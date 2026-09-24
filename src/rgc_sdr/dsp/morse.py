"""Decode received CW into text.

Morse is on/off keying, so the signal is in the *envelope* of the narrow CW filter, not
in the audio tone. The decoder therefore works on |filtered signal| rather than on the
demodulated audio, which removes the beat note and leaves only the keying.

Two things have to be learned from the signal rather than assumed:

* **The threshold**, because absolute levels mean nothing. It is tracked from the
  envelope's own low and high percentiles, so the same code works on a weak signal and a
  loud one.
* **The dot length**, because operators send anywhere from 5 to 50 words per minute and
  change speed mid-transmission. Marks cluster around one and three units, which is
  enough to track it.

Pure NumPy, no Qt, no device access.
"""

from __future__ import annotations

from collections import deque

import numpy as np

MORSE_TO_CHAR: dict[str, str] = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":",
    "-.-.-.": ";", "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
    ".-..-.": '"', "...-..-": "$", ".--.-.": "@",
    # Prosigns worth showing rather than dropping.
    "-.-.-": "<KA>", "...-.-": "<SK>", "-...-.-": "<BK>", "........": "<ERR>",
}

CHAR_TO_MORSE: dict[str, str] = {
    char: code for code, char in MORSE_TO_CHAR.items() if len(char) == 1
}

#: Envelope sample rate the decoder works at. 1.5 kHz is 0.67 ms resolution, ample for
#: elements tens of milliseconds long, and cheap to run.
ENVELOPE_DECIM = 32

#: Dot length limits, which bound the speed the decoder will track: 6 to 60 WPM.
MIN_DOT_MS = 20.0
MAX_DOT_MS = 200.0
DEFAULT_DOT_MS = 60.0          # 20 WPM

#: Runs shorter than this are keying transients or noise, not elements.
#:
#: The channel filter is several hundred taps, so its impulse response smears each keying
#: edge by about 10 ms and can leave a short spurious run at the start of a transmission.
#: Still below the 20 ms dot of the fastest speed tracked.
MIN_RUN_MS = 12.0

#: Durations below this fraction of the current dot estimate are not used for learning.
#:
#: One spurious 10 ms run was enough to become the whole low cluster and drag the estimate
#: onto its floor, after which a 60 ms dot read as a dash and "SOS" decoded as "U OS".
#: 0.4 still allows tracking an operator who speeds up by more than twice.
LEARN_FLOOR_RATIO = 0.4

#: How far above the noise the envelope must swing before anything is decoded.
MIN_DEPTH_RATIO = 2.0

#: Seconds of envelope kept for estimating the threshold.
#:
#: This has to span several elements. Estimating from one block instead -- 21 ms, shorter
#: than a single dot -- made a long dash look like a constant level with no keying in it,
#: and the decoder reset mid-character and emitted a stream of E's.
THRESHOLD_WINDOW_S = 1.5


def text_to_morse(text: str) -> str:
    """Render text as morse, for generating test signals."""
    words = []
    for word in text.upper().split():
        words.append(" ".join(CHAR_TO_MORSE[c] for c in word if c in CHAR_TO_MORSE))
    return "   ".join(words)


class EnvelopeSampler:
    """Reduce a complex block to an envelope at a lower rate.

    A plain block mean is enough: the envelope of keying is slowly varying compared with
    the sample rate, so there is nothing near the new Nyquist to alias. Carries the
    remainder between blocks so no samples are lost at the seams.
    """

    def __init__(self, decim: int = ENVELOPE_DECIM) -> None:
        if decim < 1:
            raise ValueError("decim must be >= 1")
        self.decim = int(decim)
        self._carry = np.zeros(0, dtype=np.float64)

    def reset(self) -> None:
        self._carry = np.zeros(0, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0 and self._carry.size == 0:
            return np.zeros(0, dtype=np.float64)
        magnitude = np.abs(block).astype(np.float64)
        data = np.concatenate([self._carry, magnitude]) if self._carry.size else magnitude
        usable = (data.size // self.decim) * self.decim
        self._carry = data[usable:].copy()
        if usable == 0:
            return np.zeros(0, dtype=np.float64)
        return data[:usable].reshape(-1, self.decim).mean(axis=1)


class CwDecoder:
    """Turn an envelope stream into text.

    Fed envelope samples and returns whatever characters completed. Keeps the last
    `history` characters so a display can show a rolling line.
    """

    def __init__(
        self,
        sample_rate: float,
        history: int = 50,
        dot_ms: float = DEFAULT_DOT_MS,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.history = int(history)
        self._dot_ms = float(dot_ms)
        self._chars: deque[str] = deque(maxlen=self.history)
        #: Recent run lengths -- marks *and* gaps -- for estimating the dot.
        #:
        #: Two earlier attempts failed. An EMA is fragile: starting from the wrong speed,
        #: a dash misread as a dot pulls the estimate further the wrong way. A low
        #: percentile of marks fails on dash-heavy text, because as dashes accumulate the
        #: percentile climbs into the dash cluster and dashes then read as dots -- "CQ"
        #: decoded as "CX" for exactly that reason.
        #:
        #: Gaps are what make it tractable: the gap *between elements of a character* is
        #: always exactly one dot. So marks (1 and 3 units) and gaps (1, 3 and 7) share a
        #: large one-unit population, and splitting the combined set in two finds it
        #: whatever the ratio of dots to dashes in the text.
        self._durations: deque[float] = deque(maxlen=48)
        #: Mark lengths of the character being received, classified only once it ends.
        #:
        #: Deciding dot-or-dash as each element arrives means the first character is read
        #: against the *initial* speed guess and is usually wrong -- at 35 wpm a dash is
        #: shorter than twice the assumed dot, so "PARIS" came out as "FARIS". Holding the
        #: durations and classifying at the end uses the estimate as it stands once the
        #: whole character has been seen, by which point it has locked on.
        self._symbol_marks: list[float] = []
        self._in_mark = False
        self._run = 0
        self._noise = 0.0
        self._peak = 0.0
        self._space_added = False
        self._window = int(max(64, THRESHOLD_WINDOW_S * self.sample_rate))
        self._recent = deque(maxlen=self._window)

    # -- reporting ---------------------------------------------------------

    @property
    def text(self) -> str:
        return "".join(self._chars)

    @property
    def dot_ms(self) -> float:
        return self._dot_ms

    @property
    def wpm(self) -> float:
        """Speed in words per minute, by the standard PARIS timing."""
        return 1200.0 / self._dot_ms if self._dot_ms > 0 else 0.0

    def reset(self) -> None:
        self._chars.clear()
        self._in_mark = False
        self._run = 0
        self._space_added = False
        self._symbol_marks.clear()
        self._durations.clear()
        self._recent.clear()

    # -- thresholds --------------------------------------------------------

    def _update_thresholds(self, envelope: np.ndarray) -> float | None:
        """Threshold from the last `THRESHOLD_WINDOW_S` of envelope, not from one block.

        Returns None when the envelope is not swinging enough to be keying, so that
        characters are not invented out of hiss.
        """
        self._recent.extend(float(v) for v in envelope)
        if len(self._recent) < 32:
            return None
        history = np.fromiter(self._recent, dtype=np.float64, count=len(self._recent))
        self._noise = float(np.percentile(history, 20.0))
        self._peak = float(np.percentile(history, 95.0))
        if self._peak <= self._noise * MIN_DEPTH_RATIO:
            return None
        return self._noise + 0.4 * (self._peak - self._noise)

    # -- element timing ----------------------------------------------------

    def _ms(self, samples: int) -> float:
        return samples * 1000.0 / self.sample_rate

    @property
    def _symbol(self) -> str:
        """The character so far, classified against the current dot estimate."""
        boundary = 2.0 * self._dot_ms
        return "".join("." if d < boundary else "-" for d in self._symbol_marks)

    def _learn_dot(self, duration_ms: float) -> None:
        """Re-estimate the dot by splitting recent run lengths into two clusters.

        The lower cluster is the one-unit population: single dots, and the gaps between
        elements of a character, which are one unit by definition.
        """
        if duration_ms < LEARN_FLOOR_RATIO * self._dot_ms:
            return                      # a transient, not an element
        self._durations.append(duration_ms)
        if len(self._durations) < 5:
            return
        runs = np.fromiter(self._durations, dtype=np.float64, count=len(self._durations))
        low, high = float(runs.min()), float(runs.max())
        if high <= low * 1.5:
            # One cluster only: everything is the same length, so it is all one unit.
            estimate = low
        else:
            for _ in range(8):
                split = (low + high) / 2.0
                short, long_ = runs[runs <= split], runs[runs > split]
                if short.size == 0 or long_.size == 0:
                    break
                # Medians, so one stray run cannot define a cluster.
                low, high = float(np.median(short)), float(np.median(long_))
            estimate = low
        self._dot_ms = float(np.clip(estimate, MIN_DOT_MS, MAX_DOT_MS))

    def _emit_symbol(self) -> str:
        symbol = self._symbol
        if not symbol:
            return ""
        char = MORSE_TO_CHAR.get(symbol, "")
        self._symbol_marks.clear()
        if not char:
            char = "*"          # heard something, could not read it
        self._chars.append(char)
        return char

    def _maybe_word_space(self, samples: int) -> str:
        """Add a word space once a gap is long enough.

        Kept separate from emitting the character because the character is flushed as
        soon as the gap reaches character length -- so by the time a word-length gap is
        confirmed there is no character left to attach the space to, and requiring one
        silently dropped every word break.
        """
        if self._space_added or self._ms(samples) < 5.0 * self._dot_ms:
            return ""
        if not self._chars or self._chars[-1] == " ":
            return ""
        self._chars.append(" ")
        self._space_added = True
        return " "

    def _close_run(self, was_mark: bool, samples: int) -> str:
        duration = self._ms(samples)
        if duration < MIN_RUN_MS:
            return ""
        if was_mark:
            # Recorded, not yet classified: that happens when the character completes.
            self._symbol_marks.append(duration)
            self._learn_dot(duration)
            self._space_added = False
            return ""
        # A space: short is within a character, medium ends one, long ends a word.
        if duration < 2.0 * self._dot_ms:
            # One-unit gaps are the most reliable evidence of the dot length there is.
            self._learn_dot(duration)
            return ""
        produced = self._emit_symbol()
        return produced + self._maybe_word_space(samples)

    # -- the stream --------------------------------------------------------

    def feed(self, envelope: np.ndarray) -> str:
        """Feed envelope samples; return any characters that completed."""
        if envelope.size == 0:
            return ""
        threshold = self._update_thresholds(envelope)
        if threshold is None:
            # Carrier gone. Close off whatever was in progress so it is not stranded.
            produced = ""
            if self._symbol:
                produced = self._emit_symbol()
            self._in_mark = False
            self._run = 0
            return produced

        marks = envelope > threshold
        produced = ""
        # Split the block into uniform runs, then fold them into the run in progress.
        changes = np.flatnonzero(np.diff(marks.view(np.int8))) + 1
        for segment in np.split(marks, changes):
            is_mark = bool(segment[0])
            if is_mark == self._in_mark:
                self._run += segment.size
            else:
                produced += self._close_run(self._in_mark, self._run)
                self._in_mark = is_mark
                self._run = segment.size

        # An open space long enough to end a character must not wait for the next mark,
        # or the final character of a transmission never appears.
        if not self._in_mark:
            if self._symbol and self._ms(self._run) >= 2.0 * self._dot_ms:
                produced += self._emit_symbol()
            produced += self._maybe_word_space(self._run)
        return produced
