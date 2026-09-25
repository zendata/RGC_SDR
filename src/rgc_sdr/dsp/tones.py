"""Sub-audible signalling for NBFM: CTCSS tones and DCS codes, sent and detected.

CTCSS is a steady tone, 67-254 Hz, under the voice. DCS sends a 23-bit Golay (23,12)
codeword over and over at 134.4 bit/s as a low-passed NRZ signal: 12 data bits -- the
3-digit octal code in 9 bits, then a fixed 100 -- and 11 check bits, least significant
bit first. A "1" is an upward frequency shift.

Verified against published DCS tables, not only against this module's own decoder: the
bitwise inverse of the 023 codeword is a rotation of 047's, and 025's of 244's -- the
inverse pairs those tables list. Code 023 encodes to 0x763813.

Both are kept below 300 Hz, where the transmit chain's voice high-pass leaves room, and
both are detected on receive from the discriminator audio, decimated to 1.5 kHz.

Pure NumPy, no Qt, no device access.
"""

from __future__ import annotations

import numpy as np

from .decimate import StreamDecimator, lowpass_taps
from .filters import Fir, fir_length_for

DCS_BAUD = 134.4
#: Golay (23,12) generator, x^11+x^10+x^6+x^5+x^4+x^2+1.
GOLAY_POLY = 0xC75

#: The standard DCS codes, as radios list them.
DCS_CODES: tuple[str, ...] = (
    "023", "025", "026", "031", "032", "036", "043", "047", "051", "053", "054", "065",
    "071", "072", "073", "074", "114", "115", "116", "122", "125", "131", "132", "134",
    "143", "145", "152", "155", "156", "162", "165", "172", "174", "205", "212", "223",
    "225", "226", "243", "244", "245", "246", "251", "252", "255", "261", "263", "265",
    "266", "271", "274", "306", "311", "315", "325", "331", "332", "343", "346", "351",
    "356", "364", "365", "371", "411", "412", "413", "423", "431", "432", "445", "446",
    "452", "454", "455", "462", "464", "465", "466", "503", "506", "516", "523", "526",
    "532", "546", "565", "606", "612", "624", "627", "631", "632", "654", "662", "664",
    "703", "712", "723", "731", "732", "734", "743", "754",
)

#: Below this, sub-audible signalling lives. Voice is high-passed above it.
SUBAUDIBLE_HZ = 300.0


def _golay(data12: int) -> int:
    """23-bit systematic Golay codeword: 11 check bits above the 12 data bits."""
    cw = data12 & 0xFFF
    for _ in range(12):
        if cw & 1:
            cw ^= GOLAY_POLY
        cw >>= 1
    return (cw << 12) | (data12 & 0xFFF)


def dcs_codeword(code: str) -> int:
    """The 23-bit codeword for an octal DCS code such as "023"."""
    value = int(code, 8)
    if not 0 <= value < 512:
        raise ValueError(f"DCS code {code!r} is not three octal digits")
    return _golay(0x800 | value)


def dcs_bits(code: str) -> np.ndarray:
    """The codeword's 23 bits in transmission order (least significant first)."""
    word = dcs_codeword(code)
    return np.array([(word >> i) & 1 for i in range(23)], dtype=np.uint8)


def _subaudible_lowpass(rate: float) -> Fir:
    cutoff = SUBAUDIBLE_HZ / rate
    return Fir(lowpass_taps(cutoff, fir_length_for(cutoff, maximum=1023)))


class CtcssGenerator:
    """A continuous sine at the tone frequency, amplitude 1."""

    def __init__(self, rate: float, hz: float) -> None:
        self.rate, self.hz = float(rate), float(hz)
        self._phase = 0.0

    def process(self, n: int) -> np.ndarray:
        step = 2 * np.pi * self.hz / self.rate
        phases = self._phase + step * np.arange(1, n + 1)
        self._phase = float(phases[-1] % (2 * np.pi)) if n else self._phase
        return np.sin(phases)


class DcsGenerator:
    """The DCS codeword repeated as NRZ (+1 for a 1), low-passed below 300 Hz."""

    def __init__(self, rate: float, code: str) -> None:
        self.rate = float(rate)
        self.code = code
        self._levels = np.where(dcs_bits(code) == 1, 1.0, -1.0)
        self._n = 0
        self._filter = _subaudible_lowpass(self.rate)

    def process(self, n: int) -> np.ndarray:
        t = (self._n + np.arange(n)) / self.rate
        self._n += n
        bit = (t * DCS_BAUD).astype(np.int64) % 23
        return self._filter.process(self._levels[bit])


def tone_generator(rate: float, kind: str, value) -> CtcssGenerator | DcsGenerator:
    if kind == "ctcss":
        return CtcssGenerator(rate, float(value))
    if kind == "dcs":
        return DcsGenerator(rate, str(value))
    raise ValueError(f"unknown tone kind {kind!r}")


class ToneSquelch:
    """Opens only while the chosen CTCSS tone or DCS code is being received.

    Works on discriminator audio. Decimated to ~1.5 kHz and low-passed below 300 Hz, the
    last half second is examined on every block:

    * CTCSS: the strongest sub-audible component must sit within 1 Hz of the tone and
      hold a good share of the sub-audible energy. Frequency is estimated from a
      zero-padded spectrum, so neighbouring tones 2.3 Hz apart are told apart.
    * DCS: normalised correlation against the code's own periodic waveform, at every
      phase, must exceed a threshold -- and with the right sign, since an inverted code
      is a different code.

    Closing waits for two misses in a row, so one bad block does not chop the audio.
    """

    WINDOW_S = 0.5
    CTCSS_TOLERANCE_HZ = 1.0
    CTCSS_SHARE = 0.3
    DCS_THRESHOLD = 0.6

    def __init__(self, audio_rate: float, kind: str, value) -> None:
        self.kind = kind
        self.value = value
        factor = 1
        while audio_rate / (factor * 2) >= 1200.0:
            factor *= 2
        self._down = StreamDecimator(factor)
        self.rate = audio_rate / factor
        self._lowpass = _subaudible_lowpass(self.rate)
        self._window = int(self.WINDOW_S * self.rate)
        self._buffer = np.zeros(0)
        self._misses = 0
        self.open = False
        if kind == "dcs":
            period = 23 * self.rate / DCS_BAUD
            gen = DcsGenerator(self.rate, str(value))
            # Settled template, long enough to slide one whole period past the window.
            warm = gen.process(int(period * 2))
            self._template = gen.process(self._window + int(np.ceil(period)) + 1)
            del warm
        elif kind != "ctcss":
            raise ValueError(f"unknown tone kind {kind!r}")

    def process(self, audio: np.ndarray) -> bool:
        x = self._down.process(np.asarray(audio, dtype=np.float64))
        if x.size:
            x = self._lowpass.process(x)
            self._buffer = np.concatenate([self._buffer, x])[-self._window:]
        if self._buffer.size < self._window:
            return self.open
        present = self._ctcss_present() if self.kind == "ctcss" else self._dcs_present()
        if present:
            self.open, self._misses = True, 0
        else:
            self._misses += 1
            if self._misses >= 2:
                self.open = False
        return self.open

    def _ctcss_present(self) -> bool:
        x = self._buffer - self._buffer.mean()
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size), n=16384)) ** 2
        freqs = np.fft.rfftfreq(16384, 1.0 / self.rate)
        band = (freqs >= 55.0) & (freqs <= 270.0)
        total = spectrum[band].sum()
        if total <= 0:
            return False
        peak = freqs[band][int(np.argmax(spectrum[band]))]
        if abs(peak - float(self.value)) > self.CTCSS_TOLERANCE_HZ:
            return False
        near = band & (np.abs(freqs - float(self.value)) <= 3.0)
        return spectrum[near].sum() / total >= self.CTCSS_SHARE

    def _dcs_present(self) -> bool:
        x = self._buffer - self._buffer.mean()
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            return False
        t = self._template
        corr = np.correlate(t, x, mode="valid")
        # Energy of each template slice the window was compared with.
        energy = np.convolve(t * t, np.ones(x.size), mode="valid")
        score = corr / (norm_x * np.sqrt(np.maximum(energy, 1e-30)))
        return float(score.max()) >= self.DCS_THRESHOLD
