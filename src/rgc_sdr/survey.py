"""DMR and P25 survey: every channel in a capture at once (VK3RQ, 2026-10-10).

Listening to one channel after another to find which carry DMR or P25, and which carry
voice, took a 3-second listen per channel -- a pass over 400-480 MHz took the best part
of half an hour, and voice, being intermittent, was mostly missed. Here one capture of
the radio's whole span gives every active channel at once:

1. **Channels** from the capture's spectrum: peaks a max-hold puts SNR_DB over the mean
   floor (a max-hold sits about 15 dB over noise, so less finds static), inside the
   span's usable middle and clear of the radio's own spike, at least 12.5 kHz apart,
   snapped to the 6.25 kHz land-mobile grid.
2. **One FFT** of the capture; each channel back out as a 48 kS/s stream by taking its
   bins and transforming back (as the Classify sweep does).
3. **The app's own decoders** on each: the NBFM discriminator into the DMR and P25
   decoders, whose counters say what was there -- colour codes or NACs, voice bursts or
   frames and on which slots, and an encrypted call.

Runs wherever the samples are: on the Mac for a radio plugged into it, and on the
network radio server for one on the Pi (a survey request), which has the radio's whole
span where the Mac has only a 256 kHz window. Pure NumPy and the app's DSP; no Qt.
"""

from __future__ import annotations

import numpy as np

from .dsp.demod import DemodChain
from .dsp.dmr import DmrDecoder
from .dsp.p25 import P25Decoder

#: How far a channel's max-hold must stand over the mean floor.
SNR_DB = 20.0
#: Channels closer than this are one.
SEPARATION_HZ = 12.5e3
GRID_HZ = 6.25e3
#: Valid frames at the spectrum's frequency that make trying its neighbours pointless.
SETTLED_SCORE = 20
#: Either side of a peak whose mean power gives the channel's centre.
CENTROID_HZ = 6.25e3
#: Each channel's stream: four-level FSK at 4800 symbols/s wants several samples a symbol.
CHANNEL_RATE = 48e3
#: Keep clear of the span's edges, where the radio's filter rolls off.
USABLE = 0.45
SPIKE_GUARD_HZ = 5e3
SEGMENT = 4096


def find_channels(iq: np.ndarray, rate: float, centre_hz: float, lo_hz: float, hi_hz: float,
                  spike_offset_hz: float = 0.0) -> list[tuple[float, float]]:
    """(frequency, SNR dB) of the active channels between `lo` and `hi`, strongest first."""
    n = SEGMENT
    count = iq.size // n
    if count < 2:
        return []
    window = np.hanning(n).astype(np.float32)
    segments = iq[:count * n].reshape(count, n) * window
    power = np.abs(np.fft.fftshift(np.fft.fft(segments, axis=1), axes=1)) ** 2
    mean_db = 10 * np.log10(power.mean(axis=0) + 1e-30)
    hold_db = 10 * np.log10(power.max(axis=0) + 1e-30)
    offsets = (np.arange(n) - n // 2) * rate / n
    freqs = centre_hz + offsets
    floor = float(np.median(mean_db))
    allowed = (freqs >= lo_hz) & (freqs <= hi_hz) & (np.abs(offsets) <= rate * USABLE)
    if spike_offset_hz:
        allowed &= np.abs(offsets - spike_offset_hz) > SPIKE_GUARD_HZ
    level = np.where(allowed, hold_db, -np.inf)
    out: list[tuple[float, float]] = []
    guard = int(SEPARATION_HZ / (rate / n))
    while True:
        k = int(np.argmax(level))
        snr = float(level[k] - floor)
        if not np.isfinite(snr) or snr < SNR_DB:
            break
        level[max(0, k - guard):k + guard + 1] = -np.inf
        # The channel's centre is where its mean power balances, not its strongest
        # max-hold bin: a TDMA burst's peak can sit off centre (measured: 463.9000 for a
        # 463.9125 MHz DMR repeater).
        half = max(1, int(CENTROID_HZ / (rate / n)))
        lo_k, hi_k = max(0, k - half), min(n, k + half + 1)
        weight = 10 ** ((mean_db[lo_k:hi_k] - floor) / 10) - 1
        weight = np.clip(weight, 0.0, None)
        centre = (float(np.sum(freqs[lo_k:hi_k] * weight) / np.sum(weight))
                  if np.sum(weight) > 0 else float(freqs[k]))
        freq = round(centre / GRID_HZ) * GRID_HZ
        if all(abs(freq - f) >= SEPARATION_HZ for f, _ in out):
            out.append((freq, snr))
    return out


def channel_stream(spectrum: np.ndarray, rate: float, centre_hz: float, freq_hz: float,
                   out_rate: float = CHANNEL_RATE) -> tuple[np.ndarray, float]:
    """The signal at `freq_hz` moved to 0 Hz, from the capture's FFT, at about `out_rate`."""
    n = spectrum.size
    m = min(n, int(round(out_rate * n / rate / 2)) * 2)
    k0 = int(round((freq_hz - centre_hz) * n / rate))
    bins = (k0 + np.arange(-m // 2, m // 2)) % n
    z = np.fft.ifft(np.fft.ifftshift(spectrum[bins])) * (m / n)
    return z.astype(np.complex64), rate * m / n


def decode_channel(stream: np.ndarray, rate: float) -> dict:
    """What the DMR and P25 decoders found in one channel's stream."""
    chain = DemodChain(rate, "nbfm", agc=False)
    dmr, p25 = DmrDecoder(chain.if_rate), P25Decoder(chain.if_rate)
    for block in np.array_split(stream, max(1, stream.size // 4800)):
        chain.process(block)
        dmr.process(chain.last_detected)
        p25.process(chain.last_detected)
    report = {"protocol": "", "colour_codes": [], "nacs": [], "voice": 0,
              "voice_slots": [], "encrypted": False,
              # Valid frames, which choose between neighbouring grid points.
              "score": p25.frames + dmr.bursts}
    if p25.frames >= 2:
        report.update(protocol="P25", nacs=sorted(p25.nacs), voice=p25.voice_frames,
                      encrypted=p25.encrypted_seen)
    elif dmr.bursts - dmr.voice_bursts >= 2 or dmr.voice_bursts >= 3:
        report.update(protocol="DMR", colour_codes=sorted(dmr.colour_codes),
                      voice=dmr.voice_bursts, voice_slots=sorted(dmr.voice_slots),
                      encrypted=dmr.encrypted_seen)
    return report


def survey_iq(iq: np.ndarray, rate: float, centre_hz: float, lo_hz: float, hi_hz: float,
              spike_offset_hz: float = 0.0, cancelled=lambda: False) -> list[dict]:
    """Every active channel in a capture, each with what its decoders found."""
    found = find_channels(iq, rate, centre_hz, lo_hz, hi_hz, spike_offset_hz)
    if not found:
        return []
    spectrum = np.fft.fft(np.asarray(iq, dtype=np.complex64))
    out = []
    seen: set[float] = set()
    for freq, snr in found:
        if cancelled():
            break
        # The spectrum places a channel to within a grid step or so (a TDMA signal's
        # power is lumpy); the decoders settle it: the grid point either side is tried
        # too, and the one decoding the most valid frames is the channel.
        best = None
        for candidate in (freq, freq - GRID_HZ, freq + GRID_HZ):
            if candidate in seen:
                continue
            stream, out_rate = channel_stream(spectrum, rate, centre_hz, candidate)
            report = decode_channel(stream, out_rate)
            report.update(freq_hz=float(candidate), snr_db=round(snr, 1))
            if best is None or report["score"] > best["score"]:
                best = report
            if candidate == freq and report["score"] >= SETTLED_SCORE:
                break                       # decoding well where the spectrum said
        if best is not None:
            seen.add(best["freq_hz"])
            out.append(best)
    return out


def merge(table: dict, reports: list[dict], when: float) -> None:
    """Fold a pass's reports into `table` (frequency -> running record): protocols and
    codes kept once seen, voice counted up, and when voice was last heard."""
    for r in reports:
        row = table.setdefault(r["freq_hz"], {"freq_hz": r["freq_hz"], "protocol": "",
                                              "colour_codes": set(), "nacs": set(),
                                              "voice": 0, "voice_slots": set(),
                                              "encrypted": False, "last_voice": None,
                                              "snr_db": r["snr_db"], "heard": 0})
        row["heard"] += 1
        row["snr_db"] = max(row["snr_db"], r["snr_db"])
        if r["protocol"]:
            row["protocol"] = r["protocol"]
        row["colour_codes"] |= set(r["colour_codes"])
        row["nacs"] |= set(r["nacs"])
        row["voice_slots"] |= set(r["voice_slots"])
        row["encrypted"] = row["encrypted"] or r["encrypted"]
        if r["voice"]:
            row["voice"] += r["voice"]
            row["last_voice"] = when


class SurveyJob:
    """One window's survey on a thread: a radio on this machine captured and surveyed
    here; a network radio asked to survey on its server (it has the whole span). Poll
    `done`; then `reports` (or `error`)."""

    def __init__(self, source, centre_hz: float, lo_hz: float, hi_hz: float,
                 seconds: float = 3.0) -> None:
        import threading

        self.source = source
        self.centre_hz, self.lo_hz, self.hi_hz = float(centre_hz), float(lo_hz), float(hi_hz)
        self.seconds = float(seconds)
        self.reports: list[dict] = []
        self.error = ""
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._remote = hasattr(source, "survey")
        self._reader = None if self._remote else source.sequential_reader()
        self._thread = threading.Thread(target=self._run, name="survey", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self) -> None:
        import time

        try:
            if self._remote:
                self.reports = self.source.survey(self.centre_hz, self.seconds,
                                                  self.lo_hz, self.hi_hz)
                return
            rate = float(self.source.sample_rate)
            wanted = int(self.seconds * rate)
            chunks, have = [], 0
            deadline = time.monotonic() + self.seconds * 3 + 2
            while have < wanted and not self._cancel.is_set():
                if time.monotonic() > deadline:
                    raise TimeoutError("the radio sent too few samples")
                n = self._reader.available()
                if n == 0:
                    time.sleep(0.02)
                    continue
                piece = self._reader.read(min(n, wanted - have))
                chunks.append(piece)
                have += piece.size
            if chunks and not self._cancel.is_set():
                self.reports = survey_iq(
                    np.concatenate(chunks), rate, float(self.source.center_freq),
                    self.lo_hz, self.hi_hz,
                    float(getattr(self.source, "dc_spike_offset_hz", 0.0) or 0.0),
                    cancelled=self._cancel.is_set)
        except Exception as exc:                          # shown, never fatal
            self.error = str(exc) or type(exc).__name__
        finally:
            self._done.set()


def plan_windows(lo_hz: float, hi_hz: float, span_hz: float) -> list[float]:
    """Window centres covering `lo`..`hi` with the usable middle of each span."""
    step = span_hz * 2 * USABLE * 0.95
    if hi_hz <= lo_hz or step <= 0:
        return []
    centres, centre = [], lo_hz + step / 2
    while centre - step / 2 < hi_hz:
        centres.append(centre)
        centre += step
    return centres
