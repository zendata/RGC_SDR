"""Four-level FSK at 4800 symbols/s, the common ground of P25 Phase 1 and DMR.

Both send dibits as four frequency deviations: 01 -> +3, 00 -> +1, 10 -> -1, 11 -> -3
(in units of 600 Hz for P25 C4FM, 648 Hz for DMR), and both begin their frames with a
known 24-symbol sync word. So recovery is built around the sync, not a free-running
clock: the FM discriminator is correlated with the sync pattern, each peak marks a frame
start to a fraction of a sample, the 24 known symbols fix the frame's own scale and
offset (so tuning error and deviation do not matter), and the symbols after it are read
at the nominal symbol period from there. Transmitter clocks are accurate to a few ppm,
far inside half a symbol over the few hundred symbols of a frame.

Pure NumPy and scipy; no Qt, no device access (PLANNING.md 7p).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .decimate import lowpass_taps
from .filters import Fir, fir_length_for

SYMBOL_RATE = 4800.0
#: Normalised correlation a sync must reach. Measured on air: real syncs reach 0.95 and
#: more; a P25 sync is all +/-3 symbols, so FM voice reaches 0.85 now and then, which the
#: frame's own error check then rejects.
SYNC_THRESHOLD = 0.8
#: Symbol value of each dibit, and the reverse.
DIBIT_SYMBOL = {0b01: 3, 0b00: 1, 0b10: -1, 0b11: -3}
_LEVELS = np.array([-3, -1, 1, 3])
_LEVEL_DIBIT = np.array([0b11, 0b10, 0b00, 0b01])
#: Seconds an identical message is held back for: control channels repeat the same
#: broadcasts many times a second.
REPEAT_SECONDS = 30.0


def sync_symbols(word: int) -> np.ndarray:
    """A 48-bit sync word -> its 24 symbols."""
    return np.array([DIBIT_SYMBOL[(word >> (46 - 2 * i)) & 3] for i in range(24)], float)


def dibits_to_bits(dibits: np.ndarray) -> np.ndarray:
    d = np.asarray(dibits, dtype=np.uint8)
    return np.ravel(np.stack([(d >> 1) & 1, d & 1], axis=1))


def bits_to_int(bits) -> int:
    value = 0
    for b in bits:
        value = value << 1 | int(b)
    return value


def crc_ccitt(bits) -> int:
    """CRC-16/CCITT (x^16 + x^12 + x^5 + 1), initial value 0, no inversion."""
    r = 0
    for b in bits:
        fb = ((r >> 15) ^ int(b)) & 1
        r = (r << 1) & 0xFFFF
        if fb:
            r ^= 0x1021
    return r


@dataclass
class SyncHit:
    """One sync found: which, where its first symbol starts (absolute sample index, in
    the decoder's filtered stream), and +1, or -1 if it matched upside down."""

    name: str
    start: float
    sign: int


class SyncFinder:
    """FM discriminator samples in; frames out, each as the symbols from its sync on.

    `frame_symbols` is how many symbols to hand over from each sync's first symbol (it
    may also start before the sync: `lead_symbols`, for DMR's fields ahead of it).

    A sync that matches upside down is turned the right way up (`upright`, for P25: a
    receiver that inverts the spectrum), or else kept as it is and read against the
    negated pattern -- for DMR, where a data sync's negative is the voice sync.
    """

    def __init__(self, sample_rate: float, syncs: dict[str, int], frame_symbols: int,
                 lead_symbols: int = 0, upright: bool = True) -> None:
        self.sample_rate = float(sample_rate)
        self.sps = self.sample_rate / SYMBOL_RATE
        if self.sps < 2:
            raise ValueError("need at least two samples per symbol")
        self.frame_symbols = int(frame_symbols)
        self.lead_symbols = int(lead_symbols)
        self.upright = bool(upright)
        self._sync_values = {name: sync_symbols(word) for name, word in syncs.items()}
        n = int(round(24 * self.sps))
        self._n = n
        self._templates = {}
        for name, symbols in self._sync_values.items():
            t = symbols[np.minimum((np.arange(n) / self.sps).astype(int), 23)]
            t = t - t.mean()
            self._templates[name] = (t / np.linalg.norm(t))[::-1]
        # Keep the symbols, lose the noise: the shaped signal fills about +/-2.4 kHz.
        cutoff = 2800.0 / self.sample_rate
        self._fir = Fir(lowpass_taps(cutoff, fir_length_for(cutoff, maximum=127)))
        self.reset()

    def reset(self) -> None:
        self._fir.reset()
        self._buf = np.zeros(0)
        self._base = 0               # absolute index of _buf[0]
        self._searched = 0           # absolute index up to which syncs have been sought
        self._pending: list[SyncHit] = []

    def _correlate(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        """Normalised correlation for start positions lo..hi-1 (absolute)."""
        n = self._n
        x = self._buf[lo - self._base: hi - self._base + n - 1]
        cs = np.cumsum(np.concatenate([[0.0], x]))
        cs2 = np.cumsum(np.concatenate([[0.0], x * x]))
        s, s2 = cs[n:] - cs[:-n], cs2[n:] - cs2[:-n]
        norm = np.sqrt(np.maximum(s2 - s * s / n, 1e-12))
        return {name: fftconvolve(x, t, mode="valid") / norm
                for name, t in self._templates.items()}

    def process(self, x: np.ndarray) -> list[tuple[SyncHit, np.ndarray]]:
        """Returns (sync, symbols) for every frame complete so far. Symbols are scaled so
        the sync's own reads exactly +/-1, +/-3; round them with `to_dibits`."""
        x = np.asarray(x, dtype=np.float64)
        if x.size:
            self._buf = np.concatenate([self._buf, self._fir.process(x)])
        end = self._base + self._buf.size
        sps = self.sps
        # A peak needs a sample after it, and runs of candidates closer than a symbol
        # are one sync, so stop the search a little short of the end.
        hi = end - self._n - int(2 * sps)
        lo = max(self._searched, self._base)
        if hi > lo:
            for name, c in self._correlate(lo, hi + 1).items():
                strong = np.flatnonzero(np.abs(c[:-1]) >= SYNC_THRESHOLD)
                if strong.size == 0:
                    continue
                # Group candidates within a symbol; keep each group's strongest.
                groups = np.split(strong, np.flatnonzero(np.diff(strong) > sps) + 1)
                for g in groups:
                    k = int(g[np.argmax(np.abs(c[g]))])
                    # Sub-sample position from a parabola through the peak.
                    a = abs(c[k - 1]) if k > 0 else 0.0
                    b, d = abs(c[k]), abs(c[k + 1])
                    den = a - 2 * b + d
                    frac = 0.5 * (a - d) / den if den < 0 and k > 0 else 0.0
                    self._pending.append(SyncHit(name, lo + k + frac,
                                                 1 if c[k] > 0 else -1))
            self._searched = hi
            self._pending.sort(key=lambda h: h.start)
        out = []
        keep = []
        for hit in self._pending:
            first = hit.start - self.lead_symbols * sps
            last = hit.start + (self.frame_symbols - self.lead_symbols) * sps
            if last + 2 >= end:
                keep.append(hit)
                continue
            if first < self._base:
                continue                                  # its start was trimmed away
            pos = first + (np.arange(self.frame_symbols) + 0.5) * sps - self._base
            v = np.interp(pos, np.arange(self._buf.size), self._buf)
            sync = self._sync_values[hit.name]
            if self.upright:
                v = v * hit.sign
            else:
                sync = sync * hit.sign
            ref = v[self.lead_symbols:self.lead_symbols + 24]
            scale, offset = np.polyfit(sync, ref, 1)
            if scale <= 0:
                continue
            out.append((hit, (v - offset) / scale))
        self._pending = keep
        # Keep what a frame may still need: back to the oldest pending frame's start,
        # and the template's length behind the search point.
        horizon = self._searched - int(self.lead_symbols * sps) - 2
        if keep:
            horizon = min(horizon, int(keep[0].start - self.lead_symbols * sps) - 2)
        drop = max(0, horizon - self._base)
        if drop:
            self._buf = self._buf[drop:]
            self._base += drop
        return out


def to_dibits(symbols: np.ndarray) -> np.ndarray:
    """Scaled symbols -> dibits, each to the nearest of the four levels."""
    s = np.asarray(symbols)
    return _LEVEL_DIBIT[np.argmin(np.abs(s[:, None] - _LEVELS), axis=1)]


class RepeatFilter:
    """Lets a message's text through once per REPEAT_SECONDS: control channels send the
    same broadcasts several times a second."""

    def __init__(self, seconds: float = REPEAT_SECONDS) -> None:
        self.seconds = float(seconds)
        self._seen: dict[str, float] = {}

    def fresh(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        last = self._seen.get(key)
        if last is not None and now - last < self.seconds:
            return False
        self._seen[key] = now
        if len(self._seen) > 4096:                       # forget the oldest half
            for k in sorted(self._seen, key=self._seen.get)[:2048]:
                del self._seen[k]
        return True
