"""DMR voice: AMBE+2 speech from a DMR channel's voice bursts (VK3RQ, 2026-10-08).

A call on one timeslot is a run of voice superframes, each six bursts A-F of that slot,
one every 60 ms (the other slot's bursts in between). Only burst A carries the voice
sync; B-F carry embedded signalling (EMB and a piece of link control) in its place. So
the bursts are found from A: the sync finder hands over a whole superframe from each
voice sync, its timing and scale fixed by the sync, and B-F are read at their places --
transmitter clocks are good to a few ppm, a small fraction of a symbol over 330 ms. A
place that holds a data sync instead (the call's terminator) ends the superframe.

Each burst carries three AMBE+2 frames of 72 bits: 108 bits, the 48 in the middle, 108
bits; the second frame straddles the middle. Each frame is put into mbelib's
ambe_fr[4][24] by DSD's DMR interleave tables (W, X, Y, Z below; DSD is ISC-licensed,
as the P25 tables in p25voice.py are) and decoded by mbelib's AMBE 3600x2450 decoder
(Homebrew's `mbelib`, as for P25: the codec comes from outside), 20 ms of 8 kHz speech
a frame.

A call whose EMB says it is encrypted (the privacy indicator) is muted, as encrypted
P25 is: without its key it would be noise.

Pure NumPy apart from the codec; no Qt, no device access.
"""

from __future__ import annotations

import ctypes

import numpy as np

from .dmr import BURST_SYMBOLS, LEAD_SYMBOLS, SYNCS, _TACT, TACT_POSITIONS
from .fsk4 import SyncFinder, bits_to_int, dibits_to_bits, to_dibits
from .p25voice import FRAME_SAMPLES, _MbeParms, _load_mbelib

AUDIO_RATE = 8000.0
#: A slot's superframe: six of its bursts, one every two burst periods.
SUPERFRAME_BURSTS = 6
SUPERFRAME_SYMBOLS = (SUPERFRAME_BURSTS - 1) * 2 * BURST_SYMBOLS + BURST_SYMBOLS
#: A burst's place holds a data burst (not voice) if this many of its 24 middle
#: dibits match a data sync.
DATA_SYNC_MATCH = 20

# DSD's DMR AMBE interleave: dibit i of a 72-bit frame puts its high bit in
# ambe_fr[W[i]][X[i]] and its low bit in ambe_fr[Y[i]][Z[i]].
W = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2)
X = (23, 10, 22, 9, 21, 8, 20, 7, 19, 6, 18, 5, 17, 4, 16, 3, 15, 2,
     14, 1, 13, 0, 12, 10, 11, 9, 10, 8, 9, 7, 8, 6, 7, 5, 6, 4)
Y = (0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0, 3, 1, 3, 1, 3, 1, 3,
     1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3)
Z = (5, 3, 4, 2, 3, 1, 2, 0, 1, 13, 0, 12, 22, 11, 21, 10, 20, 9,
     19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2, 12, 1, 11, 0)

_DATA_SYNC_DIBITS = None


def _data_sync_dibits() -> np.ndarray:
    global _DATA_SYNC_DIBITS
    if _DATA_SYNC_DIBITS is None:
        word = SYNCS["bs"]
        bits = np.array([(word >> (47 - i)) & 1 for i in range(48)], dtype=np.uint8)
        _DATA_SYNC_DIBITS = bits[0::2] * 2 + bits[1::2]
    return _DATA_SYNC_DIBITS


def deinterleave(frame72: np.ndarray) -> np.ndarray:
    """72 frame bits -> mbelib's ambe_fr[4][24] (0/1)."""
    bits = np.asarray(frame72, dtype=np.uint8)
    out = np.zeros((4, 24), dtype=np.uint8)
    out[W, X] = bits[0::2]
    out[Y, Z] = bits[1::2]
    return out


def burst_frames(bits: np.ndarray) -> list[np.ndarray]:
    """A voice burst's bits (CACH first, as the DMR decoder reads them) -> its three
    72-bit AMBE frames."""
    first, second = bits[24:132], bits[180:288]
    return [first[:72], np.concatenate([first[72:], second[:36]]), second[36:]]


def emb_privacy(bits: np.ndarray) -> int:
    """A burst B-F's EMB privacy indicator (the bit after the 4-bit colour code)."""
    return int(bits[132 + 4])


class AmbeDecoder:
    """One AMBE+2 frame at a time, through mbelib (keeps the codec's state)."""

    SCALE = 1.0 / 32768.0
    UV_QUALITY = 3

    def __init__(self) -> None:
        self._mbe = _load_mbelib()
        if self._mbe is None:
            raise RuntimeError("mbelib is not installed: brew install mbelib")
        self._cur, self._prev, self._enh = _MbeParms(), _MbeParms(), _MbeParms()
        self._mbe.mbe_initMbeParms(ctypes.byref(self._cur), ctypes.byref(self._prev),
                                   ctypes.byref(self._enh))
        self._out = (ctypes.c_float * FRAME_SAMPLES)()
        self._fr = ((ctypes.c_char * 24) * 4)()
        self._d = (ctypes.c_char * 49)()
        self._err_str = ctypes.create_string_buffer(64)
        self.errors = 0

    def decode(self, frame: np.ndarray) -> np.ndarray:
        """ambe_fr[4][24] (0/1) -> 160 float samples, about -1..1."""
        flat = np.asarray(frame, dtype=np.uint8).reshape(4, 24)
        for r in range(4):
            ctypes.memmove(self._fr[r], flat[r].tobytes(), 24)
        errs, errs2 = ctypes.c_int(0), ctypes.c_int(0)
        self._mbe.mbe_processAmbe3600x2450Framef(
            self._out, ctypes.byref(errs), ctypes.byref(errs2), self._err_str, self._fr,
            self._d, ctypes.byref(self._cur), ctypes.byref(self._prev),
            ctypes.byref(self._enh), self.UV_QUALITY)
        self.errors = int(errs2.value)
        return np.frombuffer(self._out, dtype=np.float32).copy() * self.SCALE


class DmrVoice:
    """FM discriminator samples in, 8 kHz speech out, from one timeslot (`slot` 1 or 2)
    or whichever is talking (0)."""

    def __init__(self, sample_rate: float, slot: int = 0) -> None:
        self.sample_rate = float(sample_rate)
        self.slot = int(slot)
        self._finder = SyncFinder(self.sample_rate, SYNCS, SUPERFRAME_SYMBOLS,
                                  LEAD_SYMBOLS, upright=False)
        self._codecs = {1: AmbeDecoder(), 2: AmbeDecoder()}
        #: The slot being played when both are allowed: the first to speak keeps it.
        self._current: int | None = None
        self._last_seen: dict[int, int] = {}
        self._superframes = 0
        #: Whether the call on each slot is encrypted (EMB privacy indicator).
        self.encrypted = {1: False, 2: False}
        self.frames = 0
        self.muted_frames = 0

    def reset(self) -> None:
        self._finder.reset()
        self._current = None

    def _bursts(self, symbols: np.ndarray) -> list[np.ndarray]:
        """The superframe's voice bursts, in order, as bits; stopping at a place that
        holds a data burst (the call ended)."""
        out = []
        sync = _data_sync_dibits()
        for k in range(SUPERFRAME_BURSTS):
            start = k * 2 * BURST_SYMBOLS
            burst = symbols[start:start + BURST_SYMBOLS]
            if burst.size < BURST_SYMBOLS:
                break
            dibits = to_dibits(burst)
            if k and np.count_nonzero(dibits[LEAD_SYMBOLS:LEAD_SYMBOLS + 24] == sync) \
                    >= DATA_SYNC_MATCH:
                break
            out.append(dibits_to_bits(dibits))
        return out

    def _slot_of(self, source: str, bits: np.ndarray) -> int | None:
        if source == "dm1":
            return 1
        if source == "dm2":
            return 2
        tact = _TACT.get(bits_to_int(bits[TACT_POSITIONS]))
        return None if tact is None else (tact >> 2 & 1) + 1

    def process(self, x: np.ndarray) -> np.ndarray:
        pieces = []
        for hit, symbols in self._finder.process(x):
            if hit.sign > 0:
                continue                                  # a data sync: not a voice burst A
            bursts = self._bursts(symbols)
            if not bursts:
                continue
            slot = self._slot_of(hit.name, bursts[0]) or 1
            self._superframes += 1
            self._last_seen[slot] = self._superframes
            if self.slot and slot != self.slot:
                continue
            if not self.slot:
                current = self._current
                # Keep the slot already playing while it is still talking.
                if current is not None and current != slot and \
                        self._superframes - self._last_seen.get(current, -99) <= 2:
                    continue
                self._current = slot
            if len(bursts) > 1:
                privacy = [emb_privacy(b) for b in bursts[1:]]
                self.encrypted[slot] = sum(privacy) * 2 > len(privacy)
            codec = self._codecs[slot]
            for bits in bursts:
                for frame in burst_frames(bits):
                    speech = codec.decode(deinterleave(frame))
                    self.frames += 1
                    if self.encrypted[slot]:
                        self.muted_frames += 1
                        speech = np.zeros_like(speech)
                    pieces.append(speech)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)
