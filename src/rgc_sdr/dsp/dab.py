"""DAB (ETSI EN 300 401), transmission mode I: OFDM to the Fast Information Channel.

Stage 1 of DAB+ (VK3RQ, 2026-10-07): the ensemble's name, its services' names and how
the multiplex is laid out -- everything in the FIC, which needs no audio codec.

Mode I at 2.048 MS/s: a 96 ms frame is a null symbol (2656 samples) then 76 OFDM
symbols of 2552 samples (504 guard + 2048 useful), 1536 carriers 1 kHz apart either
side of the centre. Symbol 1 is the phase reference; data is differentially QPSK
modulated against the previous symbol, so the reference's own table is not needed.
Symbols 2-4 carry the FIC: 9216 bits, four blocks of 2304, each a punctured rate 1/4
convolutional code (K=7) of three 256-bit FIBs, energy-dispersed, each FIB with its
own CRC-16.

Pure NumPy; no Qt, no device access. The Viterbi decoder loops over trellis steps, not
samples, and runs every block of a frame side by side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

RATE = 2.048e6
NULL = 2656
TU = 2048
GUARD = 504
SYMBOL = TU + GUARD
SYMBOLS = 76                       # phase reference + 75 data symbols
FRAME = NULL + SYMBOLS * SYMBOL    # 196608 samples, 96 ms
K = 1536                           # carriers
FIC_SYMBOLS = 3
#: While a station plays, the FIC is read every this many frames (about 0.4 s).
FIC_EVERY = 4
FIC_BLOCK_BITS = 2304
FIB_BITS = 256


# -- frequency interleaving -----------------------------------------------------------

def _carrier_order() -> np.ndarray:
    """Carrier index (-768..768, no 0) for each QPSK symbol n, from the standard's
    permutation: PI(i) = (13 PI(i-1) + 511) mod 2048, keeping 256..1792 but 1024."""
    pi, out = 0, []
    for _ in range(2048):
        if 256 <= pi <= 1792 and pi != 1024:
            out.append(pi - 1024)
        pi = (13 * pi + 511) % 2048
    return np.array(out)


CARRIERS = _carrier_order()
#: FFT bin (of a 2048-point FFT, DC at 0) holding QPSK symbol n.
_BINS = CARRIERS % TU


# -- the convolutional code ----------------------------------------------------------

#: Taps (delays) of the four generators, octal 133, 171, 145, 133.
_TAPS = ((0, 2, 3, 5, 6), (0, 1, 2, 3, 6), (0, 1, 4, 6), (0, 2, 3, 5, 6))


def _outputs() -> np.ndarray:
    """[state, input] -> the four output bits; state = the last six inputs, newest
    in bit 5."""
    out = np.zeros((64, 2, 4), dtype=np.int8)
    for s in range(64):
        for b in range(2):
            reg = [b] + [(s >> (5 - i)) & 1 for i in range(6)]       # a_i, a_{i-1}..a_{i-6}
            for g, taps in enumerate(_TAPS):
                out[s, b, g] = sum(reg[t] for t in taps) & 1
    return out


_OUT = _outputs()
_NEXT = np.array([[(b << 5) | (s >> 1) for b in range(2)] for s in range(64)])


def convolve(bits: np.ndarray) -> np.ndarray:
    """Mother code with six zero tail bits: n bits -> 4 (n + 6) bits."""
    state, out = 0, []
    for b in list(bits) + [0] * 6:
        out.extend(_OUT[state, int(b)])
        state = _NEXT[state, int(b)]
    return np.array(out, dtype=np.uint8)


def _trellis() -> tuple[np.ndarray, np.ndarray]:
    """Predecessors of each state -- the two states whose shift leads here -- and the
    input bit that led there."""
    prev = np.zeros((64, 2), dtype=np.int64)
    prev_bit = np.zeros(64, dtype=np.int64)
    for s in range(64):
        for b in range(2):
            prev[_NEXT[s, b]][s & 1] = s
            prev_bit[_NEXT[s, b]] = b
    return prev, prev_bit


_PREV, _PREV_BIT = _trellis()
_EXPECT = (1.0 - 2.0 * _OUT.astype(np.float64)).reshape(128, 4).T          # [4, 128]


def viterbi(soft: np.ndarray) -> np.ndarray:
    """Decode a batch of codewords at once: soft is [batch, 4 * (n + 6)], +1 for a
    likely 0 and -1 for a likely 1, 0 for a punctured (unknown) bit. -> [batch, n].

    State n = 32 b + j is reached from states 2j and 2j + 1 with input b, so with the
    metrics viewed as [32, 2] both candidates for every state come from one broadcast
    add, and every branch metric from one matrix product before the trellis is walked.
    The step loop's NumPy call overhead had been most of the cost, and made DAB+ too slow
    for a Raspberry Pi 5 (measured 2026-10-10: 9.9 s to decode 6 s of an ensemble)."""
    soft = np.atleast_2d(soft)
    batch, steps = soft.shape[0], soft.shape[1] // 4
    gain = soft[:, : steps * 4].reshape(batch, steps, 4).astype(np.float64) @ _EXPECT
    # Columns are [state s = 2j + k, input b]; -> [step, batch, b, j, k].
    gain = np.ascontiguousarray(gain.reshape(batch, steps, 32, 2, 2).transpose(1, 0, 4, 2, 3))
    metric = np.full((batch, 1, 32, 2), -1e9)
    metric[:, 0, 0, 0] = 0.0
    decisions = np.zeros((steps, batch, 2, 32), dtype=bool)
    for t in range(steps):
        cand = metric + gain[t]                                          # [batch, 2, 32, 2]
        decisions[t] = cand[..., 1] > cand[..., 0]   # a tie keeps state 2j, as argmax did
        metric = cand.max(axis=3).reshape(batch, 1, 32, 2)
    decisions = decisions.reshape(steps, batch, 64)
    state = np.zeros(batch, dtype=np.int64)                              # tail ends at 0
    bits = np.zeros((batch, steps), dtype=np.uint8)
    rows = np.arange(batch)
    for t in range(steps - 1, -1, -1):
        bits[:, t] = state >> 5
        state = 2 * (state & 31) + decisions[t, rows, state]
    return bits[:, : steps - 6]


# -- puncturing -------------------------------------------------------------------------

#: Puncturing vectors V_PI (EN 300 401 table 13): which of 32 mother-code bits are sent.
#: Copied from the standard: a rule that spread the ones evenly matched only some of them
#: (PI 1, 8, 15, 16, 24), not PI 2, 9 and most others.
_TABLE_13 = (
    "11001000100010001000100010001000", "11001000100010001100100010001000",
    "11001000110010001100100010001000", "11001000110010001100100011001000",
    "11001100110010001100100011001000", "11001100110010001100110011001000",
    "11001100110011001100110011001000", "11001100110011001100110011001100",
    "11101100110011001100110011001100", "11101100110011001110110011001100",
    "11101100111011001110110011001100", "11101100111011001110110011101100",
    "11101110111011001110110011101100", "11101110111011001110111011101100",
    "11101110111011101110111011101100", "11101110111011101110111011101110",
    "11111110111011101110111011101110", "11111110111011101111111011101110",
    "11111110111111101111111011101110", "11111110111111101111111011111110",
    "11111111111111101111111011111110", "11111111111111101111111111111110",
    "11111111111111111111111111111110", "11111111111111111111111111111111",
)


PI = {x: np.array([c == "1" for c in v]) for x, v in enumerate(_TABLE_13, start=1)}
assert all(PI[x].sum() == 8 + x for x in PI)
#: The tail's vector: 12 of 24 bits.
PI_TAIL = np.array([1, 1, 0, 0] * 6, dtype=bool)


def fic_pattern() -> np.ndarray:
    """Which of a FIC block's 3096 mother-code bits are sent: 21 x 128 with PI_16,
    3 x 128 with PI_15, then the tail -- 2304 in all."""
    parts = [np.tile(PI[16], 4)] * 21 + [np.tile(PI[15], 4)] * 3 + [PI_TAIL]
    return np.concatenate(parts)


FIC_PATTERN = fic_pattern()
assert FIC_PATTERN.sum() == FIC_BLOCK_BITS


def depuncture(soft: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Put received soft bits back where the mother code had them; 0 elsewhere."""
    soft = np.atleast_2d(soft)
    out = np.zeros((soft.shape[0], pattern.size))
    out[:, pattern] = soft
    return out


# -- energy dispersal and CRC ------------------------------------------------------------

def prbs(n: int) -> np.ndarray:
    """x^9 + x^5 + 1 from all ones: the energy-dispersal sequence."""
    reg = [1] * 9
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        bit = reg[8] ^ reg[4]
        out[i] = bit
        reg = [bit] + reg[:8]
    return out


_PRBS_FIC = prbs(768)


def crc16(bits) -> int:
    """CRC-CCITT, initial all ones, result inverted (the FIB check)."""
    r = 0xFFFF
    for b in bits:
        fb = ((r >> 15) ^ int(b)) & 1
        r = (r << 1) & 0xFFFF
        if fb:
            r ^= 0x1021
    return r ^ 0xFFFF


# -- FIGs ------------------------------------------------------------------------------

def _label(data: bytes) -> str:
    return data.decode("latin-1", errors="replace").rstrip(" \x00")


@dataclass
class Ensemble:
    """What the FIC has said so far."""

    eid: int | None = None
    label: str = ""
    services: dict[int, str] = field(default_factory=dict)
    #: SId -> subchannel id of its primary audio component.
    components: dict[int, int] = field(default_factory=dict)
    #: subchannel id -> (start address in CUs, size in CUs, EEP option (0 = A, 1 = B, -1
    #: for UEP), protection level 1-4, protection text)
    subchannels: dict[int, tuple[int, int, int, int, str]] = field(default_factory=dict)
    #: SId -> whether its audio is DAB+ (ASCTy 63) rather than DAB (MP2).
    dab_plus: dict[int, bool] = field(default_factory=dict)


#: EEP subchannel sizes per CU for each protection level: (option, level) -> n per 8 kbit/s
#: (EN 300 401 6.2.1, Table 7): option 0 (A) levels 1-4 use 12, 8, 6, 4 CU per n.
_EEP_A_CU = {1: 12, 2: 8, 3: 6, 4: 4}
_EEP_B_CU = {1: 27, 2: 21, 3: 18, 4: 15}


def parse_fib(data: bytes, ensemble: Ensemble) -> None:
    """Fold one FIB's FIGs into `ensemble`."""
    i = 0
    while i < 30:
        header = data[i]
        if header == 0xFF:
            break
        kind, length = header >> 5, header & 31
        body = data[i + 1:i + 1 + length]
        i += 1 + length
        if len(body) < length or length == 0:
            break
        if kind == 0:
            _fig0(body, ensemble)
        elif kind == 1:
            _fig1(body, ensemble)


def _fig0(body: bytes, e: Ensemble) -> None:
    pd, ext = body[0] >> 5 & 1, body[0] & 31
    d = body[1:]
    if ext == 0 and len(d) >= 2:
        e.eid = d[0] << 8 | d[1]
    elif ext == 1:
        j = 0
        while j + 3 <= len(d):
            sub = d[j] >> 2
            start = (d[j] & 3) << 8 | d[j + 1]
            if d[j + 2] & 0x80:                                     # long form: EEP
                option, level = d[j + 2] >> 4 & 7, (d[j + 2] >> 2 & 3) + 1
                size = (d[j + 2] & 3) << 8 | d[j + 3]
                e.subchannels[sub] = (start, size, option, level,
                                      f"EEP {level}-{'AB'[option] if option < 2 else '?'}")
                j += 4
            else:                                                   # short form: UEP
                e.subchannels[sub] = (start, 0, -1, 0, f"UEP table {d[j + 2] & 63}")
                j += 3
    elif ext == 2:
        j = 0
        wide = pd == 1
        while j + (4 if wide else 2) < len(d):
            sid_len = 4 if wide else 2
            sid = int.from_bytes(d[j:j + sid_len], "big")
            count = d[j + sid_len] & 15
            j += sid_len + 1
            for _ in range(count):
                if j + 2 > len(d):
                    return
                tmid = d[j] >> 6
                if tmid == 0:                                       # audio in a subchannel
                    ascty, sub, primary = d[j] & 63, d[j + 1] >> 2, d[j + 1] >> 1 & 1
                    if primary:
                        e.components[sid] = sub
                        e.dab_plus[sid] = ascty == 63
                j += 2


def _fig1(body: bytes, e: Ensemble) -> None:
    ext = body[0] & 7
    if ext == 0 and len(body) >= 19:
        e.eid = body[1] << 8 | body[2]
        e.label = _label(body[3:19])
    elif ext == 1 and len(body) >= 19:
        e.services[body[1] << 8 | body[2]] = _label(body[3:19])


# -- the receiver ---------------------------------------------------------------------

@dataclass
class DabMessage:
    text: str
    ensemble: str = ""
    fields: dict = field(default_factory=dict)
    channel: str = ""
    received: float = field(default_factory=time.time)

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        return f"{stamp}  DAB  {self.text}"


class DabReceiver:
    """IQ at 2.048 MS/s, centred on the ensemble, in; FIC contents out."""

    name = "DAB"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        # 2.048 MS/s times a power of two: faster, then halved here through a proper
        # filter. *Measured:* at 2.048 MS/s the HackRF's analogue filter (1.75 MHz at its
        # narrowest) lets the next ensemble, 1.712 MHz away, fold onto this one -- 9A
        # and 9B decoded almost nothing until sampled at 4.096 MS/s.
        ratio = sample_rate / RATE
        factor = int(round(ratio))
        if factor < 1 or abs(ratio - factor) > 1e-6 or factor & (factor - 1):
            raise ValueError(f"DAB needs {RATE / 1e6:g} MS/s or a power-of-two multiple "
                             f"(4.096 is best); this is {sample_rate / 1e6:g} MS/s")
        from .decimate import StreamDecimator

        self._decimator = StreamDecimator(factor) if factor > 1 else None
        #: How far into the guard interval each FFT starts, in samples.
        self.backoff = GUARD // 4
        #: Share of samples at full scale lately: over a few tenths of a percent, the
        #: radio's gain is too high for this signal.
        self.clipped = 0.0
        self.channel = channel
        self.ensemble = Ensemble()
        self.frames = 0
        self.good_fibs = 0
        self.bad_fibs = 0
        #: Carrier frequency offset found, in Hz (after the radio's own correction).
        self.offset_hz = 0.0
        self._reported: set = set()
        #: The station being listened to (DAB+), and its sub-channel decoder.
        self.service: int | None = None
        self._audio = None
        self._pcm: list[np.ndarray] = []
        self.reset()

    def select(self, sid: int | None) -> bool:
        """Listen to service `sid` (DAB+ only). False if the FIC has not yet said where
        it is, or it is not DAB+."""
        from .dabplus import DabPlusAudio, codec_available

        self.service, self._audio = sid, None
        if sid is None:
            return True
        sub = self.ensemble.components.get(sid)
        layout = self.ensemble.subchannels.get(sub) if sub is not None else None
        if layout is None or layout[2] < 0 or not self.ensemble.dab_plus.get(sid):
            return False
        if not codec_available():
            return False
        self._audio = DabPlusAudio(layout[0], layout[1], layout[2], layout[3])
        return True

    @property
    def audio(self):
        """The selected station's sub-channel decoder (rates, error counts), or None."""
        return self._audio

    def take_audio(self) -> np.ndarray:
        """Audio decoded since last asked: [n, 2] at `audio.sample_rate`."""
        if not self._pcm:
            return np.zeros((0, 2), dtype=np.float32)
        out, self._pcm = np.concatenate(self._pcm), []
        return out

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.complex64)
        self._offset: int | None = None
        self._last_candidate: int | None = None

    # -- synchronisation
    def _find_frame(self, x: np.ndarray) -> int | None:
        """Start of the first null symbol in x (the quietest NULL-long stretch)."""
        if x.size < FRAME + NULL:
            return None
        power = np.abs(x[:FRAME + NULL]) ** 2
        c = np.cumsum(np.concatenate([[0.0], power]))
        window = c[NULL:] - c[:-NULL]
        start = int(np.argmin(window[:FRAME]))
        if window[start] > 0.3 * np.mean(window):
            return None                                             # no clear null: no DAB
        return start

    def _fine(self, x: np.ndarray, start: int) -> tuple[int, float]:
        """Refine the symbol timing and find the fractional frequency offset from the
        guard intervals (each is a copy of its symbol's end)."""
        best, best_mag, phase = start, -1.0, 0.0
        for shift in range(-32, 33):
            corr = 0j
            for l in range(1, 6):
                s = start + NULL + shift + l * SYMBOL
                a = x[s:s + GUARD]
                b = x[s + TU:s + TU + GUARD]
                corr += np.vdot(a, b)
            if abs(corr) > best_mag:
                best, best_mag, phase = start + shift, abs(corr), np.angle(corr)
        return best, phase / (2 * np.pi)                         # offset in carrier units

    def process(self, iq: np.ndarray) -> list[DabMessage]:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size:
            # Share of samples at full scale. *Measured:* the HackRF at its default gains
            # clipped 20-30 % of samples on Melbourne's Band III, and decoded little.
            clipped = float(np.mean((np.abs(iq.real) > 0.95) | (np.abs(iq.imag) > 0.95)))
            self.clipped += 0.2 * (clipped - self.clipped)
        if self._decimator is not None:
            iq = self._decimator.process(iq).astype(np.complex64)
        self._buf = np.concatenate([self._buf, iq])
        out: list[DabMessage] = []
        while self._buf.size >= 2 * FRAME + NULL:
            start = self._find_frame(self._buf)
            if start is None:
                self._buf = self._buf[FRAME:]
                continue
            start, frac = self._fine(self._buf, start)
            if start < 0:
                self._buf = self._buf[FRAME:]
                continue
            frame = self._buf[start:start + FRAME]
            self._buf = self._buf[start + FRAME - NULL:]
            out += self._frame(frame, frac)
        return out

    def _frame(self, frame: np.ndarray, frac: float) -> list[DabMessage]:
        # FFT each symbol's useful part, starting a little into the guard (a constant
        # phase per carrier, which the differential demodulation cancels).
        backoff = self.backoff
        # Every symbol, even for the FIC alone: the carriers' reliability is judged over
        # the whole frame.
        count = SYMBOLS
        starts = NULL + np.arange(count) * SYMBOL + GUARD - backoff
        # The whole-carrier offset, from eight symbols' power across the frame and held
        # unless a new value repeats: *measured*, one symbol's power put single frames 6
        # or 12 carriers out (and lost them) while the true offset never moved.
        sample = starts[:: max(1, count // 8)][:8]
        rotate = np.exp(-2j * np.pi * frac * np.arange(TU) / TU)
        power = sum(np.abs(np.fft.fft(frame[s:s + TU] * rotate)) ** 2 for s in sample)
        candidate = self._integer_offset(power)
        if self._offset is None or candidate == self._last_candidate:
            self._offset = candidate
        self._last_candidate = candidate
        offset = self._offset
        # The whole offset comes out in time, not by moving bins: an offset also turns
        # each carrier's phase from one symbol to the next (2552 samples is not a whole
        # number of its cycles), which would wreck the differential QPSK.
        # exp(-j w (s + m)) = exp(-j w s) exp(-j w m): one symbol's ramp and a phase per
        # symbol, not an exponential across the whole frame (a fifth of a Pi 5's time).
        w = 2 * np.pi * (frac + offset) / TU
        ramp = np.exp(-1j * w * np.arange(TU)).astype(np.complex64)
        phase = np.exp(-1j * w * starts).astype(np.complex64)[:, None]
        windows = frame[starts[:, None] + np.arange(TU)] * ramp * phase
        spectra = np.fft.fft(windows, axis=1)
        cells = spectra[:, _BINS]                                   # [symbol, n]
        z = cells[1:] * np.conj(cells[:-1])                         # differential
        # Channel-state weighting: each carrier by how well its points sit on the QPSK
        # constellation over the frame (z^4 takes the data out: an ideal point gives -1).
        # *Measured:* the carriers next to the HackRF's LO leak (+200 kHz) erred 20-50 %,
        # always in the same FIC positions, and cost alternate FIBs every frame -- the
        # code corrects scattered errors, not ones that recur in one place.
        unit = z / (np.abs(z) + 1e-12)
        quality = np.clip(-np.mean((unit ** 4).real, axis=0), 0.0, 1.0)     # [n]
        scale = np.mean(np.abs(z)) + 1e-12
        weighted = z * quality / scale
        soft = np.concatenate([weighted.real, weighted.imag], axis=1)   # [symbols, 3072]
        self.frames += 1
        self.offset_hz = (offset + frac) * RATE / TU
        # The FIC every frame until a station plays, then every FIC_EVERY: the station's
        # place in the multiplex is already known, and the FIC's Viterbi decoding is half
        # the work (too much for a Raspberry Pi 5 every frame, measured 2026-10-10).
        messages = (self._fic(soft[:FIC_SYMBOLS].ravel())
                    if self._audio is None or self.frames % FIC_EVERY == 1 else [])
        if self._audio is not None:
            # The MSC: 72 symbols, four 24 ms CIFs of 55296 bits; the station's
            # sub-channel is its CUs (64 bits each) in every CIF.
            cifs = soft[FIC_SYMBOLS:].ravel().reshape(4, 55296)
            a, b = self._audio.start_cu * 64, (self._audio.start_cu + self._audio.size_cu) * 64
            pcm = self._audio.push([cif[a:b].astype(np.float32) for cif in cifs])
            if pcm.size:
                self._pcm.append(pcm)
        return messages

    def _integer_offset(self, power: np.ndarray) -> int:
        """Whole carriers the ensemble sits off centre: where the 1536 occupied carriers
        hold the most power. (Fooled by the neighbouring ensembles when they had folded
        in at 2.048 MS/s; sampled at 4.096 and filtered, they are gone.)"""
        best, best_power = 0, -1.0
        for k in range(-20, 21):
            p = power[(_BINS + k) % TU].sum()
            if p > best_power:
                best, best_power = k, p
        return best

    def _fic(self, soft: np.ndarray) -> list[DabMessage]:
        blocks = soft.reshape(4, FIC_BLOCK_BITS)
        bits = viterbi(depuncture(blocks, FIC_PATTERN)) ^ _PRBS_FIC   # [4, 768]
        before = (self.ensemble.label, dict(self.ensemble.services))
        for block in bits:
            for f in range(3):
                fib = block[f * FIB_BITS:(f + 1) * FIB_BITS]
                if crc16(fib[:240]) != int("".join(map(str, fib[240:])), 2):
                    self.bad_fibs += 1
                    continue
                self.good_fibs += 1
                parse_fib(np.packbits(fib[:240]).tobytes(), self.ensemble)
        return self._news(before)

    def _news(self, before) -> list[DabMessage]:
        e = self.ensemble
        out = []
        if e.label and e.label != before[0]:
            out.append(DabMessage(f"ensemble {e.label}" + (f" ({e.eid:04X})" if e.eid else ""),
                                  e.label, {"eid": e.eid}))
        for sid, name in e.services.items():
            key = (sid, name, e.components.get(sid))
            if key in self._reported or sid not in e.components:
                continue
            self._reported.add(key)
            sub = e.components[sid]
            kind = "DAB+" if e.dab_plus.get(sid) else "DAB"
            layout = e.subchannels.get(sub)
            extra = f", {layout[1]} CU, {layout[4]}" if layout and layout[1] else ""
            out.append(DabMessage(f"{name}  ({kind}, service {sid:04X}, subchannel {sub}{extra})",
                                  e.label, {"sid": sid, "subchannel": sub}))
        return out
