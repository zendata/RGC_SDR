"""POCSAG pager decoding (ITU-R M.584), from the NBFM discriminator.

Two-level FSK at 512, 1200 or 2400 baud, all three sliced in parallel since a channel
may carry any of them, and either polarity accepted. A batch is the sync codeword
followed by 8 frames of 2 codewords. Each 32-bit codeword is a BCH(31,21) code plus an
even parity bit; up to two bit errors are corrected from a syndrome table.

Privacy (PLANNING.md 7p): message text can carry names, addresses and medical details.
`PagerMessage.summary` leaves it out unless asked, and nothing here stores or logs it.

Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .bitsync import BitSlicer

BAUDS = (512, 1200, 2400)
SYNC = 0x7CD215D8
IDLE = 0x7A89C197
#: BCH(31,21) generator: x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1.
GENERATOR = 0b11101101001
#: Bit errors tolerated in the sync codeword itself.
SYNC_TOLERANCE = 2
BATCH_BITS = 32 + 16 * 32
NUMERIC = "0123456789*U -)("


def _syndrome_one(word31: int) -> int:
    """Remainder of a 31-bit word divided by the generator."""
    for bit in range(30, 9, -1):
        if word31 >> bit & 1:
            word31 ^= GENERATOR << (bit - 10)
    return word31


#: Syndrome contributed by each of the 31 code bits (bit i of the 31-bit word).
_BIT_SYNDROMES = np.array([_syndrome_one(1 << i) for i in range(31)], dtype=np.int64)


def syndromes(codewords: np.ndarray) -> np.ndarray:
    """BCH syndromes of 32-bit codewords (the parity bit, bit 0, is not part of it)."""
    words = np.asarray(codewords, dtype=np.int64) >> 1
    bits = (words[:, None] >> np.arange(31)) & 1
    return np.bitwise_xor.reduce(np.where(bits == 1, _BIT_SYNDROMES, 0), axis=1)


def _build_corrections() -> dict[int, int]:
    """Syndrome -> 32-bit error pattern, for every one- and two-bit error."""
    table = {}
    for i in range(31):
        table[int(_BIT_SYNDROMES[i])] = 1 << (i + 1)
    for i, j in combinations(range(31), 2):
        table[int(_BIT_SYNDROMES[i] ^ _BIT_SYNDROMES[j])] = (1 << (i + 1)) | (1 << (j + 1))
    return table


_CORRECTIONS = _build_corrections()


def correct(codeword: int) -> tuple[int, int] | None:
    """(corrected codeword, bits fixed), or None when it is beyond repair."""
    s = int(syndromes(np.array([codeword]))[0])
    fixed = 0
    if s:
        pattern = _CORRECTIONS.get(s)
        if pattern is None:
            return None
        codeword ^= pattern
        fixed = pattern.bit_count()
    if codeword.bit_count() % 2:
        if fixed == 2:
            return None            # a third error, somewhere
        codeword ^= 1              # the parity bit itself was wrong
        fixed += 1
    return codeword, fixed


def _reverse(value: int, width: int) -> int:
    return int(f"{value:0{width}b}"[::-1], 2)


def numeric_text(chunks: list[int]) -> str:
    """Numeric messages: 4-bit BCD, each digit sent least significant bit first."""
    out = []
    for chunk in chunks:
        for shift in (16, 12, 8, 4, 0):
            out.append(NUMERIC[_reverse(chunk >> shift & 0xF, 4)])
    return "".join(out).rstrip()


#: Control codes that end an alphanumeric message (ETX, EOT).
_ALPHA_END = (3, 4)


def _alpha_codes(chunks: list[int]) -> tuple[list[int], bool]:
    """7-bit characters, least significant bit first, packed across the 20-bit chunks
    without regard to codeword boundaries, up to the end-of-text code; and whether
    that code was found."""
    stream = "".join(f"{c:020b}" for c in chunks)
    codes = []
    for i in range(0, len(stream) - 6, 7):
        code = int(stream[i:i + 7][::-1], 2)
        if code in _ALPHA_END:
            return codes, True
        codes.append(code)
    return codes, False


def alpha_text(chunks: list[int]) -> str:
    """Alphanumeric messages, printable characters and line breaks only."""
    text = "".join(chr(c) if 32 <= c < 127 else ("\n" if c == 10 else "")
                   for c in _alpha_codes(chunks)[0])
    return text.rstrip()


#: How much of a numeric reading must be digits, spaces and dashes to believe it.
NUMERIC_CONFIDENCE = 0.95


def classify(chunks: list[int]) -> tuple[str, str]:
    """("numeric" or "alpha", text), judged by the content, not the function code.

    Function 0 is only *conventionally* numeric. Measured 2026-10-05 on a Melbourne
    network at 148.68 MHz: its function-0 pages were text (56 % dictionary words read
    as alpha), and reading them as numeric gave digits strewn with U * ( ) -. A real
    numeric page is all digits, spaces and dashes; text read as numeric is not, and
    digits read as text are full of control codes.
    """
    codes, ended = _alpha_codes(chunks)
    printable = sum(32 <= c < 127 or c in (10, 13) for c in codes) / max(1, len(codes))
    # A text page fills its codewords up to the last, so its end code lies in the final
    # one; an "end code" with whole codewords still to come is digits read as text.
    ends_in_last = ended and len(codes) * 7 + 7 > (len(chunks) - 1) * 20
    clean_text = bool(codes) and ends_in_last and printable == 1.0
    numeric = numeric_text(chunks)
    digits = sum(c in "0123456789 -" for c in numeric) / max(1, len(numeric))
    clean_digits = bool(numeric) and digits >= NUMERIC_CONFIDENCE and digits >= printable
    if clean_text and clean_digits:
        # Both readings look right, which only happens for very short pages: a couple
        # of letters is the likelier accident, so a few characters are asked of text.
        return ("alpha", alpha_text(chunks)) if len(codes) >= 3 else ("numeric", numeric)
    if clean_text:
        # Properly closed text: its padding alone could read as a run of zeros.
        return "alpha", alpha_text(chunks)
    if clean_digits:
        return "numeric", numeric
    return "alpha", alpha_text(chunks)


@dataclass
class PagerMessage:
    baud: int
    address: int
    function: int
    kind: str                     # "numeric", "alpha" or "tone"
    text: str = field(repr=False)  # kept out of reprs and logs on purpose
    corrected_bits: int = 0
    received: float = field(default_factory=time.time)

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        head = (f"{stamp}  POCSAG{self.baud}  addr {self.address:7d}  func {self.function}  "
                f"{self.kind}")
        if self.kind == "tone":
            return head
        body = self.text if show_text else f"[{len(self.text)} characters hidden]"
        return f"{head}  {body}"


class _Framer:
    """Sync search and codeword assembly for one baud rate."""

    def __init__(self, baud: int) -> None:
        self.baud = baud
        self._bits = np.zeros(0, dtype=np.uint8)
        self._pending: dict | None = None
        self._since_batch = 0
        self.batches = 0

    def reset(self) -> None:
        self._bits = np.zeros(0, dtype=np.uint8)
        self._pending = None
        self._since_batch = 0

    def feed(self, bits: np.ndarray) -> list[PagerMessage]:
        out: list[PagerMessage] = []
        buf = np.concatenate([self._bits, bits])
        pos = 0
        while buf.size - pos >= 32:
            window = sliding_window_view(buf[pos:], 32)
            words = window.astype(np.int64) @ (1 << np.arange(31, -1, -1, dtype=np.int64))
            normal = np.bitwise_count(words ^ SYNC) <= SYNC_TOLERANCE
            inverted = np.bitwise_count(words ^ (~SYNC & 0xFFFFFFFF)) <= SYNC_TOLERANCE
            hits = np.flatnonzero(normal | inverted)
            if hits.size == 0:
                self._since_batch += buf.size - 31 - pos
                pos = buf.size - 31
                break
            start = pos + int(hits[0])
            if buf.size - start < BATCH_BITS:
                pos = start
                break
            batch = buf[start + 32:start + BATCH_BITS]
            if inverted[hits[0]]:
                batch = 1 - batch
            out.extend(self._batch(batch))
            self.batches += 1
            self._since_batch = 0
            pos = start + BATCH_BITS
        self._bits = buf[pos:]
        if self._pending is not None and self._since_batch > 2 * BATCH_BITS:
            out.extend(self._flush())       # the transmission ended mid-message
        return out

    def _batch(self, bits: np.ndarray) -> list[PagerMessage]:
        out = []
        words = bits.reshape(16, 32).astype(np.int64) @ (
            1 << np.arange(31, -1, -1, dtype=np.int64))
        for i, raw in enumerate(words):
            fixed = correct(int(raw))
            if fixed is None:
                out.extend(self._flush())   # cannot trust what follows in this message
                continue
            word, nbits = fixed
            if word in (IDLE, 0):
                # All zeros is a valid codeword (address 0, function 0) that no pager
                # uses, but a dead carrier slices to exactly that.
                out.extend(self._flush())
            elif word >> 31 == 0:
                out.extend(self._flush())
                self._pending = {
                    "address": ((word >> 13) & 0x3FFFF) << 3 | (i // 2),
                    "function": (word >> 11) & 0x3,
                    "chunks": [],
                    "fixed": nbits,
                }
            elif self._pending is not None:
                self._pending["chunks"].append((word >> 11) & 0xFFFFF)
                self._pending["fixed"] += nbits
        return out

    def _flush(self) -> list[PagerMessage]:
        pending, self._pending = self._pending, None
        if pending is None:
            return []
        chunks = pending["chunks"]
        if not chunks:
            kind, text = "tone", ""
        else:
            kind, text = classify(chunks)
        return [PagerMessage(self.baud, pending["address"], pending["function"], kind, text,
                             pending["fixed"])]


class PocsagDecoder:
    """Discriminator samples in, pager messages out, at all three rates at once."""

    name = "POCSAG"

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = float(sample_rate)
        self._slicers = [BitSlicer(self.sample_rate, b) for b in BAUDS]
        self._framers = [_Framer(b) for b in BAUDS]

    def reset(self) -> None:
        for slicer, framer in zip(self._slicers, self._framers):
            slicer.reset()
            framer.reset()

    def process(self, x: np.ndarray) -> list[PagerMessage]:
        out: list[PagerMessage] = []
        for slicer, framer in zip(self._slicers, self._framers):
            out.extend(framer.feed(slicer.process(x)))
        return out
