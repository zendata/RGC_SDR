"""Four-level FSK sync and symbol recovery, shared by P25 and DMR."""

import numpy as np

from src.rgc_sdr.dsp.fsk4 import (
    DIBIT_SYMBOL, RepeatFilter, SyncFinder, bits_to_int, crc_ccitt, dibits_to_bits,
    sync_symbols, to_dibits,
)

RATE = 62_500.0           # the NBFM chain's IF rate at 2 MS/s: 13.02 samples per symbol
SYNC = 0x5575F5FF77FF


def word_dibits(word: int, n: int) -> list[int]:
    return [(word >> (2 * (n - 1 - i))) & 3 for i in range(n)]


def fsk4_wave(dibits, rate=RATE, unit=0.24, dc=0.08, noise=0.04, lead=500, seed=1):
    """Discriminator output for a run of dibits: `unit` per symbol step (600 Hz on a
    2500 Hz scale, as P25), smoothed like a shaped transmitter, off tune by `dc`, with
    noise. Starts `lead` samples in."""
    sps = rate / 4800
    symbols = np.array([DIBIT_SYMBOL[int(d)] for d in dibits], float)
    n = lead + int(np.ceil(symbols.size * sps)) + lead
    idx = np.floor((np.arange(n) - lead) / sps).astype(int)
    inside = (idx >= 0) & (idx < symbols.size)
    x = np.zeros(n)
    x[inside] = symbols[idx[inside]] * unit
    k = max(1, int(sps * 0.6))
    x = np.convolve(x, np.ones(k) / k, mode="same")
    rng = np.random.default_rng(seed)
    return x + dc + noise * rng.standard_normal(n)


def run(decoder, x, block=3125):
    found = []
    for i in range(0, x.size, block):
        found += decoder.process(x[i:i + block])
    return found


def test_crc_ccitt_matches_the_published_check_value():
    # CRC-16/XMODEM (poly 0x1021, init 0) of "123456789" is 0x31C3.
    bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    assert crc_ccitt(bits) == 0x31C3


def test_dibit_helpers():
    assert list(sync_symbols(SYNC)[:6]) == [3, 3, 3, 3, 3, -3]
    assert list(dibits_to_bits([0b01, 0b10])) == [0, 1, 1, 0]
    assert bits_to_int([1, 0, 1, 1]) == 11
    assert list(to_dibits(np.array([2.6, 0.7, -1.2, -3.4]))) == [0b01, 0b00, 0b10, 0b11]


def test_finds_frames_across_any_block_boundaries():
    payload = list(np.random.default_rng(3).integers(0, 4, 200))
    dibits = (word_dibits(SYNC, 24) + payload) * 3
    x = fsk4_wave(dibits)
    finder = SyncFinder(RATE, {"fs": SYNC}, frame_symbols=224)
    found = []
    for i in range(0, x.size, 997):                       # boundaries anywhere
        found += finder.process(x[i:i + 997])
    assert len(found) == 3
    for hit, symbols in found:
        assert hit.sign == 1
        assert list(to_dibits(symbols)[24:]) == payload


def test_scale_and_offset_come_from_the_sync():
    payload = [0, 1, 2, 3] * 10
    x = -fsk4_wave(word_dibits(SYNC, 24) + payload, unit=0.31, dc=-0.3)
    finder = SyncFinder(RATE, {"fs": SYNC}, frame_symbols=64)
    (hit, symbols), = finder.process(x)
    assert hit.sign == -1                                  # upside down, and still read
    assert list(to_dibits(symbols)[24:]) == payload


def test_noise_finds_no_syncs():
    x = 0.3 * np.random.default_rng(4).standard_normal(int(RATE * 5))
    assert SyncFinder(RATE, {"fs": SYNC}, frame_symbols=64).process(x) == []


def test_repeat_filter():
    f = RepeatFilter(seconds=30)
    assert f.fresh("a", now=0) and not f.fresh("a", now=10) and f.fresh("b", now=10)
    assert f.fresh("a", now=31)
