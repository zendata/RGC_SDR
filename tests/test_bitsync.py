"""Bit recovery from transition timing."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.bitsync import BitSlicer


def nrz(bits, sps, clock=1.0, rng=None, noise=0.0, dc=0.0):
    """Bits as a +/-1 waveform at `sps` samples per bit, the sender's clock `clock`
    times nominal, band-limited a little so the edges are not instant."""
    n = int(len(bits) * sps / clock)
    idx = np.minimum((np.arange(n) * clock / sps).astype(int), len(bits) - 1)
    x = np.where(np.asarray(bits)[idx] > 0, 1.0, -1.0)
    x = np.convolve(x, np.ones(5) / 5, mode="same")
    if rng is not None and noise:
        x = x + noise * rng.standard_normal(n)
    return x + dc


def recovered_matches(sent, got, skip=16):
    """The received bits contain the sent ones (after the first few, while timing
    settles) without a single error."""
    sent = np.asarray(sent, dtype=np.uint8)
    probe = sent[skip:skip + 64]
    for offset in range(0, len(got) - len(probe)):
        if np.array_equal(got[offset:offset + len(probe)], probe):
            tail = sent[skip:]
            return np.array_equal(got[offset:offset + len(tail)], tail[:len(got) - offset])
    return False


@pytest.mark.parametrize("rate, baud", [(48_000, 1200), (57_000, 2400), (62_500, 512)])
def test_recovers_random_bits(rate, baud):
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 2000)
    slicer = BitSlicer(rate, baud)
    x = nrz(bits, rate / baud, rng=rng, noise=0.3, dc=0.2)
    # Fed in uneven blocks, as the audio thread delivers them.
    cuts = np.cumsum(rng.integers(300, 3000, 200))
    got = np.concatenate([slicer.process(b) for b in np.split(x, cuts[cuts < x.size])])
    assert recovered_matches(bits, got)


def test_follows_a_clock_that_is_off_frequency():
    """0.3% fast: three bits' slip over 1000 bits, which a fixed clock would not survive."""
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 3000)
    slicer = BitSlicer(48_000, 1200)
    got = slicer.process(nrz(bits, 40.0, clock=1.003))
    assert recovered_matches(bits, got)


def test_a_long_steady_level_is_not_held_back():
    slicer = BitSlicer(48_000, 1200)
    bits = np.r_[np.tile([1, 0], 50), np.ones(300, dtype=int)]
    got = slicer.process(nrz(bits, 40.0))
    assert got[-200:].all() and got.size > 300


def test_needs_two_samples_per_bit():
    with pytest.raises(ValueError):
        BitSlicer(2000, 1200)
