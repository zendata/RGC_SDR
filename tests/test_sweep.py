"""The Classify button: every strong signal on screen, merged by system (sweep.py)."""

import numpy as np
import pytest

from src.rgc_sdr.classify import Classification
from src.rgc_sdr.sweep import (
    COLOURS, Channeliser, find_candidates, merge, on_grid, sweep,
)
from tests.test_fsk4 import fsk4_wave
from tests.test_p25 import CONTROL

RATE = 1e6
CENTRE = 420.5e6


def at(baseband, offset, rate=RATE, n=None):
    x = np.asarray(baseband, dtype=np.complex128)
    if n is not None:
        x = np.resize(x, n)
    return x * np.exp(2j * np.pi * offset * np.arange(x.size) / rate)


def fm(dev, rate=RATE):
    return np.exp(2j * np.pi * np.cumsum(dev) / rate)


def scene(seconds=1.6, seed=5):
    """Two channels of one P25 network, an NBFM voice, a strong noise-like signal no
    classifier can name, a weak carrier, and the radio's spike -- in noise."""
    n = int(seconds * RATE)
    p25 = fm(fsk4_wave(CONTROL * 3, rate=RATE, unit=600.0, dc=0.0, noise=0.0))
    t = np.arange(n) / RATE
    voice = fm(2500 * np.sin(2 * np.pi * 800 * t))
    rng = np.random.default_rng(seed)
    hiss = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    mush = np.convolve(hiss, np.ones(200) / 200, mode="same") * 3   # ~5 kHz of noise
    x = (at(p25, 150e3, n=n) + 0.7 * at(p25, 250e3, n=n) + 0.8 * at(voice, -200e3)
         + at(mush, -100e3) + 0.003 * at(np.ones(n), 50e3) + 0.5 * at(np.ones(n), 200e3))
    return (x + 0.002 * hiss).astype(np.complex64)


def test_candidates_are_strong_inside_the_view_and_clear_of_the_spike():
    iq = scene()
    found = find_candidates(iq, RATE, CENTRE, CENTRE - RATE / 2, CENTRE + RATE / 2,
                            spike_offset_hz=200e3)
    offsets = [round((c.freq_hz - CENTRE) / 1e3) for c in found]
    assert set(offsets) >= {150, 250, -200, -100}
    assert 200 not in offsets                   # the spike
    assert 50 not in offsets                    # too weak: under 20 dB
    only_right = find_candidates(iq, RATE, CENTRE, CENTRE + 100e3, CENTRE + 400e3, 200e3)
    assert {round((c.freq_hz - CENTRE) / 1e3) for c in only_right} == {150, 250}


def test_the_channeliser_moves_a_signal_to_zero():
    n = 2 ** 18
    iq = at(np.ones(n), 300e3 + 1000.0).astype(np.complex64)
    z, fs = Channeliser(iq, RATE, CENTRE).channel(CENTRE + 300e3, 250e3)
    assert fs == pytest.approx(250e3, rel=0.01)
    spectrum = np.abs(np.fft.fft(z))
    assert np.fft.fftfreq(z.size, 1 / fs)[np.argmax(spectrum)] == pytest.approx(1000.0, abs=5)


def test_a_systems_channels_share_one_coloured_label():
    a = Classification(420.0125e6, "P25", "NAC 161, control channel", certain=True,
                       system="P25 NAC 161")
    b = Classification(420.0625e6, "P25", "NAC 161", certain=True, system="P25 NAC 161")
    c = Classification(466.04e6, "NBFM", "about +/-2.5 kHz deviation")
    labels = merge([a, c, b])
    assert [l.freq_hz for l in labels] == [420.0125e6, 466.04e6]
    assert labels[0].others == [420.0625e6] and labels[0].text.endswith("+1 channel")
    assert labels[0].colour != COLOURS[0] and labels[1].colour == COLOURS[0]


def test_labels_snap_to_the_channel_grid():
    assert on_grid(97.4948e6, "WBFM") == 97.5e6
    assert on_grid(420.0136e6, "P25") == 420.0125e6
    assert on_grid(131.5512e6, "AM") == 131.55e6                    # airband: 25 kHz
    assert on_grid(118.0089e6, "AM") == pytest.approx(118.008333e6, abs=1)  # or 8.33
    assert on_grid(150.12346e6, "NBFM") == 150.1235e6     # 1.5 kHz off the grid: 100 Hz


def test_the_sweep_names_merges_and_skips():
    results = sweep(scene(), RATE, CENTRE, CENTRE - RATE / 2, CENTRE + RATE / 2,
                    spike_offset_hz=200e3)
    labels = merge(results)
    texts = [l.text for l in labels]
    assert any(t.startswith("420.6500 MHz  P25 (NAC 2A7") and t.endswith("+1 channel")
               for t in texts)
    assert any("NBFM" in t for t in texts)
    assert not any(word in t for t in texts for word in ("Unknown", "LSB", "USB"))
    assert len(labels) == 2                     # the noise-like signal is not named
    p25 = next(l for l in labels if "P25" in l.text)
    assert p25.others == [pytest.approx(CENTRE + 250e3, abs=2e3)]
