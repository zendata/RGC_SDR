"""Transmit modulators, checked by demodulating their output with the receive chain.

The receiver is already verified on air, so it is the independent check: if the
receiver hears the tone the modulator was given, the modulator is right.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import DemodChain
from src.rgc_sdr.dsp.modulate import (
    AUDIO_RATE, TX_MODES, Interpolator, Modulator, nearest_tx_rate,
)

IQ_RATE = 1.92e6          # 40 x 48 kHz and 8 x 240 kHz: suits every mode


def tone(freq, seconds=0.5, level=0.5, rate=AUDIO_RATE):
    t = np.arange(int(rate * seconds)) / rate
    return level * np.sin(2 * np.pi * freq * t)


def modulate(mode, audio, block=1024, rate=IQ_RATE):
    mod = Modulator(mode, rate)
    return np.concatenate([mod.process(audio[i:i + block]) for i in range(0, audio.size, block)])


def demodulate(mode, iq, block=65536, agc=True):
    chain = DemodChain(IQ_RATE, mode, volume=1.0, agc=agc)
    out = [chain.process(iq[i:i + block]) for i in range(0, iq.size, block)]
    audio = np.concatenate([o for o in out if o.size])
    return (audio[:, 0] if audio.ndim == 2 else audio), chain.audio_rate


def peak_hz(audio, rate):
    audio = audio[len(audio) // 4:]
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    return np.fft.rfftfreq(audio.size, 1 / rate)[int(np.argmax(spectrum))]


@pytest.mark.parametrize("mode", TX_MODES)
def test_the_receiver_hears_what_was_transmitted(mode):
    iq = modulate(mode, tone(1000.0))
    audio, rate = demodulate(mode, iq)
    assert peak_hz(audio, rate) == pytest.approx(1000.0, abs=30.0)


@pytest.mark.parametrize("mode", TX_MODES)
def test_output_never_exceeds_full_scale(mode):
    iq = modulate(mode, tone(1000.0, level=5.0))          # grossly overdriven mic
    assert np.max(np.abs(iq)) <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", TX_MODES)
def test_blocks_join_seamlessly(mode):
    audio = tone(700.0, seconds=0.2)
    whole = Modulator(mode, IQ_RATE).process(audio)
    pieces = modulate(mode, audio, block=333)
    assert np.max(np.abs(whole - pieces)) < 1e-4


def test_usb_is_one_sideband():
    """Tuned as LSB, a USB transmission should be all but silent."""
    iq = modulate("usb", tone(1000.0))
    # AGC off: it would lift the residue in the wrong sideband to full level.
    right, _ = demodulate("usb", iq, agc=False)
    wrong, _ = demodulate("lsb", iq, agc=False)
    n = len(right) // 4
    ratio_db = 20 * np.log10(np.std(right[n:]) / np.std(wrong[n:]))
    assert ratio_db > 25.0, f"only {ratio_db:.1f} dB of opposite-sideband rejection"


def test_nbfm_deviation_is_as_specified():
    """Instantaneous frequency of a full-scale tone peaks at the deviation."""
    mod = Modulator("nbfm", 48e3)                     # no interpolation, easier to read
    iq = mod.process(tone(1000.0, level=1.0, seconds=0.3))
    inst = np.angle(iq[1:] * np.conj(iq[:-1])) * 48e3 / (2 * np.pi)
    peak = np.percentile(np.abs(inst[2000:]), 99.5)
    assert peak == pytest.approx(2.5e3, rel=0.15)


def test_cw_is_refused():
    with pytest.raises(ValueError, match="CW"):
        Modulator("cw", IQ_RATE)


def test_an_unusable_rate_is_refused_with_the_nearest_good_one():
    with pytest.raises(ValueError, match="1.92"):
        Modulator("wbfm", 2e6)
    assert nearest_tx_rate("wbfm", 2e6) == 1.92e6
    assert nearest_tx_rate("usb", 2e6) == pytest.approx(2.016e6)


def test_interpolator_passes_a_tone_at_unity_gain():
    x = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48e3)
    y = Interpolator(5).process(x)
    assert y.size == 5 * x.size
    assert np.std(y[5000:]) == pytest.approx(np.std(x[1000:]), rel=0.02)
