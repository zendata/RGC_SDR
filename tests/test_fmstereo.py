"""FM stereo and RDS through the full WBFM chain, from synthetic broadcasts.

The multiplex here is built to the broadcast standard -- sin pilot, sin(2wt) subcarrier,
RDS on the third harmonic -- then FM-modulated, so these exercise the real demodulator
path end to end.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.fmstereo import PILOT_HZ, StereoDecoder

MPX_RATE = 192_000.0


def multiplex(n, left_hz=None, right_hz=None, pilot=True, fs=MPX_RATE, level=0.4):
    t = np.arange(n) / fs
    left = level * np.sin(2 * np.pi * left_hz * t) if left_hz else np.zeros(n)
    right = level * np.sin(2 * np.pi * right_hz * t) if right_hz else np.zeros(n)
    w = 2 * np.pi * PILOT_HZ * t
    mpx = 0.45 * (left + right) + (0.45 * (left - right) * np.sin(2 * w) if pilot else 0)
    if pilot:
        mpx = mpx + 0.09 * np.sin(w)
    return mpx


def decode(mpx, block=8192, **kw):
    dec = StereoDecoder(MPX_RATE, **kw)
    lefts, rights, flags = [], [], []
    for i in range(0, mpx.size, block):
        out = dec.process(mpx[i:i + block])
        lefts.append(out.left); rights.append(out.right); flags.append(out.stereo)
    return np.concatenate(lefts), np.concatenate(rights), flags, dec


def level_at(x, hz, fs=MPX_RATE):
    x = x[x.size // 4:]
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    return float(spectrum[np.argmin(np.abs(freqs - hz))])


def test_a_left_only_tone_stays_on_the_left():
    left, right, flags, _ = decode(multiplex(int(MPX_RATE * 0.5), left_hz=1000))
    assert flags[-1]
    separation = 20 * np.log10(level_at(left, 1000) / level_at(right, 1000))
    assert separation > 25.0, f"only {separation:.1f} dB of separation"


def test_a_right_only_tone_stays_on_the_right():
    left, right, _, _ = decode(multiplex(int(MPX_RATE * 0.5), right_hz=2500))
    separation = 20 * np.log10(level_at(right, 2500) / level_at(left, 2500))
    assert separation > 25.0


def test_different_programmes_either_side():
    left, right, _, _ = decode(multiplex(int(MPX_RATE * 0.5), left_hz=800, right_hz=3000))
    assert level_at(left, 800) > 10 * level_at(left, 3000)
    assert level_at(right, 3000) > 10 * level_at(right, 800)


def test_without_a_pilot_it_falls_back_to_mono():
    left, right, flags, dec = decode(multiplex(int(MPX_RATE * 0.5), left_hz=1000,
                                               right_hz=1000, pilot=False))
    assert not flags[-1]
    assert np.allclose(left, right)


def test_mono_can_be_forced_for_a_noisy_station():
    """Stereo is noisier on a weak signal; the listener can choose mono."""
    dec = StereoDecoder(MPX_RATE)
    dec.force_mono = True
    out = None
    mpx = multiplex(int(MPX_RATE * 0.3), left_hz=1000)
    for i in range(0, mpx.size, 8192):
        out = dec.process(mpx[i:i + 8192])
    assert not out.stereo
    assert np.allclose(out.left, out.right)


def test_the_pilot_itself_is_removed_from_the_audio():
    left, _, _, _ = decode(multiplex(int(MPX_RATE * 0.5), left_hz=1000))
    assert level_at(left, PILOT_HZ) < 0.01 * level_at(left, 1000)


def test_decoder_is_seamless_across_blocks():
    """Odd block sizes must not change the filtered result.

    Compared once stereo has engaged in both runs: the stereo decision is a smoothed
    per-block measurement, so *when* it switches on depends on block size, but the
    filtering it gates must be identical.
    """
    mpx = multiplex(int(MPX_RATE * 0.5), left_hz=1000)
    a, _, fa, _ = decode(mpx, block=8192)
    b, _, fb, _ = decode(mpx, block=3001)
    assert fa[-1] and fb[-1]
    tail = a.size // 2
    assert np.allclose(a[tail:], b[tail:], atol=1e-9)


def test_too_low_a_rate_is_refused():
    with pytest.raises(ValueError):
        StereoDecoder(96_000.0)


# -- RDS over the air --------------------------------------------------------

from src.rgc_sdr.dsp.rds import BIT_RATE, RdsDemodulator, encode_block  # noqa: E402
from tests.test_rds import ps_groups, rt_groups  # noqa: E402


def rds_waveform(groups, n, fs=MPX_RATE, start_bit=0.37):
    """Biphase, differentially encoded RDS bits as a +/-1 waveform at the MPX rate."""
    bits = []
    for a, b, c, d in groups:
        for info, offset in ((a, "A"), (b, "B"), (c, "C"), (d, "D")):
            block = encode_block(info, offset)
            bits.extend((block >> (25 - i)) & 1 for i in range(26))
    encoded, previous = [], 0
    for bit in bits:
        previous ^= bit
        encoded.append(previous)
    t_bits = (np.arange(n) / fs) * BIT_RATE + start_bit
    index = np.minimum(t_bits.astype(int), len(encoded) - 1)
    first_half = (t_bits % 1.0) < 0.5
    symbol = np.where(np.array(encoded)[index] == 1, 1.0, -1.0)
    return np.where(first_half, symbol, -symbol)


def with_rds(mpx, groups, fs=MPX_RATE, level=0.04):
    t = np.arange(mpx.size) / fs
    return mpx + level * rds_waveform(groups, mpx.size, fs) * np.sin(3 * 2 * np.pi * PILOT_HZ * t)


def rds_through_stereo(mpx, block=8192):
    stereo = StereoDecoder(MPX_RATE)
    rds = RdsDemodulator(MPX_RATE)
    for i in range(0, mpx.size, block):
        out = stereo.process(mpx[i:i + block])
        rds.process(out.mpx, out.pilot)
    return rds


def test_station_name_is_decoded_from_the_multiplex():
    groups = ps_groups(0xC202, "RGC SDR") * 12
    seconds = len(groups) * 104 / BIT_RATE
    mpx = with_rds(multiplex(int(MPX_RATE * seconds), left_hz=1000), groups)
    rds = rds_through_stereo(mpx)
    assert rds.info.ps_name == "RGC SDR", f"got {rds.info.ps_name!r}"
    assert rds.info.pi == 0xC202


def test_radio_text_is_decoded_from_the_multiplex():
    groups = (ps_groups(0xC202, "RGC SDR") + rt_groups(0xC202, "Hello from RGC")) * 4
    seconds = len(groups) * 104 / BIT_RATE
    mpx = with_rds(multiplex(int(MPX_RATE * seconds), left_hz=400, right_hz=900), groups)
    rds = rds_through_stereo(mpx)
    assert rds.info.radio_text == "Hello from RGC"


def test_rds_survives_noise():
    rng = np.random.default_rng(2)
    groups = ps_groups(0x1234, "NOISY FM") * 14
    n = int(MPX_RATE * len(groups) * 104 / BIT_RATE)
    mpx = with_rds(multiplex(n, left_hz=1000), groups) + 0.01 * rng.standard_normal(n)
    assert rds_through_stereo(mpx).info.ps_name == "NOISY FM"


def test_no_station_name_is_invented_without_rds():
    mpx = multiplex(int(MPX_RATE * 2.0), left_hz=1000)
    rds = rds_through_stereo(mpx)
    assert rds.info.ps_name == ""


# -- the whole receive chain -------------------------------------------------

from src.rgc_sdr.dsp.demod import DEEMPHASIS_S, DemodChain  # noqa: E402

IQ_RATE = 768e3


def broadcast(mpx_at_iq_rate, deviation=75e3, fs=IQ_RATE):
    """FM-modulate a multiplex, as the transmitter does."""
    phase = 2 * np.pi * deviation * np.cumsum(mpx_at_iq_rate) / fs
    return np.exp(1j * phase).astype(np.complex64)


def receive(iq, block=65536, **kw):
    chain = DemodChain(IQ_RATE, "wbfm", volume=1.0, agc=False, **kw)
    out = [chain.process(iq[i:i + block]) for i in range(0, iq.size, block)]
    return np.concatenate([o for o in out if o.size]), chain


def test_chain_delivers_stereo_from_a_real_broadcast():
    n = int(IQ_RATE * 0.8)
    mpx = multiplex(n, left_hz=1000, fs=IQ_RATE)
    audio, chain = receive(broadcast(mpx))
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert chain.stereo
    left, right = audio[:, 0], audio[:, 1]
    separation = 20 * np.log10(level_at(left, 1000, chain.audio_rate)
                               / level_at(right, 1000, chain.audio_rate))
    assert separation > 20.0, f"only {separation:.1f} dB through the full chain"


def test_chain_reads_rds_from_a_real_broadcast():
    groups = (ps_groups(0xC202, "RGC SDR") + rt_groups(0xC202, "Stereo and RDS")) * 5
    seconds = len(groups) * 104 / BIT_RATE
    n = int(IQ_RATE * seconds)
    mpx = with_rds(multiplex(n, left_hz=700, right_hz=1500, fs=IQ_RATE), groups, fs=IQ_RATE)
    _, chain = receive(broadcast(mpx))
    assert chain.rds.ps_name == "RGC SDR", f"got {chain.rds.ps_name!r}"
    assert chain.rds.radio_text == "Stereo and RDS"


def test_mono_station_plays_the_same_on_both_sides():
    n = int(IQ_RATE * 0.5)
    audio, chain = receive(broadcast(multiplex(n, left_hz=1000, right_hz=1000,
                                               pilot=False, fs=IQ_RATE)))
    assert not chain.stereo
    assert np.allclose(audio[:, 0], audio[:, 1])


def test_de_emphasis_is_the_european_standard():
    """75 us is the Americas; 50 us is right for the UK and Europe."""
    assert DEEMPHASIS_S == pytest.approx(50e-6)


def test_the_channel_is_wide_enough_for_stereo_and_rds():
    chain = DemodChain(IQ_RATE, "wbfm")
    assert chain.bandwidth_hz >= 180e3
    assert chain.mpx_rate >= 150e3            # RDS sits at 57 kHz


@pytest.mark.parametrize("rate", [912e3, 768e3, 650e3, 456e3, 384e3, 228e3, 192e3,
                                  2.048e6, 4e6])
def test_every_supported_rate_can_carry_the_multiplex(rate):
    chain = DemodChain(rate, "wbfm")
    assert chain.mpx_rate >= 150e3
    assert 20e3 <= chain.audio_rate <= 96e3


def test_noise_is_not_mistaken_for_stereo():
    """The failure found on air: an amplitude test called pure noise stereo.

    An FM discriminator turns noise into a multiplex with plenty of energy near 19 kHz,
    so this drives the real chain with noise rather than a clean mono signal -- the
    earlier tests only checked the clean case, which is how the fault got through.
    """
    rng = np.random.default_rng(1)
    n = int(IQ_RATE * 1.0)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    _, chain = receive(noise)
    assert not chain.stereo
    assert chain._stereo.pilot_coherence < 0.3


def test_a_weak_but_real_pilot_is_still_found():
    """Coherence must not demand a strong signal, only a steady one."""
    rng = np.random.default_rng(4)
    n = int(IQ_RATE * 1.0)
    iq = broadcast(multiplex(n, left_hz=1000, fs=IQ_RATE))
    iq = iq + 0.25 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    _, chain = receive(iq.astype(np.complex64))
    assert chain.stereo, f"coherence {chain._stereo.pilot_coherence:.2f}"
