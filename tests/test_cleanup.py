"""P10's receiver refinements (PLANNING.md 7r): noise blanker, noise reduction, notches,
the passband and IF shift, synchronous AM, AGC modes and the NBFM noise squelch."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.cleanup import AudioCleanup, NoiseBlanker, blanker_threshold
from src.rgc_sdr.dsp.demod import (
    DemodChain, ReceiverOptions, SyncAmDetector, passband_for, width_and_shift,
)

RATE = 48_000.0


def _tone(f, n, rate=RATE, amp=1.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / rate)


def _power_at(x, f, rate=RATE):
    """Power of `x` at frequency f (dB), from a windowed FFT."""
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / rate)
    k = int(np.argmin(np.abs(freqs - f)))
    return 10 * np.log10(spec[max(0, k - 2):k + 3].sum() + 1e-30)


def _run(cleanup, x, block=1024):
    return np.concatenate([cleanup.process(x[i:i + block]) for i in range(0, x.size, block)])


# -- noise blanker -------------------------------------------------------------------


def test_the_blanker_removes_impulses_and_leaves_the_signal():
    rate = 2e6
    n = 200_000
    rng = np.random.default_rng(0)
    signal = 0.01 * np.exp(2j * np.pi * 50e3 * np.arange(n) / rate)
    noise = 0.002 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    impulses = np.zeros(n, complex)
    impulses[rng.integers(0, n, 20)] = 0.5                    # 30x the signal
    blanker = NoiseBlanker(rate, level=5)
    out = blanker.process(signal + noise + impulses)
    assert np.abs(out).max() < 0.05                            # every impulse gone
    assert blanker.blanked < 0.01                              # and little else
    kept = np.abs(out) > 0
    assert np.allclose(out[kept], (signal + noise + impulses)[kept])


def test_the_blanker_is_off_at_level_zero_and_its_threshold_runs_down():
    x = np.ones(1000, complex)
    x[500] = 100
    assert NoiseBlanker(1e6, level=0).process(x) is x
    assert blanker_threshold(1) == pytest.approx(15.5)
    assert blanker_threshold(10) == pytest.approx(2.0)
    assert blanker_threshold(1) > blanker_threshold(5) > blanker_threshold(10)


# -- audio cleanup -------------------------------------------------------------------


def test_cleanup_off_is_a_pure_delay():
    c = AudioCleanup(RATE)
    x = np.random.default_rng(1).standard_normal(48_000)
    y = _run(c, x)
    lag = c.n - c.hop
    assert y.size == x.size
    assert np.allclose(y[lag:], x[:-lag])


def test_frames_reassemble_the_audio_exactly_when_nothing_is_cut():
    c = AudioCleanup(RATE, notches_hz=(23_500.0,))           # active, but above the audio
    x = _tone(1000, 48_000)
    y = _run(c, x)
    lag = c.n - c.hop
    assert y.size == x.size - (x.size % c.hop) or y.size <= x.size
    m = min(y.size, x.size) - lag
    assert np.allclose(y[lag + c.n:lag + m], x[c.n:m], atol=1e-4)   # bar the cut leakage


def test_noise_reduction_raises_the_signal_to_noise_ratio():
    rng = np.random.default_rng(2)
    n = 96_000
    x = _tone(800, n, amp=0.1) + 0.05 * rng.standard_normal(n)
    plain = _run(AudioCleanup(RATE), x)[-48_000:]
    cleaned = _run(AudioCleanup(RATE, nr_level=6), x)[-48_000:]

    def snr(y):
        spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        k = int(round(800 * y.size / RATE))
        tone = spectrum[k - 3:k + 4].sum()
        return 10 * np.log10(tone / (spectrum.sum() - tone))

    # Measured: +4 dB at level 1, +9 at level 6, +12 at level 10.
    assert snr(cleaned) - snr(plain) > 6.0


def test_the_auto_notch_removes_a_steady_tone_but_not_the_rest():
    rng = np.random.default_rng(3)
    n = 3 * 48_000
    speechlike = 0.05 * rng.standard_normal(n)
    whistle = _tone(1500, n, amp=0.2)
    c = AudioCleanup(RATE, auto_notch=True)
    y = _run(c, speechlike + whistle)[-48_000:]
    before = _power_at((speechlike + whistle)[-48_000:], 1500)
    assert before - _power_at(y, 1500) > 20.0
    assert any(abs(f - 1500) < 100 for f in c.notched_hz)
    assert abs(_power_at(y, 3000) - _power_at(speechlike[-48_000:], 3000)) < 3.0


def test_a_manual_notch_removes_its_frequency_only():
    x = _tone(1000, 48_000, amp=0.3) + _tone(2000, 48_000, amp=0.3)
    y = _run(AudioCleanup(RATE, notches_hz=(1000.0,)), x)[-24_000:]
    assert _power_at(y, 2000) - _power_at(y, 1000) > 30.0


# -- passband and IF shift -----------------------------------------------------------


@pytest.mark.parametrize("mode,width,shift,side", [
    ("usb", 2700, 0, "both"), ("usb", 2400, 300, "both"), ("lsb", 2400, -200, "both"),
    ("am", 9000, 1500, "both"), ("cw", 500, 100, "both"), ("sam", 9000, 0, "upper"),
    ("sam", 6000, 500, "lower"), ("nbfm", 12500, 0, "both")])
def test_width_and_shift_invert_the_passband(mode, width, shift, side):
    lo, hi = passband_for(mode, width, shift, side)
    assert width_and_shift(mode, lo, hi, side) == pytest.approx((width, shift))


def test_fm_ignores_the_shift_and_usb_sits_above_the_dial():
    assert passband_for("nbfm", 12500, 3000) == (-6250, 6250)
    assert passband_for("usb", 2700) == (0, 2700)
    assert passband_for("lsb", 2700, 100) == (-2600, 100)


def _ssb_chain(shift):
    return DemodChain(RATE, "usb", agc=False, volume=1.0,
                      options=ReceiverOptions(if_shift_hz=shift))


def _tone_iq(offset, n=48_000, amp=0.1):
    return amp * np.exp(2j * np.pi * offset * np.arange(n) / RATE)


def test_the_if_shift_moves_what_usb_hears():
    low, high = _tone_iq(200), _tone_iq(3000)
    plain, moved = _ssb_chain(0), _ssb_chain(600)
    assert _power_at(plain.process(low)[-24000:], 200) > _power_at(moved.process(low)[-24000:], 200) + 20
    assert _power_at(moved.process(high)[-24000:], 3000) > _power_at(_ssb_chain(0).process(high)[-24000:], 3000) + 20


# -- synchronous AM ------------------------------------------------------------------


def _am(carrier_offset, n=96_000, depth=0.7, audio=1000.0, rate=RATE):
    t = np.arange(n) / rate
    return (1 + depth * np.sin(2 * np.pi * audio * t)) * np.exp(2j * np.pi * carrier_offset * t) * 0.01


def test_sam_finds_an_offset_carrier_and_demodulates_cleanly():
    chain = DemodChain(RATE, "sam", volume=1.0)
    out = np.concatenate([chain.process(b) for b in np.array_split(_am(250.0), 48)])
    assert chain.sam_locked
    assert chain._detector.freq_hz == pytest.approx(250.0, abs=2.0)
    tail = out[-24_000:]
    # The tone, and little distortion: its second harmonic far below it.
    assert _power_at(tail, 1000) - _power_at(tail, 2000) > 30.0


def test_sam_with_one_sideband_rejects_interference_on_the_other():
    n = 96_000
    wanted = _am(0.0, n)
    interferer = 0.01 * np.exp(2j * np.pi * 3000 * np.arange(n) / RATE)   # upper side
    both = DemodChain(RATE, "sam", volume=1.0)
    lower = DemodChain(RATE, "sam", volume=1.0, options=ReceiverOptions(sam_sideband="lower"))
    x = wanted + interferer
    a = np.concatenate([both.process(b) for b in np.array_split(x, 48)])[-24_000:]
    b = np.concatenate([lower.process(b) for b in np.array_split(x, 48)])[-24_000:]
    assert _power_at(a, 3000) - _power_at(b, 3000) > 20.0
    assert abs(_power_at(b, 1000) - _power_at(a, 1000)) < 8.0


def test_sam_detector_reacquires_when_the_carrier_is_lost():
    det = SyncAmDetector(RATE, 9000)
    rng = np.random.default_rng(4)
    det.process(_am(100.0, 4800))
    assert det.freq_hz is not None
    for _ in range(det.LOST_BLOCKS):
        det.process(0.01 * (rng.standard_normal(2400) + 1j * rng.standard_normal(2400)))
    assert det.freq_hz is None or not det.locked


# -- AGC modes -------------------------------------------------------------------------


def test_agc_off_applies_the_fixed_gain_and_fm_keeps_its_level():
    x = _tone_iq(1000, amp=1e-3)
    off = DemodChain(RATE, "usb", volume=1.0, options=ReceiverOptions(agc_mode="off",
                                                                      manual_gain_db=40))
    assert off.agc_gain == pytest.approx(100.0)
    fm = DemodChain(RATE, "nbfm", volume=1.0, options=ReceiverOptions(agc_mode="off"))
    assert fm.agc_gain == 1.0
    raw = DemodChain(RATE, "usb", volume=1.0, agc=False).process(x)
    assert np.allclose(off.process(x), np.clip(raw * 100.0, -1, 1), atol=1e-6)


def test_fast_agc_recovers_sooner_than_slow():
    loud, quiet = _tone_iq(1000, 9600, 0.1), _tone_iq(1000, 48_000, 0.001)
    gains = {}
    for mode in ("fast", "slow"):
        chain = DemodChain(RATE, "usb", volume=1.0, options=ReceiverOptions(agc_mode=mode))
        chain.process(loud)
        for block in np.array_split(quiet, 40):
            chain.process(block)
        gains[mode] = chain.agc_gain
    assert gains["fast"] > 5 * gains["slow"]


# -- NBFM noise squelch ------------------------------------------------------------------


def _fm(n, snr_db, rng, rate=RATE):
    phase = 2 * np.pi * 2500 * np.cumsum(np.sin(2 * np.pi * 1000 * np.arange(n) / rate)) / rate
    signal = np.exp(1j * phase)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return 0.01 * (signal + noise * 10 ** (-snr_db / 20) * np.sqrt(rate / 12.5e3))


def test_the_noise_squelch_closes_on_noise_and_opens_on_a_signal():
    rng = np.random.default_rng(5)
    opts = ReceiverOptions(noise_squelch=True, quieting_db=10.0)
    dead = DemodChain(RATE, "nbfm", volume=1.0, options=opts)
    noise = 0.01 * (rng.standard_normal(48_000) + 1j * rng.standard_normal(48_000))
    out = np.concatenate([dead.process(b) for b in np.array_split(noise, 20)])
    assert not np.any(out[-4800:]) and abs(dead.quieting_db) < 3.0
    live = DemodChain(RATE, "nbfm", volume=1.0, options=opts)
    out = np.concatenate([live.process(b) for b in np.array_split(_fm(48_000, 25, rng), 20)])
    assert np.any(out[-4800:]) and live.quieting_db > 15.0


def test_options_change_in_place():
    chain = DemodChain(RATE, "usb", volume=1.0)
    chain.set_options(ReceiverOptions(if_shift_hz=500, nb_level=3, nr_level=4,
                                      notch_offsets_hz=(1200.0, -300.0)))
    assert chain.passband() == (500, 3200)
    assert chain._blanker.level == 3 and chain._cleanup.nr_level == 4
    assert chain._cleanup.notches_hz == (1200.0,)       # LSB-side offset is not heard on USB
    lsb = DemodChain(RATE, "lsb", options=ReceiverOptions(notch_offsets_hz=(-800.0,)))
    assert lsb._cleanup.notches_hz == (800.0,)
    cw = DemodChain(RATE, "cw", pitch_hz=600, options=ReceiverOptions(notch_offsets_hz=(100.0,)))
    assert cw._cleanup.notches_hz == (700.0,)


def test_noise_reduction_keeps_a_steady_tone():
    """A CW note is as steady as noise; it must not be learned as noise and removed."""
    rng = np.random.default_rng(6)
    n = 5 * 48_000
    x = _tone(700, n, amp=0.1) + 0.02 * rng.standard_normal(n)
    y = _run(AudioCleanup(RATE, nr_level=10), x)[-48_000:]
    assert _power_at(x[-48_000:], 700) - _power_at(y, 700) < 3.0
