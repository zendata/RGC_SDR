"""Demodulator tests by round trip: modulate a known tone, demodulate, check it returns.

Synthetic signals throughout -- no radio, no audio device.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import (
    MODE_SPECS,
    MODES,
    AmDetector,
    Deemphasis,
    DemodChain,
    Fir,
    FmDetector,
    Mixer,
    channel_taps,
    fir_length_for,
    sideband_taps,
)

FS = 768e3


def peak_freq(audio, rate, ignore_below_hz=50.0):
    """Frequency of the strongest audio component. Stereo is averaged to mono."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / rate)
    spectrum[freqs < ignore_below_hz] = 0.0
    return float(freqs[int(np.argmax(spectrum))])


def am_signal(n, tone_hz, fs=FS, depth=0.6, offset_hz=0.0):
    t = np.arange(n) / fs
    envelope = 1.0 + depth * np.cos(2 * np.pi * tone_hz * t)
    return (envelope * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def fm_signal(n, tone_hz, deviation_hz, fs=FS, offset_hz=0.0):
    """FM by integrating the modulating tone.

    phase(t) = 2*pi * integral f(t) dt = (deviation/tone) * sin(2*pi*tone*t).
    The 2*pi belongs to the integral, not in front of it -- an extra factor here inflates
    the deviation 6.28x, which overflows the channel filter and clips into odd harmonics.
    """
    t = np.arange(n) / fs
    phase = (deviation_hz / tone_hz) * np.sin(2 * np.pi * tone_hz * t)
    return (np.exp(1j * (phase + 2 * np.pi * offset_hz * t))).astype(np.complex64)


def ssb_signal(n, tone_hz, fs=FS, upper=True):
    """A single audio tone on one sideband: one complex exponential."""
    t = np.arange(n) / fs
    sign = 1.0 if upper else -1.0
    return np.exp(2j * np.pi * sign * tone_hz * t).astype(np.complex64)


def run_blocks(chain, signal, block=8192):
    out = [chain.process(signal[i : i + block]) for i in range(0, signal.size, block)]
    return np.concatenate([b for b in out if b.size])


# -- AM ----------------------------------------------------------------------

def test_am_recovers_the_modulating_tone():
    chain = DemodChain(FS, "am", volume=1.0)
    audio = run_blocks(chain, am_signal(FS // 2, 1000.0))
    assert audio.size > 1000
    assert peak_freq(audio[500:], chain.audio_rate) == pytest.approx(1000.0, abs=30.0)


@pytest.mark.parametrize("tone", [300.0, 1000.0, 3000.0])
def test_am_recovers_tones_across_the_audio_band(tone):
    chain = DemodChain(FS, "am", volume=1.0)
    audio = run_blocks(chain, am_signal(FS // 2, tone))
    assert peak_freq(audio[500:], chain.audio_rate) == pytest.approx(tone, abs=40.0)


def test_am_removes_the_carrier_dc():
    """An unmodulated carrier is a constant envelope: the output must settle near silence.

    AGC off: it would amplify the residual toward its target and mask the DC removal.
    """
    chain = DemodChain(FS, "am", volume=1.0, agc=False)
    carrier = np.ones(int(FS), dtype=np.complex64)
    audio = run_blocks(chain, carrier)
    assert np.abs(audio[-2000:]).mean() < 0.02


def test_am_depth_tracks_amplitude():
    quiet = DemodChain(FS, "am", volume=1.0, agc=False)
    loud = DemodChain(FS, "am", volume=1.0, agc=False)
    a = run_blocks(quiet, am_signal(FS // 2, 1000.0, depth=0.2))
    b = run_blocks(loud, am_signal(FS // 2, 1000.0, depth=0.8))
    assert np.std(b[500:]) > 2.5 * np.std(a[500:])


def test_am_detector_dc_blocker_is_smooth_across_blocks():
    det = AmDetector()
    steady = np.full(4096, 0.5, dtype=np.complex64)
    first = det.process(steady)
    later = det.process(steady)
    assert abs(float(later.mean())) < abs(float(first.mean())) + 1e-9
    for _ in range(200):
        later = det.process(steady)
    assert abs(float(later.mean())) < 1e-3


# -- FM ----------------------------------------------------------------------

def test_nbfm_recovers_the_modulating_tone():
    chain = DemodChain(FS, "nbfm", volume=1.0)
    spec = MODE_SPECS["nbfm"]
    audio = run_blocks(chain, fm_signal(FS // 2, 1000.0, spec.deviation_hz))
    assert peak_freq(audio[500:], chain.audio_rate) == pytest.approx(1000.0, abs=40.0)


def test_wbfm_recovers_the_modulating_tone():
    chain = DemodChain(FS, "wbfm", volume=1.0)
    audio = run_blocks(chain, fm_signal(FS // 2, 1000.0, 50e3))
    assert audio.size > 1000
    assert peak_freq(audio[800:], chain.audio_rate) == pytest.approx(1000.0, abs=60.0)


def test_fm_output_scales_with_deviation():
    spec = MODE_SPECS["nbfm"]
    low = run_blocks(DemodChain(FS, "nbfm", volume=1.0, agc=False),
                     fm_signal(FS // 2, 1000.0, spec.deviation_hz * 0.25))
    high = run_blocks(DemodChain(FS, "nbfm", volume=1.0, agc=False),
                      fm_signal(FS // 2, 1000.0, spec.deviation_hz))
    assert np.std(high[500:]) > 2.5 * np.std(low[500:])


def test_fm_detector_is_seamless_across_blocks():
    """Without carrying the last sample, each block starts with a bogus phase step."""
    det = FmDetector(48e3, 5e3)
    t = np.arange(4096) / 48e3
    x = np.exp(2j * np.pi * 1000.0 * t).astype(np.complex64)
    joined = np.concatenate([det.process(x[:2048]), det.process(x[2048:])])
    # A constant frequency offset must give a constant discriminator output.
    assert np.std(joined[5:]) < 1e-6


def test_fm_squelch_mutes_a_weak_channel():
    chain = DemodChain(FS, "nbfm", volume=1.0, squelch_dbfs=-20.0)
    faint = fm_signal(FS // 4, 1000.0, 2.5e3) * 1e-4
    audio = run_blocks(chain, faint)
    assert np.max(np.abs(audio)) == 0.0
    assert chain.muted_blocks > 0


def test_fm_squelch_passes_a_strong_channel():
    chain = DemodChain(FS, "nbfm", volume=1.0, squelch_dbfs=-20.0)
    audio = run_blocks(chain, fm_signal(FS // 4, 1000.0, 2.5e3))
    assert np.max(np.abs(audio)) > 0.01
    assert chain.muted_blocks == 0


def test_deemphasis_attenuates_treble_more_than_bass():
    rate = 192e3
    deemph = Deemphasis(rate, 75e-6)
    def level(freq):
        d = Deemphasis(rate, 75e-6)
        t = np.arange(8192) / rate
        x = np.sin(2 * np.pi * freq * t)
        return float(np.std(d.process(x)[1000:]))
    assert level(10e3) < level(1e3) * 0.5


# -- SSB ---------------------------------------------------------------------

def test_usb_recovers_an_upper_sideband_tone():
    chain = DemodChain(FS, "usb", volume=1.0)
    audio = run_blocks(chain, ssb_signal(FS // 2, 1200.0, upper=True))
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(1200.0, abs=40.0)


def test_lsb_recovers_a_lower_sideband_tone():
    chain = DemodChain(FS, "lsb", volume=1.0)
    audio = run_blocks(chain, ssb_signal(FS // 2, 1200.0, upper=False))
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(1200.0, abs=40.0)


def test_usb_rejects_the_opposite_sideband():
    """The whole point of SSB: the wrong sideband must be strongly suppressed."""
    wanted = run_blocks(DemodChain(FS, "usb", volume=1.0, agc=False),
                        ssb_signal(FS // 2, 1200.0, upper=True))
    unwanted = run_blocks(DemodChain(FS, "usb", volume=1.0, agc=False),
                          ssb_signal(FS // 2, 1200.0, upper=False))
    ratio = 20 * np.log10(np.std(wanted[2000:]) / (np.std(unwanted[2000:]) + 1e-20))
    assert ratio > 30.0, f"opposite-sideband rejection only {ratio:.1f} dB"


def test_lsb_rejects_the_opposite_sideband():
    wanted = run_blocks(DemodChain(FS, "lsb", volume=1.0, agc=False),
                        ssb_signal(FS // 2, 1200.0, upper=False))
    unwanted = run_blocks(DemodChain(FS, "lsb", volume=1.0, agc=False),
                          ssb_signal(FS // 2, 1200.0, upper=True))
    ratio = 20 * np.log10(np.std(wanted[2000:]) / (np.std(unwanted[2000:]) + 1e-20))
    assert ratio > 30.0


def test_sideband_taps_are_complex_and_asymmetric():
    """Measure at a real audio offset, not over each half-band.

    The passband starts at 0 Hz, so its transition skirt straddles DC; comparing whole
    half-spectra just measures that rolloff rather than opposite-sideband leakage.
    """
    fs = 48e3
    taps = sideband_taps(2.7e3, fs, upper=True)
    assert np.iscomplexobj(taps)

    def response(freq):
        n = np.arange(taps.size)
        return abs(complex(np.sum(taps * np.exp(-2j * np.pi * freq * n / fs))))

    for audio in (800.0, 1200.0):
        assert response(audio) > 20 * response(-audio), f"leak at -{audio:g} Hz"


# -- offset tuning -----------------------------------------------------------

def test_offset_brings_a_signal_away_from_centre_into_the_channel():
    """Listening off-centre is what lets you watch wide and demodulate a nearby signal."""
    offset = 40e3
    chain = DemodChain(FS, "am", volume=1.0, offset_hz=offset)
    audio = run_blocks(chain, am_signal(FS // 2, 1000.0, offset_hz=offset))
    assert peak_freq(audio[500:], chain.audio_rate) == pytest.approx(1000.0, abs=40.0)


def test_signal_outside_the_channel_is_rejected():
    chain = DemodChain(FS, "am", volume=1.0)           # listening at centre
    far = run_blocks(chain, am_signal(FS // 2, 1000.0, offset_hz=120e3))
    centred = run_blocks(DemodChain(FS, "am", volume=1.0), am_signal(FS // 2, 1000.0))
    assert np.std(far[500:]) < 0.05 * np.std(centred[500:])


def test_mixer_phase_is_continuous_across_blocks():
    mixer = Mixer(48e3, 1000.0)
    x = np.ones(2048, dtype=np.complex64)
    joined = np.concatenate([mixer.process(x), mixer.process(x)])
    step = np.angle(joined[1:] * np.conj(joined[:-1]))
    assert np.std(step) < 1e-9, "phase discontinuity at the block seam"


def test_mixer_phase_stays_bounded_over_long_runs():
    mixer = Mixer(48e3, 1234.0)
    for _ in range(2000):
        mixer.process(np.ones(1024, dtype=np.complex64))
    assert abs(mixer._phase) <= 2 * np.pi


def test_zero_offset_is_a_passthrough():
    mixer = Mixer(48e3, 0.0)
    x = np.ones(8, dtype=np.complex64)
    assert mixer.process(x) is x


# -- rates and plumbing ------------------------------------------------------

@pytest.mark.parametrize("rate", [912e3, 768e3, 650e3, 456e3, 384e3, 228e3, 192e3])
@pytest.mark.parametrize("mode", MODES)
def test_every_device_rate_and_mode_gives_a_sane_audio_rate(rate, mode):
    chain = DemodChain(rate, mode)
    assert 20e3 <= chain.audio_rate <= 96e3, (
        f"{mode} at {rate/1e3:g} kS/s gives {chain.audio_rate:.0f} Hz"
    )


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_produces_audio_from_noise_without_error(mode):
    rng = np.random.default_rng(0)
    chain = DemodChain(768e3, mode)
    n = chain.input_for_audio(4096)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.01
    audio = run_blocks(chain, iq)
    assert audio.size > 0
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


@pytest.mark.parametrize("mode", MODES)
def test_reset_clears_state(mode):
    chain = DemodChain(768e3, mode, volume=1.0)
    run_blocks(chain, am_signal(200000, 1000.0))
    chain.reset()
    quiet = chain.process(np.zeros(chain.input_for_audio(8192), dtype=np.complex64))
    assert np.max(np.abs(quiet)) < 1e-6


def test_volume_scales_and_output_never_clips_the_dac():
    loud = DemodChain(FS, "am", volume=50.0)
    audio = run_blocks(loud, am_signal(FS // 4, 1000.0))
    assert np.max(np.abs(audio)) <= 1.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        DemodChain(FS, "ssb-ish")


def test_fir_length_scales_inversely_with_cutoff():
    assert fir_length_for(0.2) < fir_length_for(0.01)
    for cutoff in (0.001, 0.01, 0.1, 0.4):
        n = fir_length_for(cutoff)
        assert 31 <= n <= 511 and n % 2 == 1


def test_fir_is_length_preserving_and_stateful():
    fir = Fir(channel_taps(9e3, 48e3))
    a = fir.process(np.ones(1000, dtype=np.complex64))
    b = fir.process(np.ones(1000, dtype=np.complex64))
    assert a.size == 1000 and b.size == 1000
    # Second block is past the start-up transient, so closer to the steady-state gain.
    assert abs(abs(b[-1]) - 1.0) < abs(abs(a[0]) - 1.0)


def test_input_for_audio_is_enough():
    for mode in MODES:
        chain = DemodChain(768e3, mode)
        n = chain.input_for_audio(4096)
        got = chain.process(np.zeros(n, dtype=np.complex64)).size
        assert got >= 4096 * 0.9, f"{mode}: asked 4096 audio samples, got {got}"


# -- audio AGC ---------------------------------------------------------------

from src.rgc_sdr.dsp.demod import AudioAgc  # noqa: E402


def test_agc_equalises_levels_across_four_orders_of_magnitude():
    """Measured on air: a -103 dBFS carrier gave audio rms 1e-5. AGC makes it audible."""
    levels = []
    for amp in (1.0, 1e-2, 1e-4):
        chain = DemodChain(FS, "am", volume=1.0)
        audio = run_blocks(chain, am_signal(int(FS * 1.5), 1000.0) * amp, block=16384)
        levels.append(float(np.std(audio[-20000:])))
    assert min(levels) > 0.02, f"too quiet: {levels}"
    assert max(levels) / min(levels) < 3.0, f"levels not equalised: {levels}"


def test_agc_is_audible_immediately_on_a_weak_signal():
    """The first block must already be at the right gain, not ramping up for seconds."""
    chain = DemodChain(FS, "am", volume=1.0)
    weak = am_signal(200_000, 1000.0) * 3e-5
    first = chain.process(weak)
    assert float(np.std(first)) > 0.01, "first block is inaudible"


def test_agc_without_it_the_output_tracks_absolute_amplitude():
    loud = DemodChain(FS, "am", volume=1.0, agc=False)
    quiet = DemodChain(FS, "am", volume=1.0, agc=False)
    a = run_blocks(loud, am_signal(200_000, 1000.0))
    b = run_blocks(quiet, am_signal(200_000, 1000.0) * 1e-4)
    assert float(np.std(a)) / float(np.std(b)) > 1000.0


def test_agc_backs_off_faster_than_it_recovers():
    """Asymmetric by design: never blast, but do not pump the noise floor either.

    Compared as the fraction of the remaining distance covered by one block, which is
    what the two rates actually control.
    """
    loud = np.full(1024, 1.0)
    quiet = np.full(1024, 0.001)

    rising = AudioAgc(target_rms=0.1, max_gain=1e6)
    for _ in range(200):
        rising.process(quiet)
    settled_high = rising.gain
    rising.process(loud)
    attack_fraction = (settled_high - rising.gain) / (settled_high - 0.1)

    falling = AudioAgc(target_rms=0.1, max_gain=1e6)
    for _ in range(200):
        falling.process(loud)
    settled_low = falling.gain
    falling.process(quiet)
    decay_fraction = (falling.gain - settled_low) / (100.0 - settled_low)

    assert attack_fraction > 5 * decay_fraction, (
        f"attack {attack_fraction:.3f} not decisively faster than decay {decay_fraction:.3f}"
    )


def test_agc_will_not_amplify_a_dead_channel_without_limit():
    agc = AudioAgc(target_rms=0.15, max_gain=100.0)
    for _ in range(50):
        agc.process(np.full(1024, 1e-9))
    assert agc.gain <= 100.0


def test_agc_handles_exact_silence():
    agc = AudioAgc()
    out = agc.process(np.zeros(1024))
    assert np.all(out == 0.0) and np.isfinite(agc.gain)


def test_agc_reset_reprimes():
    agc = AudioAgc()
    agc.process(np.full(1024, 1.0))
    agc.reset()
    assert agc.gain == 1.0
    agc.process(np.full(1024, 0.001))
    assert agc.gain > 1.0


def test_agc_output_still_never_clips():
    chain = DemodChain(FS, "am", volume=1.0)
    audio = run_blocks(chain, am_signal(200_000, 1000.0) * 1e-5)
    assert np.max(np.abs(audio)) <= 1.0


# -- CW ----------------------------------------------------------------------

def keyed_carrier(n, fs=FS, offset_hz=0.0, on=True):
    """An on/off carrier, which is what CW actually is."""
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * offset_hz * t)
    return (x if on else x * 0.0).astype(np.complex64)


def test_cw_turns_a_tuned_carrier_into_an_audible_tone():
    """Tuned exactly, a keyed carrier is at DC and therefore silent without a BFO."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.5)), block=16384)
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(chain.pitch_hz, abs=25.0)
    assert np.std(audio[2000:]) > 0.1


def test_cw_key_up_is_silence():
    chain = DemodChain(FS, "cw", volume=1.0, agc=False)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.2), on=False), block=16384)
    assert np.max(np.abs(audio)) < 1e-6


def test_cw_pitch_follows_mistuning():
    """Tuning off by 200 Hz should move the tone by 200 Hz -- that is how you zero-beat."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.5), offset_hz=200.0), block=16384)
    expected = chain.pitch_hz + 200.0
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(expected, abs=30.0)


def test_cw_is_narrow_enough_to_reject_a_nearby_signal():
    """A 500 Hz filter must reject something 3 kHz away, which SSB width would pass."""
    on_channel = run_blocks(DemodChain(FS, "cw", volume=1.0, agc=False),
                            keyed_carrier(int(FS * 0.5)), block=16384)
    off_channel = run_blocks(DemodChain(FS, "cw", volume=1.0, agc=False),
                             keyed_carrier(int(FS * 0.5), offset_hz=3000.0), block=16384)
    ratio = 20 * np.log10(np.std(on_channel[2000:]) / (np.std(off_channel[2000:]) + 1e-20))
    assert ratio > 30.0, f"only {ratio:.1f} dB of rejection 3 kHz away"


def test_cw_offset_keeps_meaning_where_you_are_listening():
    """The BFO is internal: the offset the UI sets is still the listening frequency."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False, offset_hz=5000.0)
    assert chain.offset_hz == pytest.approx(5000.0)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.5), offset_hz=5000.0), block=16384)
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(chain.pitch_hz, abs=30.0)


def test_cw_bandwidth_can_be_narrowed():
    chain = DemodChain(FS, "cw", volume=1.0, agc=False, bandwidth_hz=250.0)
    assert chain.bandwidth_hz == pytest.approx(250.0)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.4)), block=16384)
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(chain.pitch_hz, abs=30.0)


def test_only_cw_has_a_pitch():
    for mode in MODES:
        chain = DemodChain(FS, mode)
        if mode == "cw":
            assert chain.pitch_hz > 0.0
        else:
            assert chain.pitch_hz == 0.0


def test_bandpass_taps_are_centred_where_asked():
    from src.rgc_sdr.dsp.demod import bandpass_taps

    fs = 48e3
    taps = bandpass_taps(700.0, 500.0, fs)

    def response(freq):
        n = np.arange(taps.size)
        return abs(complex(np.sum(taps * np.exp(-2j * np.pi * freq * n / fs))))

    assert response(700.0) > 10 * response(0.0)        # rejects DC
    assert response(700.0) > 10 * response(-700.0)     # asymmetric, as it must be
    assert response(700.0) > 10 * response(2000.0)     # and narrow


def test_cw_pitch_defaults_lower_than_the_traditional_700():
    """Lower beat notes are less tiring over a long session."""
    assert DemodChain(FS, "cw").pitch_hz == pytest.approx(500.0)


@pytest.mark.parametrize("pitch", [400.0, 500.0, 600.0, 700.0, 800.0])
def test_cw_tone_lands_on_whatever_pitch_is_asked_for(pitch):
    chain = DemodChain(FS, "cw", volume=1.0, agc=False, pitch_hz=pitch)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.5)), block=16384)
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(pitch, abs=25.0)


def test_changing_pitch_moves_the_tone_and_keeps_it_audible():
    """The filter and the mixer must move together, or the tone falls outside its own
    passband and goes silent."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False, pitch_hz=700.0)
    run_blocks(chain, keyed_carrier(int(FS * 0.2)), block=16384)
    chain.set_pitch(400.0)
    chain.reset()
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.5)), block=16384)
    assert peak_freq(audio[2000:], chain.audio_rate) == pytest.approx(400.0, abs=25.0)
    assert np.std(audio[2000:]) > 0.1, "tone went quiet after moving the pitch"


def test_setting_the_same_pitch_is_a_no_op():
    chain = DemodChain(FS, "cw", pitch_hz=500.0)
    before = chain._channel
    chain.set_pitch(500.0)
    assert chain._channel is before


def test_cw_audio_has_no_significant_harmonics():
    """Answers 'is something additive happening?': measured -100 dB and below."""
    chain = DemodChain(FS, "cw", volume=0.4, pitch_hz=500.0)
    audio = run_blocks(chain, keyed_carrier(int(FS * 0.6)), block=16384)[4000:]
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / chain.audio_rate)
    spectrum = spectrum / spectrum.max()

    def level_at(hz, tol=40.0):
        mask = np.abs(freqs - hz) < tol
        return 20 * np.log10(spectrum[mask].max() + 1e-20) if mask.any() else -200.0

    assert freqs[int(np.argmax(spectrum))] == pytest.approx(500.0, abs=25.0)
    assert level_at(1000.0) < -40.0, "second harmonic present"
    assert level_at(1500.0) < -40.0, "third harmonic present"
