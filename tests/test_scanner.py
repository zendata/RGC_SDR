"""Scanner state machine, driven by synthetic spectra with a controlled clock."""

import numpy as np
import pytest

from src.rgc_sdr.scanner import ScanAction, ScanConfig, Scanner, ScanState

SPAN = 768e3
BINS = 4096


def spectrum(centre_hz, carriers=(), floor=-118.0, seed=5):
    rng = np.random.default_rng(seed)
    freqs = centre_hz + (np.arange(BINS) - BINS // 2) * (SPAN / BINS)
    dbfs = floor + rng.normal(0.0, 0.7, BINS)
    for freq, excess in carriers:
        idx = int(np.argmin(np.abs(freqs - freq)))
        for offset, share in ((0, 1.0), (-1, 0.6), (1, 0.6)):
            j = idx + offset
            if 0 <= j < BINS:
                dbfs[j] = max(dbfs[j], floor + excess * share)
    return freqs, dbfs


def airband(**kw):
    kw.setdefault("start_hz", 118e6)
    kw.setdefault("end_hz", 137e6)
    kw.setdefault("step_hz", 25e3)
    return ScanConfig(**kw)


def settle(scanner, centre, t):
    """Advance past the settle delay, leaving the scanner ready to search.

    The frame that arrives on the settle boundary is deliberately skipped rather than
    analysed -- the samples behind it may span the retune -- so one frame is consumed
    making the transition and the *next* one is the first to be scanned.
    """
    freqs, dbfs = spectrum(centre)
    t += scanner.config.settle_s + 0.01
    scanner.on_frame(centre, freqs, dbfs, now=t)
    assert scanner.state is ScanState.SEARCHING
    return t


# -- configuration -----------------------------------------------------------

def test_config_rejects_a_backwards_range():
    with pytest.raises(ValueError):
        Scanner(ScanConfig(start_hz=137e6, end_hz=118e6))


def test_config_rejects_a_zero_step():
    with pytest.raises(ValueError):
        Scanner(airband(step_hz=0))


# -- sweeping ----------------------------------------------------------------

def test_start_tunes_to_the_first_window():
    scanner = Scanner(airband())
    step = scanner.start(SPAN, now=0.0)
    assert step.action is ScanAction.TUNE
    assert 118e6 < step.centre_hz < 118e6 + SPAN
    assert scanner.state is ScanState.SETTLING
    assert scanner.window_count > 20


def test_nothing_is_analysed_until_the_front_end_settles():
    """Frames arriving during the settle window must be ignored, not scanned."""
    scanner = Scanner(airband(settle_s=0.5))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    freqs, dbfs = spectrum(centre, carriers=[(centre + 100e3, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=0.2)
    assert step.action is ScanAction.NONE
    assert scanner.hits() == []


def test_the_frame_on_the_settle_boundary_is_skipped_not_scanned():
    """Its samples may straddle the retune, so it is discarded deliberately."""
    scanner = Scanner(airband(settle_s=0.5, min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    freqs, dbfs = spectrum(centre, carriers=[(centre + 100e3, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=0.51)
    assert step.action is ScanAction.NONE
    assert scanner.state is ScanState.SEARCHING
    assert scanner.hits() == [], "boundary frame was scanned"
    # The next frame is the one that counts.
    step = scanner.on_frame(centre, freqs, dbfs, now=0.55)
    assert step.action is ScanAction.DWELL
    assert len(scanner.hits()) == 1


def test_a_quiet_window_advances_to_the_next():
    scanner = Scanner(airband())
    first = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, first, 0.0)
    freqs, dbfs = spectrum(first)
    step = scanner.on_frame(first, freqs, dbfs, now=t)
    assert step.action is ScanAction.TUNE
    assert step.centre_hz != first
    assert scanner.hits() == []


def test_a_carrier_is_logged_and_dwelled_on():
    scanner = Scanner(airband(min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    target = centre + 150e3
    freqs, dbfs = spectrum(centre, carriers=[(target, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)

    assert step.action is ScanAction.DWELL
    assert step.dwell_hz == pytest.approx(round(target / 25e3) * 25e3)
    assert scanner.state is ScanState.DWELLING
    assert len(step.new_hits) == 1
    assert len(scanner.hits()) == 1


def test_dwelling_does_not_retune_the_radio():
    """A hit is inside the window already received, so listening needs no tune."""
    scanner = Scanner(airband())
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(centre + 150e3, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)
    assert step.action is ScanAction.DWELL
    assert step.centre_hz == pytest.approx(centre), "the window must not move"


def test_survey_mode_logs_without_stopping():
    scanner = Scanner(airband(stop_on_signal=False, min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(centre + 150e3, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)
    assert step.action is ScanAction.TUNE          # moved on regardless
    assert len(scanner.hits()) == 1


def test_a_full_pass_wraps_and_counts():
    scanner = Scanner(airband(start_hz=118e6, end_hz=119e6))
    t = 0.0
    step = scanner.start(SPAN, now=t)
    seen = []
    for _ in range(scanner.window_count):
        seen.append(step.centre_hz)
        t = settle(scanner, step.centre_hz, t)
        freqs, dbfs = spectrum(step.centre_hz)
        step = scanner.on_frame(step.centre_hz, freqs, dbfs, now=t)
    assert step.finished_pass
    assert scanner.passes == 1
    assert step.centre_hz == pytest.approx(seen[0]), "should wrap to the first window"


# -- dwell behaviour ---------------------------------------------------------

def _dwell_on(scanner, centre, target, t):
    t = settle(scanner, centre, t)
    freqs, dbfs = spectrum(centre, carriers=[(target, 30)])
    scanner.on_frame(centre, freqs, dbfs, now=t)
    return t


def test_dwell_holds_while_the_signal_is_present():
    scanner = Scanner(airband(resume_after_quiet_s=2.0))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    target = centre + 150e3
    t = _dwell_on(scanner, centre, target, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(target, 30)])
    for _ in range(10):
        t += 0.5
        step = scanner.on_frame(centre, freqs, dbfs, now=t)
        assert step.action is ScanAction.NONE
    assert scanner.state is ScanState.DWELLING


def test_dwell_resumes_after_the_signal_stops():
    scanner = Scanner(airband(resume_after_quiet_s=2.0))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = _dwell_on(scanner, centre, centre + 150e3, 0.0)
    quiet_freqs, quiet_dbfs = spectrum(centre)
    t += 0.5
    assert scanner.on_frame(centre, quiet_freqs, quiet_dbfs, now=t).action is ScanAction.NONE
    t += 2.1
    step = scanner.on_frame(centre, quiet_freqs, quiet_dbfs, now=t)
    assert step.action is ScanAction.TUNE
    assert scanner.dwell_hz is None


def test_max_dwell_breaks_out_of_a_continuous_carrier():
    """A permanent signal such as ATIS must not trap the sweep forever."""
    scanner = Scanner(airband(max_dwell_s=5.0, resume_after_quiet_s=999.0))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    target = centre + 150e3
    t = _dwell_on(scanner, centre, target, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(target, 30)])
    t += 2.0
    assert scanner.on_frame(centre, freqs, dbfs, now=t).action is ScanAction.NONE
    t += 4.0
    assert scanner.on_frame(centre, freqs, dbfs, now=t).action is ScanAction.TUNE


def test_skip_abandons_the_current_transmission():
    scanner = Scanner(airband())
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = _dwell_on(scanner, centre, centre + 150e3, 0.0)
    step = scanner.skip(now=t)
    assert step.action is ScanAction.TUNE
    assert scanner.dwell_hz is None


def test_skip_does_nothing_when_not_dwelling():
    scanner = Scanner(airband())
    scanner.start(SPAN, now=0.0)
    assert scanner.skip(now=0.1).action is ScanAction.NONE


# -- lockout -----------------------------------------------------------------

def test_lock_out_current_stops_it_being_found_again():
    scanner = Scanner(airband())
    centre = scanner.start(SPAN, now=0.0).centre_hz
    target = centre + 150e3
    t = _dwell_on(scanner, centre, target, 0.0)

    locked = scanner.lock_out_current(now=t)
    assert locked == pytest.approx(round(target / 25e3) * 25e3)
    assert locked in scanner.lockout
    assert scanner.hits() == [], "a locked channel should not stay in the results"

    # Sweep the same window again: it must not be reported or dwelled on.
    scanner._index = 0
    scanner._state = ScanState.SEARCHING
    freqs, dbfs = spectrum(centre, carriers=[(target, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t + 10.0)
    assert step.action is ScanAction.TUNE
    assert step.new_hits == ()
    assert scanner.hits() == []


def test_lock_out_current_when_not_dwelling_is_harmless():
    scanner = Scanner(airband())
    assert scanner.lock_out_current(now=0.0) is None


def test_lock_out_snaps_to_the_channel_grid():
    scanner = Scanner(airband())
    assert scanner.lock_out(118_247_300) == pytest.approx(118_250_000)


def test_unlock_restores_a_channel():
    scanner = Scanner(airband())
    scanner.lock_out(118_250_000)
    assert scanner.unlock(118_250_000) is True
    assert scanner.unlock(118_250_000) is False
    assert scanner.lockout == set()


def test_preexisting_lockout_is_respected_from_the_start():
    scanner = Scanner(airband(), lockout={118_250_000})
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(118_250_000, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)
    assert step.new_hits == ()


# -- hit accumulation --------------------------------------------------------

def test_repeated_sightings_accumulate_rather_than_duplicate():
    # A range narrow enough for one window, so the sweep keeps revisiting it.
    scanner = Scanner(airband(start_hz=118e6, end_hz=118.5e6, stop_on_signal=False))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    assert scanner.window_count == 1
    target = 118.4e6
    t = 0.0
    for level in (20, 30, 25):
        t = settle(scanner, centre, t)
        freqs, dbfs = spectrum(centre, carriers=[(target, level)])
        scanner.on_frame(centre, freqs, dbfs, now=t)
        t += 1.0
    hits = scanner.hits()
    assert len(hits) == 1
    assert hits[0].freq_hz == pytest.approx(118.4e6)
    assert hits[0].count == 3
    assert hits[0].level_dbfs == pytest.approx(-118.0 + 30, abs=1.0)  # strongest kept
    assert hits[0].last_seen > hits[0].first_seen


def test_hits_are_sorted_by_frequency():
    scanner = Scanner(airband(stop_on_signal=False, min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(centre + 200e3, 25), (centre - 200e3, 25)])
    scanner.on_frame(centre, freqs, dbfs, now=t)
    found = [h.freq_hz for h in scanner.hits()]
    assert found == sorted(found) and len(found) == 2


def test_clear_hits():
    scanner = Scanner(airband(stop_on_signal=False, min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(centre + 150e3, 30)])
    scanner.on_frame(centre, freqs, dbfs, now=t)
    assert scanner.hits()
    scanner.clear_hits()
    assert scanner.hits() == []


def test_stop_halts_the_state_machine():
    scanner = Scanner(airband())
    centre = scanner.start(SPAN, now=0.0).centre_hz
    scanner.stop()
    assert not scanner.running
    freqs, dbfs = spectrum(centre, carriers=[(centre + 150e3, 30)])
    assert scanner.on_frame(centre, freqs, dbfs, now=5.0).action is ScanAction.NONE


def test_hits_outside_the_requested_range_are_not_reported():
    """The end windows overrun the range; anything beyond it must be discarded."""
    scanner = Scanner(airband(start_hz=118e6, end_hz=118.5e6, stop_on_signal=False,
                              min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    # One inside the asked-for range, one past its top edge but inside the window.
    freqs, dbfs = spectrum(centre, carriers=[(118.4e6, 30), (118.58e6, 30)])
    scanner.on_frame(centre, freqs, dbfs, now=t)
    found = [h.freq_hz for h in scanner.hits()]
    assert found == [pytest.approx(118.4e6)], f"out-of-range hit reported: {found}"


# -- confirmation by persistence ---------------------------------------------

def _one_window(**kw):
    kw.setdefault("start_hz", 118e6)
    kw.setdefault("end_hz", 118.5e6)
    kw.setdefault("stop_on_signal", False)
    return airband(**kw)


def _look_again(scanner, centre, target, level, t):
    """One more pass over the same single-window range."""
    t = settle(scanner, centre, t)
    freqs, dbfs = spectrum(centre, carriers=[(target, level)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)
    return step, t + 1.0


def test_a_single_sighting_is_not_reported_by_default():
    """Noise spikes reach 7-24 dB above the median, so one look is not evidence."""
    scanner = Scanner(_one_window())
    centre = scanner.start(SPAN, now=0.0).centre_hz
    step, _ = _look_again(scanner, centre, 118.4e6, 30, 0.0)
    assert step.new_hits == ()
    assert scanner.hits() == []
    # It is remembered as a candidate, just not reported.
    assert len(scanner.candidates()) == 1


def test_a_channel_is_reported_once_confirmed():
    scanner = Scanner(_one_window(min_sightings=2))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    step, t = _look_again(scanner, centre, 118.4e6, 30, 0.0)
    assert step.new_hits == ()
    step, t = _look_again(scanner, centre, 118.4e6, 30, t)
    assert len(step.new_hits) == 1
    assert [h.freq_hz for h in scanner.hits()] == [pytest.approx(118.4e6)]


def test_a_confirmed_channel_is_reported_only_once():
    """Later sightings must update it, not announce it again."""
    scanner = Scanner(_one_window(min_sightings=2))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = 0.0
    announcements = 0
    for _ in range(5):
        step, t = _look_again(scanner, centre, 118.4e6, 30, t)
        announcements += len(step.new_hits)
    assert announcements == 1
    assert scanner.hits()[0].count == 5


def test_confirmation_rejects_a_spike_that_moves():
    """A noise spike lands somewhere different each pass and is never confirmed."""
    scanner = Scanner(_one_window(min_sightings=2))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = 0.0
    for target in (118.20e6, 118.30e6, 118.40e6, 118.45e6):
        step, t = _look_again(scanner, centre, target, 30, t)
        assert step.new_hits == (), f"{target} confirmed after one sighting"
    assert scanner.hits() == []
    assert len(scanner.candidates()) == 4


def test_confirmation_can_be_turned_off():
    scanner = Scanner(_one_window(min_sightings=1))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    step, _ = _look_again(scanner, centre, 118.4e6, 30, 0.0)
    assert len(step.new_hits) == 1


def test_min_sightings_must_be_sane():
    with pytest.raises(ValueError):
        Scanner(airband(min_sightings=0))


def test_dwell_still_happens_on_a_first_sighting():
    """Short transmissions must not be missed while waiting for confirmation."""
    scanner = Scanner(airband(min_sightings=3))
    centre = scanner.start(SPAN, now=0.0).centre_hz
    t = settle(scanner, centre, 0.0)
    freqs, dbfs = spectrum(centre, carriers=[(centre + 150e3, 30)])
    step = scanner.on_frame(centre, freqs, dbfs, now=t)
    assert step.action is ScanAction.DWELL
    assert step.new_hits == (), "reported before confirmation"
