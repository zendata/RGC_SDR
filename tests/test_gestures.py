"""Swipe accumulation: no Qt, no synthetic trackpad events needed."""

import pytest

from src.rgc_sdr.ui.gestures import SwipeAccumulator, horizontal_dominates


def test_a_single_notch_is_one_step():
    acc = SwipeAccumulator(units_per_step=120.0)
    assert acc.add(120.0) == 1
    assert acc.add(-120.0) == -1


def test_small_deltas_accumulate_instead_of_firing_each_time():
    """A trackpad sends many tiny deltas; acting on each would tune far too fast."""
    acc = SwipeAccumulator(units_per_step=120.0)
    steps = [acc.add(20.0) for _ in range(5)]
    assert steps == [0, 0, 0, 0, 0]
    assert acc.add(20.0) == 1


def test_the_remainder_is_carried_forward():
    acc = SwipeAccumulator(units_per_step=100.0)
    assert acc.add(150.0) == 1
    assert acc.pending == pytest.approx(50.0)
    assert acc.add(50.0) == 1


def test_a_large_delta_yields_several_steps():
    acc = SwipeAccumulator(units_per_step=100.0)
    assert acc.add(350.0) == 3
    assert acc.pending == pytest.approx(50.0)


def test_reversing_direction_takes_effect_at_once():
    """Built-up momentum one way must not have to be undone before tuning back."""
    acc = SwipeAccumulator(units_per_step=100.0)
    acc.add(90.0)               # nearly a step upward
    assert acc.add(-90.0) == 0  # not -1 yet, but the upward total is discarded
    assert acc.add(-20.0) == -1


def test_direction_is_symmetric():
    up = SwipeAccumulator(units_per_step=100.0)
    down = SwipeAccumulator(units_per_step=100.0)
    assert up.add(500.0) == -down.add(-500.0)


def test_zero_delta_does_nothing():
    acc = SwipeAccumulator()
    assert acc.add(0.0) == 0
    assert acc.pending == 0.0


def test_reset_discards_partial_movement():
    acc = SwipeAccumulator(units_per_step=100.0)
    acc.add(90.0)
    acc.reset()
    assert acc.pending == 0.0
    assert acc.add(90.0) == 0


def test_units_per_step_must_be_positive():
    for bad in (0.0, -10.0):
        with pytest.raises(ValueError):
            SwipeAccumulator(units_per_step=bad)


def test_horizontal_dominates_decides_tune_versus_zoom():
    assert horizontal_dominates(100.0, 10.0) is True
    assert horizontal_dominates(-100.0, 10.0) is True
    assert horizontal_dominates(10.0, 100.0) is False
    assert horizontal_dominates(0.0, 0.0) is False
    # A diagonal swipe goes to whichever axis is larger; a tie means zoom, since that is
    # the gesture that already existed.
    assert horizontal_dominates(50.0, 50.0) is False
