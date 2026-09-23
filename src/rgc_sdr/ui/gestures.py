"""Trackpad gesture handling for the plot views.

A two-finger vertical swipe already zooms the frequency axis, which pyqtgraph does for
us. A horizontal swipe should tune instead. Both arrive as wheel events, so they are
told apart by which axis dominates.

The accumulator is plain Python with no Qt, so the threshold behaviour can be tested
without synthesising trackpad events.
"""

from __future__ import annotations

#: Direction of a sideways swipe. +1 means a rightward wheel delta tunes upward.
#: Trackpad sign depends on the system's "natural scrolling" setting, so this is the one
#: constant to flip if the gesture feels backwards.
TUNE_DIRECTION = 1.0

#: Wheel units per tuning step. One mouse notch is 120, so a notch moves one step and a
#: trackpad swipe -- which arrives as a stream of much smaller deltas -- moves gradually.
UNITS_PER_STEP = 120.0


class SwipeAccumulator:
    """Turns a stream of small wheel deltas into discrete, signed steps.

    Trackpads deliver many tiny deltas rather than notches, so acting on each one would
    make tuning uncontrollably fast. Deltas are summed instead and a step is emitted each
    time the total crosses the threshold, leaving the remainder for the next event.
    """

    def __init__(self, units_per_step: float = UNITS_PER_STEP) -> None:
        if units_per_step <= 0:
            raise ValueError("units_per_step must be positive")
        self.units_per_step = float(units_per_step)
        self._total = 0.0

    def reset(self) -> None:
        self._total = 0.0

    @property
    def pending(self) -> float:
        return self._total

    def add(self, delta: float) -> int:
        """Accumulate `delta`; return how many whole steps it completed (signed)."""
        if delta == 0.0:
            return 0
        # A reversal should take effect immediately rather than first working through
        # momentum built up in the other direction.
        if (delta > 0) != (self._total > 0) and self._total != 0.0:
            self._total = 0.0
        self._total += float(delta)
        steps = int(self._total / self.units_per_step)
        if steps:
            self._total -= steps * self.units_per_step
        return steps


def horizontal_dominates(delta_x: float, delta_y: float) -> bool:
    """True when a wheel event is a sideways swipe rather than an up/down one."""
    return abs(delta_x) > abs(delta_y)


#: A trackpad pixel is worth much less than a mouse notch, so pixel deltas are scaled up
#: to the same units before accumulating. One notch is 120 units and a deliberate swipe
#: covers on the order of a hundred pixels.
PIXELS_TO_UNITS = 3.0


def wheel_deltas(event) -> tuple[float, float]:
    """Pull (dx, dy) out of a QWheelEvent in consistent units.

    macOS trackpads populate `pixelDelta` and may leave `angleDelta` empty or coarse,
    while mice populate `angleDelta` only. Reading just one of them is why a sideways
    swipe did nothing. Whichever axis pair carries more information is used.
    """
    angle = event.angleDelta()
    pixel = event.pixelDelta()
    ax, ay = float(angle.x()), float(angle.y())
    px, py = float(pixel.x()) * PIXELS_TO_UNITS, float(pixel.y()) * PIXELS_TO_UNITS
    if abs(px) + abs(py) > abs(ax) + abs(ay):
        return px, py
    return ax, ay
