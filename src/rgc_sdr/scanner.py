"""Band scanner: sweep a frequency range, log what is active, stop on transmissions.

No Qt and no device access. The scanner is a state machine driven one frame at a time by
whatever owns the display loop: it is handed a spectrum and returns an instruction. That
keeps it testable against synthetic spectra and leaves all I/O to the caller.

Dwelling moves the *audio offset*, not the radio. A hit is by construction inside the
window already being received, so listening to it needs no retune and therefore no settle
delay -- resuming the sweep is instant, and a short transmission is not missed while the
front end recovers from a tune.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from .dsp.detect import Detection, detect_channels, noise_floor_dbfs, plan_windows, snap_to_grid


class ScanState(Enum):
    IDLE = "idle"
    #: Waiting for the front end to settle after a retune.
    SETTLING = "settling"
    #: Looking at the current window for activity.
    SEARCHING = "searching"
    #: Listening to a hit inside the current window.
    DWELLING = "dwelling"


class ScanAction(Enum):
    NONE = "none"
    #: Retune the radio to `centre_hz` and clear the audio offset.
    TUNE = "tune"
    #: Keep the tuning; listen at `dwell_hz` via the audio offset.
    DWELL = "dwell"
    #: Stop listening; the sweep continues next frame.
    RESUME = "resume"


@dataclass(frozen=True)
class ScanConfig:
    start_hz: float
    end_hz: float
    #: Channel spacing. 25 kHz for airband, 8.33 kHz for the newer European allocations.
    step_hz: float = 25e3
    #: Detection threshold *above the measured noise floor*, so one setting suits any band.
    threshold_db: float = 10.0
    usable_fraction: float = 0.8
    #: Time for the front end to settle after a retune; the source also discards ~150 ms.
    settle_s: float = 0.35
    #: How long a channel must be quiet before the sweep moves on.
    resume_after_quiet_s: float = 2.0
    #: Cap on one transmission, so a continuous carrier (ATIS, data) cannot trap the sweep.
    max_dwell_s: float = 30.0
    #: False surveys the range without stopping, which builds a list fast.
    stop_on_signal: bool = True
    dc_guard_hz: float = 1.5e3
    #: Passes a channel must appear on before it is reported as found.
    #:
    #: Measured 2026-09-23: the loudest *noise* bin in a window sits 7-24 dB above the
    #: median, so no fixed threshold separates signal from noise, and a narrowband AM
    #: carrier is only 2-3 bins wide -- the same as a noise spike -- so width cannot
    #: separate them either. What does is persistence: a real channel reappears at the
    #: same frequency every pass, a noise spike does not. The scanner still *dwells* on a
    #: first sighting, so short transmissions are not missed; only the stored list waits
    #: for confirmation.
    min_sightings: int = 2

    def validate(self) -> None:
        if self.end_hz <= self.start_hz:
            raise ValueError("scan end must be above scan start")
        if self.step_hz <= 0:
            raise ValueError("step must be positive")
        if self.min_sightings < 1:
            raise ValueError("min_sightings must be at least 1")


@dataclass
class ScanHit:
    """A channel the scanner has heard, and how often."""

    freq_hz: float
    level_dbfs: float
    snr_db: float
    first_seen: float
    last_seen: float
    count: int = 1

    def merge(self, detection: Detection, now: float) -> "ScanHit":
        return replace(
            self,
            level_dbfs=max(self.level_dbfs, detection.level_dbfs),
            snr_db=max(self.snr_db, detection.snr_db),
            last_seen=now,
            count=self.count + 1,
        )


@dataclass(frozen=True)
class ScanStep:
    """What the caller should do with the radio this frame."""

    action: ScanAction = ScanAction.NONE
    centre_hz: float | None = None
    dwell_hz: float | None = None
    new_hits: tuple[ScanHit, ...] = ()
    finished_pass: bool = False


@dataclass
class Scanner:
    config: ScanConfig
    lockout: set[float] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.config.validate()
        self._state = ScanState.IDLE
        self._windows: list[float] = []
        self._index = 0
        self._settle_until = 0.0
        self._dwell_hz: float | None = None
        self._dwell_started = 0.0
        self._last_active = 0.0
        self._hits: dict[float, ScanHit] = {}
        self.passes = 0

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> ScanState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is not ScanState.IDLE

    @property
    def dwell_hz(self) -> float | None:
        return self._dwell_hz

    @property
    def window_centre(self) -> float | None:
        if not self._windows:
            return None
        return self._windows[self._index % len(self._windows)]

    @property
    def window_count(self) -> int:
        return len(self._windows)

    @property
    def window_index(self) -> int:
        return self._index

    def hits(self) -> list[ScanHit]:
        """Channels confirmed on enough passes to be worth reporting."""
        return [
            self._hits[key]
            for key in sorted(self._hits)
            if self._hits[key].count >= self.config.min_sightings
        ]

    def candidates(self) -> list[ScanHit]:
        """Everything seen, confirmed or not. Useful for diagnosing a threshold."""
        return [self._hits[key] for key in sorted(self._hits)]

    def clear_hits(self) -> None:
        self._hits.clear()

    # -- control -----------------------------------------------------------

    def start(self, span_hz: float, now: float | None = None) -> ScanStep:
        """Begin sweeping. Returns the first instruction."""
        now = time.monotonic() if now is None else now
        self._windows = plan_windows(
            self.config.start_hz,
            self.config.end_hz,
            span_hz,
            self.config.usable_fraction,
            grid_hz=self.config.step_hz,
        )
        self._index = 0
        self.passes = 0
        self._dwell_hz = None
        return self._tune_current(now)

    def stop(self) -> ScanStep:
        self._state = ScanState.IDLE
        self._dwell_hz = None
        return ScanStep(action=ScanAction.RESUME)

    def skip(self, now: float | None = None) -> ScanStep:
        """Abandon the current transmission and carry on."""
        if self._state is not ScanState.DWELLING:
            return ScanStep()
        return self._resume(time.monotonic() if now is None else now)

    def lock_out_current(self, now: float | None = None) -> float | None:
        """Lock out what is being listened to, then carry on.

        Returns the locked frequency so the caller can persist it.
        """
        if self._dwell_hz is None:
            return None
        frequency = snap_to_grid(self._dwell_hz, self.config.step_hz)
        self.lockout.add(frequency)
        self._hits.pop(frequency, None)
        self._resume(time.monotonic() if now is None else now)
        return frequency

    def lock_out(self, freq_hz: float) -> float:
        frequency = snap_to_grid(freq_hz, self.config.step_hz)
        self.lockout.add(frequency)
        self._hits.pop(frequency, None)
        return frequency

    def unlock(self, freq_hz: float) -> bool:
        frequency = snap_to_grid(freq_hz, self.config.step_hz)
        if frequency in self.lockout:
            self.lockout.discard(frequency)
            return True
        return False

    # -- the frame loop ----------------------------------------------------

    def _tune_current(self, now: float) -> ScanStep:
        self._state = ScanState.SETTLING
        self._settle_until = now + self.config.settle_s
        return ScanStep(action=ScanAction.TUNE, centre_hz=self.window_centre)

    def _advance(self, now: float) -> ScanStep:
        self._index += 1
        finished = False
        if self._index >= len(self._windows):
            self._index = 0
            self.passes += 1
            finished = True
        step = self._tune_current(now)
        return replace(step, finished_pass=finished)

    def _resume(self, now: float) -> ScanStep:
        self._dwell_hz = None
        return replace(self._advance(now), action=ScanAction.TUNE)

    def on_frame(
        self, centre_hz: float, freqs: np.ndarray, dbfs: np.ndarray, now: float | None = None
    ) -> ScanStep:
        """Feed one spectrum. Returns what to do with the radio."""
        now = time.monotonic() if now is None else now
        if self._state is ScanState.IDLE:
            return ScanStep()
        if self._state is ScanState.SETTLING:
            if now < self._settle_until:
                return ScanStep()
            self._state = ScanState.SEARCHING
            return ScanStep()
        if self._state is ScanState.DWELLING:
            return self._dwell_frame(freqs, dbfs, now)
        return self._search_frame(centre_hz, freqs, dbfs, now)

    def _search_frame(
        self, centre_hz: float, freqs: np.ndarray, dbfs: np.ndarray, now: float
    ) -> ScanStep:
        half = self.config.usable_fraction * float(freqs[-1] - freqs[0]) / 2.0
        # Clamped to the range that was actually asked for: the first and last windows
        # extend past it, and reporting hits outside the requested band is wrong.
        detections = detect_channels(
            freqs,
            dbfs,
            threshold_db=self.config.threshold_db,
            step_hz=self.config.step_hz,
            usable_lo=max(centre_hz - half, self.config.start_hz),
            usable_hi=min(centre_hz + half, self.config.end_hz),
            dc_guard_hz=self.config.dc_guard_hz,
            centre_hz=centre_hz,
            lockout=frozenset(self.lockout),
        )
        new: list[ScanHit] = []
        needed = self.config.min_sightings
        for detection in detections:
            existing = self._hits.get(detection.freq_hz)
            if existing is None:
                hit = ScanHit(
                    detection.freq_hz, detection.level_dbfs, detection.snr_db, now, now
                )
                self._hits[detection.freq_hz] = hit
                if hit.count >= needed:
                    new.append(hit)
            else:
                merged = existing.merge(detection, now)
                self._hits[detection.freq_hz] = merged
                if merged.count == needed:
                    # Just confirmed: report it once, not on every later sighting.
                    new.append(merged)

        if detections and self.config.stop_on_signal:
            strongest = max(detections, key=lambda d: d.level_dbfs)
            self._dwell_hz = strongest.freq_hz
            self._dwell_started = now
            self._last_active = now
            self._state = ScanState.DWELLING
            return ScanStep(
                action=ScanAction.DWELL,
                centre_hz=centre_hz,
                dwell_hz=strongest.freq_hz,
                new_hits=tuple(new),
            )
        return replace(self._advance(now), new_hits=tuple(new))

    def _dwell_frame(self, freqs: np.ndarray, dbfs: np.ndarray, now: float) -> ScanStep:
        if self._dwell_hz is None:
            return self._resume(now)
        if now - self._dwell_started >= self.config.max_dwell_s:
            return self._resume(now)

        half = max(self.config.step_hz / 2.0, 1.0)
        channel = np.abs(freqs - self._dwell_hz) <= half
        if channel.any():
            level = float(np.max(dbfs[channel]))
            # Hysteresis: hold on a few dB below the trigger, so a signal fading at the
            # threshold does not chatter between dwelling and sweeping.
            if level > noise_floor_dbfs(dbfs) + self.config.threshold_db - 3.0:
                self._last_active = now
        if now - self._last_active >= self.config.resume_after_quiet_s:
            return self._resume(now)
        return ScanStep(action=ScanAction.NONE, dwell_hz=self._dwell_hz)
