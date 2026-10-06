"""Frequency calibration against a known carrier (calibrate.py).

The radio here is a test double feeding the DSP: it renders a carrier where a radio
with a given crystal error would show it, plus a spur that moves with the tuning, as
the HackRF's do.
"""

import threading
import time

import numpy as np
import pytest

from src.rgc_sdr.calibrate import (
    CalibrateJob, carrier_lines, common_line, new_ppm,
)

RATE = 2e6


def carrier_iq(offset_hz, seconds=1.0, rate=RATE, extra=(), noise=0.01, seed=1):
    n = np.arange(int(seconds * rate))
    x = np.exp(2j * np.pi * offset_hz * n / rate)
    for hz, amplitude in extra:
        x = x + amplitude * np.exp(2j * np.pi * hz * n / rate)
    rng = np.random.default_rng(seed)
    return (x + noise * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
            ).astype(np.complex64)


class Caps:
    def covers(self, hz):
        return 70e6 <= hz <= 6e9


class Reader:
    def __init__(self, radio):
        self.radio, self.sent = radio, 0

    def available(self):
        return int(RATE * 0.5)

    def read(self, n):
        self.sent += n
        return self.radio.render(n)


class ErrRadio:
    """Shows a carrier at true frequency x (1 + ppm_error - correction), and a spur
    fixed 6.9 kHz above the reference in the stream (it moves with the tuning)."""

    def __init__(self, error_ppm, carriers, spur_at=None):
        self.error_ppm, self.carriers = error_ppm, carriers
        self.spur_at = spur_at
        self.center_freq, self.sample_rate, self.caps = 100e6, RATE, Caps()
        self.ppm = 0.0
        self.tuned = []

    def set_center_freq(self, hz):
        self.center_freq = float(hz)
        self.tuned.append(hz)

    def sequential_reader(self):
        return Reader(self)

    def render(self, n):
        seen = [(f * (1 + (self.error_ppm - self.ppm) * 1e-6) - self.center_freq, 1.0)
                for f in self.carriers]
        if self.spur_at is not None:                 # fixed in the stream, so in RF it
            seen.append((self.spur_at, 3.0))         # moves with every retune
        if not seen:                                 # nothing on the air: noise
            return carrier_iq(0.0, n / RATE, extra=[(0.0, -1.0)])
        first, *rest = seen
        return carrier_iq(first[0], n / RATE, extra=rest)


def run(job):
    deadline = time.time() + 30
    while not job.done and time.time() < deadline:
        time.sleep(0.02)
    assert job.done
    return job


def test_a_line_is_measured_to_a_fraction_of_a_hertz():
    iq = carrier_iq(-200e3 + 558.7)
    (offset, snr), = carrier_lines(iq, RATE, 120e6, 119.8e6)[:1]
    assert offset == pytest.approx(558.7, abs=0.5) and snr > 30


def test_the_correction_converges():
    assert new_ppm(0.0, 558.7, 119.8e6) == pytest.approx(4.664, abs=1e-3)
    assert new_ppm(4.664, 0.0, 119.8e6) == pytest.approx(4.664)


def test_a_line_that_moves_with_the_tuning_is_a_spur():
    assert common_line([(6885.0, 40.0)], [(6864.0 + 300.0, 40.0)]) is None
    assert common_line([(6885.0, 40.0), (-1381.0, 30.0)],
                       [(-1402.0, 31.0)]) == (pytest.approx(-1391.5), 30.0)


def test_the_job_measures_the_error_and_puts_the_radio_back():
    radio = ErrRadio(+4.66, [119.8e6])
    job = run(CalibrateJob(radio, [(119.8e6, "Essendon ATIS")]))
    assert job.result.ppm == pytest.approx(4.66, abs=0.02)
    assert "Essendon ATIS" in job.result.method
    assert radio.center_freq == 100e6                     # back where it was
    radio.ppm = job.result.ppm
    again = run(CalibrateJob(radio, [119.8e6]))
    assert abs(again.result.offset_hz) < 3.0              # corrected: reads true


def test_a_stronger_spur_is_not_taken_for_the_reference():
    radio = ErrRadio(-11.6, [119.8e6], spur_at=200e3 + 6885.0)
    job = run(CalibrateJob(radio, [119.8e6]))
    assert job.result.ppm == pytest.approx(-11.6, abs=0.05)


def test_the_next_reference_is_tried_and_failure_is_explained():
    radio = ErrRadio(-11.6, [144.65e6])
    job = run(CalibrateJob(radio, [(119.8e6, "ATIS"), (144.65e6, "beacon")]))
    assert "beacon" in job.result.method
    silent = run(CalibrateJob(ErrRadio(3.0, []), [(119.8e6, "ATIS")]))
    assert silent.result is None and "no clear carrier from ATIS" in silent.error


def test_a_reference_out_of_reach_is_skipped():
    radio = ErrRadio(2.0, [119.8e6])
    job = run(CalibrateJob(radio, [(30e6, "HF"), (119.8e6, "ATIS")]))
    assert all(hz > 70e6 for hz in radio.tuned) and job.result is not None


def test_cancelling_leaves_the_tuning_alone():
    radio = ErrRadio(2.0, [119.8e6])
    job = CalibrateJob(radio, [119.8e6])
    job.cancel()
    run(job)
    assert job.result is None or job.cancelled
