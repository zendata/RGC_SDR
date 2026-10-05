"""The decode worker: its own NBFM chain and decoder over the source's IQ."""

import time

import numpy as np
import pytest

from src.rgc_sdr.decoding import DECODERS, DecodeWorker
from src.rgc_sdr.device.source import SequentialReader, _Ring
from tests.test_pocsag import _alpha_chunks, encode, fsk_iq

RATE = 192_000.0


class RingSource:
    """A ring and a sample rate: all the worker reads."""

    def __init__(self, rate=RATE):
        self.sample_rate = rate
        self.ring = _Ring(2_000_000)

    def sequential_reader(self):
        return SequentialReader(self.ring)


def page_iq(offset_hz, text="WORKER TEST", baud=1200):
    """A POCSAG page `offset_hz` away from the centre of a 192 kS/s stream."""
    iq = fsk_iq(encode(4321, 3, _alpha_chunks(text)), baud, rate=RATE)
    n = np.arange(iq.size)
    tail = np.zeros(int(RATE), np.complex64)          # a second of quiet after it
    shifted = np.concatenate([iq * np.exp(2j * np.pi * offset_hz * n / RATE), tail])
    rng = np.random.default_rng(11)                   # a receiver's noise floor
    noise = 0.02 * (rng.standard_normal(shifted.size) + 1j * rng.standard_normal(shifted.size))
    return (shifted + noise).astype(np.complex64)


def test_decodes_at_the_listening_offset():
    worker = DecodeWorker(RingSource(), "pocsag", offset_hz=25e3)
    found = []
    for block in np.array_split(page_iq(25e3), 40):
        found += worker.process(block)
    assert [(m.address, m.text) for m in found] == [(4321, "WORKER TEST")]
    assert worker.take() == found and worker.take() == []


def test_off_the_offset_there_is_nothing():
    worker = DecodeWorker(RingSource(), "pocsag", offset_hz=-40e3)
    found = []
    for block in np.array_split(page_iq(25e3), 40):
        found += worker.process(block)
    assert found == []


def test_runs_on_its_own_thread():
    src = RingSource()
    worker = DecodeWorker(src, "pocsag", offset_hz=0.0)
    worker.start()
    try:
        for block in np.array_split(page_iq(0.0), 20):
            src.ring.write(block)
            time.sleep(0.01)
        deadline = time.time() + 5
        got = []
        while not got and time.time() < deadline:
            got = worker.take()
            time.sleep(0.05)
    finally:
        worker.stop()
    assert [m.text for m in got] == ["WORKER TEST"] and worker.errors == 0


def test_unknown_decoder_is_refused():
    with pytest.raises(ValueError):
        DecodeWorker(RingSource(), "morse-by-smoke")


def test_every_decoder_is_described():
    for key, spec in DECODERS.items():
        assert spec.label and spec.bandwidth_hz > 0
