import numpy as np

from src.rgc_sdr.device.spike import stream_device, synthesize_iq


def test_synthesized_stream_stats():
    iq = synthesize_iq(n=200_000)
    stats = stream_device(0.3, plant_samples=iq)
    assert stats.samples > 0
    assert 100_000 < stats.rate() < 10_000_000
    assert stats.snr_db(stats.last_block) > 6.0
