import os

import pytest

# Qt must run windowless under pytest; set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: needs a real SDR attached; skipped automatically when absent"
    )


def _devices():
    try:
        from src.rgc_sdr.device.source import enumerate_devices

        return enumerate_devices()
    except Exception:
        return []


@pytest.fixture(scope="session")
def sdr_devices():
    return _devices()


def pytest_collection_modifyitems(config, items):
    if _devices():
        return
    skip = pytest.mark.skip(reason="no SDR attached")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
