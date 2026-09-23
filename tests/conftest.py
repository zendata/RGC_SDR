import os

import pytest

# Qt must run windowless under pytest; set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: needs a real SDR attached; skipped automatically when absent"
    )


_STATE = {}


def _devices():
    if "list" not in _STATE:
        try:
            from src.rgc_sdr.device.source import enumerate_devices

            _STATE["list"] = enumerate_devices()
        except Exception:
            _STATE["list"] = []
    return _STATE["list"]


def _usable():
    """Enumerable *and* openable.

    A radio that is plugged in but held by another process (the app running in another
    window) still enumerates, so checking only for its presence made every hardware test
    fail with "Unable to open" instead of skipping. Probed once per session, and released
    straight away so the probe itself does not hold the device.
    """
    if "usable" not in _STATE:
        devices = _devices()
        if not devices:
            _STATE["usable"] = False
            return False
        import gc

        try:
            import SoapySDR  # type: ignore

            driver = devices[0].get("driver", "airspyhf")
            handle = SoapySDR.Device(f"driver={driver}")
            del handle
            gc.collect()
            _STATE["usable"] = True
        except Exception:
            _STATE["usable"] = False
    return _STATE["usable"]


@pytest.fixture(scope="session")
def sdr_devices():
    return _devices()


def pytest_collection_modifyitems(config, items):
    if _usable():
        return
    reason = (
        "SDR attached but in use by another process"
        if _devices()
        else "no SDR attached"
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
