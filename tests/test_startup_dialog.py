"""The start-up radio chooser (ui/startup_dialog.py): what it lists, and that a radio
that fails to open leaves it showing why rather than closing."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from src.rgc_sdr.device.profiles import PROFILES, Availability, profile_for  # noqa: E402
from src.rgc_sdr.ui.startup_dialog import StartupRadioDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _there(*keys):
    """Every profile, with `keys` connected."""
    return lambda: [Availability(p, True, p.key in keys) for p in PROFILES]


def _keys(dialog):
    return [dialog.radios.item(i).data(0x0100) for i in range(dialog.radios.count())]


def test_it_lists_the_radios_there_and_always_the_wifi_705(qapp):
    dialog = StartupRadioDialog("Could not open IC-705 (WiFi): timed out",
                                lambda key: None, _there("hackrf", "rtlsdr"))
    assert _keys(dialog) == ["hackrf", "rtlsdr", "icom705net"]
    assert "timed out" in dialog.problem.text()
    assert dialog.open_button.isEnabled()
    dialog.close()


def test_a_radio_that_fails_leaves_the_reason_and_the_list(qapp):
    attempts = []

    def opener(key):
        attempts.append(key)
        if key == "hackrf":
            raise RuntimeError("Unable to open")
        return f"source:{key}"

    dialog = StartupRadioDialog("first problem", opener, _there("hackrf", "rtlsdr"))
    dialog._open()                                      # the HackRF, first in the list
    assert dialog.source is None and "Unable to open" in dialog.problem.text()
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    dialog.radios.setCurrentRow(1)
    dialog._open()
    assert attempts == ["hackrf", "rtlsdr"] and dialog.source == "source:rtlsdr"
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_look_again_finds_a_radio_plugged_in_since(qapp):
    there = {"keys": ("rtlsdr",)}
    dialog = StartupRadioDialog("x", lambda key: None,
                                lambda: _there(*there["keys"])())
    assert _keys(dialog) == ["rtlsdr", "icom705net"]
    there["keys"] = ("rtlsdr", "airspyhf")
    dialog.again_button.click()
    assert _keys(dialog) == ["airspyhf", "rtlsdr", "icom705net"]
    dialog.close()


def test_a_cancelled_login_stays_in_the_chooser(qapp):
    dialog = StartupRadioDialog("x", lambda key: None, _there())
    assert _keys(dialog) == ["icom705net"]
    dialog._open()                                      # the login was cancelled: None
    assert dialog.source is None
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    dialog.close()
    assert profile_for("icom705net") is not None


def test_the_radio_that_failed_is_last_and_not_chosen(qapp):
    dialog = StartupRadioDialog("x", lambda key: None, _there("rtlsdr"), failed="icom705net")
    assert _keys(dialog) == ["rtlsdr", "icom705net"] and dialog.chosen_key() == "rtlsdr"
    dialog.close()
    other = StartupRadioDialog("x", lambda key: None, _there("hackrf", "rtlsdr"),
                               failed="hackrf")
    assert _keys(other) == ["rtlsdr", "icom705net", "hackrf"]
    other.close()
