"""Entry-point tests.

These exist because a syntax error in `__main__.py` once shipped while 417 tests passed:
nothing imported the module, so the single most user-facing file was the least covered.
Importing it is most of the value here.
"""

import json

import pytest

from src.rgc_sdr import __main__ as entry


def test_the_module_imports():
    """A syntax error here means the app cannot start at all."""
    assert callable(entry.main)
    assert callable(entry.build_parser)


def test_parser_builds_and_accepts_no_arguments():
    args = entry.build_parser().parse_args([])
    assert args.freq is None            # so saved state can supply it
    assert args.driver is None          # so the remembered or connected radio is used
    assert args.debug_gestures is False


def test_every_documented_flag_parses():
    args = entry.build_parser().parse_args([
        "--driver", "airspyhf", "--serial", "abc",
        "--freq", "7.1e6", "--rate", "768e3", "--zoom", "8", "--fft", "4096",
        "--fps", "30", "--rows", "256", "--bins", "512", "--colormap", "viridis",
        "--min-db", "-120", "--max-db", "-60", "--agc",
        "--mode", "usb", "--volume", "0.5", "--offset", "1500",
        "--squelch", "-95", "--step", "100", "--bandwidth", "2700",
        "--no-audio", "--recordings", "/tmp/x", "--debug-gestures",
        "--no-restore",
    ])
    assert args.freq == pytest.approx(7.1e6)
    assert args.zoom == 8
    assert args.mode == "usb"
    assert args.step == pytest.approx(100.0)
    assert args.debug_gestures is True
    assert args.no_audio is True


def test_step_and_zoom_reject_nonsense():
    parser = entry.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--zoom", "7"])          # not a power of two
    with pytest.raises(SystemExit):
        parser.parse_args(["--fft", "1000"])        # not an offered size
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "fmx"])


def test_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exit_info:
        entry.build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "spectrum and waterfall" in capsys.readouterr().out


def test_list_devices_returns_a_status_code():
    """Runs the real enumeration; 0 with a radio attached, 1 without."""
    assert entry.list_devices() in (0, 1)


def test_list_memories_on_an_empty_store(monkeypatch, tmp_path, capsys):
    from src.rgc_sdr.settings import Settings

    monkeypatch.setattr(entry.Settings, "load",
                        classmethod(lambda cls, path=None: Settings(tmp_path / "s.json")))
    assert entry.main(["--list-memories"]) == 0
    assert "No saved memories" in capsys.readouterr().out


def test_list_memories_prints_what_is_stored(monkeypatch, tmp_path, capsys):
    from src.rgc_sdr.settings import Settings, Snapshot

    store = Settings(tmp_path / "s.json")
    store.add_memory("Radio 4 LW", Snapshot(freq_hz=198e3, mode="am"))
    monkeypatch.setattr(entry.Settings, "load", classmethod(lambda cls, path=None: store))
    assert entry.main(["--list-memories"]) == 0
    out = capsys.readouterr().out
    assert "Radio 4 LW" in out and "0.1980 MHz" in out


def test_forget_clears_the_file(monkeypatch, tmp_path, capsys):
    from src.rgc_sdr.settings import Settings, Snapshot

    path = tmp_path / "s.json"
    store = Settings(path)
    store.add_memory("gone", Snapshot(freq_hz=7.1e6))
    store.last = Snapshot(freq_hz=7.1e6)
    store.save()
    assert json.loads(path.read_text())["memories"]

    # Bound before patching, or the patched loader calls itself forever.
    real_load = Settings.load
    monkeypatch.setattr(entry.Settings, "load",
                        classmethod(lambda cls, p=None: real_load(path)))
    assert entry.main(["--forget"]) == 0
    assert json.loads(path.read_text())["memories"] == []
    assert "cleared" in capsys.readouterr().out


def test_unknown_memory_is_an_error(monkeypatch, tmp_path, capsys):
    from src.rgc_sdr.settings import Settings

    monkeypatch.setattr(entry.Settings, "load",
                        classmethod(lambda cls, path=None: Settings(tmp_path / "s.json")))
    assert entry.main(["--memory", "nope"]) == 2
    assert "No memory named" in capsys.readouterr().err


def test_a_missing_driver_reports_cleanly(monkeypatch, tmp_path, capsys):
    """An unopenable device must exit with a message, not a traceback."""
    from src.rgc_sdr.settings import Settings

    monkeypatch.setattr(entry.Settings, "load",
                        classmethod(lambda cls, path=None: Settings(tmp_path / "s.json")))
    assert entry.main(["--driver", "definitely-not-a-driver"]) == 2
    assert "Could not open" in capsys.readouterr().err


def test_run_accepts_every_keyword_main_passes_it():
    """Guards the call that actually broke: argument order and names must line up."""
    import inspect

    from src.rgc_sdr.ui.main_window import run

    accepted = inspect.signature(run).parameters
    assert "debug_gestures" in accepted
    # `source` must stay positional-or-keyword and first.
    assert list(accepted)[0] == "source"

    window_params = inspect.signature(
        __import__("src.rgc_sdr.ui.main_window", fromlist=["MainWindow"]).MainWindow.__init__
    ).parameters
    for name in ("fft_size", "fps", "history_rows", "waterfall_bins", "colormap",
                 "levels", "decimation", "peak_hold", "mode", "volume", "offset_hz",
                 "squelch_dbfs", "bandwidth_hz", "step_hz", "enable_audio",
                 "recordings_dir", "settings"):
        assert name in window_params, f"main() passes {name}, MainWindow does not accept it"


def test_choose_driver_honours_an_explicit_request():
    assert entry.choose_driver("hackrf", "airspyhf") == "hackrf"


def test_choose_driver_prefers_the_remembered_radio_when_present(monkeypatch):
    from src.rgc_sdr.device import profiles

    def fake(devices=None, modules=None):
        return [profiles.Availability(p, True, p.key in ("airspyhf", "hackrf"))
                for p in profiles.PROFILES]
    monkeypatch.setattr(profiles, "availability", fake)
    assert entry.choose_driver(None, "hackrf") == "hackrf"


def test_choose_driver_falls_back_to_what_is_connected(monkeypatch):
    """The remembered radio may have been unplugged since last time."""
    from src.rgc_sdr.device import profiles

    def fake(devices=None, modules=None):
        return [profiles.Availability(p, True, p.key == "airspyhf") for p in profiles.PROFILES]
    monkeypatch.setattr(profiles, "availability", fake)
    assert entry.choose_driver(None, "hackrf") == "airspyhf"


def test_list_shows_every_supported_radio(capsys):
    entry.list_devices()
    out = capsys.readouterr().out
    for name in ("Airspy HF+", "HackRF", "RTL-SDR", "Pluto"):
        assert name in out
