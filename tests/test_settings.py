"""Memory store and last-state persistence. Uses tmp_path; never touches the real file."""

import json

import pytest

from src.rgc_sdr.settings import Memory, Settings, Snapshot


def a_snapshot(**kw):
    snap = Snapshot(freq_hz=7.1e6, sample_rate=768e3, decimation=4)
    for k, v in kw.items():
        setattr(snap, k, v)
    return snap


# -- round trip --------------------------------------------------------------

def test_save_and_reload_last_state(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(path)
    s.last = a_snapshot(freq_hz=14.2e6, decimation=8, colormap="turbo")
    s.save()

    again = Settings.load(path)
    assert again.last is not None
    assert again.last.freq_hz == pytest.approx(14.2e6)
    assert again.last.decimation == 8
    assert again.last.colormap == "turbo"


def test_save_and_reload_memories(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(path)
    s.add_memory("Radio 4 LW", a_snapshot(freq_hz=198e3, sample_rate=192e3))
    s.add_memory("40m", a_snapshot(freq_hz=7.1e6))
    s.save()

    again = Settings.load(path)
    assert again.names() == ["40m", "Radio 4 LW"]  # sorted, case-insensitive
    assert again.get_memory("radio 4 lw").snapshot.freq_hz == pytest.approx(198e3)


def test_none_colour_levels_survive_a_round_trip(tmp_path):
    """None means auto-fit; it must not come back as 0.0."""
    path = tmp_path / "s.json"
    s = Settings(path)
    s.last = a_snapshot(min_db=None, max_db=None)
    s.save()
    assert Settings.load(path).last.min_db is None
    assert Settings.load(path).last.max_db is None


def test_explicit_colour_levels_survive(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(path)
    s.last = a_snapshot(min_db=-120.0, max_db=-70.0)
    s.save()
    assert Settings.load(path).last.min_db == pytest.approx(-120.0)


# -- robustness --------------------------------------------------------------

def test_missing_file_gives_empty_settings(tmp_path):
    s = Settings.load(tmp_path / "nope.json")
    assert s.last is None and s.memories == []


def test_corrupt_file_does_not_raise(tmp_path):
    """A bad settings file must never stop the radio from starting."""
    path = tmp_path / "s.json"
    path.write_text("{this is not json")
    s = Settings.load(path)
    assert s.last is None and s.memories == []


@pytest.mark.parametrize("content", ["[]", "null", '"text"', "123"])
def test_unexpected_json_shapes_are_ignored(tmp_path, content):
    path = tmp_path / "s.json"
    path.write_text(content)
    s = Settings.load(path)
    assert s.last is None and s.memories == []


def test_partial_and_junk_fields_fall_back_to_defaults(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "last": {"freq_hz": 5e6, "decimation": "not a number", "bogus_key": 1},
        "memories": [{"name": "ok", "snapshot": {"freq_hz": 1e6}},
                     {"name": "   "},            # blank name, dropped
                     "not a dict"],              # wrong type, dropped
    }))
    s = Settings.load(path)
    assert s.last.freq_hz == pytest.approx(5e6)
    assert s.last.decimation == 1                # junk replaced by the default
    assert s.last.fft_size == 4096               # absent, so default
    assert s.names() == ["ok"]


def test_negative_decimation_is_clamped():
    assert Snapshot.from_dict({"decimation": -4}).decimation == 1
    assert Snapshot.from_dict({"decimation": 0}).decimation == 1


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(path)
    s.last = a_snapshot()
    s.save()
    s.save()
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_save_creates_the_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "s.json"
    s = Settings(path)
    s.last = a_snapshot()
    s.save()
    assert path.is_file()


# -- memory management -------------------------------------------------------

def test_adding_the_same_name_replaces_rather_than_duplicates():
    s = Settings()
    assert s.add_memory("40m", a_snapshot(freq_hz=7.1e6)) is False
    assert s.add_memory("40m", a_snapshot(freq_hz=7.2e6)) is True
    assert len(s.memories) == 1
    assert s.get_memory("40m").snapshot.freq_hz == pytest.approx(7.2e6)


def test_name_matching_ignores_case_and_surrounding_space():
    s = Settings()
    s.add_memory("  Radio 4  ", a_snapshot())
    assert s.names() == ["Radio 4"]
    assert s.get_memory("RADIO 4") is not None


def test_blank_name_is_rejected():
    s = Settings()
    for bad in ("", "   ", "\t"):
        with pytest.raises(ValueError):
            s.add_memory(bad, a_snapshot())


def test_remove_memory():
    s = Settings()
    s.add_memory("a", a_snapshot())
    assert s.remove_memory("A") is True
    assert s.names() == []
    assert s.remove_memory("a") is False


def test_describe_mentions_zoom_only_when_zoomed():
    assert Snapshot(freq_hz=7.1e6, decimation=1).describe() == "7.1000 MHz"
    assert Snapshot(freq_hz=7.1e6, decimation=8).describe() == "7.1000 MHz (8x)"


def test_default_path_is_under_application_support():
    s = Settings()
    assert s.path.parent.name == "RGC_SDR"
    assert "Application Support" in str(s.path)
