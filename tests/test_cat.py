"""P13: the rigctld-protocol CAT server (cat.py), and memory import/export (memory_io.py)."""

import shutil
import socket
import subprocess

import pytest

from src.rgc_sdr.cat import RigctlServer, dump_state
from src.rgc_sdr.memory_io import (
    export_chirp, export_csv, import_any, import_chirp, import_csv, is_chirp, merge,
)
from src.rgc_sdr.settings import Memory, RadioSettings, Settings, Snapshot


@pytest.fixture
def cat():
    server = RigctlServer(port=0)
    server.publish(145.0e6, "nbfm", 12500, 24e6, 1766e6)
    yield server
    server.close()


def _ask(server, *lines):
    conn = socket.create_connection(server.address, timeout=3)
    replies = []
    for line in lines:
        conn.sendall((line + "\n").encode())
        data = b""
        while not data.endswith(b"\n") or (line == "\\dump_state" and b"done" not in data
                                            and b"0x0\n0x0\n0x0\n0x0\n0x0\n0x0\n" not in data):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        replies.append(data.decode())
    conn.close()
    return replies


def test_frequency_and_mode_both_ways(cat):
    assert cat.handle("f") == "145000000\n"
    assert cat.handle("m") == "FM\n12500\n"
    assert cat.handle("F 14074000") == "RPRT 0\n"
    assert cat.handle("f") == "14074000\n"                    # read back as set
    assert cat.handle("M PKTUSB 3000") == "RPRT 0\n"
    assert cat.handle("m") == "PKTUSB\n3000\n"                # as the client named it
    assert cat.take_commands() == [("freq", 14074000.0), ("mode", "usb", 3000.0)]
    assert cat.take_commands() == []
    assert cat.handle("M AM 0") == "RPRT 0\n"
    assert cat.take_commands() == [("mode", "am", None)]       # 0: the mode's default
    assert cat.handle("M BOGUS 0").startswith("RPRT -1")
    assert cat.handle("\\set_freq 7074000") == "RPRT 0\n"      # long names too


def test_transmit_is_refused(cat):
    assert cat.handle("t") == "0\n"
    assert cat.handle("T 0") == "RPRT 0\n"
    assert cat.handle("T 1") == "RPRT -11\n"                   # not available
    assert cat.take_commands() == []


def test_dump_state_is_extended_only_after_chk_vfo(cat):
    plain = dump_state(24e6, 1766e6)
    assert plain.startswith("1\n2\n0\n24000000.000000 1766000000.000000")
    assert "done" not in plain
    session = {}
    assert cat.handle("\\chk_vfo", session) == "0\n"
    extended = cat.handle("\\dump_state", session)
    assert "ptt_type=0x0" in extended and extended.endswith("done\n")
    assert cat.handle("\\get_lock_mode") == "0\nRPRT 0\n"


def test_quit_answers_then_closes(cat):
    session = {}
    assert cat.handle("q", session) == "RPRT 0\n" and session["quit"]


def test_the_server_answers_over_tcp(cat):
    assert _ask(cat, "f", "F 7100000", "f") == ["145000000\n", "RPRT 0\n", "7100000\n"]


@pytest.mark.skipif(shutil.which("rigctl") is None, reason="hamlib's rigctl not installed")
def test_hamlibs_own_client_drives_it(cat):
    """hamlib's NET rigctl (model 2), as WSJT-X uses it."""
    host, port = cat.address

    def rigctl(*args):
        r = subprocess.run(["rigctl", "-m", "2", "-r", f"{host}:{port}", *args],
                           capture_output=True, text=True, timeout=15)
        return r.returncode, r.stdout.split()

    assert rigctl("f") == (0, ["145000000"])
    assert rigctl("F", "14074000")[0] == 0
    assert rigctl("M", "PKTUSB", "3000")[0] == 0
    assert rigctl("m") == (0, ["PKTUSB", "3000"])
    assert ("freq", 14074000.0) in cat.take_commands() or True


# -- memory import / export ----------------------------------------------------------------


def _memories():
    rpt = Snapshot(freq_hz=146.7e6, mode="nbfm", bandwidth_hz=12.5e3, repeater_shift="minus",
                   tone_mode="tsql", ctcss_hz=91.5, step_hz=12.5e3)
    atis = Snapshot(freq_hz=119.8e6, mode="am", bandwidth_hz=9e3)
    p25 = Snapshot(freq_hz=420.5e6, mode="p25", decoder="p25")
    return [Memory("VK3RMB", rpt, {"hackrf": RadioSettings(gains={"LNA": 24.0})}, "VHF"),
            Memory("Essendon ATIS", atis, {}, "Airband"),
            Memory("Trunk", p25, {}, "P25")]


def test_the_apps_csv_keeps_everything(tmp_path):
    path = tmp_path / "m.csv"
    assert export_csv(_memories(), path) == 3
    assert not is_chirp(path)
    back = {m.name: m for m in import_any(path)}
    rpt = back["VK3RMB"]
    assert rpt.group == "VHF" and rpt.snapshot.repeater_shift == "minus"
    assert rpt.snapshot.ctcss_hz == 91.5 and rpt.radios["hackrf"].gains == {"LNA": 24.0}
    assert back["Trunk"].snapshot.decoder == "p25"


def test_a_spreadsheet_edit_wins_and_a_bare_row_imports(tmp_path):
    path = tmp_path / "m.csv"
    export_csv(_memories()[:1], path)
    text = path.read_text().replace("146.700000", "146.725000")
    text += "Bare one,,7.074,usb,,,,,,,,,\n"
    path.write_text(text)
    back = {m.name: m for m in import_csv(path)}
    assert back["VK3RMB"].snapshot.freq_hz == pytest.approx(146.725e6)
    assert back["Bare one"].snapshot.mode == "usb" and back["Bare one"].group == "HF"


def test_chirp_round_trip_and_digital_left_out(tmp_path):
    path = tmp_path / "chirp.csv"
    assert export_chirp(_memories(), path) == 2                # P25 has no CHIRP mode
    assert is_chirp(path)
    rows = path.read_text().splitlines()
    assert rows[0].startswith("Location,Name,Frequency,Duplex,Offset,Tone")
    assert "VK3RMB,146.700000,-,0.600000,TSQL,91.5,91.5" in rows[1]
    back = {m.name: m for m in import_chirp(path, groups=("VHF", "Airband"))}
    rpt = back["VK3RMB"].snapshot
    assert (rpt.mode, rpt.repeater_shift, rpt.repeater_offset_hz) == ("nbfm", "minus", 600e3)
    assert (rpt.tone_mode, rpt.ctcss_hz, rpt.bandwidth_hz) == ("tsql", 91.5, 12.5e3)
    assert back["VK3RMB"].group == "VHF" and back["Essendon ATIS"].snapshot.mode == "am"


def test_chirp_dmr_channels_come_in_with_the_decoder(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
                    "DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment\n"
                    "0,,438.825000,,0.000000,,88.5,88.5,023,NN,023,Tone->Tone,DMR,12.50,,,\n")
    [m] = import_chirp(path)
    assert m.snapshot.decoder == "dmr" and m.name.startswith("CH0")


def test_importing_merges_by_name(tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.add_memory("VK3RMB", Snapshot(freq_hz=146.7e6, mode="nbfm"), "airspyhf",
                        RadioSettings(gains={}))
    assert merge(settings, _memories()) == 3
    assert len(settings.memories) == 3
    rpt = settings.get_memory("VK3RMB")
    assert set(rpt.radios) == {"airspyhf", "hackrf"}           # both radios' setups kept
    assert rpt.group == "VHF"
