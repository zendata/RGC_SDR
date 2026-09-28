"""IcomSource's control layer, against a stand-in radio that answers CI-V as the
IC-705 did when probed (PLANNING.md 7o), including refusing AGC changes in FM."""

import pytest

from src.rgc_sdr.device import civ
from src.rgc_sdr.device.icom import IcomSource, s_meter_text


class StandInRadio:
    """Just enough IC-705 to answer the app, with the values read from VK3RQ's radio
    (2026-09-27/28): frequency, mode, levels, switches, meters, VFOs, tones, offset."""

    def __init__(self):
        self.freq = 145_650_000
        self.mode, self.filter = 0x05, 0x01           # FM, FIL1
        self.levels = {0x01: 72, 0x02: 255, 0x03: 85, 0x0A: 3, 0x12: 128, 0x06: 88,
                       0x0D: 129, 0x0E: 151, 0x16: 128, 0x15: 128}
        self.switches = {0x02: 0, 0x12: 1, 0x22: 0, 0x40: 1, 0x41: 0, 0x48: 0, 0x44: 0,
                         0x45: 0, 0x46: 0, 0x47: 1, 0x50: 0, 0x5D: 0}
        self.att = 0x00
        self.smeter = 109
        self.ptt = 0
        self.po, self.swr = 143, 48              # 50 % power, SWR 1.5
        self.meters = {0x13: 60, 0x14: 130, 0x15: 189, 0x16: 121}
        self.squelch = 1                          # open
        self.data_off_mod = 0x02                  # MIC,USB, as found on VK3RQ's radio
        self.menu = {(0x01, 0x10): bytes.fromhex("0128"), (0x01, 0x11): b"\x00"}
        self.span = None
        self.split = 0x01
        self.rit = {0x01: 0, 0x02: 0}
        self.rit_hz = -1200
        self.parser = civ.FrameParser()
        self.out = bytearray()
        self.written = []

    def _reply(self, cmd, payload=b""):
        self.out += civ.encode(cmd, payload, to=civ.CONTROLLER_ADDRESS, frm=civ.IC705_ADDRESS)

    def write(self, data):
        for f in self.parser.feed(data):
            self.written.append(f)
            self._answer(f.cmd, f.payload)
        return len(data)

    def _answer(self, cmd, p):
        ng = lambda: self._reply(civ.NG)
        if cmd == 0x03:
            self._reply(0x03, civ.encode_freq(self.freq))
        elif cmd == 0x05:
            self.freq = civ.decode_freq(p); self._reply(civ.OK)
        elif cmd == 0x04:
            self._reply(0x04, bytes([self.mode, self.filter]))
        elif cmd == 0x06:
            self.mode, self.filter = p[0], p[1]; self._reply(civ.OK)
        elif cmd == 0x14:
            if p[0] not in self.levels:
                return ng()                                  # e.g. twin PBT in FM/WFM
            if len(p) == 1:
                self._reply(0x14, bytes([p[0]]) + civ.encode_level(self.levels[p[0]]))
            else:
                self.levels[p[0]] = civ.decode_level(p[1:3]); self._reply(civ.OK)
        elif cmd == 0x16:
            if len(p) == 1:
                self._reply(0x16, bytes([p[0], self.switches[p[0]]]))
            elif p[0] == 0x12 and self.mode == 0x05:         # AGC is fixed in FM
                ng()
            else:
                self.switches[p[0]] = p[1]; self._reply(civ.OK)
        elif cmd == 0x11:
            if p:
                self.att = p[0]; self._reply(civ.OK)
            else:
                self._reply(0x11, bytes([self.att]))
        elif cmd == 0x15:
            if p[0] == 0x01:
                self._reply(0x15, bytes([0x01, self.squelch]))
            elif p[0] == 0x07:
                self._reply(0x15, b"\x07\x00")
            else:
                value = {0x02: self.smeter, 0x11: self.po, 0x12: self.swr, **self.meters}[p[0]]
                self._reply(0x15, bytes([p[0]]) + civ.encode_level(value))
        elif cmd == 0x1C and len(p) == 2:
            self.ptt = p[1]; self._reply(civ.OK)
        elif cmd == 0x0F:
            if not p:
                self._reply(0x0F, bytes([self.split]))
            else:
                self.split = {0x00: 0x00, 0x10: 0x00}.get(p[0], p[0]); self._reply(civ.OK)
        elif cmd == 0x21:
            if p[0] == 0x00:
                digits = civ.encode_freq(abs(self.rit_hz), nbytes=2)
                self._reply(0x21, b"\x00" + digits + (b"\x01" if self.rit_hz < 0 else b"\x00"))
            elif len(p) == 1:
                self._reply(0x21, bytes([p[0], self.rit[p[0]]]))
            else:
                self.rit[p[0]] = p[1]; self._reply(civ.OK)
        elif cmd == 0x25:
            hz = 101_900_000 if p[0] == 0 else 437_225_000
            self._reply(0x25, bytes([p[0]]) + civ.encode_freq(hz))
        elif cmd == 0x26:
            self._reply(0x26, bytes([p[0], 0x06 if p[0] == 0 else 0x05, 0x00, 0x01]))
        elif cmd == 0x0C:
            self._reply(0x0C, bytes.fromhex("006000"))       # 600 kHz, as read
        elif cmd == 0x1B:
            self._reply(0x1B, bytes([p[0]]) + (bytes.fromhex("000023") if p[0] == 2
                                               else bytes.fromhex("000885")))
        elif cmd == 0x1A and p[0] == 0x05 and (p[1], p[2]) in self.menu:
            if len(p) == 3:
                self._reply(0x1A, p + self.menu[(p[1], p[2])])
            else:
                self.menu[(p[1], p[2])] = bytes(p[3:]); self._reply(civ.OK)
        elif cmd == 0x1A and p[:3] == bytes([0x05, 0x01, 0x18]):
            if len(p) == 3:
                self._reply(0x1A, p + bytes([self.data_off_mod]))
            else:
                self.data_off_mod = p[3]; self._reply(civ.OK)
        elif cmd == 0x27 and p[:2] == bytes([0x15, 0x00]) and len(p) == 7:
            self.span = civ.decode_freq(p[2:7]); self._reply(civ.OK)
        elif cmd == 0x27:
            self._reply(civ.OK if len(p) > 1 else 0x27, p if len(p) == 1 else b"")

    def read(self, n):
        data, self.out = bytes(self.out[:n]), self.out[n:]
        return data

    def close(self):
        pass


def radio_and_source():
    radio = StandInRadio()
    src = IcomSource(transport=radio)
    return radio, src


def settle(src, rounds=5):
    for _ in range(rounds):
        src._pump_once()


def test_opens_where_the_radio_is_tuned():
    radio, src = radio_and_source()
    assert src.center_freq == 145_650_000


def test_poll_reads_every_setting_mode_and_the_meter():
    radio, src = radio_and_source()
    src.poll(now=0.0)
    settle(src)
    assert src.mode == "fm" and src.filter == 1
    expected = {"af": 72, "rf": 255, "sql": 85, "power": 3, "preamp": 0,
                "agc": 1, "nb": 0, "nr": 1, "att": 0, "bkin": 1, "split": 1,
                "dup": 0x10, "notch": 0, "comp_level": 151, "notch_pos": 129}
    assert {k: src.state.get(k) for k in expected} == expected
    assert src.smeter == 109


def test_setting_a_level_reaches_the_radio():
    radio, src = radio_and_source()
    src.set_control("rf", 200)
    settle(src)
    assert radio.levels[0x02] == 200 and src.state["rf"] == 200 and src.refused is None


def test_attenuator_is_20_db_or_off():
    radio, src = radio_and_source()
    src.set_control("att", 1)
    settle(src)
    assert radio.att == 0x20


def test_a_refused_setting_is_reported_and_read_back():
    """Measured: the 705 answers FA to AGC MID in FM, and keeps FAST."""
    radio, src = radio_and_source()
    src.set_control("agc", 2)
    settle(src)
    assert src.refused == "agc"
    assert src.state["agc"] == 1                     # the radio's real value, not ours


def test_refusals_are_matched_to_the_right_write():
    radio, src = radio_and_source()
    src.set_control("rf", 100)                        # OK
    src.set_control("agc", 3)                         # NG in FM
    src.set_control("nb", 1)                          # OK
    settle(src)
    assert src.refused == "agc"
    assert radio.levels[0x02] == 100 and radio.switches[0x22] == 1


def test_mode_and_filter():
    radio, src = radio_and_source()
    src.set_mode("usb", 2)
    settle(src)
    assert (radio.mode, radio.filter) == (0x01, 0x02)
    src.set_mode("lsb")                               # filter kept
    settle(src)
    assert (radio.mode, radio.filter) == (0x00, 0x02)


def test_tuning_writes_bcd_frequency():
    radio, src = radio_and_source()
    src.set_center_freq(146.5e6)
    settle(src)
    assert radio.freq == 146_500_000


def test_front_panel_changes_are_followed():
    """CI-V Transceive: the radio reports its own dial and mode changes unasked."""
    radio, src = radio_and_source()
    radio.out += civ.encode(0x00, civ.encode_freq(7_100_000), to=0x00, frm=civ.IC705_ADDRESS)
    radio.out += civ.encode(0x01, bytes([0x00, 0x03]), to=0x00, frm=civ.IC705_ADDRESS)
    settle(src)
    assert src.center_freq == 7_100_000 and src.mode == "lsb" and src.filter == 3


@pytest.mark.parametrize("raw,text", [(0, "S0"), (109, "S8"), (120, "S9"),
                                      (160, "S9+20 dB"), (241, "S9+60 dB")])
def test_s_meter_text(raw, text):
    assert s_meter_text(raw) == text


# -- transmit ------------------------------------------------------------------------

from src.rgc_sdr.device.icom import power_percent, swr_value  # noqa: E402


def test_ptt_keys_and_unkeys_the_radio():
    radio, src = radio_and_source()
    src.set_ptt(True)
    settle(src)
    assert radio.ptt == 1 and src.transmitting and src.refused is None
    src.set_ptt(False)
    settle(src)
    assert radio.ptt == 0 and not src.transmitting


def test_meters_while_transmitting_are_power_and_swr():
    radio, src = radio_and_source()
    src.set_ptt(True)
    src.poll(now=0.0)
    settle(src)
    assert power_percent(src.po) == pytest.approx(50.0)
    assert swr_value(src.swr) == pytest.approx(1.5)
    polled = [f.payload[0] for f in radio.written if f.cmd == 0x15]
    assert 0x11 in polled and 0x12 in polled and 0x02 not in polled


def test_closing_never_leaves_the_radio_keyed():
    radio, src = radio_and_source()
    src.set_ptt(True)
    settle(src)
    src.close()
    assert radio.ptt == 0


@pytest.mark.parametrize("raw,percent", [(0, 0.0), (143, 50.0), (213, 100.0), (178, 75.0)])
def test_power_meter_scale(raw, percent):
    assert power_percent(raw) == pytest.approx(percent)


@pytest.mark.parametrize("raw,swr", [(0, 1.0), (48, 1.5), (80, 2.0), (120, 3.0), (100, 2.5)])
def test_swr_meter_scale(raw, swr):
    assert swr_value(raw) == pytest.approx(swr)



# -- the radio's squelch, which the 705 does not apply to its USB audio ---------------

def test_squelch_state_is_followed():
    radio, src = radio_and_source()
    src.poll(now=0.0)
    settle(src)
    assert src.squelch_open is True
    radio.squelch = 0
    src.poll(now=1.0)
    settle(src)
    assert src.squelch_open is False


def test_squelch_is_polled_often_but_not_while_transmitting():
    radio, src = radio_and_source()
    for i in range(10):
        src.poll(now=i * 0.1)
    asked = sum(1 for f in radio.written if f.cmd == 0x15 and f.payload[:1] == b"\x01")
    assert asked == 10
    src.set_ptt(True)
    radio.written.clear()
    src.poll(now=5.0)
    assert not any(f.cmd == 0x15 and f.payload[:1] == b"\x01" for f in radio.written)



# -- taking over the speaker and TX audio source, and handing them back ---------------

import json  # noqa: E402

from src.rgc_sdr.device.icom import IcomSource as _Source  # noqa: E402


def source_with_file(radio, tmp_path):
    return _Source(transport=radio, restore_file=tmp_path / "restore.json")


def test_the_app_silences_the_speaker_and_takes_tx_audio_from_usb(tmp_path):
    radio = StandInRadio()
    src = source_with_file(radio, tmp_path)
    src._take_over()
    settle(src)
    assert radio.levels[0x01] == 0 and radio.data_off_mod == 0x01
    # USB audio at full level, muted by the radio itself when its squelch closes.
    assert radio.menu[(0x01, 0x10)] == bytes.fromhex("0255")
    assert radio.menu[(0x01, 0x11)] == b"\x01"
    saved = json.loads((tmp_path / "restore.json").read_text())
    assert saved == {"af": "0072", "data_off_mod": "02", "usb_af_level": "0128",
                     "usb_af_sql": "00"}


def test_handing_back_restores_the_radios_own_settings(tmp_path):
    radio = StandInRadio()
    src = source_with_file(radio, tmp_path)
    src._take_over(); settle(src)
    src._hand_back(); settle(src)
    assert radio.levels[0x01] == 72 and radio.data_off_mod == 0x02
    assert radio.menu == {(0x01, 0x10): bytes.fromhex("0128"), (0x01, 0x11): b"\x00"}
    assert not (tmp_path / "restore.json").exists()


def test_after_a_crash_the_real_originals_come_back(tmp_path):
    """The radio still has the app's values (AF 0, USB); the file has the truth."""
    (tmp_path / "restore.json").write_text(json.dumps({"af": "0072", "data_off_mod": "02"}))
    radio = StandInRadio()
    radio.levels[0x01], radio.data_off_mod = 0, 0x01
    src = source_with_file(radio, tmp_path)
    src._take_over(); settle(src)
    src._hand_back(); settle(src)
    assert radio.levels[0x01] == 72 and radio.data_off_mod == 0x02


def test_span_is_sent_as_a_bcd_half_span():
    radio, src = radio_and_source()
    src.set_span(100e3)
    settle(src)
    assert radio.span == 100_000



# -- the radio's display: VFOs, RIT, offset, tones, meters -----------------------------

from src.rgc_sdr.device.icom import comp_db, drain_amps, supply_volts  # noqa: E402


def test_display_state_decodes_as_read_from_the_radio():
    radio, src = radio_and_source()
    src.poll(now=0.0)
    settle(src, rounds=10)
    st = src.status
    assert st["vfo_sel_hz"] == 101_900_000 and st["vfo_other_hz"] == 437_225_000
    assert st["vfo_sel_mode"] == "wfm" and st["vfo_other_mode"] == "fm"
    assert st["vfo_sel_filter"] == 1 and st["vfo_sel_data"] is False
    assert st["offset_hz"] == 600_000                  # 00 60 00, in 100 Hz units
    assert st["tone_hz"] == pytest.approx(88.5) and st["tsql_hz"] == pytest.approx(88.5)
    assert st["dtcs"] == "023"
    assert st["rit_hz"] == -1200
    assert supply_volts(st["vd"]) == pytest.approx(12.55, abs=0.05)


def test_transmit_meters_include_alc_comp_and_current():
    radio, src = radio_and_source()
    src.set_ptt(True)
    src.poll(now=0.0)
    settle(src, rounds=10)
    assert comp_db(src.status["comp_meter"]) == pytest.approx(15.0)
    assert drain_amps(src.status["id"]) == pytest.approx(2.0)
    assert src.status["alc"] == 60


def test_split_and_duplex_are_separate_keys_on_one_command():
    """0F carries both (Basic Manual p. 2-6 has SPLIT and DUP keys): one read sets both
    keys, and turning one on turns the other off."""
    radio, src = radio_and_source()
    src.set_control("dup", 0x11)                       # DUP-
    settle(src)
    assert radio.split == 0x11
    src._read_control("split"); settle(src)
    assert src.state["dup"] == 0x11 and src.state["split"] == 0
    src.set_control("split", 1)
    settle(src)
    assert radio.split == 0x01
    src._read_control("split"); settle(src)
    assert src.state["dup"] == 0x10 and src.state["split"] == 1
    src.set_control("split", 0)
    settle(src)
    assert radio.split == 0x00


@pytest.mark.parametrize("value,auto,manual", [(1, 1, 0), (2, 0, 1), (0, 0, 0)])
def test_notch_is_one_key_off_auto_manual(value, auto, manual):
    radio, src = radio_and_source()
    radio.switches[0x41], radio.switches[0x48] = 1, 1 - auto   # something else on first
    src.set_control("notch", value)
    settle(src)
    assert (radio.switches[0x41], radio.switches[0x48]) == (auto, manual)
    src._read_control("notch"); settle(src)
    assert src.state["notch"] == value


def test_a_refused_read_is_not_blamed_on_a_write():
    """Twin PBT reads come back FA in FM/WFM (measured). That must not mark a
    change made in the same batch as refused, nor shift the matching after it."""
    radio, src = radio_and_source()
    src._send(0x14, b"\x07")                          # PBT1 read: FA
    src.set_control("rf", 100)                         # a write: FB
    src.set_control("agc", 2)                          # a write refused in FM: FA
    settle(src)
    assert src.refused == "agc"
    assert src._pending == [] or all(not w for _c, _l, w in src._pending)


@pytest.mark.parametrize("key,value,cmd,payload", [
    ("comp", 1, 0x16, b"\x44\x01"), ("bkin", 2, 0x16, b"\x47\x02"),
    ("tone", 2, 0x16, b"\x5d\x02"), ("rit", 1, 0x21, b"\x01\x01"),
    ("notch_pos", 200, 0x14, b"\x0d\x02\x00"), ("anti_vox", 255, 0x14, b"\x17\x02\x55"),
    ("bkin_delay", 0, 0x14, b"\x0f\x00\x00"), ("tone", 9, 0x16, b"\x5d\x09"),
    ("dup", 0x12, 0x0F, b"\x12")])
def test_function_controls_write_the_documented_commands(key, value, cmd, payload):
    radio, src = radio_and_source()
    src.set_control(key, value)
    written = radio.written[-1]
    assert (written.cmd, written.payload) == (cmd, payload)



@pytest.mark.parametrize("hz,payload", [(120, b"\x00\x20\x01\x00"),
                                        (-1234, b"\x00\x34\x12\x01"),
                                        (20000, b"\x00\x99\x99\x00")])
def test_rit_offset_is_written_as_the_radio_takes_it(hz, payload):
    """Formats checked on the radio 2026-09-28; the limit is +-9.999 kHz."""
    radio, src = radio_and_source()
    src.set_rit(hz)
    assert (radio.written[-1].cmd, radio.written[-1].payload) == (0x21, payload)
