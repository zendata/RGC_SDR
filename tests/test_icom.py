"""IcomSource's control layer, against a stand-in radio that answers CI-V as the
IC-705 did when probed (PLANNING.md 7o), including refusing AGC changes in FM."""

import pytest

from src.rgc_sdr.device import civ
from src.rgc_sdr.device.icom import IcomSource, s_meter_text


class StandInRadio:
    """Just enough IC-705 to answer the app: frequency, mode, settings, meter."""

    def __init__(self):
        self.freq = 145_650_000
        self.mode, self.filter = 0x05, 0x01           # FM, FIL1
        self.levels = {0x01: 0, 0x02: 255, 0x03: 85, 0x0A: 3}
        self.switches = {0x02: 0, 0x12: 1, 0x22: 0, 0x40: 1}
        self.att = 0x00
        self.smeter = 109
        self.ptt = 0
        self.squelch = 1                          # open
        self.po, self.swr = 143, 48              # 50 % power, SWR 1.5
        self.parser = civ.FrameParser()
        self.out = bytearray()
        self.written = []

    def _reply(self, cmd, payload=b""):
        self.out += civ.encode(cmd, payload, to=civ.CONTROLLER_ADDRESS, frm=civ.IC705_ADDRESS)

    def write(self, data):
        for f in self.parser.feed(data):
            self.written.append(f)
            cmd, p = f.cmd, f.payload
            if cmd == 0x03:
                self._reply(0x03, civ.encode_freq(self.freq))
            elif cmd == 0x05:
                self.freq = civ.decode_freq(p); self._reply(civ.OK)
            elif cmd == 0x04:
                self._reply(0x04, bytes([self.mode, self.filter]))
            elif cmd == 0x06:
                self.mode, self.filter = p[0], p[1]; self._reply(civ.OK)
            elif cmd == 0x14 and len(p) == 1:
                self._reply(0x14, bytes([p[0]]) + civ.encode_level(self.levels[p[0]]))
            elif cmd == 0x14:
                self.levels[p[0]] = civ.decode_level(p[1:3]); self._reply(civ.OK)
            elif cmd == 0x16 and len(p) == 1:
                self._reply(0x16, bytes([p[0], self.switches[p[0]]]))
            elif cmd == 0x16:
                if p[0] == 0x12 and self.mode == 0x05:        # AGC is fixed in FM
                    self._reply(civ.NG)
                else:
                    self.switches[p[0]] = p[1]; self._reply(civ.OK)
            elif cmd == 0x11 and not p:
                self._reply(0x11, bytes([self.att]))
            elif cmd == 0x11:
                self.att = p[0]; self._reply(civ.OK)
            elif cmd == 0x15 and p[0] == 0x01:        # squelch status: one byte, as measured
                self._reply(0x15, bytes([0x01, self.squelch]))
            elif cmd == 0x15:
                value = {0x02: self.smeter, 0x11: self.po, 0x12: self.swr}[p[0]]
                self._reply(0x15, bytes([p[0]]) + civ.encode_level(value))
            elif cmd == 0x1C and len(p) == 2:
                self.ptt = p[1]; self._reply(civ.OK)
            elif cmd == 0x27:
                self._reply(civ.OK if len(p) > 1 else 0x27, p if len(p) == 1 else b"")
        return len(data)

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
    assert src.state == {"af": 0, "rf": 255, "sql": 85, "power": 3, "preamp": 0,
                         "agc": 1, "nb": 0, "nr": 1, "att": 0}
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
