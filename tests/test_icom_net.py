"""The IC-705's network protocol (device/icom_net.py).

The packet layouts are checked against packets captured from real radios, as quoted in
kappanhang's source (github.com/nonoo/kappanhang). The login and the CI-V and audio
paths are then exercised end to end against a stand-in radio on localhost -- a test
double for the protocol, not a device mode of the app.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest

from src.rgc_sdr.device import civ
from src.rgc_sdr.device import icom_net as net


def hexbytes(text: str) -> bytes:
    return bytes(int(x, 16) for x in text.replace(",", " ").split())


# -- packets ---------------------------------------------------------------------------


def test_passcode_follows_icoms_table():
    assert net.passcode("") == bytes(16)
    assert net.passcode("a")[:1] == b"\x38"                 # 97 -> 0x38
    assert net.passcode("aa")[:2] == b"\x38\x2b"            # second place: 98 -> 0x2b
    assert net.passcode("~~")[1:2] == b"\x47"               # 127 wraps to 32 -> 0x47
    assert len(net.passcode("a-very-long-user-name")) == 16


def test_renewal_packet_matches_one_from_a_radio():
    # kappanhang controlstream.go, "Example request from PC".
    captured = hexbytes("""
        0x40 0x00 0x00 0x00 0x00 0x00 0x0d 0x00 0xbb 0x41 0x3f 0x2b 0xe6 0xb2 0x7b 0x7b
        0x00 0x00 0x00 0x30 0x01 0x05 0x00 0x02 0x00 0x00 0x5d 0x37 0x12 0x82 0x3b 0xde
    """) + bytes(32)
    pkt = net.token_packet(0xBB413F2B, 0xE6B27B7B, 2, 0x05, hexbytes("5d 37 12 82 3b de"))
    struct.pack_into("<H", pkt, 6, 0x0D)
    assert bytes(pkt) == captured


def test_ping_request_and_answer_match_ones_from_a_radio():
    assert net.ping(0xBED9F263, 0xE435DD72, 9, False, hexbytes("78 40 f6 02")) == hexbytes(
        "15 00 00 00 07 00 09 00 be d9 f2 63 e4 35 dd 72 00 78 40 f6 02")
    from_radio = hexbytes("00 00 00 00 07 00 1c 0e e4 35 dd 72 be d9 f2 63 00 57 2b 12 00")
    assert net.is_ping(from_radio)
    assert net.ping(0xBED9F263, 0xE435DD72, 0x0E1C, True, from_radio[17:21]) == hexbytes(
        "15 00 00 00 07 00 1c 0e be d9 f2 63 e4 35 dd 72 01 57 2b 12 00")


def test_header_fields():
    head = net.parse_header(hexbytes("10 00 00 00 04 00 00 00 8c 7d 45 7a 1d f6 e9 0b"))
    assert head == (16, net.T_I_AM_HERE, 0, 0x8C7D457A, 0x1DF6E90B)
    assert net.control(net.T_READY, 0x1DF6E90B, 0x8C7D457A, seq=1) == hexbytes(
        "10 00 00 00 06 00 01 00 1d f6 e9 0b 8c 7d 45 7a")


def test_login_packet_layout():
    p = net.login_packet(1, 2, 0, b"\xaa\xbb", "user", "pass")
    assert len(p) == 0x80 and p[:6] == b"\x80\x00\x00\x00\x00\x00"
    assert p[16:23] == bytes([0, 0, 0, 0x70, 0x01, 0, 0]) and p[26:28] == b"\xaa\xbb"
    assert p[64:80] == net.passcode("user") and p[80:96] == net.passcode("pass")
    assert p[96:104] == b"icom-pc\x00"


def test_connection_request_asks_for_48k_pcm_both_ways():
    p = net.conninfo_packet(1, 2, 3, b"T" * 6, b"G" * 16, b"IC-705", "user")
    assert len(p) == 0x90 and p[26:32] == b"T" * 6 and p[32:48] == b"G" * 16
    assert p[64:72] == b"IC-705\x00\x00" and p[96:112] == net.passcode("user")
    assert p[112:116] == bytes([1, 1, 4, 4])
    assert struct.unpack_from(">IIIII", p, 116) == (48000, 48000, 50002, 50003, 300)


def test_civ_packets_both_ways():
    frame = civ.encode(civ.CMD_READ_FREQ, to=0xA4)
    p = net.civ_packet(1, 2, 0x0102, frame)
    assert p[0] == 0x15 + len(frame) and p[16:21] == bytes([0xC1, len(frame), 0, 1, 2])
    assert net.civ_payload(bytes(p)) == frame
    assert net.civ_payload(net.control(0, 1, 2)) is None
    opening = net.civ_open_packet(1, 2, 0, True)
    assert len(opening) == 0x16 and opening[16:] == bytes([0xC0, 1, 0, 0, 0, 0x05])


def test_transmit_audio_goes_as_two_packets_per_20_ms():
    pcm = bytes(range(256)) * 7 + bytes(128)
    assert len(pcm) == 1920
    first, second = net.audio_packets(1, 2, 5, pcm)
    assert len(first) == 0x18 + 1364 and first[:2] == b"\x6c\x05"
    assert len(second) == 0x18 + 556 and second[:2] == b"\x44\x02"
    assert first[16:24] == bytes([0x80, 0, 0, 4, 0, 0, 0x05, 0x54])
    assert second[18:20] == b"\x00\x05"
    assert net.audio_payload(bytes(first)) + net.audio_payload(bytes(second)) == pcm


def test_pcm_conversion():
    block = np.array([0.0, 0.5, -0.5, -1.0, 2.0], dtype=np.float32)
    back = net.pcm_to_float(net.float_to_pcm(block))
    assert np.allclose(back, [0.0, 0.5, -0.5, -1.0, 32767 / 32768])


def test_radio_packets_from_kappanhang_parse():
    # The 0x90 answer that opens the CI-V and audio ports (controlstream.go).
    answer = hexbytes("""
        0x90 0x00 0x00 0x00 0x00 0x00 0x19 0x00 0xc6 0x5f 0x6f 0x0c 0x5f 0x8b 0x1e 0x89
        0x00 0x00 0x00 0x80 0x03 0x00 0x00 0x00 0x00 0x00 0x31 0x30 0x31 0x47 0x39 0x07
        0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x10 0x80 0x00 0x00 0x90 0xc7 0x0e 0x86 0x01
        0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
        0x49 0x43 0x2d 0x37 0x30 0x35 0x00 0x00
    """) + bytes(24) + hexbytes("01 00 00 00 69 63 6f 6d 2d 70 63 00") + bytes(36)
    assert len(answer) == 0x90 and answer[96] == 1
    assert answer[64:70] == b"IC-705"
    assert struct.unpack_from(">II", answer, 8) == (0xC65F6F0C, 0x5F8B1E89)


# -- a stand-in radio on localhost --------------------------------------------------------


class StandInRadio:
    """Speaks the radio's side of the protocol well enough to log in, answer a frequency
    query over CI-V, send received audio and take transmit audio."""

    RADIO_ID = 0x0A0B0C0D
    TOKEN = b"\x11\x22\x33\x44\x55\x66"
    GUID = bytes(range(16))

    def __init__(self, user="vk3rq", password="secret", wrong_answer=None):
        self.user, self.password = user, password
        self.socks = []
        for _ in range(3):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            s.settimeout(0.02)
            self.socks.append(s)
        self.ports = tuple(s.getsockname()[1] for s in self.socks)
        self.peer = [None, None, None]
        self.client_id = [0, 0, 0]
        self.civ_received = []
        self.audio_received = bytearray()
        self.conninfo = None
        self.pings_answered = 0
        self.goodbyes = 0
        self.freq_hz = 145_650_000
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(1)
        for s in self.socks:
            s.close()

    def send(self, i, data):
        if self.peer[i] is not None:
            self.socks[i].sendto(bytes(data), self.peer[i])

    def _packet(self, i, size, kind=0):
        return net.header(size, kind, 0, self.RADIO_ID, self.client_id[i])

    def send_audio(self, pcm: bytes):
        p = self._packet(2, 24 + len(pcm))
        p[16:24] = bytes([0x97, 0x81, 0, 1, 0, 0]) + struct.pack(">H", len(pcm))
        self.send(2, p + pcm)

    def ping_client(self, i):
        self.send(i, net.ping(self.RADIO_ID, self.client_id[i], 7, False, b"\x01\x02\x03\x04"))

    def _run(self):
        while not self._stop.is_set():
            for i, s in enumerate(self.socks):
                try:
                    data, addr = s.recvfrom(4096)
                except (socket.timeout, OSError):
                    continue
                self.peer[i] = addr
                self._handle(i, data)

    def _handle(self, i, p):
        kind = p[4] if len(p) >= 16 else None
        if len(p) == 16 and kind == net.T_ARE_YOU_THERE:
            self.client_id[i] = struct.unpack_from(">I", p, 8)[0]
            self.send(i, net.control(net.T_I_AM_HERE, self.RADIO_ID, self.client_id[i]))
        elif len(p) == 16 and kind == net.T_READY:
            self.send(i, net.control(net.T_READY, self.RADIO_ID, self.client_id[i], seq=1))
        elif len(p) == 16 and kind == net.T_DISCONNECT:
            self.goodbyes += 1
        elif net.is_ping(p) and p[16] == 1:
            self.pings_answered += 1
        elif i == 0:
            self._control(p)
        elif i == 1:
            data = net.civ_payload(p)
            if data:
                self.civ_received.append(data)
                if data[4] == civ.CMD_READ_FREQ:
                    answer = civ.encode(civ.CMD_READ_FREQ, civ.encode_freq(self.freq_hz),
                                        to=0xE0, frm=0xA4)
                    self.send(1, self._packet(1, 0x15 + len(answer))
                              + bytes([0xC1, len(answer), 0, 0, 1]) + answer)
        elif i == 2:
            pcm = net.audio_payload(p)
            if pcm:
                self.audio_received += pcm

    def _control(self, p):
        if len(p) == 0x80:
            ok = (p[64:80] == net.passcode(self.user) and p[80:96] == net.passcode(self.password))
            reply = self._packet(0, 0x60) + bytes(0x60 - 16)
            reply[26:32] = self.TOKEN
            if not ok:
                reply[48:52] = b"\xff\xff\xff\xfe"
            struct.pack_into("<H", reply, 6, 1)
            self.send(0, reply)
        elif len(p) == 0x40 and p[21] == 0x05:
            ok = self._packet(0, 0x40) + bytes(48)
            ok[21] = 0x05
            self.send(0, ok)
            caps = self._packet(0, 0xA8) + bytes(0xA8 - 16)
            caps[66:82] = self.GUID
            caps[82:88] = b"IC-705"
            self.send(0, caps)
        elif len(p) == 0x90:
            self.conninfo = bytes(p)
            reply = self._packet(0, 0x90) + bytes(0x90 - 16)
            reply[26:32] = self.TOKEN
            reply[64:70] = b"IC-705"
            reply[96] = 1
            self.send(0, reply)


@pytest.fixture
def radio():
    r = StandInRadio()
    yield r
    r.close()


def link_to(radio, password="secret"):
    return net.IcomLink("127.0.0.1", "vk3rq", password, timeout=1.0, ports=radio.ports)


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_logs_in_and_asks_for_the_radios_ports_and_codec(radio):
    link = link_to(radio)
    try:
        assert link.connected and link.radio_name == "IC-705"
        info = radio.conninfo
        assert info[26:32] == radio.TOKEN and info[32:48] == radio.GUID
        assert struct.unpack_from(">II", info, 124) == radio.ports[1:]
    finally:
        link.close()
    assert wait_until(lambda: radio.goodbyes >= 3)          # every session signed off


def test_a_wrong_password_is_reported(radio):
    with pytest.raises(net.LoginError):
        link_to(radio, password="nope")


def test_no_radio_is_reported_quickly():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))                          # a port nobody answers on
    port = s.getsockname()[1]
    start = time.monotonic()
    with pytest.raises(ConnectionError):
        net.IcomLink("127.0.0.1", "u", "p", timeout=0.3, ports=(port, port, port))
    assert time.monotonic() - start < 5
    s.close()


def test_answers_the_radios_pings(radio):
    link = link_to(radio)
    try:
        radio.ping_client(0)
        assert wait_until(lambda: radio.pings_answered >= 1)
    finally:
        link.close()


def test_civ_both_ways_through_icom_source(radio, tmp_path):
    from src.rgc_sdr.device.icom import IcomSource

    link = link_to(radio)
    try:
        src = IcomSource(transport=link, restore_file=tmp_path / "restore.json")
        assert src.center_freq == 145_650_000
        assert src.wlan and src.link is link
        assert src.caps.driver == "icom705net" and src.profile.key == "icom705net"
        assert src.takeover["data_off_mod"] == b"\x03"     # TX audio from WLAN
        src.set_center_freq(7_100_000)
        assert wait_until(lambda: any(f[4] == civ.CMD_SET_FREQ for f in radio.civ_received))
    finally:
        link.close()


def test_received_audio_reaches_the_sink_and_tx_audio_reaches_the_radio(radio):
    link = link_to(radio)
    got = []
    link.audio_sink = got.append
    try:
        radio.send_audio(net.float_to_pcm(np.full(480, 0.25, dtype=np.float32)))
        assert wait_until(lambda: got)
        assert np.allclose(got[0], 0.25, atol=1e-4) and got[0].size == 480
        link.send_audio(np.full(net.TX_BLOCK_SAMPLES, -0.5, dtype=np.float32))
        assert wait_until(lambda: len(radio.audio_received) == 1920)
        assert np.allclose(net.pcm_to_float(bytes(radio.audio_received)), -0.5)
    finally:
        link.close()


def test_network_audio_output_sends_the_shaped_microphone(radio):
    from src.rgc_sdr.audio import NetworkAudioOutput

    class Mic:
        def available(self):
            return 10_000

        def read(self, n):
            return np.full(n, 0.1, dtype=np.float32)

    link = link_to(radio)
    out = NetworkAudioOutput(link, level=0.5)
    try:
        out.start(Mic())
        assert wait_until(lambda: len(radio.audio_received) >= 1920 * 3)
    finally:
        out.stop()
        link.close()
    assert np.allclose(net.pcm_to_float(bytes(radio.audio_received[:1920])), 0.05, atol=1e-3)


def test_login_details_are_saved_without_the_password(tmp_path):
    path = tmp_path / "net.json"
    net.save_login(net.NetworkLogin("192.168.1.50", "vk3rq", "secret"), path, keychain=False)
    assert "secret" not in path.read_text()
    login = net.load_login(path, keychain=False)
    assert (login.host, login.user, login.password) == ("192.168.1.50", "vk3rq", "")
    assert login.complete
    assert not net.load_login(tmp_path / "missing.json", keychain=False).complete
