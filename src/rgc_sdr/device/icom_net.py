"""The IC-705 over WiFi: Icom's network remote protocol, the one its RS-BA1 software uses.

Icom does not publish it. The packet layouts here follow the two open implementations,
kappanhang (github.com/nonoo/kappanhang, written for the IC-705) and wfview, and the
packets they quote from real radios are the tests' reference (tests/test_icom_net.py).

Three UDP sessions, one per port on the radio (SET > WLAN Set > Remote Settings):

* control, 50001 -- login with the radio's Network User ID and password, then a request
  for the other two; kept alive with pings and a re-authorisation every minute;
* CI-V, 50002 -- the same CI-V bytes as on the USB cable, a frame or two to a packet;
* audio, 50003 -- received audio from the radio, transmit audio to it: 16-bit PCM,
  mono, 48 kHz, 20 ms a packet pair.

Every session opens the same way ("are you there" -> "I am here" with the radio's id ->
"are you ready" -> "ready") and every packet starts with the same 16-byte header: length,
type, sequence (little-endian), then the sender's and receiver's ids. Packets with a
sequence number are kept a while, because the radio may ask for one again.

`IcomLink` holds all three and hands the CI-V side to `IcomSource` as its transport --
anything with read/write/close, as the serial port is -- and the audio to
`audio.NetworkRadioAudio` and `audio.NetworkAudioOutput`.
"""

from __future__ import annotations

import json
import os
import select
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTROL_PORT = 50001
SERIAL_PORT = 50002
AUDIO_PORT = 50003

#: Received and transmitted audio: 16-bit little-endian PCM, one channel.
AUDIO_RATE = 48_000
#: Codec 0x04 is "LPCM 1 ch 16 bit" in the connection request (both directions).
AUDIO_CODEC = 0x04
#: Transmit audio goes in 20 ms blocks, each sent as two packets of these sizes.
TX_BLOCK_SAMPLES = 960
_TX_PARTS = (1364, 556)
#: The jitter buffer the radio is asked to keep for our transmit audio, ms.
TX_BUFFER_MS = 300

#: Packet types (the 16-bit field after the length).
T_DATA, T_RETRANSMIT, T_ARE_YOU_THERE, T_I_AM_HERE, T_DISCONNECT, T_READY, T_PING = \
    0, 1, 3, 4, 5, 6, 7

#: How often to ping, to re-authorise, and how long the radio may stay silent.
PING_S = 3.0
REAUTH_S = 60.0
SILENT_S = 5.0
#: Idle packets keep the sequence moving: often just after real traffic, then slowly.
IDLE_FAST_S, IDLE_SLOW_S, IDLE_FAST_FOR_S = 0.1, 1.0, 1.0
#: How long to wait for each step of the login.
STEP_TIMEOUT_S = 2.0

#: The client name sent at login; RS-BA1 sends this.
CLIENT_NAME = b"icom-pc"


# -- packets (pure functions, tested against packets from real radios) --------------------

# Icom's substitution table for the user name and password in the login packet, indexed
# by (character code + position); kappanhang's passcode.go.
_PASSCODE = bytes.fromhex(
    "475d4c4266202346"  # 32-39
    "4e57453d67766041"  # 40-47
    "6239592d687e7c65"  # 48-55
    "7d4929727378216e"  # 56-63
    "5a5e4a3e712c2a54"  # 64-71
    "3c3a634f43752779"  # 72-79
    "5b3570486b566f34"  # 80-87
    "326c30616d7b2f4b"  # 88-95
    "64382b2e50403f55"  # 96-103
    "333725772426746a"  # 104-111
    "28534d69225c4431"  # 112-119
    "36583b7a515f52"    # 120-126
)


def passcode(text: str) -> bytes:
    """A user name or password as the login packet carries it: 16 bytes, obscured."""
    out = bytearray(16)
    for i, ch in enumerate(text.encode("ascii", "replace")[:16]):
        p = ch + i
        if p > 126:
            p = 32 + p % 127
        if 32 <= p <= 126:
            out[i] = _PASSCODE[p - 32]
    return bytes(out)


def header(length: int, kind: int, seq: int, local_id: int, remote_id: int) -> bytearray:
    """The 16 bytes every packet starts with. The ids go big-endian, as the radio's own
    packets carry them (it echoes ours back); the rest is little-endian."""
    return bytearray(struct.pack("<IHH", length, kind, seq & 0xFFFF)
                     + struct.pack(">II", local_id, remote_id))


def control(kind: int, local_id: int, remote_id: int, seq: int = 0) -> bytes:
    """A bare 16-byte packet: are you there, ready, disconnect, idle, retransmit."""
    return bytes(header(16, kind, seq, local_id, remote_id))


def ping(local_id: int, remote_id: int, seq: int, reply: bool, token: bytes) -> bytes:
    """A 21-byte ping: a request (reply False) or the answer to the radio's, which
    carries its 4-byte token back."""
    return bytes(header(21, T_PING, seq, local_id, remote_id)) + bytes([int(reply)]) + token[:4]


def parse_header(pkt: bytes) -> tuple[int, int, int, int, int] | None:
    """(length, type, seq, sender id, receiver id), or None if too short."""
    if len(pkt) < 16:
        return None
    length, kind, seq = struct.unpack_from("<IHH", pkt)
    sender, receiver = struct.unpack_from(">II", pkt, 8)
    return length, kind, seq, sender, receiver


def is_ping(pkt: bytes) -> bool:
    # The radio's own pings start 00 rather than 15: the length is not checked.
    return len(pkt) == 21 and pkt[1:6] == b"\x00\x00\x00\x07\x00"


def login_packet(local_id: int, remote_id: int, inner_seq: int, token_request: bytes,
                 user: str, password: str) -> bytearray:
    p = header(0x80, T_DATA, 0, local_id, remote_id)
    p += bytes([0x00, 0x00, 0x00, 0x70, 0x01, 0x00, 0x00]) + struct.pack("<H", inner_seq)
    p += b"\x00" + token_request[:2] + bytes(36)
    p += passcode(user) + passcode(password)
    p += CLIENT_NAME.ljust(32, b"\x00")
    assert len(p) == 0x80
    return p


def token_packet(local_id: int, remote_id: int, inner_seq: int, magic: int,
                 token: bytes) -> bytearray:
    """Re-authorisation with the radio's token: magic 02 after login, 05 to renew (every
    minute), 01 to sign off."""
    p = header(0x40, T_DATA, 0, local_id, remote_id)
    p += bytes([0x00, 0x00, 0x00, 0x30, 0x01, magic, 0x00]) + struct.pack("<H", inner_seq)
    p += b"\x00" + token[:6] + bytes(32)
    assert len(p) == 0x40
    return p


def conninfo_packet(local_id: int, remote_id: int, inner_seq: int, token: bytes,
                    radio_guid: bytes, radio_name: bytes, user: str,
                    ports: tuple[int, int] = (SERIAL_PORT, AUDIO_PORT)) -> bytearray:
    """Ask for the CI-V and audio sessions: which ports, which codec, which rate."""
    p = header(0x90, T_DATA, 0, local_id, remote_id)
    p += bytes([0x00, 0x00, 0x00, 0x80, 0x01, 0x03, 0x00]) + struct.pack("<H", inner_seq)
    p += b"\x00" + token[:6] + radio_guid[:16].ljust(16, b"\x00") + bytes(16)
    p += radio_name[:32].ljust(32, b"\x00")
    p += passcode(user)
    p += bytes([0x01, 0x01, AUDIO_CODEC, AUDIO_CODEC])            # RX on, TX on, codecs
    p += struct.pack(">II", AUDIO_RATE, AUDIO_RATE)
    p += struct.pack(">II", *ports)
    p += struct.pack(">I", TX_BUFFER_MS) + bytes([0x01]) + bytes(7)
    assert len(p) == 0x90
    return p


def civ_packet(local_id: int, remote_id: int, civ_seq: int, data: bytes) -> bytearray:
    """CI-V bytes for the radio: C1, the length, then a sequence of its own (big-endian)."""
    p = header(0x15 + len(data), T_DATA, 0, local_id, remote_id)
    p += bytes([0xC1, len(data), 0x00]) + struct.pack(">H", civ_seq & 0xFFFF)
    return p + data


def civ_open_packet(local_id: int, remote_id: int, civ_seq: int, open_: bool) -> bytearray:
    p = header(0x16, T_DATA, 0, local_id, remote_id)
    p += bytes([0xC0, 0x01, 0x00]) + struct.pack(">H", civ_seq & 0xFFFF)
    return p + bytes([0x05 if open_ else 0x00])


def civ_payload(pkt: bytes) -> bytes | None:
    """The CI-V bytes in a packet from the radio, or None if it is not a CI-V packet."""
    if (len(pkt) >= 22 and pkt[16] == 0xC1
            and struct.unpack_from("<H", pkt, 17)[0] == len(pkt) - 0x15):
        return bytes(pkt[21:])
    return None


def audio_packets(local_id: int, remote_id: int, audio_seq: int, pcm: bytes) -> list[bytearray]:
    """20 ms of transmit audio (1920 bytes) as the radio wants it: two packets."""
    out = []
    start = 0
    for size in _TX_PARTS:
        part = pcm[start:start + size]
        start += size
        p = header(0x18 + len(part), T_DATA, 0, local_id, remote_id)
        p += bytes([0x80, 0x00]) + struct.pack(">H", (audio_seq - 1) & 0xFFFF)
        p += b"\x00\x00" + struct.pack(">H", len(part))
        out.append(p + part)
        audio_seq += 1
    return out


def audio_payload(pkt: bytes) -> bytes | None:
    """The PCM in an audio packet from the radio, or None."""
    if len(pkt) <= 24 or len(pkt) in (16, 21):
        return None
    head = parse_header(pkt)
    if head is None or head[0] != len(pkt) or head[1] != T_DATA:
        return None
    return bytes(pkt[24:])


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(block: np.ndarray) -> bytes:
    return (np.clip(block, -1.0, 32767 / 32768) * 32768.0).astype("<i2").tobytes()


# -- one UDP session ---------------------------------------------------------------------


class _Stream:
    """One of the three sessions: its socket, ids, sequence numbers and resend history."""

    HISTORY = 512

    def __init__(self, name: str, host: str, port: int) -> None:
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # The radio's own clients use the same port number at both ends.
            self.sock.bind(("", port))
        except OSError:
            self.sock.bind(("", 0))
        self.sock.connect((host, port))
        ip, local_port = self.sock.getsockname()
        self.local_id = ((struct.unpack(">I", socket.inet_aton(ip))[0] << 16)
                         | (local_port & 0xFFFF)) & 0xFFFFFFFF
        self.remote_id = 0
        self.seq = 1
        self.ping_seq = 1
        self._ping_inner = 0x8304
        self._history: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self.last_heard = time.monotonic()
        self.last_tracked = 0.0
        self.next_ping = 0.0
        self.next_idle = 0.0
        self.pinging = False
        self.lost = 0
        self._expect_seq: int | None = None

    def send(self, data: bytes) -> None:
        try:
            self.sock.send(data)
        except OSError:
            pass

    def send_tracked(self, pkt: bytearray, idle: bool = False) -> None:
        """Number it, keep it for a resend, send it."""
        with self._lock:
            struct.pack_into("<H", pkt, 6, self.seq)
            data = bytes(pkt)
            self._history[self.seq] = data
            if len(self._history) > self.HISTORY:
                del self._history[min(self._history)]
            self.seq = (self.seq + 1) & 0xFFFF
            if not idle:
                self.last_tracked = time.monotonic()
        self.send(data)

    def recv(self) -> bytes | None:
        try:
            data = self.sock.recv(2048)
        except OSError:
            return None
        self.last_heard = time.monotonic()
        return data

    def wait_for(self, match, timeout: float = STEP_TIMEOUT_S) -> bytes | None:
        """Read until a packet `match` accepts (answering pings meanwhile), or time out."""
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([self.sock], [], [], left)
            if not ready:
                break
            pkt = self.recv()
            if pkt is None:
                continue
            if self.handle_common(pkt):
                continue
            if match(pkt):
                return pkt
        return None

    def open(self) -> None:
        """The handshake every session starts with."""
        hello = control(T_ARE_YOU_THERE, self.local_id, 0)
        for _attempt in range(3):
            self.send(hello)
            here = self.wait_for(lambda p: len(p) == 16 and p[4] == T_I_AM_HERE, 1.0)
            if here is not None:
                self.remote_id = parse_header(here)[3]
                break
        else:
            raise ConnectionError(self._no_answer())
        ready = control(T_READY, self.local_id, self.remote_id, seq=1)
        self.send(ready)
        if self.wait_for(lambda p: len(p) == 16 and p[4] == T_READY) is None:
            raise ConnectionError(f"the radio's {self.name} port did not get ready")

    def _no_answer(self) -> str:
        """Why nothing answered, as far as can be told: usually the Mac is on another
        network (in AP mode the radio is only on its own, "IC-705")."""
        host, port = self.sock.getpeername()
        mine = self.sock.getsockname()[0]
        text = f"no answer from the radio's {self.name} port ({host}:{port})"
        if mine.rsplit(".", 1)[0] != host.rsplit(".", 1)[0]:
            text += (f" -- this Mac is on {mine}, another network: join the radio's "
                     "network (in AP mode, \"IC-705\") or check its address")
        return text

    def handle_common(self, pkt: bytes) -> bool:
        """Answer pings and resend requests. True if the packet needs nothing more."""
        if is_ping(pkt):
            if pkt[16] == 0x00 and self.pinging:          # the radio pinging us
                self.send(ping(self.local_id, self.remote_id,
                               struct.unpack_from("<H", pkt, 6)[0], True, pkt[17:21]))
            return True
        if len(pkt) >= 16 and pkt[4] == T_RETRANSMIT and pkt[5] == 0:
            if len(pkt) == 16:
                self._resend(struct.unpack_from("<H", pkt, 6)[0])
            else:
                for i in range(16, len(pkt) - 3, 4):
                    first, last = struct.unpack_from("<HH", pkt, i)
                    for seq in range(first, last + 1 if last >= first else first + 1):
                        self._resend(seq & 0xFFFF)
            return True
        return False

    def _resend(self, seq: int) -> None:
        with self._lock:
            data = self._history.get(seq)
        # One we no longer have is answered with an idle packet of that number.
        self.send(data if data is not None else control(T_DATA, self.local_id,
                                                         self.remote_id, seq=seq))

    def note_seq(self, pkt: bytes) -> None:
        """Count packets the radio sent that never arrived (for the status line)."""
        seq = struct.unpack_from("<H", pkt, 6)[0]
        if self._expect_seq is not None:
            gap = (seq - self._expect_seq) & 0xFFFF
            if 0 < gap < 1000:
                self.lost += gap
        self._expect_seq = (seq + 1) & 0xFFFF

    def tick(self, now: float, idle: bool) -> None:
        if self.pinging and now >= self.next_ping:
            self.next_ping = now + PING_S
            self.send(ping(self.local_id, self.remote_id, self.ping_seq, False,
                           bytes([os.urandom(1)[0], self._ping_inner & 0xFF,
                                  self._ping_inner >> 8 & 0xFF, 0x06])))
            self.ping_seq = (self.ping_seq + 1) & 0xFFFF
            self._ping_inner = (self._ping_inner + 1) & 0xFFFF
        if idle and now >= self.next_idle:
            fast = now - self.last_tracked < IDLE_FAST_FOR_S
            self.next_idle = now + (IDLE_FAST_S if fast else IDLE_SLOW_S)
            self.send_tracked(bytearray(control(T_DATA, self.local_id, self.remote_id)),
                              idle=True)

    def close(self, say_goodbye: bool) -> None:
        if say_goodbye and self.remote_id:
            bye = control(T_DISCONNECT, self.local_id, self.remote_id)
            self.send(bye)
            self.send(bye)
        self.sock.close()


# -- the link: login, then CI-V and audio ---------------------------------------------------


class LoginError(ConnectionError):
    """The radio answered but would not let us in."""


class IcomLink:
    """A logged-in network connection to the radio: CI-V both ways, audio both ways.

    `read`/`write`/`close` make it an `IcomSource` transport, as the USB serial port is.
    Received audio goes to `audio_sink` (a callable taking float32 blocks) when one is set.
    """

    #: The radio is on WiFi: the app takes TX audio from WLAN, not USB.
    wlan = True

    def __init__(self, host: str, user: str, password: str,
                 timeout: float = STEP_TIMEOUT_S,
                 ports: tuple[int, int, int] = (CONTROL_PORT, SERIAL_PORT, AUDIO_PORT)) -> None:
        self.host = host
        self.ports = ports
        self.user = user
        self.radio_name = ""
        #: Why the link stopped, once it has.
        self.error: str | None = None
        self.audio_sink = None
        self._civ_in = bytearray()
        self._civ_ready = threading.Condition()
        self._civ_seq = 0
        self._audio_seq = 1
        self._audio_lock = threading.Lock()
        self._inner = 0
        self._token = b""
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_reauth = 0.0
        self.control = self.serial = self.audio = None
        try:
            self._login(user, password, timeout)
        except Exception:
            self._shut(goodbye=False)
            raise
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="ic705-net", daemon=True)
        self._thread.start()

    # -- login ---------------------------------------------------------------------------

    def _next_inner(self) -> int:
        value, self._inner = self._inner, (self._inner + 1) & 0xFFFF
        return value

    def _login(self, user: str, password: str, timeout: float) -> None:
        try:
            ctl = self.control = _Stream("control", self.host, self.ports[0])
        except OSError as exc:
            raise ConnectionError(f"cannot reach {self.host}: {exc}") from exc
        ctl.open()
        ctl.send_tracked(login_packet(ctl.local_id, ctl.remote_id, self._next_inner(),
                                      os.urandom(2), user, password))
        reply = ctl.wait_for(lambda p: len(p) == 0x60 and p[:6] == b"\x60\x00\x00\x00\x00\x00",
                             timeout)
        if reply is None:
            raise ConnectionError("the radio did not answer the login")
        if reply[48:52] == b"\xff\xff\xff\xfe":
            raise LoginError("the radio refused the user name or password")
        self._token = bytes(reply[26:32])
        ctl.pinging = True
        ctl.ping_seq = 2
        ctl.send_tracked(token_packet(ctl.local_id, ctl.remote_id, self._next_inner(), 0x02,
                                      self._token))
        ctl.send_tracked(token_packet(ctl.local_id, ctl.remote_id, self._next_inner(), 0x05,
                                      self._token))
        guid = name = None
        authorised = False
        deadline = time.monotonic() + timeout * 2
        while time.monotonic() < deadline and not (authorised and guid is not None):
            pkt = ctl.wait_for(lambda p: True, max(0.05, deadline - time.monotonic()))
            if pkt is None:
                break
            self._check_status(pkt)
            if len(pkt) == 0x40 and pkt[:6] == b"\x40\x00\x00\x00\x00\x00" and pkt[21] == 0x05:
                authorised = True
            elif len(pkt) == 0xA8 and pkt[:6] == b"\xa8\x00\x00\x00\x00\x00":
                guid = bytes(pkt[66:82])
                name = bytes(pkt[82:114]).split(b"\x00", 1)[0]
        if not authorised or guid is None:
            raise ConnectionError("the radio did not finish the login")
        name = name or b"IC-705"
        self.radio_name = name.decode("ascii", "replace")
        ctl.send_tracked(conninfo_packet(ctl.local_id, ctl.remote_id, self._next_inner(),
                                         self._token, guid, name, user, self.ports[1:]))

        def opened(p: bytes) -> bool:
            self._check_status(p)
            return len(p) == 0x90 and p[:6] == b"\x90\x00\x00\x00\x00\x00" and p[96] == 1

        reply = ctl.wait_for(opened, timeout * 2)
        if reply is None:
            raise ConnectionError("the radio did not open its CI-V and audio ports "
                                  "(is another program connected to it?)")
        ctl.remote_id, ctl.local_id = struct.unpack_from(">II", reply, 8)
        self._token = bytes(reply[26:32])
        self._next_reauth = time.monotonic() + REAUTH_S

        self.serial = _Stream("CI-V", self.host, self.ports[1])
        self.serial.open()
        self.serial.pinging = True
        self.serial.send_tracked(civ_open_packet(self.serial.local_id, self.serial.remote_id,
                                                 self._civ_seq, True))
        self._civ_seq += 1
        self.audio = _Stream("audio", self.host, self.ports[2])
        self.audio.open()
        self.audio.pinging = True
        self.audio.seq = 0                    # this session's numbering starts at 0

    def _check_status(self, pkt: bytes) -> None:
        """The radio's status packet says when it has thrown us out."""
        if len(pkt) == 0x50 and pkt[:6] == b"\x50\x00\x00\x00\x00\x00":
            if pkt[48:51] == b"\xff\xff\xff":
                raise LoginError("the radio refused the connection (try switching it off "
                                 "and on)")
            if pkt[48:51] == b"\x00\x00\x00" and pkt[64] == 0x01:
                raise ConnectionError("the radio closed the connection")

    # -- running -------------------------------------------------------------------------

    def _run(self) -> None:
        streams = [s for s in (self.control, self.serial, self.audio) if s is not None]
        by_sock = {s.sock: s for s in streams}
        try:
            while self._running.is_set():
                ready, _, _ = select.select(list(by_sock), [], [], 0.02)
                for sock in ready:
                    stream = by_sock[sock]
                    pkt = stream.recv()
                    if pkt is not None and not stream.handle_common(pkt):
                        self._dispatch(stream, pkt)
                now = time.monotonic()
                self.control.tick(now, idle=True)
                self.serial.tick(now, idle=True)
                self.audio.tick(now, idle=False)
                if now >= self._next_reauth:
                    self._next_reauth = now + REAUTH_S
                    ctl = self.control
                    ctl.send_tracked(token_packet(ctl.local_id, ctl.remote_id,
                                                  self._next_inner(), 0x05, self._token))
                if now - self.control.last_heard > SILENT_S:
                    raise ConnectionError("lost the radio (no reply for "
                                          f"{SILENT_S:.0f} s)")
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self._running.clear()
            with self._civ_ready:
                self._civ_ready.notify_all()

    def _dispatch(self, stream: _Stream, pkt: bytes) -> None:
        if stream is self.control:
            self._check_status(pkt)
        elif stream is self.serial:
            data = civ_payload(pkt)
            if data is not None or (len(pkt) == 16 and pkt[4] == T_DATA):
                stream.note_seq(pkt)          # idle packets take sequence numbers too
            if data is not None:
                with self._civ_ready:
                    self._civ_in += data
                    del self._civ_in[:-65536]
                    self._civ_ready.notify_all()
        elif stream is self.audio:
            pcm = audio_payload(pkt)
            sink = self.audio_sink
            if pcm is not None:
                stream.note_seq(pkt)
                if sink is not None:
                    sink(pcm_to_float(pcm))

    @property
    def connected(self) -> bool:
        return self._running.is_set()

    @property
    def stats(self) -> dict:
        return {"lost_civ": self.serial.lost if self.serial else 0,
                "lost_audio": self.audio.lost if self.audio else 0}

    # -- the transport face (IcomSource) -----------------------------------------------

    def read(self, n: int) -> bytes:
        """Up to n CI-V bytes from the radio, waiting at most 50 ms (as the serial port's
        timeout). Empty once the link has stopped."""
        with self._civ_ready:
            if not self._civ_in and self._running.is_set():
                self._civ_ready.wait(0.05)
            data = bytes(self._civ_in[:n])
            del self._civ_in[:n]
        if not data and not self._running.is_set():
            time.sleep(0.05)                  # do not spin a reader on a dead link
        return data

    def write(self, data: bytes) -> int:
        """CI-V to the radio, one frame to a packet."""
        if self.serial is None or not self._running.is_set():
            return 0
        for frame in data.split(b"\xfd"):
            if not frame:
                continue
            frame += b"\xfd"
            self.serial.send_tracked(civ_packet(self.serial.local_id, self.serial.remote_id,
                                                self._civ_seq, frame))
            self._civ_seq = (self._civ_seq + 1) & 0xFFFF
        return len(data)

    def send_audio(self, block: np.ndarray) -> None:
        """20 ms (TX_BLOCK_SAMPLES) of transmit audio, -1..1."""
        if self.audio is None or not self._running.is_set():
            return
        pcm = float_to_pcm(np.asarray(block, dtype=np.float32)[:TX_BLOCK_SAMPLES])
        with self._audio_lock:
            for pkt in audio_packets(self.audio.local_id, self.audio.remote_id,
                                     self._audio_seq, pcm):
                self.audio.send_tracked(pkt)
            self._audio_seq = (self._audio_seq + len(_TX_PARTS)) & 0xFFFF

    def close(self) -> None:
        self._shut(goodbye=True)

    def _shut(self, goodbye: bool) -> None:
        was_running = self._running.is_set()
        self._running.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        goodbye = goodbye and was_running
        if goodbye and self.serial is not None:
            self.serial.send_tracked(civ_open_packet(self.serial.local_id,
                                                     self.serial.remote_id, self._civ_seq,
                                                     False))
        if goodbye and self.control is not None and self._token:
            ctl = self.control
            ctl.send_tracked(token_packet(ctl.local_id, ctl.remote_id, self._next_inner(),
                                          0x01, self._token))
            # A moment to answer any resend request for the sign-off.
            ctl.wait_for(lambda p: False, 0.3)
        for stream in (self.audio, self.serial, self.control):
            if stream is not None:
                stream.close(goodbye)
        self.control = self.serial = self.audio = None
        with self._civ_ready:
            self._civ_ready.notify_all()


# -- where the radio is, and how to log in ---------------------------------------------------

#: The radio's address and Network User ID; the password goes in the macOS Keychain.
LOGIN_FILE = Path.home() / "Library" / "Application Support" / "RGC_SDR" / "ic705_network.json"
KEYCHAIN_SERVICE = "RGC_SDR IC-705 network"


@dataclass
class NetworkLogin:
    host: str = ""
    user: str = ""
    password: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.host and self.user)


def _keychain(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["security", *args], capture_output=True, text=True,
                              timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None


def load_login(path: Path | None = None, keychain: bool = True) -> NetworkLogin:
    path = Path(path) if path is not None else LOGIN_FILE
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return NetworkLogin()
    if not isinstance(data, dict):
        return NetworkLogin()
    login = NetworkLogin(str(data.get("host", "")).strip(), str(data.get("user", "")).strip())
    if keychain and login.complete:
        found = _keychain("find-generic-password", "-s", KEYCHAIN_SERVICE,
                          "-a", f"{login.user}@{login.host}", "-w")
        if found is not None and found.returncode == 0:
            login.password = found.stdout.rstrip("\n")
    return login


def save_login(login: NetworkLogin, path: Path | None = None, keychain: bool = True) -> None:
    path = Path(path) if path is not None else LOGIN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"host": login.host, "user": login.user}))
    if keychain and login.complete:
        _keychain("add-generic-password", "-U", "-s", KEYCHAIN_SERVICE,
                  "-a", f"{login.user}@{login.host}", "-w", login.password)


def open_link(login: NetworkLogin | None = None) -> IcomLink:
    """Log in to the radio as last set up (SDR menu > Icom IC-705 (WiFi))."""
    login = login or load_login()
    if not login.complete:
        raise ConnectionError("its WiFi address and user are not set up yet")
    return IcomLink(login.host, login.user, login.password)
