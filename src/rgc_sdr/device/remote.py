"""A radio on the network radio server (P9, PLANNING.md 7q), as an ordinary IQ source.

`RemoteIQSource` connects to `rgc_sdr.netserver` on another machine (the Pi 5, over
Tailscale) and opens one of its radios. The server sends two things: IQ for a window
around the tuned frequency, which fills a local ring -- its rate is this source's
`sample_rate`, so the demodulators and decoders need nothing new -- and spectrum lines of
the radio's whole span, which the window draws instead of an FFT of its own
(`take_spectrum_lines`). The radio's own rate is `span_rate`, chosen from the Rate list.

Changes are sent without waiting for the reply, so dragging a digit over the internet
does not stall the window: the frequency is predicted (clamped to the radio's range),
and each flushing change bumps a generation number the server tags its IQ with, so
blocks made before it are dropped. A dropped link is reconnected, the radio reopened
where it was.

Servers are named in a small file of their own (`servers.json`), like the IC-705's
login: a Tailscale name ("radiopi") is enough, so no address need be stored.

No Qt and no DSP imports (layering rule, PLANNING.md section 5).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .remote_protocol import (
    DEFAULT_PORT, FRAME_IQ, FRAME_SPECTRUM, PROTOCOL_VERSION, ProtocolError,
    caps_from_dict, decode_iq, decode_spectrum, iq_rate_for, json_frame, parse_json,
    read_frame,
)
from .source import IQSource, SequentialReader, _Ring

SERVERS_FILE = Path.home() / "Library" / "Application Support" / "RGC_SDR" / "servers.json"
#: How long to wait for a server that may be across the internet.
CONNECT_TIMEOUT_S = 3.0
REPLY_TIMEOUT_S = 8.0          # opening a radio on the Pi takes a second or two
#: Listing is asked for whenever the radio list is shown: cached, so an unreachable
#: server costs one timeout now and then, not one per click.
LIST_CACHE_S = 20.0
LIST_TIMEOUT_S = 1.5
RING_SECONDS = 2.0
#: Spectrum lines kept for the window to draw: a second or two at 25 a second.
SPECTRUM_QUEUE = 50
#: The highest radio rate offered remotely: the Pi's own ceiling, as the app's.
MAX_REMOTE_RADIO_RATE = 10e6


@dataclass(frozen=True)
class SpectrumLine:
    """One line of the radio's spectrum, as the server computed it."""

    centre_hz: float
    span_hz: float
    dbfs: np.ndarray


class RemoteError(RuntimeError):
    """The server refused a request, or could not be reached."""


# -- the servers to look on ------------------------------------------------------------


@dataclass(frozen=True)
class ServerAddress:
    host: str
    port: int = DEFAULT_PORT

    @property
    def name(self) -> str:
        """For keys and menus: "radiopi" of "radiopi" or "radiopi.example.ts.net"."""
        return self.host.split(".")[0].lower() if not _is_ip(self.host) else self.host

    def text(self) -> str:
        return self.host if self.port == DEFAULT_PORT else f"{self.host}:{self.port}"


def _is_ip(host: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET6 if ":" in host else socket.AF_INET, host)
        return True
    except OSError:
        return False


def parse_servers(text: str) -> list[ServerAddress]:
    """'radiopi, other:55134' -> addresses. Blank entries are skipped."""
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        host, port = part, DEFAULT_PORT
        if part.count(":") == 1:
            host, _, number = part.partition(":")
            port = int(number)
        out.append(ServerAddress(host.strip(), port))
    return out


def load_servers(path: Path | None = None) -> list[ServerAddress]:
    path = Path(path) if path is not None else SERVERS_FILE
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for entry in data.get("servers", []) if isinstance(data, dict) else []:
        try:
            out.append(ServerAddress(str(entry["host"]), int(entry.get("port", DEFAULT_PORT))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_servers(servers: list[ServerAddress], path: Path | None = None) -> None:
    path = Path(path) if path is not None else SERVERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"servers": [{"host": s.host, "port": s.port} for s in servers]}, indent=2))
    _list_cache.clear()


# -- a connection ----------------------------------------------------------------------


class _Connection:
    """One TCP connection: requests out, replies matched by id, IQ handed to `on_iq`."""

    def __init__(self, server: ServerAddress, on_iq=None, on_close=None,
                 timeout: float = CONNECT_TIMEOUT_S, on_spectrum=None) -> None:
        self.server = server
        self._sock = socket.create_connection((server.host, server.port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(None)
        self._on_iq = on_iq
        self._on_spectrum = on_spectrum
        self._on_close = on_close
        self._send_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, list] = {}
        self._pending_lock = threading.Lock()
        #: Called with each reply's state, in order, whether waited for or not.
        self.on_state = None
        self.closed = False
        self._thread = threading.Thread(target=self._receive, name="remote-rx", daemon=True)
        self._thread.start()

    @property
    def next_id(self) -> int:
        """The id the next request will carry."""
        return self._next_id + 1

    def request(self, op: str, wait: bool = True, timeout: float = REPLY_TIMEOUT_S,
                **fields) -> dict:
        """Send a request; with `wait`, its reply (RemoteError if refused)."""
        with self._pending_lock:
            self._next_id += 1
            ident = self._next_id
            slot = [threading.Event(), None]
            if wait:
                self._pending[ident] = slot
        try:
            with self._send_lock:
                self._sock.sendall(json_frame({"id": ident, "op": op, **fields}))
        except OSError as exc:
            with self._pending_lock:
                self._pending.pop(ident, None)
            raise RemoteError(f"{self.server.name}: {exc}") from None
        if not wait:
            return {"id": ident}
        if not slot[0].wait(timeout):
            with self._pending_lock:
                self._pending.pop(ident, None)
            raise RemoteError(f"{self.server.name}: no reply to {op}")
        reply = slot[1]
        if reply is None:
            raise RemoteError(f"{self.server.name}: connection lost")
        if not reply.get("ok"):
            raise RemoteError(f"{self.server.name}: {reply.get('error', 'refused')}")
        return reply

    def _receive(self) -> None:
        try:
            while True:
                kind, payload = read_frame(self._sock)
                if kind == FRAME_IQ:
                    if self._on_iq is not None:
                        self._on_iq(*decode_iq(payload))
                    continue
                if kind == FRAME_SPECTRUM:
                    if self._on_spectrum is not None:
                        self._on_spectrum(*decode_spectrum(payload))
                    continue
                reply = parse_json(payload)
                state = reply.get("state")
                if state is not None and self.on_state is not None:
                    self.on_state(reply.get("id"), state)
                with self._pending_lock:
                    slot = self._pending.pop(reply.get("id"), None)
                if slot is not None:
                    slot[1] = reply
                    slot[0].set()
        except (ConnectionError, OSError, ProtocolError):
            pass
        finally:
            self.closed = True
            with self._pending_lock:
                pending, self._pending = list(self._pending.values()), {}
            for slot in pending:
                slot[0].set()                       # their reply is None: lost
            if self._on_close is not None:
                self._on_close()

    def close(self) -> None:
        self._on_close = None
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def hello(conn: _Connection) -> str:
    """Check the protocol; returns the server's host name."""
    return str(conn.request("hello", version=PROTOCOL_VERSION).get("server", ""))


_list_cache: dict[ServerAddress, tuple[float, list[dict]]] = {}


def list_radios(server: ServerAddress, use_cache: bool = True) -> list[dict]:
    """The radios attached to `server` ([] if it cannot be reached), cached briefly."""
    now = time.monotonic()
    cached = _list_cache.get(server)
    if use_cache and cached is not None and now - cached[0] < LIST_CACHE_S:
        return cached[1]
    try:
        conn = _Connection(server, timeout=LIST_TIMEOUT_S)
        try:
            hello(conn)
            radios = [r for r in conn.request("list", timeout=LIST_TIMEOUT_S * 2)
                      .get("radios", []) if isinstance(r, dict) and r.get("driver")]
        finally:
            conn.close()
    except (OSError, RemoteError):
        radios = []
    _list_cache[server] = (now, radios)
    return radios


# -- profiles for remote radios -------------------------------------------------------


def remote_key(driver: str, server: ServerAddress) -> str:
    """The app's key for a radio on a server: "rtlsdr@radiopi"."""
    return f"{driver}@{server.name}"


def remote_profile(base, server: ServerAddress):
    """The profile of model `base` on `server`: its own rates as the span, receive only,
    and opened over the network."""
    ceiling = min(base.max_rate, MAX_REMOTE_RADIO_RATE)
    rates = tuple(sorted({r for r in base.sample_rates + base.extra_rates if r <= ceiling},
                         reverse=True)) or base.sample_rates[:1]
    iq_rate, _ = iq_rate_for(base.default_rate)
    return replace(
        base,
        key=remote_key(base.driver, server),
        label=f"{base.label} on {server.name}",
        driver=remote_key(base.driver, server),
        sample_rates=rates,
        max_rate=ceiling,
        extra_rates=(),
        module="network",
        install=f"the network radio server running on {server.name} (PLANNING.md 7q)",
        notes=f"{base.label} plugged into {server.name}, over the network: its whole span "
              f"on the waterfall, and about {iq_rate / 1e3:g} kHz around the tuned "
              f"frequency for listening and decoding. Receive only.",
        tx=None,
        lo_offset_hz=0.0,
        remote=(server.host, server.port, base.driver),
    )


# -- the source ------------------------------------------------------------------------


class RemoteIQSource(IQSource):
    """A radio on a network radio server. See the module docstring."""

    def __init__(self, server: ServerAddress, driver: str, serial: str | None = None,
                 sample_rate: float | None = None, center_freq: float = 100e6,
                 profile=None) -> None:
        self.server = server
        self._driver = driver
        self._serial = serial or ""
        self.profile = profile
        self._generation = 0
        self._lock = threading.Lock()
        self._closed = False
        self._running = False
        self._samples = 0
        self._stale = 0
        self._reconnects = 0
        self._last_tune_id = 0
        #: The radio's rate wanted: the span. None for the profile's default.
        self._wanted_rate = float(sample_rate) if sample_rate else (
            profile.default_rate if profile else 0.0)
        self._freq = float(center_freq)
        self._state: dict = {}
        self._lines: deque[SpectrumLine] = deque(maxlen=SPECTRUM_QUEUE)
        self._latest_line: SpectrumLine | None = None
        #: (fft size, lines a second, zoom) asked of the server.
        self._display = (4096, 25.0, 1)
        self._conn: _Connection | None = None
        self._open()
        self._ring = _Ring(self._ring_capacity())

    # -- connecting

    def _open(self) -> None:
        conn = _Connection(self.server, on_iq=self._on_iq, on_close=self._on_closed,
                           on_spectrum=self._on_spectrum)
        fft_size, fps, zoom = self._display
        try:
            hello(conn)
            reply = conn.request("open", driver=self._driver, serial=self._serial,
                                 radio_rate=self._wanted_rate, hz=self._freq,
                                 generation=self._generation, fft_size=fft_size, fps=fps,
                                 zoom=zoom)
        except Exception:
            conn.close()
            raise
        caps = caps_from_dict(reply["caps"])
        # Its key, not the model's driver: reopening after a failed switch goes through
        # the factory by `caps.driver`.
        self._caps = replace(caps, driver=self.profile.key if self.profile else caps.driver,
                             label=f"{caps.label} on {self.server.name}")
        self._state = dict(reply["state"])
        self._freq = float(self._state.get("freq", self._freq))
        conn.on_state = self._on_state
        self._conn = conn

    def _on_closed(self) -> None:
        """The link dropped: reconnect in the background and carry on where we were."""
        if self._closed:
            return
        threading.Thread(target=self._reconnect, name="remote-reconnect", daemon=True).start()

    def _reconnect(self) -> None:
        delay = 1.0
        while not self._closed:
            time.sleep(delay)
            try:
                gains = dict(self._state.get("gains", {}))
                agc = self._state.get("agc")
                self._open()
                for name, db in gains.items():
                    self._conn.request("gain", wait=False, name=name, db=db)
                if agc is not None:
                    self._conn.request("agc", wait=False, on=bool(agc))
                if self._running:
                    self._conn.request("start")
                self._reconnects += 1
                return
            except (OSError, RemoteError, KeyError):
                delay = min(delay * 2, 15.0)

    def _send(self, op: str, wait: bool = False, **fields) -> dict:
        conn = self._conn
        if conn is None or conn.closed:
            return {}                      # reconnecting: the reopen restores the state
        try:
            return conn.request(op, wait=wait, **fields)
        except RemoteError:
            if wait:
                raise
            return {}

    def _on_state(self, ident, state: dict) -> None:
        with self._lock:
            self._state = dict(state)
            # A reply to an older tune would put the dial back: only the latest counts.
            if ident is not None and ident >= self._last_tune_id:
                self._freq = float(state.get("freq", self._freq))

    def _on_iq(self, generation: int, iq: np.ndarray) -> None:
        if generation != self._generation:
            self._stale += iq.size             # made before the latest retune
            return
        self._ring.write(iq)
        self._samples += iq.size

    def _on_spectrum(self, centre: float, span: float, dbfs: np.ndarray) -> None:
        line = SpectrumLine(centre, span, dbfs)
        self._lines.append(line)
        self._latest_line = line

    def _ring_capacity(self) -> int:
        return max(int(self.sample_rate * RING_SECONDS), 1 << 16)

    # -- IQSource

    @property
    def caps(self):
        return self._caps

    @property
    def sample_rate(self) -> float:
        """The IQ window's rate: what the demodulators and decoders receive."""
        return float(self._state.get("iq_rate", 0.0)) or iq_rate_for(self.span_rate)[0]

    @property
    def span_rate(self) -> float:
        """The radio's own rate, which is the span the spectrum lines cover."""
        return float(self._state.get("radio_rate", self._wanted_rate))

    # -- the spectrum (P9b): drawn from the server's lines, not an FFT of the IQ

    def take_spectrum_lines(self) -> list[SpectrumLine]:
        """Every line received since the last call, oldest first."""
        out = []
        while self._lines:
            try:
                out.append(self._lines.popleft())
            except IndexError:
                break
        return out

    @property
    def display_center_freq(self) -> float:
        line = self._latest_line
        return line.centre_hz if line is not None else float(
            self._state.get("radio_freq", self._freq))

    @property
    def display_span(self) -> float:
        line = self._latest_line
        return line.span_hz if line is not None else self.span_rate / self._display[2]

    def set_display(self, fft_size: int, fps: float, zoom: int) -> None:
        """What the spectrum lines should be: bins, lines a second, and zoom."""
        wanted = (int(fft_size), float(fps), max(1, int(zoom)))
        if wanted != self._display:
            self._display = wanted
            self._send("display", fft_size=wanted[0], fps=wanted[1], zoom=wanted[2])

    @property
    def center_freq(self) -> float:
        return self._freq

    @property
    def dc_spike_offset_hz(self) -> float:
        return float(self._state.get("dc_spike_offset", 0.0))

    def start(self) -> None:
        if not self._running:
            self._send("start", wait=True)
            self._running = True

    def stop(self) -> None:
        if self._running:
            self._running = False
            try:
                self._send("stop", wait=True)
            except RemoteError:
                pass

    def read_latest(self, n: int) -> np.ndarray:
        return self._ring.read_latest(n)

    def sequential_reader(self) -> SequentialReader:
        return SequentialReader(self._ring)

    def set_center_freq(self, hz: float, flush: bool = True) -> float:
        wanted = self._caps.clamp_freq(float(hz))
        with self._lock:
            if flush:
                self._generation += 1
                self._ring.clear()
            self._freq = wanted
        conn = self._conn
        if conn is not None:
            # Before sending, or its reply could arrive first and be taken as stale.
            self._last_tune_id = conn.next_id
        self._send("tune", hz=wanted, flush=flush, generation=self._generation)
        return wanted

    def set_sample_rate(self, hz: float) -> float:
        """Set the radio's rate -- the span -- which the Rate list, memories and settings
        hold; the IQ window follows from it. Returns the span rate."""
        target = self._caps.nearest_sample_rate(float(hz))
        if target == self.span_rate:
            return target
        self._wanted_rate = target
        with self._lock:
            self._generation += 1
        reply = self._send("rate", wait=True, radio_rate=target, generation=self._generation)
        self._state = dict(reply.get("state", self._state))
        self._ring = _Ring(self._ring_capacity())
        self._lines.clear()
        self._latest_line = None
        return self.span_rate

    def set_gain(self, name: str, db: float) -> None:
        self._state.setdefault("gains", {})[name] = float(db)
        self._send("gain", name=name, db=float(db))

    def get_gain(self, name: str) -> float:
        return float(self._state.get("gains", {}).get(name, 0.0))

    def set_agc(self, enabled: bool) -> None:
        if self._caps.has_agc:
            self._state["agc"] = bool(enabled)
            self._send("agc", on=bool(enabled))

    def get_agc(self) -> bool:
        return bool(self._state.get("agc", False))

    @property
    def bandwidth(self) -> float:
        return float(self._state.get("bandwidth", 0.0))

    def set_bandwidth(self, hz: float) -> float:
        if self._caps.bandwidths:
            self._state["bandwidth"] = float(hz)
            self._send("bandwidth", hz=float(hz))
        return self.bandwidth

    def read_setting(self, key: str) -> bool:
        return bool(self._state.get("settings", {}).get(key, False))

    def write_setting(self, key: str, enabled: bool) -> None:
        self._state.setdefault("settings", {})[key] = bool(enabled)
        self._send("setting", key=key, on=bool(enabled))

    @property
    def ppm(self) -> float:
        return float(self._state.get("ppm", 0.0))

    def set_ppm(self, ppm: float) -> None:
        self._state["ppm"] = float(ppm)
        self._send("ppm", ppm=float(ppm))

    @property
    def stats(self) -> dict[str, int]:
        return {"samples": self._samples, "dropped": self._stale,
                "overflows": int(self._state.get("lost", 0)), "timeouts": 0,
                "errors": self._reconnects}

    def close(self) -> None:
        self._closed = True
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.request("close", timeout=2.0)
            except RemoteError:
                pass
            conn.close()
