"""The network radio server (P9a, PLANNING.md 7q): runs on the Pi, serves its SDRs.

    python3 -m rgc_sdr.netserver --bind 100.x.y.z

A client lists the radios attached, opens one, and receives its IQ decimated on the Pi
to a link rate the network can carry; tuning, gains and the rest are JSON requests. The
radio is opened with the app's own `SoapyIQSource`, so every driver quirk, LO offset and
ppm correction behaves here as it does on the Mac. Receive only. Any number of clients may
connect and list radios; one at a time has a radio open, and opening one closes the
previous client's (a dropped link would otherwise hold it until TCP gives up).

No Qt. Python 3.13-clean: it runs on the Pi's Debian Python.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time

from .device.profiles import profile_for
from .device.remote_protocol import (
    DEFAULT_PORT, FRAME_JSON, PROTOCOL_VERSION, ProtocolError, caps_to_dict, encode_iq,
    json_frame, link_plan, parse_json, read_frame,
)
from .dsp.decimate import StreamDecimator

#: IQ is sent in blocks of about this long: short enough for a responsive waterfall,
#: long enough that the 12-byte headers and per-block work stay negligible.
BLOCK_S = 0.02


def list_radios() -> list[dict]:
    """The SDRs attached here that the app knows (not the sound cards and other things
    Soapy modules also enumerate)."""
    from .device.source import enumerate_devices

    out = []
    for found in enumerate_devices():
        driver = found.get("driver", "")
        profile = profile_for(driver)
        if profile is None or profile.kind != "sdr":
            continue
        out.append({"driver": driver, "serial": found.get("serial", ""),
                    "label": found.get("label", profile.label)})
    # Some drivers do not list a radio that is open: the one in use is still here.
    for session in list(RadioSession.open_sessions):
        entry = {"driver": session.driver, "serial": session.serial,
                 "label": session.source.caps.label, "in_use": True}
        if not any(r["driver"] == entry["driver"] and r["serial"] in ("", entry["serial"])
                   for r in out):
            out.append(entry)
    return out


class RadioSession:
    """One open radio and the pump that sends its decimated IQ."""

    #: Sessions with a radio open, for `list_radios`.
    open_sessions: set["RadioSession"] = set()

    def __init__(self, send, driver: str, serial: str | None, link_rate: float,
                 centre_hz: float, generation: int, source_factory=None) -> None:
        from .device.source import SoapyIQSource

        profile = profile_for(driver)
        if profile is None or profile.kind != "sdr":
            raise ValueError(f"not an SDR this server knows: {driver!r}")
        factory = source_factory or SoapyIQSource
        self._send = send
        self.driver, self.serial = driver, serial or ""
        self.source = factory(driver=driver, serial=serial or None, center_freq=centre_hz)
        self.plan = link_plan(self.source.caps.sample_rates, profile.default_rate)
        if not self.plan:
            self.source.close()
            raise ValueError(f"{profile.label}: no link rate fits")
        self._lock = threading.Lock()
        self.generation = int(generation)
        self.link_rate = 0.0
        self.factor = 1
        self._decimator = StreamDecimator(1)
        self._reader = None
        self._apply_rate(link_rate)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        RadioSession.open_sessions.add(self)

    # -- the radio's state, as sent to the client

    def state(self) -> dict:
        src, caps = self.source, self.source.caps
        gains = {g.name: src.get_gain(g.name) for g in caps.gain_elements}
        settings = {s.key: src.read_setting(s.key) for s in caps.settings}
        return {
            "freq": src.center_freq, "link_rate": self.link_rate,
            "radio_rate": src.sample_rate, "factor": self.factor, "gains": gains,
            "agc": src.get_agc(), "bandwidth": getattr(src, "bandwidth", 0.0),
            "settings": settings, "ppm": getattr(src, "ppm", 0.0),
            "dc_spike_offset": src.dc_spike_offset_hz, "generation": self.generation,
            "lost": getattr(self._reader, "lost", 0),
        }

    def caps(self) -> dict:
        """The radio's capabilities, with the link rates as its sample rates."""
        out = caps_to_dict(self.source.caps)
        out["sample_rates"] = list(self.plan)
        return out

    # -- changes

    def _apply_rate(self, link_rate: float) -> None:
        link = min(self.plan, key=lambda r: abs(r - float(link_rate)))
        radio_rate, factor = self.plan[link]
        self.source.set_sample_rate(radio_rate)
        self.link_rate, self.factor = link, factor
        self._decimator = StreamDecimator(factor)
        # A rate change rebuilds the source's ring: read from the new one.
        self._reader = self.source.sequential_reader()

    def set_rate(self, link_rate: float, generation: int) -> None:
        with self._lock:
            self._apply_rate(link_rate)
            self.generation = int(generation)

    def tune(self, hz: float, flush: bool, generation: int) -> None:
        with self._lock:
            self.source.set_center_freq(float(hz), flush=flush)
            if flush:
                self._decimator.reset()
                self._reader.skip_to_latest()
                self.generation = int(generation)

    def apply(self, op: str, message: dict) -> None:
        """The simple setters: gain, AGC, IF bandwidth, a driver switch, ppm."""
        src = self.source
        if op == "gain":
            src.set_gain(str(message["name"]), float(message["db"]))
        elif op == "agc":
            src.set_agc(bool(message["on"]))
        elif op == "bandwidth":
            src.set_bandwidth(float(message["hz"]))
        elif op == "setting":
            src.write_setting(str(message["key"]), bool(message["on"]))
        elif op == "ppm":
            src.set_ppm(float(message["ppm"]))
        else:
            raise ValueError(f"unknown op {op!r}")

    # -- streaming

    def start(self) -> None:
        if self._thread is not None:
            return
        self.source.start()
        with self._lock:
            self._reader = self.source.sequential_reader()
            self._decimator.reset()
        self._running.set()
        self._thread = threading.Thread(target=self._pump, name="iq-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.stop()

    def close(self) -> None:
        RadioSession.open_sessions.discard(self)
        self.stop()
        self.source.close()

    def _pump(self) -> None:
        while self._running.is_set():
            with self._lock:
                reader, decimator = self._reader, self._decimator
                block = int(self.source.sample_rate * BLOCK_S) // self.factor * self.factor
                if reader.available() < block:
                    out = None
                else:
                    generation = self.generation
                    out = decimator.process(reader.read(block))
            if out is None:
                time.sleep(BLOCK_S / 4)
                continue
            if out.size:
                try:
                    self._send(encode_iq(out, generation))
                except OSError:
                    self._running.clear()            # the client has gone
                    return


class ClientHandler:
    """One client's requests, in order, on its own thread."""

    def __init__(self, conn: socket.socket, source_factory=None, claim=None) -> None:
        self.conn = conn
        #: Called before opening a radio, to close any other client's.
        self._claim = claim
        self._send_lock = threading.Lock()
        self.session: RadioSession | None = None
        self._source_factory = source_factory

    def send(self, data: bytes) -> None:
        with self._send_lock:
            self.conn.sendall(data)

    def reply(self, request: dict, **fields) -> None:
        self.send(json_frame({"id": request.get("id"), "ok": True, **fields}))

    def run(self) -> None:
        try:
            while True:
                kind, payload = read_frame(self.conn)
                if kind != FRAME_JSON:
                    raise ProtocolError("a client sends only control messages")
                request = parse_json(payload)
                try:
                    self.handle(request)
                except Exception as exc:          # the request failed, not the link
                    self.send(json_frame({"id": request.get("id"), "ok": False,
                                          "error": f"{type(exc).__name__}: {exc}"}))
        except (ConnectionError, OSError, ProtocolError):
            pass
        finally:
            self.close()

    def handle(self, request: dict) -> None:
        op = request.get("op")
        if op == "hello":
            if request.get("version") != PROTOCOL_VERSION:
                raise ProtocolError(f"protocol {request.get('version')}, server speaks "
                                    f"{PROTOCOL_VERSION}")
            self.reply(request, version=PROTOCOL_VERSION, server=socket.gethostname())
            return
        if op == "list":
            self.reply(request, radios=list_radios())
            return
        if op == "open":
            self._close_session()
            if self._claim is not None:
                self._claim(self)
            self.session = RadioSession(
                self.send, str(request["driver"]), request.get("serial"),
                float(request.get("link_rate", 0)), float(request.get("hz", 100e6)),
                int(request.get("generation", 0)), self._source_factory)
            self.reply(request, caps=self.session.caps(), state=self.session.state())
            return
        session = self.session
        if session is None:
            raise ValueError("no radio open")
        if op == "start":
            session.start()
        elif op == "stop":
            session.stop()
        elif op == "close":
            self._close_session()
            self.reply(request)
            return
        elif op == "tune":
            session.tune(float(request["hz"]), bool(request.get("flush", True)),
                         int(request.get("generation", session.generation)))
        elif op == "rate":
            session.set_rate(float(request["link_rate"]),
                             int(request.get("generation", session.generation)))
        elif op != "state":
            session.apply(op, request)
        self.reply(request, state=session.state())

    def _close_session(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            session.close()

    def close(self) -> None:
        self._close_session()
        try:
            self.conn.close()
        except OSError:
            pass


class Server:
    """Accepts any number of clients; one at a time has a radio open."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, source_factory=None) -> None:
        self.listener = socket.create_server((host, port))
        self._source_factory = source_factory
        self._handlers: set[ClientHandler] = set()
        self._lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        return self.listener.getsockname()[:2]

    def claim(self, handler: ClientHandler) -> None:
        """`handler` is about to open a radio: disconnect every other client holding one,
        and wait until its radio is released, or the open finds it busy."""
        with self._lock:
            others = [h for h in self._handlers if h is not handler and h.session is not None]
        for other in others:
            print("replacing the previous client", file=sys.stderr, flush=True)
            try:
                other.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            for _ in range(60):
                if other.session is None:
                    break
                time.sleep(0.05)

    def _run(self, handler: ClientHandler) -> None:
        try:
            handler.run()
        finally:
            with self._lock:
                self._handlers.discard(handler)

    def serve_forever(self) -> None:
        while True:
            try:
                conn, peer = self.listener.accept()
            except OSError:
                return                                  # closed
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"client {peer[0]} connected", file=sys.stderr, flush=True)
            handler = ClientHandler(conn, self._source_factory, self.claim)
            with self._lock:
                self._handlers.add(handler)
            threading.Thread(target=self._run, args=(handler,), name="client",
                             daemon=True).start()

    def close(self) -> None:
        self.listener.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RGC_SDR network radio server")
    parser.add_argument("--bind", required=True,
                        help="address to listen on (the Tailscale address), optionally :port")
    args = parser.parse_args(argv)
    host, _, port = args.bind.rpartition(":") if args.bind.count(":") == 1 else (args.bind, "", "")
    server = Server(host or args.bind, int(port) if port else DEFAULT_PORT)
    print(f"serving on {server.address[0]}:{server.address[1]}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
