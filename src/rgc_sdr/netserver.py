"""The network radio server (P9, PLANNING.md 7q): runs on the Pi, serves its SDRs.

    python3 -m rgc_sdr.netserver --bind 100.x.y.z

A client lists the radios attached and opens one. It then receives spectrum lines of the
radio's whole span, and IQ for a window around the tuned frequency, decimated on the Pi to
a rate the network can carry; tuning inside the span only moves the window. Tuning, gains
and the rest are JSON requests. The
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
import numpy as np

from .device.remote_protocol import (
    DEFAULT_PORT, FRAME_JSON, PROTOCOL_VERSION, ProtocolError, caps_to_dict, encode_iq,
    encode_spectrum, iq_rate_for, json_frame, needs_recentre, parse_json, read_frame,
)
from .device.source import _Nco
from .dsp.decimate import Decimator, StreamDecimator
from .dsp.spectrum import SpectrumAnalyzer

#: IQ is sent in blocks of about this long: short enough for responsive audio, long
#: enough that the 12-byte headers and per-block work stay negligible.
BLOCK_S = 0.02
#: Of the radio's samples, at most this many seconds' worth go into one spectrum line,
#: as in the app (ui/main_window.py).
FRAME_INPUT_BUDGET = 0.6


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
    """One open radio: an IQ window around the tuned frequency, and spectrum lines of the
    radio's whole span, each sent by a thread of its own (PLANNING.md 7q, P9b)."""

    #: Sessions with a radio open, for `list_radios`.
    open_sessions: set["RadioSession"] = set()

    def __init__(self, send, driver: str, serial: str | None, radio_rate: float,
                 centre_hz: float, generation: int, source_factory=None) -> None:
        from .device.source import SoapyIQSource

        profile = profile_for(driver)
        if profile is None or profile.kind != "sdr":
            raise ValueError(f"not an SDR this server knows: {driver!r}")
        factory = source_factory or SoapyIQSource
        self._send = send
        self.driver, self.serial = driver, serial or ""
        self.source = factory(driver=driver, serial=serial or None, center_freq=centre_hz)
        self._lock = threading.Lock()
        self.generation = int(generation)
        #: The tuned frequency: the IQ window's centre, which need not be the radio's.
        self.tuned = float(self.source.center_freq)
        self.iq_rate, self.factor = iq_rate_for(self.source.sample_rate)
        self._decimator = StreamDecimator(self.factor)
        self._nco = _Nco(0.0, self.source.sample_rate)
        self._reader = None
        self.fft_size, self.fps, self.zoom = 4096, 25.0, 1
        self._analyzer = SpectrumAnalyzer(self.fft_size)
        if radio_rate:
            self._apply_rate(radio_rate)
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []
        RadioSession.open_sessions.add(self)

    # -- the radio's state, as sent to the client

    def state(self) -> dict:
        src, caps = self.source, self.source.caps
        gains = {g.name: src.get_gain(g.name) for g in caps.gain_elements}
        settings = {s.key: src.read_setting(s.key) for s in caps.settings}
        return {
            "freq": self.tuned, "radio_freq": src.center_freq, "radio_rate": src.sample_rate,
            "iq_rate": self.iq_rate, "factor": self.factor, "gains": gains,
            "agc": src.get_agc(), "bandwidth": getattr(src, "bandwidth", 0.0),
            "settings": settings, "ppm": getattr(src, "ppm", 0.0),
            # Where the radio's spike is, from the tuned frequency.
            "dc_spike_offset": src.center_freq + src.dc_spike_offset_hz - self.tuned,
            "generation": self.generation, "lost": getattr(self._reader, "lost", 0),
            "fft_size": self.fft_size, "zoom": self.zoom,
        }

    def caps(self) -> dict:
        return caps_to_dict(self.source.caps)

    # -- changes

    def _set_mixer(self) -> None:
        """Shift the tuned frequency to the IQ window's centre."""
        self._nco = _Nco(self.source.center_freq - self.tuned, self.source.sample_rate)

    def _apply_rate(self, radio_rate: float) -> None:
        self.source.set_sample_rate(float(radio_rate))
        self.iq_rate, self.factor = iq_rate_for(self.source.sample_rate)
        self._decimator = StreamDecimator(self.factor)
        # A rate change rebuilds the source's ring: read from the new one.
        self._reader = self.source.sequential_reader()
        if needs_recentre(self.source.center_freq, self.source.sample_rate, self.iq_rate,
                          self.tuned):
            self.source.set_center_freq(self.tuned)
        self._set_mixer()

    def set_rate(self, radio_rate: float, generation: int) -> None:
        with self._lock:
            self._apply_rate(radio_rate)
            self.generation = int(generation)

    def tune(self, hz: float, flush: bool, generation: int) -> None:
        """Move the IQ window; the radio only when the window would leave its span."""
        with self._lock:
            caps = self.source.caps
            self.tuned = caps.clamp_freq(float(hz))
            if needs_recentre(self.source.center_freq, self.source.sample_rate,
                              self.iq_rate, self.tuned):
                self.source.set_center_freq(self.tuned, flush=True)
                flush = True
            self._set_mixer()
            if flush:
                # Replaced, not reset: the pump may be using the old one outside the lock.
                self._decimator = StreamDecimator(self.factor)
                self._reader.skip_to_latest()
                self.generation = int(generation)

    def set_display(self, fft_size: int | None, fps: float | None, zoom: int | None) -> None:
        with self._lock:
            if fft_size and int(fft_size) != self.fft_size:
                self.fft_size = int(fft_size)
                self._analyzer = SpectrumAnalyzer(self.fft_size)
            if fps:
                self.fps = min(50.0, max(1.0, float(fps)))
            if zoom:
                self.zoom = max(1, int(zoom))

    def survey(self, centre_hz: float, seconds: float, lo_hz: float, hi_hz: float,
               generation: int | None = None) -> list:
        """A DMR/P25 survey (survey.py) of the radio's whole span about `centre_hz`: the
        radio moved there, `seconds` of its full-rate IQ captured, every channel in it
        decoded. Done here because the client has only the IQ window; the reports are
        a few hundred bytes."""
        from .survey import survey_iq

        seconds = min(max(float(seconds), 0.5), 10.0)
        self.tune(float(centre_hz), True,
                  self.generation + 1 if generation is None else int(generation))
        time.sleep(0.25)                                  # the front end settles
        reader = self.source.sequential_reader()
        wanted = int(seconds * self.source.sample_rate)
        chunks, have = [], 0
        deadline = time.monotonic() + seconds * 3 + 2
        while have < wanted and time.monotonic() < deadline:
            n = reader.available()
            if n == 0:
                time.sleep(0.02)
                continue
            piece = reader.read(min(n, wanted - have))
            chunks.append(piece)
            have += piece.size
        if not chunks:
            return []
        iq = np.concatenate(chunks)
        src = self.source
        return survey_iq(iq, src.sample_rate, src.center_freq, float(lo_hz), float(hi_hz),
                         src.dc_spike_offset_hz)

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
        if self._threads:
            return
        self.source.start()
        with self._lock:
            self._reader = self.source.sequential_reader()
            self._decimator.reset()
        self._running.set()
        for target, name in ((self._iq_pump, "iq-pump"), (self._spectrum_pump, "spectrum")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._running.clear()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        self.source.stop()

    def close(self) -> None:
        RadioSession.open_sessions.discard(self)
        self.stop()
        self.source.close()

    def _sendable(self, data: bytes) -> bool:
        try:
            self._send(data)
            return True
        except OSError:
            self._running.clear()            # the client has gone
            return False

    def _iq_pump(self) -> None:
        while self._running.is_set():
            # Only the read is locked: the filtering, most of the work, runs outside it,
            # so tuning and the spectrum thread never wait for it. A change made
            # meanwhile swaps in a new mixer and filter and a new generation, so this
            # block is filtered by the objects it was read for and tagged as stale.
            with self._lock:
                reader = self._reader
                block = int(self.source.sample_rate * BLOCK_S) // self.factor * self.factor
                if reader.available() < block:
                    iq = None
                else:
                    generation, nco, decimator = self.generation, self._nco, self._decimator
                    iq = reader.read(block)
            if iq is None:
                time.sleep(BLOCK_S / 4)
                continue
            nco.process(iq)
            out = decimator.process(iq)
            if out.size and not self._sendable(encode_iq(out, generation)):
                return

    def spectrum_line(self) -> bytes | None:
        """The newest spectrum line: the radio's whole span, or zoomed, span/zoom around
        the tuned frequency. None until there are samples enough."""
        with self._lock:
            analyzer, zoom = self._analyzer, self.zoom
            rate, radio_centre, tuned = (self.source.sample_rate, self.source.center_freq,
                                         self.tuned)
        decimator = Decimator(zoom)
        budget = int(FRAME_INPUT_BUDGET * rate)
        wanted = min(analyzer.samples_wanted(), max(analyzer.fft_size, budget // zoom))
        iq = self.source.read_latest(decimator.input_for_output(wanted))
        if iq.size < decimator.input_for_output(analyzer.fft_size):
            return None
        centre = radio_centre
        if zoom > 1:
            # Centred on the tuned frequency, as the app's own zoom is.
            n = np.arange(iq.size)
            iq = iq * np.exp(-2j * np.pi * (tuned - radio_centre) / rate * n).astype(np.complex64)
            iq = decimator.process(iq)
            centre = tuned
        if iq.size < analyzer.fft_size:
            return None
        return encode_spectrum(centre, rate / zoom, analyzer.psd_dbfs(iq))

    def _spectrum_pump(self) -> None:
        next_at = time.monotonic()
        while self._running.is_set():
            line = self.spectrum_line()
            if line is not None and not self._sendable(line):
                return
            next_at += 1.0 / self.fps
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()          # fell behind: do not try to catch up


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
                float(request.get("radio_rate", 0)), float(request.get("hz", 100e6)),
                int(request.get("generation", 0)), self._source_factory)
            self.session.set_display(request.get("fft_size"), request.get("fps"),
                                     request.get("zoom"))
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
            session.set_rate(float(request["radio_rate"]),
                             int(request.get("generation", session.generation)))
        elif op == "survey":
            reports = session.survey(request["centre_hz"], request.get("seconds", 3.0),
                                     request.get("lo_hz", 0.0), request.get("hi_hz", 1e12),
                                     request.get("generation"))
            self.reply(request, reports=reports, state=session.state())
            return
        elif op == "display":
            session.set_display(request.get("fft_size"), request.get("fps"),
                                request.get("zoom"))
        elif op != "state":
            session.apply(op, request)
        self.reply(request, state=session.state())

    def _close_session(self) -> None:
        # The radio is released *before* the session is cleared: a client opening the
        # radio waits for this one's session to clear (Server.claim), and clearing it
        # first let the open race the release and find the radio still in use
        # (an intermittent test failure, 2026-10-08).
        session = self.session
        if session is not None:
            session.close()
        self.session = None

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
