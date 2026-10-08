"""A CAT server speaking hamlib's rigctld protocol (P13, PLANNING.md 7u).

WSJT-X, fldigi, JTDX, loggers and most ham software can control a radio through
`rigctld`, hamlib's network daemon, by choosing hamlib's "NET rigctl" radio and an
address. This server answers that protocol for RGC_SDR, so those programs read and set
the frequency and mode of whatever radio the app is using -- an SDR, a network radio or
the IC-705 -- with no hamlib driver for it.

The server's threads never touch the window. What they read comes from a snapshot the
window publishes (`publish`), and what they set is queued for the window to apply on
its own thread (`take_commands`). A set is answered at once, as rigctld does.

Transmit is not offered: `T 1` is refused (RPRT -11, "not available"). Keying from
another program is a new transmit path, and the first key-up of any new transmit path is
the operator's to make (CLAUDE.md); it can be added once that has been done.

The reply layouts follow hamlib 4.7's own rigctld, read from its dummy radio
(`rigctld -m 1`), and are checked against hamlib's `rigctl -m 2` in the tests.

No Qt; plain sockets and threads.
"""

from __future__ import annotations

import queue
import socket
import threading

#: rigctld's port.
DEFAULT_PORT = 4532
#: hamlib error codes used here.
RIG_OK, RIG_EINVAL, RIG_ENIMPL, RIG_ENAVAIL = 0, -1, -4, -11

#: hamlib mode bits (rig.h), for the dump_state ranges and filters.
MODE_BITS = {"AM": 0x1, "CW": 0x2, "USB": 0x4, "LSB": 0x8, "RTTY": 0x10, "FM": 0x20,
             "WFM": 0x40, "CWR": 0x80, "SAM": 0x10000}
ALL_MODES = sum(MODE_BITS.values())

#: The app's modes as hamlib names them.
TO_HAMLIB = {"usb": "USB", "lsb": "LSB", "cw": "CW", "am": "AM", "sam": "SAM",
             "nbfm": "FM", "wbfm": "WFM", "p25": "FM", "dab": "WFM", "off": "USB",
             # The IC-705's own modes.
             "fm": "FM", "wfm": "WFM", "rtty": "RTTY"}
#: hamlib's modes as the app has them: the packet modes (what WSJT-X asks for) are
#: their sidebands; the app has no RTTY demodulator, so RTTY listens in USB.
FROM_HAMLIB = {"USB": "usb", "PKTUSB": "usb", "LSB": "lsb", "PKTLSB": "lsb",
               "CW": "cw", "CWR": "cw", "AM": "am", "SAM": "sam", "AMS": "sam",
               "FM": "nbfm", "FMN": "nbfm", "PKTFM": "nbfm", "WFM": "wbfm",
               "RTTY": "usb", "RTTYR": "lsb"}

#: Long command names -> the short letters this server handles them as.
LONG_NAMES = {
    "get_freq": "f", "set_freq": "F", "get_mode": "m", "set_mode": "M",
    "get_vfo": "v", "set_vfo": "V", "get_ptt": "t", "set_ptt": "T",
    "get_split_vfo": "s", "set_split_vfo": "S", "get_split_freq": "i",
    "set_split_freq": "I", "get_split_mode": "x", "set_split_mode": "X",
    "get_level": "l", "set_level": "L", "get_rit": "j", "set_rit": "J",
    "get_xit": "z", "set_xit": "Z", "get_info": "_", "quit": "q",
}


#: What follows the fixed part of dump_state for a client that asked \chk_vfo first
#: (hamlib 4.7's rigctld does the same): capabilities as key=value lines, then "done".
#: ptt_type 0 is RIG_PTT_NONE: no transmit from here.
EXTENDED_STATE = (
    "vfo_ops=0x0", "ptt_type=0x0", "targetable_vfo=0x0", "has_set_vfo=1",
    "has_get_vfo=1", "has_set_freq=1", "has_get_freq=1", "has_set_conf=0",
    "has_get_conf=0", "has_power2mW=0", "has_mW2power=0", "has_get_ant=0",
    "has_set_ant=0", "timeout=0", "rig_model=2", "rigctld_version=RGC_SDR", "done",
)


def dump_state(low_hz: float, high_hz: float, extended: bool = False) -> str:
    """rigctld's capabilities answer, laid out as hamlib 4.7's own (protocol 1); with
    `extended`, the key=value lines a newer client reads after it."""
    lines = [
        "1",                     # protocol version
        "2",                     # radio model: NET rigctl
        "0",                     # ITU region, unused
        f"{low_hz:.6f} {high_hz:.6f} 0x{ALL_MODES:x} -1 -1 0x3 0x1",
        "0 0 0 0 0 0 0",
        "0 0 0 0 0 0 0",         # no transmit ranges: receive only here
        f"0x{ALL_MODES:x} 1",    # tuning steps
        "0 0",
        f"0x{MODE_BITS['USB'] | MODE_BITS['LSB']:x} 2700",       # filters
        f"0x{MODE_BITS['CW']:x} 500",
        f"0x{MODE_BITS['AM'] | MODE_BITS['SAM']:x} 9000",
        f"0x{MODE_BITS['FM']:x} 12500",
        f"0x{MODE_BITS['WFM']:x} 200000",
        "0 0",
        "0",                     # max RIT
        "0",                     # max XIT
        "0",                     # max IF shift
        "0",                     # announces
        "0 ",                    # preamps: none (hamlib ends the list with 0)
        "0 ",                    # attenuators: none
        "0x0", "0x0",            # get, set functions
        "0x0", "0x0",            # get, set levels
        "0x0", "0x0",            # get, set parameters
    ]
    if extended:
        lines += list(EXTENDED_STATE)
    return "\n".join(lines) + "\n"


class RigctlServer:
    """Serves the rigctld protocol on `host:port` (localhost by default)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.listener = socket.create_server((host, port))
        self._state = {"freq": 0.0, "mode": "usb", "passband": 2700.0,
                       "low": 0.0, "high": 6e9}
        self._commands: queue.Queue = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self.clients = 0
        threading.Thread(target=self._accept, name="cat-accept", daemon=True).start()

    @property
    def address(self) -> tuple[str, int]:
        return self.listener.getsockname()[:2]

    # -- the window's side

    def publish(self, freq_hz: float, mode: str, passband_hz: float,
                low_hz: float = 0.0, high_hz: float = 6e9) -> None:
        """What the radio is doing now (called on the window's thread)."""
        self._state = {"freq": float(freq_hz), "mode": mode, "passband": float(passband_hz),
                       "low": float(low_hz), "high": float(high_hz),
                       "hamlib_mode": self._state.get("hamlib_mode")}

    def take_commands(self) -> list[tuple]:
        """Changes asked for since last time: ("freq", hz) and ("mode", mode, passband
        or None)."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._running.clear()
        try:
            self.listener.close()
        except OSError:
            pass

    # -- the network's side

    def _accept(self) -> None:
        while self._running.is_set():
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), name="cat-client",
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        self.clients += 1
        buffer = b""
        session = {"chk_vfo": False}
        try:
            while self._running.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    reply = self.handle(line.decode("ascii", "replace").strip(), session)
                    if reply:
                        conn.sendall(reply.encode("ascii"))
                    if session.get("quit"):
                        return
        except OSError:
            pass
        finally:
            self.clients -= 1
            conn.close()

    def handle(self, line: str, session: dict | None = None) -> str | None:
        """One command line -> its reply ("" for none, None to close). `session` is
        the connection's own state: whether it asked \\chk_vfo, as newer clients do."""
        session = session if session is not None else {}
        if not line:
            return ""
        parts = line.split()
        command, args = parts[0], parts[1:]
        if command.startswith("\\"):
            name = command[1:]
            if name == "dump_state":
                return dump_state(self._state["low"], self._state["high"],
                                  extended=session.get("chk_vfo", False))
            if name == "chk_vfo":
                session["chk_vfo"] = True
                return "0\n"
            if name in ("get_powerstat",):
                return "1\n"
            if name in ("get_lock_mode",):
                return f"0\nRPRT {RIG_OK}\n"           # value, then a report (hamlib 4.7)
            if name in ("set_powerstat", "set_lock_mode"):
                return f"RPRT {RIG_OK}\n"
            command = LONG_NAMES.get(name, "")
            if not command:
                return f"RPRT {RIG_ENIMPL}\n"
        return self._short(command, args, session)

    def _short(self, command: str, args: list[str], session: dict | None = None) -> str:
        state = self._state
        mode_name = TO_HAMLIB.get(state["mode"], "USB")
        # A client that set a packet mode (PKTUSB) reads it back as it set it, as from a
        # radio: WSJT-X checks.
        asked = state.get("hamlib_mode")
        if asked and FROM_HAMLIB.get(asked) == state["mode"]:
            mode_name = asked
        if command in ("q", "Q"):
            if session is not None:
                session["quit"] = True
            return f"RPRT {RIG_OK}\n"
        if command == "f":
            return f"{int(round(state['freq']))}\n"
        if command == "F":
            try:
                hz = float(args[0])
            except (IndexError, ValueError):
                return f"RPRT {RIG_EINVAL}\n"
            self._commands.put(("freq", hz))
            self._state = dict(state, freq=hz)        # read back as set, as a radio does
            return f"RPRT {RIG_OK}\n"
        if command in ("m", "x"):
            return f"{mode_name}\n{int(round(state['passband']))}\n"
        if command in ("M", "X"):
            if not args or args[0].upper() not in FROM_HAMLIB:
                return f"RPRT {RIG_EINVAL}\n"
            mode = FROM_HAMLIB[args[0].upper()]
            passband = None
            if len(args) > 1:
                try:
                    width = float(args[1])
                except ValueError:
                    return f"RPRT {RIG_EINVAL}\n"
                passband = width if width > 0 else None    # 0 default, -1 unchanged
            if command == "M":
                self._commands.put(("mode", mode, passband))
                self._state = dict(state, mode=mode, hamlib_mode=args[0].upper(),
                                   passband=passband or state["passband"])
            return f"RPRT {RIG_OK}\n"
        if command == "v":
            return "VFOA\n"
        if command == "t":
            return "0\n"
        if command == "T":
            if args and args[0] not in ("0",):
                return f"RPRT {RIG_ENAVAIL}\n"        # see the module docstring
            return f"RPRT {RIG_OK}\n"
        if command == "s":
            return "0\nVFOA\n"
        if command == "i":
            return f"{int(round(state['freq']))}\n"
        if command in ("j", "z"):
            return "0\n"
        if command == "l":
            return "0\n"
        if command == "_":
            return "RGC_SDR\n"
        if command in ("V", "S", "I", "J", "Z", "L"):
            return f"RPRT {RIG_OK}\n"
        return f"RPRT {RIG_ENIMPL}\n"
