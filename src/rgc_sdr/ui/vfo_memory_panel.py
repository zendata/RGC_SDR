"""The IC-705's VFO/MEMORY indicator and screen, on the Mac.

On the radio, right of the frequency, "VFO A", "VFO B" or "MEMO" with the channel
number; touching it opens the VFO/MEMORY screen (Basic Manual p. 3-1):

    VFO   MEMO    CALL    GROUP   A/B
    MW    M-CLR   SELECT  M->VFO  (back)

Laid out and behaving the same way here. As on the radio, MW, M-CLR and M->VFO act on a
one-second hold (a right-click does too), and A/B held copies A to B.

The radio does not report its VFO/memory state over CI-V, so the indicator shows what
was last chosen from the Mac, and "VFO/MEMO ?" until then.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ..device.ic705_memory import CALL_GROUP, MemoryChannel
from .function_panel import FunctionButton

_KEY = ("QPushButton { color: #e6edf3; background: #2a3342; border: 1px solid #3a4658;"
        " border-radius: 4px; padding: 6px; min-width: 76px; min-height: 30px; }"
        " QPushButton:disabled { color: #5c6773; background: #1a2029; }")
_ON = ("QPushButton { color: #10141a; background: #4fc3f7; border: 1px solid #4fc3f7;"
       " border-radius: 4px; padding: 6px; min-width: 76px; min-height: 30px;"
       " font-weight: bold; }")
_INDICATOR = ("QPushButton { color: #e6edf3; background: transparent; border: 1px solid"
              " #3a4658; border-radius: 4px; padding: 1px 6px; text-align: left; }"
              " QPushButton:hover { border-color: #4fc3f7; }")


def indicator_text(src) -> tuple[str, str]:
    """(top line, second line) for the indicator: ("VFO A", ""), ("MEMO", "00-07 40m QRP")."""
    mode = getattr(src, "vfo_mode", None)
    if mode in ("A", "B"):
        return f"VFO {mode}", ""
    memo = getattr(src, "memo_contents", None)
    if mode == "CALL":
        label = MemoryChannel(CALL_GROUP, getattr(src, "call_channel", 0)).label
    elif mode == "MEMO":
        label = MemoryChannel(getattr(src, "memo_group", 0),
                              getattr(src, "memo_channel", 0)).label
    else:
        return "VFO/MEMO ?", ""
    name = memo.name if isinstance(memo, MemoryChannel) else \
        "(blank)" if memo == "blank" else ""
    return ("CALL" if mode == "CALL" else "MEMO"), f"{label} {name}".strip()


class VfoMemoryPanel(QtWidgets.QFrame):
    """The VFO/MEMORY screen: two rows of five keys."""

    def __init__(self, source, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self._source = source
        self.setStyleSheet("QFrame { background: #11161d; border: 1px solid #2c3645; }")
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(6)
        title = QtWidgets.QLabel("VFO/MEMORY")
        title.setStyleSheet("color: #8b98a8; border: none;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(title, 0, 0, 1, 5)
        self.keys: dict[str, FunctionButton] = {}
        layout = (("VFO", "MEMO", "CALL", "GROUP", "A/B"),
                  ("MW", "M-CLR", "SELECT", "M→VFO", "↩"))
        tips = {
            "VFO": "VFO mode", "MEMO": "Memory mode", "CALL": "Call channel mode",
            "GROUP": "Choose the memory group", "A/B": "Swap VFO A/B; hold: A=B",
            "MW": "Hold 1 s: write the VFO into the selected memory channel",
            "M-CLR": "Hold 1 s: clear the selected memory channel",
            "SELECT": "Mark this channel as select channel ★1-★3, or clear it",
            "M→VFO": "Hold 1 s: copy the memory channel to the VFO",
            "↩": "Back",
        }
        for r, row in enumerate(layout, start=1):
            for c, name in enumerate(row):
                key = FunctionButton(name)
                key.setToolTip(tips[name])
                key.setStyleSheet(_KEY)
                grid.addWidget(key, r, c)
                self.keys[name] = key
        k = self.keys
        k["VFO"].clicked.connect(lambda: self._do(self._source.select_vfo))
        k["MEMO"].clicked.connect(lambda: self._do(self._source.select_memory))
        k["CALL"].clicked.connect(lambda: self._do(self._source.select_call))
        k["GROUP"].clicked.connect(self._choose_group)
        k["A/B"].clicked.connect(lambda: self._do(self._source.swap_vfo))
        k["A/B"].long_pressed.connect(lambda: self._do(self._source.equalize_vfo))
        # The radio needs a one-second touch for these; a plain click only says so.
        for name, action in (("MW", self._source.memory_write),
                             ("M-CLR", self._source.memory_clear),
                             ("M→VFO", self._source.memory_to_vfo)):
            k[name].long_pressed.connect(lambda a=action: self._do(a, close=True))
            k[name].clicked.connect(lambda n=name: self.note.setText(f"Hold {n} for a second"))
        k["SELECT"].clicked.connect(self._cycle_select)
        k["↩"].clicked.connect(self.close)
        self.note = QtWidgets.QLabel("")
        self.note.setStyleSheet("color: #8b98a8; border: none;")
        grid.addWidget(self.note, 3, 0, 1, 5)
        self.sync()

    def _do(self, action, close: bool = False) -> None:
        action()
        self.sync()
        if close:
            self.close()

    def _choose_group(self) -> None:
        menu = QtWidgets.QMenu(self)
        for g in range(100):
            act = menu.addAction(f"Group {g:02d}")
            act.setData(g)
        chosen = menu.exec(self.keys["GROUP"].mapToGlobal(
            QtCore.QPoint(0, self.keys["GROUP"].height())))
        if chosen is not None:
            self._do(lambda: self._source.select_memory(group=int(chosen.data()), channel=0))

    def _cycle_select(self) -> None:
        memo = getattr(self._source, "memo_contents", None)
        current = memo.select if isinstance(memo, MemoryChannel) else 0
        self._do(lambda: self._source.set_select((current + 1) % 4))
        self.note.setText("Select channel: " + ("off" if current == 3 else f"★{current + 1}"))

    def sync(self) -> None:
        mode = getattr(self._source, "vfo_mode", None)
        in_memory = mode in ("MEMO", "CALL")
        lit = {"VFO": mode in ("A", "B"), "MEMO": mode == "MEMO", "CALL": mode == "CALL"}
        for name, key in self.keys.items():
            key.setStyleSheet(_ON if lit.get(name) else _KEY)
        for name in ("GROUP", "M-CLR", "SELECT", "M→VFO"):
            self.keys[name].setEnabled(mode == "MEMO" or (in_memory and name != "GROUP"))
        self.keys["A/B"].setEnabled(mode not in ("MEMO", "CALL"))
