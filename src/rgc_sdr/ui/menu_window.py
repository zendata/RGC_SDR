"""The IC-705's SET menu (and the other menus' SET screens) in a Mac window.

Built from `ic705_menu.MENU_ITEMS`, which is generated from Icom's CI-V reference guide,
so every item number, range and choice is Icom's. Values are read from the radio in the
background when the window opens -- 360 items take a few seconds -- and a change is sent
as soon as it is made. A few items the app itself depends on are shown but locked.

Items whose data the guide describes elsewhere (filters, edges, symbols, dates) are
shown read-only, as the radio's raw value.
"""

from __future__ import annotations

import threading

from PyQt6 import QtCore, QtGui, QtWidgets

from ..device.ic705_menu import MENU_ITEMS
from ..device.icom import MENU_LOCKED


class ScaledSpin(QtWidgets.QSpinBox):
    """A spin box over the radio's codes that shows them in the radio's units."""

    def __init__(self, low: int, high: int, scale) -> None:
        super().__init__()
        self._scale = scale
        self.setRange(low, high)
        self.setKeyboardTracking(False)

    def _value_of(self, code: int) -> float:
        a, va, b, vb, _unit = self._scale
        return va if b == a else va + (code - a) * (vb - va) / (b - a)

    def textFromValue(self, code: int) -> str:  # noqa: N802  (Qt naming)
        if self._scale is None:
            return str(code)
        value = self._value_of(code)
        unit = self._scale[4]
        return f"{value:.0f}{unit}" if float(value).is_integer() else f"{value:.1f}{unit}"

    def valueFromText(self, text: str) -> int:  # noqa: N802  (Qt naming)
        if self._scale is None:
            return int(text)
        a, va, b, vb, unit = self._scale
        try:
            number = float(text.replace(unit, "").strip() if unit else text.strip())
        except ValueError:
            return self.value()
        code = round(a + (number - va) * (b - a) / (vb - va)) if vb != va else a
        return max(self.minimum(), min(self.maximum(), code))

    def validate(self, text, pos):  # noqa: D102
        return (QtGui.QValidator.State.Acceptable, text, pos) if self._scale else \
            super().validate(text, pos)


class _Reader(QtCore.QObject):
    """Reads the items off the UI thread and reports each value as it arrives."""

    value_read = QtCore.pyqtSignal(int, object)
    finished = QtCore.pyqtSignal()

    def __init__(self, source, numbers) -> None:
        super().__init__()
        self._source = source
        self._numbers = list(numbers)
        self._stop = threading.Event()

    def run(self) -> None:
        for number in self._numbers:
            if self._stop.is_set():
                break
            self.value_read.emit(number, self._source.read_menu(number))
        self.finished.emit()

    def stop(self) -> None:
        self._stop.set()


class MenuWindow(QtWidgets.QDialog):
    """Every SET-menu item: a tree by menu, a search box, and an editor per item."""

    def __init__(self, source, parent=None, read_now: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("IC-705 menus")
        self.resize(720, 640)
        self._source = source
        self._rows: dict[int, tuple] = {r[0]: r for r in MENU_ITEMS}
        self.editors: dict[int, QtWidgets.QWidget] = {}
        self._items: dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the menus (e.g. \"beep\", \"USB\", \"VOX\")")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter)
        top.addWidget(self.search, 1)
        self.reload = QtWidgets.QPushButton("Read from radio")
        self.reload.clicked.connect(self.read_all)
        top.addWidget(self.reload)
        layout.addLayout(top)
        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Setting", "Value"])
        self.tree.setColumnWidth(0, 400)
        layout.addWidget(self.tree, 1)
        self._build()

        self._thread: QtCore.QThread | None = None
        self._reader: _Reader | None = None
        if read_now:
            self.read_all()

    # -- building -------------------------------------------------------------------------

    def _branch(self, path: str) -> QtWidgets.QTreeWidgetItem:
        node = self.tree.invisibleRootItem()
        for part in path.split(" > "):
            for i in range(node.childCount()):
                if node.child(i).text(0) == part and node.child(i).data(0, 256) is None:
                    node = node.child(i)
                    break
            else:
                child = QtWidgets.QTreeWidgetItem([part, ""])
                node.addChild(child)
                node = child
        return node

    def _build(self) -> None:
        for row in MENU_ITEMS:
            number, path, title, digits, low, high, choices, scale, readonly = row
            item = QtWidgets.QTreeWidgetItem([title, ""])
            item.setData(0, 256, number)                 # Qt.UserRole
            tip = f"Menu item {number:04d}"
            if number in MENU_LOCKED:
                tip += f"\n{MENU_LOCKED[number]}"
            item.setToolTip(0, tip)
            self._branch(path).addChild(item)
            self._items[number] = item
            editor = self._editor(row)
            if editor is not None:
                editor.setToolTip(tip)
                self.tree.setItemWidget(item, 1, editor)
                self.editors[number] = editor

    def _editor(self, row) -> QtWidgets.QWidget | None:
        number, _path, _title, digits, low, high, choices, scale, readonly = row
        if readonly or not digits:
            return None                                  # shown as text in column 1
        if choices:
            combo = QtWidgets.QComboBox()
            for code, label in sorted(choices.items()):
                combo.addItem(label, code)
            combo.setEnabled(False)                      # until its value has been read
            combo.currentIndexChanged.connect(
                lambda _i, n=number, c=combo: self._changed(n, c.currentData()))
            return combo
        spin = ScaledSpin(low, high, scale)
        spin.setEnabled(False)
        spin.valueChanged.connect(lambda v, n=number: self._changed(n, v))
        return spin

    # -- values -----------------------------------------------------------------------------

    def read_all(self) -> None:
        """Read every item from the radio, in the background."""
        if self._thread is not None:
            return
        self._thread = QtCore.QThread(self)
        self._reader = _Reader(self._source, [r[0] for r in MENU_ITEMS])
        self._reader.moveToThread(self._thread)
        self._thread.started.connect(self._reader.run)
        self._reader.value_read.connect(self.show_value)
        self._reader.finished.connect(self._read_done)
        self._count = 0
        self.reload.setEnabled(False)
        self._thread.start()

    def _read_done(self) -> None:
        self.status.setText(f"Read {self._count} of {len(MENU_ITEMS)} items from the radio.")
        self.reload.setEnabled(True)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = self._reader = None

    def show_value(self, number: int, value) -> None:
        """Put a value read from the radio into its row (without sending it back)."""
        self._count = getattr(self, "_count", 0) + 1
        self.status.setText(f"Reading from the radio … {self._count}/{len(MENU_ITEMS)}")
        if value is None:
            return
        row, item = self._rows[number], self._items[number]
        editor = self.editors.get(number)
        self._loading = True
        try:
            if isinstance(editor, QtWidgets.QComboBox):
                index = editor.findData(value)
                if index < 0:                            # a code the guide did not list
                    editor.addItem(str(value), value)
                    index = editor.count() - 1
                editor.setCurrentIndex(index)
                editor.setEnabled(number not in MENU_LOCKED)
            elif isinstance(editor, ScaledSpin):
                if not editor.minimum() <= int(value) <= editor.maximum():
                    # The radio knows better than the guide (REF Adjust reads 260 of 255).
                    editor.setRange(min(editor.minimum(), int(value)),
                                    max(editor.maximum(), int(value)))
                editor.setValue(int(value))
                editor.setEnabled(number not in MENU_LOCKED)
            else:
                digits = row[3] or 2
                item.setText(1, f"{value:0{digits}d}")
        finally:
            self._loading = False

    def _changed(self, number: int, value) -> None:
        if self._loading or value is None:
            return
        try:
            self._source.write_menu(number, int(value), self._rows[number][3])
        except PermissionError as exc:
            self.status.setText(str(exc))

    # -- search -----------------------------------------------------------------------------

    def filter(self, text: str) -> None:
        """Show only items whose title or menu path contains every word typed."""
        words = text.lower().split()
        for number, item in self._items.items():
            row = self._rows[number]
            hay = f"{row[1]} {row[2]} {number:04d}".lower()
            item.setHidden(not all(w in hay for w in words))
        self._hide_empty(self.tree.invisibleRootItem())
        if words:
            self.tree.expandAll()

    def _hide_empty(self, node) -> bool:
        """Hide branches with nothing visible under them. True if `node` shows anything."""
        visible = False
        for i in range(node.childCount()):
            child = node.child(i)
            if child.data(0, 256) is not None:
                visible |= not child.isHidden()
            else:
                shown = self._hide_empty(child)
                child.setHidden(not shown)
                visible |= shown
        return visible

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        if self._reader is not None:
            self._reader.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
