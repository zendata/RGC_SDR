"""The Memory page: memories in groups, dragged between them (VK3RQ, 2026-10-07).

Groups are headings (LW, AM broadcast, FM broadcast, HF, VHF, UHF, Airband, Satellites,
Air Nav/Data, P25, DMR, DAB+ to start), each memory under its own with its frequency
and mode. Drag one or more memories onto a group, or onto any memory in it, to move
them. Double-click recalls; the memory in use (the last recalled) is shown in bold, and
Update saves the radio's settings now over it (VK3RQ, 2026-10-07). The panel edits the settings' groups itself and says so
(`changed`); recalling, saving and deleting go to the window, which knows the radio.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..settings import Settings

GROUP_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
NAME_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
#: Group heading colours, in turn.
GROUP_COLOURS = ("#ffe082", "#ffcc80", "#ef9a9a", "#ce93d8", "#9fa8da", "#81d4fa",
                 "#80cbc4", "#a5d6a7", "#e6ee9c", "#ffab91", "#bcaaa4", "#b0bec5")
MODE_NAMES = {"off": "", "nbfm": "NBFM", "wbfm": "WBFM", "am": "AM", "usb": "USB",
              "lsb": "LSB", "cw": "CW", "p25": "P25", "dab": "DAB"}


class MemoryTree(QtWidgets.QTreeWidget):
    """A tree whose memories can be dropped on a group, and nowhere else."""

    #: (memory names, group) when memories are dropped on a group.
    dropped = QtCore.pyqtSignal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["Memory", "Frequency", "Mode / decoder"])
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)

    def group_at(self, pos: QtCore.QPoint) -> str | None:
        item = self.itemAt(pos)
        if item is None:
            return None
        return item.data(0, GROUP_ROLE)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self.group_at(event.position().toPoint()) is None:
            event.ignore()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        group = self.group_at(event.position().toPoint())
        names = [i.data(0, NAME_ROLE) for i in self.selectedItems() if i.data(0, NAME_ROLE)]
        # The tree is rebuilt from the settings: Qt must not move the items itself.
        event.setDropAction(QtCore.Qt.DropAction.IgnoreAction)
        event.accept()
        if group and names:
            self.dropped.emit(names, group)


class MemoryPanel(QtWidgets.QWidget):
    HINT = "Drag memories onto a group to move them"
    OWN_FILTER = "RGC_SDR memories (*.csv)"
    CHIRP_FILTER = "CHIRP (*.csv)"

    recallRequested = QtCore.pyqtSignal(str)
    #: Save the radio's current settings: (name, group or "" for the automatic one).
    saveRequested = QtCore.pyqtSignal(str, str)
    #: Save the radio's current settings over the memory in use, by name.
    updateRequested = QtCore.pyqtSignal(str)
    deleteRequested = QtCore.pyqtSignal(str)
    #: The settings' groups or names changed here: save them.
    changed = QtCore.pyqtSignal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        self.recall_button = self._button(buttons, "Recall", self._recall,
                                          "Tune to the selected memory (or double-click it)")
        self.update_button = self._button(buttons, "Update", self._update,
                                          "Save the radio's settings now over the memory in "
                                          "use (in bold), for this radio")
        self.save_button = self._button(buttons, "Save current…", self._save,
                                        "Store the radio's settings now, in the selected group")
        self.rename_button = self._button(buttons, "Rename…", self._rename,
                                          "Rename the selected memory or group")
        self.delete_button = self._button(buttons, "Delete", self._delete,
                                          "Delete the selected memories, or an empty group")
        buttons.addSpacing(20)
        self.group_button = self._button(buttons, "New group…", self._new_group,
                                         "Add a memory group")
        buttons.addSpacing(20)
        self.import_button = self._button(
            buttons, "Import…", self._import,
            "Add memories from a CSV file: this app's own, or CHIRP's (for radios CHIRP\n"
            "programs). A memory with a name already used is replaced.")
        self.export_button = self._button(
            buttons, "Export…", self._export,
            "Save the selected memories (all, if none is selected) as CSV: this app's own,\n"
            "which keeps everything, or CHIRP's, for programming a radio")
        buttons.addStretch(1)
        self.hint = QtWidgets.QLabel(self.HINT)
        self.hint.setStyleSheet("color: #888;")
        buttons.addWidget(self.hint)
        outer.addLayout(buttons)
        self.tree = MemoryTree()
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.dropped.connect(self._on_dropped)
        outer.addWidget(self.tree, 1)
        self._expanded: set[str] | None = None
        #: The memory last recalled, which Update saves over ("" for none).
        self.in_use = ""
        self.refresh()

    @staticmethod
    def _button(row, text, slot, tip) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tip)
        button.clicked.connect(slot)
        row.addWidget(button)
        return button

    # -- contents
    def refresh(self) -> None:
        """Rebuild from the settings, keeping which groups are open."""
        if self._expanded is not None:
            self._expanded = {self.tree.topLevelItem(i).data(0, GROUP_ROLE)
                              for i in range(self.tree.topLevelItemCount())
                              if self.tree.topLevelItem(i).isExpanded()}
        selected = {i.data(0, NAME_ROLE) for i in self.tree.selectedItems()}
        self.tree.clear()
        font = self.tree.font()
        font.setBold(True)
        for k, group in enumerate(self.settings.memory_groups):
            members = self.settings.memories_in(group)
            head = QtWidgets.QTreeWidgetItem([f"{group}  ({len(members)})", "", ""])
            head.setData(0, GROUP_ROLE, group)
            head.setFont(0, font)
            colour = QtGui.QColor(GROUP_COLOURS[k % len(GROUP_COLOURS)])
            for col in range(3):
                head.setBackground(col, colour)
                head.setForeground(col, QtGui.QColor("black"))
            head.setFlags(head.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            self.tree.addTopLevelItem(head)
            for memory in members:
                snap = memory.snapshot
                what = MODE_NAMES.get(snap.mode, snap.mode.upper())
                if snap.decoder:
                    what = f"{what} / {snap.decoder.upper()}".strip(" /")
                item = QtWidgets.QTreeWidgetItem(
                    [memory.name, f"{snap.freq_hz / 1e6:.4f} MHz", what])
                item.setData(0, NAME_ROLE, memory.name)
                item.setData(0, GROUP_ROLE, group)
                if memory.name == self.in_use:
                    for col in range(3):
                        item.setFont(col, font)
                    item.setToolTip(0, "In use: Update saves over this one")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)
                head.addChild(item)
                if memory.name in selected:
                    item.setSelected(True)
            head.setExpanded(self._expanded is not None and group in self._expanded)
        if self._expanded is None:
            self._expanded = set()
        for col in range(3):
            self.tree.resizeColumnToContents(col)
        has = bool(self.settings.memories)
        for button in (self.recall_button, self.rename_button, self.delete_button):
            button.setEnabled(has)
        self.update_button.setEnabled(self.settings.get_memory(self.in_use) is not None)
        self.update_button.setToolTip(
            f"Save the radio's settings now over \u201c{self.in_use}\u201d, for this radio"
            if self.update_button.isEnabled() else "Recall a memory first: Update saves over it")

    def set_in_use(self, name: str) -> None:
        """Mark `name` as the memory in use ("" for none)."""
        memory = self.settings.get_memory(name)
        self.in_use = memory.name if memory is not None else ""
        self.refresh()

    def collapse_all(self) -> None:
        """Every group closed, so any of them is reached without scrolling past the
        others (VK3RQ, 2026-10-09): done each time the Memory tab is opened."""
        self.tree.collapseAll()
        self._expanded = set()

    def selected_names(self) -> list[str]:
        return [i.data(0, NAME_ROLE) for i in self.tree.selectedItems() if i.data(0, NAME_ROLE)]

    def selected_group(self) -> str:
        """The group of the selection (a heading or a memory), or ""."""
        items = self.tree.selectedItems()
        return (items[0].data(0, GROUP_ROLE) or "") if items else ""

    # -- actions
    def move(self, names: list[str], group: str) -> None:
        for name in names:
            self.settings.move_memory(name, group)
        self.refresh()
        self.changed.emit()

    def _on_dropped(self, names: list, group: str) -> None:
        self.move(list(names), group)

    def _on_double_click(self, item, _column) -> None:
        name = item.data(0, NAME_ROLE)
        if name:
            self.recallRequested.emit(name)

    def _recall(self) -> None:
        names = self.selected_names()
        if names:
            self.recallRequested.emit(names[0])

    def _update(self) -> None:
        if self.settings.get_memory(self.in_use) is not None:
            self.updateRequested.emit(self.in_use)

    def _save(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save memory", "Name for this memory:")
        if ok and name.strip():
            self.saveRequested.emit(name.strip(), self.selected_group())

    def _rename(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        name, group = items[0].data(0, NAME_ROLE), items[0].data(0, GROUP_ROLE)
        old = name or group
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename", "New name:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        done = (self.settings.rename_memory(name, new) if name
                else self.settings.rename_group(group, new))
        if done and name and name == self.in_use:
            self.in_use = new.strip()
        if done:
            self.refresh()
            self.changed.emit()

    def _delete(self) -> None:
        names = self.selected_names()
        if names:
            text = f'Delete "{names[0]}"?' if len(names) == 1 else f"Delete {len(names)} memories?"
            answer = QtWidgets.QMessageBox.question(self, "Delete", text)
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                for name in names:
                    self.deleteRequested.emit(name)
            return
        group = self.selected_group()
        if group and self.settings.remove_group(group):
            self.refresh()
            self.changed.emit()

    def _say(self, text: str) -> None:
        """A note in place of the hint, for a few seconds."""
        self.hint.setText(text)
        QtCore.QTimer.singleShot(8000, lambda: self.hint.setText(self.HINT))

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import memories", "", "CSV (*.csv);;All files (*)")
        if path:
            self.import_from(path)

    def import_from(self, path) -> int:
        from ..memory_io import import_any, is_chirp, merge

        try:
            memories = import_any(path, self.settings.memory_groups)
            chirp = is_chirp(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self._say(f"could not import: {exc}")
            return 0
        count = merge(self.settings, memories)
        self.refresh()
        self.changed.emit()
        self._say(f"imported {count} memories{' from CHIRP' if chirp else ''}")
        return count

    def _export(self) -> None:
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export memories", "memories.csv",
            f"{self.OWN_FILTER};;{self.CHIRP_FILTER}")
        if path:
            self.export_to(path, chirp=chosen == self.CHIRP_FILTER)

    def export_to(self, path, chirp: bool = False) -> int:
        from ..memory_io import export_any

        names = set(self.selected_names())
        memories = [m for m in self.settings.memories if not names or m.name in names]
        try:
            count = export_any(memories, path, chirp=chirp)
        except OSError as exc:
            self._say(f"could not export: {exc}")
            return 0
        skipped = len(memories) - count
        self._say(f"exported {count} memories" +
                  (f" ({skipped} digital ones CHIRP cannot hold left out)" if skipped else ""))
        return count

    def _new_group(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New group", "Name for the group:")
        if ok and self.settings.add_group(name):
            self.refresh()
            self.changed.emit()
