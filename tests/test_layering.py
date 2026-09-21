"""Enforce the layering rule from PLANNING.md section 5.

`dsp/` must not import Qt, and `device/` must not import Qt or `dsp`. Stated as an
invariant in the handover docs, so it is checked rather than trusted -- this is what
keeps both layers testable headless and independently swappable.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "rgc_sdr"

QT = ("PyQt6", "pyqtgraph", "PySide6")


def _module_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import; record it as a dotted suffix.
            names.add(("." * node.level) + (node.module or ""))
    return names


def _py_files(subdir: str) -> list[pathlib.Path]:
    return sorted((SRC / subdir).rglob("*.py"))


@pytest.mark.parametrize("path", _py_files("dsp"), ids=lambda p: p.name)
def test_dsp_does_not_import_qt(path):
    offenders = [i for i in _module_imports(path) if i.split(".")[0] in QT]
    assert not offenders, f"{path.name} imports Qt: {offenders}"


@pytest.mark.parametrize("path", _py_files("device"), ids=lambda p: p.name)
def test_device_does_not_import_qt(path):
    offenders = [i for i in _module_imports(path) if i.split(".")[0] in QT]
    assert not offenders, f"{path.name} imports Qt: {offenders}"


@pytest.mark.parametrize("path", _py_files("device"), ids=lambda p: p.name)
def test_device_does_not_import_dsp(path):
    offenders = [
        i for i in _module_imports(path) if i.endswith("dsp") or ".dsp." in i or i == "..dsp"
    ]
    assert not offenders, f"{path.name} imports dsp: {offenders}"


def test_dsp_is_importable_without_qt(monkeypatch):
    """dsp must load even if Qt is absent from the environment."""
    import sys

    for name in list(sys.modules):
        if name.split(".")[0] in QT:
            monkeypatch.setitem(sys.modules, name, None)
    import importlib

    for mod in ("src.rgc_sdr.dsp.spectrum", "src.rgc_sdr.dsp.waterfall"):
        importlib.reload(importlib.import_module(mod))
