"""Diagnostic for trackpad gestures.

Sideways swipe-to-tune depends on how the platform chooses to deliver a two-finger
horizontal gesture, and that varies with the trackpad settings: it may arrive as a
horizontal wheel event, as a native pan, or be swallowed by the system's own
"swipe between pages". Rather than guess, run with `--debug-gestures` and swipe: every
relevant event is printed with its raw numbers.
"""

from __future__ import annotations

from PyQt6 import QtCore


class GestureLogger(QtCore.QObject):
    """Prints wheel and native-gesture events as they arrive."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._seen = 0

    def eventFilter(self, obj, event) -> bool:  # noqa: N802  (Qt naming)
        kind = event.type()
        if kind == QtCore.QEvent.Type.Wheel:
            angle, pixel = event.angleDelta(), event.pixelDelta()
            self._seen += 1
            print(
                f"[gesture {self._seen:4d}] Wheel on {type(obj).__name__:<18} "
                f"angleDelta=({angle.x():+5d},{angle.y():+5d})  "
                f"pixelDelta=({pixel.x():+5d},{pixel.y():+5d})  "
                f"phase={event.phase().name} inverted={event.inverted()}",
                flush=True,
            )
        elif kind == QtCore.QEvent.Type.NativeGesture:
            self._seen += 1
            try:
                name = event.gestureType().name
            except AttributeError:
                name = str(event.gestureType())
            print(
                f"[gesture {self._seen:4d}] NativeGesture on {type(obj).__name__:<10} "
                f"type={name} value={event.value():+.4f}",
                flush=True,
            )
        return False


def install(app) -> GestureLogger:
    logger = GestureLogger(app)
    app.installEventFilter(logger)
    print(
        "Gesture logging on. Swipe two fingers sideways over the spectrum, then\n"
        "up and down for comparison, and send the lines that appear.",
        flush=True,
    )
    return logger
