"""Rolling waterfall history and bin-to-pixel reduction.

Pure NumPy, no Qt -- the UI layer wraps this in a pyqtgraph image item.
"""

from __future__ import annotations

import numpy as np

FILL_DBFS = -140.0  # below the measured noise floor (~-111 dBFS), so new panes look empty


def reduce_max(values: np.ndarray, width: int) -> np.ndarray:
    """Max-pool `values` down to `width` bins.

    Max, not mean: when the FFT has more bins than the widget has pixels, averaging
    dilutes a narrow carrier into the surrounding noise and it visually disappears.

    Uses `maximum.reduceat` over edges at `arange(width)*n//width`, which handles widths
    that do not divide `n`. The obvious `reshape(width, k).max(1)` alternative is wrong:
    with n=100, width=30 it pads by 20 while buckets are only 4 wide, so five entire
    trailing buckets become padding.
    """
    n = values.size
    if width <= 0:
        raise ValueError("width must be positive")
    if width >= n:
        return values
    edges = (np.arange(width) * n) // width
    return np.maximum.reduceat(values, edges)


class WaterfallBuffer:
    """Fixed-height history of spectrum rows, newest first.

    Row 0 is the most recent. Scrolling is a memmove of the whole array: at 512x1024
    float32 that is 2 MB per frame, ~50 MB/s at 25 FPS, which is nothing on an M3. The
    zero-copy double-buffer trick is held in reserve until profiling asks for it
    (PLANNING.md section 7).
    """

    def __init__(self, rows: int = 512, cols: int = 1024, fill: float = FILL_DBFS) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self._fill = float(fill)
        self._buf = np.full((int(rows), int(cols)), self._fill, dtype=np.float32)
        # Counted explicitly rather than inferred by comparing against `fill`: the
        # measured noise floor (-134.7 dBFS) sits close to the sentinel, so a genuinely
        # quiet bin could otherwise be mistaken for unwritten history.
        self._written = 0

    @property
    def rows(self) -> int:
        return self._buf.shape[0]

    @property
    def cols(self) -> int:
        return self._buf.shape[1]

    @property
    def image(self) -> np.ndarray:
        """The history as (rows, cols); row 0 newest. Live view -- do not mutate."""
        return self._buf

    @property
    def written_rows(self) -> int:
        """How many rows hold real data; the rest is still fill."""
        return self._written

    def clear(self) -> None:
        self._buf[:] = self._fill
        self._written = 0

    def resize_cols(self, cols: int) -> None:
        """Change bin count (e.g. FFT size changed); history is discarded."""
        if cols == self.cols:
            return
        self._buf = np.full((self.rows, int(cols)), self._fill, dtype=np.float32)
        self._written = 0

    def push(self, row: np.ndarray) -> None:
        """Insert a spectrum row at the top, scrolling the rest down."""
        if row.size != self.cols:
            row = reduce_max(row, self.cols)
        self._buf[1:] = self._buf[:-1]
        self._buf[0] = row
        self._written = min(self.rows, self._written + 1)

    def percentile_levels(
        self, low: float = 5.0, high: float = 99.5, margin_db: float = 3.0
    ) -> tuple[float, float]:
        """Colour limits fitted to the visible history, for the auto-fit control."""
        if self._written == 0:
            return (-115.0, -40.0)
        live = self._buf[: self._written]
        lo, hi = np.percentile(live, [low, high])
        if hi - lo < 6.0:  # near-flat history: keep a usable span
            hi = lo + 6.0
        return (float(lo - margin_db), float(hi + margin_db))
