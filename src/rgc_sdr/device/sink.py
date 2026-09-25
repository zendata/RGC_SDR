"""The transmit side of a radio: where modulated IQ will go (PLANNING.md, P6).

Only the interface for now. No radio is keyed anywhere in this application yet; the TX
button runs microphone -> modulator and meters the result without an `IQSink`, so the
whole audio chain can be checked with no RF produced.

A Soapy implementation will open a TX stream (CF32, |x| <= 1), and must enforce the
interlocks the plan lists -- licensed bands only, a callsign set, a transmit timeout, and
for a half-duplex radio (`TxCaps.full_duplex` False, as on the HackRF) the receive stream
stopped first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .source import TxCaps


class IQSink(ABC):
    """A transmit stream of complex baseband samples."""

    @property
    @abstractmethod
    def caps(self) -> TxCaps: ...

    @property
    @abstractmethod
    def sample_rate(self) -> float: ...

    @property
    @abstractmethod
    def center_freq(self) -> float: ...

    @abstractmethod
    def set_center_freq(self, hz: float) -> float: ...

    @abstractmethod
    def start(self) -> None:
        """Key the transmitter."""

    @abstractmethod
    def write(self, iq: np.ndarray) -> int:
        """Queue samples for transmission. Returns how many were accepted."""

    @abstractmethod
    def stop(self) -> None:
        """Unkey. Must be safe to call at any time, including from a timeout."""

    def close(self) -> None:
        self.stop()
