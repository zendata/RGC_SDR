"""The transmit side of a radio: where modulated IQ goes (PLANNING.md, P6 and 7m).

`SoapyIQSink` transmits on the same Soapy device the receiver has open -- a radio can
only be opened once. For a half-duplex radio (`TxCaps.full_duplex` False, the HackRF)
it stops the receive stream first and restarts it on unkey; SoapyHackRF keeps separate
receive and transmit frequency, rate and gains and re-applies them on each activation.

No band, mode or power limits by the owner's choice (VK3RQ, in-house receiver testing).
What remains is operational: the application's 3-minute timeout, and a low starting
gain the user can raise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .source import LO_OFFSET_HZ, SOAPY_TX, TxCaps, _Nco

#: Microseconds to wait for the radio to accept a block before counting a stall.
WRITE_TIMEOUT_US = 200_000


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


class SoapyIQSink(IQSink):
    """Transmit through the Soapy device behind a `SoapyIQSource`."""

    def __init__(self, source, center_freq: float, sample_rate: float,
                 gains: dict[str, float] | None = None) -> None:
        if source.caps.tx is None:
            raise RuntimeError(f"{source.caps.label or 'this radio'} cannot transmit")
        self._source = source
        self._dev = source.soapy_device
        self._caps = source.caps.tx
        self._rate = float(sample_rate)
        self._freq = float(center_freq)
        self._gains = dict(gains or {})
        # A spiky radio leaks its LO too: transmit from LO_OFFSET_HZ away and shift the
        # signal back down to the wanted frequency, as on receive.
        profile = getattr(source, "profile", None)
        self._lo_offset = LO_OFFSET_HZ if (profile and profile.dc_offset) else 0.0
        self._nco = _Nco(-self._lo_offset, self._rate)
        self._stream = None
        self._paused_receiver = False
        self.samples_written = 0
        self.stalls = 0

    @property
    def caps(self) -> TxCaps:
        return self._caps

    @property
    def sample_rate(self) -> float:
        return self._rate

    @property
    def center_freq(self) -> float:
        return self._freq

    @property
    def keyed(self) -> bool:
        return self._stream is not None

    def set_center_freq(self, hz: float) -> float:
        self._freq = float(hz)
        if self._stream is not None:
            self._dev.setFrequency(SOAPY_TX, 0, self._freq + self._lo_offset)
        return self._freq

    def set_gain(self, name: str, db: float) -> None:
        self._gains[name] = float(db)
        if self._stream is not None:
            self._dev.setGain(SOAPY_TX, 0, name, float(db))

    def start(self) -> None:
        if self._stream is not None:
            return
        if not self._caps.full_duplex:
            self._source.stop()
            self._paused_receiver = True
        try:
            d = self._dev
            d.setSampleRate(SOAPY_TX, 0, self._rate)
            d.setFrequency(SOAPY_TX, 0, self._freq + self._lo_offset)
            for name, db in self._gains.items():
                d.setGain(SOAPY_TX, 0, name, float(db))
            stream = d.setupStream(SOAPY_TX, "CF32")
            d.activateStream(stream)
            self._stream = stream
        except Exception:
            self._resume_receiver()
            raise

    def write(self, iq: np.ndarray) -> int:
        if self._stream is None or iq.size == 0:
            return 0
        buf = np.ascontiguousarray(iq, dtype=np.complex64).copy()
        self._nco.process(buf)
        sent = 0
        while sent < buf.size and self._stream is not None:
            result = self._dev.writeStream(self._stream, [buf[sent:]], buf.size - sent,
                                           timeoutUs=WRITE_TIMEOUT_US)
            if result.ret > 0:
                sent += result.ret
            else:
                self.stalls += 1
                if self.stalls % 16 == 0:
                    break            # the radio has stopped taking samples; do not hang
        self.samples_written += sent
        return sent

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                self._dev.deactivateStream(stream)
                self._dev.closeStream(stream)
            finally:
                self._resume_receiver()
        else:
            self._resume_receiver()

    def _resume_receiver(self) -> None:
        if self._paused_receiver:
            self._paused_receiver = False
            self._source.start()
