"""Transmit: microphone -> modulator -> (one day) the radio. PLANNING.md, P6.

For now there is no `IQSink` behind it, so pressing TX is a *dry run*: the MacBook Air
microphone is captured, modulated in the current mode exactly as it would be for the air,
and metered -- and nothing is radiated. That exercises the whole audio side of the chain
before a transmitter is ever keyed.

A worker thread does the modulating, for the same reason the receive side has one: the
PortAudio callback must never do real work. The timeout is checked by the caller (the
UI's timer), so the transmitter is always unkeyed from one place.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .dsp.modulate import TX_MODES, Modulator, nearest_tx_rate

#: Transmit IQ rate: 50 x 48 kHz and 10 x 240 kHz, so every mode interpolates by a whole
#: number; and at or above the HackRF's recommended 2 MS/s minimum.
TX_IQ_RATE = nearest_tx_rate("wbfm", 2.4e6)

#: A stuck TX button must not transmit forever. Three minutes, as on many transceivers.
TX_TIMEOUT_S = 180.0


class Transmitter:
    """Microphone audio, modulated in `mode`, written to `sink` if there is one."""

    def __init__(self, mode: str, mic=None, sink=None, iq_rate: float = TX_IQ_RATE,
                 timeout_s: float = TX_TIMEOUT_S, block: int = 1024, clock=time.monotonic):
        if mode not in TX_MODES:
            raise ValueError(
                "CW transmit is not supported" if mode == "cw"
                else f"choose a mode to transmit in ({', '.join(m.upper() for m in TX_MODES)})"
            )
        if mic is None:
            from .audio import Microphone

            mic = Microphone()
        self.mode = mode
        self.mic = mic
        self.sink = sink
        self.modulator = Modulator(mode, iq_rate)
        self.timeout_s = float(timeout_s)
        self._block = int(block)
        self._clock = clock
        self._started_at: float | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        #: IQ produced so far, and the peak magnitude of the latest block (full scale 1).
        self.iq_samples = 0
        self.iq_peak = 0.0

    @property
    def dry_run(self) -> bool:
        """True when nothing is being radiated: there is no sink."""
        return self.sink is None

    @property
    def active(self) -> bool:
        return self._running.is_set()

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._started_at is None else self._clock() - self._started_at

    def expired(self) -> bool:
        return self.active and self.elapsed_s >= self.timeout_s

    def start(self) -> None:
        if self.active:
            return
        self.mic.start()
        if self.sink is not None:
            self.sink.start()
        self._started_at = self._clock()
        self._running.set()
        self._thread = threading.Thread(target=self._pump, name="tx-modulate", daemon=True)
        self._thread.start()

    def pump_once(self) -> int:
        """Modulate one block if the microphone has one. Returns IQ samples produced."""
        if self.mic.available() < self._block:
            return 0
        iq = self.modulator.process(self.mic.read(self._block))
        if self.sink is not None:
            self.sink.write(iq)
        self.iq_samples += iq.size
        self.iq_peak = float(np.max(np.abs(iq))) if iq.size else 0.0
        return iq.size

    def _pump(self) -> None:
        while self._running.is_set():
            if not self.pump_once():
                time.sleep(0.005)

    def stop(self) -> None:
        """Unkey first, then stop listening. Safe to call repeatedly."""
        self._running.clear()
        if self.sink is not None:
            self.sink.stop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.mic.stop()
        self._started_at = None
