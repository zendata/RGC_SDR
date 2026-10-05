"""Data decoders, run from the IQ stream on their own thread (PLANNING.md 7p).

Each decoder has its own gapless reader and its own NBFM chain, so it decodes whether or
not audio is on, whatever the audio mode, and through mute and squelch. The chain's raw
discriminator is what the decoder slices (`DemodChain.last_discriminator`).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .device.source import IQSource
from .dsp.demod import DemodChain
from .dsp.pocsag import PocsagDecoder


@dataclass(frozen=True)
class DecoderSpec:
    label: str
    factory: Callable[[float], object]
    #: Channel width the decoder's NBFM chain listens through.
    bandwidth_hz: float
    #: Whether its messages carry text that should be hidden by default.
    private: bool = False


DECODERS: dict[str, DecoderSpec] = {
    # +/-4.5 kHz deviation plus up to 2400 baud: wider than a 12.5 kHz voice channel.
    "pocsag": DecoderSpec("POCSAG", PocsagDecoder, 16e3, private=True),
}

#: Seconds of IQ handed to the chain at a time.
BLOCK_SECONDS = 0.05
#: Messages kept for the UI to collect; older ones are dropped if it never does.
QUEUE_LIMIT = 1000


class DecodeWorker:
    """Runs one decoder over the source's IQ, at the listening offset."""

    def __init__(self, source: IQSource, name: str, offset_hz: float = 0.0) -> None:
        if name not in DECODERS:
            raise ValueError(f"unknown decoder {name!r}")
        self.name = name
        self.spec = DECODERS[name]
        self.source = source
        self._offset = float(offset_hz)
        self._lock = threading.Lock()
        self._chain = DemodChain(source.sample_rate, "nbfm", offset_hz=self._offset,
                                 bandwidth_hz=self.spec.bandwidth_hz, agc=False)
        self.decoder = self.spec.factory(self._chain.if_rate)
        self._messages: deque = deque(maxlen=QUEUE_LIMIT)
        self._reader = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._block = max(4096, int(source.sample_rate * BLOCK_SECONDS))
        self.errors = 0
        self.decoded = 0

    @property
    def running(self) -> bool:
        return self._thread is not None

    def set_offset(self, offset_hz: float) -> None:
        self._offset = float(offset_hz)
        with self._lock:
            self._chain.set_offset(self._offset)
            self.decoder.reset()

    def reset(self) -> None:
        """Drop everything in flight: called on retune, when it was another channel."""
        with self._lock:
            self._chain.reset()
            self.decoder.reset()
            if self._reader is not None:
                self._reader.skip_to_latest()

    def take(self) -> list:
        """Messages decoded since the last call, oldest first."""
        out = []
        while self._messages:
            out.append(self._messages.popleft())
        return out

    def process(self, iq: np.ndarray) -> list:
        """Decode one block of IQ. The thread's step, and callable directly in tests."""
        with self._lock:
            self._chain.process(iq)
            found = self.decoder.process(self._chain.last_discriminator)
        self.decoded += len(found)
        self._messages.extend(found)
        return found

    def start(self) -> None:
        if self._thread is not None:
            return
        self._reader = self.source.sequential_reader()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name=f"decode-{self.name}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while self._running.is_set():
            if self._reader.available() < self._block:
                time.sleep(0.02)
                continue
            iq = self._reader.read(self._block)
            if iq.size == 0:
                continue
            try:
                self.process(iq)
            except Exception:
                # A DSP error must not kill decoding silently or spin the thread.
                self.errors += 1
