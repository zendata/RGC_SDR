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
from .dsp.ais import CHANNELS as AIS_CHANNELS
from .dsp.ais import AisDecoder
from .dsp.aprs import AprsDecoder
from .dsp.pocsag import PocsagDecoder


@dataclass(frozen=True)
class DecoderSpec:
    label: str
    #: Builds a decoder for an IF rate; for a fixed-channel decoder, also its channel name.
    factory: Callable[..., object]
    #: Channel width the decoder's NBFM chain listens through.
    bandwidth_hz: float
    #: Whether its messages carry text that should be hidden by default.
    private: bool = False
    #: Fixed channels (name -> Hz) decoded wherever the radio is tuned, provided they are
    #: in view, instead of the listening frequency. AIS has two, 50 kHz apart.
    channels: tuple[tuple[str, float], ...] = ()


DECODERS: dict[str, DecoderSpec] = {
    # +/-4.5 kHz deviation plus up to 2400 baud: wider than a 12.5 kHz voice channel.
    "pocsag": DecoderSpec("POCSAG", PocsagDecoder, 16e3, private=True),
    # Amateur traffic, so nothing to hide. 145.175 MHz in VK.
    "aprs": DecoderSpec("APRS", AprsDecoder, 12.5e3),
    # GMSK 9600 with +/-2.4 kHz deviation in a 25 kHz channel; both channels at once.
    "ais": DecoderSpec("AIS", AisDecoder, 20e3, channels=tuple(AIS_CHANNELS.items())),
}

#: Seconds of IQ handed to the chain at a time.
BLOCK_SECONDS = 0.05
#: Messages kept for the UI to collect; older ones are dropped if it never does.
QUEUE_LIMIT = 1000
#: A fixed channel counts as in view when it sits this far inside the span's edge.
EDGE_FRACTION = 0.45


class _Lane:
    """One channel: its NBFM chain and its decoder."""

    def __init__(self, chain: DemodChain, decoder, name: str = "",
                 freq_hz: float | None = None) -> None:
        self.chain, self.decoder, self.name, self.freq_hz = chain, decoder, name, freq_hz
        self.in_view = True


class DecodeWorker:
    """Runs one decoder over the source's IQ: at the listening offset, or on the
    decoder's own fixed channels."""

    def __init__(self, source: IQSource, name: str, offset_hz: float = 0.0) -> None:
        if name not in DECODERS:
            raise ValueError(f"unknown decoder {name!r}")
        self.name = name
        self.spec = DECODERS[name]
        self.source = source
        self._offset = float(offset_hz)
        self._lock = threading.Lock()
        self._lanes: list[_Lane] = []
        rate, bw = source.sample_rate, self.spec.bandwidth_hz
        if self.spec.channels:
            for label, freq in self.spec.channels:
                chain = DemodChain(rate, "nbfm", offset_hz=0.0, bandwidth_hz=bw, agc=False)
                self._lanes.append(_Lane(chain, self.spec.factory(chain.if_rate, label),
                                         label, freq))
            # One ship, one name, whichever channel it was heard on.
            shared = getattr(self._lanes[0].decoder, "names", None)
            for lane in self._lanes[1:]:
                if shared is not None and hasattr(lane.decoder, "names"):
                    lane.decoder.names = shared
            self.follow_tuning()
        else:
            chain = DemodChain(rate, "nbfm", offset_hz=self._offset, bandwidth_hz=bw,
                               agc=False)
            self._lanes.append(_Lane(chain, self.spec.factory(chain.if_rate)))
        self._messages: deque = deque(maxlen=QUEUE_LIMIT)
        self._reader = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._block = max(4096, int(source.sample_rate * BLOCK_SECONDS))
        self.errors = 0
        self.decoded = 0

    # The first lane, for a single-channel decoder (and its tests).
    @property
    def _chain(self) -> DemodChain:
        return self._lanes[0].chain

    @property
    def decoder(self):
        return self._lanes[0].decoder

    @property
    def running(self) -> bool:
        return self._thread is not None

    @property
    def fixed_channels(self) -> bool:
        return bool(self.spec.channels)

    def channels_in_view(self) -> list[tuple[str, float, bool]]:
        """(name, Hz, in view) for each fixed channel."""
        return [(lane.name, lane.freq_hz, lane.in_view) for lane in self._lanes
                if lane.freq_hz is not None]

    def follow_tuning(self) -> None:
        """Re-aim fixed channels after a retune, and note which are still in view."""
        if not self.fixed_channels:
            return
        centre, half = self.source.center_freq, self.source.sample_rate * EDGE_FRACTION
        with self._lock:
            for lane in self._lanes:
                offset = lane.freq_hz - centre
                lane.in_view = abs(offset) + self.spec.bandwidth_hz / 2 <= half
                lane.chain.set_offset(offset)

    def set_offset(self, offset_hz: float) -> None:
        """Follow the listening offset; fixed channels stay where they are."""
        self._offset = float(offset_hz)
        if self.fixed_channels:
            return
        with self._lock:
            self._chain.set_offset(self._offset)
            self.decoder.reset()

    def reset(self) -> None:
        """Drop everything in flight: called on retune, when it was another channel."""
        with self._lock:
            for lane in self._lanes:
                lane.chain.reset()
                lane.decoder.reset()
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
        found = []
        with self._lock:
            for lane in self._lanes:
                if not lane.in_view:
                    continue
                lane.chain.process(iq)
                found += lane.decoder.process(lane.chain.last_discriminator)
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
