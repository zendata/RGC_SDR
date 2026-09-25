"""Audio output: runs the demodulator on a worker thread and feeds the sound device.

Three threads meet here, which is the whole difficulty:

* the device reader thread fills the IQ ring (see `device.source`),
* a worker thread pulls IQ *sequentially*, demodulates, and pushes audio into a FIFO,
* PortAudio's callback thread drains the FIFO into the output buffer.

The worker exists so the callback never does real work: a callback that allocates or
blocks produces a dropout, and a dropout is audible. The FIFO absorbs the jitter between
them, and its depth is the latency.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from .device.source import IQSource
from .dsp.demod import MODES, DemodChain
from .dsp.morse import ENVELOPE_DECIM, CwDecoder, EnvelopeSampler

_SOUNDDEVICE_HINT = (
    "sounddevice is not installed, so audio is unavailable. Install with:\n"
    "    pip install sounddevice"
)


def _import_sounddevice():
    try:
        import sounddevice  # type: ignore
    except (ImportError, OSError) as exc:
        # OSError: the wheel is present but PortAudio is missing.
        raise RuntimeError(f"{_SOUNDDEVICE_HINT}\n({exc})") from None
    return sounddevice


def audio_available() -> bool:
    try:
        _import_sounddevice()
    except RuntimeError:
        return False
    return True


class AudioFifo:
    """Thread-safe queue of audio samples, bounded so a stalled sink cannot grow it.

    Both ends are counted rather than inferred: `underrun_samples` says how much silence
    was invented for the callback, `dropped_samples` how much audio was thrown away
    because the sink stopped draining. Either one is a real fault worth surfacing.
    """

    def __init__(self, capacity: int, channels: int = 1) -> None:
        self._capacity = int(capacity)
        #: 1 holds plain sample arrays; 2 holds (frames, 2) stereo blocks. Capacity and
        #: counts are in frames either way.
        self.channels = int(channels)
        self._blocks: deque[np.ndarray] = deque()
        self._size = 0
        self._lock = threading.Lock()
        self.underrun_samples = 0
        self.dropped_samples = 0

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._size = 0

    def push(self, block: np.ndarray) -> None:
        if len(block) == 0:
            return
        with self._lock:
            self._blocks.append(block)
            self._size += len(block)
            while self._size > self._capacity and self._blocks:
                oldest = self._blocks.popleft()
                self._size -= len(oldest)
                self.dropped_samples += len(oldest)

    def pull(self, n: int) -> np.ndarray:
        """Exactly `n` frames, padding with silence if starved."""
        shape = (n,) if self.channels == 1 else (n, self.channels)
        out = np.zeros(shape, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._blocks:
                block = self._blocks[0]
                take = min(n - filled, len(block))
                out[filled : filled + take] = block[:take]
                filled += take
                if take == len(block):
                    self._blocks.popleft()
                else:
                    self._blocks[0] = block[take:]
                self._size -= take
            if filled < n:
                self.underrun_samples += n - filled
        return out


class AudioSink:
    """Demodulate an `IQSource` and play the result."""

    def __init__(
        self,
        source: IQSource,
        mode: str = "am",
        offset_hz: float = 0.0,
        volume: float = 0.4,
        squelch_dbfs: float | None = None,
        bandwidth_hz: float | None = None,
        pitch_hz: float | None = None,
        blocksize: int = 1024,
        buffer_blocks: int = 5,
        device=None,
    ) -> None:
        self._sd = _import_sounddevice()
        self.source = source
        self.blocksize = int(blocksize)
        self.buffer_blocks = int(buffer_blocks)
        self.device = device

        self._mode = mode
        self._offset = float(offset_hz)
        self._volume = float(volume)
        self._squelch = squelch_dbfs
        self._bandwidth = bandwidth_hz
        self._pitch = pitch_hz
        self._muted = False
        self._force_mono = False

        self._chain: DemodChain | None = None
        self._reader = None
        self._fifo = AudioFifo(self.blocksize * self.buffer_blocks)
        self._stream = None
        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._chain_errors = 0
        self._sampler: EnvelopeSampler | None = None
        self._decoder: CwDecoder | None = None
        #: Decoded CW, as a plain string so the UI can read it without a lock. String
        #: assignment is atomic, so a reader sees one version or the next, never a
        #: half-built line.
        self._cw_text = ""
        #: Optional tap, called on the worker thread with every produced audio block.
        #: Used for recording; must not block, or it becomes a dropout.
        self.on_audio = None

    # -- configuration -----------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def audio_rate(self) -> float:
        return self._chain.audio_rate if self._chain else 0.0

    @property
    def channel_dbfs(self) -> float | None:
        return self._chain.channel_dbfs if self._chain else None

    @property
    def bandwidth_hz(self) -> float:
        return self._chain.bandwidth_hz if self._chain else 0.0

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        with self._lock:
            if self._chain is not None:
                self._chain.set_bandwidth(bandwidth_hz)

    @property
    def pitch_hz(self) -> float:
        return self._chain.pitch_hz if self._chain else 0.0

    def set_pitch(self, pitch_hz: float) -> None:
        self._pitch = float(pitch_hz)
        with self._lock:
            if self._chain is not None:
                self._chain.set_pitch(pitch_hz)

    @property
    def offset_hz(self) -> float:
        return self._offset

    def _build_chain(self) -> DemodChain:
        return DemodChain(
            self.source.sample_rate,
            self._mode,
            offset_hz=self._offset,
            volume=self._volume,
            squelch_dbfs=self._squelch,
            bandwidth_hz=self._bandwidth,
            pitch_hz=self._pitch,
        )

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        """Silence the output without touching the volume setting.

        The demodulator keeps running, so unmuting is instant and at the right level
        rather than waiting for the AGC to find its feet again.
        """
        self._muted = bool(muted)
        if self._muted:
            self._fifo.clear()      # drop what is already queued, or it plays on briefly

    def set_volume(self, volume: float) -> None:
        self._volume = float(volume)
        with self._lock:
            if self._chain is not None:
                self._chain.volume = self._volume

    def set_squelch(self, squelch_dbfs: float | None) -> None:
        self._squelch = squelch_dbfs
        with self._lock:
            if self._chain is not None:
                self._chain.squelch_dbfs = squelch_dbfs

    def set_offset(self, offset_hz: float) -> None:
        self._offset = float(offset_hz)
        with self._lock:
            if self._chain is not None:
                self._chain.set_offset(self._offset)

    def set_mode(self, mode: str) -> None:
        """Switch demodulator. Restarts the stream, since the audio rate may change."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == self._mode and self.running:
            return
        self._mode = mode
        # A width chosen for the old mode is meaningless for the new one.
        self._bandwidth = None
        if self.running:
            self.restart()

    @property
    def channels(self) -> int:
        """1, or 2 for broadcast FM, which is always delivered as stereo."""
        return self._chain.channels if self._chain else 1

    @property
    def stereo(self) -> bool:
        return bool(self._chain and self._chain.stereo)

    @property
    def rds(self):
        return self._chain.rds if self._chain else None

    def set_force_mono(self, mono: bool) -> None:
        self._force_mono = bool(mono)
        with self._lock:
            if self._chain is not None:
                self._chain.force_mono = self._force_mono

    @property
    def cw_text(self) -> str:
        return self._cw_text

    @property
    def cw_wpm(self) -> float:
        decoder = self._decoder
        return decoder.wpm if decoder is not None else 0.0

    def reset(self) -> None:
        """Drop filter state and buffered audio: called on retune.

        Everything in flight was received at the old frequency, so playing it out would
        be a burst of the previous station.
        """
        with self._lock:
            if self._chain is not None:
                self._chain.reset()
        self._fifo.clear()
        # Decoded text belongs to the station that was being received.
        if self._decoder is not None:
            self._decoder.reset()
            self._cw_text = ""
        if self._sampler is not None:
            self._sampler.reset()
        if self._reader is not None:
            self._reader.skip_to_latest()

    def restart(self) -> None:
        """Rebuild for the source's current sample rate."""
        was_running = self.running
        self.stop()
        if was_running:
            self.start()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._chain = self._build_chain()
            self._chain.force_mono = self._force_mono
            if self._mode == "cw":
                self._sampler = EnvelopeSampler(ENVELOPE_DECIM)
                self._decoder = CwDecoder(self._chain.if_rate / ENVELOPE_DECIM)
            else:
                self._sampler = self._decoder = None
        self._cw_text = ""
        self._reader = self.source.sequential_reader()
        self._fifo = AudioFifo(self.blocksize * self.buffer_blocks)

        self._fifo = AudioFifo(self.blocksize * self.buffer_blocks, self.channels)
        self._stream = self._sd.OutputStream(
            samplerate=self._chain.audio_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._running.set()
        self._worker = threading.Thread(target=self._pump, name="audio-demod", daemon=True)
        self._worker.start()
        self._prefill()
        self._stream.start()

    def _prefill(self, timeout_s: float = 0.4) -> None:
        """Let the worker get ahead before the stream opens.

        Starting the stream immediately guarantees the callback arrives before any audio
        exists, which is one buffer of silence -- a click on every start. Bounded, so a
        starved or stopped IQ stream cannot hang startup.
        """
        import time

        target = max(1, self._fifo.capacity // 3)
        deadline = time.monotonic() + timeout_s
        while len(self._fifo) < target and time.monotonic() < deadline:
            time.sleep(0.005)

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._fifo.clear()

    # -- the two hot paths -------------------------------------------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        block = self._fifo.pull(frames)
        if block.ndim == 1:
            outdata[:, 0] = block
        else:
            outdata[:] = block
        self._wake.set()          # room has been freed; let the worker refill

    def _pump(self) -> None:
        """Keep the FIFO around half full, no fuller."""
        target = self._fifo.capacity // 2
        while self._running.is_set():
            if len(self._fifo) >= target:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            with self._lock:
                chain = self._chain
            if chain is None:
                break
            need = chain.input_for_audio(self.blocksize)
            if self._reader.available() < need:
                self._wake.wait(timeout=0.01)
                self._wake.clear()
                continue
            iq = self._reader.read(need)
            if iq.size == 0:
                continue
            try:
                audio = chain.process(iq)
            except Exception:
                # Never let a DSP error kill audio silently or spin the thread.
                self._chain_errors += 1
                continue
            decoder, sampler = self._decoder, self._sampler
            if decoder is not None and sampler is not None:
                try:
                    decoder.feed(sampler.process(chain.last_channel))
                    self._cw_text = decoder.text
                except Exception:
                    self._chain_errors += 1

            # The recorder is fed before muting: muting is a choice about the room,
            # not about the recording, and a silent file would be a nasty surprise.
            tap = self.on_audio
            if tap is not None and audio.size:
                try:
                    tap(audio)
                except Exception:
                    self._chain_errors += 1
            if self._muted:
                # Pushed as silence rather than skipped, so the stream stays fed and the
                # callback never reports an underrun for something deliberate.
                audio = np.zeros_like(audio)
            self._fifo.push(audio)

    # -- reporting ---------------------------------------------------------

    @property
    def stats(self) -> dict[str, float]:
        chain = self._chain
        return {
            "audio_rate": self.audio_rate,
            "queued": len(self._fifo),
            "underrun_samples": self._fifo.underrun_samples,
            "dropped_samples": self._fifo.dropped_samples,
            "lost_iq": getattr(self._reader, "lost", 0),
            "muted_blocks": chain.muted_blocks if chain else 0,
            "agc_gain": chain.agc_gain if chain else 1.0,
            "channel_dbfs": chain.channel_dbfs if chain else -200.0,
            "chain_errors": self._chain_errors,
            "muted": self._muted,
            "cw_wpm": self.cw_wpm,
        }


# -- transmit audio: the microphone -----------------------------------------------
# Groundwork for transmitting (PLANNING.md, P6): captured here, modulated by
# dsp.modulate, and one day written to a radio. Nothing in this section transmits.

#: The input to use for transmit audio, by the name CoreAudio gives it.
PREFERRED_MIC = "MacBook Air Microphone"


def choose_input_device(devices, preferred: str = PREFERRED_MIC) -> int | None:
    """Index of the microphone to use, or None for the system default input.

    By name rather than by index or "the default": indexes shift as devices come and
    go, and the default input can be a headset or a phone that happens to be paired
    (this machine lists an iPhone-style "light Microphone" ahead of the built-in one).
    """
    inputs = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    for i, d in inputs:
        if d.get("name") == preferred:
            return i
    for i, d in inputs:
        if preferred.lower() in str(d.get("name", "")).lower():
            return i
    return None


class Microphone:
    """Mono audio from the microphone, into a FIFO the modulator pulls from.

    The PortAudio callback only copies and measures; anything heavier belongs on the
    consumer's thread, as on the receive side.
    """

    def __init__(
        self,
        samplerate: float = 48_000.0,
        blocksize: int = 1024,
        device_name: str = PREFERRED_MIC,
        buffer_seconds: float = 1.0,
    ) -> None:
        self.samplerate = float(samplerate)
        self.blocksize = int(blocksize)
        self.device_name = device_name
        self._fifo = AudioFifo(int(self.samplerate * buffer_seconds))
        self._stream = None
        #: Peak level of the most recent block, dBFS: for a mic-gain meter.
        self.level_dbfs = -200.0
        self.device_index: int | None = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        block = indata[:, 0].astype(np.float32, copy=True)
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        self.level_dbfs = 20.0 * np.log10(peak + 1e-12)
        self._fifo.push(block)

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _import_sounddevice()
        self.device_index = choose_input_device(sd.query_devices(), self.device_name)
        stream = sd.InputStream(
            device=self.device_index, channels=1, samplerate=self.samplerate,
            blocksize=self.blocksize, dtype="float32", callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def available(self) -> int:
        return len(self._fifo)

    def read(self, n: int) -> np.ndarray:
        """Exactly `n` samples; silence-padded if the microphone has not caught up."""
        return self._fifo.pull(n)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self._fifo.clear()
