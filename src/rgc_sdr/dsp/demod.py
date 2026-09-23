"""Demodulators: AM, narrow/wide FM, and SSB.

Every block here is *stateful and continuous*. The display can restart its filters each
frame because it only shows the newest block, but audio is a single unbroken stream: any
discontinuity at a block boundary is an audible click or buzz at the block rate. So the
mixer carries its phase, the FIRs carry their history, and the FM detector carries its
previous sample.

Pure NumPy plus scipy for the one stateful IIR (FM de-emphasis). No Qt, no device access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decimate import StreamDecimator, lowpass_taps

#: Modes the UI offers, in the order the roadmap introduces them.
MODES = ("am", "nbfm", "wbfm", "usb", "lsb")


#: Channel widths offered per mode, in Hz. The mode's own spec value is the default.
BANDWIDTH_PRESETS: dict[str, tuple[float, ...]] = {
    "am": (3e3, 4.5e3, 6e3, 9e3, 12e3, 16e3),
    "nbfm": (6e3, 8e3, 12.5e3, 16e3, 25e3),
    "wbfm": (100e3, 150e3, 200e3),
    "usb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
    "lsb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
}


@dataclass(frozen=True)
class ModeSpec:
    """Per-mode plan: how wide the channel is and what rates the chain needs."""

    bandwidth_hz: float
    if_target_hz: float
    audio_target_hz: float | None = None
    deviation_hz: float = 5e3
    deemphasis_s: float | None = None
    audio_cutoff_hz: float | None = None
    squelch_capable: bool = False


MODE_SPECS: dict[str, ModeSpec] = {
    # Double-sideband AM broadcast: 9 kHz channel spacing in ITU regions 1 and 3.
    "am": ModeSpec(bandwidth_hz=9e3, if_target_hz=48e3),
    "nbfm": ModeSpec(
        bandwidth_hz=12.5e3, if_target_hz=48e3, deviation_hz=2.5e3,
        audio_cutoff_hz=3.4e3, squelch_capable=True,
    ),
    # Broadcast FM. 150 kHz rather than the nominal 180: the decimation cascade's own
    # anti-alias filters retain about +/-0.41 of the output rate, so 180 kHz would sit in
    # the transition band. Mono only -- no stereo pilot decoding.
    "wbfm": ModeSpec(
        bandwidth_hz=150e3, if_target_hz=160e3, audio_target_hz=44e3,
        deviation_hz=75e3, deemphasis_s=75e-6, audio_cutoff_hz=15e3,
        squelch_capable=True,
    ),
    "usb": ModeSpec(bandwidth_hz=2.7e3, if_target_hz=48e3),
    "lsb": ModeSpec(bandwidth_hz=2.7e3, if_target_hz=48e3),
}


def fir_length_for(cutoff: float, minimum: int = 31, maximum: int = 511) -> int:
    """Odd tap count giving a transition band proportional to the cutoff."""
    transition = max(cutoff * 0.35, 0.004)
    n = int(4.0 / transition)
    n = max(minimum, min(maximum, n))
    return n | 1


class Mixer:
    """Frequency shift, continuous across blocks.

    The phase accumulator is what makes it continuous: restarting the exponential at zero
    each block would step the phase and click once per block.
    """

    def __init__(self, sample_rate: float, offset_hz: float = 0.0) -> None:
        self._fs = float(sample_rate)
        self._offset = float(offset_hz)
        self._phase = 0.0

    @property
    def offset_hz(self) -> float:
        return self._offset

    def set_offset(self, offset_hz: float) -> None:
        self._offset = float(offset_hz)

    def reset(self) -> None:
        self._phase = 0.0

    def process(self, iq: np.ndarray) -> np.ndarray:
        if self._offset == 0.0 or iq.size == 0:
            return iq
        step = -2.0 * np.pi * self._offset / self._fs
        phase = self._phase + step * np.arange(iq.size)
        # Wrapped, or the accumulator loses precision after a few minutes of audio.
        self._phase = float((self._phase + step * iq.size) % (2.0 * np.pi))
        return iq * np.exp(1j * phase).astype(np.complex128)


class Fir:
    """Stateful FIR. Taps may be complex, for a single-sideband filter."""

    def __init__(self, taps: np.ndarray) -> None:
        self._taps = np.asarray(taps)
        self._history: np.ndarray | None = None

    @property
    def taps(self) -> np.ndarray:
        return self._taps

    def reset(self) -> None:
        self._history = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        dtype = np.result_type(x.dtype, self._taps.dtype)
        if self._history is None:
            self._history = np.zeros(self._taps.size - 1, dtype=dtype)
        padded = np.concatenate([self._history.astype(dtype), x.astype(dtype)])
        self._history = padded[-(self._taps.size - 1) :]
        return np.convolve(padded, self._taps, mode="valid")


def channel_taps(bandwidth_hz: float, sample_rate: float) -> np.ndarray:
    """Real low-pass keeping +/- bandwidth/2 of a complex signal."""
    cutoff = min(bandwidth_hz / 2.0 / sample_rate, 0.49)
    return lowpass_taps(cutoff, fir_length_for(cutoff))


def sideband_taps(bandwidth_hz: float, sample_rate: float, upper: bool) -> np.ndarray:
    """Complex band-pass isolating one sideband.

    With the carrier at 0 Hz the upper sideband occupies 0..+B and the lower -B..0, so a
    real low-pass cannot separate them -- it is symmetric. Modulating a half-width
    low-pass up to +/-B/2 gives an asymmetric filter that passes only one side.
    """
    half = min(bandwidth_hz / 2.0 / sample_rate, 0.24)
    taps = lowpass_taps(half, fir_length_for(half))
    n = np.arange(taps.size) - (taps.size - 1) / 2.0
    direction = 1.0 if upper else -1.0
    return taps * np.exp(2j * np.pi * direction * half * n)


class AmDetector:
    """Envelope detector with a slow DC blocker.

    The DC estimate is a per-block running mean rather than a per-sample IIR: it updates
    smoothly across boundaries, needs no Python loop, and AM only requires the carrier
    term removed, not a sharp high-pass.
    """

    def __init__(self, smoothing: float = 0.05) -> None:
        self._alpha = float(smoothing)
        self._dc: float | None = None

    def reset(self) -> None:
        self._dc = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)
        envelope = np.abs(x)
        mean = float(envelope.mean())
        self._dc = mean if self._dc is None else self._dc + self._alpha * (mean - self._dc)
        return envelope - self._dc


class FmDetector:
    """Quadrature discriminator: the phase advance between consecutive samples.

    Keeping the previous block's last sample is what makes it seamless; without it every
    block starts with a bogus phase step.
    """

    def __init__(self, sample_rate: float, deviation_hz: float) -> None:
        self._gain = float(sample_rate) / (2.0 * np.pi * float(deviation_hz))
        self._last: complex | None = None

    def reset(self) -> None:
        self._last = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)
        previous = x[0] if self._last is None else self._last
        shifted = np.empty_like(x)
        shifted[0] = previous
        shifted[1:] = x[:-1]
        self._last = complex(x[-1])
        return np.angle(x * np.conj(shifted)) * self._gain


class Deemphasis:
    """First-order de-emphasis for broadcast FM (75 us in ITU region 2, 50 us elsewhere).

    A stateful IIR, so scipy carries the filter state between blocks.
    """

    def __init__(self, sample_rate: float, tau_s: float) -> None:
        from scipy.signal import lfilter, lfilter_zi

        self._lfilter = lfilter
        # Bilinear transform of 1 / (1 + s*tau).
        k = 1.0 / np.tan(1.0 / (2.0 * float(sample_rate) * float(tau_s)))
        self._b = np.array([1.0, 1.0]) / (1.0 + k)
        self._a = np.array([1.0, (1.0 - k) / (1.0 + k)])
        self._zi_template = lfilter_zi(self._b, self._a)
        self._zi = None

    def reset(self) -> None:
        self._zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self._zi is None:
            self._zi = self._zi_template * x[0]
        out, self._zi = self._lfilter(self._b, self._a, x, zi=self._zi)
        return out


class AudioAgc:
    """Normalise demodulated audio to a usable level.

    Without this the output is proportional to absolute signal strength, so a weak
    station is inaudible however high the volume: measured on air, a -103 dBFS AM carrier
    gave an audio RMS of 0.00001. Every real receiver normalises here.

    Asymmetric time constants, as an AGC should be: it backs off quickly when a signal
    gets louder so it cannot blast, and recovers slowly so a pause does not pump the noise
    floor up. `max_gain` stops it amplifying pure noise to full scale on a dead channel.
    """

    def __init__(
        self,
        target_rms: float = 0.15,
        attack: float = 0.35,
        decay: float = 0.02,
        max_gain: float = 20000.0,
        floor_rms: float = 1e-7,
    ) -> None:
        self.target_rms = float(target_rms)
        self.attack = float(attack)
        self.decay = float(decay)
        self.max_gain = float(max_gain)
        self.floor_rms = float(floor_rms)
        self._gain = 1.0
        self._primed = False

    @property
    def gain(self) -> float:
        return self._gain

    def reset(self) -> None:
        self._gain = 1.0
        self._primed = False

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        level = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if level < self.floor_rms:
            return audio * self._gain
        wanted = min(self.target_rms / level, self.max_gain)
        if not self._primed:
            # Jump straight to the right gain on the first block. Easing in from 1.0 at
            # the slow decay rate would take ten seconds of near-silence to become
            # audible on a weak signal, which reads as broken rather than as an AGC.
            self._gain = wanted
            self._primed = True
        else:
            # Louder than wanted -> attack (fast); quieter -> decay (slow).
            rate = self.attack if wanted < self._gain else self.decay
            self._gain += rate * (wanted - self._gain)
        return audio * self._gain


class DemodChain:
    """Mixer -> decimate -> channel filter -> detect -> audio shaping -> gain.

    Produces real audio at `audio_rate`, which is the input rate divided by powers of two
    and is handed straight to the output device (CoreAudio resamples odd rates itself).
    """

    def __init__(
        self,
        sample_rate: float,
        mode: str = "am",
        offset_hz: float = 0.0,
        volume: float = 0.4,
        squelch_dbfs: float | None = None,
        agc: bool = True,
        bandwidth_hz: float | None = None,
    ) -> None:
        if mode not in MODE_SPECS:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.sample_rate = float(sample_rate)
        self.mode = mode
        self.spec = MODE_SPECS[mode]
        self.volume = float(volume)
        self.squelch_dbfs = squelch_dbfs
        #: In-channel power of the most recent block, for the S-meter. dBFS, so it is
        #: relative to full scale and not calibrated to dBm -- see PLANNING.md 7d.
        self.channel_dbfs = -200.0

        self.if_decim = self._pick_factor(self.sample_rate, self.spec.if_target_hz)
        self.if_rate = self.sample_rate / self.if_decim

        self._mixer = Mixer(self.sample_rate, offset_hz)
        self._decimator = StreamDecimator(self.if_decim)

        self.bandwidth_hz = float(
            bandwidth_hz if bandwidth_hz is not None else self.spec.bandwidth_hz
        )
        self._channel = Fir(self._channel_taps())

        if mode == "am":
            self._detector = AmDetector()
        elif mode in ("nbfm", "wbfm"):
            self._detector = FmDetector(self.if_rate, self.spec.deviation_hz)
        else:
            self._detector = None  # SSB needs no detector beyond taking the real part

        self._deemph = (
            Deemphasis(self.if_rate, self.spec.deemphasis_s)
            if self.spec.deemphasis_s
            else None
        )

        self.audio_decim = 1
        if self.spec.audio_target_hz is not None:
            self.audio_decim = self._pick_factor(self.if_rate, self.spec.audio_target_hz)
        self._audio_decimator = StreamDecimator(self.audio_decim)
        self.audio_rate = self.if_rate / self.audio_decim

        self._agc = AudioAgc() if agc else None

        cutoff = self.spec.audio_cutoff_hz
        self._audio_fir = None
        if cutoff is not None and cutoff < self.audio_rate / 2.0:
            c = cutoff / self.audio_rate
            self._audio_fir = Fir(lowpass_taps(c, fir_length_for(c)))

        self.muted_blocks = 0

    def _channel_taps(self) -> np.ndarray:
        if self.mode in ("usb", "lsb"):
            return sideband_taps(self.bandwidth_hz, self.if_rate, upper=(self.mode == "usb"))
        return channel_taps(self.bandwidth_hz, self.if_rate)

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        """Retune the channel filter in place.

        Only the filter is rebuilt, not the chain: the decimators and detector hold state
        that is still valid, and tearing them down would click.
        """
        bandwidth_hz = float(bandwidth_hz)
        if bandwidth_hz == self.bandwidth_hz:
            return
        self.bandwidth_hz = bandwidth_hz
        self._channel = Fir(self._channel_taps())

    @staticmethod
    def _pick_factor(rate: float, target: float) -> int:
        """Largest power of two that keeps rate/factor at or above `target`."""
        factor = 1
        while rate / (factor * 2) >= target:
            factor *= 2
        return factor

    @property
    def offset_hz(self) -> float:
        return self._mixer.offset_hz

    def set_offset(self, offset_hz: float) -> None:
        self._mixer.set_offset(offset_hz)

    def input_for_audio(self, n_audio: int) -> int:
        """Input IQ samples needed for roughly `n_audio` output samples."""
        return int(n_audio * self.if_decim * self.audio_decim)

    def reset(self) -> None:
        """Drop all filter state. Call on retune: the history is a different signal."""
        self._mixer.reset()
        self._decimator.reset()
        self._channel.reset()
        if self._detector is not None:
            self._detector.reset()
        if self._deemph is not None:
            self._deemph.reset()
        self._audio_decimator.reset()
        if self._audio_fir is not None:
            self._audio_fir.reset()
        if self._agc is not None:
            self._agc.reset()

    def process(self, iq: np.ndarray) -> np.ndarray:
        if iq.size == 0:
            return np.zeros(0, dtype=np.float32)

        shifted = self._mixer.process(iq)
        narrow = self._decimator.process(shifted)
        if narrow.size == 0:
            return np.zeros(0, dtype=np.float32)
        channel = self._channel.process(narrow)
        if channel.size == 0:
            return np.zeros(0, dtype=np.float32)

        # Measured after the channel filter, so it is the level of the signal actually
        # being listened to rather than of everything in the span.
        power = float(np.mean(np.abs(channel) ** 2))
        self.channel_dbfs = 10.0 * np.log10(power + 1e-30)

        squelched = (
            self.squelch_dbfs is not None
            and self.spec.squelch_capable
            and self.channel_dbfs < self.squelch_dbfs
        )

        if self._detector is not None:
            audio = self._detector.process(channel)
        else:
            audio = channel.real          # SSB: the sideband filter did the work

        if self._deemph is not None:
            audio = self._deemph.process(audio)
        if self.audio_decim > 1:
            audio = self._audio_decimator.process(audio)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
        if self._audio_fir is not None:
            audio = self._audio_fir.process(audio)

        if squelched:
            self.muted_blocks += 1
            return np.zeros(audio.size, dtype=np.float32)

        if self._agc is not None:
            audio = self._agc.process(audio)

        return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)

    @property
    def agc_gain(self) -> float:
        return self._agc.gain if self._agc is not None else 1.0
