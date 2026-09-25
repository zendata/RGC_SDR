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
from .filters import Fir, bandpass_taps, fir_length_for  # noqa: F401  (re-exported)

#: Modes the UI offers, in the order the roadmap introduces them.
MODES = ("am", "nbfm", "wbfm", "usb", "lsb", "cw")


#: Channel widths offered per mode, in Hz. The mode's own spec value is the default.
BANDWIDTH_PRESETS: dict[str, tuple[float, ...]] = {
    "am": (3e3, 4.5e3, 6e3, 9e3, 12e3, 16e3),
    "nbfm": (6e3, 8e3, 12.5e3, 16e3, 25e3),
    "wbfm": (150e3, 180e3, 200e3, 250e3),
    "usb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
    "lsb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
    # CW filters are narrow: the signal is an on/off carrier, so bandwidth buys nothing
    # but noise. 250-500 Hz is typical, and 100 Hz is for digging one signal out of a pile.
    "cw": (100.0, 250.0, 500.0, 800.0, 1.5e3),
}

#: Beat-note pitches offered for CW, in Hz. Operator preference varies a lot, and a
#: pitch that suits one pair of ears is fatiguing to another.
CW_PITCHES: tuple[float, ...] = (400.0, 450.0, 500.0, 550.0, 600.0, 700.0, 800.0)


#: FM broadcast de-emphasis. 50 us is the standard in Europe, Africa, Asia and
#: Australia; the Americas and South Korea use 75 us. The chain used 75 us until the
#: receiver was found to be in Australia (Melbourne), where that over-cuts the treble.
DEEMPHASIS_S = 50e-6

#: The FM multiplex must reach 57 kHz for RDS, so it is kept at 150 kHz or more.
MPX_TARGET_HZ = 150e3


#: AM audio level per unit of modulation depth: a 100% modulated peak reaches this.
AM_AUDIO_GAIN = 0.5
#: Ceiling on the carrier-referenced gain, so exact silence cannot divide by nothing.
AM_MAX_GAIN = 1e7
#: How long the audio AGC holds its gain through a pause before recovering.
AGC_HANG_S = 0.6


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
    #: CW only. A keyed carrier tuned exactly produces DC, which is silent, so the chain
    #: mixes it to this audio pitch -- the job a BFO does in a conventional receiver.
    pitch_hz: float = 0.0


MODE_SPECS: dict[str, ModeSpec] = {
    # Double-sideband AM broadcast: 9 kHz channel spacing in ITU regions 1 and 3.
    "am": ModeSpec(bandwidth_hz=9e3, if_target_hz=48e3, squelch_capable=True),
    "nbfm": ModeSpec(
        bandwidth_hz=12.5e3, if_target_hz=48e3, deviation_hz=2.5e3,
        audio_cutoff_hz=3.4e3, squelch_capable=True,
    ),
    # Broadcast FM, in stereo with RDS. The IF runs at 300 kHz or more so a full 200 kHz
    # channel fits inside the decimation cascade's passband (it keeps about +/-0.41 of the
    # output rate); the stereo subcarrier and RDS live in the upper part of the
    # multiplex, and a narrower channel strips them first.
    "wbfm": ModeSpec(
        bandwidth_hz=200e3, if_target_hz=300e3, audio_target_hz=44e3,
        deviation_hz=75e3, deemphasis_s=DEEMPHASIS_S, audio_cutoff_hz=15e3,
        squelch_capable=True,
    ),
    "usb": ModeSpec(bandwidth_hz=2.7e3, if_target_hz=48e3),
    "lsb": ModeSpec(bandwidth_hz=2.7e3, if_target_hz=48e3),
    # 500 Hz rather than the traditional 700: lower notes are markedly less tiring
    # over a long session, and it is selectable anyway.
    "cw": ModeSpec(bandwidth_hz=500.0, if_target_hz=48e3, pitch_hz=500.0),
}


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


def channel_taps(bandwidth_hz: float, sample_rate: float) -> np.ndarray:
    """Real low-pass keeping +/- bandwidth/2 of a complex signal."""
    cutoff = min(bandwidth_hz / 2.0 / sample_rate, 0.49)
    return lowpass_taps(cutoff, fir_length_for(cutoff))


def sideband_taps(bandwidth_hz: float, sample_rate: float, upper: bool) -> np.ndarray:
    """Complex band-pass isolating one sideband.

    With the carrier at 0 Hz the upper sideband occupies 0..+B and the lower -B..0, so a
    real low-pass cannot separate them -- it is symmetric. A band-pass centred on +/-B/2
    passes only one side.
    """
    half = min(bandwidth_hz / 2.0, 0.24 * sample_rate)
    centre = half if upper else -half
    return bandpass_taps(centre, bandwidth_hz, sample_rate)


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

    @property
    def carrier(self) -> float:
        """Smoothed carrier amplitude: the envelope's mean, which the DC blocker removes."""
        return self._dc or 0.0

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

    Asymmetric, as an AGC should be. It backs off at once when a signal gets louder, so
    it cannot blast; the change is ramped across the block so it does not click. It then
    *holds* its gain through a pause (`hang_samples`) before recovering slowly, so the
    gaps between words do not pump the noise up and make the next word start loud -- the
    fault heard on SSB speech with an attack of 0.35 per block and no hang. `max_gain`
    stops it amplifying pure noise to full scale on a dead channel.

    AM does not use this: see DemodChain, which references AM to its carrier instead.
    """

    def __init__(
        self,
        target_rms: float = 0.15,
        attack: float = 1.0,
        decay: float = 0.02,
        max_gain: float = 20000.0,
        floor_rms: float = 1e-7,
        hang_samples: int = 0,
    ) -> None:
        self.target_rms = float(target_rms)
        self.attack = float(attack)
        self.decay = float(decay)
        self.max_gain = float(max_gain)
        self.floor_rms = float(floor_rms)
        self.hang_samples = int(hang_samples)
        self._hang = 0
        self._gain = 1.0
        self._primed = False

    @property
    def gain(self) -> float:
        return self._gain

    def reset(self) -> None:
        self._gain = 1.0
        self._primed = False
        self._hang = 0

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
            self._hang = self.hang_samples
            return audio * self._gain
        previous = self._gain
        if wanted <= self._gain * 1.5:
            # A signal at (or near) full level: restart the hang, and attack if louder.
            self._hang = self.hang_samples
            if wanted < self._gain:
                self._gain += self.attack * (wanted - self._gain)
        elif self._hang > 0:
            # A pause between words: hold, rather than winding the gain up.
            self._hang -= len(audio)
        else:
            self._gain += self.decay * (wanted - self._gain)
        if self._gain == previous:
            return audio * self._gain
        ramp = np.linspace(previous, self._gain, len(audio), endpoint=True)
        if audio.ndim == 2:
            ramp = ramp[:, None]
        return audio * ramp


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
        pitch_hz: float | None = None,
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

        # Only CW has a beat note. The UI passes its pitch whatever the mode, and it once
        # reached SSB too: USB then passed -500..+2200 Hz while the display shaded
        # 0..+2700, and every voice came out 500 Hz high.
        self.pitch_hz = self.spec.pitch_hz
        if mode == "cw" and pitch_hz is not None:
            self.pitch_hz = float(pitch_hz)
        self._user_offset = float(offset_hz)
        # The BFO is folded into the mixer, so the UI's offset keeps meaning "where I am
        # listening" rather than having to know about CW's pitch.
        self._mixer = Mixer(self.sample_rate, self._mix_offset())
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
            # SSB and CW need no detector: the filter did the work and the real part of
            # the result is the audio.
            self._detector = None

        self._deemph = (
            Deemphasis(self.if_rate, self.spec.deemphasis_s)
            if self.spec.deemphasis_s
            else None
        )

        #: 2 for broadcast FM, which is always delivered as stereo -- duplicated mono when
        #: the station sends no pilot -- so the audio stream never changes shape mid-way.
        self.channels = 2 if mode == "wbfm" else 1
        self._stereo = None
        self._rds = None
        audio_source_rate = self.if_rate
        if mode == "wbfm":
            from .fmstereo import StereoDecoder
            from .rds import RdsDemodulator

            self.mpx_decim = self._pick_factor(self.if_rate, MPX_TARGET_HZ)
            self.mpx_rate = self.if_rate / self.mpx_decim
            self._mpx_decimator = StreamDecimator(self.mpx_decim)
            self._stereo = StereoDecoder(self.mpx_rate)
            self._rds = RdsDemodulator(self.mpx_rate)
            audio_source_rate = self.mpx_rate
            # De-emphasis belongs on left and right, *after* the stereo matrix. On the
            # whole multiplex it would flatten the 38 kHz subcarrier and the RDS with it.
            self._deemph_pair = [
                Deemphasis(self.mpx_rate, self.spec.deemphasis_s) for _ in range(2)
            ]
            self._deemph = None

        self.audio_decim = 1
        if self.spec.audio_target_hz is not None:
            self.audio_decim = self._pick_factor(audio_source_rate, self.spec.audio_target_hz)
        self._audio_decimator = StreamDecimator(self.audio_decim)
        self.audio_rate = audio_source_rate / self.audio_decim
        if mode == "wbfm":
            self._audio_decimator_pair = [StreamDecimator(self.audio_decim) for _ in range(2)]

        # AM is levelled by its carrier, which is steady whatever the programme does:
        # the audio becomes the modulation depth, so a pause cannot wind the gain up and
        # the next word cannot start loud. Every other mode has no carrier to go by and
        # uses the audio AGC, holding through pauses of up to AGC_HANG_S.
        self._carrier_agc = bool(agc) and mode == "am"
        self._am_gain = 1.0
        self._agc = (AudioAgc(hang_samples=int(AGC_HANG_S * self.audio_rate))
                     if agc and not self._carrier_agc else None)

        cutoff = self.spec.audio_cutoff_hz
        self._audio_fir = None
        self._audio_fir_pair = None
        if cutoff is not None and cutoff < self.audio_rate / 2.0:
            c = cutoff / self.audio_rate
            self._audio_fir = Fir(lowpass_taps(c, fir_length_for(c)))
            if mode == "wbfm":
                self._audio_fir_pair = [Fir(lowpass_taps(c, fir_length_for(c)))
                                        for _ in range(2)]

        self.muted_blocks = 0
        #: The most recent channel-filtered block, complex. CW decoding needs the
        #: envelope of this rather than the audio, which carries the beat note.
        self.last_channel = np.zeros(0, dtype=np.complex128)

    def _mix_offset(self) -> float:
        """Mixer shift, which for CW puts the carrier at the wanted audio pitch."""
        return self._user_offset - self.pitch_hz

    def _channel_taps(self) -> np.ndarray:
        if self.mode == "cw":
            # Centred on the pitch, so the keyed carrier lands inside the passband.
            return bandpass_taps(self.pitch_hz, self.bandwidth_hz, self.if_rate)
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
        """Where the user is listening, with any CW pitch already accounted for."""
        return self._user_offset

    def set_pitch(self, pitch_hz: float) -> None:
        """Move the beat note. Rebuilds the filter and re-aims the mixer together.

        Both have to change: the mixer decides where the carrier lands and the filter
        decides what is passed, so changing one alone would put the tone outside its own
        passband.
        """
        pitch_hz = float(pitch_hz)
        if self.mode != "cw" or pitch_hz == self.pitch_hz:
            return
        self.pitch_hz = pitch_hz
        self._mixer.set_offset(self._mix_offset())
        self._channel = Fir(self._channel_taps())

    def set_offset(self, offset_hz: float) -> None:
        self._user_offset = float(offset_hz)
        self._mixer.set_offset(self._mix_offset())

    def input_for_audio(self, n_audio: int) -> int:
        """Input IQ samples needed for roughly `n_audio` output samples."""
        mpx = getattr(self, "mpx_decim", 1)
        return int(n_audio * self.if_decim * mpx * self.audio_decim)

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
        if self._stereo is not None:
            mono = self._stereo.force_mono
            self._stereo.reset()
            self._stereo.force_mono = mono
            self._rds.reset()
            self._mpx_decimator.reset()
            for part in (*self._deemph_pair, *self._audio_decimator_pair,
                         *(self._audio_fir_pair or ())):
                part.reset()

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

        self.last_channel = channel
        # Measured after the channel filter, so it is the level of the signal actually
        # being listened to rather than of everything in the span.
        power = float(np.mean(np.abs(channel) ** 2))
        self.channel_dbfs = 10.0 * np.log10(power + 1e-30)

        squelched = (
            self.squelch_dbfs is not None
            and self.spec.squelch_capable
            and self.channel_dbfs < self.squelch_dbfs
        )

        if self.mode == "wbfm":
            audio = self._broadcast_fm(channel)
            if audio.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            if squelched:
                self.muted_blocks += 1
                return np.zeros(audio.shape, dtype=np.float32)
            if self._agc is not None:
                audio = self._agc.process(audio)     # one gain for both channels
            return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)

        if self._detector is not None:
            audio = self._detector.process(channel)
            if self._carrier_agc:
                self._am_gain = min(AM_AUDIO_GAIN / max(self._detector.carrier, 1e-30),
                                    AM_MAX_GAIN)
                audio = audio * self._am_gain
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

    def _broadcast_fm(self, channel: np.ndarray) -> np.ndarray:
        """Discriminate, split the multiplex into left and right, and read the RDS."""
        mpx = self._detector.process(channel)
        mpx = self._mpx_decimator.process(mpx)
        if mpx.size == 0:
            return np.zeros((0, 2))
        stereo = self._stereo.process(mpx)
        self._rds.process(stereo.mpx, stereo.pilot)
        sides = []
        for i, side in enumerate((stereo.left, stereo.right)):
            side = self._deemph_pair[i].process(side)
            if self.audio_decim > 1:
                side = self._audio_decimator_pair[i].process(side)
            if self._audio_fir_pair is not None:
                side = self._audio_fir_pair[i].process(side)
            sides.append(side)
        n = min(sides[0].size, sides[1].size)
        return np.column_stack([sides[0][:n], sides[1][:n]])

    @property
    def stereo(self) -> bool:
        """True when a broadcast FM station's pilot is present and stereo is in use."""
        return bool(self._stereo is not None and self._stereo.stereo)

    @property
    def force_mono(self) -> bool:
        return bool(self._stereo is not None and self._stereo.force_mono)

    @force_mono.setter
    def force_mono(self, value: bool) -> None:
        if self._stereo is not None:
            self._stereo.force_mono = bool(value)

    @property
    def rds(self):
        """Decoded RDS for broadcast FM, else None."""
        return self._rds.info if self._rds is not None else None

    @property
    def agc_gain(self) -> float:
        if self._carrier_agc:
            return self._am_gain
        return self._agc.gain if self._agc is not None else 1.0
