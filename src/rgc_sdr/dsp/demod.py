"""Demodulators: AM, synchronous AM, narrow/wide FM, and SSB.

Every block here is *stateful and continuous*. The display can restart its filters each
frame because it only shows the newest block, but audio is a single unbroken stream: any
discontinuity at a block boundary is an audible click or buzz at the block rate. So the
mixer carries its phase, the FIRs carry their history, and the FM detector carries its
previous sample.

Pure NumPy plus scipy for the one stateful IIR (FM de-emphasis). No Qt, no device access.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .decimate import StreamDecimator, lowpass_taps
from .filters import Fir, bandpass_taps, fir_length_for  # noqa: F401  (re-exported)

#: Modes the UI offers, in the order the roadmap introduces them.
MODES = ("am", "sam", "nbfm", "wbfm", "usb", "lsb", "cw", "p25", "dmr", "dab")


#: Channel widths offered per mode, in Hz. The mode's own spec value is the default.
BANDWIDTH_PRESETS: dict[str, tuple[float, ...]] = {
    "am": (3e3, 4.5e3, 6e3, 9e3, 12e3, 16e3),
    "sam": (3e3, 4.5e3, 6e3, 9e3, 12e3, 16e3),
    "nbfm": (6e3, 8e3, 12.5e3, 16e3, 25e3),
    "wbfm": (150e3, 180e3, 200e3, 250e3),
    "usb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
    "lsb": (1.8e3, 2.1e3, 2.4e3, 2.7e3, 3.0e3, 3.6e3),
    # CW filters are narrow: the signal is an on/off carrier, so bandwidth buys nothing
    # but noise. 250-500 Hz is typical, and 100 Hz is for digging one signal out of a pile.
    "cw": (100.0, 250.0, 500.0, 800.0, 1.5e3),
    # A P25 channel is 12.5 kHz; the voice decoder needs all of it.
    "p25": (12.5e3,),
    "dmr": (12.5e3,),
    # A DAB ensemble is 1.536 MHz, all of it needed.
    "dab": (1.536e6,),
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
    # Synchronous AM (P10): the carrier tracked and the signal demodulated coherently.
    "sam": ModeSpec(bandwidth_hz=9e3, if_target_hz=48e3, squelch_capable=True),
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
    # P25 Phase 1 digital voice (dsp/p25voice.py): the FM discriminator's four-level
    # symbols, decoded to IMBE frames and through mbelib to 8 kHz audio.
    "p25": ModeSpec(bandwidth_hz=12.5e3, if_target_hz=48e3, deviation_hz=2.5e3),
    # DMR voice (dsp/dmrvoice.py): the same four-level FSK, AMBE+2 through mbelib.
    "dmr": ModeSpec(bandwidth_hz=12.5e3, if_target_hz=48e3, deviation_hz=2.5e3),
    # DAB+ (dsp/dab.py, dsp/dabplus.py): the whole ensemble at 2.048 MS/s, a station's
    # HE-AAC through FAAD2.
    "dab": ModeSpec(bandwidth_hz=1.536e6, if_target_hz=2.048e6),
}


#: Modes whose passband the IF shift moves (FM's stays centred: shifted, the
#: discriminator would only hear one side of the deviation).
SHIFT_MODES = ("am", "sam", "usb", "lsb", "cw")
#: Modes the noise blanker, noise reduction and notches apply to: single-channel audio
#: from a demodulator of the app's own.
CLEANUP_MODES = ("am", "sam", "nbfm", "usb", "lsb", "cw")
#: Audio AGC presets: (hang seconds, decay per block). "off" uses a fixed gain.
AGC_MODES = ("fast", "medium", "slow", "off")
AGC_PRESETS = {"fast": (0.1, 0.1), "medium": (AGC_HANG_S, 0.02), "slow": (1.5, 0.005)}
SAM_SIDEBANDS = ("both", "upper", "lower")
#: Modes the AGC-off fixed gain applies to.
FIXED_GAIN_MODES = ("am", "sam", "usb", "lsb", "cw")


@dataclass(frozen=True)
class ReceiverOptions:
    """The DSP row's settings (P10, PLANNING.md 7r)."""

    #: Passband moved this far from the listening frequency (SHIFT_MODES).
    if_shift_hz: float = 0.0
    #: Noise blanker, 0 off, 1-10.
    nb_level: int = 0
    #: Noise reduction, 0 off, 1-10.
    nr_level: int = 0
    auto_notch: bool = False
    #: Manual notches, as RF offsets from the listening frequency.
    notch_offsets_hz: tuple[float, ...] = field(default_factory=tuple)
    agc_mode: str = "medium"
    #: Gain with the AGC off, dB.
    manual_gain_db: float = 60.0
    sam_sideband: str = "both"
    #: NBFM: squelch on the noise above the voice instead of on level.
    noise_squelch: bool = False
    #: Noise squelch: opens when the noise is this far below a dead channel's.
    quieting_db: float = 10.0
    #: DMR: the timeslot to hear, 1 or 2, or 0 for whichever is talking.
    dmr_slot: int = 0


def passband_for(mode: str, bandwidth_hz: float, shift_hz: float = 0.0,
                 sideband: str = "both") -> tuple[float, float]:
    """(low, high) of what is heard, in Hz from the listening frequency. The same edges
    build the channel filter and shade the spectrum."""
    w = float(bandwidth_hz)
    shift = float(shift_hz) if mode in SHIFT_MODES else 0.0
    if mode == "usb":
        lo, hi = 0.0, w
    elif mode == "lsb":
        lo, hi = -w, 0.0
    elif mode == "sam" and sideband == "upper":
        lo, hi = 0.0, w / 2
    elif mode == "sam" and sideband == "lower":
        lo, hi = -w / 2, 0.0
    else:
        lo, hi = -w / 2, w / 2
    return lo + shift, hi + shift


def width_and_shift(mode: str, low_hz: float, high_hz: float,
                    sideband: str = "both") -> tuple[float, float]:
    """The inverse of `passband_for`: the width and shift giving edges `low`..`high` (as
    a dragged passband asks). FM modes stay centred."""
    lo, hi = sorted((float(low_hz), float(high_hz)))
    if mode == "usb":
        return hi - lo, lo
    if mode == "lsb":
        return hi - lo, hi
    if mode == "sam" and sideband in ("upper", "lower"):
        return 2 * (hi - lo), (lo if sideband == "upper" else hi)
    if mode in SHIFT_MODES:
        return hi - lo, (lo + hi) / 2
    return 2 * max(abs(lo), abs(hi)), 0.0


class SyncAmDetector:
    """Synchronous AM: track the carrier, remove its phase, detect coherently.

    The carrier is found in the first block by an FFT within +/-ACQUIRE_HZ, then followed
    by a frequency-locked mixer updated once a block from the carrier's own rotation.
    Its phase comes from a narrow one-pole low-pass of the derotated signal -- an IIR,
    so scipy carries its state and no Python loop runs per sample. With the carrier's
    phase removed, its sidebands are real: both are the real part, and one alone is the
    receiver's sideband filter applied to them. Through selective fading this keeps the
    audio clean where an envelope detector distorts, and choosing one sideband steps
    away from interference on the other.
    """

    ACQUIRE_HZ = 1000.0
    CARRIER_HZ = 30.0
    FLL_GAIN = 0.3
    #: Below this ratio of carrier to signal for LOST_BLOCKS blocks, find it again.
    LOCK_RATIO = 0.15
    LOST_BLOCKS = 20

    def __init__(self, sample_rate: float, bandwidth_hz: float, sideband: str = "both"):
        from scipy.signal import lfilter

        self._lfilter = lfilter
        self.rate = float(sample_rate)
        a = float(np.exp(-2 * np.pi * self.CARRIER_HZ / self.rate))
        self._b, self._a = np.array([1.0 - a]), np.array([1.0, -a])
        self.sideband = sideband
        self._ssb = None
        if sideband in ("upper", "lower"):
            self._ssb_taps = sideband_taps(bandwidth_hz / 2, self.rate, sideband == "upper")
            from .filters import Fir as _Fir

            self._ssb = _Fir(self._ssb_taps)
        self.reset()

    def reset(self) -> None:
        #: Carrier frequency found, Hz from the listening frequency; None until found.
        self.freq_hz: float | None = None
        self._phase = 0.0
        self._zi = np.zeros(1, dtype=np.complex128)
        self._lost = 0
        self.carrier = 0.0
        if self._ssb is not None:
            self._ssb.reset()

    @property
    def locked(self) -> bool:
        return self.freq_hz is not None and self._lost == 0

    def _acquire(self, x: np.ndarray) -> None:
        n = 1 << max(13, int(np.ceil(np.log2(x.size))))
        spectrum = np.abs(np.fft.fft(x * np.hanning(x.size), n))
        freqs = np.fft.fftfreq(n, 1.0 / self.rate)
        near = np.abs(freqs) <= self.ACQUIRE_HZ
        self.freq_hz = float(freqs[near][int(np.argmax(spectrum[near]))])
        self._zi = np.zeros(1, dtype=np.complex128)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0)
        if self.freq_hz is None:
            self._acquire(x)
        step = -2.0 * np.pi * self.freq_hz / self.rate
        phase = self._phase + step * np.arange(x.size)
        self._phase = float((self._phase + step * x.size) % (2 * np.pi))
        z = x * np.exp(1j * phase)
        c, self._zi = self._lfilter(self._b, self._a, z, zi=self._zi)
        if c.size > 1:
            # The carrier's remaining rotation, per sample, steers the mixer.
            turn = np.angle(np.sum(c[1:] * np.conj(c[:-1])))
            self.freq_hz += self.FLL_GAIN * turn * self.rate / (2 * np.pi)
        magnitude = np.abs(c)
        self.carrier = float(magnitude.mean())
        level = float(np.sqrt(np.mean(np.abs(z) ** 2)))
        if level > 0 and self.carrier < self.LOCK_RATIO * level:
            self._lost += 1
            if self._lost >= self.LOST_BLOCKS:
                self.freq_hz, self._lost = None, 0
        else:
            self._lost = 0
        unit = c / np.maximum(magnitude, 1e-30)
        base = z * np.conj(unit) - magnitude          # carrier removed, sidebands real
        if self._ssb is None:
            return base.real
        return 2.0 * self._ssb.process(base).real


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
        ramp_attack: bool = True,
    ) -> None:
        self.target_rms = float(target_rms)
        #: Ease a cut in gain across the block (no click), or make it at once: decoded
        #: digital voice is levelled per codec frame, and a frame louder than the last
        #: wants its gain from its first sample.
        self.ramp_attack = bool(ramp_attack)
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

    @property
    def primed(self) -> bool:
        return self._primed

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
        if self._gain == previous or (self._gain < previous and not self.ramp_attack):
            return audio * self._gain
        ramp = np.linspace(previous, self._gain, len(audio), endpoint=True)
        if audio.ndim == 2:
            ramp = ramp[:, None]
        return audio * ramp


class AudioPacer:
    """Even audio from a decoder that delivers it in bursts (P25: 180 ms per frame; DAB+:
    120 ms superframes, half a second late). Each call returns exactly the audio its IQ
    stands for, so the IQ is read steadily in real time; playback waits until
    PACER_PREBUFFER_S is queued, and silence fills only when the queue runs dry.

    *Measured 2026-10-07:* without it DAB played for a moment and went silent -- a burst
    overfilled the audio FIFO, the worker stopped reading IQ while the FIFO was full, the
    radio's ring overran and skipped, and DAB lost its frames."""

    def __init__(self, channels: int, prebuffer: int) -> None:
        self.channels, self.prebuffer = channels, int(prebuffer)
        self._queue: list[np.ndarray] = []
        self._queued = 0
        self._owed = 0.0
        self.playing = False
        self.underruns = 0

    def reset(self) -> None:
        self._queue, self._queued, self._owed, self.playing = [], 0, 0.0, False

    def push(self, audio: np.ndarray) -> None:
        if audio.shape[0]:
            self._queue.append(audio)
            self._queued += audio.shape[0]

    def pull(self, frames: float) -> np.ndarray:
        """`frames` more frames are due (fractional, carried over)."""
        self._owed += frames
        n = int(self._owed)
        self._owed -= n
        shape = (n, self.channels) if self.channels > 1 else (n,)
        out = np.zeros(shape, dtype=np.float32)
        if not self.playing and self._queued >= self.prebuffer:
            self.playing = True
        if not self.playing or n == 0:
            return out
        got = 0
        while got < n and self._queue:
            head = self._queue[0]
            take = min(n - got, head.shape[0])
            out[got:got + take] = head[:take]
            got += take
            if take == head.shape[0]:
                self._queue.pop(0)
            else:
                self._queue[0] = head[take:]
        self._queued -= got
        if got < n:                                  # ran dry: wait for a cushion again
            self.playing = False
            self.underruns += 1
        return out


#: Audio queued before a bursty decoder (P25, DAB+) starts to play.
PACER_PREBUFFER_S = 0.3


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
        options: ReceiverOptions | None = None,
    ) -> None:
        if mode not in MODE_SPECS:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.sample_rate = float(sample_rate)
        self.mode = mode
        self.spec = MODE_SPECS[mode]
        self.options = options or ReceiverOptions()
        self._agc_enabled = bool(agc)
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
        elif mode == "sam":
            self._detector = SyncAmDetector(self.if_rate, self.bandwidth_hz,
                                            self.options.sam_sideband)
        elif mode in ("nbfm", "wbfm", "p25", "dmr"):
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
        self.channels = 2 if mode in ("wbfm", "dab") else 1
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
        #: DAB: the ensemble receiver, which makes a station's audio itself.
        self._dab = None
        self.dab_problem = ""
        if mode == "dab":
            from .dab import DabReceiver, RATE as DAB_RATE

            try:
                # Given the full rate: the receiver halves 4.096 MS/s itself.
                self._dab = DabReceiver(self.sample_rate)
            except ValueError as exc:
                self.dab_problem = str(exc)
            self.audio_rate = 48000.0
            self._pacer = AudioPacer(2, PACER_PREBUFFER_S * self.audio_rate)
            self._dab_up = None
            self.dab_messages: list = []
        #: P25's voice decoder, which makes its own 8 kHz audio from the discriminator.
        self._p25 = None
        if mode == "p25":
            from .modulate import Interpolator
            from .p25voice import AUDIO_RATE, P25Voice

            self._p25 = P25Voice(self.if_rate)
            # mbelib speaks at 8 kHz; six times that is a rate every output device takes.
            self._p25_up = Interpolator(6)
            self.audio_rate = 6 * AUDIO_RATE
            # Voice arrives in 180 ms bursts and not at all between calls: paced.
            self._pacer = AudioPacer(1, PACER_PREBUFFER_S * self.audio_rate)
        #: DMR's voice decoder: AMBE+2 at 8 kHz, paced as P25's is.
        self._dmr = None
        if mode == "dmr":
            from .dmrvoice import AUDIO_RATE as DMR_RATE, DmrVoice
            from .modulate import Interpolator

            self._dmr = DmrVoice(self.if_rate, self.options.dmr_slot)
            self._dmr_up = Interpolator(6)
            self.audio_rate = 6 * DMR_RATE
            self._pacer = AudioPacer(1, PACER_PREBUFFER_S * self.audio_rate)
        if mode == "wbfm":
            self._audio_decimator_pair = [StreamDecimator(self.audio_decim) for _ in range(2)]

        self._am_gain = 1.0
        self._build_agc()

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
        self._tone_squelch = None
        self._tone_highpass = None
        from .cleanup import AudioCleanup, NoiseBlanker

        cleanable = mode in CLEANUP_MODES
        self._blanker = (NoiseBlanker(self.sample_rate, self.options.nb_level)
                         if cleanable else None)
        self._cleanup = (AudioCleanup(self.audio_rate, self.options.nr_level,
                                      self.options.auto_notch, self._notch_audio_hz())
                         if cleanable else None)
        #: NBFM noise squelch: the discriminator's noise above the voice, and what a dead
        #: channel gives through the same filters (PLANNING.md 7r).
        self.quieting_db: float | None = None
        self._noise_open = False
        self._noise_filter = None
        self._noise_reference = None
        if mode == "nbfm":
            self._build_noise_squelch()
        #: The most recent channel-filtered block, complex. CW decoding needs the
        #: envelope of this rather than the audio, which carries the beat note.
        self.last_channel = np.zeros(0, dtype=np.complex128)
        #: The most recent raw detector output -- the NBFM discriminator or the AM
        #: envelope -- real, at `if_rate`: before the audio low-pass, AGC and squelch,
        #: which would distort or silence exactly what a data decoder needs to slice.
        self.last_detected = np.zeros(0, dtype=np.float64)

    def _mix_offset(self) -> float:
        """Mixer shift, which for CW puts the carrier at the wanted audio pitch."""
        return self._user_offset - self.pitch_hz

    def _channel_taps(self) -> np.ndarray:
        shift = self.options.if_shift_hz if self.mode in SHIFT_MODES else 0.0
        if self.mode == "cw":
            # Centred on the pitch, so the keyed carrier lands inside the passband.
            return bandpass_taps(self.pitch_hz + shift, self.bandwidth_hz, self.if_rate)
        if self.mode in ("usb", "lsb"):
            if not shift:
                return sideband_taps(self.bandwidth_hz, self.if_rate,
                                     upper=(self.mode == "usb"))
            lo, hi = passband_for(self.mode, self.bandwidth_hz, shift)
            return bandpass_taps((lo + hi) / 2, hi - lo, self.if_rate)
        if self.mode in ("am", "sam") and shift:
            # Both sidebands, wherever the shift puts them: SAM picks one afterwards.
            return bandpass_taps(shift, self.bandwidth_hz, self.if_rate)
        return channel_taps(self.bandwidth_hz, self.if_rate)

    def passband(self) -> tuple[float, float]:
        """What is heard, Hz from the listening frequency."""
        return passband_for(self.mode, self.bandwidth_hz, self.options.if_shift_hz,
                            self.options.sam_sideband)

    #: Digital voice is levelled a codec frame at a time.
    VOICE_FRAME_S = 0.02

    def _level_voice(self, audio: np.ndarray) -> np.ndarray:
        """The AGC over decoded P25 or DMR voice, which arrives a superframe at a time
        (DMR: 18 codec frames, 360 ms). Levelled whole, a call starting quietly set a high
        gain that the loud speech after it ramped down over the next 360 ms: the start of
        every call blasted (VK3RQ, 2026-10-10). Levelled per 20 ms frame, first set by
        the loudest frame, and cut at a frame's start when it is louder, it does not."""
        agc = self._agc
        if agc is None or audio.size == 0:
            return audio
        n = max(1, int(self.VOICE_FRAME_S * self.audio_rate))
        frames = [audio[i:i + n] for i in range(0, audio.size, n)]
        if not agc.primed:
            agc.process(max(frames, key=lambda f: float(np.mean(np.square(f)))))
        return np.concatenate([agc.process(f) for f in frames])

    def _build_agc(self) -> None:
        """AM and SAM are levelled by their carrier, which is steady whatever the
        programme does: the audio becomes the modulation depth, so a pause cannot wind
        the gain up and the next word cannot start loud. Every other mode has no carrier
        to go by and uses the audio AGC, holding through pauses. With the AGC off, a
        fixed gain."""
        mode = self.options.agc_mode if self.options.agc_mode in AGC_MODES else "medium"
        on = self._agc_enabled and mode != "off"
        # A fixed gain only where the audio has no level of its own: FM's discriminator
        # is already scaled to full deviation, and 60 dB on it would only clip.
        self._fixed_gain = (10.0 ** (self.options.manual_gain_db / 20.0)
                            if self._agc_enabled and mode == "off"
                            and self.mode in FIXED_GAIN_MODES else None)
        self._carrier_agc = on and self.mode in ("am", "sam")
        self._agc = None
        if on and not self._carrier_agc:
            hang_s, decay = AGC_PRESETS[mode]
            self._agc = AudioAgc(decay=decay, hang_samples=int(hang_s * self.audio_rate),
                                 ramp_attack=self.mode not in ("p25", "dmr"))

    def _notch_audio_hz(self) -> list[float]:
        """Manual notches (RF offsets from the listening frequency) as audio frequencies:
        where each lands after this mode's demodulator."""
        out = []
        for offset in self.options.notch_offsets_hz:
            if self.mode == "usb":
                f = offset
            elif self.mode == "lsb":
                f = -offset
            elif self.mode == "cw":
                f = offset + self.pitch_hz
            else:
                f = abs(offset)
            if f > 0:
                out.append(f)
        return out

    def _build_noise_squelch(self) -> None:
        """The band above the voice (8-16 kHz of discriminator output), and its power
        on a dead channel, found by running the chain's own channel filter and
        discriminator on a fixed sample of Gaussian noise. A discriminator's output on
        noise does not depend on the noise's level, so this is the reference whatever
        the radio's gain."""
        rate = self.if_rate
        high = min(16e3, 0.45 * rate)
        low = min(8e3, 0.5 * high)
        taps = lowpass_taps(high / rate, fir_length_for(low / rate)) - lowpass_taps(
            low / rate, fir_length_for(low / rate))
        self._noise_filter = Fir(taps)
        rng = np.random.default_rng(705)
        n = int(0.25 * rate)
        noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        channel = Fir(self._channel.taps).process(noise)
        detected = FmDetector(rate, self.spec.deviation_hz).process(channel)
        band = Fir(taps).process(detected)[taps.size:]
        self._noise_reference = float(np.mean(band ** 2)) + 1e-30
        self._noise_power: float | None = None

    def _noise_squelched(self, detected: np.ndarray) -> bool:
        band = self._noise_filter.process(detected)
        if band.size:
            power = float(np.mean(band ** 2)) + 1e-30
            self._noise_power = (power if self._noise_power is None
                                 else 0.5 * self._noise_power + 0.5 * power)
            self.quieting_db = 10.0 * np.log10(self._noise_reference / self._noise_power)
        if self.quieting_db is None:
            return True
        threshold = self.options.quieting_db
        # Two decibels of hysteresis, so a signal near the threshold does not chatter.
        if self._noise_open:
            self._noise_open = self.quieting_db >= threshold - 2.0
        else:
            self._noise_open = self.quieting_db >= threshold
        return not self._noise_open

    def set_options(self, options: ReceiverOptions) -> None:
        """Apply the DSP row's settings in place: only what changed is rebuilt."""
        old, self.options = self.options, options
        if (old.if_shift_hz, old.sam_sideband) != (options.if_shift_hz, options.sam_sideband):
            self._channel = Fir(self._channel_taps())
            if self.mode == "sam":
                self._detector = SyncAmDetector(self.if_rate, self.bandwidth_hz,
                                                options.sam_sideband)
            if self.mode == "nbfm":
                self._build_noise_squelch()
        if (old.agc_mode, old.manual_gain_db) != (options.agc_mode, options.manual_gain_db):
            self._build_agc()
        if self._dmr is not None:
            self._dmr.slot = int(options.dmr_slot)
        if self._blanker is not None:
            self._blanker.level = options.nb_level
        if self._cleanup is not None:
            self._cleanup.nr_level = options.nr_level
            self._cleanup.auto_notch = options.auto_notch
            self._cleanup.set_notches(self._notch_audio_hz())

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
        if self.mode == "sam":
            self._detector = SyncAmDetector(self.if_rate, self.bandwidth_hz,
                                            self.options.sam_sideband)
        if self.mode == "nbfm":
            self._build_noise_squelch()

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
        if self._cleanup is not None:
            self._cleanup.set_notches(self._notch_audio_hz())

    def set_tone_squelch(self, tone: tuple[str, object] | None) -> None:
        """Only let audio through while a CTCSS tone or DCS code is received (NBFM).

        The audio is high-passed at 300 Hz while this is on, as a radio does, so the
        tone that opens the squelch is not heard in the speaker.
        """
        if tone is None or self.mode != "nbfm":
            self._tone_squelch = None
            self._tone_highpass = None
            return
        from .tones import SUBAUDIBLE_HZ, ToneSquelch

        self._tone_squelch = ToneSquelch(self.audio_rate, *tone)
        cutoff = SUBAUDIBLE_HZ / self.audio_rate
        n = fir_length_for(cutoff, maximum=1023)
        highpass = -lowpass_taps(cutoff, n)
        highpass[n // 2] += 1.0
        self._tone_highpass = Fir(highpass)

    @property
    def tone_open(self) -> bool | None:
        """Whether the tone squelch is open; None when there is no tone squelch."""
        return None if self._tone_squelch is None else self._tone_squelch.open

    def set_offset(self, offset_hz: float) -> None:
        self._user_offset = float(offset_hz)
        self._mixer.set_offset(self._mix_offset())

    def input_for_audio(self, n_audio: int) -> int:
        """Input IQ samples needed for roughly `n_audio` output samples: the ratio of the
        rates, which is the decimation for the demodulators and also holds for P25 and
        DAB, whose decoders make their audio at a rate of their own."""
        return int(np.ceil(n_audio * self.sample_rate / self.audio_rate))

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
        if self._blanker is not None:
            self._blanker.reset()
        if self._cleanup is not None:
            self._cleanup.reset()
        if self._noise_filter is not None:
            self._noise_filter.reset()
            self._noise_power, self.quieting_db, self._noise_open = None, None, False
        if self._p25 is not None:
            self._p25.reset()
            self._p25_up.reset()
            self._pacer.reset()
        if self._dmr is not None:
            self._dmr.reset()
            self._dmr_up.reset()
            self._pacer.reset()
        if self._dab is not None:
            # A retune is another ensemble: a fresh receiver, or the last one's stations
            # stay in the list and the new ones are added to them (VK3RQ, 2026-10-07).
            from .dab import DabReceiver

            self._dab = DabReceiver(self.sample_rate)
            self._dab_up = None
            self._pacer.reset()
            self.dab_messages = []
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
        # Cleared first: a block that yields no output must not hand a decoder the
        # previous block's discriminator a second time.
        self.last_detected = np.zeros(0, dtype=np.float64)
        if iq.size == 0:
            return np.zeros(0, dtype=np.float32)

        if self._blanker is not None:
            iq = self._blanker.process(iq)
        shifted = self._mixer.process(iq)
        if self.mode == "dab":
            return self._dab_audio(shifted)
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

        if self._dmr is not None:
            self.last_detected = self._detector.process(channel)
            voice = self._dmr.process(self.last_detected)
            if voice.size:
                audio = self._level_voice(self._dmr_up.process(voice.astype(np.float64)))
                self._pacer.push(audio.astype(np.float32))
            audio = self._pacer.pull(channel.size / self.if_rate * self.audio_rate)
            return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)

        if self._p25 is not None:
            self.last_detected = self._detector.process(channel)
            voice = self._p25.process(self.last_detected)
            if voice.size:
                audio = self._p25_up.process(voice.astype(np.float64))
                if not self._p25.encrypted:
                    audio = self._level_voice(audio)
                self._pacer.push(audio.astype(np.float32))
            audio = self._pacer.pull(channel.size / self.if_rate * self.audio_rate)
            return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)

        if self._detector is not None:
            audio = self._detector.process(channel)
            if self.mode in ("nbfm", "am"):
                self.last_detected = audio
            if self._noise_filter is not None and self.options.noise_squelch:
                if self._noise_squelched(audio):
                    squelched = True
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

        if self._tone_squelch is not None:
            # Decided on the audio *with* the tone in it, before it is filtered out.
            if not self._tone_squelch.process(audio):
                squelched = True
            audio = self._tone_highpass.process(audio)

        if self._cleanup is not None:
            audio = self._cleanup.process(audio)

        if squelched:
            self.muted_blocks += 1
            return np.zeros(audio.size, dtype=np.float32)

        if self._agc is not None:
            audio = self._agc.process(audio)
        elif self._fixed_gain is not None:
            audio = audio * self._fixed_gain

        return np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)

    def _dab_audio(self, iq: np.ndarray) -> np.ndarray:
        """The ensemble to the receiver; its station's audio out at 48 kHz stereo, silence
        filling the gaps (decoding runs half a second behind, and in 120 ms bursts)."""
        pcm = np.zeros((0, 2), dtype=np.float32)
        if self._dab is not None:
            self.dab_messages += self._dab.process(iq)
            if self._dab.service is None:
                # The first DAB+ station the FIC names, so there is something to hear.
                for sid, plus in self._dab.ensemble.dab_plus.items():
                    if plus and self._dab.select(sid):
                        break
            pcm = self._dab.take_audio()
            audio = self._dab.audio
            if pcm.size and audio is not None and audio.sample_rate == 32000.0:
                if self._dab_up is None:
                    from .modulate import Interpolator

                    self._dab_up = [(Interpolator(3), StreamDecimator(2)) for _ in range(2)]
                pcm = np.column_stack([
                    dec.process(up.process(pcm[:, c].astype(np.float64)))
                    for c, (up, dec) in enumerate(self._dab_up)])
        self._pacer.push(pcm.astype(np.float32))
        out = self._pacer.pull(iq.size / self.sample_rate * self.audio_rate)
        return np.clip(out * self.volume, -1.0, 1.0).astype(np.float32)

    @property
    def dab(self):
        """The DAB ensemble receiver (stations, `select`), in DAB mode."""
        return self._dab

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
    def dmr_voice(self):
        """The DMR voice decoder, in DMR mode."""
        return self._dmr

    @property
    def p25_voice(self):
        """The P25 voice decoder (who is talking, `take_calls`), in P25 mode."""
        return self._p25

    @property
    def rds(self):
        """Decoded RDS for broadcast FM, else None."""
        return self._rds.info if self._rds is not None else None

    @property
    def agc_gain(self) -> float:
        if self._carrier_agc:
            return self._am_gain
        if self._fixed_gain is not None:
            return self._fixed_gain
        return self._agc.gain if self._agc is not None else 1.0

    @property
    def sam_locked(self) -> bool | None:
        """SAM: whether the carrier is being tracked; None in other modes."""
        return self._detector.locked if self.mode == "sam" else None

    @property
    def notched_hz(self) -> list[float]:
        """Audio frequencies the automatic notch is removing now."""
        return self._cleanup.notched_hz if self._cleanup is not None else []

    @property
    def blanked(self) -> float:
        """Share of samples the noise blanker is removing lately."""
        return self._blanker.blanked if self._blanker is not None else 0.0
