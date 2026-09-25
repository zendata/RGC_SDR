"""Transmit-side DSP: microphone audio to complex baseband IQ.

Groundwork for transmitting with a HackRF (PLANNING.md, P6). Nothing here touches a
radio: it turns audio into the IQ a transmit stream would carry, and every mode is
checked by demodulating its own output with the receive chain.

    audio (48 kHz) -> limit -> band-limit -> [pre-emphasis] -> modulate at a baseband
    rate -> integer interpolation to the radio's IQ rate

All modes except CW, by request: CW wants a keyer and shaped keying envelopes, not a
microphone.

The IQ rate must be a whole multiple of the mode's baseband rate (48 kHz, or 240 kHz for
broadcast FM), so interpolation is by an integer and needs no fractional resampler. The
HackRF takes any rate from 2 to 20 MS/s, so `nearest_tx_rate` can always find one:
1.92 MS/s (40 x 48 kHz, 8 x 240 kHz) suits every mode.

Pure NumPy/SciPy, no Qt, no device access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .decimate import lowpass_taps
from .filters import Fir, fir_length_for

#: Modes that can be transmitted. CW is deliberately absent.
TX_MODES = ("am", "nbfm", "wbfm", "usb", "lsb")

#: The microphone's rate. CoreAudio gives the MacBook Air Microphone 48 kHz natively.
AUDIO_RATE = 48_000.0

#: CTCSS/DCS share of full deviation. 15% is typical: enough for a repeater's decoder,
#: not enough to eat into the voice.
TONE_LEVEL = 0.15


@dataclass(frozen=True)
class TxModeSpec:
    """Per-mode transmit plan: audio passband, modulation rate and parameters."""

    audio_low_hz: float
    audio_high_hz: float
    #: Rate the modulator runs at; a whole multiple of AUDIO_RATE.
    baseband_rate: float
    deviation_hz: float = 0.0
    preemphasis_s: float | None = None


TX_SPECS: dict[str, TxModeSpec] = {
    # Speech bandwidth: 300-3000 Hz keeps an AM or FM signal inside a narrow channel.
    "am": TxModeSpec(300.0, 3000.0, AUDIO_RATE),
    "nbfm": TxModeSpec(300.0, 3000.0, AUDIO_RATE, deviation_hz=2.5e3),
    # Mono broadcast FM: 75 kHz deviation needs ~180 kHz (Carson), so 240 kHz.
    "wbfm": TxModeSpec(30.0, 15e3, 5 * AUDIO_RATE, deviation_hz=75e3, preemphasis_s=50e-6),
    # SSB: the usual 2.4 kHz of speech, 300-2700 Hz.
    "usb": TxModeSpec(300.0, 2700.0, AUDIO_RATE),
    "lsb": TxModeSpec(300.0, 2700.0, AUDIO_RATE),
}


def nearest_tx_rate(mode: str, wanted_hz: float) -> float:
    """The IQ rate nearest `wanted_hz` that `mode` can interpolate to exactly."""
    base = TX_SPECS[mode].baseband_rate
    return max(1, round(wanted_hz / base)) * base


class Interpolator:
    """Raise the sample rate by an integer factor, seamlessly across blocks.

    Zero-stuffing followed by a low-pass, done polyphase by `scipy.signal.upfirdn`. Each
    block is prefixed with enough of the previous one to fill the filter, so the output
    is exactly what one long call would have produced.
    """

    def __init__(self, factor: int) -> None:
        from scipy.signal import upfirdn

        self._upfirdn = upfirdn
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("factor must be >= 1")
        cutoff = 0.45 / self.factor
        # Scaled by the factor: zero-stuffing spreads the energy over `factor` samples.
        self._taps = lowpass_taps(cutoff, fir_length_for(cutoff)) * self.factor
        self._keep = math.ceil(self._taps.size / self.factor)
        self._history: np.ndarray | None = None

    def reset(self) -> None:
        self._history = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1 or x.size == 0:
            return x
        if self._history is None:
            self._history = np.zeros(self._keep, dtype=x.dtype)
        extended = np.concatenate([self._history.astype(x.dtype), x])
        y = self._upfirdn(self._taps, extended, up=self.factor)
        start = self._keep * self.factor
        self._history = extended[-self._keep:]
        return y[start: start + x.size * self.factor]


class Preemphasis:
    """First-order treble boost, the inverse of the receiver's de-emphasis.

    (1 + s*tau) on its own rises without limit, so it is shelved off at a corner well
    above the audio band. Bilinear transform, state carried across blocks.
    """

    def __init__(self, sample_rate: float, tau_s: float) -> None:
        from scipy.signal import lfilter

        self._lfilter = lfilter
        fs = float(sample_rate)
        tau_shelf = 1.0 / (2 * np.pi * 0.4 * fs)
        k = 2.0 * fs
        b = np.array([1 + k * tau_s, 1 - k * tau_s])
        a = np.array([1 + k * tau_shelf, 1 - k * tau_shelf])
        self._b, self._a = b / a[0], a / a[0]
        self._zi = np.zeros(1)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        out, self._zi = self._lfilter(self._b, self._a, x, zi=self._zi)
        return out


def _audio_bandpass(low_hz: float, high_hz: float, rate: float) -> Fir:
    """Real band-pass: a low-pass at `high` minus one at `low`, equal lengths."""
    low, high = low_hz / rate, min(high_hz / rate, 0.45)
    n = fir_length_for(low, maximum=1023)
    return Fir(lowpass_taps(high, n) - lowpass_taps(low, n))


class Modulator:
    """Microphone audio in, IQ at `iq_rate` out, for one transmit mode.

    Output magnitude never exceeds 1.0, which is full scale for a CF32 transmit stream.
    """

    def __init__(
        self,
        mode: str,
        iq_rate: float,
        audio_rate: float = AUDIO_RATE,
        am_depth: float = 0.8,
        tone: tuple[str, object] | None = None,
    ) -> None:
        if mode == "cw":
            raise ValueError("CW transmit is not supported: it needs a keyer, not a microphone")
        if mode not in TX_SPECS:
            raise ValueError(f"cannot transmit {mode!r}; expected one of {TX_MODES}")
        self.mode = mode
        self.spec = TX_SPECS[mode]
        self.audio_rate = float(audio_rate)
        self.iq_rate = float(iq_rate)
        base = self.spec.baseband_rate

        audio_up = base / self.audio_rate
        iq_up = self.iq_rate / base
        if audio_up != int(audio_up):
            raise ValueError(f"{mode} needs audio at a divisor of {base:g} Hz")
        if iq_up != int(iq_up) or iq_up < 1:
            raise ValueError(
                f"{mode} needs an IQ rate that is a multiple of {base / 1e3:g} kHz; "
                f"nearest is {nearest_tx_rate(mode, self.iq_rate) / 1e6:g} MS/s"
            )

        self.am_depth = float(am_depth)
        if tone is not None and mode != "nbfm":
            raise ValueError("CTCSS and DCS are NBFM features")
        #: ("ctcss", Hz) or ("dcs", "023"), sent under the voice; None for neither.
        self.tone = tone
        self._tone = None
        if tone is not None:
            from .tones import tone_generator

            self._tone = tone_generator(self.audio_rate, *tone)
        self._band = _audio_bandpass(self.spec.audio_low_hz, self.spec.audio_high_hz,
                                     self.audio_rate)
        self._pre = (Preemphasis(self.audio_rate, self.spec.preemphasis_s)
                     if self.spec.preemphasis_s else None)
        self._audio_up = Interpolator(int(audio_up))
        self._iq_up = Interpolator(int(iq_up))
        self._phase = 0.0
        self._ssb = None
        if mode in ("usb", "lsb"):
            from .demod import sideband_taps

            # The receiver's own sideband filter, applied to real audio, leaves one side.
            self._ssb = Fir(sideband_taps(self.spec.audio_high_hz, base, upper=(mode == "usb")))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Modulate a block of mono audio in -1..1. Returns complex64 IQ."""
        if audio.size == 0:
            return np.zeros(0, dtype=np.complex64)
        a = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
        a = self._band.process(a)
        if self._pre is not None:
            a = self._pre.process(a)
        if self._tone is not None:
            # Added after the 300 Hz high-pass, which keeps the voice out of the tone's
            # band. 15% of the deviation (375 Hz of 2.5 kHz), the voice taking the rest.
            a = (1.0 - TONE_LEVEL) * a + TONE_LEVEL * self._tone.process(a.size)
        a = self._audio_up.process(a)
        base = self.spec.baseband_rate

        if self.mode == "am":
            bb = ((1.0 + self.am_depth * np.clip(a, -1.0, 1.0)) / (1.0 + self.am_depth))
            bb = bb.astype(np.complex128)
        elif self.mode in ("nbfm", "wbfm"):
            step = 2 * np.pi * self.spec.deviation_hz / base
            phase = self._phase + np.cumsum(step * np.clip(a, -1.0, 1.0))
            self._phase = float(phase[-1] % (2 * np.pi))
            bb = np.exp(1j * phase)
        else:
            # A real signal's one side carries half its amplitude, hence the 2.
            bb = 2.0 * self._ssb.process(a)

        iq = self._iq_up.process(bb)
        # Full scale is 1.0. Interpolation ringing can poke just over; hold it there.
        peak = np.abs(iq)
        over = peak > 1.0
        if np.any(over):
            iq[over] /= peak[over]
        return iq.astype(np.complex64)
