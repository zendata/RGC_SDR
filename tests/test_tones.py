"""CTCSS and DCS: encoded by the transmit chain, received by the NBFM chain, decoded.

The DCS codeword itself is checked against published tables (inverse pairs), not only
against this module's own decoder.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import DemodChain
from src.rgc_sdr.dsp.modulate import Modulator
from src.rgc_sdr.dsp.tones import (
    DCS_CODES, ToneSquelch, dcs_bits, dcs_codeword,
)
from src.rgc_sdr.repeater import CTCSS_TONES

IQ_RATE = 2.4e6


def _rotations(word):
    return {((word >> k) | (word << (23 - k))) & 0x7FFFFF for k in range(23)}


def test_dcs_023_is_the_published_codeword():
    assert dcs_codeword("023") == 0x763813


@pytest.mark.parametrize("code,inverse", [("023", "047"), ("025", "244")])
def test_dcs_inverse_pairs_match_the_published_tables(code, inverse):
    flipped = ~dcs_codeword(code) & 0x7FFFFF
    assert flipped in _rotations(dcs_codeword(inverse))


def test_dcs_is_sent_least_significant_bit_first():
    bits = dcs_bits("023")
    assert bits[:9].tolist() == [1, 1, 0, 0, 1, 0, 0, 0, 0]    # 023 octal = 0b000010011
    assert bits[9:12].tolist() == [0, 0, 1]                      # the fixed 100, backwards


def test_standard_lists():
    assert len(DCS_CODES) == 104 and len(set(DCS_CODES)) == 104
    assert CTCSS_TONES[0] == 67.0 and CTCSS_TONES[-1] == 254.1


def received_audio(tone, seconds=1.5, voice=True):
    """NBFM with a 1 kHz 'voice' and the given tone, through the receiver."""
    t = np.arange(int(48e3 * seconds)) / 48e3
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t) if voice else np.zeros(t.size)
    mod = Modulator("nbfm", IQ_RATE, tone=tone)
    iq = np.concatenate([mod.process(audio[i:i + 1024]) for i in range(0, audio.size, 1024)])
    chain = DemodChain(IQ_RATE, "nbfm", volume=1.0, agc=False)
    out = [chain.process(iq[i:i + 65536]) for i in range(0, iq.size, 65536)]
    return np.concatenate([o for o in out if o.size]), chain.audio_rate


def opens(audio, rate, kind, value):
    squelch = ToneSquelch(rate, kind, value)
    for i in range(0, audio.size, 2048):
        squelch.process(audio[i:i + 2048])
    return squelch.open


@pytest.mark.parametrize("hz", [67.0, 88.5, 123.0, 254.1])
def test_ctcss_opens_on_its_own_tone(hz):
    audio, rate = received_audio(("ctcss", hz))
    assert opens(audio, rate, "ctcss", hz)


def test_ctcss_tells_neighbouring_tones_apart():
    audio, rate = received_audio(("ctcss", 67.0))
    assert not opens(audio, rate, "ctcss", 69.3)


def test_ctcss_stays_shut_without_a_tone():
    audio, rate = received_audio(None)
    assert not opens(audio, rate, "ctcss", 88.5)


@pytest.mark.parametrize("code", ["023", "205", "754"])
def test_dcs_opens_on_its_own_code(code):
    audio, rate = received_audio(("dcs", code))
    assert opens(audio, rate, "dcs", code)


def test_dcs_rejects_other_codes_including_the_inverse():
    audio, rate = received_audio(("dcs", "023"))
    for other in ("025", "047", "754"):
        assert not opens(audio, rate, "dcs", other), other


def test_dcs_stays_shut_on_a_ctcss_signal():
    audio, rate = received_audio(("ctcss", 100.0))
    assert not opens(audio, rate, "dcs", "023")


def test_tones_are_only_for_nbfm():
    with pytest.raises(ValueError, match="NBFM"):
        Modulator("am", IQ_RATE, tone=("ctcss", 88.5))


# -- tone squelch inside the NBFM receive chain --------------------------------------

def chain_output(tx_tone, rx_tone, seconds=2.0):
    t = np.arange(int(48e3 * seconds)) / 48e3
    mod = Modulator("nbfm", IQ_RATE, tone=tx_tone)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    iq = np.concatenate([mod.process(audio[i:i + 1024]) for i in range(0, audio.size, 1024)])
    chain = DemodChain(IQ_RATE, "nbfm", volume=1.0, agc=False)
    chain.set_tone_squelch(rx_tone)
    out = [chain.process(iq[i:i + 65536]) for i in range(0, iq.size, 65536)]
    return np.concatenate([o for o in out if o.size]), chain


def test_chain_is_silent_without_the_right_tone():
    for tx in (None, ("ctcss", 100.0), ("dcs", "047")):
        out, chain = chain_output(tx, ("dcs", "023"))
        assert np.max(np.abs(out)) == 0.0, tx
        assert chain.tone_open is False


def test_chain_passes_voice_with_the_right_tone_and_filters_the_tone_out():
    out, chain = chain_output(("ctcss", 88.5), ("ctcss", 88.5))
    assert chain.tone_open
    tail = out[out.size // 2:]
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(tail.size)))
    freqs = np.fft.rfftfreq(tail.size, 1 / chain.audio_rate)
    voice = spectrum[np.abs(freqs - 1000) < 20].max()
    tone = spectrum[np.abs(freqs - 88.5) < 5].max()
    assert 20 * np.log10(voice / tone) > 30.0, "the CTCSS tone is audible"


def test_tone_squelch_is_ignored_outside_nbfm():
    chain = DemodChain(IQ_RATE, "am")
    chain.set_tone_squelch(("ctcss", 88.5))
    assert chain.tone_open is None
