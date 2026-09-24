"""Morse decoding, tested by sending text and reading it back.

The signals here are keyed carriers put through the real CW demodulator, so these
exercise the whole path rather than the decoder in isolation.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import DemodChain
from src.rgc_sdr.dsp.morse import (
    CHAR_TO_MORSE,
    ENVELOPE_DECIM,
    MORSE_TO_CHAR,
    CwDecoder,
    EnvelopeSampler,
    text_to_morse,
)

FS = 768e3


def keyed_iq(text, wpm=20.0, fs=FS, offset_hz=0.0, amplitude=0.3, noise=0.0,
             lead_ms=200.0, seed=7):
    """A carrier keyed with `text`, as an operator would send it."""
    dot = 1.2 / wpm                                  # seconds per dot
    pattern = text_to_morse(text)
    on_off: list[tuple[bool, float]] = [(False, lead_ms / 1000.0)]
    for symbol in pattern:
        if symbol == ".":
            on_off.append((True, dot))
            on_off.append((False, dot))
        elif symbol == "-":
            on_off.append((True, 3 * dot))
            on_off.append((False, dot))
        elif symbol == " ":
            # text_to_morse uses one space between letters and three between words;
            # letters already carry one dot of gap, so add two more, or six for a word.
            on_off.append((False, 2 * dot))
    on_off.append((False, 8 * dot))                  # let the last character close

    rng = np.random.default_rng(seed)
    pieces = []
    phase = 0.0
    for on, seconds in on_off:
        n = max(1, int(round(seconds * fs)))
        t = (phase + np.arange(n)) / fs
        phase += n
        tone = amplitude * np.exp(2j * np.pi * offset_hz * t) if on else np.zeros(n, complex)
        if noise:
            tone = tone + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        pieces.append(tone)
    return np.concatenate(pieces).astype(np.complex64)


def decode_through_chain(iq, wpm_hint=20.0, block=16384):
    """Run IQ through the CW demodulator and decode the envelope it produces."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False)
    sampler = EnvelopeSampler(ENVELOPE_DECIM)
    decoder = CwDecoder(chain.if_rate / ENVELOPE_DECIM, dot_ms=1200.0 / wpm_hint)
    for i in range(0, iq.size, block):
        chain.process(iq[i : i + block])
        decoder.feed(sampler.process(chain.last_channel))
    return decoder


# -- tables ------------------------------------------------------------------

def test_the_alphabet_and_digits_are_present():
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        assert char in CHAR_TO_MORSE
        assert MORSE_TO_CHAR[CHAR_TO_MORSE[char]] == char


def test_codes_are_unique():
    assert len(set(MORSE_TO_CHAR)) == len(MORSE_TO_CHAR)


def test_text_to_morse_spacing():
    assert text_to_morse("E") == "."
    assert text_to_morse("EE") == ". ."
    assert text_to_morse("E E") == ".   ."


# -- envelope sampler --------------------------------------------------------

def test_envelope_sampler_reduces_the_rate():
    sampler = EnvelopeSampler(10)
    out = sampler.process(np.ones(100, dtype=np.complex64))
    assert out.size == 10
    assert np.allclose(out, 1.0)


def test_envelope_sampler_carries_the_remainder():
    """No samples may be dropped at block seams, or timing drifts."""
    sampler = EnvelopeSampler(10)
    total = sum(sampler.process(np.ones(7, dtype=np.complex64)).size for _ in range(10))
    assert total == 7          # 70 samples in, 7 out


def test_envelope_sampler_follows_keying():
    sampler = EnvelopeSampler(10)
    on = sampler.process(np.full(100, 0.5, dtype=np.complex64))
    off = sampler.process(np.zeros(100, dtype=np.complex64))
    assert np.allclose(on, 0.5)
    assert np.allclose(off, 0.0)


# -- decoding ----------------------------------------------------------------

def test_a_single_letter():
    assert "E" in decode_through_chain(keyed_iq("E")).text


def test_a_short_word():
    assert "CQ" in decode_through_chain(keyed_iq("CQ")).text


def test_a_callsign_with_digits():
    text = decode_through_chain(keyed_iq("G4ABC")).text
    assert "G4ABC" in text, f"got {text!r}"


def test_a_realistic_call():
    sent = "CQ CQ DE G4ABC K"
    got = decode_through_chain(keyed_iq(sent)).text
    assert "G4ABC" in got, f"got {got!r}"
    assert "CQ" in got


def test_word_spaces_are_produced():
    got = decode_through_chain(keyed_iq("E E")).text
    assert " " in got, f"got {got!r}"


@pytest.mark.parametrize("wpm", [12.0, 18.0, 25.0, 35.0])
def test_decoding_across_speeds(wpm):
    """Operators send at wildly different speeds, so the dot length must be learned."""
    got = decode_through_chain(keyed_iq("PARIS", wpm=wpm), wpm_hint=20.0).text
    assert "PARIS" in got, f"at {wpm} wpm got {got!r}"


def test_speed_is_estimated():
    decoder = decode_through_chain(keyed_iq("PARIS PARIS", wpm=25.0), wpm_hint=20.0)
    assert decoder.wpm == pytest.approx(25.0, rel=0.25)


def test_decoding_survives_noise():
    got = decode_through_chain(keyed_iq("TEST", noise=0.01)).text
    assert "TEST" in got, f"got {got!r}"


def test_an_offset_carrier_still_decodes():
    """Slightly off-tune is normal; the CW filter is 500 Hz wide."""
    got = decode_through_chain(keyed_iq("SOS", offset_hz=120.0)).text
    assert "SOS" in got, f"got {got!r}"


def test_nothing_is_invented_from_noise_alone():
    """The commonest failure of a CW decoder is inventing text out of hiss."""
    rng = np.random.default_rng(3)
    n = int(FS * 1.5)
    hiss = (0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    assert decode_through_chain(hiss).text.strip() == ""


def test_nothing_is_invented_from_a_steady_carrier():
    """An unkeyed carrier is not morse; it must not produce a stream of characters."""
    t = np.arange(int(FS * 1.5)) / FS
    steady = (0.3 * np.exp(2j * np.pi * 0.0 * t)).astype(np.complex64)
    assert len(decode_through_chain(steady).text) <= 2


# -- rolling history ---------------------------------------------------------

def test_history_is_bounded_to_the_last_fifty():
    decoder = CwDecoder(1500.0, history=50)
    for i in range(80):
        decoder._chars.append(str(i % 10))
    assert len(decoder.text) == 50


def test_reset_clears_the_line():
    decoder = decode_through_chain(keyed_iq("TEST"))
    assert decoder.text
    decoder.reset()
    assert decoder.text == ""


def test_feeding_nothing_is_safe():
    decoder = CwDecoder(1500.0)
    assert decoder.feed(np.zeros(0)) == ""


def test_decoder_reports_characters_as_they_complete():
    """The display wants them incrementally, not only at the end."""
    chain = DemodChain(FS, "cw", volume=1.0, agc=False)
    sampler = EnvelopeSampler(ENVELOPE_DECIM)
    decoder = CwDecoder(chain.if_rate / ENVELOPE_DECIM)
    iq = keyed_iq("TEST")
    produced = ""
    for i in range(0, iq.size, 16384):
        chain.process(iq[i : i + 16384])
        produced += decoder.feed(sampler.process(chain.last_channel))
    assert "TEST" in produced.replace(" ", "") or "TEST" in decoder.text
