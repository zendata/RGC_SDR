"""POCSAG decoding.

The encoder below is test scaffolding. It could share a misunderstanding with the
decoder, so the BCH code is also checked against constants from the standard: the sync
and idle codewords are valid codewords, which they are only under the right generator.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import DemodChain
from src.rgc_sdr.dsp.pocsag import (
    IDLE, NUMERIC, SYNC, PocsagDecoder, alpha_text, classify, correct, numeric_text, syndromes,
)

# -- an encoder, for tests only -----------------------------------------------------


def _bch(data21: int) -> int:
    """21 data bits -> 32-bit codeword with check bits and even parity."""
    rem = data21 << 10
    for bit in range(30, 9, -1):
        if rem >> bit & 1:
            rem ^= 0b11101101001 << (bit - 10)
    word = ((data21 << 10) | rem) << 1
    return word | (word.bit_count() & 1)


def _rev(v, w):
    return int(f"{v:0{w}b}"[::-1], 2)


def _alpha_chunks(text):
    stream = "".join(f"{_rev(ord(c), 7):07b}" for c in text + "\x04")
    stream += "0" * (-len(stream) % 20)
    return [int(stream[i:i + 20], 2) for i in range(0, len(stream), 20)]


def _numeric_chunks(digits):
    nibbles = [_rev(NUMERIC.index(d), 4) for d in digits]
    nibbles += [_rev(NUMERIC.index(" "), 4)] * (-len(nibbles) % 5)
    return [int("".join(f"{n:04b}" for n in nibbles[i:i + 5]), 2)
            for i in range(0, len(nibbles), 5)]


def encode(address, function, chunks):
    """Bits of a complete transmission: preamble, then batches."""
    frame = address & 7
    words = [IDLE] * (2 * frame)
    words.append(_bch(((address >> 3) << 2 | function) << 0))
    words += [_bch(1 << 20 | c) for c in chunks]
    words += [IDLE] * (-len(words) % 16)
    bits = [int(b) for b in "10" * 288]
    for i in range(0, len(words), 16):
        for w in [SYNC] + words[i:i + 16]:
            bits += [int(b) for b in f"{w:032b}"]
    return np.array(bits, dtype=np.uint8)


def fsk_iq(bits, baud, rate=48_000.0, deviation=4500.0, invert=False, noise=0.0, seed=1):
    """2-FSK at baseband: POCSAG sends a 1 as the lower frequency."""
    sps = rate / baud
    idx = (np.arange(int(len(bits) * sps)) / sps).astype(int)
    level = np.where(bits[idx] == 1, -1.0, 1.0) * (-1 if invert else 1)
    phase = 2 * np.pi * np.cumsum(level * deviation) / rate
    iq = np.exp(1j * phase)
    if noise:
        rng = np.random.default_rng(seed)
        iq = iq + noise * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size))
    return iq.astype(np.complex64)


def discriminator(iq, rate=48_000.0):
    chain = DemodChain(rate, "nbfm", bandwidth_hz=16e3)
    out = []
    for block in np.array_split(iq, max(1, iq.size // 4096)):
        chain.process(block)
        out.append(chain.last_detected)
    return np.concatenate(out)


def decode(x, rate=48_000.0):
    dec = PocsagDecoder(rate)
    out = []
    for block in np.array_split(x, max(1, x.size // 3000)):
        out += dec.process(block)
    out += dec.process(np.zeros(int(rate)))     # a second of nothing ends the message
    return out


# -- the code itself ------------------------------------------------------------------


def test_standard_codewords_are_valid_bch_codewords():
    assert list(syndromes(np.array([SYNC, IDLE]))) == [0, 0]
    assert SYNC.bit_count() % 2 == 0 and IDLE.bit_count() % 2 == 0


def test_one_and_two_bit_errors_are_corrected():
    word = _bch(0x12345)
    for flips in ([3], [0], [5, 17], [1, 31]):
        bad = word
        for f in flips:
            bad ^= 1 << f
        fixed, nbits = correct(bad)
        assert fixed == word and nbits == len(flips)


def test_three_errors_are_not_passed_off_as_good():
    word = _bch(0x0ABCD)
    bad = word ^ (1 << 4) ^ (1 << 9) ^ (1 << 20)
    result = correct(bad)
    assert result is None or result[0] != word      # never silently "fixed" to a lie


def test_text_codings_round_trip():
    assert alpha_text(_alpha_chunks("TEST 123")) == "TEST 123"
    assert numeric_text(_numeric_chunks("0412 555-12")) == "0412 555-12"


# -- end to end, from IQ through the NBFM discriminator --------------------------------


@pytest.mark.parametrize("baud", [512, 1200, 2400])
def test_decodes_an_alpha_page_at_every_rate(baud):
    bits = encode(1234567, 3, _alpha_chunks("ROUND TRIP AT SPEED"))
    msgs = decode(discriminator(fsk_iq(bits, baud)))
    found = [m for m in msgs if m.baud == baud]
    assert len(found) == 1
    m = found[0]
    assert (m.address, m.function, m.kind, m.text) == (1234567, 3, "alpha",
                                                       "ROUND TRIP AT SPEED")


def test_numeric_and_tone_only_pages():
    bits = np.concatenate([encode(8, 0, _numeric_chunks("123")), encode(42, 1, [])])
    msgs = decode(discriminator(fsk_iq(bits, 1200)))
    assert [(m.address, m.kind, m.text) for m in msgs] == [(8, "numeric", "123"),
                                                           (42, "tone", "")]


def test_either_polarity_is_accepted():
    bits = encode(100, 3, _alpha_chunks("UPSIDE DOWN"))
    msgs = decode(discriminator(fsk_iq(bits, 1200, invert=True)))
    assert [m.text for m in msgs] == ["UPSIDE DOWN"]


def test_survives_noise():
    bits = encode(200, 3, _alpha_chunks("NOISY BUT READABLE"))
    msgs = decode(discriminator(fsk_iq(bits, 1200, noise=0.25)))
    assert [m.text for m in msgs] == ["NOISY BUT READABLE"]


def test_noise_alone_decodes_nothing():
    rng = np.random.default_rng(5)
    iq = (rng.standard_normal(48_000 * 3) + 1j * rng.standard_normal(48_000 * 3)) * 0.1
    assert decode(discriminator(iq.astype(np.complex64))) == []


def test_summary_hides_the_text_unless_asked():
    bits = encode(77, 3, _alpha_chunks("PRIVATE DETAILS"))
    (m,) = decode(discriminator(fsk_iq(bits, 1200)))
    assert "PRIVATE" not in m.summary() and "15 characters hidden" in m.summary()
    assert "PRIVATE DETAILS" in m.summary(show_text=True)
    assert "PRIVATE" not in repr(m)


# -- numeric or text: by content, not by function code ------------------------------


def test_text_sent_with_function_0_is_read_as_text():
    """A Melbourne network sends its text pages with function 0 (measured 2026-10-05);
    read as numeric they came out as digits strewn with U * ( ) -."""
    bits = encode(555, 0, _alpha_chunks("ALERT STRUCTURE FIRE"))
    (m,) = decode(discriminator(fsk_iq(bits, 512)))
    assert (m.function, m.kind, m.text) == (0, "alpha", "ALERT STRUCTURE FIRE")


def test_digits_sent_with_function_3_are_read_as_numeric():
    bits = encode(556, 3, _numeric_chunks("0412 555 123"))
    (m,) = decode(discriminator(fsk_iq(bits, 512)))
    assert (m.function, m.kind, m.text) == (3, "numeric", "0412 555 123")


def test_classification_over_many_random_messages():
    """Very short pages can read cleanly both ways, so this asks for 99 %, not 100 %."""
    rng = np.random.default_rng(42)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:/-"
    right_text = right_digits = 0
    trials = 2000
    for _ in range(trials):
        text = "".join(rng.choice(list(letters), rng.integers(3, 80)))
        right_text += classify(_alpha_chunks(text)) == ("alpha", text.rstrip())
        digits = "".join(rng.choice(list("0123456789 -"), rng.integers(3, 40))).strip() or "7"
        right_digits += classify(_numeric_chunks(digits)) == ("numeric", digits)
    assert right_text / trials >= 0.99 and right_digits / trials >= 0.99


def test_text_stops_at_the_end_of_text_code():
    chunks = _alpha_chunks("SHORT")             # EOT, then padding
    assert alpha_text(chunks) == "SHORT"
