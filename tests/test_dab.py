"""DAB mode I: OFDM synchronisation and the FIC, against an encoder written here.

The coding chain's parts are checked against what the standard fixes -- PI_16 and PI_1
as printed, the FIB CRC -- and the whole receiver on a synthetic ensemble with noise and
a frequency offset. On-air confirmation is in PLANNING.md.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp import dab as D


def fig1(ext: int, ident: int, label: str) -> bytes:
    text = label.encode("latin-1").ljust(16)
    body = bytes([ext]) + ident.to_bytes(2, "big") + text + b"\xff\xff"
    return bytes([1 << 5 | len(body)]) + body


def fig0_2(sid: int, sub: int, dab_plus: bool) -> bytes:
    body = bytes([2]) + sid.to_bytes(2, "big") + bytes([1]) + \
        bytes([(63 if dab_plus else 0) & 63, sub << 2 | 0b10])
    return bytes([0 << 5 | len(body)]) + body


def fig0_1(sub: int, start: int, size: int, level: int = 3) -> bytes:
    body = bytes([1, sub << 2 | start >> 8, start & 255,
                  0x80 | (level - 1) << 2 | size >> 8, size & 255])
    return bytes([0 << 5 | len(body)]) + body


def fib(figs: bytes) -> np.ndarray:
    data = (figs + b"\xff" * 30)[:30]
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    crc = D.crc16(bits)
    return np.concatenate([bits, [(crc >> (15 - i)) & 1 for i in range(16)]]).astype(np.uint8)


FIGS = [fig1(0, 0x1001, "TEST ENSEMBLE"), fig1(1, 0x1A01, "RADIO ONE"),
        fig1(1, 0x1A02, "RADIO TWO"), fig0_2(0x1A01, 1, True), fig0_2(0x1A02, 2, False),
        fig0_1(1, 0, 72), fig0_1(2, 72, 84)]


def frame_signal(rng) -> np.ndarray:
    fibs = [fib(FIGS[(3 * b + f) % len(FIGS)]) for b in range(4) for f in range(3)]
    blocks = [np.concatenate(fibs[3 * b:3 * b + 3]) ^ D._PRBS_FIC for b in range(4)]
    coded = np.concatenate([D.convolve(b)[D.FIC_PATTERN] for b in blocks])     # 9216
    z = np.zeros((D.SYMBOLS - 1, D.K), dtype=complex)
    for l in range(D.SYMBOLS - 1):
        if l < D.FIC_SYMBOLS:
            bits = coded[l * 2 * D.K:(l + 1) * 2 * D.K]
        else:
            bits = rng.integers(0, 2, 2 * D.K)
        z[l] = ((1 - 2.0 * bits[:D.K]) + 1j * (1 - 2.0 * bits[D.K:])) / np.sqrt(2)
    cells = np.exp(1j * np.pi / 2 * rng.integers(0, 4, D.K))          # the reference
    symbols = [cells]
    for l in range(D.SYMBOLS - 1):
        cells = cells * z[l]
        symbols.append(cells)
    out = [np.zeros(D.NULL, dtype=complex)]
    for cells in symbols:
        spectrum = np.zeros(D.TU, dtype=complex)
        spectrum[D._BINS] = cells
        t = np.fft.ifft(spectrum) * np.sqrt(D.TU)
        out.append(np.concatenate([t[-D.GUARD:], t]))
    return np.concatenate(out)


def ensemble_iq(frames=4, offset_carriers=2.3, noise=0.15, seed=4):
    rng = np.random.default_rng(seed)
    x = np.concatenate([frame_signal(rng) for _ in range(frames)])
    x = np.concatenate([x[5000:], x[:5000]])          # starting mid-frame, as a radio does
    n = np.arange(x.size)
    x = x * np.exp(2j * np.pi * offset_carriers * n / D.TU)
    x = x + noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x.astype(np.complex64)


def test_puncturing_vectors_are_the_standards():
    assert "".join(map(str, D.PI[16].astype(int))) == "1110" * 8
    assert "".join(map(str, D.PI[1].astype(int))) == "1100" + "1000" * 7
    assert D.PI[24].all() and D.FIC_PATTERN.sum() == 2304
    # Not an even spread of ones: what an evenly spread rule got wrong.
    assert "".join(map(str, D.PI[2].astype(int))) == "1100100010001000" "1100100010001000"
    assert "".join(map(str, D.PI[9].astype(int))) == "1110110011001100" "1100110011001100"


def test_carrier_order_is_a_permutation_of_the_1536_carriers():
    assert sorted(D.CARRIERS) == [k for k in range(-768, 769) if k != 0]


def test_viterbi_corrects_a_noisy_codeword():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 768).astype(np.uint8)
    soft = 1 - 2.0 * D.convolve(bits)[D.FIC_PATTERN] + 0.7 * rng.standard_normal(2304)
    assert (D.viterbi(D.depuncture(soft, D.FIC_PATTERN))[0] == bits).all()


def test_the_receiver_reads_the_ensemble_and_its_services():
    rx = D.DabReceiver(D.RATE)
    found = []
    x = ensemble_iq()
    for i in range(0, x.size, 50000):
        found += rx.process(x[i:i + 50000])
    texts = [m.text for m in found]
    assert "ensemble TEST ENSEMBLE (1001)" in texts
    assert any(t.startswith("RADIO ONE  (DAB+, service 1A01, subchannel 1, 72 CU, EEP 3-A")
               for t in texts)
    assert any(t.startswith("RADIO TWO  (DAB, service 1A02, subchannel 2") for t in texts)
    assert rx.bad_fibs == 0 and rx.good_fibs >= 24
    assert rx.offset_hz == pytest.approx(2.3 * 1000, abs=30)


def test_needs_2048_ks():
    with pytest.raises(ValueError):
        D.DabReceiver(2e6)


def test_noise_is_not_an_ensemble():
    rx = D.DabReceiver(D.RATE)
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(600000) + 1j * rng.standard_normal(600000)).astype(np.complex64)
    assert rx.process(x) == [] and rx.good_fibs == 0


def test_the_decode_worker_reads_an_ensemble_at_the_listening_offset():
    from src.rgc_sdr.decoding import DecodeWorker
    from tests.test_decoding import RingSource

    x = ensemble_iq(offset_carriers=0.0)
    n = np.arange(x.size)
    shifted = (x * np.exp(2j * np.pi * 300e3 * n / D.RATE)).astype(np.complex64)
    worker = DecodeWorker(RingSource(D.RATE), "dab", offset_hz=300e3)
    assert worker.problem == ""
    found = []
    for i in range(0, shifted.size, 100000):
        found += worker.process(shifted[i:i + 100000])
    assert any(m.text == "ensemble TEST ENSEMBLE (1001)" for m in found)


def test_the_worker_explains_the_rate_dab_needs():
    from src.rgc_sdr.decoding import DecodeWorker
    from tests.test_decoding import RingSource

    assert "2.048 MS/s" in DecodeWorker(RingSource(2e6), "dab").problem
