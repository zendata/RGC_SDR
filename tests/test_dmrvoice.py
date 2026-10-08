"""DMR voice (dsp/dmrvoice.py): bursts found from the voice sync, AMBE+2 frames cut out
and deinterleaved for mbelib, slots kept apart, encrypted calls muted."""

import numpy as np
import pytest

from src.rgc_sdr.dsp import dmrvoice
from src.rgc_sdr.dsp.dmr import SYNCS, TACT_POSITIONS, bptc_encode, golay20_encode, tact_encode
from src.rgc_sdr.dsp.dmrvoice import (
    W, X, Y, Z, AmbeDecoder, DmrVoice, burst_frames, deinterleave,
)
from tests.test_dmr import BS_VOICE, DMR_UNIT, FILLER, PREAMBLE
from tests.test_fsk4 import RATE, fsk4_wave
from tests.test_p25 import bits_of


def _cach(slot):
    cach = np.zeros(24, dtype=int)
    cach[TACT_POSITIONS] = bits_of(tact_encode(0b1000 | (slot - 1) << 2), 7)
    return list(cach)


def voice_burst(slot, info216, first, privacy=0, rng=None):
    """A voice burst: A carries the voice sync, B-F the EMB (privacy bit 4) and a piece
    of embedded signalling."""
    if first:
        middle = bits_of(BS_VOICE, 48)
    else:
        middle = list((rng or np.random.default_rng(0)).integers(0, 2, 48))
        middle[4] = privacy
    bits = _cach(slot) + list(info216[:108]) + list(middle) + list(info216[108:])
    return [2 * bits[i] + bits[i + 1] for i in range(0, 288, 2)]


def data_burst(slot, colour=7, data_type=3):
    info = bptc_encode(PREAMBLE)
    slot_type = bits_of(golay20_encode(colour << 4 | data_type), 20)
    bits = (_cach(slot) + list(info[:98]) + slot_type[:10] + bits_of(SYNCS["bs"], 48)
            + slot_type[10:] + list(info[98:]))
    return [2 * bits[i] + bits[i + 1] for i in range(0, 288, 2)]


def superframe(slot, infos, privacy=0, other=None):
    """Six voice bursts on `slot`, the other slot's bursts (data) in between."""
    rng = np.random.default_rng(5)
    dibits = []
    for k, info in enumerate(infos):
        dibits += voice_burst(slot, info, k == 0, privacy, rng)
        dibits += other if other is not None else data_burst(3 - slot)
    return dibits


def _record(monkeypatch):
    frames = []

    def decode(self, frame):
        frames.append(np.array(frame))
        return np.full(160, 0.1, np.float32)

    monkeypatch.setattr(AmbeDecoder, "decode", decode)
    return frames


def _infos(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, 216) for _ in range(n)]


def test_the_interleave_fills_the_codec_frame_exactly_once():
    cells = list(zip(W, X)) + list(zip(Y, Z))
    assert len(set(cells)) == 72
    assert [sum(1 for r, _ in cells if r == k) for k in range(4)] == [24, 23, 11, 14]
    frame = np.arange(72) % 2
    grid = deinterleave(frame)
    assert grid.sum() == frame.sum()


def test_a_burst_holds_three_frames_the_second_across_the_middle():
    bits = np.arange(288)
    f1, f2, f3 = burst_frames(bits)
    assert list(f1) == list(range(24, 96))
    assert list(f2) == list(range(96, 132)) + list(range(180, 216))
    assert list(f3) == list(range(216, 288))


def test_a_superframe_gives_its_eighteen_frames_in_order(monkeypatch):
    frames = _record(monkeypatch)
    infos = _infos()
    voice = DmrVoice(RATE)
    audio = voice.process(fsk4_wave(superframe(1, infos) + FILLER, unit=DMR_UNIT, dc=-0.05))
    expected = [deinterleave(f) for info in infos
                for f in burst_frames(np.array(_cach(1) + list(info[:108]) + [0] * 48
                                               + list(info[108:])))]
    assert len(frames) == 18
    for got, want in zip(frames, expected):
        assert np.array_equal(got, want)
    assert audio.size == 18 * 160 and voice.encrypted[1] is False


def test_the_call_ending_stops_the_superframe(monkeypatch):
    frames = _record(monkeypatch)
    infos = _infos(2)
    dibits = (voice_burst(1, infos[0], True) + data_burst(2) + voice_burst(1, infos[1], False)
              + data_burst(2) + data_burst(1, data_type=2) + data_burst(2)   # terminator
              + data_burst(1) * 0 + FILLER * 4)
    DmrVoice(RATE).process(fsk4_wave(dibits, unit=DMR_UNIT, dc=-0.05))
    assert len(frames) == 6                               # bursts A and B only


def test_only_the_chosen_slot_is_heard(monkeypatch):
    frames = _record(monkeypatch)
    dibits = superframe(2, _infos(seed=2)) + FILLER
    assert DmrVoice(RATE, slot=1).process(fsk4_wave(dibits, unit=DMR_UNIT)).size == 0
    assert frames == []
    assert DmrVoice(RATE, slot=2).process(fsk4_wave(dibits, unit=DMR_UNIT)).size == 18 * 160


def test_an_encrypted_call_is_muted(monkeypatch):
    _record(monkeypatch)
    voice = DmrVoice(RATE)
    audio = voice.process(fsk4_wave(superframe(1, _infos(), privacy=1) + FILLER,
                                    unit=DMR_UNIT))
    assert voice.encrypted[1] is True and not np.any(audio)
    assert voice.muted_frames == 18


@pytest.mark.skipif(not dmrvoice._load_mbelib(), reason="mbelib not installed")
def test_mbelib_decodes_an_ambe_frame():
    codec = AmbeDecoder()
    speech = codec.decode(deinterleave(np.random.default_rng(3).integers(0, 2, 72)))
    assert speech.shape == (160,) and speech.dtype == np.float32
    assert np.all(np.abs(speech) <= 1.5)


def test_dmr_mode_plays_through_the_chain(monkeypatch):
    from src.rgc_sdr.dsp.demod import DemodChain, ReceiverOptions

    _record(monkeypatch)
    chain = DemodChain(48_000.0, "dmr", volume=1.0, options=ReceiverOptions(dmr_slot=1))
    assert chain.audio_rate == 48_000.0 and chain.dmr_voice.slot == 1
    chain.set_options(ReceiverOptions(dmr_slot=2))
    assert chain.dmr_voice.slot == 2
