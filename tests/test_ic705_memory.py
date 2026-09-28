"""IC-705 memory channels: decoded from bytes read off VK3RQ's radio (2026-09-28)."""

import pytest

from src.rgc_sdr.device.ic705_memory import MemoryChannel, MemorySide, address

# The 111 bytes after group and channel, exactly as the radio sent them.
CH_00_00 = bytes.fromhex(
    "00 00 10 62 00 00 02 01 00 00 00 00 08 85 00 08 85 00 00 23 00 00 50 00 43 51 43 51 "
    "43 51 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 00 10 62 00 00 02 02 00 00 "
    "00 00 08 85 00 08 85 00 00 23 00 00 50 00 43 51 43 51 43 51 20 20 20 20 20 20 20 20 20 "
    "20 20 20 20 20 20 20 20 20 52 4e 20 20 20 20 20 20 20 20 20 20 20 20 20 20")
CALL_144_C1 = bytes.fromhex(
    "00 00 00 44 46 01 05 01 00 04 00 00 08 85 00 08 85 00 01 31 00 00 60 00 43 51 43 51 "
    "43 51 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 00 00 52 46 01 05 01 00 00 "
    "00 00 08 85 00 08 85 00 00 23 00 00 60 00 43 51 43 51 43 51 20 20 20 20 20 20 20 20 20 "
    "20 20 20 20 20 20 20 20 20 50 72 69 76 61 74 65 20 32 6d 20 20 20 20 20 20")
RHF_OLINDA = bytes.fromhex(   # group 02, channel 00: a DUP- repeater with a tone
    "00 00 00 75 38 04 05 01 00 11 00 00 08 85 00 08 85 00 00 23 00 00 00 05 43 51 43 "
    "51 43 51 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 00 00 10 14 00 00 02 00 "
    "00 00 00 08 85 00 08 85 00 00 23 00 00 50 00 43 51 43 51 43 51 20 20 20 20 20 20 20 20 "
    "20 20 20 20 20 20 20 20 20 20 52 48 46 20 4f 6c 69 6e 64 61 20 20 20 20 20 20")


def test_a_broadcast_channel():
    m = MemoryChannel.decode(0, 0, CH_00_00)
    assert m.label == "00-00" and m.name == "RN"
    assert m.rx.freq_hz == 621_000 and m.rx.mode == "am" and m.rx.filter == 1
    assert not m.split and m.rx.tone_text == ""


def test_a_call_channel_with_dtcs():
    m = MemoryChannel.decode(100, 0, CALL_144_C1)
    assert m.label == "144 C1" and m.name == "Private 2m"
    assert m.rx.freq_hz == 146_440_000 and m.rx.mode == "fm"
    assert m.rx.tone_mode == 4 and m.rx.dtcs == "131" and m.rx.tone_text == "DTCS(T) 131"
    assert m.rx.offset_hz == 600_000


def test_a_repeater_channel():
    m = MemoryChannel.decode(2, 0, RHF_OLINDA)
    assert m.name == "RHF Olinda" and m.rx.freq_hz == 438_750_000
    assert m.rx.duplex == 1 and m.rx.tone_mode == 1 and m.rx.tone_text == "TONE 88.5"
    assert m.rx.offset_hz == 5_000_000                       # 70 cm: 5 MHz


@pytest.mark.parametrize("raw", [CH_00_00, CALL_144_C1, RHF_OLINDA])
def test_channels_round_trip_byte_for_byte(raw):
    assert MemoryChannel.decode(1, 2, raw).encode() == raw


def test_a_blank_answer_is_not_a_channel():
    assert MemoryChannel.decode(0, 50, b"\xff") is None


def test_new_channels_carry_the_same_data_on_both_sides():
    side = MemorySide(freq_hz=146_900_000, mode="fm", duplex=1, tone_mode=1, tone_hz=91.5,
                      offset_hz=600_000)
    m = MemoryChannel.simplex(5, 7, side, name="VK3RMM")
    raw = m.encode()
    back = MemoryChannel.decode(5, 7, raw)
    assert back.rx == back.tx == side and back.name == "VK3RMM"
    assert address(5, 7) == bytes.fromhex("00050007")
    assert address(100, 2) == bytes.fromhex("01000002")
