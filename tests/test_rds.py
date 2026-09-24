"""RDS decoding. Codec tests use bit streams directly; RF tests are in test_fmstereo.py."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.rds import (
    GENERATOR,
    OFFSETS,
    PUBLISHED_SYNDROMES,
    GroupAssembler,
    StationInfo,
    checkword,
    encode_block,
    offset_of,
    syndrome,
)


def _reverse(value: int, bits: int) -> int:
    return int(f"{value:0{bits}b}"[::-1], 2)


def test_constants_reproduce_the_published_syndrome_table():
    """The independent check on the polynomial and offsets.

    IEC 62106 lists its syndromes least-significant-bit first. Converting ours to that
    convention must reproduce all five; a wrong polynomial or offset would not.
    """
    reversed_generator = _reverse(GENERATOR, 11)

    def mod(value, g):
        while value.bit_length() > 10:
            value ^= g << (value.bit_length() - 11)
        return value

    for name, offset in OFFSETS.items():
        published = _reverse(mod(_reverse(offset, 10) << 16, reversed_generator), 10)
        assert published == PUBLISHED_SYNDROMES[name], name


def ps_groups(pi, name, pty=10):
    """The four type-0A groups carrying an eight-character station name."""
    name = name.ljust(8)[:8]
    groups = []
    for segment in range(4):
        b = (0 << 12) | (0 << 11) | (pty << 5) | segment
        d = (ord(name[2 * segment]) << 8) | ord(name[2 * segment + 1])
        groups.append((pi, b, 0xE0CD, d))
    return groups


def rt_groups(pi, text, ab=0, pty=10):
    """Type-2A groups carrying radio text, four characters each."""
    text = (text + "\r").ljust(64)[:64]
    groups = []
    for segment in range(16):
        b = (2 << 12) | (pty << 5) | (ab << 4) | segment
        chunk = text[4 * segment: 4 * segment + 4]
        c = (ord(chunk[0]) << 8) | ord(chunk[1])
        d = (ord(chunk[2]) << 8) | ord(chunk[3])
        groups.append((pi, b, c, d))
        if "\r" in chunk:
            break
    return groups


def group_bits(groups):
    bits = []
    for a, b, c, d in groups:
        for info, offset in ((a, "A"), (b, "B"), (c, "C"), (d, "D")):
            block = encode_block(info, offset)
            bits.extend((block >> (25 - i)) & 1 for i in range(26))
    return np.array(bits, dtype=np.uint8)


def test_every_offset_is_recognised():
    for name in OFFSETS:
        block = encode_block(0x1234, name)
        assert offset_of(block) == name


def test_an_intact_block_has_its_offset_as_syndrome():
    assert syndrome(encode_block(0xBEEF, "B")) == OFFSETS["B"]


def test_a_single_bit_error_is_detected():
    block = encode_block(0xC202, "A")
    for bit in range(26):
        assert offset_of(block ^ (1 << bit)) is None, f"bit {bit} error went unnoticed"


def test_checkword_depends_on_the_offset():
    assert checkword(0x1234, "A") != checkword(0x1234, "B")


def test_station_name_is_decoded():
    asm = GroupAssembler()
    asm.push(group_bits(ps_groups(0xC202, "BBC R2") * 2))
    assert asm.synced
    assert asm.info.ps_name == "BBC R2"
    assert asm.info.pi == 0xC202
    assert asm.info.pty_name == "Pop Music"


def test_sync_is_found_from_an_arbitrary_starting_bit():
    """A receiver joins mid-stream; it must find the block boundaries itself."""
    bits = group_bits(ps_groups(0xC202, "RGC SDR") * 3)
    asm = GroupAssembler()
    asm.push(bits[37:])                   # start part-way through a block
    assert asm.info.ps_name == "RGC SDR"


def test_radio_text_is_decoded():
    asm = GroupAssembler()
    asm.push(group_bits(rt_groups(0xC202, "Now playing: Test Card") * 2))
    assert asm.info.radio_text == "Now playing: Test Card"


def test_new_radio_text_replaces_the_old_when_the_flag_flips():
    asm = GroupAssembler()
    asm.push(group_bits(rt_groups(0xC202, "First message here", ab=0)))
    asm.push(group_bits(rt_groups(0xC202, "Second", ab=1)))
    assert asm.info.radio_text == "Second"


def test_a_corrupt_block_is_dropped_not_decoded_as_nonsense():
    bits = group_bits(ps_groups(0xC202, "ABCDEFGH") * 3)
    # Corrupt the D block (characters) of the second segment in the first repeat.
    start = 26 * (4 * 1 + 3)
    bits[start + 5] ^= 1
    asm = GroupAssembler()
    asm.push(bits)
    assert asm.info.ps_name == "ABCDEFGH"
    assert asm.bad_blocks >= 1


def test_random_bits_never_produce_a_station_name():
    rng = np.random.default_rng(5)
    asm = GroupAssembler()
    asm.push(rng.integers(0, 2, 26 * 4 * 60, dtype=np.uint8))
    assert asm.info.ps_name == ""


def test_sync_is_lost_after_a_run_of_bad_blocks():
    asm = GroupAssembler(lose_after=5)
    asm.push(group_bits(ps_groups(0xC202, "BBC R2")))
    assert asm.synced
    asm.push(np.zeros(26 * 8, dtype=np.uint8) ^ 1)
    assert not asm.synced


def test_station_name_is_not_shown_until_complete():
    info = StationInfo()
    info.apply([0xC202, (0 << 12) | (10 << 5) | 0, 0, (ord("B") << 8) | ord("B")])
    assert info.ps_name == ""


def test_version_b_groups_are_handled():
    info = StationInfo()
    for segment, pair in enumerate(("Hi", "\r ")):
        b = (2 << 12) | (1 << 11) | (10 << 5) | segment
        info.apply([0xC202, b, 0xC202, (ord(pair[0]) << 8) | ord(pair[1])])
    assert info.radio_text == "Hi"


def test_partial_radio_text_is_not_shown():
    """Half a message on screen reads as garbage; show it only once complete."""
    groups = rt_groups(0xC202, "Now playing on Triple M Melbourne")
    info = StationInfo()
    for a, b, c, d in groups[:-1]:
        info.apply([a, b, c, d])
    assert info.radio_text == ""
    info.apply(list(groups[-1]))
    assert info.radio_text == "Now playing on Triple M Melbourne"


def test_the_previous_message_stays_up_while_the_next_arrives():
    info = StationInfo()
    for g in rt_groups(0xC202, "First message", ab=0):
        info.apply(list(g))
    for g in rt_groups(0xC202, "Second message here", ab=1)[:2]:
        info.apply(list(g))
    assert info.radio_text == "First message"


def test_a_missing_segment_keeps_the_message_back():
    info = StationInfo()
    groups = rt_groups(0xC202, "Complete sentence please")
    for i, g in enumerate(groups):
        if i != 2:
            info.apply(list(g))
    assert info.radio_text == ""
