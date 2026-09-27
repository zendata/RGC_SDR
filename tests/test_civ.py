"""CI-V protocol, checked against bytes captured from VK3RQ's IC-705 (2026-09-27)."""

import pathlib

import numpy as np
import pytest

from src.rgc_sdr.device import civ

CAPTURE = (pathlib.Path(__file__).parent / "data" / "ic705_scope_2s.bin").read_bytes()


def test_read_frequency_request_matches_what_the_radio_answered():
    assert civ.encode(civ.CMD_READ_FREQ) == bytes.fromhex("fefea4e003fd")


def test_the_radios_frequency_reply_decodes():
    """The 705 answered FE FE E0 A4 03 00 00 65 45 01 FD when on 145.65 MHz."""
    [frame] = civ.FrameParser().feed(bytes.fromhex("fefee0a4030000654501fd"))
    assert (frame.to, frame.frm, frame.cmd) == (civ.CONTROLLER_ADDRESS, civ.IC705_ADDRESS, 0x03)
    assert civ.decode_freq(frame.payload) == 145_650_000


@pytest.mark.parametrize("hz", [0, 7_100_000, 145_650_000, 433_920_000, 1_234_567_890])
def test_frequency_bcd_round_trips(hz):
    assert civ.decode_freq(civ.encode_freq(hz)) == hz


def test_frequency_bcd_is_least_significant_byte_first():
    assert civ.encode_freq(145_650_000) == bytes.fromhex("0000654501")


def test_levels_are_two_byte_bcd_most_significant_first():
    assert civ.encode_level(128) == bytes.fromhex("0128")
    assert civ.decode_level(bytes.fromhex("0255")) == 255
    with pytest.raises(ValueError):
        civ.encode_level(256)


def test_mode_reply_from_the_radio():
    """04 -> 05 01: FM, filter 1."""
    [frame] = civ.FrameParser().feed(bytes.fromhex("fefee0a4040501fd"))
    assert civ.MODES[frame.payload[0]] == "fm" and frame.payload[1] == 1


def test_parser_reassembles_frames_split_across_reads():
    parser = civ.FrameParser()
    got = []
    for i in range(0, len(CAPTURE), 7):                 # 7-byte dribbles
        got += parser.feed(CAPTURE[i:i + 7])
    assert len(got) == 90 and parser.discarded == 0


def test_parser_skips_noise_and_collisions():
    parser = civ.FrameParser()
    noise = b"\x00\x13\x37"
    jammed = bytes.fromhex("fefee0a4fcfcfd")
    good = bytes.fromhex("fefee0a4fbfd")
    frames = parser.feed(noise + jammed + good)
    assert len(frames) == 1 and frames[0].is_ok


def test_parser_holds_a_half_preamble_for_the_next_read():
    parser = civ.FrameParser()
    assert parser.feed(bytes.fromhex("fe")) == []
    assert len(parser.feed(bytes.fromhex("fee0a4fbfd"))) == 1


def test_data_cannot_contain_the_end_marker():
    with pytest.raises(ValueError):
        civ.encode(0x14, b"\x01\xfd")


def _lines(data):
    parser, assembler = civ.FrameParser(), civ.ScopeAssembler()
    lines = []
    for frame in parser.feed(data):
        line = assembler.feed(frame)
        if line is not None:
            lines.append(line)
    return lines, assembler


def test_real_scope_capture_gives_whole_475_point_lines():
    lines, assembler = _lines(CAPTURE)
    assert len(lines) >= 8
    for line in lines:
        assert line.amplitudes.size == 475
        assert line.centre_mode
        assert line.centre_hz == pytest.approx(145.65e6)
        assert line.span_hz == pytest.approx(20e3)          # +/-10 kHz
        assert line.amplitudes.max() <= 160


def test_a_line_missing_a_division_is_dropped_not_torn():
    scope = [f for f in civ.FrameParser().feed(CAPTURE) if f.cmd == civ.CMD_SCOPE]
    first_line = next(i for i, f in enumerate(scope) if f.payload[2] == 0x01)
    without_div5 = scope[:first_line + 4] + scope[first_line + 5:]   # lose division 5 once
    assembler = civ.ScopeAssembler()
    lines = [line for f in without_div5 if (line := assembler.feed(f)) is not None]
    assert assembler.dropped >= 1
    assert all(line.amplitudes.size == 475 for line in lines)


def test_fixed_mode_header_gives_the_edges():
    # 27 00 | main, seq 1 of 1 | fixed | lower edge | upper edge | in range | amplitudes
    frame = civ.Frame(civ.CONTROLLER_ADDRESS, civ.IC705_ADDRESS, 0x27,
                      bytes([0x00, 0x00, 0x01, 0x01, 0x01]) + civ.encode_freq(144e6)
                      + civ.encode_freq(146e6) + b"\x00" + bytes(range(10)))
    line = civ.ScopeAssembler().feed(frame)
    assert not line.centre_mode
    assert (line.low_hz, line.high_hz) == (144e6, 146e6)
    assert line.amplitudes.tolist() == list(range(10))
