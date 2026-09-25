"""Repeater splits and CTCSS tones for NBFM transmit.

A repeater listens on one frequency and retransmits on another, so to work one you
receive on its output and transmit on its input: the listening frequency plus or minus
the band's standard offset. Australian band plan (WIA): 600 kHz on 2 m, 5 MHz on 70 cm.
Some older VK 70 cm repeaters use 7 MHz, so the offset is editable.

CTCSS is a sub-audible tone sent under the voice; many repeaters only open for the right
one. The standard EIA/TIA-603 tones are listed here, with 150.0 Hz.

Pure Python, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

SIMPLEX = "simplex"
PLUS = "plus"
MINUS = "minus"
SHIFTS = (SIMPLEX, PLUS, MINUS)


@dataclass(frozen=True)
class RepeaterBand:
    name: str
    low_hz: float
    high_hz: float
    #: The standard input/output split in Australia.
    offset_hz: float

    def contains(self, hz: float) -> bool:
        return self.low_hz <= hz <= self.high_hz


BANDS: tuple[RepeaterBand, ...] = (
    RepeaterBand("2 m", 144e6, 148e6, 600e3),
    RepeaterBand("70 cm", 420e6, 450e6, 5e6),
)


def band_for(hz: float) -> RepeaterBand | None:
    for band in BANDS:
        if band.contains(hz):
            return band
    return None


def tx_frequency(listen_hz: float, shift: str, offset_hz: float) -> float:
    """Where to transmit to be heard through a repeater whose output is `listen_hz`."""
    if shift == PLUS:
        return listen_hz + offset_hz
    if shift == MINUS:
        return listen_hz - offset_hz
    return listen_hz


#: Standard CTCSS tones, Hz.
CTCSS_TONES: tuple[float, ...] = (
    67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5,
    94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3,
    131.8, 136.5, 141.3, 146.2, 150.0, 151.4, 156.7, 159.8, 162.2, 165.5,
    167.9, 171.3, 173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8, 196.6,
    199.5, 203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3,
    254.1,
)
