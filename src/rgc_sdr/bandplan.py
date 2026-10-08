"""A simplified Australian band plan, for the spectrum's overlay (P11, PLANNING.md 7s).

For orientation, not authority: the ACMA's Australian Radiofrequency Spectrum Plan and
the amateur licence conditions are that, and the WIA band plan says how amateur bands
are used. Ranges here are the broad allocations a listener tunes by -- the amateur
bands as Australian licences allow (all licence grades together), the ITU shortwave
broadcast bands, aviation, marine, CB and the ISM bands.

No Qt; data and one lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Kinds, with the colour each is drawn in.
KIND_COLOURS = {
    "amateur": "#66bb6a",
    "broadcast": "#42a5f5",
    "aviation": "#ab47bc",
    "marine": "#26c6da",
    "cb": "#ffa726",
    "other": "#9e9e9e",
}


@dataclass(frozen=True)
class Band:
    low_hz: float
    high_hz: float
    name: str
    kind: str


def _b(low_mhz: float, high_mhz: float, name: str, kind: str) -> Band:
    return Band(low_mhz * 1e6, high_mhz * 1e6, name, kind)


BANDS: tuple[Band, ...] = (
    # Amateur (VK). 2200 m and 630 m are low-power LF/MF allocations.
    _b(0.1357, 0.1378, "2200 m amateur", "amateur"),
    _b(0.472, 0.479, "630 m amateur", "amateur"),
    _b(1.800, 1.875, "160 m amateur", "amateur"),
    _b(3.500, 3.700, "80 m amateur", "amateur"),
    _b(3.776, 3.800, "80 m amateur (DX)", "amateur"),
    _b(7.000, 7.300, "40 m amateur", "amateur"),
    _b(10.100, 10.150, "30 m amateur", "amateur"),
    _b(14.000, 14.350, "20 m amateur", "amateur"),
    _b(18.068, 18.168, "17 m amateur", "amateur"),
    _b(21.000, 21.450, "15 m amateur", "amateur"),
    _b(24.890, 24.990, "12 m amateur", "amateur"),
    _b(28.000, 29.700, "10 m amateur", "amateur"),
    _b(50.000, 54.000, "6 m amateur", "amateur"),
    _b(144.000, 148.000, "2 m amateur", "amateur"),
    _b(420.000, 450.000, "70 cm amateur", "amateur"),
    _b(1240.0, 1300.0, "23 cm amateur", "amateur"),
    _b(2400.0, 2450.0, "13 cm amateur", "amateur"),
    # Broadcast: MF (9 kHz channels), the ITU shortwave bands, FM, Band III (TV, DAB+).
    _b(0.5265, 1.6065, "MW broadcast", "broadcast"),
    _b(1.6065, 1.705, "MW narrowcast", "broadcast"),
    _b(2.300, 2.495, "120 m broadcast", "broadcast"),
    _b(3.200, 3.400, "90 m broadcast", "broadcast"),
    _b(3.900, 4.000, "75 m broadcast", "broadcast"),
    _b(4.750, 5.060, "60 m broadcast", "broadcast"),
    _b(5.900, 6.200, "49 m broadcast", "broadcast"),
    _b(7.300, 7.450, "41 m broadcast", "broadcast"),
    _b(9.400, 9.900, "31 m broadcast", "broadcast"),
    _b(11.600, 12.100, "25 m broadcast", "broadcast"),
    _b(13.570, 13.870, "22 m broadcast", "broadcast"),
    _b(15.100, 15.800, "19 m broadcast", "broadcast"),
    _b(17.480, 17.900, "16 m broadcast", "broadcast"),
    _b(18.900, 19.020, "15 m broadcast", "broadcast"),
    _b(21.450, 21.850, "13 m broadcast", "broadcast"),
    _b(25.670, 26.100, "11 m broadcast", "broadcast"),
    _b(87.500, 108.000, "FM broadcast", "broadcast"),
    _b(174.000, 230.000, "Band III: TV, DAB+", "broadcast"),
    # Aviation.
    _b(108.000, 117.975, "Air navigation (VOR, ILS)", "aviation"),
    _b(118.000, 137.000, "Airband (AM voice, ACARS)", "aviation"),
    _b(1089.0, 1091.0, "ADS-B 1090", "aviation"),
    # Marine.
    _b(156.000, 162.025, "Marine VHF", "marine"),
    _b(161.950, 162.050, "AIS", "marine"),
    # Citizens band.
    _b(26.965, 27.405, "27 MHz CB", "cb"),
    _b(476.425, 477.400, "UHF CB", "cb"),
    # Other.
    _b(137.000, 138.000, "Weather satellites", "other"),
    _b(433.050, 434.790, "433 MHz ISM / LIPD", "other"),
    _b(915.000, 928.000, "915-928 MHz LIPD", "other"),
    _b(1574.42, 1576.42, "GPS L1", "other"),
)


def bands_in(low_hz: float, high_hz: float) -> list[Band]:
    """Bands overlapping `low`..`high`, widest first (so narrow ones draw on top)."""
    found = [b for b in BANDS if b.high_hz > low_hz and b.low_hz < high_hz]
    return sorted(found, key=lambda b: b.low_hz - b.high_hz)
