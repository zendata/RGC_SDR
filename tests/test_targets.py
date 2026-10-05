"""Folding decoded messages into targets for the map."""

import pytest

from src.rgc_sdr.dsp.ais import AisMessage
from src.rgc_sdr.targets import EXPIRY_S, TargetStore


def ais(msg_type, mmsi, received=1000.0, name="", **fields):
    return AisMessage(msg_type, mmsi, fields, [], "A", name, received)


def test_positions_and_names_fold_into_one_target():
    store = TargetStore()
    store.update([ais(1, 503020660, position=(-37.81, 144.91), sog=4.2, cog=190.0, heading=188)])
    store.update([ais(5, 503020660, name="SVITZER OTWAY", callsign="VJN4647",
                      destination="to MELBOURNE")])
    store.update([ais(1, 503020660, received=1010.0, position=(-37.82, 144.91), sog=4.0,
                      cog=191.0, heading=None)])
    (t,) = store.targets.values()
    assert (t.kind, t.label, t.ident, t.messages) == ("ship", "SVITZER OTWAY", "503020660", 3)
    assert (t.lat, t.lon, t.speed_kn, t.course) == (-37.82, 144.91, 4.0, 191.0)
    assert t.heading is None and t.bearing == 191.0          # course when no heading
    assert list(t.trail) == [(-37.81, 144.91), (-37.82, 144.91)]
    assert t.details["callsign"] == "VJN4647"
    assert "SVITZER OTWAY" in t.describe()


def test_kinds():
    store = TargetStore()
    store.update([ais(21, 995036196, name="ECC-1", position=(-38.3, 144.63)),
                  ais(4, 5030280, position=(-37.87, 144.90)),
                  ais(18, 503084570, position=(-37.94, 144.99))])
    assert {t.ident: t.kind for t in store.targets.values()} == {
        "995036196": "aid", "005030280": "base", "503084570": "ship"}


def test_a_name_alone_is_known_but_not_placed():
    store = TargetStore()
    store.update([ais(24, 503181170, name="MONTY")])
    assert len(store.targets) == 1 and store.placed() == []


def test_old_targets_expire():
    store = TargetStore()
    store.update([ais(1, 1, received=0.0, position=(0.0, 0.0)),
                  ais(21, 2, received=0.0, position=(0.0, 0.0))])
    assert store.expire(now=EXPIRY_S["ship"] + 1) == 1          # the ship, not the mark
    assert [t.ident for t in store.targets.values()] == ["000000002"]


def test_version_changes_only_with_news():
    store = TargetStore()
    v = store.version
    store.update([])
    assert store.version == v
    store.update([ais(1, 7, position=(1.0, 2.0))])
    assert store.version == v + 1


def test_acars_places_aircraft_and_ground_stations():
    from src.rgc_sdr.dsp.acars import AcarsMessage

    store = TargetStore()
    store.update([AcarsMessage("2", "VH-ABC", "NAK", "H1", "1", "no position here")])
    assert store.targets == {}                            # nothing to place yet
    store.update([AcarsMessage("2", "VH-ABC", "NAK", "3L", "7", "S 37.894/E144.735",
                               "JQ0737", "M93A", position=(-37.894, 144.735)),
                  AcarsMessage("2", "", "NAK", "SQ", "", "02XSMELYMML03741S14451E",
                               position=(-37.68, 144.85))])
    plane, station = store.targets["acars:VH-ABC"], store.targets["acars:YMML"]
    assert (plane.kind, plane.name, plane.lat) == ("aircraft", "JQ0737", -37.894)
    assert station.kind == "base"
    store.update([AcarsMessage("2", "VH-ABC", "NAK", "H1", "2", "later, no position",
                               "JQ0737")])
    assert plane.messages == 2                            # known, so still counted


def test_adsb_aircraft_by_icao_address():
    from src.rgc_sdr.dsp.adsb import AdsbMessage

    store = TargetStore()
    store.update([AdsbMessage("7C6B2D", 17, 4, {"callsign": "QFA401"}, "")])
    assert store.placed() == [] and store.targets["adsb:7C6B2D"].name == "QFA401"
    store.update([AdsbMessage("7C6B2D", 17, 11, {"altitude_ft": 12000,
                                                 "position": (-37.7, 144.8)}, ""),
                  AdsbMessage("7C6B2D", 17, 19, {"speed_kn": 250.0, "track": 160.0,
                                                 "vertical_fpm": -1000}, "")])
    t = store.targets["adsb:7C6B2D"]
    assert (t.kind, t.altitude_ft, t.speed_kn, t.bearing) == ("aircraft", 12000, 250.0, 160.0)
    assert (t.lat, t.lon) == (-37.7, 144.8) and t.details["climb"] == "-1000 ft/min"
