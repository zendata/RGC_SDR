"""What each supported SDR is, before it is plugged in.

Probing (`SoapyIQSource._probe_caps`) is still the authority once a radio is open -- it
has already caught one driver claiming an AGC it ignores. These profiles cover what
probing cannot: naming the radio in a menu, saying how to install its driver, and
choosing sensible starting values the driver does not volunteer.

Two limits here come from this application rather than the hardware. `max_rate` is where
the NumPy DSP runs out of headroom: measured 2026-09-25, the audio chain takes 22% of a
core at 6 MS/s, 37% at 10 and 74% at 20, so rates above 10 MS/s are not offered even
where the radio supports them. And `dc_offset` marks radios with a spike at the centre
frequency, which the scanner must step around.

No Qt and no DSP imports (layering rule, PLANNING.md section 5).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .source import LO_OFFSET_HZ, DeviceCaps, FreqRange, GainElement, TxCaps

#: Above this the audio chain cannot keep up in pure NumPy (see module docstring).
APP_MAX_RATE = 10e6
#: The Pluto's USB link, measured: above this it silently drops samples.
PLUTO_MAX_RATE = 6e6


@dataclass(frozen=True)
class SdrProfile:
    key: str
    label: str
    driver: str
    #: Nominal coverage, shown before the radio is connected.
    freq_ranges: tuple[FreqRange, ...]
    #: Rates to offer. Used as-is when the driver only reports a continuous range.
    sample_rates: tuple[float, ...]
    default_rate: float
    max_rate: float
    #: Where to start when switching to this radio and the current frequency is outside
    #: its coverage -- somewhere with something to see, rather than a band edge.
    default_freq: float
    gain_elements: tuple[GainElement, ...]
    has_agc: bool
    bias_tee: bool
    #: A spike at the tuned centre the scanner must step around.
    dc_offset: bool
    #: Soapy module filename stem, for telling "not installed" from "not connected".
    module: str
    install: str
    notes: str = ""
    #: Gains to start from, where the driver's own defaults are measurably poor.
    default_gains: tuple[tuple[str, float], ...] = ()
    #: Rates the driver takes but does not list, offered as well: DAB's 2.048 MS/s, and
    #: 4.096 MS/s, which DAB needs on a radio whose own filter is too wide at 2.048.
    extra_rates: tuple[float, ...] = ()
    #: The transmitter, for radios that have one. Probing wins once the radio is open.
    tx: TxCaps | None = None
    #: "sdr" (IQ over SoapySDR) or "transceiver" (a radio controlled over its own
    #: protocol, such as the IC-705 over CI-V, which demodulates itself).
    kind: str = "sdr"
    #: How far to tune the hardware from the wanted frequency, shifting the stream back
    #: in software. None means LO_OFFSET_HZ for a radio with a DC spike, else none.
    lo_offset_hz: float | None = None
    #: A radio on a network radio server (device/remote.py): (host, port, its driver
    #: there). None for a radio attached to this machine.
    remote: tuple[str, int, str] | None = None

    @property
    def lo_offset(self) -> float:
        if self.lo_offset_hz is not None:
            return float(self.lo_offset_hz)
        return LO_OFFSET_HZ if self.dc_offset else 0.0

    def covers(self, hz: float) -> bool:
        return any(r.contains(hz) for r in self.freq_ranges)

    def describe_ranges(self) -> str:
        return ", ".join(
            f"{r.min_hz / 1e6:g}-{r.max_hz / 1e6:g} MHz" for r in self.freq_ranges
        )


PROFILES: tuple[SdrProfile, ...] = (
    SdrProfile(
        key="airspyhf",
        label="Airspy HF+",
        driver="airspyhf",
        freq_ranges=(FreqRange(9e3, 31e6), FreqRange(60e6, 260e6)),
        sample_rates=(912e3, 768e3, 650e3, 456e3, 384e3, 228e3, 192e3),
        default_rate=768e3,
        max_rate=912e3,
        default_freq=7.1e6,
        gain_elements=(),
        has_agc=False,           # claimed by the driver, ignored by it (section 3)
        bias_tee=False,
        dc_offset=False,         # measured: centre bin within 1 dB of the floor
        # No spike to dodge (re-measured 2026-10-05: centre 0-3.5 dB over the floor from
        # 3.7 to 230 MHz, i.e. noise), but tuned 100 kHz away all the same at VK3RQ's
        # request, as good practice, so nothing listened to sits on the LO.
        lo_offset_hz=100e3,
        module="airspyhfSupport",
        install="brew install airspyhf, plus the SoapyAirspyHF module",
        notes="HF and VHF. No gain controls: the driver exposes none.",
    ),
    SdrProfile(
        key="airspy",
        label="Airspy R2 / Mini",
        driver="airspy",
        freq_ranges=(FreqRange(24e6, 1800e6),),
        sample_rates=(10e6, 6e6, 3e6, 2.5e6),
        default_rate=2.5e6,
        max_rate=APP_MAX_RATE,
        default_freq=100.0e6,
        gain_elements=(
            GainElement("LNA", 0.0, 15.0, 1.0),
            GainElement("MIX", 0.0, 15.0, 1.0),
            GainElement("VGA", 0.0, 15.0, 1.0),
        ),
        has_agc=True,
        bias_tee=True,
        dc_offset=False,
        module="airspySupport",
        install="./tools/install_drivers.sh airspy (SoapyAirspy is not in Homebrew)",
        notes="The R2 offers 2.5 and 10 MS/s, the Mini 3 and 6.",
    ),
    SdrProfile(
        key="hackrf",
        label="HackRF One",
        driver="hackrf",
        freq_ranges=(FreqRange(1e6, 6000e6),),
        sample_rates=(10e6, 8e6, 6e6, 4e6, 2e6),
        extra_rates=(2.048e6, 4.096e6),
        default_rate=4e6,
        max_rate=APP_MAX_RATE,
        default_freq=100.0e6,
        gain_elements=(
            GainElement("AMP", 0.0, 14.0, 14.0),
            GainElement("LNA", 0.0, 40.0, 8.0),
            GainElement("VGA", 0.0, 62.0, 2.0),
        ),
        has_agc=False,
        bias_tee=True,
        dc_offset=True,
        module="HackRFSupport",
        install="brew install soapyhackrf",
        notes="Up to 20 MS/s in hardware; capped at 10 here for the NumPy DSP.",
        # Measured 2026-09-25 on Melbourne FM: the driver's LNA 16 / VGA 16 left
        # stations 10 dB over the floor, too weak for stereo. LNA 32 / VGA 30 gave
        # 25-27 dB, stereo and RDS on The Fox and SmoothFM, and IQ peaks near 0.16 --
        # plenty of headroom. AMP off: it added nothing but floor.
        default_gains=(("LNA", 32.0), ("VGA", 30.0)),
        # Probed 2026-09-25 (receive open, nothing keyed). Half duplex.
        tx=TxCaps(
            freq_ranges=(FreqRange(1e6, 6000e6),),
            gain_elements=(GainElement("VGA", 0.0, 47.0, 1.0),
                           GainElement("AMP", 0.0, 14.0, 14.0)),
            sample_rates=(2e6, 4e6, 8e6, 10e6),
            full_duplex=False,
        ),
    ),
    SdrProfile(
        key="rtlsdr",
        label="RTL-SDR",
        driver="rtlsdr",
        freq_ranges=(FreqRange(24e6, 1766e6),),
        sample_rates=(2.4e6, 2.048e6, 1.92e6, 1.8e6, 1.4e6, 1.024e6),
        default_rate=2.048e6,
        max_rate=2.4e6,          # the dongle drops samples above this
        default_freq=100.0e6,
        gain_elements=(GainElement("TUNER", 0.0, 49.6, 0.1),),
        has_agc=True,
        bias_tee=True,
        dc_offset=True,
        module="rtlsdrSupport",
        install="brew install soapyrtlsdr",
        notes="R820T coverage. HF needs direct sampling, not yet supported here.",
    ),
    SdrProfile(
        key="plutosdr",
        label="ADALM-Pluto",
        driver="plutosdr",
        freq_ranges=(FreqRange(325e6, 3800e6),),
        sample_rates=(6e6, 4e6, 2.5e6, 2e6, 1e6),
        extra_rates=(2.048e6, 4.096e6),
        default_rate=2e6,
        # Measured 2026-10-03 over USB: every sample arrives up to 6 MS/s, then the
        # link tops out near 7.7 MS/s (95% delivered at 8, 78% at 10) with no overflow
        # reported -- the gaps were only heard, as scratchy FM audio.
        max_rate=PLUTO_MAX_RATE,
        default_freq=433.92e6,
        gain_elements=(GainElement("PGA", 0.0, 73.0, 1.0),),
        has_agc=True,
        bias_tee=False,
        dc_offset=True,
        module="PlutoSDRSupport",
        install="./tools/install_drivers.sh pluto (builds libiio, libad9361, SoapyPlutoSDR)",
        notes="325-3800 MHz as shipped; 70-6000 MHz with the well-known firmware change.",
        # Measured 2026-09-25 (VK3RQ's Pluto, which reports 70-6000 MHz): the driver
        # starts at PGA 71 dB, where Melbourne FM overloads it -- 3 dB over a raised
        # floor, stereo 40% of the time, no radio text. At 50 dB: 47 dB over the floor,
        # stereo 99%, RDS 453 good / 0 bad, IQ peaks 0.32.
        default_gains=(("PGA", 50.0),),
        # Probed 2026-09-25, nothing keyed. Unlike the HackRF it is full duplex, so the
        # receiver keeps running while it transmits.
        tx=TxCaps(
            freq_ranges=(FreqRange(70e6, 6000e6),),
            gain_elements=(GainElement("PGA", 0.0, 89.0, 1.0),),
            sample_rates=(1e6, 2e6, 4e6, 6e6),
            full_duplex=True,
        ),
    ),
)

# Not an SDR, but chosen from the same menu: the IC-705, viewed and controlled over CI-V
# (device/icom.py). Its "rate" is the scope's span; it has no IQ.
PROFILES = PROFILES + (
    SdrProfile(
        key="icom705",
        label="Icom IC-705",
        driver="icom705",
        freq_ranges=(FreqRange(30e3, 199.999999e6), FreqRange(400e6, 470e6)),
        sample_rates=(20e3,),
        default_rate=20e3,
        max_rate=20e3,
        default_freq=145e6,
        gain_elements=(),
        has_agc=False,
        bias_tee=False,
        dc_offset=False,
        module="pyserial",
        install="pip install pyserial (./setup.sh does it), and a USB cable to the radio",
        notes="Transceiver over CI-V: the radio's own scope as the waterfall. No IQ.",
        kind="transceiver",
    ),
    # The same radio over WiFi (device/icom_net.py): Icom's network remote protocol.
    SdrProfile(
        key="icom705net",
        label="Icom IC-705 (WiFi)",
        driver="icom705net",
        freq_ranges=(FreqRange(30e3, 199.999999e6), FreqRange(400e6, 470e6)),
        sample_rates=(20e3,),
        default_rate=20e3,
        max_rate=20e3,
        default_freq=145e6,
        gain_elements=(),
        has_agc=False,
        bias_tee=False,
        dc_offset=False,
        module="network",
        install="the radio on WiFi with Network Control on (SET > WLAN Set > Remote Settings)",
        notes="The IC-705 over WiFi, as on its USB cable: scope, control, audio and TX.",
        kind="transceiver",
    ),
)

_BY_DRIVER = {p.driver: p for p in PROFILES}
_BY_KEY = {p.key: p for p in PROFILES}


#: Profiles of radios found on network radio servers, by key ("rtlsdr@radiopi"): made as
#: they are found, and kept, so a memory or setting naming one still resolves.
_REMOTE: dict[str, SdrProfile] = {}


def profile_for(driver_or_key: str) -> SdrProfile | None:
    return (_BY_DRIVER.get(driver_or_key) or _BY_KEY.get(driver_or_key)
            or _REMOTE.get(driver_or_key) or _remote_from_key(driver_or_key))


def register_remote(profile: SdrProfile) -> SdrProfile:
    _REMOTE[profile.key] = profile
    return profile


def _remote_from_key(key: str) -> SdrProfile | None:
    """A remote radio's profile from its key alone ("rtlsdr@radiopi" -> the RTL-SDR on the
    server named radiopi), for a key remembered from before its server was asked."""
    driver, at, host = key.partition("@")
    base = _BY_DRIVER.get(driver)
    if not at or not host or base is None or base.kind != "sdr":
        return None
    from .remote import ServerAddress, load_servers, remote_profile

    server = next((s for s in load_servers() if s.name == host), ServerAddress(host))
    return register_remote(remote_profile(base, server))


def installed_modules() -> set[str]:
    """Soapy module filename stems, e.g. {"HackRFSupport", "airspyhfSupport"}."""
    try:
        import SoapySDR  # type: ignore
    except ImportError:
        return set()
    stems = set()
    for path in SoapySDR.listModules():
        name = path.rsplit("/", 1)[-1]
        for suffix in (".so", ".dylib", ".dll"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if name.startswith("lib"):
            name = name[3:]
        stems.add(name)
    return stems


@dataclass(frozen=True)
class Availability:
    profile: SdrProfile
    installed: bool
    connected: bool
    serial: str = ""

    @property
    def status(self) -> str:
        if self.connected:
            return "connected"
        if self.installed:
            return "not connected"
        return "driver not installed"


def availability(
    devices: list[dict[str, str]] | None = None, modules: set[str] | None = None,
    servers: list | None = None,
) -> list[Availability]:
    """Every profile, with whether its driver is installed and a radio attached; then
    the radios on network radio servers (`servers`, by default those set up when the
    devices are probed for real)."""
    if modules is None:
        modules = installed_modules()
    if devices is None:
        from .source import enumerate_devices

        devices = enumerate_devices()
        if servers is None:
            from .remote import load_servers

            servers = load_servers()
    found = {}
    for device in devices:
        found.setdefault(device.get("driver", ""), device.get("serial", ""))
    # Transceivers are not SoapySDR devices: look for them on USB serial.
    from .icom import find_ic705_ports

    try:
        import serial  # noqa: F401
        modules = set(modules) | {"pyserial"}
    except ImportError:
        pass
    if find_ic705_ports():
        found.setdefault("icom705", "")
    # A network radio cannot be seen without logging in: it counts as there once its
    # address is set up, and the login says whether it really is.
    modules = set(modules) | {"network"}
    from .icom_net import load_login

    if load_login(keychain=False).complete:
        found.setdefault("icom705net", "")
    out = []
    for profile in PROFILES:
        # Exact stem match: "airspySupport" must not match "airspyhfSupport".
        installed = profile.module in modules
        connected = profile.driver in found
        out.append(Availability(profile, installed or connected, connected,
                                found.get(profile.driver, "")))
    return out + remote_availability(servers or [])


def remote_availability(servers) -> list[Availability]:
    """The SDRs attached to each server that answers, as remote profiles."""
    from .remote import list_radios, remote_profile

    out = []
    for server in servers:
        seen = set()
        for radio in list_radios(server):
            base = _BY_DRIVER.get(str(radio.get("driver")))
            if base is None or base.kind != "sdr" or base.driver in seen:
                continue
            seen.add(base.driver)
            profile = register_remote(remote_profile(base, server))
            out.append(Availability(profile, True, True, str(radio.get("serial", ""))))
    return out


def refine_caps(caps: DeviceCaps, profile: SdrProfile | None) -> DeviceCaps:
    """Fill gaps in probed capabilities and apply this application's rate ceiling.

    Probing wins wherever it has an answer. The profile only supplies rates when the
    driver reports a continuous range instead of a list (Pluto does), and drops rates the
    DSP cannot sustain.
    """
    if profile is None:
        return caps
    rates = tuple(sorted(set(caps.sample_rates or profile.sample_rates)
                         | set(profile.extra_rates), reverse=True))
    ceiling = min(profile.max_rate, APP_MAX_RATE)
    usable = tuple(r for r in rates if r <= ceiling) or tuple(sorted(rates)[:1])
    ranges = caps.freq_ranges or profile.freq_ranges
    return replace(caps, sample_rates=usable, freq_ranges=ranges)


def caps_from_profile(profile: SdrProfile, serial: str = "") -> DeviceCaps:
    """Capabilities as the profile describes them, for a radio not yet probed."""
    return DeviceCaps(
        driver=profile.driver,
        label=profile.label,
        serial=serial,
        sample_rates=tuple(sorted({r for r in profile.sample_rates + profile.extra_rates
                                   if r <= profile.max_rate}, reverse=True)),
        freq_ranges=profile.freq_ranges,
        gain_elements=profile.gain_elements,
        has_agc=profile.has_agc,
        formats=("CF32",),
        tx=profile.tx,
    )


def starting_frequency(profile: SdrProfile, current_hz: float) -> float:
    """Keep the current frequency if the new radio can tune it, else its default."""
    return float(current_hz) if profile.covers(current_hz) else float(profile.default_freq)
