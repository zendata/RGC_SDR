# 5. Features to add

What the established SDR applications (SDR Console, SDRuno, SDR++, SDR#, SDRangel, HDSDR,
GQRX) offer that RGC_SDR does not yet, and a plan for adding them. These are proposals
for discussion, not decisions: PLANNING.md section 6 holds the roadmap, and each phase is
agreed, built and verified on air before the next one starts.

## Where RGC_SDR stands

It already has some things few of those applications combine:

- DAB+ audio, and P25 voice with who is talking
- P25 and DMR trunking metadata and packet data, with positions on a map
- AIS, ADS-B, ACARS, APRS and POCSAG on one map and panel
- a protocol classifier that labels the whole screen
- per-radio frequency calibration
- the IC-705 controlled over USB and WiFi
- radios on a Pi used over Tailscale at their full span

The gaps are mostly in three areas: **receiver refinements** (noise blanking, noise
reduction, notches, adjustable passbands, synchronous AM), **more than one receiver in
the span at once**, and **integration** with other software (CAT control, audio routing,
time-shift).

## The gaps

Effort: **S** is days, **M** a week or two, **L** several weeks. "Value" is judged for
this station: an Australian amateur listening to HF, VHF airband, P25 and DMR, broadcast,
and data.

### Receiving better

| Feature | What it is | Effort | Value |
|---|---|---|---|
| Noise blanker | Blank impulse noise (ignition, power lines) in the IQ before the channel filter | S | High on HF |
| Noise reduction | Spectral subtraction or Wiener filtering of the audio | M | High on SSB and AM |
| Notch filters | Automatic notch of carriers (heterodynes) in the passband; manual notches set by clicking | M | High on HF |
| Adjustable passband | Drag the passband edges on the spectrum; IF shift; separate low and high cut for SSB | S | High |
| Synchronous AM | Lock to the carrier and demodulate coherently, with selectable sideband, to beat selective fading and adjacent-channel splatter | M | High on shortwave and MW |
| AGC modes | Fast, medium, slow and off, with a threshold; manual IF gain | S | Medium |
| NBFM noise squelch | Squelch on the noise above the voice band rather than on level, as FM radios do | S | Medium |
| Audio equaliser and output choice | Bass and treble, choice of output device | S | Medium |
| Calibrated S-meter | dBm and S-units from a measured gain table per radio and gain setting | M | Medium; needs a signal generator |

### Seeing more

| Feature | What it is | Effort | Value |
|---|---|---|---|
| Markers | Click-placed markers with frequency and level, delta markers, peak search, channel power, occupied bandwidth | S | High as a learning tool |
| Band plan overlay | ACMA allocations and the WIA amateur band plan drawn under the spectrum | S | High |
| Memories on the spectrum | Memory names shown at their frequencies | S | Medium |
| Shortwave schedules | EiBi or HFCC broadcast schedules labelling HF signals, by time of day | M | Medium |
| Persistence display | A density (heat-map) spectrum that shows how often each level occurs, revealing bursty and hidden signals | M | Medium |
| Display controls | Waterfall speed, averaging, window function, timestamps on the waterfall | S | Medium |
| Wideband panorama | A sweep stitched from many tunings, hundreds of MHz on one screen, from the scanner's windowing | M | High for surveying |
| Detachable panels | Panels in their own windows, for a second screen | M | Low |

### More than one receiver at once

| Feature | What it is | Effort | Value |
|---|---|---|---|
| Multiple VFOs | Several independent receivers in one span, each with its own mode, passband, squelch, volume, recorder and decoder, mixed to the speaker | L | Very high: a hallmark of high-end SDR |
| P25 trunk following | Follow the control channel's voice grants to the voice channel automatically, with talkgroup names, priorities and lockouts | M, after VFOs | Very high for P25 |
| DMR trunk following | The same for DMR Tier III or Capacity Plus, where present | M | Medium |

### More protocols

| Feature | What it is | Effort | Value |
|---|---|---|---|
| DMR voice | AMBE+2, through mbelib's AMBE decoder as P25 uses its IMBE one. The codec's patent status needs checking first. | M | High |
| D-STAR and System Fusion | Amateur digital voice common in VK: GMSK 4800 or C4FM, with AMBE / AMBE+2 | M each | Medium |
| P25 Phase 2 | TDMA, H-DQPSK, AMBE+2 half rate. PLANNING.md notes the local network carries its voice on Phase 2. | L | High if that network matters |
| NXDN, dPMR | Other narrowband digital voice | M each | Low unless heard locally |
| HF weather fax | The Bureau of Meteorology's HF radio-fax broadcasts (check they are still on air) | M | Medium, a classic learning project |
| NAVTEX, SITOR-B | Marine safety text on 518 and 490 kHz | M | Low here |
| RTTY, PSK31 | Classic amateur text modes | M | Medium |
| FT8, FT4, WSPR | The busiest amateur digital modes. Hand audio and CAT to WSJT-X (see integration) rather than reimplementing them. | S via WSJT-X | High |
| HFDL, VDL Mode 2 | Aircraft data on HF and on 136.975 MHz (check VDL2 is active in VK) | L, M | Medium |
| Weather satellites | Meteor-M LRPT on 137 MHz, with Doppler correction from orbital elements. The NOAA APT satellites have been retired. | L | Medium; the memories already have a Satellites group |
| Satellite Doppler | Tune automatically for Doppler from TLEs (SGP4), for the ISS and amateur satellites | M | Medium |
| DRM | Digital shortwave; already in PLANNING.md, waiting for a signal heard here and the HF+ | L | Low until a signal is found |
| DAB (MP2) | Original DAB audio, and UEP sub-channels. Melbourne's ensembles are DAB+, so check whether any MP2 service exists here first. | M | Low |
| ISM sensors | 433 MHz weather stations, tyre pressure and the like (as rtl_433). Some carry identifiers, so keep the privacy rule. | M | Low |
| ADS-B extras | An aircraft database (type, operator), altitude colouring on the map, Mode A/C | S–M | Medium |

### Recording and playback

| Feature | What it is | Effort | Value |
|---|---|---|---|
| Time shift | A rolling buffer of the last few minutes of IQ, so you can go back and hear what was missed | M | Very high |
| SigMF recordings | Read and write the SigMF standard (data file plus JSON metadata), so recordings move between RGC_SDR, SDR++, GNU Radio and inspectrum | S | High |
| Squelch-triggered recording | Record audio only while a channel is open, one file per transmission, per VFO | S | High for monitoring |
| Scheduled recording | Start and stop a recording at set times on a set memory | S | Medium |
| Playback controls | Seek, speed, loop, and a whole-file waterfall overview to jump to signals | M | High |

### Working with other software

| Feature | What it is | Effort | Value |
|---|---|---|---|
| CAT server | Serve the hamlib `rigctld` protocol (and later TCI), so WSJT-X, fldigi and loggers can read and set frequency and mode | S | Very high |
| Audio routing | Send a VFO's audio to a virtual device (BlackHole), for digital-mode software | S | Very high with CAT |
| Memory import and export | CSV, CHIRP and SDR# frequency lists in and out | S | Medium |
| DX cluster | Spots from a DX cluster drawn on the spectrum | S | Medium on HF |
| Logging | Log what was heard (ADIF for contacts) | S | Low to medium |
| Knobs and controllers | A USB tuning knob or MIDI controller, and keyboard shortcuts | S | Medium |

### Network radio server

| Feature | What it is | Effort | Value |
|---|---|---|---|
| Audio-only mode | Demodulate on the Pi and send compressed audio (Opus), for a phone connection | M | High away from home |
| Several listeners | More than one client sharing a radio, each with their own IQ window | M | Low |
| Transmit over the network | The HackRF on the Pi keyed from the Mac. No interlocks are to be added, at the operator's choice, and the first key-up is the operator's. | M | Medium |
| Classify on the Pi | Run the whole-screen sweep on the Pi, so it covers the full span and not just the IQ window | S | Medium |
| A web client | Listen from a browser | L | Low |

### Hardware and speed

| Feature | What it is | Effort | Value |
|---|---|---|---|
| More radios | SDRplay (SoapySDRPlay3), LimeSDR, bladeRF and USRP are already SoapySDR radios: a profile and a test each | S each | Depends on the radio |
| KiwiSDR client | Use public KiwiSDR receivers worldwide as sources | M | Medium |
| Faster DSP | Numba or Accelerate for the inner loops that NumPy cannot vectorise (phase-locked loops, LMS filters), and above 10 MS/s | M | Enables SAM, LMS notch, HackRF at 20 MS/s |

Not planned: a CW keyer or CW transmit, which the operator has declined; and band, mode
or power interlocks on transmit, likewise.

## The plan

Ordered by value for effort, and so each phase builds on the one before. Each ends
runnable, tested, and verified on air.

### P10: Receiver refinements (M) ✅ done 2026-10-08

The biggest difference on HF, and good DSP to learn from.

1. **Adjustable passband and IF shift.** Drag the passband's edges on the spectrum. Store
   the low and high cut per mode in the snapshot, so memories keep them. The channel
   filter already takes an arbitrary band-pass (`bandpass_taps`); the work is in
   `ui/spectrum_view.py` and `DemodChain.set_passband(low, high)`.
2. **Noise blanker.** In the IQ, before decimation: blank samples whose magnitude exceeds
   *k* × a running mean, with a short window either side. Vectorised: a moving average,
   a mask, and dilating the mask by convolution. Test with synthetic impulses on a tone.
   On air: 40 m at night, or a known noisy appliance.
3. **Noise reduction.** On the audio: short-time FFT with overlap-add, a noise spectrum
   learned during the quietest frames, then spectral subtraction with a floor to avoid
   "musical noise". It works a block at a time, so it fits the chain. A strength
   control, 0–10.
4. **Notches.** Automatic: find steady tones in the audio spectrum (a carrier is a peak
   persisting across frames) and notch them with narrow FIR band-stops. Manual: click a
   carrier inside the passband. An adaptive LMS filter needs a per-sample loop, so the
   FFT approach comes first.
5. **Synchronous AM.** Estimate the carrier's phase block by block, as the RDS decoder
   already tracks its subcarrier in 64-sample chunks with a small loop, then demodulate
   coherently. Offer LSB, USB or both sidebands to dodge interference on one side. Test
   with a synthetic carrier plus modulation plus a frequency offset; verify on 774 kHz
   (ABC Melbourne) and shortwave at night.
6. **AGC modes** (fast, medium, slow, off) and an NBFM noise squelch.

### P11: Markers and band plan (S) ✅ done 2026-10-08

7. Markers, delta markers, peak search, channel power and occupied bandwidth, on the
   spectrum. All are computed from the Welch spectrum the frame already has.
8. A band-plan overlay from a small data file of ACMA and WIA allocations, drawn as
   coloured bands under the spectrum, plus memory names at their frequencies.

### P12: Time shift and SigMF (M)

9. **SigMF.** Write recordings as `.sigmf-data` plus `.sigmf-meta`, and read SigMF and
   the existing complex64 + JSON format. It is a small change to `recorder.py` and
   `device/playback.py`, and makes every recording usable in other tools.
10. **Time shift.** A second, longer ring (minutes, sized from the rate; spilled to disk
    above a memory limit), written by the reader thread alongside the live ring.
    "Rewind" makes playback read from it. Since playback is already just another source,
    the display, audio and decoders need no changes.
11. **Playback controls**: seek, speed, loop, and an overview waterfall of the whole
    file.

### P13: Working with other software (S)

12. **A `rigctld` server.** Hamlib's TCP protocol (`f`, `F`, `m`, `M`, `t`, `T` and so
    on) on localhost: WSJT-X, fldigi, JTDX and most loggers speak it. It maps to
    `_retune` and the mode combo, and works for the SDRs and the IC-705 alike.
13. **Audio to a virtual device.** A second `AudioSink` on a chosen output, such as
    BlackHole. With item 12, WSJT-X decodes FT8 and WSPR from the HF+ with no new
    decoder.
14. Memory import and export (CSV, CHIRP).

### P14: Multiple VFOs (L)

The largest change, and the one that most separates high-end software.

15. **Model.** A `Vfo` holds a frequency, mode, passband, squelch, volume, recorder and
    decoder. The window has a list of them, one "active" for the controls.
16. **DSP.** One `DemodChain` per VFO, each with its own sequential reader on the shared
    ring. The chain is already self-contained, and `decoding.py` already runs several
    lanes on one reader, so this reuses existing pieces. CPU is the limit: share the
    first decimation stages between VFOs where the rates allow.
17. **Audio.** A mixer that sums the VFOs' audio (each with volume and left/right pan)
    into the one output, or routes a VFO to its own device (item 13).
18. **Interface.** Each VFO's passband in its own colour on the spectrum, a strip of VFO
    tabs, and a click to make one active.
19. **Network radios.** The server sends one IQ window per VFO.

### P15: P25 trunk following (M, after P14)

20. The control channel already gives voice grants (talkgroup, radio, channel) and the
    channel plan. A trunking VFO follows each grant to its voice channel, which is
    usually inside the span, with talkgroup names, priorities and lockouts from a small
    table. Several calls at once, one VFO each. Verify on the local P25 network,
    remembering its voice may be Phase 2, which needs P20.

### P16: Digital voice (M each)

21. **DMR voice**, through mbelib's AMBE decoder, once the codec's patent position is
    checked. The DMR decoder already finds voice bursts by their sync.
22. **D-STAR and System Fusion**, for the VK amateur repeaters that use them.

### P17: Network server, phase 3 (M)

23. **Audio-only mode with Opus** for phone connections: the Pi demodulates (the same
    `DemodChain`) and sends compressed audio plus spectrum lines.
24. **Classify on the Pi**, so the sweep covers the whole span.
25. **Transmit over the network**, keyed by the operator as every new transmit path is.

### Later, as wanted

- Weather fax and RTTY or PSK31 (P18)
- Satellites: TLE Doppler tracking, then Meteor LRPT (P19)
- P25 Phase 2 (P20)
- A persistence display, the wideband panorama, and shortwave schedules (P21)
- DRM, once a signal is heard here
- Faster DSP, if a feature above runs short of CPU

## How each phase should be done

The same way as every phase so far:

- **Plan first.** A section in PLANNING.md before the code, including what will be
  measured.
- **Layering.** DSP in `dsp/` with no Qt and no device access, tested headless with
  synthetic signals. A test double may feed a pure DSP function; the app never gets a
  fake radio.
- **Vectorised.** NumPy with no per-sample Python loops. Where a loop is unavoidable
  (a PLL, an LMS filter), run it on blocks or chunks, or move it to compiled code.
- **Verified on air**, with the measurement recorded in PLANNING.md section 3 or the
  phase's section. A claim checked against the standard and a real signal beats one
  checked only against this code's own encoder.
- **Local defaults.** Australian band plans, and the European RDS PTY table.
- **Privacy.** New decoders that can carry personal data hide text on request and write
  nothing to disk.
