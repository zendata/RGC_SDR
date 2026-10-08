# 2. The interface

The window has three parts, top to bottom:

1. **The tab row**: six coloured toggles and the frequency display.
2. **The panels** the tabs open. Several can be open at once, and the dividers between
   them drag.
3. **The spectrum and waterfall**, which are always there.

Which tabs were open is remembered. The window opens maximised, and it is laid out to
fit a 1280-point-wide screen.

## The tab row

| Tab | Opens |
|---|---|
| **Settings** | Radio, tuning, display, audio, recording and transmit controls |
| **Decode** | The data decoders and their messages |
| **Classify** | "What is this signal?" and the whole-screen sweep |
| **Memory** | Memories in groups |
| **Map** | Ships, aircraft, APRS stations and radios that reported a position |
| **Scan** | The band scanner |

Classify and Map are hidden while the IC-705 is in use, because the radio sends no IQ.

### The frequency display

The large yellow readout is the tuned frequency, in MHz to the hertz.

- **Drag a digit** up or down to step that digit: one step every 8 pixels.
- **Swipe** up or down with two fingers over a digit, or turn the mouse wheel, to step
  that digit by one. The tuning step plays no part. Small trackpad movements add up to a
  step, so it does not race away.
- **Click** without dragging to put a text cursor there, then type a frequency.

A frequency beyond the current radio's reach hands over to a radio that reaches it (see
[More than one radio](01-iq-stream.md#more-than-one-radio)).

## The spectrum and waterfall

- **Click** either to tune there. Snap applies if it is on.
- **Swipe sideways** over the spectrum to tune by the step. If macOS claims sideways
  swipes ("Swipe between pages"), hold **Shift** and swipe vertically instead.
- **Swipe vertically** to zoom the frequency axis of the plot. This only magnifies the
  picture; **Zoom** in Settings changes what is computed.
- The shaded band on the spectrum is the **passband**: what the demodulator is
  listening to, one-sided for USB and LSB. **Drag its edges** to change the width and
  IF shift (not for P25 or DAB). The dotted line marks the tuned centre.
- **Cmd-click** the spectrum to add a manual notch there (a dashed red line), or remove
  one near the click. Notches are radio frequencies, so they stay on their carrier when
  you retune slightly.
- **Option-click** the spectrum to place a **marker**, or remove one near the click. A
  marker snaps onto the strongest bin within a few bins, and shows its frequency and
  level. The second and later markers also show their difference from the first
  (Δ kHz, Δ dB).
- **The band plan** (tick **Bands**) is a coloured strip along the bottom: amateur
  green, broadcast blue, aviation purple, marine cyan, CB orange, others grey. It is
  labelled where there is room, with the full name and range on hover. It is a
  simplified Australian plan for orientation; the ACMA spectrum plan is the authority.
- **Memory names** (tick **Names**) are shown at their frequencies along the top.
- **Labels** on the waterfall come from Classify. Each has an arrow over its signal.

## Settings

### Tuning row

| Field | What it does |
|---|---|
| **SDR** | The radio: every supported one, detected ones highlighted yellow, radios on network servers, and **Network radio servers…** at the end to name them. Choosing one makes it first choice wherever it reaches. |
| **Step** | Tuning increment for sideways swipes: 10 Hz, 100 Hz, 500 Hz, 1, 5, 9, 10, 25 or 100 kHz, or 1 MHz. |
| **Snap** | Round tuning to a multiple of the step, for channelised bands. Recalled memories and scanner hits are left exactly where they are. |
| **Rate** | The radio's sample rate, which is the span shown. Changing it restarts the stream, the audio and the decoders. |
| **Zoom** | Decimate 1–32×: a narrower span with proportionally finer bins, centred on the tuned frequency. On a network radio the server zooms its own lines. |
| **Span** | IC-705 only: the radio's scope span (centre mode). Disabled when its scope is in fixed mode. |
| **Bands** | Show the band plan along the bottom of the spectrum. Remembered. |
| **Names** | Show memory names at their frequencies. Remembered. |
| **Marker** | Put a marker on the strongest signal in view. |
| **✕** | Clear the markers. |

### Radio row

Built from what the radio reports, so it differs from radio to radio.

| Field | What it does |
|---|---|
| **AGC** | The radio's hardware AGC, where the driver has one that works. |
| **Gain stages** | One field per stage, by the driver's names: HackRF AMP, LNA and VGA; RTL-SDR TUNER; Pluto PGA; Airspy LNA, MIX and VGA. A stage that is only on or off (HackRF AMP, +14 dB) is a tick box. The HF+ has none. |
| **Driver switches** | Tick boxes for the driver's on/off settings: Bias Tee, Offset Tune, I/Q Swap, Digital AGC and so on. |
| **IF BW** | The radio's hardware filter, where it has a choice. The width it actually uses is shown even when it is not in the list (the Pluto opens at 18 MHz). |
| **PPM** | This radio's frequency error, + when it reads high. It is corrected in tuning, so the display shows the true frequency. Saved per radio; measured automatically the first time a radio is connected. |
| **Cal** | Measure the error now from a carrier at the listening frequency. Tune exactly to a signal whose frequency you know first. |

### DSP row

Receiver refinements (P10), shown for the app's own demodulators. See
[Demodulation](03-demodulation.md#receiver-refinements) for how each works.

| Field | What it does |
|---|---|
| **Shift** | IF shift: moves the passband off the listening frequency (AM, SAM, USB, LSB, CW). Reset to 0 on a change of mode. |
| **NB** | Noise blanker, Off or 1–10: removes impulses (ignition, electric fences, switching supplies). |
| **NR** | Noise reduction, Off or 1–10. Measured on a tone in white noise: +4 dB S/N at 1, +9 at 6, +12 at 10. |
| **Notch** | Automatic notch: removes steady whistles from the audio. |
| **Clear notches** | Shown when there are manual notches; removes them. |
| **AGC** | Fast, Medium (the default), Slow or Off. |
| (gain) | With the AGC off, the fixed audio gain in dB (AM, SAM, SSB and CW only; FM has a level of its own). |
| (sideband) | SAM only: Both, Upper or Lower. |
| **Noise sq** | NBFM only: squelch on the noise above the voice, as an FM radio does. |
| (quieting) | How far the noise must fall to open the noise squelch, default 10 dB. |
| (state) | The passband's channel power (dBFS) and occupied bandwidth (the span holding 99% of its power); then what the DSP is doing: SAM's lock and carrier offset, the quieting, impulses blanked, bins notched. |

All of it is saved with each memory and in the last state.

### NBFM row

Shown in NBFM mode.

| Field | What it does |
|---|---|
| **Shift** | Simplex, + or −: transmit on the listening frequency, or above or below it, to work a repeater. |
| **Rpt offset** | The repeater split. Australia: 600 kHz on 2 m and 5 MHz on 70 cm (7 MHz on some older repeaters). The band's standard is filled in. |
| (TX frequency) | Where a transmission would go. |
| **Tone** | **Off**; **Tone** (CTCSS sent on transmit); **TSQL** (CTCSS sent, and receive muted unless it is heard); **DCS** (the code sent, and receive muted unless it is heard). |
| (tone value) | The CTCSS frequency (the EIA/TIA-603 list, with 150.0 Hz) or the DCS code. |
| (tone state) | Whether the tone squelch is open. |

### Audio row

| Field | What it does |
|---|---|
| **Mode** | Off, AM, SAM (synchronous AM), NBFM, WBFM, USB, LSB, CW, P25 or DAB. The IC-705 lists its own modes (LSB, USB, AM, CW, RTTY, FM, WFM) and changes the radio's mode. |
| **BW** | The channel filter's width. The choices depend on the mode (see [Demodulation](03-demodulation.md)); on the IC-705, its filters FIL1–3. |
| **Pitch** | CW only: the beat-note pitch, 400–800 Hz, and what **Zero beat** tunes to. |
| **Stereo** | WBFM only: decode stereo when the station sends it. Untick for mono, which is quieter on a weak station. |
| **Vol** | Output volume. |
| **Mute** | Silence the output without losing the volume. The demodulator keeps running, so unmuting is instant and at the right level. |
| **TX** | Key the transmitter (also the **space bar**, unless text is being typed). Shown for radios that can transmit. See [Transmit](03-demodulation.md#transmit). |
| **Offset** | Listen this far from the tuned centre without moving the radio. Keeps a wide span on screen while you hear one signal in it. Limited to the span (on a network radio, to the IQ window). |
| **Squelch** | Mute while the channel level is below the threshold (AM, NBFM, WBFM). |
| **A** | Set the squelch 3 dB above the level in the channel now, so it just goes quiet. Press it while only noise is there. |
| **R** | Reset the squelch to −100 dBFS. |
| (threshold) | The squelch level, dBFS. |
| (S-meter) | Channel level in dBFS and an S/N estimate (see [The S-meter](01-iq-stream.md#the-s-meter)). On the IC-705, the radio's own S-meter, or Po and SWR while transmitting. |

### Record row

| Field | What it does |
|---|---|
| **Audio** | Record the demodulated audio to WAV. |
| **IQ** | Record raw IQ as SigMF (`.sigmf-data` + `.sigmf-meta`). About 6 MB/s at 768 kS/s; stops at 2 GiB, or on a real retune. |
| **Replay** | Go back over the last 10, 30 or 60 seconds received, or all that is kept. They play as a recording, then the radio comes back by itself. The menu also sets how much is kept: none, 30 s (the default), 1, 2 or 5 minutes, within 512 MB (about 16 s at 4 MS/s, 87 s at 768 kS/s). A real retune empties it. |
| **Play…** | Play an IQ recording in place of the radio: SigMF (from this app, SDR++, GNU Radio and others; complex float32, int16, int8 or uint8), or this app's older `.cf32`. Choosing any radio from the list also ends playback. |
| **Pause**, **Stop** | While playing: pause, or go back to the radio. |
| (seek bar) | While playing: where it is; drag to go elsewhere. |
| (speed) | While playing: 0.5× to 8×. Audio is right only at 1×. |
| **Loop** | While playing: start again at the end. |
| (overview) | While playing, a strip below the row: the whole recording, time left to right, frequency up. Click to go there. |

### Display row

| Field | What it does |
|---|---|
| **FFT** | Bins per spectrum: 1024 to 16384. More bins give finer resolution and slower response. |
| **Colour** | Waterfall colour map: inferno, magma, plasma, viridis, turbo, CET-L9 or CET-R4. |
| **dBFS … to …** | The waterfall's colour range. Set by hand, it is remembered per radio. |
| **Auto** | Fit the colour range to the visible history. It also happens once, automatically, after 20 rows. |
| **Peak hold** | Show the decaying peak trace on the spectrum. |

### Info row

| Field | What it does |
|---|---|
| **DAB station** | DAB mode: the stations the ensemble names, to choose which to hear. |
| (info line) | Decoded CW text with its speed; for broadcast FM, stereo or mono and the RDS name, programme type and radio text; for DAB, the ensemble. |
| **Zero beat** | CW only. Hold it to tune a nearby carrier, within 500 Hz, onto the chosen pitch. |

### Transmit row

Shown for radios with a transmitter (the HackRF; the Pluto reports one too).

| Field | What it does |
|---|---|
| **TX gain stages** | The radio's transmit gains, by the driver's names (HackRF VGA 0–47 dB, AMP +14 dB). They start at minimum. |
| **Mic** | Microphone gain before the modulator, 0–40 dB, default 25. A limiter holds peaks at full deviation, so raise it until the audio is loud enough. The status line's "drive" shows how close the peaks come. It can be changed while transmitting. |

### IC-705 controls

With the IC-705 in use, a panel rebuilt from its CI-V state replaces the SDR rows:

- **The radio's display.** The operating VFO large and the other small, mode, filter,
  TX/RX, split or duplex, tone, RIT, function indicators, and a meter.
- **The VFO/MEMORY screen.** VFO, MEMO, CALL, GROUP and A/B, as on the radio. MW, M-CLR
  and M→VFO act on a one-second hold or a right-click.
- **The FUNCTION keys.** P.AMP, ATT, AGC, NOTCH, NB, NR, SPLIT, VOX and the rest light
  when on, and step through their settings on each click. A right-click or long press
  opens the levels behind a key, as on the radio.
- **AF, RF, SQL and Power**, and **Mic** (the Mac microphone level sent to the radio).
- **The radio's SET menu**, generated from Icom's CI-V reference: every item, read in the
  background and sent as soon as it is changed. A few the app depends on are locked.
- **The radio's memory channels**: tune one, copy it into the app's memories, rename,
  retune, clear, or fill it from what the radio is on. Every write is read back.

Changes made on the radio's own screen show in the app.

## Decode

| Field | What it does |
|---|---|
| **Decode** | The decoder: POCSAG, APRS, AIS, ACARS, ADS-B, P25, DMR or DAB. Choosing one switches to the mode its signal uses (NBFM for most, AM for ACARS, P25 for P25). |
| (status) | Where it is listening and how wide, or why it cannot decode (for example, a sample rate it cannot use). |
| **Show text** | Message text is shown by default. Untick it to hide pager, P25 and DMR text for this session. Nothing decoded is ever written to disk. |
| **Map** | Open the Map tab. |
| **Clear** | Clear the messages. |

Decoders run on their own thread with their own demodulator, so they keep decoding
whatever the audio mode, and through mute and squelch. AIS listens on both of its
channels wherever the radio is tuned, provided they are in view. How each protocol is
decoded is in [Decoders](04-decoders.md).

## Classify

| Field | What it does |
|---|---|
| **?** | Classify the signal at the listening frequency: a few seconds of IQ, then its name labelled on the waterfall. |
| **Classify** | Label up to twelve strong, clean signals across what the waterfall shows, wherever the radio is tuned. Unidentified ones are skipped. A trunked system's channels share one label, and its other channels are marked in the same colour. |
| **Clear** | Clear the list and the labels. |
| (list) | What was found, newest first. Double-click to tune there. |

A name with "?" is a judgement from the signal's shape. One without was confirmed by a
decoder.

## Memory

Memories are filed in groups: LW, AM broadcast, FM broadcast, HF, VHF, UHF, Airband,
Satellites, Air Nav/Data, P25, DMR and DAB+ to start. A new memory is placed by its mode,
decoder and frequency.

| Control | What it does |
|---|---|
| **Recall** (or double-click) | Tune to the selected memory, changing radio if the memory is set up for one that is there and reaches it. |
| **Update** | Save the current settings over the memory in use, shown in bold (the last one recalled or saved). |
| **Save current…** | Store the current settings under a new name, in the selected group. |
| **Rename…** | Rename the selected memory or group. |
| **Delete** | Delete the selected memories, or an empty group. |
| **New group…** | Add a group. |
| Drag and drop | Drag memories onto a group, or onto any memory in it, to move them. |

**What a memory holds.** The station once: frequency, mode, channel width, squelch,
offset, step, snap, CW pitch, stereo, repeater shift and offset, tone, decoder, volume,
FFT size, colour map and the DSP row (shift, blanker, noise reduction, notches, AGC,
SAM sideband, noise squelch). Then each radio's own setup separately: sample rate, zoom,
gains, AGC, IF bandwidth, driver switches and colour range. Saving 621 kHz on the HackRF
leaves its Airspy HF+ setup alone, and recalling it on either radio brings back that
radio's setup. A radio recalling a memory for the first time uses its own last
settings, with the zoom chosen to give about the same span. A radio's frequency error is
never stored in a memory.

## Map

OpenStreetMap tiles, cached on disk for a week and fetched at most two at a time, under
OSM's usage policy. Targets keep a trail and drop off after a while unheard:

| Kind | From | Kept for |
|---|---|---|
| Ship (arrow), aid to navigation, base station | AIS | 30 min ships, 60 min others |
| Aircraft (arrow) | ADS-B, ACARS positions | 20 min |
| APRS station or object (orange) | APRS | 60 min |
| Radio (green) | P25 or DMR location reports (LRRP) | 30 min |

Hover over a target to see its details, and click to select it. The table lists name,
ID, kind, speed and how long ago it was heard. **Fit all** zooms to show everything.

## Scan

A spectrum scanner. Instead of stepping channel by channel, it sweeps the range a whole
window at a time and finds every active channel in each window with one FFT.

| Field | What it does |
|---|---|
| **Band preset** | Airband (25 or 8.33 kHz), Marine VHF, 2 m, VHF FM broadcast, MW broadcast, 49 m shortwave, 40 m. |
| **From**, **To** | The range, MHz. |
| **Step** | Channel spacing results are snapped to. |
| **Threshold** | How far above the measured noise floor counts as a signal (default 10 dB). |
| **Confirm** | Passes a channel must appear on before it is stored (default 2). Noise spikes rarely recur at the same frequency; real channels do. |
| **Stop on signal** | Dwell on each transmission (by moving the audio offset, so no retune), up to 30 s, resuming 2 s after it goes quiet. Untick to survey the range without stopping. |
| **Scan**, **Skip**, **Lock out** | Start or stop; leave this transmission; never stop here again. |
| **Found** | Channels found, separate from memories. **Tune**, **Lock**, **To memory**, **Clear**. Double-click to tune. |
| **Locked out** | Frequencies never stopped on. Double-click or **Unlock selected** to release. |

## The status line

For an SDR: frequency, sample rate (for a network radio, its rate and the IQ window's),
zoom, span, hertz per bin, FFT size, frame rate, peak and floor in dBFS, the radio's
overflow, timeout and error counts, and the audio: mode, rate, AGC gain, underruns and
squelched blocks. While transmitting: the frequency, mode, mic level, drive and elapsed
time. For the IC-705: its scope line count, audio, and WiFi link losses. Notices such as
"now using…" are held for a few seconds before the frame's own line replaces them.

## Start-up

The last radio used is reopened at its last frequency and settings. If it cannot be
opened, a chooser lists the radios that are there instead, plus the IC-705 over WiFi with
its login. The one that failed is listed last. **Look again** finds a radio plugged in
since, and **Quit** quits.

Two desktop launchers exist:

- **VK3RQ Super SDR** reopens the last radio.
- **VK3RQ Super SDR WiFi** goes straight to the IC-705 over WiFi. It starts through
  Terminal, which macOS's Local Network privacy has allowed onto the LAN when a Finder
  launch was not.

Command-line options (`./run.sh --help`) include `--driver`, `--freq`, `--rate`,
`--memory`, `--list`, `--list-memories`, `--play` and `--no-audio`.
