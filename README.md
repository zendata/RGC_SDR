# RGC_SDR

An incremental, learning-focused SDR receiver for macOS (Apple silicon), built on an
Airspy HF+ over SoapySDR. Live spectrum, scrolling waterfall, click-to-tune, zoom,
named memories, **audio demodulation** (AM, NBFM, WBFM, USB, LSB), a signal meter,
recording and **IQ playback**, a **band scanner**, and **data decoders** (POCSAG pagers,
APRS). Next: AIS, ACARS and ADS-B, then P25 and DMR metadata.

See [PLANNING.md](PLANNING.md) for the roadmap, architecture and measured hardware facts.

Real hardware only — there is no simulated device mode, by design
([PLANNING.md](PLANNING.md) §1).

## Installing on a new Mac

One script does everything -- Xcode Command Line Tools, Homebrew, the SDR drivers, the
Python environment, the tests and the Desktop icon:

```bash
git clone https://github.com/zendata/RGC_SDR.git ~/Code/RGC_SDR
cd ~/Code/RGC_SDR && ./install.sh
```

Apple silicon, macOS 11 or later; about 10-20 minutes the first time, and it asks for
your password once (Homebrew). If the Command Line Tools are missing, macOS opens a dialog
to install them; run `./install.sh` again when that finishes. It is safe to re-run.

What it does, if you would rather do it by hand:

1. `xcode-select --install`, then [Homebrew](https://brew.sh).
2. `brew install soapysdr soapyhackrf airspy airspyhf libusb cmake pkgconf`
3. `./tools/install_drivers.sh` -- the Airspy HF+, Airspy R2/Mini and ADALM-Pluto
   drivers are not in Homebrew, so this builds SoapyAirspyHF, SoapyAirspy and (with
   libiio v0.25 and libad9361) SoapyPlutoSDR into `/opt/homebrew`. `SoapySDRUtil --info`
   should then list airspy, airspyhf, hackrf and plutosdr.
4. `./setup.sh` -- the `.venv`, made with the Homebrew Python that SoapySDR's bindings
   are built for, with those bindings linked in (they are not on PyPI).
5. `./tools/make_app.sh` -- the "VK3RQ Super SDR" icon on the Desktop.

An RTL-SDR also needs `brew install soapyrtlsdr`.

## Usage

```bash
./run.sh                                     # or double-click "VK3RQ Super SDR.app"
python -m src.rgc_sdr                        # resumes where you left off
python -m src.rgc_sdr --list                 # show attached SDRs
python -m src.rgc_sdr --freq 0.909e6         # medium wave
python -m src.rgc_sdr --freq 7.1e6 --zoom 8  # 40 m, zoomed in
python -m src.rgc_sdr --mode am --freq 0.684e6      # listen to a MW station
python -m src.rgc_sdr --mode usb --freq 14.2e6      # 20 m SSB
python -m src.rgc_sdr --memory "Radio 4 LW"  # start from a saved memory
python -m src.rgc_sdr --list-memories
python -m src.rgc_sdr --help                 # all options
```

**It starts up where you left it.** Frequency, rate, zoom, FFT size and colour map are
saved on exit and restored next launch. Any flag you pass overrides just that one setting;
`--no-restore` ignores the saved state for one run, and `--forget` clears it.

### Icom IC-705

The IC-705 is not an SDR, but it is in the same **SDR** menu: plug in its USB cable and
choose **Icom IC-705**. The spectrum and waterfall are the radio's own scope (475 points,
about 4 lines a second -- the 705's limit over USB), the frequency follows its dial, and
tuning in the app (frequency box, step, click, memories) tunes the radio. Zoom and FFT
are the radio's to set, so they are hidden.

**Mode** lists the 705's modes and **BW** its filters FIL1-3. Squelch is the radio's own
**SQL** on the Radio row (the app's Squelch box is hidden for the 705); the 705 does not
squelch the audio it sends over USB, so the app follows its squelch and silences the
audio while it is shut. The **Radio** row has its
AF, RF gain, squelch, preamp, attenuator, AGC, noise blanker, noise reduction and TX
power, as percentages like the radio's own screen, and they follow the radio's knobs.
The meter reads the radio's S-meter in S-units. If the radio refuses a setting -- the
705 fixes AGC in FM -- the status bar says so and the control shows the radio's real
value.

Its **audio** plays on the Mac (Vol and Mute work as usual). **TX** -- the red button
or the space bar -- keys the 705 with the Mac's microphone as the audio, on the 705's
frequency and at the power set on the Radio row; the meter shows power and SWR while
transmitting, and the receiver is muted.

While the app has the 705 it **silences the radio's speaker** (AF 0 -- the Mac's audio is
unaffected) and sets **DATA OFF MOD to USB**, so only the Mac microphone goes on air, not
the radio's own as well. Both are put back when you quit or switch radio, and are kept on
disk meanwhile, so even a crash cannot leave the radio's speaker and microphone off.

**Span** (in place of Zoom) sets the radio's scope span, ±2.5 to ±500 kHz, and follows
changes made on the radio; in the radio's fixed scope mode the display shows its edges. CW and WFM are not transmitted from the
app. **Mic** on the Radio row sets how loud the Mac microphone is sent -- 25% to start,
with a limiter on the peaks; lower it if you are reported distorted or over-compressed,
raise it if quiet. It works while transmitting and is remembered for the 705. Keep the scope showing on the radio: it sends no scope data
while in a menu.

**Over WiFi** -- no cable: choose **Icom IC-705 (WiFi)** in the SDR menu. The first time
it asks for the radio's address and its network user and password (kept for next time;
the password in the macOS Keychain). Everything works as on USB -- scope, controls,
meters, audio and TX -- over Icom's own remote protocol (the one RS-BA1 uses). On the
radio, once:

1. **SET > WLAN Set > WLAN** ON, and **Connection Type** "Station" to join your home
   network (then **Access Point** to pick it), or "Access Point" to have the Mac join
   the radio's own network.
2. **SET > WLAN Set > Remote Settings**: **Network Control** ON, and a **Network User1
   ID** and **Password**. Leave the ports at 50001-50003.
3. The address: **SET > WLAN Set > Connection Settings (Station) > IP Address** (fix
   it there, or reserve it on your router, so it does not change).

Over WiFi the app takes TX audio from WLAN instead of USB (DATA OFF MOD and DATA MOD =
WLAN) and sets the WLAN audio output to AF with the squelch; all put back afterwards. If
the link drops the status line says so; choose the radio again to reconnect. With no
radio on USB at all, the app offers **Connect IC-705 over WiFi…** when it starts, and
once connected it opens straight onto the WiFi radio next time. The **VK3RQ Super SDR WiFi** icon on
the Desktop (made by `./tools/make_app.sh` with the other) always opens the 705 over
WiFi, asking for the address and login if it cannot connect. Only one
program can be connected to the radio at a time.

### Choosing the radio

The **SDR** selector at the top left lists every supported radio — Airspy HF+, Airspy
R2/Mini, HackRF One, RTL-SDR and ADALM-Pluto — and says which are connected. Pick one and
the window rebuilds for it: its frequency range and sample rates, and whatever gain stages,
AGC and bias-tee it actually has. The HF+ has none of those, so none are shown.

Radios need their SoapySDR driver. `python -m src.rgc_sdr --list` shows what is installed,
connected, and how to install what isn't (`brew install soapyrtlsdr` for an RTL-SDR; the
Airspy R2 and Pluto drivers must be built from source). The app starts on the radio you
used last, or whichever is plugged in.

### Tuning

- **Click** anywhere on the spectrum or the waterfall to tune there. A dotted line marks
  where the receiver is actually tuned.
- **Freq** box takes a frequency directly; **Step** sets the tuning increment, from
  **10 Hz** up to 1 MHz — including 100 Hz for SSB and 9 kHz for MW channel spacing.
- **Snap** rounds tuning to a multiple of the step, for channelised bands like airband
  (25 kHz) or medium wave (9 kHz). Recalled memories and scanner hits are left exactly
  where they are.
- **Two-finger swipe left or right over the FFT display** tunes gradually by that step,
  so you can pitch an SSB voice by ear. Up and down still zooms.
- **Shift + two-finger swipe up or down** does the same thing, and is the reliable one:
  macOS may claim a sideways swipe for its own "Swipe between pages" gesture, in which
  case the application never sees it. A vertical swipe always arrives.

Zooming with a two-finger vertical swipe is kept when you tune: the view stays at the
same width and recentres on the new frequency, so you can zoom into a busy patch and hop
between stations without losing your place. Changing the sample rate or the Zoom
decimation resets it, since those change the span deliberately.

Tuning by gesture works on the spectrum only, not the waterfall — the waterfall is what
you read history from. If a sideways swipe does nothing, run with `--debug-gestures` to
see exactly which events your trackpad delivers.
- **Rate** selects any of the seven supported sample rates (192 kHz to 912 kHz), which
  changes how much spectrum you see at once.

Tuning outside a tunable range snaps to the nearest one and the Freq box updates to show
where you really are — the Airspy HF+ has a gap between 31 and 60 MHz.

Small adjustments are treated as adjustments: a step under one display bin (750 Hz at
768 kHz) leaves the waterfall history and the audio running, so tuning by ear does not
click or wipe the screen on every step. A larger move clears them, as it should.

### Desktop shortcut

`VK3RQ Super SDR.app` on the Desktop launches the current working copy — it is a thin bundle that
just runs [run.sh](run.sh), so it picks up code changes with no rebuild. Rebuild the bundle
itself only if the repo moves:

```bash
./tools/make_app.sh              # onto the Desktop
./tools/make_app.sh /Applications
```

With the radio unplugged it shows a dialog rather than failing silently. The icon is
generated from the app's own colour map by [tools/make_icon.py](tools/make_icon.py)
(then `tools/make_icns.sh`). Note the Dock tile says "Python" while running, because the
bundle execs a system interpreter rather than embedding one.

### Zoom

**Zoom** decimates the IQ stream: each step halves the span and doubles the resolution,
from 768 kHz / 187 Hz-per-bin at 1x down to 24 kHz / 5.9 Hz-per-bin at 32x. Useful for
pulling a narrow carrier out of what looks like flat noise at full span. Frame rate holds
at 25 FPS throughout.

### Audio

Pick a mode from the **Audio** dropdown — AM, NBFM, WBFM, USB, LSB or CW — and the shaded
band on the spectrum shows exactly what is being demodulated.

**CW** adds a beat-frequency oscillator: a keyed carrier tuned exactly would sit at 0 Hz
and be silent, so it is mixed up to an audible tone. **Pitch** selects that tone, 400 to
800 Hz, defaulting to 500 — lower notes are less tiring over a long session. Filters are
narrow (100 Hz to 1.5 kHz, 500 Hz by default) and mistuning moves the pitch, so you can
zero-beat by ear — which is what the 10 Hz and 100 Hz steps are for.

**Zero beat** (next to Peak hold, CW only) does it for you. Hold the button and it finds
the carrier within 500 Hz, works out which way to go, and tunes it onto whatever pitch you
have selected — accurate to a couple of hertz. Hold it for a second or two rather than
tapping: it re-measures several times a second and converges, which also rides out the
gaps between CW elements. It only tunes while held, does nothing if there is no
signal nearby, and ignores Snap, since a channel grid is the opposite of what
zero-beating needs. Audio keeps playing while it converges.

- **Offset** listens that far from the tuned centre without moving the radio, so you can
  watch a wide span and hear one signal inside it. The shaded band follows it.
- **Vol** is a plain output gain. An automatic gain control runs ahead of it, so a weak
  station is still audible: without it a −104 dBFS carrier gives an audio level of
  0.00001, which is silence at any volume. On **AM** the level is set from the carrier,
  which is steady whether anyone is talking or not, so speech after a pause does not
  come in loud. Other modes hold their gain through pauses of up to 0.6 s.
- **Mute** silences the output without losing your volume setting, and keeps the
  demodulator running so unmuting is instant. A recording in progress still captures
  audio — muting is about the room, not the file.
- **Squelch** mutes below a threshold, in AM, NBFM and WBFM. (SSB and CW have no
  carrier to measure, so a threshold would chop the speech.)
- **BW** sets the channel filter width — narrow it to pull one AM station out of a
  crowded band, or widen it for better fidelity.

The **signal meter** shows in-channel power in dBFS with a peak marker and an SNR
estimate. It reads dBFS rather than S-units on purpose: S9 means −73 dBm at the antenna,
which needs the whole gain chain calibrated, and this receiver reports no gain at all — an
S-reading would be invented. SNR is the number that actually tells you if a signal is
copyable.

Audio is continuous and independent of the display: the waterfall may drop frames, the
audio path never skips samples. The status bar reports the audio rate, the AGC gain in
use, and any underruns -- or, when there is no sound, why: "audio off (choose a Mode)"
or the reason audio could not start.

**WBFM** is decoded in **stereo** when the station sends a pilot, with **RDS**: the info
line beside Peak hold shows STEREO or MONO, the station name, programme type and radio
text (hover for the PI code). The station name usually appears within a second or two;
the radio text is shown only once the whole message has arrived, which takes about 10
seconds. Untick **Stereo** for mono, which is quieter on a weak
station. De-emphasis is 50 us, the standard in Australia and Europe.

SSB is a proper single-sideband filter, measured at over 30 dB rejection of the opposite
sideband.

### Transmit

The red **TX** button -- click it, or press the **space bar** -- transmits on the
frequency you are listening to (tuned frequency plus Offset), with audio from the MacBook
Air Microphone, in AM, NBFM, WBFM, USB or LSB (not CW). The mic level and TX frequency
show in the status bar. It is enabled only for radios with a transmitter (the HackRF).

- **TX gains** are on the Radio row (TX VGA, TX AMP), saved per radio. They start at
  minimum; raise them as needed. They can be changed while transmitting.
- The HackRF is half duplex, so the receiver stops while TX is on and comes back on
  release. Tuning, rate, radio and memories are locked while keyed.
- TX switches itself off after **three minutes**, on a mode or radio change, and when
  the window closes.
- No band, mode or power limits are applied: this is for in-house receiver testing
  (VK3RQ). The HackRF's harmonics are strong -- use a dummy load or a filter.
- The first press asks macOS for microphone permission.

**NBFM row** (shown only in NBFM, on any frequency):
- **Simplex / Duplex + / Duplex −** with a **Rpt offset**: transmit on the listening
  frequency, or above or below it to work a repeater. The TX frequency shows in red. The
  offset defaults to the Australian standard for the band -- 600 kHz on 2 m, 5 MHz on
  70 cm -- and an edited one (7 MHz for an older 70 cm repeater, say) is kept for that
  band. Shown only for radios that transmit.
- **Tone**: *Tone* sends a CTCSS tone; *TSQL* sends it and keeps receive muted until
  it is heard; *DCS* does the same with a DCS code. Pick the tone frequency or the code
  beside it. The tone is filtered out of the speaker, and "open"/"closed" shows whether
  it is being received. Tone squelch works on receive-only radios too.
- All of it is saved with memories.

### Scanner

The **Scan** button, right of Zoom, opens the **Scanner** panel. It sweeps a frequency range, logs every active channel it finds, and
stops on transmissions so you can hear them.

Set **From** / **To**, pick a **Step** matching the band's channel spacing, and press
**Scan** — or choose a **Band preset** (airband, marine, 2 m, FM broadcast, MW, shortwave)
which fills all three in. Airband is 118–137 MHz at 25 kHz: about 31 windows, swept in
**10 s**.

It scans by spectrum, not channel by channel: the receiver sees 768 kHz at once, so one
FFT finds every active channel in a window. That makes a sweep roughly 25x faster than
retuning to each channel in turn. Stopping on a signal moves only the audio offset, not
the radio, so resuming is instant and short transmissions are not clipped.

- **Threshold** is how far above the noise floor counts as a signal. Higher finds less but
  is more certain.
- **Confirm** is how many passes a channel must appear on before it is stored. The default
  of 2 is doing real work: the loudest *noise* peak in a window sits 7–24 dB above the
  median, so a single sighting is not evidence. In testing this cut 44 candidates to 22
  real ones. Set it to 1 to store everything, including noise.
- **Stop on signal** unchecked surveys the range without pausing, which builds a list fast.
- **Skip** leaves the current transmission; **Lock out** means never stop there again.

**Found** lists what the scan discovered — double-click to tune. This is a **separate list
from your memories**, so a scan can never bury a frequency you saved by hand; use **To
memory** to promote one. **Locked out** holds channels the scanner skips on later passes
(double-click to unlock) — useful for silencing a continuous ATIS or a local data carrier.
Both lists persist between sessions.

Tuning manually stops the scan, rather than the two fighting over the dial.

### Recording

**Record → Audio** writes the demodulated audio to a 16-bit WAV. **Record → IQ** writes
raw complex64 baseband plus a JSON sidecar describing the sample rate, centre frequency
and driver, so the capture can be replayed or analysed later. Both show elapsed time and
file size as they run.

Files land in `~/Documents/RGC_SDR` (change with `--recordings DIR`), named by timestamp
and frequency, e.g. `2026-09-23_143512_0.6840MHz_am.wav`.

IQ is big: about **5.7 MB/s** at 768 kS/s, or 345 MB a minute. Captures stop themselves at
2 GiB rather than filling the disk. Retuning also stops an IQ capture, since the sidecar
names one centre frequency and continuing would make the file describe itself wrongly.

Reading a capture back:

```python
import json, numpy as np
iq = np.fromfile("2026-09-23_143512_0.6840MHz.cf32", dtype=np.complex64)
meta = json.load(open("2026-09-23_143512_0.6840MHz.cf32.json"))
print(meta["sample_rate_hz"], meta["center_freq_hz"], iq.size)
```

### Playing a recording

**Play…** on the Record row opens an IQ recording (`.cf32` with its `.json` sidecar) and
plays it in place of the radio, in real time and looping; or start with `--play FILE`.
Everything works on it as on the radio: click-to-tune moves around inside the recorded
span, and demodulation, decoders and zoom all apply. **Pause** holds it; **Stop**, or
choosing a radio in the SDR list, goes back to the radio where it was. Nothing is saved
while a recording plays, so the next launch still opens the radio.

### Decoders

The **Decode** button, beside Scan, opens the **Decode** panel. Choose a decoder and it
runs on the listening frequency (the tuned frequency plus Offset) on its own thread,
whatever the audio is doing, including with audio off.

- **POCSAG** pagers, 512, 1200 and 2400 baud at once, either polarity, correcting up to
  two bit errors per codeword. Each page shows its address (capcode), function and type.
  Whether a page is text or numeric is judged from its content, since networks differ in
  how they use the function code. Pager messages can carry names, addresses and medical
  details, so the text is hidden unless **Show text** is ticked; that is not remembered,
  and nothing is saved to disk.
- **APRS** (AX.25 over 1200 baud AFSK; 145.175 MHz in Australia): `SOURCE>DEST,PATH:info`,
  plus the position in decimal degrees when the packet has one, in any of the plain,
  compressed or Mic-E formats.

Messages wrap to the panel's width. A memory remembers the decoder that was running when
it was saved, so recalling it starts that decoder again; memories for the national APRS
channels (145.175 MHz, WICEN 145.200 MHz, and 439.100 MHz on 70 cm, out of the HF+'s
range) are a good start.

### Memories

**Save…** stores the current station -- frequency, mode, bandwidth, step, snap, squelch,
volume, offset -- under a name you type, together with how the *current radio* was set
for it: rate, zoom, gains, AGC, IF bandwidth, bias-tee and colour range. Pick a name from
the **Memory** dropdown to jump straight back to it, or **Delete** to remove it.

Memories work on any radio. Saving the same name on another radio adds that radio's setup
and keeps the others. Recalling a station on a radio it was never saved on uses that
radio's own last settings, with the zoom picked to give about the same span.

Each radio type also remembers its own last settings -- a HackRF's gains, say -- and gets
them back whenever it is opened or switched to. Everything lives in
`~/Library/Application Support/RGC_SDR/settings.json`.

### Display

The colour range **auto-fits** to what the antenna is actually receiving a second after
launch, because real levels swing by tens of dB between setups. Override it with
`--min-db` / `--max-db`, or re-fit at any time with the **Auto** button.

Also in the window: FFT size, colour map, dBFS range and peak hold. Gain and bandwidth
controls are built from what the driver *actually honours*, so for the Airspy HF+ you get
neither: it exposes no gain stages, no bandwidth control, and although it claims to have
an AGC toggle it ignores the setting entirely (verified by trying it — the mode will not
change and the level moves 0.05 dB). Radios that do honour these grow the controls.

Device diagnostic / P0 check:

```bash
python -m src.rgc_sdr.spike --freq 7.1e6
```

## Tests

```bash
pytest                    # DSP + UI, no radio needed (Qt runs offscreen)
pytest -m hardware        # streams from the attached device
```

## Layout

| Path | Role |
|---|---|
| [src/rgc_sdr/device/source.py](src/rgc_sdr/device/source.py) | `IQSource`, `SoapyIQSource`, `DeviceCaps`, ring buffer |
| [src/rgc_sdr/dsp/spectrum.py](src/rgc_sdr/dsp/spectrum.py) | Welch-averaged spectrum in dBFS |
| [src/rgc_sdr/dsp/waterfall.py](src/rgc_sdr/dsp/waterfall.py) | Rolling history, max-pool bin reduction |
| [src/rgc_sdr/dsp/decimate.py](src/rgc_sdr/dsp/decimate.py) | Zoom decimation, stateless and streaming |
| [src/rgc_sdr/dsp/demod.py](src/rgc_sdr/dsp/demod.py) | AM / FM / SSB detectors, channel filters, audio AGC |
| [src/rgc_sdr/audio.py](src/rgc_sdr/audio.py) | Demod worker thread, FIFO, sound device |
| [src/rgc_sdr/recorder.py](src/rgc_sdr/recorder.py) | WAV and raw-IQ recording, on their own threads |
| [src/rgc_sdr/device/playback.py](src/rgc_sdr/device/playback.py) | Plays an IQ recording as a source |
| [src/rgc_sdr/decoding.py](src/rgc_sdr/decoding.py) | Decoder registry and worker thread |
| [src/rgc_sdr/dsp/bitsync.py](src/rgc_sdr/dsp/bitsync.py) | Bit recovery from transition timing |
| [src/rgc_sdr/dsp/pocsag.py](src/rgc_sdr/dsp/pocsag.py) | POCSAG framing, BCH correction, messages |
| [src/rgc_sdr/dsp/aprs.py](src/rgc_sdr/dsp/aprs.py) | AFSK, HDLC, AX.25, APRS positions |
| [src/rgc_sdr/ui/decoder_panel.py](src/rgc_sdr/ui/decoder_panel.py) | Decode dock |
| [src/rgc_sdr/ui/smeter.py](src/rgc_sdr/ui/smeter.py) | Signal meter (dBFS + SNR, no invented S-units) |
| [src/rgc_sdr/dsp/detect.py](src/rgc_sdr/dsp/detect.py) | Sweep planning and carrier detection |
| [src/rgc_sdr/scanner.py](src/rgc_sdr/scanner.py) | Scanner state machine, lockout, confirmation |
| [src/rgc_sdr/ui/scanner_panel.py](src/rgc_sdr/ui/scanner_panel.py) | Scanner dock: range, found list, lockout |
| [src/rgc_sdr/ui/](src/rgc_sdr/ui/) | pyqtgraph spectrum + waterfall, main window |

`dsp/` never imports Qt and `device/` never imports Qt or `dsp`, so both are testable
headless.
