# RGC_SDR

An incremental, learning-focused SDR receiver for macOS (Apple silicon), built on an
Airspy HF+ over SoapySDR. Live spectrum, scrolling waterfall, click-to-tune, zoom,
named memories, **audio demodulation** (AM, NBFM, WBFM, USB, LSB), a signal meter,
recording, and a **band scanner**. Next: IQ playback, multi-device, networking.

See [PLANNING.md](PLANNING.md) for the roadmap, architecture and measured hardware facts.

Real hardware only — there is no simulated device mode, by design
([PLANNING.md](PLANNING.md) §1).

## Requirements

The native SoapySDR stack comes from Homebrew, not pip:

```bash
brew install soapysdr soapyairspyhf
```

Then the Python side:

```bash
./setup.sh
source .venv/bin/activate
```

## Usage

```bash
./run.sh                                     # or double-click "RGC SDR.app"
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

`RGC SDR.app` on the Desktop launches the current working copy — it is a thin bundle that
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
  0.00001, which is silence at any volume.
- **Mute** silences the output without losing your volume setting, and keeps the
  demodulator running so unmuting is instant. A recording in progress still captures
  audio — muting is about the room, not the file.
- **Squelch** mutes below a threshold, and is only enabled for the FM modes.
- **BW** sets the channel filter width — narrow it to pull one AM station out of a
  crowded band, or widen it for better fidelity.

The **signal meter** shows in-channel power in dBFS with a peak marker and an SNR
estimate. It reads dBFS rather than S-units on purpose: S9 means −73 dBm at the antenna,
which needs the whole gain chain calibrated, and this receiver reports no gain at all — an
S-reading would be invented. SNR is the number that actually tells you if a signal is
copyable.

Audio is continuous and independent of the display: the waterfall may drop frames, the
audio path never skips samples. The status bar reports the audio rate, the AGC gain in
use, and any underruns.

**WBFM** is decoded in **stereo** when the station sends a pilot, with **RDS**: the info
line at the top right shows STEREO or MONO, the station name, programme type and radio
text (hover for the PI code). The station name usually appears within a second or two;
the radio text is shown only once the whole message has arrived, which takes about 10
seconds. Untick **Stereo** for mono, which is quieter on a weak
station. De-emphasis is 50 us, the standard in Australia and Europe.

SSB is a proper single-sideband filter, measured at over 30 dB rejection of the opposite
sideband.

### Scanner

The **Scanner** panel sweeps a frequency range, logs every active channel it finds, and
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

### Memories

**Save…** stores the current frequency, rate, zoom and display settings under a name you
type. Pick a name from the **Memory** dropdown to jump straight back to it, or
**Delete** to remove it. They live in
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
| [src/rgc_sdr/ui/smeter.py](src/rgc_sdr/ui/smeter.py) | Signal meter (dBFS + SNR, no invented S-units) |
| [src/rgc_sdr/dsp/detect.py](src/rgc_sdr/dsp/detect.py) | Sweep planning and carrier detection |
| [src/rgc_sdr/scanner.py](src/rgc_sdr/scanner.py) | Scanner state machine, lockout, confirmation |
| [src/rgc_sdr/ui/scanner_panel.py](src/rgc_sdr/ui/scanner_panel.py) | Scanner dock: range, found list, lockout |
| [src/rgc_sdr/ui/](src/rgc_sdr/ui/) | pyqtgraph spectrum + waterfall, main window |

`dsp/` never imports Qt and `device/` never imports Qt or `dsp`, so both are testable
headless.
