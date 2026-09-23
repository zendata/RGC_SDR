# RGC_SDR

An incremental, learning-focused SDR receiver for macOS (Apple silicon), built on an
Airspy HF+ over SoapySDR. Currently at **P2: tuning** — live spectrum, scrolling waterfall,
and click-to-tune. Next is P3 (demodulation and audio).

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
python -m src.rgc_sdr --list                 # show attached SDRs
python -m src.rgc_sdr --freq 7.1e6           # 40 m, 768 kS/s
python -m src.rgc_sdr --freq 0.909e6         # medium wave
python -m src.rgc_sdr --freq 100.1e6 --rate 384e3
python -m src.rgc_sdr --help                 # all options
```

### Tuning

- **Click** anywhere on the spectrum or the waterfall to tune there. A dotted line marks
  where the receiver is actually tuned.
- **Freq** box takes a frequency directly; **Step** sets the arrow-key/scroll increment
  (1 kHz to 1 MHz, including 9 kHz for MW channel spacing).
- **Rate** selects any of the seven supported sample rates (192 kHz to 912 kHz), which
  changes how much spectrum you see at once.

Tuning outside a tunable range snaps to the nearest one and the Freq box updates to show
where you really are — the Airspy HF+ has a gap between 31 and 60 MHz.

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

### Display

The colour range **auto-fits** to what the antenna is actually receiving a second after
launch, because real levels swing by tens of dB between setups. Override it with
`--min-db` / `--max-db`, or re-fit at any time with the **Auto** button.

Also in the window: FFT size, colour map, dBFS range and peak hold. Gain and bandwidth
controls are built from what the driver reports, so for the Airspy HF+ you get an AGC
toggle and nothing else — it exposes no gain stages and no bandwidth control.

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
| [src/rgc_sdr/ui/](src/rgc_sdr/ui/) | pyqtgraph spectrum + waterfall, main window |

`dsp/` never imports Qt and `device/` never imports Qt or `dsp`, so both are testable
headless.
