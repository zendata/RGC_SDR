# RGC_SDR

An incremental, learning-focused SDR receiver for macOS (Apple silicon), built on an
Airspy HF+ over SoapySDR. Currently at **P1: live spectrum + scrolling waterfall**.

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
python -m src.rgc_sdr --list                 # show attached SDRs
python -m src.rgc_sdr --freq 7.1e6           # 40 m, 768 kS/s
python -m src.rgc_sdr --freq 0.909e6         # medium wave
python -m src.rgc_sdr --freq 100.1e6 --rate 384e3
python -m src.rgc_sdr --help                 # all options
```

The colour range **auto-fits** to what the antenna is actually receiving a second after
launch, because real levels swing by tens of dB between setups. Override it with
`--min-db` / `--max-db`, or re-fit at any time with the **Auto** button.

In the window: FFT size, colour map, dBFS range, peak hold, and whatever gain controls the
driver actually exposes (for the Airspy HF+ that is AGC only — it reports no gain stages).

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
