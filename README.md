# RGC_SDR
An incremental, learning-focused SDR receiver for macOS (Apple silicon) using an Airspy HF+.

See [PLANNING.md](PLANNING.md) for the roadmap, architecture, and handover guidance.

The plan recommends Python + (SoapySDR / airspyhf bindings), NumPy DSP, PyQt6/pyqtgraph UI — finalized in `PLANNING.md`.

## Setup

```bash
./setup.sh          # core env (numpy + pytest) in .venv/
./setup.sh --all    # also install UI (PyQt6, pyqtgraph) and DSP (scipy)
source .venv/bin/activate
python -m src.rgc_sdr.device.spike --simulate   # verify pipeline
```
