# RGC_SDR — Planning & Implementation
Status: draft v1 · 2026-09-21 · Target: macOS (Apple M3, 16GB RAM) · Hardware: Airspy HF+ over USB-C

## 1. Goal
A desktop SDR receiver application (like SDR++ or Airspy SDR#) built incrementally for learning.
Scope starts small: live spectrum/waterfall + audio demodulation, then grows feature by feature.

## 2. Language recommendation
**Recommendation: Python for the application, with C/C++ kept behind library bindings for the hot DSP path.**
Rationale:
- Fast iteration while learning; rich ecosystem for DSP (NumPy/SciPy), GUI (Qt), and plotting.
- Real-time performance comes from NumPy's vectorized (C-backed) ops and SoapySDR/airspyhf bindings, which are native code underneath.
- If a hotspot ever needs more speed, it can be dropped into a small C extension or `numba`/`cython` without rewriting the app.
- Pure C (or C++/Rust) is a valid alternative for the DSP core; defer that decision until profiling shows a need.

Alternatives considered: Rust + egui (harder learning curve), C++/Qt (SDR++/SDR# style, slower iteration).

## 3. Key dependencies (recommended)
- `soapyairspyhf` (SoapySDR module for Airspy HF+) — abstracts USB/libusb access. Fallback: direct `libairspyhf` C API via `ctypes`.
- `numpy`, `scipy` (filters), `sounddevice` or `pyaudio` (audio out), `PyQt6` + `pyqtgraph` (or ` matplotlib` embedded) for UI + waterfall.
- Optional later: `numba` for jittered DSP, `pytest` for tests.

## 4. Architecture (layered, swappable)
```
Hardware layer   airspyhf driver (SoapySDR) → IQ sample stream
DSP layer        tweakable chain: decimation → filters → demod (AM/FM/SSB) → audio
UI layer         PyQt6 widgets + pyqtgraph waterfall/spectrum (10-30 FPS)
App layer        main window, device control, bookmarks (later)
```
Keep the DSP chain decoupled from the UI so models/LLMs can swap modules.

## 5. Roadmap (incremental)
P0 — Spike: open device, stream IQ, print sample rate & SNR. Prove USB + driver works. ✅ (simulated; run with `--simulate` on the CLI, real device when connected)
P1 — Spectrum: IQ → FFT → scrolling waterfall UI.  
P2 — Tune: set frequency, gain (LNA/AGC), bandwidth from UI.  
P3 — Demod: AM audio out. Then FM (WBFM/NBFM), then SSB (USB/LSB).  
P4 — UX polish: click-to-tune on waterfall, gain/squelch sliders, recording (WAV/IQ).  
P5 — Extras: bookmarks, scanner, remote/network (e.g. SpyServer-style), plugins.

Each phase ends runnable and tested; pick the next phase you want to learn.

## 6. Testing & quality
- Unit tests with synthesized IQ (no hardware needed): FFT output, filter response, demod of generated AM/FM signals.
- pytest; optional CI (GitHub Actions) later.
- Manual smoke checklist per phase (device attach → stream → demod → audio).

## 7. Risks / unknowns
- macOS driver for Airspy HF+ on arm64 (M3): verify `soapyairspyhf`/`libairspyhf` availability via Homebrew; else use `libairspyhf` via `ctypes`.
- USB throughput & buffer handling on Airspy HF+ (it’s modest ~ up to 768 kHz — fine).
- Audio latency/glitches: choose `sounddevice` with callback; tune block sizes.
- PyQt6 GPL/commercial licensing — confirm acceptable use; else switch to PySide6 (LGPL).

## 8. Repo layout (initial)
```
RGC_SDR/
  PLANNING.md
  README.md
  .gitignore
  src/rgc_sdr/        (package; modules: device, dsp, ui, app)
  tests/
```
Populate as phases land; keep modules small and independently testable.

## 9. Handover notes for any AI/model
- Read this file first; roadmap P0–P5 defines done-ness per step.
- Hardware specifics: Airspy HF+ (HF/VHF, 0.5–31 MHz & 60–260 MHz, 18-bit ADC, ~648/768 kHz BW). No clock/tuner issues; simple gain blocks.
- Conventions: Python 3.11+ (or use `uv`), type hints, NumPy vectorized DSP, avoid per-sample Python loops except in proven hotspots with numba.
- Don’t break driver abstraction; use Soapy where possible.
