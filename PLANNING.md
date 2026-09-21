# RGC_SDR — Planning & Implementation
Status: v2 · 2026-09-21 · Target: macOS 15 (Apple M3, 16 GB) · Hardware: Airspy HF+ over USB-C

## 1. Goal
A desktop SDR receiver application (in the spirit of SDR++ / SDR#), built incrementally for
learning. Scope starts small — live spectrum + waterfall — then grows feature by feature toward
tuning, demodulation and audio.

**Real hardware only.** There is no simulated device mode and none should be added: the Airspy HF+
is present and streaming, and a fake device path costs maintenance while hiding driver-level
behaviour (read granularity, overflows, capability gaps) that this project exists to learn.
Unit tests still build synthetic IQ *arrays* to feed pure DSP functions — that is test data for a
function, not a simulated device, and it keeps the DSP layer verifiable without the radio attached.

## 2. Language
**Python for the application; native code stays behind library bindings for the hot path.**
- Fast iteration while learning; strong DSP (NumPy), GUI (Qt) and plotting ecosystems.
- Real-time performance comes from NumPy's vectorised C loops and SoapySDR's native driver.
- If profiling ever finds a hotspot, drop it into `numba`/`cython` or a small C extension without
  rewriting the app.

Deferred until profiling justifies it: rewriting the DSP core in C/C++/Rust.
Alternatives considered and rejected for now: Rust + egui (steeper curve), C++/Qt (slower iteration).

## 3. Verified environment (measured, not assumed)
Probed 2026-09-21 on the actual device — these numbers override vendor-datasheet guesses.

| Property | Value |
|---|---|
| Soapy modules installed | `libairspyhfSupport.so`, `libHackRFSupport.so` |
| Device | `driver=airspyhf`, `AirSpy HF+ [3b52d65dfdbf3f9e]` |
| Sample rates | 912, 768, 650, 456, 384, 228, 192 kHz |
| Frequency ranges | 0.009–31 MHz, 60–260 MHz |
| Stream formats | CF32, CS16, CS8, CU16, CU8 (CF32 normalised to ±1.0) |
| `getStreamMTU` | 65536 |
| **Actual `readStream` return** | **2048 samples per call, regardless of buffer size** |
| Gain elements | **none** — `listGains()` empty, `getGainRange()` = 0.0–0.0, `getSettingInfo()` empty |
| AGC | `hasGainMode()` = true (on/off only) |
| Measured noise floor | ≈ −111 dBFS median on 40 m |
| Measured strongest carrier | ≈ −47 dBFS |
| Python / NumPy | 3.14.7 / 2.5.3 |
| Qt stack | PyQt6 6.11.0, pyqtgraph 0.14.0 |
| Not installed | `scipy`, `sounddevice` (neither needed until P3) |

Three consequences that shape the design:
1. **2048-sample reads mean 375 `readStream` calls/sec at 768 kHz.** That cannot run on the GUI
   thread — a reader thread feeding a ring buffer is mandatory, not an optimisation.
2. **This driver exposes no gain controls at all.** Any "set LNA/VGA gain" UI is unimplementable
   here; gain control reduces to an AGC toggle. Other drivers (e.g. HackRF) *do* expose elements,
   so the UI must be built from probed capabilities rather than hardcoded per device.
3. **Signal levels are very low** (floor ≈ −111 dBFS), so sensible default colour limits and an
   auto-fit control matter more than they would on a strong-signal receiver.

## 4. Device access & multi-SDR support
SoapySDR is the abstraction seam. Future radios (HackRF is already installed) must work without
touching DSP or UI code, so:
- One generic `SoapyIQSource`, parameterised by driver/serial — **not** a class per radio.
- Capabilities are **probed at open time** into a `DeviceCaps` record (rates, frequency ranges,
  gain elements, AGC, formats). UI controls are generated from `DeviceCaps`.
- Device-specific quirks are data, not subclasses.

Note: `SoapySDR.Device("driver=airspyhf")` (string form) opens the device;
`SoapySDR.Device(dict(driver="airspyhf"))` fails with `Device::make() no match`. Use the string form.
The `libairspyhf`-via-`ctypes` fallback in v1 of this plan is dropped as unnecessary.

## 5. Architecture (layered, swappable)
```
Hardware layer   SoapySDR driver → reader thread → IQ ring buffer
DSP layer        window → FFT → Welch average → dBFS → (later) decimate/filter/demod
UI layer         PyQt6 + pyqtgraph spectrum & waterfall (~25 FPS)
App layer        entry point, device selection, (later) bookmarks
```
**Hard rule: `dsp/` never imports Qt, and `device/` never imports Qt or `dsp`.** This keeps the DSP
headless-testable and lets modules be swapped independently.

## 6. Roadmap (incremental)
- **P0 — Spike.** Open device, stream IQ, report rate & levels. ✅ *(real hardware; verified above)*
- **P1 — Spectrum & waterfall.** IQ → FFT → live spectrum + scrolling waterfall UI. ← **current**
- **P2 — Tune.** Set frequency, sample rate and bandwidth from the UI; click-to-tune on the
  waterfall; AGC toggle. *(Corrected from v1: no LNA/VGA gain sliders — this driver has no gain
  elements. Build gain controls from `DeviceCaps.gain_elements`, which is empty here.)*
- **P3 — Demod.** AM audio out, then NBFM/WBFM, then SSB (USB/LSB). Needs `sounddevice`; verify
  Python 3.14 wheels for it and `scipy` before starting.
- **P4 — UX polish.** Gain/squelch where supported, S-meter, recording (WAV/IQ).
- **P5 — Extras.** Bookmarks, scanner, multi-device, network (SpyServer-style), plugins.

Each phase ends runnable and tested.

## 7. P1 design (current phase)
**Threading.** Reader thread owns the device and loops `readStream` into a ~0.5 s ring buffer
(384 k × complex64 ≈ 3 MB), holding a lock only for the index update. The GUI thread runs a 25 FPS
`QTimer`, snapshots the newest samples and renders. Dropping intermediate samples is correct for
display; the ring buffer is nonetheless the right shape to feed P3 audio, which needs continuity.

**Spectrum.** FFT size 4096 (188 Hz/bin at 768 kHz), periodic Hann window. At 25 FPS, 30720 samples
arrive per frame but one 4096-point FFT uses only 13 % of them — so average ~16 segments at 50 %
overlap (Welch). This smooths the noise floor *and* stops narrow bursts being missed, for almost no
cost. Average **power**, then convert to dB (averaging dB values is wrong).

**dBFS normalisation.** `20·log10(|X| / Σw)`. For complex IQ a tone lands in one bin and sums
coherently, so the normaliser is `Σw` — **not** `Σw/2`, which is the real-signal case and reads
6 dB hot. A full-scale complex tone must read 0.0 dBFS; this is pinned by a unit test.

**Waterfall.** 512 rows ≈ 20 s of history, newest at top. Roll by memmove (`buf[1:] = buf[:-1]`):
2 MB/frame ≈ 50 MB/s, negligible on an M3. The zero-copy double-buffer trick (write each row at
`p` and `p+H`, display the contiguous `buf[p+1:p+H+1]`) is held in reserve for when profiling asks.

**Bin → pixel reduction uses max-pooling, not averaging.** When `fft_size` exceeds the widget width,
averaging makes narrow carriers fade out; max preserves them. Implemented with
`np.maximum.reduceat` over `arange(width)*n//width` edges, which handles non-divisible widths
correctly (naive `reshape`-based pooling silently drops whole trailing buckets).

**Defaults.** Colour range −115 to −40 dBFS (seeded from measurement), with a percentile auto-fit.
Exponential smoothing (α ≈ 0.3) on the spectrum curve only — the waterfall stays unsmoothed so
transients remain visible.

## 8. Testing & quality
- Pure-DSP tests run headless with synthetic IQ arrays, no radio and no Qt:
  tone lands in the expected bin; full-scale complex tone reads 0.0 dBFS; no mirror image
  (catches `fftshift`/sideband errors); waterfall ordering correct across many pushes;
  max-pooling preserves a single-bin carrier that averaging would bury.
- Hardware tests are marked `@pytest.mark.hardware` and skip automatically when no device is
  enumerated, so `pytest` stays green unplugged and verifies real streaming when plugged in.
- Manual smoke check per phase (attach → stream → render → shutdown with no dangling thread).

## 9. Risks / unknowns
- 912 kHz rate is untested for sustained USB throughput; default to 768 kHz and expose the rest.
- pyqtgraph 0.14 + PyQt6 6.11 on Python **3.14** is a very new stack, and `ImageItem`
  colormap/axis-order APIs have shifted between versions — pin behaviour with an early smoke run.
- `scipy` / `sounddevice` wheels for Python 3.14 are unverified; both are P3 concerns, not P1.
- PyQt6 is GPL. Fine for private/learning use; switch to PySide6 (LGPL) if this is ever distributed.
- Overflow handling: the driver reports `SOAPY_SDR_OVERFLOW`; surface the count in the UI rather
  than swallowing it, so buffer tuning is observable.

## 10. Repo layout
```
RGC_SDR/
  PLANNING.md  README.md  setup.sh  pyproject.toml
  src/rgc_sdr/
    __main__.py          entry point
    spike.py             P0 CLI diagnostic (--list, stream stats)
    device/source.py     IQSource ABC, SoapyIQSource, DeviceCaps, ring buffer
    dsp/spectrum.py      SpectrumAnalyzer (no Qt)
    dsp/waterfall.py     WaterfallBuffer, max-pool reduction (no Qt)
    ui/spectrum_view.py  live spectrum curve + peak hold
    ui/waterfall.py      pyqtgraph waterfall image view
    ui/main_window.py    layout, frame pump, wiring
  tests/
```

## 11. Handover notes for any AI/model
- Read this file first; §6 defines done-ness per phase.
- §3 is measured fact — trust it over datasheets, and re-measure rather than assume if hardware changes.
- **Do not add a simulated device mode.** See §1.
- Conventions: Python 3.14, type hints, NumPy-vectorised DSP, no per-sample Python loops.
- Respect the layering rule in §5 and keep device support driver-agnostic (§4).
