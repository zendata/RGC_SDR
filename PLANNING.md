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
| Bandwidth control | **none** — `listBandwidths()` = `()`, `getBandwidthRange()` = `[]`; `setBandwidth()` is silently accepted while `getBandwidth()` stays 0.0 |
| Sustained throughput | 912 kHz: 99.2 % of nominal over 8 s, 0 overflows. 768 kHz: 99.0 %. Both fine |
| Restart settling | first ~4 frames after a stream restart read ≈8 dB hot broadband (peak −93.5 vs −110 dBFS steady) |
| Retune | LO change on a live stream shows no transient; only a *rate* change restarts the stream |
| Measured noise floor | ≈ −111 dBFS median on 40 m |
| Measured strongest carrier | ≈ −47 dBFS |
| Python / NumPy | 3.14.7 / 2.5.3 |
| Qt stack | PyQt6 6.11.0, pyqtgraph 0.14.0 |
| Audio stack | `sounddevice` 0.5.6, `scipy` 1.18.1 — both have Python 3.14 wheels |
| Default output | MacBook Air Speakers, 48 kHz |
| CoreAudio rates | accepts **arbitrary** rates (57000, 40625, 28500 all played) and resamples |
| Rates dividing to 48 kHz by powers of two | 768, 384, 192 kHz only — the other four land on 57.0 / 40.6 / 28.5 kHz |

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
- **P1 — Spectrum & waterfall.** IQ → FFT → live spectrum + scrolling waterfall UI. ✅
- **P2 — Tune.** Frequency entry with a step size, sample-rate selection, click-to-tune on
  spectrum and waterfall, AGC toggle. ✅
  *(Corrected twice from v1: there are no LNA/VGA gain sliders and **no bandwidth control** —
  this driver exposes neither. Gain and bandwidth UI are both built from `DeviceCaps`, which
  reports empty tuples here, so neither control appears. A radio that does offer them, such as
  HackRF, grows the controls with no code change.)*
- **P3 — Demod.** AM, NBFM, WBFM, USB and LSB with audio out. ✅ See section 7c.
- **P4 — UX polish.** S-meter, recording (WAV/IQ), audio bandwidth control. ← **current**
- **P5 — Extras.** Scanner, multi-device, network (SpyServer-style), plugins.

**Pulled forward out of order (requested 2026-09-23), see section 7b:** decimation/zoom
(originally part of the P3 DSP chain) and named memories with last-state restore
(originally P5 bookmarks). Both are done. P3 remains the next phase.

Each phase ends runnable and tested.

## 7. P1 design (spectrum & waterfall)
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

## 7a. P2 design (tuning)
**Retuning drops derived state.** The IQ ring, waterfall history, spectrum smoothing and peak
hold all describe the *previous* tuning. Keeping any of them across a retune smears stale signal
across the new span, so all four are cleared together.

**Settling.** After a stream restart the front end needs ~150 ms; samples read during that window
are discarded rather than rendered, or they paint one bright row across the waterfall and skew the
auto-ranging. Implemented as a sample-count threshold (`_drop_until`) written only by the calling
thread and compared against a counter written only by the reader thread, so it needs no lock.

**Rate changes restart the stream.** SoapySDR will not accept a rate change on an active stream,
so the reader thread is stopped and restarted around it and the ring is rebuilt (its capacity is
derived from the rate). Frequency changes need none of this.

**Two disjoint tuning ranges.** 0.009–31 and 60–260 MHz means a requested frequency can fall
*between* ranges rather than merely out of bounds, so `DeviceCaps.clamp_freq` snaps to the nearest
range edge and the UI reflects the clamped value back into the frequency box rather than letting
the display and the hardware disagree.

**Click-to-tune** is a left-click on either the spectrum or the waterfall; pyqtgraph raises
`sigMouseClicked` only for a click without a drag, so it does not fight panning. A dotted centre
marker shows where the receiver is actually tuned.

## 7b. Decimation and memories

**Decimation is a cascade of halving stages, not one filter per factor.** A single
anti-alias filter sized for 32x needs roughly 40*M taps to keep its transition band
narrower than the passband it is protecting — about 1280 taps, whose polyphase
temporaries run to tens of megabytes. Each halving stage instead faces the same relative
problem, so a fixed 63-tap Blackman-windowed sinc is correct at every stage and the cost
falls geometrically. A flat 129-tap filter was measured first and rejected: at 32x its
transition band is *wider than the surviving passband*, so it would not actually
anti-alias while appearing to work.

Measured 2026-09-23: per-stage stopband **-75.3 dB**, and a full-scale out-of-band tone
reaches the zoomed display at **-100.5 dBFS** — below the receiver's own noise floor, so
aliasing is not a practical concern at any offered factor.

**Frame cost against the 40 ms budget for 25 FPS:**

| Zoom | 1x | 2x | 4x | 8x | 16x | 32x |
|---|---|---|---|---|---|---|
| ms/frame | 0.8 | 1.9 | 4.4 | 9.5 | 16.4 | 16.9 |
| span (kHz) | 768 | 384 | 192 | 96 | 48 | 24 |
| resolution (Hz/bin) | 187.5 | 93.8 | 46.9 | 23.4 | 11.7 | 5.9 |

**Welch segments are traded away as zoom increases.** Deep zoom needs `fft_size * factor`
input samples for even a single FFT, so a frame's input is capped at a fraction of the
sample rate and the segment count falls out of whatever that allows. This is why the cost
plateaus between 16x and 32x. The ring also has a floor (`MIN_RING_SAMPLES`) independent
of sample rate, because 16384 bins at 32x needs ~530k input samples, which is more than a
second's worth at the lower rates.

**Memories** are the radio sense of the word: named presets of frequency, rate, zoom and
display choices. Stored as one JSON file under
`~/Library/Application Support/RGC_SDR/`, written atomically via a temp file and
`os.replace` so an interrupted write cannot corrupt it. Loading is deliberately forgiving
— a missing, truncated or hand-edited file falls back to defaults rather than stopping
the application, since losing a preset must never cost you the radio.

**A fitted colour range is not a preference.** The auto-fit suits today's antenna and
conditions, so it is saved as `null` and re-fitted on the next launch; only a range the
user typed is restored verbatim. Same reasoning as the auto-fit itself (section 7).

**Startup precedence** is explicit flag, then `--memory NAME`, then the last-used state.
`--no-restore` ignores saved state for one run and `--forget` clears the file.

## 7c. P3 design (demodulation and audio)

**Audio needs a gapless reader; the display does not.** `read_latest` is lossy by design —
a display only ever wants the newest frame and skipping is invisible. Audio cannot skip:
every dropped sample is an audible click. So the ring gained a `SequentialReader` holding
an absolute cursor, which resyncs and *counts what it lost* if the writer laps it. The two
readers are independent, so the display's lossy peeking cannot disturb audio.

This also exposed a latent bug: `_Ring.clear()` used to rewind the physical write pointer
to 0, which breaks the invariant that absolute sample index maps to `index % capacity` —
so a reader resyncing after a retune was handed pre-retune samples. `clear()` now leaves
the pointer alone.

**Everything in the chain is stateful and continuous.** Any discontinuity at a block
boundary is audible as a buzz at the block rate, so the mixer carries its phase (wrapped,
or it loses precision after a few minutes), the FIRs carry their history, the FM detector
carries its previous sample, and `StreamDecimator` carries both filter history *and*
decimation phase — when a stage's input length is odd, the next block's `[::2]` point
must shift or the output rate drifts and audio eventually starves.

**Three threads meet in `audio.py`:** the device reader fills the IQ ring, a worker pulls
sequentially and demodulates, and PortAudio's callback drains a FIFO. The worker exists so
the callback never allocates or blocks — either produces a dropout. The FIFO is bounded
(a stalled sink discards rather than growing), counts underrun and dropped samples
separately, and is pre-filled before the stream opens: starting the stream immediately
guarantees one buffer of silence, which is a click on every start.

**No fractional resampler is needed.** Only 3 of the 7 device rates divide to 48 kHz by
powers of two, but CoreAudio accepts arbitrary output rates and resamples itself
(measured: 57000, 40625 and 28500 Hz all play), so the chain decimates by a power of two
into the 24–96 kHz range and hands that rate straight to the device.

**Audio AGC is not optional.** AM envelope output is proportional to absolute signal
strength: measured on air, a −103.9 dBFS carrier produced an audio RMS of **0.00001** —
inaudible at any volume setting. With AGC the same signal gives **0.126** RMS. It uses
asymmetric time constants (fast attack so it cannot blast, slow decay so it does not pump
the noise floor) and jumps straight to the right gain on its first block, because easing
in from unity at the slow rate takes ten seconds of near-silence and reads as broken.
`max_gain` stops a dead channel being amplified to full scale.

**Mode choices.** WBFM is treated as 150 kHz rather than the nominal 180: the decimation
cascade's own anti-alias filters retain about ±0.41 of the output rate, so 180 kHz would
sit in the transition band. It is mono — no stereo pilot decoding. SSB uses a *complex*
asymmetric band-pass: with the carrier at 0 Hz the upper sideband occupies 0..+B and the
lower −B..0, so a real low-pass cannot separate them (it is symmetric); modulating a
half-width low-pass up to ±B/2 passes one side only, measured at >30 dB rejection of the
other. Squelch applies only to the FM modes, which is what `ModeSpec.squelch_capable`
drives in the UI.

**Measured on air:** AM on a −103.9 dBFS carrier 225 kHz off centre, 8 seconds of
continuous playback with zero underruns, zero dropped samples, zero lost IQ and zero chain
errors, while the display held 25 FPS.

## 8. Testing & quality
- Pure-DSP tests run headless with synthetic IQ arrays, no radio and no Qt:
  tone lands in the expected bin; full-scale complex tone reads 0.0 dBFS; no mirror image
  (catches `fftshift`/sideband errors); waterfall ordering correct across many pushes;
  max-pooling preserves a single-bin carrier that averaging would bury.
- Hardware tests are marked `@pytest.mark.hardware` and skip automatically when no device is
  enumerated, so `pytest` stays green unplugged and verifies real streaming when plugged in.
- Manual smoke check per phase (attach → stream → render → shutdown with no dangling thread).

## 9. Risks / unknowns
- ~~912 kHz sustained USB throughput~~ — **closed 2026-09-23**: 99.2 % of nominal over 8 s with
  zero overflows. 768 kHz remains the default; all seven rates are selectable.
- pyqtgraph 0.14 + PyQt6 6.11 on Python **3.14** is a very new stack, and `ImageItem`
  colormap/axis-order APIs have shifted between versions — pin behaviour with an early smoke run.
- ~~`scipy` / `sounddevice` wheels for Python 3.14~~ — **closed 2026-09-23**: sounddevice
  0.5.6 and scipy 1.18.1 both install and work. scipy is used only for FM de-emphasis.
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
