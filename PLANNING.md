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
| AGC | **claimed but not honoured** — `hasGainMode()` = true, yet `setGainMode(False)` leaves `getGainMode()` reporting true and the level unchanged to 0.05 dB |
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
2. **This driver exposes no gain controls at all, and its AGC toggle is a lie.** There are no
   gain elements, and `hasGainMode()` returns true while `setGainMode` is silently ignored
   (measured 2026-09-23: the mode will not change and the received level moves 0.05 dB, i.e.
   nothing). Capabilities are therefore **verified, not trusted** — `_probe_caps` tries to change
   the gain mode and puts it back, and reports `has_agc=False` when the change does not stick. A
   control that does nothing is worse than no control. Other drivers (HackRF) do honour it, and
   there the toggle appears.
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
- **P4 — UX polish.** S-meter, recording (WAV/IQ), audio bandwidth control. ✅ See section 7d.
- **P5 — Extras.** Scanner ✅ (section 7e). Remaining: IQ *playback* (the recorder's
  sidecar format is designed for it), multi-device, network (SpyServer-style), plugins.
  ← **current**

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

## 7d. P4 design (metering, recording, bandwidth)

**The S-meter reads dBFS, deliberately not S-units.** S9 means −73 dBm at the antenna
terminals, and converting to it requires the whole gain chain calibrated: antenna factor,
feedline loss and receiver gain. This driver reports *no gain at all* (section 3), so any
S-reading would be fabricated. dBFS plus an SNR estimate is honest, and on an uncalibrated
receiver more useful — SNR is what actually predicts whether a signal is copyable.

The level is measured after the channel filter when audio is running, so it is the signal
being *listened to* rather than everything in the span. With audio off it is integrated
from the displayed spectrum over the same width, so the meter still works as a tuning aid.
SNR subtracts the noise contribution across the channel's bins rather than comparing peak
to median, which would flatter narrow signals.

**Recording never applies back-pressure to the signal path.** Each recorder owns a bounded
queue drained by its own thread, and drops with a count rather than blocking: a stalled
disk write in the audio path is a dropout, and on the Qt thread it is a freeze.

The IQ recorder pulls from its *own* gapless reader rather than being fed, so a capture is
continuous regardless of what the display or audio path is doing — verified by a test that
interleaves lossy display reads and asserts the file has no discontinuity.

**The size limit counts submitted bytes, not written bytes.** Checking bytes-on-disk lets
a full queue overshoot the limit by megabytes, since the feeder runs ahead of the writer.
IQ is heavy — complex64 at 768 kS/s measured **5.74 MB/s**, about 345 MB a minute — so the
default 2 GiB ceiling stops a capture rather than filling the disk.

**Retuning stops an IQ capture.** The sidecar names one centre frequency, so continuing
across a retune would make the file a lie about itself. The stop reason and the file path
are reported in one message, because two separate status updates meant the first was
always overwritten by the second.

**Audio bandwidth swaps the channel filter in place** rather than rebuilding the chain:
the decimators and detector hold state that is still valid, and tearing them down clicks.
Switching *mode* does reset a chosen width, since a figure picked for AM is meaningless
for WBFM.

**Measured 2026-09-23:** 5 s of AM to WAV (48 kHz, 16-bit mono, zero dropped blocks, audio
band 38.6 dB above the 12 kHz+ region) and 3.93 s of IQ (23 MB, zero lost samples) written
concurrently while the display held 25 FPS and audio reported zero underruns. Replaying
the IQ file on its own puts the carrier at +0 Hz from the recorded centre.

## 7e. Scanner

**Spectrum-based, not step-and-dwell.** A step-and-dwell scanner retunes to every channel
in turn: airband's 19 MHz at 25 kHz spacing is 760 retunes. This receiver already sees
768 kHz at once, so the same range is **31 windows** and one FFT finds every active
channel in each. Measured 2026-09-23: **0.34 s per window, a full airband sweep in
10.5 s**, and an FM broadcast sweep found 33 stations in 11.6 s.

**Dwelling moves the audio offset, not the radio.** A hit is by construction inside the
window already being received, so listening to it needs no retune and therefore no settle
delay. Resuming the sweep is instant, and a short transmission is not missed while the
front end recovers from a tune.

**The false-positive problem, and what actually solves it.** The first working sweep
returned 29 airband "channels" spaced almost exactly 0.61 MHz apart — one per scan window,
which is the signature of reporting the loudest noise peak in each. Measured on quiet
windows, **the loudest noise bin sits 7–24 dB above the median (mean 16.3 dB)**, so *no
fixed dB-above-noise threshold separates signal from noise*: even 20 dB left about 4 false
hits per sweep.

Peak *width* was the next candidate and does not work either. Quiet-VHF noise peaks
measured 2–3 bins wide, FM broadcast signals up to 1017 bins — but the one unambiguously
real airband carrier (122.875 MHz, 36 dB SNR) was itself only **2 bins**, the same as
noise. Narrowband AM cannot be separated from a spur by width.

What does separate them is **persistence**: a real channel reappears at the same frequency
every pass, a noise spike does not. So a channel is stored only after appearing on
`min_sightings` passes (default 2). Measured effect: **44 candidates reduced to 22
confirmed**, and the one-per-window pattern disappeared. The scanner still *dwells* on a
first sighting, so short transmissions are not missed — only the stored list waits for
confirmation, because that list is what accumulates rubbish over time.

**No DC guard is needed on this radio, but windows avoid DC anyway.** The centre bin
measured within ±1 dB of the noise floor at 20, 125 and 130 MHz, so the HF+ leaks no
detectable LO. Window centres are still planned half a channel *off* the grid, so a real
channel can never land on DC — free insurance for drivers that do leak, such as HackRF.

**Found channels are a separate list from memories.** Memories are deliberate, named
choices; found channels accumulate automatically and get cleared between sweeps. Mixing
them would let a scan bury a hand-saved frequency. A found channel can be promoted into
the memories explicitly.

**Other boundaries.** Detection is clamped to the range actually requested, since the
first and last windows overrun it. A manual tune stops the sweep rather than fighting the
user for the dial. Sweeping does not persist the frequency it lands on, which would
otherwise rewrite the saved state many times a second.

**The S-meter's SNR is computed from the spectrum on both sides of the ratio.** Mixing the
demodulator's post-filter level with a spectrum-derived noise figure compares two
different normalisations, which produced readings like "S/N −203 dB". It is also clamped
at zero: a channel at or below the noise floor has no measurable SNR, and 0 dB says
"indistinguishable from the noise" rather than dressing noise up as a measurement.

## 7f. Fine tuning and swipe gestures

**Steps go down to 10 Hz.** 1 kHz was the smallest increment, which is useless for SSB
where a few hundred hertz is the difference between intelligible speech and a comedy
voice. The hardware honours it: measured 2026-09-23, 100 Hz and 10 Hz steps come back
exact.

**A sideways two-finger swipe tunes; up and down still zooms.** Both arrive as wheel
events, told apart by which axis dominates. Trackpads deliver a stream of small deltas
rather than notches, so they are accumulated and a step is emitted each time the total
crosses a threshold -- acting on each delta would make tuning uncontrollable. A direction
reversal discards the built-up total so it takes effect at once rather than first working
through momentum the other way. `TUNE_DIRECTION` in `ui/gestures.py` is the single
constant to flip if the gesture feels backwards, since the sign depends on the system's
natural-scrolling setting.

**Two delivery paths, because one cannot be relied on.** A sideways swipe did nothing on
the first attempt. Two causes were found by instrumenting it rather than guessing:

* Wheel events are delivered by Qt to the scroll area's **viewport**, not to the view.
  Sending one to the view reaches nothing. Unit tests that call `wheelEvent()` directly
  therefore pass even when the real path is broken, so there is now a test that drives it
  through the viewport as Qt does.
* macOS trackpads populate `pixelDelta` and may leave `angleDelta` empty, so reading only
  `angleDelta` misses the gesture entirely. `wheel_deltas` takes whichever pair carries
  more information.

Beyond that, macOS may claim a two-finger sideways swipe for its own "Swipe between
pages" and never deliver it at all, which no amount of application code can fix. So
**shift plus a vertical swipe** tunes as well: a vertical swipe always arrives, since it
is what already drives the zoom. A native pan gesture is handled too, for trackpad
configurations that send one. `--debug-gestures` logs every wheel and gesture event with
its raw numbers, so the question can be settled by observation rather than theory.

**Swipe tuning is on the spectrum only**, not the waterfall: the waterfall is for reading
history, and retuning while scrolling back through it is more confusing than useful.

**A fine tune is not a retune.** This was the important discovery. `_retune` cleared the
waterfall, reset the spectrum smoothing, reset the audio chain and -- via
`set_center_freq` -- flushed the IQ ring and armed a 150 ms settle period. Doing all that
for a 100 Hz nudge defeats the entire point of tuning by ear.

Measured on hardware, 30 nudges of 100 Hz with the old behaviour cost **178,176 underrun
samples (3.7 seconds of silence)** and 18,432 lost IQ samples, because each step
discarded 150 ms of the stream. Treating a sub-bin move as an adjustment instead brought
that to **5,120 underruns and zero lost samples** — a 35x improvement — while the
waterfall history kept accumulating rather than being wiped 30 times.

The threshold is one display bin (`effective_rate / waterfall columns`, so 750 Hz at
768 kHz and proportionally less when zoomed). Below it the waterfall would shift by under
a pixel, so its history is still honest, and the move is small against any channel filter,
so the demodulator's state remains valid. `set_center_freq` therefore takes a `flush`
flag; the settle period is kept for stream restarts, where section 3 measured it is
genuinely needed. A fine nudge also no longer stops an IQ capture.

## 7g. CW, mute and snap

**CW needs a BFO.** A keyed carrier tuned exactly sits at 0 Hz, which is silent. The chain
mixes it to an audio pitch so it is audible, which is the job a beat-frequency oscillator
does in a conventional receiver. The pitch is **selectable (400-800 Hz, default 500)**,
because it is a matter of ears rather than engineering — 700 Hz was the first default and
proved tiring. Moving it retunes the mixer *and* rebuilds the channel filter together,
since one decides where the carrier lands and the other what is passed; changing either
alone would put the tone outside its own passband. The filter is a *complex band-pass centred on
the pitch* rather than a low-pass — 500 Hz wide by default, down to 100 Hz — so it passes
the tone and rejects its mirror image. Measured: a tuned carrier produces 700.4 Hz, key-up
is silence, mistuning by 200 Hz moves the tone by 200 Hz (which is how you zero-beat), and
a signal 3 kHz away is rejected by over 30 dB.

The BFO is folded into the mixer offset, so the UI's offset keeps meaning "where I am
listening" and nothing outside the chain needs to know CW is special. The passband overlay
is drawn at the pitch, since that is where the tone actually appears.

**Mute is not volume zero.** It silences the output while leaving the volume setting and
the demodulator running, so unmuting is instant and at the right level rather than waiting
for the AGC to recover. Two deliberate choices: the recorder is fed *before* muting, since
muting is a decision about the room and a silent recording would be a nasty surprise; and
mute is **not** persisted, because coming back to a silent radio with no explanation looks
like a fault.

**Snap rounds tuning to a multiple of the step**, for channelised bands — 25 kHz airband,
9 kHz medium wave. It applies to the things the user drives directly: clicking the
spectrum or waterfall, typing in the box, and swipe nudges. It deliberately does *not*
apply to a recalled memory or a scanner hit: a saved frequency is an exact choice someone
made, and scanner results are already on their own grid. Enabling it realigns immediately
rather than waiting for the next tune, and a snapped entry is written back into the
frequency box — comparing against the requested value rather than the snapped one, or the
box keeps showing what was typed while the radio sits on the nearest channel.

## 7h. CW zero beat

**Hold the button and it tunes the nearby carrier onto the 700 Hz beat note.** Only in CW
mode, only while held, and only if there is actually a carrier within 500 Hz.

**The display spectrum is useless for this.** 4096 bins across 768 kHz is 187.5 Hz per
bin, and the task is to place a carrier on a 700 Hz tone to within a few hertz. So
`dsp/zerobeat.py` runs its own measurement: a 32768-point transform over the narrow
region of interest, plus **parabolic interpolation** across the peak and its two
neighbours. That reaches a couple of hertz from 23.4 Hz bins; without the interpolation
the best possible tuning would be half a bin, an audible 12 Hz error on a beat note.

The window length is chosen for CW specifically. 32768 samples is 43 ms, short enough to
sit inside a single dot; a longer window would average across the gaps between elements
and smear the very carrier it is looking for. When a measurement lands in a gap there is
no carrier to find, the step reports nothing, and the next one tries again.

**The correction is applied in one move, not hunted.** The measurement is signed and
accurate, so the radio goes straight to the right place — up or down as needed — rather
than stepping and re-checking. Each correction is clamped to the search width so a bad
measurement cannot throw the tuning, there is a 3 Hz deadband to stop it jittering once
it is right, and snapping is deliberately bypassed: a channel grid is exactly what
zero-beating must ignore. Corrections are under 500 Hz and therefore *fine* tunes by
section 7f, so the audio keeps running while it converges.

**The detection threshold came from on-air failure, not taste.** Verified against a real
40 m CW signal at 7.04020 MHz: with the threshold at 8 dB, the first attempt converged
perfectly when mistuned downward but wandered upward, acting on 8-10 dB readings and
dragging the tuning 770 Hz onto a different signal. The genuine carrier consistently read
26-29 dB, so the threshold is 15 dB, which separates them cleanly. Re-measured afterwards,
all four cases converge in one or two steps:

| Mistuned | Tone before | After | Residual |
|---|---|---|---|
| −250 Hz | 941 Hz | 700 Hz | 0 Hz |
| +250 Hz | 443 Hz | 700 Hz | 0 Hz |
| −420 Hz | 915 Hz | 701 Hz | +1 Hz |
| +420 Hz | 768 Hz | 699 Hz | −1 Hz |

**Checked for a compounding error, and there is none.** The question came up naturally —
a beat note plus a tuning offset sounds like it could add twice. Two measurements say it
does not. Harmonic content in the CW audio is at **-105 dB** and below (-163 dB with AGC
off), so nothing is being generated. And run against identical data, `measure_carrier` and
the demodulator agree to within **1.6 Hz** across the whole +/-300 Hz range, with the
measurement itself accurate to 0.35 Hz — so the tuner and the chain share one idea of
where the carrier is.

An on-air run did once finish 79 Hz low, which is worth recording as a usage note rather
than a defect: only a single correction had been applied before the station stopped
keying, so no verifying measurement followed. Holding the button re-measures every 150 ms
and converges, which is what produced the sub-hertz residuals above. One press is not the
intended use.

Worth recording, because it shaped the tests: a carrier at amplitude 2e-4 in 1e-3 of
noise is **negative** SNR in the time domain yet over 25 dB in the spectrum. A
32768-point transform concentrates a coherent carrier into one bin while spreading noise
across all of them, roughly 42 dB of processing gain. "Weak" therefore has to be defined
spectrally, which is why the threshold test measures the signal's own SNR rather than
guessing an amplitude.

## 7i. Zoom is kept across tuning

**A retune preserves the view zoom and recentres it.** `_apply_geometry` used to reset the
frequency axis to the full span every time the tuning moved, which threw the zoom away
exactly when it was most useful: zoom in to inspect a crowded patch, click the station
next door, and the view snapped back to the whole 768 kHz.

Only the *tuning* preserves it. Changing the sample rate or the decimation changes the
span itself, which the user asked for explicitly, so those still reset the view. A
preserved window is clamped inside the available span so tuning near an edge cannot
scroll the view off the spectrum.

**This uncovered a test-suite fault worth recording.** Adding the tests produced a
segmentation fault in a test that passed perfectly well on its own. The cause was leaked
widgets: `close()` only hides a window, and with a hundred-odd of them accumulating in one
process, pyqtgraph's registry of axis-linked views ended up holding closed ones, and
following a stale link crashed. The suite now destroys each test's windows
(`setParent(None)` plus `deleteLater()`, then a collection), and `closeEvent` drops the
shared-axis link on the way out, which is correct teardown for the application too. The
suite also got faster, and stopped depending on test order.

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
