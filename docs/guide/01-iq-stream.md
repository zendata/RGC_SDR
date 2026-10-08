# 1. The IQ stream

Everything RGC_SDR does starts from one stream of complex samples. This document follows
that stream from the radio to the screen and to every consumer that reads it.

## What IQ is

An SDR mixes a slice of the radio spectrum down to 0 Hz and samples it twice at each
instant: the in-phase (I) and quadrature (Q) parts. Each sample is one complex number,
`I + jQ`. With both parts, positive and negative frequencies can be told apart, so a
stream at *R* samples per second shows the whole band from *centre − R/2* to
*centre + R/2*. A signal 30 kHz above the tuned frequency is a phasor turning at
+30 kHz, and one 30 kHz below is turning at −30 kHz.

RGC_SDR keeps samples as `complex64`: two 32-bit floats, where magnitude 1.0 is the
radio's full scale. That is why levels are shown in **dBFS** (decibels relative to full
scale), not in dBm. Converting to dBm would need the whole gain chain calibrated, and the
Airspy HF+ reports no gain at all (PLANNING.md section 3).

## Radios

Every SDR is opened through **SoapySDR**, by one generic class, `SoapyIQSource`
(`device/source.py`). There is no class per radio. What each radio can do is probed when
it opens and stored as a `DeviceCaps` record: its sample rates, frequency ranges, gain
stages, AGC, IF bandwidths, on/off driver settings such as a bias-tee, and its
transmitter if it has one. The interface builds its controls from that record, so a new
radio needs no interface code.

Probing is checked rather than trusted. SoapyAirspyHF claims an AGC and then ignores it,
so the AGC claim is tested by switching it and reading it back.

`device/profiles.py` adds what probing cannot tell: a menu name, how to install the
driver, sensible starting rates and gains, and two limits that come from this
application:

| Radio | Coverage | Default rate | Highest rate offered | LO offset | Gains |
|---|---|---|---|---|---|
| Airspy HF+ | 0.009–31 and 60–260 MHz | 768 kS/s | 912 kS/s | 100 kHz | none |
| Airspy R2 / Mini | 24–1800 MHz | 2.5 MS/s | 10 MS/s | none | LNA, MIX, VGA |
| HackRF One | 1–6000 MHz | 4 MS/s | 10 MS/s | 200 kHz | AMP, LNA, VGA |
| RTL-SDR | 24–1766 MHz | 2.048 MS/s | 2.4 MS/s | 200 kHz | TUNER |
| ADALM-Pluto | 325–3800 MHz | 2 MS/s | 6 MS/s | 200 kHz | PGA |
| Icom IC-705 (USB or WiFi) | 0.03–200, 400–470 MHz | its scope | — | — | the radio's own |

- **10 MS/s ceiling.** Above that the NumPy audio chain cannot keep up. Measured: 22% of
  a core at 6 MS/s, 37% at 10, 74% at 20.
- **Pluto at 6 MS/s.** Its USB link silently drops samples above that (measured).

### The LO offset and the DC spike

Most SDRs show a spike at exactly the tuned frequency: leakage from their own local
oscillator. Listening on top of it ruins the signal. So radios are tuned **away** from
the wanted frequency by their LO offset (200 kHz for most, 100 kHz for the HF+), and the
stream is shifted back in software. The spike then sits at ±200 kHz on the display, and
the wanted frequency is clean at the centre.

The shift is done by `_Nco` in the reader thread: it multiplies each block by a rotating
phasor. When the shift divides the sample rate into a short cycle (200 kHz into 4 MS/s
is 20 samples), it uses one period of the phasor as a lookup table: one multiply a
sample instead of one complex exponential. Near the top of a radio's range the offset
flips below the wanted frequency instead.

The HF+ has no significant spike (measured 2026-10-05), but it is offset by 100 kHz at
the operator's request, and that is automatic. The scanner and the Classify sweep step
around wherever the spike is, using `dc_spike_offset_hz`.

### Frequency correction (ppm)

Cheap crystals run a few parts per million off: the Pluto reads about +4.7 ppm and the
HackRF about −11.7 ppm, which is 1.5 kHz at 131 MHz. Each radio keeps a correction, and
tuning asks the hardware for `wanted × (1 + ppm·10⁻⁶)`. The displayed frequency is then
the true frequency. The correction is measured from a known carrier (see
[Decoders: calibration](04-decoders.md#calibration)), saved per radio, and never stored
in a memory: it belongs to the radio, not the station.

## The reader thread and the ring buffer

A background thread calls `readStream` in a loop and writes each block into a **ring
buffer** (`_Ring`): about one second of samples, and never less than 1.2 million, so deep
zoom always has enough history. The thread is required, not an optimisation. The HF+
driver returns 2048 samples a call, about 375 calls a second, far too many for the Qt
event loop to service.

The ring holds its lock only around index arithmetic and the copy, never around device
I/O. It counts every sample ever written, so a reader can hold an absolute position and
tell whether it has been overtaken.

### Two ways to read

| Reader | Used by | Behaviour |
|---|---|---|
| `read_latest(n)` | spectrum, waterfall, S-meter, scanner, zero beat | The newest *n* samples. Lossy: frames in between are skipped, which is invisible on a display. |
| `sequential_reader()` | audio, decoders, recorders, Classify, calibration | Each reader keeps its own cursor and gets every sample in order. If it falls more than a ring behind, it jumps to the oldest valid sample and counts what it lost (`lost`), rather than reading torn data. |

### Retuning

A retune can either flush the stream or leave it running:

- **A real retune** clears the ring (without moving its write position, so every reader's
  cursor stays meaningful), discards about 150 ms while the front end settles, and resets
  the spectrum, waterfall, audio and decoders. Rendering samples from the old frequency
  would smear a stale signal across the new span. The settling matters: the first frames
  after a restart read about 8 dB hot across the whole band.
- **A fine adjustment**, smaller than one waterfall column, keeps everything. Clearing on
  every 100 Hz nudge used to cost 3.7 seconds of silence over 30 nudges. An error that
  small is inaudible inside a channel, so the audio runs on.

A **sample-rate change** stops the stream, sets the rate, rebuilds the ring at the new
size and restarts. SoapySDR refuses a rate change on an active stream.

## From IQ to the spectrum and waterfall

The window runs a 25 frames-a-second timer. Each frame:

1. **Fetch** the newest samples: enough for 16 overlapping FFT segments, capped at
   0.6 seconds of input.
2. **Zoom** (decimate) if asked: 2× to 32×. Each 2× is one *halving stage*: a 63-tap
   windowed-sinc low-pass at a quarter of the rate, then every other sample kept. The
   stage filter is half-band, so nearly half its taps are exactly zero, and it is
   symmetric. `HalvingFilter` computes only the samples kept, from the 16 distinct
   nonzero taps. On the Pi 5 that took decimation by 8 at 2.048 MS/s from 41% of a core
   to 8%, with identical output. Zooming narrows the span and makes each bin
   proportionally finer, centred on the tuned frequency.
3. **Welch spectrum** (`dsp/spectrum.py`): FFTs of 1024 to 16384 points, a periodic Hann
   window, 50% overlap, up to 16 segments, all in one batched FFT. Power is averaged
   *then* converted to dB; averaging dB values would be wrong. It is normalised by the
   window's sum, so a full-scale complex tone reads exactly 0 dBFS. A test pins that.
4. **Spectrum curve**: lightly smoothed (each frame moves 30% of the way to the new
   value), with a peak-hold trace that falls 0.5 dB a frame.
5. **Waterfall**: the same row, unsmoothed so short bursts stay visible. It is reduced
   to 1024 columns by **max-pooling**, because averaging would make narrow carriers fade
   out. It keeps 512 rows (about 20 seconds). Empty rows are drawn at −140 dBFS.
   **Auto** fits the colours to the 5th to 99.5th percentile of the history plus 3 dB.
6. **S-meter, scanner, decoders and status** are updated from the same spectrum.

### The S-meter

The meter shows the level in the channel being listened to, in dBFS, plus a
signal-to-noise estimate. It deliberately shows no S-units: S9 is defined in dBm at the
antenna, which this receiver cannot know.

- **Level.** With audio on, it comes from the demodulator, measured after the channel
  filter, so it is exactly what is heard. With audio off, it is integrated from the
  spectrum over the channel width.
- **S/N.** Always from the spectrum, on both sides of the ratio: channel power against
  the median bin times the number of bins.

## Recording and playback

- **Audio recording** writes the demodulated audio to a WAV file.
- **IQ recording** writes raw `complex64` (interleaved little-endian float32 I/Q, no
  header; about 6 MB/s at 768 kS/s). A JSON sidecar (`<file>.json`) records the sample
  rate, centre frequency, radio driver, start time (UTC), samples written and samples
  lost.

Both run on their own threads with bounded queues. A slow disk drops samples and counts
them; it never stalls the audio or the window. IQ recording stops itself at 2 GiB rather
than filling the disk, and a real retune ends it, because the sidecar records only one
centre frequency. Files go to `~/Documents/RGC_SDR`, named by time and
frequency.

**Playback** (`device/playback.py`) is a source like any radio. A thread writes the file
into the same kind of ring, paced to the recorded rate, so the display, audio, decoders
and recorders cannot tell it from a live radio. Tuning moves a virtual centre within the
recorded span, using the same `_Nco`. This is not a simulated device: it only replays
what a real radio captured.

## The IC-705: no IQ at all

The IC-705 is a transceiver: it demodulates by itself and sends no IQ. Over USB
(`device/icom.py`) or WiFi (`device/icom_net.py`, Icom's network protocol), the app
receives its **spectrum scope**: 475 one-byte points a line, about 4.3 lines a second.
The window draws those lines in place of its own FFT. The source reports the scope's
span as its "sample rate" and the scope's centre as its centre, so tuning, memories and
geometry work unchanged. Audio comes from the radio itself, over USB audio or WiFi.

Classify, the map, zoom and recording need IQ, so they are hidden while the IC-705 is
in use.

## Radios on a network radio server

A radio plugged into another machine, such as a Raspberry Pi at home, is used over
Tailscale through RGC_SDR's own server (`netserver.py`, PLANNING.md section 7q). Raw IQ
is too much for the link: Tailscale carried about 30 Mbit/s here, and an RTL-SDR at
2.048 MS/s needs more. So the Pi does the reducing, and sends two streams:

```
radio on the Pi --SoapyIQSource--> ring
   |-- spectrum thread: Welch FFT of the whole span (or span/zoom about the tuned
   |   frequency), quantised to one byte a bin --> spectrum lines, 25 a second, < 1 Mbit/s
   '-- IQ thread: mix the tuned frequency to 0 Hz (_Nco), decimate by 2^k to 200-400 kS/s
       --> block-scaled int16 IQ, ~8 Mbit/s
```

On the Mac, `RemoteIQSource` puts the IQ window into a local ring. To everything else in
the app it is an ordinary source whose `sample_rate` is the window's rate. The window
draws the server's lines (`take_spectrum_lines`) instead of computing a spectrum, and the
**Rate** list sets the radio's own rate (`span_rate`).

- **Tuning inside the span** moves only the window, so the waterfall stays where it is.
- **Tuning within 5% of the span's edge** retunes the radio, and the waterfall starts
  again.
- **No waiting.** Each flushing change carries a generation number, and IQ made before it
  is dropped on arrival, so tuning never waits for a round trip.
- **Receive only.**
- **Wire format.** IQ uses block floating point: each block of int16 pairs is scaled to
  that block's own peak, which keeps a quiet band's noise floor intact.

## More than one radio

The radio list shows every supported radio. Ones detected now are highlighted yellow,
and radios on network servers appear as "RTL-SDR on radiopi". Tuning to a frequency the
current radio cannot reach hands over to one that can: the radio chosen from the list
first, wherever it reaches, then any other detected one in list order. Each radio keeps
its own rate, gains, colour range and frequency correction.

At start the last radio used is reopened. If it cannot be, a chooser lists the radios
that are there instead.
