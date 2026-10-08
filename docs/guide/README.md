# RGC_SDR guide

How the receiver works inside, and how to use it. Written for anyone using RGC_SDR, or
reading its code to learn how a software-defined radio works.

| Document | What it covers |
|---|---|
| [1. The IQ stream](01-iq-stream.md) | Where the samples come from, how they are buffered, shifted, decimated and turned into the spectrum and waterfall. Radios on a network server. |
| [2. The interface](02-interface.md) | Every tab, panel, field and button, and what it does. |
| [3. Demodulation](03-demodulation.md) | How each mode becomes audio: AM, NBFM, WBFM (stereo and RDS), USB, LSB, CW, P25 voice and DAB+. Squelch, tone squelch, AGC. Transmit. |
| [4. Decoders](04-decoders.md) | How each data protocol is decoded: POCSAG, APRS, AIS, ACARS, ADS-B, P25, DMR, DAB, plus Classify and calibration. |
| [5. Features to add](05-features-to-add.md) | What higher-end SDR applications offer that RGC_SDR does not, and a plan for adding it. |

[PLANNING.md](../../PLANNING.md) is still the source of truth for decisions, measurements
and the roadmap. These documents describe the code as it stands. When the two disagree,
the code is right and this guide needs updating.

## The whole receiver on one page

```
radio (SoapySDR driver)                  IC-705 (CI-V)             network radio server
        |                                      |                          |
  reader thread: readStream, NCO              scope lines             spectrum lines + IQ window
  undoes the LO offset, ppm in tuning          (no IQ)                        |
        |                                      |                          |
        v                                      |                          v
   IQ ring buffer  <---------------------------+------------------  local ring (IQ window)
   (complex64, ~1 s)                                                     |
        |                                                                |
  +-----+--------------------+----------------------+----------------+   |
  |                          |                      |                |   |
display (lossy)        audio worker            decode worker      recorders
read_latest()          sequential reader       sequential reader  sequential reader
decimate (zoom)        DemodChain              DemodChain/IQ      WAV / complex64 IQ
Welch FFT, dBFS        -> FIFO -> PortAudio    -> decoder         + JSON sidecar
spectrum, waterfall,                           -> Decode panel,
S-meter, scanner                                  map, memories
```

The display only ever needs the newest samples, so it reads lossily. Everything that
must not skip — audio, decoders, recording — has its own gapless reader. Each of those
consumers runs on its own thread, so a slow decoder never stalls the waterfall, and a
busy window never makes the audio drop out.

The code is in layers (PLANNING.md section 5):

| Layer | Directory | Rule |
|---|---|---|
| Hardware | `src/rgc_sdr/device/` | No Qt, no DSP imports |
| Signal processing | `src/rgc_sdr/dsp/` | Pure NumPy (plus a little SciPy); no Qt, no device access |
| Application | `src/rgc_sdr/*.py` | Glue: audio, decoding, scanning, recording, settings |
| Interface | `src/rgc_sdr/ui/` | PyQt6 and pyqtgraph |

Because `dsp/` never touches Qt or a radio, every demodulator and decoder is tested
headless with synthetic signals. Each one has also been checked on air in Melbourne;
PLANNING.md records where and when.
