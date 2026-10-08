# 3. Demodulation

How each mode turns IQ into audio. The code is `dsp/demod.py` (`DemodChain`), with the
stereo decoder in `dsp/fmstereo.py`, RDS in `dsp/rds.py`, tone squelch in `dsp/tones.py`,
the CW decoder in `dsp/morse.py`, and the audio plumbing in `audio.py`.

## Where the audio runs

```
IQ ring --sequential reader--> audio worker thread --> FIFO --> PortAudio callback --> speaker
                                (DemodChain)           (~0.1 s)  (copies only)
```

Three threads meet here. The device reader fills the ring. A worker pulls IQ gaplessly,
demodulates it and keeps a FIFO about half full. PortAudio's callback only copies from
the FIFO, because a callback that allocates or blocks is an audible dropout. The FIFO
holds 5 blocks of 1024 frames. P25 and DAB deliver audio in bursts, so they get 20
blocks (about 0.4 s).

Every stage is **stateful and continuous**. The mixer carries its phase, every filter
carries its history from block to block, and the FM detector keeps its last sample. A
display can restart its filters every frame, but restarting audio filters at a block
boundary is a click at the block rate.

## The common chain

```
IQ --> mixer --> decimate --> channel filter --> detector --> audio shaping --> squelch --> AGC --> volume
       (offset)  (2^k, to the     (FIR)           (per mode)    (de-emphasis,
                  mode's IF)                                      audio decimation,
                                                                  audio low-pass)
```

1. **Mixer.** Shifts the listening frequency (centre + Offset) to 0 Hz. Its phase
   accumulator runs across blocks and is wrapped, so it neither clicks nor loses
   precision over hours. For CW the beat-note pitch is folded in here too.
2. **Decimation** to the mode's IF rate: the largest power-of-two factor that keeps the
   rate at or above the mode's target (48 kHz for narrow modes, 300 kHz for broadcast
   FM). Each factor of two is a 63-tap half-band stage (see
   [The IQ stream](01-iq-stream.md#from-iq-to-the-spectrum-and-waterfall)). The rates
   are therefore the radio's rate divided by powers of two: 4 MS/s gives a 62.5 kHz IF.
   CoreAudio resamples odd output rates itself.
3. **Channel filter.** A windowed-sinc FIR, 31 to 511 taps depending on how sharp it must
   be, keeping ±bandwidth/2. The channel power after it is the S-meter's level and what
   the squelch compares.
4. **Detector**, per mode, below.
5. **Squelch.** Below the threshold, the block becomes silence (counted as "sq" in the
   status line).
6. **AGC and volume.** Output is clipped to ±1.

### Channel widths

| Mode | IF rate (at least) | Width choices (default **bold**) | Detector |
|---|---|---|---|
| AM | 48 kHz | 3, 4.5, 6, **9**, 12, 16 kHz | envelope |
| NBFM | 48 kHz | 6, 8, **12.5**, 16, 25 kHz | FM discriminator, ±2.5 kHz |
| WBFM | 300 kHz | 150, 180, **200**, 250 kHz | FM discriminator, ±75 kHz |
| USB, LSB | 48 kHz | 1.8, 2.1, 2.4, **2.7**, 3.0, 3.6 kHz | sideband filter |
| CW | 48 kHz | 100, 250, **500**, 800, 1500 Hz | band-pass at the pitch |
| P25 | 48 kHz | **12.5** kHz | discriminator, then the P25 voice decoder |
| DAB | 2.048 MS/s | **1.536** MHz | the DAB receiver |

## AM

`|x|`, the envelope of the channel-filtered signal, is the audio plus the carrier. A slow
DC blocker removes the carrier. Its estimate is a running mean updated per block (5% of
the way each block), so it needs no per-sample loop.

AM is **levelled by its carrier**, not by an audio AGC. The audio is multiplied by
0.5 ÷ carrier, which makes it the modulation depth: a 100% modulated peak reaches 0.5.
The carrier is steady whatever the programme does, so a pause cannot wind the gain up and
make the next word start loud. Measured on air, an audio AGC would have needed a gain of
about 10⁵ for a −103 dBFS station. The carrier gain is capped at 10⁷, so silence cannot
divide by nothing.

There is no audio low-pass: the channel filter's width sets the audio bandwidth. 9 kHz,
the ITU region 3 channel spacing, is the default.

## NBFM

The **quadrature discriminator** measures how far the phase turns from one sample to
the next:

```
audio = angle(x[n] · conj(x[n−1])) · IF rate / (2π · 2.5 kHz)
```

That is the instantaneous frequency, scaled so ±2.5 kHz deviation is ±1. The previous
block's last sample is kept, so the first sample of each block is right. A 3.4 kHz
low-pass leaves the speech band, and the audio AGC levels it.

### Tone squelch (CTCSS and DCS)

With **TSQL** or **DCS** chosen, the audio is passed only while the right tone or code is
received (`ToneSquelch`):

- The discriminator audio is decimated to about 1.5 kHz and low-passed below 300 Hz.
  Every block, the last half-second is examined.
- **CTCSS.** The strongest component between 55 and 270 Hz must be within 1 Hz of the
  tone, measured from a zero-padded 16384-point spectrum, and hold at least 30% of the
  sub-audible energy. That tells apart neighbouring tones only 2.3 Hz apart.
- **DCS.** Normalised correlation against the code's own repeating waveform, at every
  phase, must exceed 0.6 with the right sign. An inverted code is a different code. The
  waveform is a 23-bit Golay(23,12) codeword sent continuously at 134.4 bit/s. Codes
  were checked against published DCS tables: 023 encodes to 0x763813, and the inverse
  pairs come out as the tables list.
- It opens on one detection and closes after two misses in a row, so one bad block does
  not chop the audio.
- While tone squelch is on, the audio is high-passed at 300 Hz, as a radio does, so the
  tone itself is not heard.

## WBFM (broadcast FM)

The discriminator, at ±75 kHz deviation and an IF of at least 300 kHz, gives the
**multiplex** (MPX):

| Frequency | Content |
|---|---|
| 0–15 kHz | L+R: the mono signal |
| 19 kHz | the stereo pilot |
| 23–53 kHz | L−R, double sideband with a suppressed 38 kHz carrier |
| 57 kHz | RDS |

The MPX is decimated to at least 150 kHz, so 57 kHz survives, and goes to the stereo
decoder (`fmstereo.py`):

- **Pilot detection by phase coherence, not amplitude.** A complex band-pass isolates the
  19 kHz pilot as a phasor. It is counter-rotated by the expected 19 kHz; a real pilot
  then holds still and scores 1.0, while noise wanders and scores about 0.09. Stereo is
  used above 0.5. An amplitude test was tried first and failed on air: the discriminator
  turns noise into plenty of 19 kHz energy, so noise was declared stereo, and the
  receiver demodulated noise into the audio.
- **The 38 kHz carrier without a loop.** Squaring the unit pilot phasor doubles both its
  frequency and its phase, which gives the subcarrier exactly: `−Im(unit²)`. A
  phase-locked loop would need a per-sample Python loop, far too slow at these rates.
- **The matrix.** L+R is low-passed to 15 kHz. L−R is `2 · MPX · subcarrier`, low-passed
  the same way. Then left = (sum + difference)/2 and right = (sum − difference)/2.
  Without a pilot both channels are the sum. Every path uses the same 255-tap FIR length,
  so every path has the same delay and nothing needs aligning by hand.
- **De-emphasis** of **50 µs**, the Australian and European standard (the Americas use
  75 µs), is applied to left and right separately, *after* the matrix. Applied to the
  whole MPX, it would flatten the subcarrier and RDS. It is a first-order IIR (the
  bilinear transform of 1/(1+sτ)), with its state carried by SciPy between blocks.
- Then decimation to about 44 kHz, a 15 kHz low-pass, and **one** AGC gain for both
  channels, so the stereo image does not wander. Output is always two channels: mono
  stations are duplicated, so the stream never changes shape mid-way.

Untick **Stereo** for mono, which is quieter on a weak station, since L−R carries more
noise.

### RDS

- **Carrier.** The 57 kHz subcarrier is regenerated by **cubing** the pilot phasor. Not
  every encoder locks RDS exactly to the pilot (Triple M Melbourne measured about 7 Hz
  off), so a small Costas-style loop tracks the remaining slow rotation. Squaring BPSK
  removes its data, which is what the loop locks on. The loop runs on 64-sample chunks.
- **Bits.** RDS is BPSK at 1187.5 bit/s, biphase-coded and differentially encoded. After
  mixing, it is decimated to about 16 kHz and low-passed at 2.4 kHz. Each bit is read by
  correlating against a half-positive, half-negative template, with clock recovery, then
  decoded differentially.
- **Blocks.** 26 bits each: 16 data and a 10-bit checkword (generator 0x5B9) with an
  offset word that names the block (A, B, C, C′, D). Synchronisation is found by
  syndrome, then followed block by block, and lost after 12 bad blocks.
- **Groups.** Group 0 gives the 8-character **station name** (PS) two characters at a
  time, plus traffic announcement and music/speech. Group 2 gives the 64-character
  **radio text**; its A/B flag flipping means a new message. Every group carries the
  **PI** code, the **programme type** (PTY, from the European table, not the North
  American RBDS one) and the traffic programme flag.

The station name, programme type and radio text appear on the info line.

## USB and LSB

With the carrier at 0 Hz, the upper sideband occupies 0 to +B and the lower −B to 0. A
real low-pass is symmetric and cannot separate them. So the channel filter is a
**complex band-pass** centred on +B/2 (USB) or −B/2 (LSB): a low-pass shifted up or
down, which passes one side only. The real part of the result is the audio; no detector
is needed.

Levelled by the audio AGC (below). SSB has no carrier, so the AGC's **hang** matters
most here: it holds the gain through the pauses between words.

## CW

A keyed carrier tuned exactly would come out at 0 Hz, which is silent. So the mixer
shifts by the offset **minus the pitch**, putting the carrier at the pitch (500 Hz by
default), and the filter is a band-pass **centred on the pitch**: 100 to 1500 Hz wide,
500 by default. This is a conventional receiver's BFO, folded into the mixer. The Offset
still means "where I am listening". The passband drawn on the spectrum is centred on the
tuned frequency, because that is where a carrier must be to sound at the pitch.

Changing the pitch moves the mixer and the filter together; changing only one would put
the tone outside its own passband.

### CW decoder

Morse lives in the **envelope**, not in the tone, so the decoder (`morse.py`) works on
`|channel|` rather than the audio. The envelope is decimated by 32.

- **Threshold.** Learned from the envelope's own low and high percentiles over the last
  1.5 s, so the same code works on weak and loud signals. Nothing is decoded until the
  envelope swings to at least twice the noise level.
- **Speed.** Marks cluster around one and three units, which tracks the dot length:
  20 to 200 ms, starting at 60 ms (20 WPM). Operators send at 5 to 50 WPM and change
  speed mid-over, so this keeps adapting.
- Runs shorter than 12 ms are ignored as glitches.
- Elements, letter gaps and word gaps are told apart by their length in units. The text
  and speed appear on the info line.

### Zero beat

Hold **Zero beat** to put a nearby carrier exactly on the pitch (`zerobeat.py`). The
display is far too coarse for this: 4096 bins across 768 kHz is 187 Hz a bin. So a
32768-point transform (23 Hz bins, covering 43 ms, short enough not to straddle the gaps
between dots) finds the carrier within ±500 Hz. A parabola through the peak bin and its
neighbours places it to a couple of hertz. Below 15 dB S/N nothing is moved.

## P25 (voice)

P25 Phase 1 voice is four-level FSK (C4FM) at 4800 symbols/s in a 12.5 kHz channel. The
chain is NBFM's up to the discriminator. Its output goes to `P25Voice`
(`dsp/p25voice.py`):

1. **Symbols** are recovered around each frame sync (see
   [Decoders: four-level FSK](04-decoders.md#four-level-fsk-p25-and-dmr)).
2. A voice call is **LDU1 and LDU2** frames, 180 ms each, alternating. Each holds nine
   **IMBE** voice frames of 144 bits.
3. LDU1's link control gives the **talkgroup** and the **talking radio's ID**, protected
   by Reed-Solomon (24,12,13) over GF(64). LDU2 carries the encryption sync.
4. Each voice frame is deinterleaved into its 8 × 23 array and handed to **mbelib**
   (Homebrew's, loaded with ctypes), which corrects errors and synthesises **8 kHz**
   speech. The app cannot decode IMBE itself, so this codec comes from outside.
5. **Encrypted** calls are muted: without a key they would be noise.
6. The 8 kHz audio is interpolated ×6 to 48 kHz, levelled, and **paced**.

Voice arrives in 180 ms bursts and not at all between calls. The `AudioPacer` reads IQ
steadily in real time, queues 0.3 s before playing, and fills silence only when the queue
runs dry. Without it, a burst overfilled the FIFO, the worker stopped reading IQ, the
ring overran, and frames were lost.

Who is talking is shown in the Decode panel, under the privacy rule: P25 and DMR IDs and
content stay on screen and are never written anywhere.

## DAB (DAB+ radio)

DAB mode takes the whole ensemble, 1.536 MHz wide, from IQ at 2.048 MS/s. The receiver
halves 4.096 MS/s itself. Choosing DAB sets one of those rates when the radio has it;
4.096 also keeps the next ensemble from folding in on a radio whose own filter is too
wide. There is no mixer-and-filter chain: the ensemble goes straight to `DabReceiver`
(see [Decoders: DAB](04-decoders.md#dab-and-dab)).

The first DAB+ station the ensemble names is selected so there is something to hear.
**DAB station** chooses another. FAAD2 decodes its HE-AAC at 48 kHz, or at 32 kHz, which
is resampled by 3/2. Decoding runs about half a second behind, in 120 ms superframes, so
the audio is paced like P25's. Retuning to another ensemble starts a fresh receiver, so
the station list does not mix two ensembles.

Only DAB+ (HE-AAC) stations on equal-error-protection (EEP) sub-channels play.
Original DAB (MP2) stations, and sub-channels with unequal protection (UEP, used by
MP2), are listed but not decoded.

## AGC

Every mode except AM uses the **audio AGC** (`AudioAgc`). It aims at an RMS of 0.15:

- **Attack.** When the signal gets louder it backs off at once, ramped across the block
  so it does not click.
- **Hang.** Through a pause it *holds* its gain for 0.6 s.
- **Decay.** Only then does it recover, slowly (2% of the way a block).
- **Limits.** Its gain is capped at 20000, so a dead channel's noise is not raised to
  full scale. On its first block it jumps straight to the right gain; easing in from 1.0
  took ten seconds to become audible on a weak signal.

Without an AGC, output is proportional to signal strength, and a weak station is
inaudible at any volume. The status line shows the current gain ("agc ×…").

## Squelch

The channel power after the channel filter, in dBFS, is compared with the threshold each
block. **A** sets the threshold 3 dB above the level now (press it on noise); **R**
resets it to −100 dBFS. AM, NBFM and WBFM can be squelched. SSB and CW cannot: they have
no steady level to squelch on.

## The IC-705

The IC-705 demodulates by itself. Its audio reaches the Mac over the USB audio codec,
or over WiFi as 16-bit 48 kHz PCM in 20 ms packets. The Mode list then sets the radio's
own mode. Muting while the radio's squelch is closed is done on the Mac, because the
radio sends its audio unsquelched.

## Transmit

Pressing **TX** (or the space bar) on a radio that can transmit runs the receive chain
in reverse (`dsp/modulate.py`):

```
microphone 48 kHz --> mic gain --> speech band-pass --> limiter --> [pre-emphasis] --> [CTCSS/DCS]
   --> modulate at a baseband rate --> integer interpolation --> radio IQ rate --> SoapySDR TX stream
```

| Mode | Audio band | Modulation |
|---|---|---|
| AM | 300–3000 Hz | `(1 + 0.8·audio) / 1.8`: 80% depth, never over full scale |
| NBFM | 300–3000 Hz | phase = running sum of audio · 2π · 2.5 kHz / rate; CTCSS or DCS at 15% of the deviation |
| WBFM | 30 Hz–15 kHz | ±75 kHz at a 240 kHz baseband, 50 µs pre-emphasis (mono) |
| USB, LSB | 300–2700 Hz | the receiver's own sideband filter applied to the audio, ×2 |
| CW | — | refused: CW wants a keyer, not a microphone |

- **Mic gain** defaults to 25 dB, chosen on air: the MacBook Air microphone peaks far
  below full scale.
- **The limiter** holds peaks at full scale. It backs off at once and recovers over
  0.3 s, ramped across the block so it never steps.
- **IQ rate.** Must be a whole multiple of the mode's baseband rate, so interpolation is
  by an integer. 1.92 MS/s (40 × 48 kHz, 8 × 240 kHz) suits every mode.
- **Checked by the receiver.** Every modulator is tested by demodulating its own output
  with the receive chain.

Operational safeguards, at the operator's choice and with no band, mode or power
interlocks:

- Transmit gains start at minimum.
- Tuning is locked while keyed.
- Transmission stops after **3 minutes**.
- On a half-duplex radio (the HackRF), receive stops while keyed and restarts after.

The first key-up of any new transmit path is the operator's to do. On the IC-705, TX keys
the radio itself, with the Mac's microphone sent as its modulation (over USB, or over
WiFi with the "USB bridge" login option).
