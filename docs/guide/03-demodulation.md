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
IQ --> [noise --> mixer --> decimate --> channel filter --> detector --> audio shaping --> [noise reduction, --> squelch --> AGC --> volume
        blanker]   (offset)  (2^k, to the    (FIR: width and    (per mode)    (de-emphasis,         notches]
                              mode's IF)      IF shift)                       audio decimation,
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
| SAM | 48 kHz | 3, 4.5, 6, **9**, 12, 16 kHz | coherent, after tracking the carrier |
| NBFM | 48 kHz | 6, 8, **12.5**, 16, 25 kHz | FM discriminator, ±2.5 kHz |
| WBFM | 300 kHz | 150, 180, **200**, 250 kHz | FM discriminator, ±75 kHz |
| USB, LSB | 48 kHz | 1.8, 2.1, 2.4, **2.7**, 3.0, 3.6 kHz | sideband filter |
| CW | 48 kHz | 100, 250, **500**, 800, 1500 Hz | band-pass at the pitch |
| P25 | 48 kHz | **12.5** kHz | discriminator, then the P25 voice decoder |
| DMR | 48 kHz | **12.5** kHz | discriminator, then the DMR voice decoder |
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

## DMR (voice)

DMR voice is the same four-level FSK, so the chain is NBFM's up to the discriminator,
whose output goes to `DmrVoice` (`dsp/dmrvoice.py`):

1. **Finding the bursts.** A call on one timeslot is a run of superframes: six bursts,
   A to F, one every 60 ms, with the other slot's bursts in between. Only burst A
   carries the voice sync; B to F carry embedded signalling (EMB) in its place. So the
   sync finder hands over a whole superframe from each voice sync, with its timing and
   scale fixed by that sync, and B to F are read at their places. A place holding a data
   sync instead is the call's terminator, and ends it.
2. **Three AMBE+2 frames a burst**, 72 bits each: the 108 bits before the middle, the
   108 after, and the second frame straddling the middle.
3. **Deinterleave and decode.** Each frame goes into mbelib's 4×24 frame by DSD's DMR
   interleave tables, and through mbelib's AMBE 3600×2450 decoder (the same library as
   P25's IMBE). Out comes 20 ms of 8 kHz speech a frame.
4. **Which slot.** The DSP row's slot choice: **Slot 1**, **Slot 2**, or **Both**, where
   the first slot to speak keeps the audio until it stops.
5. **Encryption.** A call whose EMB privacy indicator is set is muted.
6. Interpolated ×6 to 48 kHz, levelled and paced, as P25 is.

Choosing the **DMR** decoder switches to this mode, so the Decode panel's talkgroup and
source show beside the audio.

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

## Receiver refinements

The DSP row (P10, PLANNING.md section 7r). The noise blanker and noise reduction apply
to AM, SAM, NBFM, SSB and CW.

### Passband and IF shift

A mode's passband is `passband_for(mode, width, shift, sideband)`, in hertz from the
listening frequency: USB 0 to W, LSB −W to 0, the rest −W/2 to W/2, each moved by the
shift. FM keeps it centred, since a shifted discriminator would hear only one side of
the deviation. The same function shades the spectrum and builds the channel filter, so
the two cannot disagree. Dragging the shaded edges runs it backwards
(`width_and_shift`). A shifted filter is a complex band-pass, a low-pass moved up or
down, as SSB's always was.

### Noise blanker

`NoiseBlanker` works on the IQ at the radio's rate, before anything else, while an
impulse is still short:

- A sample counts as an impulse if it is over *k* times the block's median magnitude.
  The median comes from a subsample, so impulses cannot move it.
- Each impulse is zeroed together with 20 µs either side, since the radio's own filters
  have already spread it a little. The mask is widened by convolution, with no loop.
- *k* runs from 15.5 at level 1 to 2 at level 10.

### Noise reduction and notches

`AudioCleanup` works in short FFT frames (about 10 ms, overlapped by half, square-root
Hann windows both ways, which reassemble the audio exactly when nothing is cut). It adds
one frame of delay.

- **Noise estimate.** Per bin: the minimum of a smoothed power, falling at once and
  rising about 1 dB a second. Then it is doubled, to correct the bias of a minimum (the
  quietest frame of steady noise lies well below its mean), and capped at twice its
  neighbours' median. Without that cap, a steady CW tone's own bin crept up until the
  tone counted as noise and was removed; a test pins that it is kept.
- **Gain.** Each bin is turned down to `max(floor, 1 − a·noise/power)`, smoothed from
  frame to frame against "musical noise". *a* runs from 0.75 to 3 and the floor from
  −8 to −26 dB as the level goes from 1 to 10.
- **Automatic notch.** Removes bins whose power, averaged over about a second, stands
  13 dB over their neighbours' median: a steady carrier. Speech moves, so it is not
  notched. It looks only between 150 Hz and 4 kHz.
- **Manual notches** are radio frequencies, mapped to audio by the mode: USB as is, LSB
  mirrored, CW plus the pitch, AM and FM by distance. Each removes ±40 Hz.

### Synchronous AM (SAM)

`SyncAmDetector`:

1. **Find the carrier.** On the first block, an FFT finds it within ±1 kHz.
2. **Follow it.** A frequency-locked mixer is updated once a block from the carrier's
   own rotation.
3. **Take its phase** from a narrow (30 Hz) one-pole low-pass of the derotated signal,
   whose state SciPy carries between blocks, so there is no per-sample loop.
4. **Demodulate coherently.** With the carrier's phase removed, the sidebands are real:
   **Both** is the real part, and **Upper** or **Lower** is SSB's sideband filter applied
   to them.

Levelled by the carrier, as AM is. Through selective fading this stays clean where an
envelope detector distorts, and choosing one sideband steps away from interference on
the other. If the carrier falls below 15% of the signal for 20 blocks, it is searched
for again.

Measured on air (an airband AM carrier received by an RTL-SDR, 2026-10-08), SAM:

- locked within a hertz
- followed a 400 Hz mistuning
- gave audio that tracks the envelope detector's (correlation 0.8)

### NBFM noise squelch

An FM radio's squelch listens to the noise above the voice, which falls as a signal
"quiets" the receiver. Here:

- The discriminator is band-passed to 8–16 kHz.
- Its power is compared with what a dead channel gives. That reference is measured once,
  by running the chain's own channel filter and discriminator on a fixed sample of
  Gaussian noise. A discriminator's output on noise does not depend on the noise's level,
  so the reference holds whatever the radio's gain.
- The squelch opens when the noise is the chosen **quieting** below it (default 10 dB),
  with 2 dB of hysteresis.

## AGC

**AGC** in the DSP row chooses **Fast** (holds 0.1 s, recovers 10% a block), **Medium**
(below), **Slow** (holds 1.5 s, 0.5% a block) or **Off**, a fixed gain for AM, SAM, SSB
and CW.

Every mode except AM and SAM uses the **audio AGC** (`AudioAgc`). It aims at an RMS of
0.15:

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
