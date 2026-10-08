# 4. Decoders

How each data protocol is recovered from the IQ stream. The decoders are in `dsp/`; the
worker that runs them is `decoding.py`. Classify and calibration, which build on them,
are at the end.

## How a decoder is run

Choosing a decoder in the **Decode** panel starts a `DecodeWorker` on its own thread,
with its own gapless reader on the IQ ring. It never shares the audio chain:

- It decodes whatever the audio mode, and through mute and squelch.
- It reads what it needs **before** any audio shaping: the raw FM discriminator or AM
  envelope at the IF rate (`DemodChain.last_detected`). The audio low-pass, AGC and
  squelch would distort or silence exactly what a data slicer needs.
- Some decoders read **raw IQ** instead, shifted to their channel (ADS-B, DAB).
- Some listen on **fixed channels** (AIS's two, ADS-B's 1090 MHz) wherever the radio is
  tuned, as long as the channel is in view (within 45% of the span from the centre).
  The others follow the listening frequency, so retuning moves them.

| Decoder | Reads | Channel width | Where |
|---|---|---|---|
| POCSAG | NBFM discriminator | 16 kHz | the listening frequency |
| APRS | NBFM discriminator | 12.5 kHz | the listening frequency (145.175 MHz in VK) |
| AIS | NBFM discriminator | 20 kHz | 161.975 and 162.025 MHz, both at once |
| ACARS | AM envelope | 10 kHz | the listening frequency (131.550 MHz here) |
| ADS-B | raw IQ magnitude | 1 MHz | 1090 MHz, at 2 MS/s or more |
| P25 | NBFM discriminator | 12.5 kHz | the listening frequency |
| DMR | NBFM discriminator | 12.5 kHz | the listening frequency |
| DAB | raw IQ | 1.536 MHz | the ensemble's centre, at 2.048 or 4.096 MS/s |

Messages are queued for the window (up to 1000), shown in the panel, and positions go to
the map. **Privacy.** Pager, P25 and DMR traffic can carry names, addresses and medical
details. Their text is shown on screen only and can be hidden with **Show text**.
Nothing decoded is written to disk or logged.

## Shared building blocks

### Two-level bit slicing (`bitsync.py`)

POCSAG, AIS and ACARS slice two-level data from a demodulated baseband without a
per-sample loop:

- Each sign change is located to a fraction of a sample by interpolation, and given a
  bit index `round((t − phase) / samples_per_bit)`.
- The **bit phase** is a running mean of the transition times modulo one bit, kept as a
  smoothed phasor and unwrapped, so clock drift is followed without ever slipping a bit.
- Every bit between two transitions has the same value, so the bit stream is the levels
  repeated by the differences in bit index. A glitch shorter than half a bit gets the
  same index at both ends and simply vanishes.
- A slow DC estimate removes any offset from mistuning.

### HDLC framing (`aprs.py`, shared with AIS)

NRZI decoding (a change is 0, no change is 1), `0x7E` flags, removal of the stuffed 0
after five 1s, least-significant-bit-first bytes, and the **CRC-16/X.25** frame check.
The check is tested against the standard value 0x906E for "123456789". A frame that
fails its CRC is discarded, never guessed at.

### Four-level FSK (P25 and DMR)

P25 Phase 1 and DMR both send **dibits as four frequency deviations** at 4800 symbols/s:
`01 → +3, 00 → +1, 10 → −1, 11 → −3`. The unit is 600 Hz for P25's C4FM and 648 Hz for
DMR. Both begin their frames with a known 24-symbol sync. So `fsk4.py` builds recovery
around the sync, not a free-running clock:

1. The discriminator output is **correlated with the sync pattern**. Each peak marks a
   frame start to a fraction of a sample.
2. The 24 known symbols fix **that frame's own scale and offset**, so mistuning and
   deviation do not matter.
3. The symbols after it are read at the nominal period from there. Transmitter clocks
   are good to a few ppm, far inside half a symbol over a frame.

## POCSAG (pagers)

ITU-R M.584, two-level FSK at **512, 1200 or 2400 baud**. All three are sliced in
parallel, since a channel may carry any of them, and either polarity is accepted.

- **Batches.** A batch is the sync codeword `0x7CD215D8` followed by 8 frames of two
  32-bit codewords. The sync is accepted with up to 2 bit errors.
- **Codewords** are BCH(31,21) plus an even parity bit. Up to **two bit errors** are
  corrected from a syndrome table.
- **Address codewords** give the 21-bit pager address (18 bits in the codeword, the
  low 3 from which frame of the batch it is in) and a 2-bit function code. **Message codewords** carry 20 bits each. Idle codewords
  (`0x7A89C197`) end a message.
- **Numeric or text.** This is judged by the **content**, not the function code, which is
  only conventionally numeric. Measured on a Melbourne network, its function-0 pages were
  text. A reading counts as numeric only if at least 95% of it is digits, spaces and
  dashes. Otherwise it is read as 7-bit characters, least significant bit first, packed
  across codewords, up to the end-of-text code.

No FLEX decoder exists: a survey of local paging channels found none (PLANNING.md P8b).

## APRS

AX.25 packets on **1200-baud Bell 202 AFSK**: mark 1200 Hz, space 2200 Hz.

1. The discriminator audio is **mixed down by the midpoint, 1700 Hz**, low-passed to
   1.2 kHz, and **FM-detected a second time**. The sign then says mark or space,
   whatever the transmitter's pre-emphasis did to the two tones' levels. A
   tone-energy detector would be fooled by that.
2. The bits are sliced, then HDLC framing and the CRC are applied (above).
3. The **AX.25** address field gives `SOURCE>DEST,PATH`; a `*` marks a digipeater that
   has repeated it.
4. **Positions** are read in all three APRS formats:
   - **Plain**: `!3749.10S/14458.00E`.
   - **Compressed**: base 91.
   - **Mic-E**, which most Kenwood and Yaesu radios send. The latitude, N/S, E/W and a
     longitude offset hide in the destination callsign.
5. **Objects** (`;NAME     *…`) and **items** (`)NAME!…`) are placed on the map under
   their own names, with "sent by". **Course and speed** come from `ccc/sss` after a
   plain position, or from Mic-E's bytes.

## AIS (ships)

ITU-R M.1371: **GMSK at 9600 baud**, ±2.4 kHz deviation, on 161.975 and 162.025 MHz.
Both channels are decoded at once, wherever the radio is tuned, provided both are in
view.

- An FM discriminator recovers the bits. GMSK's filtering only softens the edges.
- The framing is **HDLC exactly as AX.25**, with the same CRC, so the APRS framer is
  shared. The one twist: AIS fields run most significant bit first, while HDLC sends each
  byte least significant bit first, so each byte's bits are reversed back.
- **Messages read:**
  - positions: types 1–3 (class A) and 18 (class B), with navigation status, speed,
    course and heading
  - base stations (4)
  - names and voyages (5, and 24 for class B)
  - aids to navigation (21)
- Every message is also given as the standard `!AIVDM` sentence, which chart plotters
  read.

## ACARS (aircraft)

ARINC 618: **2400-baud MSK on an AM carrier**, tones of 1200 and 2400 Hz.

- The AM envelope is high-passed at 600 Hz first. In a burst a fraction of a second
  long, the AM detector's slowly removed carrier would otherwise swamp the tones.
- **The bit rule, settled on air.** 2400 Hz means "same as the previous bit" and 1200 Hz
  means "the opposite", so the data is the running XOR of the 1200 Hz decisions. On a
  4-minute capture at 131.550 MHz, that reading found SYN SYN SOH in all 30 bursts. The
  obvious readings found none. Its polarity depends on where decoding starts, so both
  are tried.
- **Characters** are 7-bit ASCII with odd parity, least significant bit first.
- **A block** is SYN SYN SOH, mode, aircraft registration (7 characters),
  acknowledgement, label (2), block ID, STX, text, then ETX or ETB, and a check.
- **The block check is CRC-16/KERMIT.** It validated all 26 complete frames of that
  capture; X.25, XMODEM and CCITT-FALSE validated none.
- **Labels** are named where common (position report, link test, squitter, free text…).
  **Positions** in the text are read in two forms: decimal degrees (`S 37.894/E144.735`),
  and ground stations' degrees and minutes in squitters.

## ADS-B (aircraft)

Mode S extended squitter on **1090 MHz**, read from the **IQ magnitude** directly, with
no AM or FM demodulator, at a whole number of samples per microsecond (2 MS/s and up:
the Pluto or HackRF).

- **Pulse-position modulation at 1 Mbit/s.** An 8 µs preamble of four pulses (0, 1, 3.5
  and 4.5 µs) must stand at least 2× above the gaps. Then 56 or 112 bits follow, each a
  pulse in the first half of its microsecond for 1, or the second half for 0.
- **Parity.** Every message carries a 24-bit parity check (generator `0xFFF409`), so a
  noisy candidate is rejected, never guessed.
- **DF17/18 extended squitters give:**
  - **identity**: callsign, from the 6-bit character set
  - **airborne position**, by CPR: globally from an even/odd pair less than 10 s apart,
    then locally from the last fix. No home location is needed or stored.
  - **altitude**
  - **velocity**: speed, track and vertical rate

Reference: Junzi Sun, *The 1090 Megahertz Riddle*.

## P25 Phase 1

TIA-102. The voice side is described in [Demodulation](03-demodulation.md#p25-voice);
this is the data side (`p25.py`).

- **Frame sync and NID.** Each frame starts with a 48-bit sync, then a 64-bit **network
  ID**: the 12-bit **NAC** (network access code) and the 4-bit **DUID** (header, LDU1,
  LDU2, terminator, TSDU, PDU…). These are protected by **BCH(63,16)**, which corrects up
  to 11 bit errors. The generator, built here from its roots, is the published
  6331141367235453 (octal); every NID on four channels decoded clean on air. A status
  symbol woven in after every 35 dibits is dropped on reading.
- **Trunking control (TSDU).** A TSDU carries one to three **TSBKs** of 96 bits: opcode,
  manufacturer (Motorola, Harris, Tait…), 64 bits of arguments and a CRC-16, which is
  CRC-CCITT inverted. They travel in a **rate-1/2 trellis code** interleaved over 98
  dibits; the deinterleave table runs from received position to decoded position,
  settled on air. Decoded:
  - voice grants: group, unit-to-unit and their updates, giving talkgroup, radio and
    channel
  - the system's **channel plan**, so channel numbers become frequencies
  - network and site status, adjacent sites, affiliations, registrations, time and date
- **Packet data (PDU).** A header coded like a TSBK, then **confirmed data blocks** of
  16 bytes, at rate 3/4 (a tribit trellis). Each has a serial number and a **CRC-9**
  (x⁹+x⁶+x⁴+x³+1, inverted), and the whole packet has a **CRC-32**. Inside are SNDCP and
  IPv4/UDP: see [Packet data](#packet-data-p25-and-dmr).
- **Voice calls** show the talkgroup and the talking radio's ID from LDU1's link
  control, and whether the call is encrypted.

## DMR

ETSI TS 102 361 (`dmr.py`). DMR is **two-slot TDMA**: a repeater sends a 30 ms burst for
each slot in turn, each 132 symbols with a 24-symbol sync in the middle.

- **CACH.** Ahead of each burst, 12 symbols say which slot it belongs to, protected by
  Hamming(7,4). On air the slot bit alternated on all 997 bursts of a continuous
  channel.
- **Sync.** A data sync's negative is the matching voice sync, so one correlation finds
  both: positive for data and control, negative for voice.
- **Slot type.** 20 bits around the sync: the **colour code** and **data type**, coded as
  the extended Golay(24,12) (generator 0xC75) shortened to 8 data bits.
- **Payload.** 196 bits, **BPTC(196,96)**. The deinterleave takes position *i* from
  received position (*i* × 181) mod 196, then Hamming row and column parities are
  checked and corrected. Decoded:
  - **CSBK** (control): opcode, manufacturer, 64 bits, CRC-CCITT inverted and XORed with
    0xA5A5
  - **Voice LC header and terminator**: group or private call, talkgroup or destination,
    source. Protected by **Reed-Solomon (12,9)** over GF(256), masked with 0x969696
    (header) or 0x999999 (terminator).
  - **Data headers and blocks**: confirmed blocks with CRC-9, then a packet CRC-32, which
    is computed over the bytes swapped in pairs (found on air)
- Every one of those codes was settled on air in Melbourne, over four channels
  (PLANNING.md P8e).

No voice: DMR's AMBE+2 codec is not available to the app.

## Packet data (P25 and DMR)

`packetdata.py` reads what a data packet carries: IPv4/UDP to Motorola's well-known
services.

| UDP port | Service | Shown |
|---|---|---|
| 4001 | LRRP: location reports | the position, which goes on the map as a "radio" |
| 4005 | ARS: registrations | which radio registered |
| 4007 | TMS: text messages | the text |
| 4008 | telemetry | — |
| 4012 | over-the-air rekeying | — |

**LRRP** layouts follow open-source decoders, since Motorola publishes none. Latitude is
sign-magnitude and longitude two's complement, checked on air against DMR reports from
Victoria.

## DAB and DAB+

ETSI EN 300 401, **transmission mode I** (`dab.py`, `dabplus.py`). At 2.048 MS/s a 96 ms
frame is a **null symbol** (2656 samples), then **76 OFDM symbols** of 2552 samples
(a 504-sample guard plus 2048 useful). Each symbol has 1536 carriers 1 kHz apart.

### Synchronisation

1. **Frame.** The null symbol is the quietest 2656-sample stretch of a frame, found from
   a running sum of power. If it is not clearly quieter than average, there is no DAB
   there.
2. **Timing and fine frequency.** Each guard interval is a copy of the end of its symbol.
   Correlating guard against end over five symbols, at shifts of ±32 samples, gives the
   symbol timing at the peak. The correlation's phase gives the **fractional** frequency
   offset.
3. **Whole-carrier offset.** Found from where the 1536 occupied carriers hold the most
   power, summed over eight symbols across the frame, within ±20 carriers. A new value is
   accepted only when it repeats. Measured: one symbol's power put single frames 6 or 12
   carriers out while the true offset never moved.
4. **The whole offset is removed in time, not by moving FFT bins.** An offset also turns
   each carrier's phase from symbol to symbol, which would wreck the differential
   demodulation.

### Demodulation

- Every symbol is FFT'd, each FFT starting a little into the guard.
- The carriers are **differentially QPSK-demodulated** against the previous symbol, so
  the phase reference's table is never needed. Then they are put back in order (the
  frequency interleave).
- **Channel-state weighting.** Each carrier is weighted by how well its points sit on
  the QPSK constellation over the frame, using z⁴, which removes the data. Measured: the
  carriers next to the HackRF's LO leak erred 20–50%, always in the same positions. That
  cost alternate FIBs every frame, because the code corrects scattered errors, not ones
  that recur in one place.
- **Clipping** is measured. The HackRF at its default gains clipped 20–30% of samples on
  Melbourne's Band III and decoded little.

### The FIC: the ensemble's directory

Symbols 2–4 carry 9216 bits in four blocks:

1. Each block is a punctured rate-1/4 convolutional code (K = 7, generators 133, 171,
   145, 133 octal). Puncturing vectors are copied from table 13 of the standard; a
   generated rule matched only some.
2. A **Viterbi decoder** decodes them. It loops over trellis steps, not samples, with
   every block of a frame side by side.
3. Energy dispersal is removed, and each 256-bit **FIB** is checked by its own CRC-16.
4. FIG 0 gives the multiplex layout: services, sub-channels, protection, and whether
   each is DAB+. FIG 1 gives the **ensemble and service names**.

The Decode panel shows the ensemble and its stations, and the station list fills the DAB
station choice.

### DAB+ audio

A station's sub-channel, from the main service channel (MSC):

1. **Extract.** The MSC is 72 symbols, four 24 ms frames of 55296 bits. The station's
   sub-channel is its capacity units (64 bits each) in every frame.
2. **Time deinterleave.** Each bit is delayed by 0–15 frames according to its position
   mod 16 (table 21, a bit reversal).
3. **Decode.** EEP depuncturing (tables 18 and 20), the same Viterbi decoder, then
   energy dispersal removed.
4. **Superframes.** Five logical frames, with **Reed-Solomon (120,110)** over GF(256)
   across the rows of a virtual interleave (Berlekamp–Massey, Chien search, Forney),
   found by the header's **Fire code**.
5. **Access units.** 20 to 60 ms of audio each, each with a CRC, decoded by **FAAD2**
   (Homebrew, through ctypes) as HE-AAC v2 with 960-sample frames.

On air, Melbourne's 9B and 9C ensembles decode 100% of FIBs and 9A about 80%.

## The map's targets

`targets.py` folds decoded messages into one target per identity:

| Identity | From |
|---|---|
| MMSI | AIS |
| ICAO address | ADS-B |
| registration, or ground station | ACARS |
| callsign, or object name | APRS |
| NAC + radio | P25 |
| colour code + radio | DMR |

Each target keeps its latest position, what is known about it (name, speed, course,
heading, altitude, path, last packet), and a trail of up to 200 points. It expires after
its kind's time unheard (see [The interface: Map](02-interface.md#map)).

## Classify: "?"

`classify.py` names the signal at the listening frequency from 2.5 seconds of IQ:

1. **Protocols, by decoding.** Each decoder is run over the capture. One that finds
   frames passing **their own error checks** names the signal beyond doubt: a P25 NID, a
   DMR slot type, a POCSAG sync and batch, an AX.25, AIS or ACARS CRC, or a Mode S
   parity. Nothing decoded is shown.
2. **Modulation, by measurement**, when no decoder recognises it, from these features:
   - occupied bandwidth
   - how constant the envelope is
   - how much power sits in a carrier
   - which side of the tuned frequency the energy is on
   - for broadcast FM, the 19 kHz pilot and RDS
   That separates WBFM, NBFM, FSK data, AM, USB, LSB, CW and a plain carrier. These
   names carry a "?": judgements, not proof.

The label goes on the waterfall over the signal.

## Classify: the whole screen

`sweep.py` labels up to **twelve** signals across what the waterfall shows, wherever the
radio is tuned:

1. **One capture** of the whole span, capped at 12 million samples.
2. **Candidates** from its 50 ms max-hold spectrum. They must stand at least **20 dB**
   over the floor, because a max-hold sits about 15 dB above the mean noise, so less
   reports transmitters in pure static. They must be inside the visible span, away from
   its edges and the radio's DC spike, and at least 12.5 kHz apart. Up to 24 are taken,
   strongest first.
3. **One FFT channeliser.** The capture is transformed once, and each candidate becomes
   a narrow stream by taking its bins and transforming back: a sharp filter for the price
   of a small inverse FFT. Mixing and decimating 12 million samples per candidate took
   seconds each.
4. **The "?" classifier** runs on each stream until twelve labels are filled. Unknown,
   USB and LSB results are skipped as unreliable at this distance. A label's frequency
   snaps to its channel grid when the measured centre is within 1.5 kHz of it: 100 kHz
   for broadcast FM, 8.33 kHz in the airband, 6.25 kHz elsewhere. A centre measured
   from the spectrum carries the radio's error and the signal's own asymmetry (97.4948
   for 97.5 MHz on an uncorrected HackRF).
5. **Merging.** P25 channels with one NAC, and DMR channels with one colour code, become
   one label at the strongest, with markers in the same colour on the rest.

ADS-B pulses never stand out in an averaged spectrum, so when 1090 MHz is on screen it
is decoded from the capture itself. On a network radio the sweep covers only the IQ
window, not the whole span shown.

## Calibration

`calibrate.py` measures a radio's frequency error from a carrier whose frequency is
known. The defaults are **Essendon ATIS on 119.8 MHz** (an AM carrier) and **the
144.650 MHz CW beacon**.

- **Two captures at different tunings.** The strongest line within ±25 ppm of the known
  frequency is measured in each, to a fraction of a hertz. It must sit at the same radio
  frequency in both, within 100 Hz, and be 30 dB above the noise. The radio's own spurs
  move with the tuning (the HackRF has combs of them) and are rejected.
- **The error.** A reading *offset* hertz high at *f*, with *ppm_now* already applied,
  means an error of `ppm_now + offset/f`. Applied again it converges.
- **When it runs.** Automatically, the first time a radio is connected and has no saved
  correction. **Cal** runs it at the listening frequency, on any known carrier.

Broadcast FM was tried and rejected. Over 4 s the programme's bass does not average
away: repeat readings of one station differed by up to 10 ppm, and five stations
scattered from −22 to +51 ppm, against +4.66 ppm from the 119.8 MHz carrier.
