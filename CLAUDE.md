# RGC_SDR

A learning-focused SDR receiver and transceiver for macOS on Apple silicon,
built over SoapySDR (Airspy HF+, HackRF, PlutoSDR and others) with an Icom
IC-705 link over USB and WiFi. The point is to understand SDR and DSP by
building it, one roadmap phase at a time.

`PLANNING.md` is the source of truth. Read it first: §6 defines when a phase
is done, §3 records measured hardware facts (trust them over datasheets, and
re-measure rather than assume if the hardware changes), and §11 has the
conventions for any model working here. This repo is **public**.

## How to work here

- Build the next unstarted phase from §6, end it runnable and tested, and
  don't start several at once.
- Real hardware only: never add a simulated device mode (§1). Test doubles
  that feed pure DSP functions are fine; a fake device path in the app is not.
- Fix factual errors in the plan when you find them rather than implementing
  a flawed spec as written, and say what you changed.
- Python 3.14 with type hints; vectorised NumPy DSP with no per-sample Python
  loops; respect the layering rule in §5 and keep device support
  driver-agnostic (§4).
- Regional defaults are Australian: 50 µs FM de-emphasis, the European RDS PTY
  table (not RBDS), 9 kHz AM spacing, ACMA band plans.

## Testing

Run `.venv/bin/pytest` before every commit. Plain `pytest` covers DSP and UI
with no radio attached (Qt runs offscreen); `pytest -m hardware` streams from
an attached device and skips when none is present.

## Hardware gotchas

- Implausibly low levels (a floor near -134 dBFS, FM stations barely above
  noise) almost always mean the antenna is unplugged. Ask before debugging DSP.
  The dBFS scale itself is pinned by a test.
- The Airspy HF+ leaves a birdie at its tuned frequency; offset-tune.
- A max-hold spectrum sits about 15 dB above the average noise floor, so a
  detection threshold below about 20 dB reports transmitters in pure static.
- Transmit (HackRF) has deliberately no band, mode or power interlocks, at the
  operator's request; don't add them. CW transmit is refused, there's a
  3-minute timeout, gain starts at minimum, and tuning is locked while keyed.
  The first key-up of any new transmit path is the operator's to do, not
  Claude's.
- Decoded pager (POCSAG) traffic can carry names, addresses and medical
  details: redact it, and never paste decoded message bodies into commits,
  issues or chat.

## Files that must never be committed

The IC-705 manuals in `docs/reference/` are Icom's copyright and git-ignored;
read only the pages you need. Never commit credentials, radio passwords,
home network addresses or hostnames — the repo is public. `.venv/` and
`.pytest_cache/` are ignored only by the self-ignoring files Python and pytest
drop inside them, so check `git status --untracked-files=all` and a
`git add -An` dry run before staging.

## Commits

Commit each coherent chunk (a phase, a fix, a doc rewrite) and push straight
to `origin/main` without being asked; there's no PR workflow. Prefer a few
logical, bisectable commits over one large one.

The IC-705 WiFi bridge (Pi Zero W running wfview's server) is a separate public
repo, `~/Code/ic705-wifi-bridge`, and must stay usable by any Icom network
client, not just this app.
