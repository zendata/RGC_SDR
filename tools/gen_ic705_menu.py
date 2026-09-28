#!/usr/bin/env python3
"""Build src/rgc_sdr/device/ic705_menu.py from Icom's IC-705 CI-V Reference Guide.

The guide's `1A 05` table lists every SET-menu item the radio will read or change over
CI-V: its number, the data it takes, and what it means. This turns that table into data
for the app's menu window, so the item numbers and choices come from Icom, not memory.

Usage:  tools/gen_ic705_menu.py ["docs/reference/icom ic705 CI-V Reference Manual.pdf"]
Needs pdftotext (brew install poppler). The PDF is Icom's and stays out of git.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "docs" / "reference" / "icom ic705 CI-V Reference Manual.pdf"
OUT = ROOT / "src" / "rgc_sdr" / "device" / "ic705_menu.py"
PAGES = range(4, 18)                 # the command table
HALF = 298                           # A4 is 595 pt wide; the table is two columns

TOP = r"(?:SET|SCOPE|AUDIO|KEYER|DECODE|RECORD|SCAN|GPS|DTMF|VOICE TX SET|DV MEMORY)"
HEADER = re.compile(rf"^(?:1A\*?\s+)?(?:05\s+)?({TOP}\b.*)$")
ITEM = re.compile(r"^(?:1A\*?\s+05\s+)?(\d{4})(\*\d)?\s+(See p\. \d+\.|\S+(?: ~ \S*)?)\s+(.*)$")
#: A lone capitalised line just above an item starts a new menu ("NB", "VOX", "CD").
SUBHEAD = re.compile(r"^[A-Z][A-Z0-9 /\-.]{0,20}$")
#: Page furniture that must not join a description.
FURNITURE = re.compile(r"^(\d+|REMOTE CONTROL|Remote control \(CI-V\) information|D?D?Command"
                       r" table|Cmd\. Sub cmd\..*)$")
#: The next command's rows: the menu table has ended.
NEXT_COMMAND = re.compile(r"^[0-9A-F]{2}\*?\d?\s+(\d|See p\.)")


#: The last few items sit under bare headings; say where they live on the radio.
MENU_NAMES = {"NB": "FUNCTION > NB", "VOX": "FUNCTION > VOX", "CD": "CD",
              "GPS Position": "GPS > GPS Position"}


def columns(pdf: Path) -> list[str]:
    lines: list[str] = []
    for page in PAGES:
        for x in (0, HALF):
            text = subprocess.run(
                ["pdftotext", "-f", str(page), "-l", str(page), "-layout", "-x", str(x),
                 "-y", "0", "-W", str(HALF), "-H", "842", str(pdf), "-"],
                check=True, capture_output=True, text=True).stdout
            lines += text.splitlines()
    return lines


def parse(lines: list[str]) -> list[dict]:
    items: list[dict] = []
    category, sub = "", ""
    started = False
    stripped = [ln.strip() for ln in lines]
    for i, line in enumerate(stripped):
        if not line or FURNITURE.match(line):
            continue
        repeat = re.match(r"^1A\*?\s+05\s+(\D.*)$", line)
        if repeat and started:
            category, sub = repeat.group(1).rstrip(" >"), ""    # a page's "1A 05 <menu>"
            continue
        head = HEADER.match(line)
        if head and "Send/read" not in line and not ITEM.match(line):
            category, sub = head.group(1).rstrip(" >"), ""
            started = True
            continue
        if not started:
            continue
        item = ITEM.match(line)
        if item:
            number = int(item.group(1))
            if items and number <= items[-1]["number"]:
                break                                   # past the end of the 1A 05 table
            items.append({"number": number, "data": item.group(3), "category": category,
                          "sub": sub, "text": item.group(4)})
            continue
        nxt = next((s for s in stripped[i + 1:] if s), "")
        if SUBHEAD.match(line) and ITEM.match(nxt):
            category, sub = line, ""                    # e.g. "NB" above its three items
            continue
        if items and NEXT_COMMAND.match(line):
            break
        if items:
            items[-1]["text"] += " " + line
    return items


def describe(entry: dict) -> tuple:
    """(number, path, title, digits, low, high, choices, scale, readonly)."""
    text = re.sub(r"\s+", " ", entry["text"]).replace("–", "-")
    text = re.split(r"\s*\bLL", text)[0]              # Icom's margin notes
    text = re.sub(r"Pp?\. \d+( and \d+\.)?\s*", "", text)   # "see page" references
    text = re.sub(r" and \d+\.", "", text)
    text = re.sub(r"^\d+ ", "", text)                   # a range's other end
    path_bits = [MENU_NAMES.get(entry["category"], entry["category"])]
    # Leading "RX > SSB >" style sub-paths belong to the path, not the title.
    while True:
        m = re.match(r"^((?:(?!Send/read| > ).)+?) > (.*)$", text)
        if not m or "Read" in m.group(1):
            break
        path_bits.append(m.group(1))
        text = m.group(2)
    title = re.sub(r"^.*?\b(Send/read|Read|Send)\s+(the\s+)?", "", text)
    title = re.sub(r"\s*\([^()]*=.*$", "", title).removesuffix(" setting").strip()
    title = re.sub(r"^[\d~ ]+", "", title).strip()
    title = title[:1].upper() + title[1:]
    paren = re.search(r"\(([^()]*=.*)\)", text)
    choices: dict[int, str] = {}
    scale = None
    data = entry["data"]
    readonly = data.startswith("See p.") or text.startswith("Read ")
    m = re.match(r"^(\d+)(?:/(\d+))*$", data)
    rng = re.match(r"^(\d+) ~ (\d*)$", data)
    if m:
        codes = [int(c) for c in data.split("/")]
        digits = len(data.split("/")[0])
        low, high = min(codes), max(codes)
    elif rng:
        digits = len(rng.group(1))
        low = int(rng.group(1))
        high = int(rng.group(2)) if rng.group(2) else low
    else:
        digits, low, high = 0, 0, 0
        readonly = True
    if paren and digits:
        body = paren.group(1)
        pairs = re.findall(r"(\d{2,8})=(.+?)(?=,\s*\d{2,8}=|\s*~|$)", body)
        if "~" in body and len(pairs) >= 2:
            (a, la), (b, lb) = pairs[0], pairs[-1]
            num = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([^\s\d]*)")
            ma, mb = num.match(la.strip()), num.match(lb.strip())
            if ma and mb and mb.group(2) not in (":", "/"):    # not times or dates
                scale = (int(a), float(ma.group(1)), int(b), float(mb.group(1)), mb.group(2))
        if scale is None and "~" not in body:
            for code, label in pairs:
                choices[int(code)] = label.strip()
    if digits > 4:
        readonly = True                                 # dates and the like
    return (entry["number"], " > ".join(path_bits), title, digits, low, high,
            choices, scale, readonly)


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF
    rows = [describe(e) for e in parse(columns(pdf))]
    body = ",\n".join("    " + repr(r) for r in rows)
    OUT.write_text(f'''"""The IC-705's SET-menu items over CI-V (command 1A 05), generated by
tools/gen_ic705_menu.py from Icom's IC-705 CI-V Reference Guide. Do not edit by hand.

Each row: (item number, menu path, title, BCD digits, lowest, highest,
{{code: label}} choices, scale (code_a, value_a, code_b, value_b, unit) or None,
read-only). {len(rows)} items.
"""

MENU_ITEMS = (
{body},
)
''')
    print(f"{len(rows)} items -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
