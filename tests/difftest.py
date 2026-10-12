"""Differential testing of the cooked view against tmux's own rendering.

A seeded generator writes plausible terminal output (text with wide and
combining characters, CR, BS, TAB, LF, cursor moves, line erases, colours)
into a real tmux pane.  tmux's rendering, with wrapped rows joined back into
lines (`capture-pane -J`), is what the cooked view promises to be: the two
must agree.

    python tests/difftest.py [SEEDS] [FIRST_SEED]    # fuzz, report mismatches

test_cooked_matches_tmux in test_difftest.py runs a fixed set of seeds.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docent_terminal_mcp.session import Session  # noqa: E402
from docent_terminal_mcp.tmux import Tmux, cell_width, expand_tabs  # noqa: E402

WIDTH, HEIGHT = 40, 20
TEXT = "abcdefghij klmnop XYZ 0123456789.,-"
WIDE = "日本語漢字😀"
COMBINING = "́̈"


VERTICAL = os.environ.get("DIFFTEST_VERTICAL", "1") == "1"
# Vertical moves as progress displays make them (pip, cargo, docker): up N
# rows, then each row rewritten within the width and erased to its end.
# Random moves also produce text overflowing a row it moved onto, which the
# cooked view keeps in its own line (tmux wraps it over the next row).
REDRAW = os.environ.get("DIFFTEST_REDRAW", "0") == "1"
# tmux keeps a tab as a cell of its own; after an erase inside its span its
# capture no longer tells which cells are blank.  Tabs can be left out.
TABS = os.environ.get("DIFFTEST_TABS", "1") == "1"
# Compare tmux's joined lines (-J) instead of screen rows (the default).
JOINED = os.environ.get("DIFFTEST_JOINED", "0") == "1"


def wrap(line: str, width: int) -> list[str]:
    """A cooked line as the screen rows a terminal of `width` shows it: a
    wide character that does not fit in the last cell goes to the next row
    (the cell left blank)."""
    rows, row, col = [], [], 0
    for ch in line:
        w = cell_width(ch)
        if col + w > width:
            rows.append("".join(row).rstrip())
            row, col = [], 0
        row.append(ch)
        col += w
    rows.append("".join(row).rstrip())
    return rows


def generate(seed: int, length: int = 120) -> bytes:
    """Terminal output the cooked view is meant to handle (no absolute
    cursor addressing, no alternate screen: those are full-screen)."""
    return render(tokens(seed, length))


def render(parts: list[str]) -> bytes:
    return ("".join(parts) + "\x1b[0m\r\nEND-OF-STREAM\r\n").encode()


def tokens(seed: int, length: int = 120) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(length):
        k = rng.random()
        if k < 0.45:
            out.append("".join(rng.choice(TEXT) for _ in range(rng.randint(1, 30))))
        elif k < 0.52:
            out.append(rng.choice(WIDE) * rng.randint(1, 4))
        elif k < 0.55:
            out.append(rng.choice("aeiou") + rng.choice(COMBINING))
        elif k < 0.65:
            out.append("\r\n")
        elif k < 0.70:
            out.append("\r")
        elif k < 0.74:
            out.append("\b" * rng.randint(1, 3))
        elif k < 0.77:
            if TABS:
                out.append("\t")
        elif k < 0.81:
            out.append(f"\x1b[{rng.randint(1, 10)}D")
        elif k < 0.84:
            out.append(f"\x1b[{rng.randint(1, 10)}C")
        elif k < 0.88:
            out.append(f"\x1b[{rng.choice(['', '0', '1', '2'])}K")
        elif k < 0.91 and REDRAW:
            n = rng.randint(1, 3)
            rows = "".join("\r" + "".join(rng.choice(TEXT) for _ in range(rng.randint(0, WIDTH - 1)))
                           + "\x1b[K\r\n" for _ in range(n))
            out.append(f"\r\n{'x' * 5}\r\n{'y' * 5}\r\n{'z' * 5}\r\n\x1b[{n}A" + rows)
        elif k < 0.91:
            if VERTICAL:
                out.append(f"\x1b[{rng.randint(1, 3)}A")
        elif k < 0.93:
            if VERTICAL:
                out.append(f"\x1b[{rng.randint(1, 2)}B")
        else:
            out.append(f"\x1b[{rng.choice(['0', '1', '31', '1;32', '7'])}m")
    return out


async def compare(seed: int, workdir: Path, stream_bytes: bytes | None = None) -> list[str] | None:
    """Play the seed's stream in a pane; None if cooked and tmux agree,
    else the differing lines."""
    socket = f"difftest-{os.getpid()}-{seed}"
    config = workdir / "tmux.conf"
    config.write_text("set -g history-limit 50000\nset -g remain-on-exit on\n")
    tmux = Tmux(socket, config)
    d = workdir / f"s{seed}-{random.randrange(1 << 30)}"
    d.mkdir()
    stream = d / "stream"
    stream.write_bytes(stream_bytes if stream_bytes is not None else generate(seed))
    raw = d / "raw.log"
    raw.touch()
    (d / "widths.jsonl").write_text(json.dumps({"offset": 0, "width": WIDTH}) + "\n")
    # stty -onlcr raw-ish: the stream's own CR/LF, as a program writes them.
    argv = ["sh", "-c", f"stty -onlcr; cat {stream}; sleep 600"]
    await tmux.new_session("s", argv, str(d), {}, WIDTH, HEIGHT, raw)
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if b"END-OF-STREAM" in raw.read_bytes():
                break
        await asyncio.sleep(0.2)
        meta = {"name": "s", "command": "sh", "cols": WIDTH, "shell_integration": False}
        session = Session("s", d, tmux, meta)
        if JOINED:
            # Lines from the cells: an empty cell after a narrow one is the
            # blank before a wide character that did not fit (a space in
            # tmux's joined capture); after a wide one, its right half.
            cooked = []
            for r in range(len(session.term.lines)):
                cells = session.term.lines[r]
                if cells is None:
                    cooked.append(session.term.line(r).rstrip())
                    continue
                cooked.append("".join(" " if c == "" and (i == 0 or cell_width((cells[i - 1] or " ")[0]) < 2)
                                      else c for i, c in enumerate(cells)).rstrip())
            out = await tmux.run("capture-pane", "-p", "-J", "-S", "-", "-t", tmux.target("s"))
            shown = [line.rstrip() for line in out.split("\n")]
        else:
            # Screen rows straight from the cooked cells (one per column):
            # the text form cannot show a cell left empty before a wrapped
            # wide character once that character is gone.
            cooked = []
            for r in range(len(session.term.lines)):
                cells = session.term.lines[r]
                if cells is None:
                    cooked += [row.rstrip() for row in wrap(session.term.line(r), WIDTH)]
                    continue
                cooked += ["".join(cells[i:i + WIDTH]).rstrip() for i in range(0, max(len(cells), 1), WIDTH)]
            out = await tmux.run("capture-pane", "-p", "-S", "-", "-t", tmux.target("s"))
            shown = [expand_tabs(line, WIDTH).rstrip() for line in out.split("\n")]
        session.close()
    finally:
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
    # From the first line on (stty prints nothing), up to END-OF-STREAM.
    def upto_end(lines):
        # (A move up can land the marker on an old row: it starts the line.)
        return lines[: next((i for i, l in enumerate(lines) if l.startswith("END-OF-STREAM")), len(lines)) + 1]
    def normal(rows):
        # An empty row right after a full one may be the erased continuation
        # of a wrapped line (tmux keeps it, the text has nothing there):
        # dropped on both sides alike.
        out = []
        for row in rows:
            if row == "" and out and sum(cell_width(c) for c in out[-1]) >= WIDTH:
                continue
            out.append(row)
        return out

    a, b = normal(upto_end(cooked)), normal(upto_end(shown))
    if a == b:
        return None
    diff = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<missing>"
        y = b[i] if i < len(b) else "<missing>"
        if x != y:
            diff.append(f"line {i + 1}:\n  cooked: {x!r}\n  tmux:   {y!r}")
    return diff


async def shrink(seed: int, workdir: Path) -> list[str]:
    """The seed's tokens cut down (delta debugging) while cooked and tmux
    still disagree: a minimal case to read."""
    parts = tokens(seed)
    chunk = len(parts) // 2
    while chunk >= 1:
        i, changed = 0, False
        while i < len(parts):
            trial = parts[:i] + parts[i + chunk:]
            if trial and await compare(seed, workdir, render(trial)):
                parts, changed = trial, True
            else:
                i += chunk
        if not changed:
            chunk //= 2
    return parts


async def main(n: int, first: int) -> int:
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(first, first + n):
            diff = await compare(seed, Path(tmp))
            if diff:
                bad += 1
                print(f"seed {seed}: {len(diff)} differing lines")
                print("\n".join(diff[:4]))
    print(f"{bad} of {n} seeds differ")
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] == ["shrink"]:
        async def one(seed):
            with tempfile.TemporaryDirectory() as tmp:
                parts = await shrink(seed, Path(tmp))
                print(f"seed {seed} minimal: {parts!r}")
                print("\n".join(await compare(seed, Path(tmp), render(parts)) or ["(agrees?)"]))
        for seed in map(int, sys.argv[2:]):
            asyncio.run(one(seed))
        sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sys.exit(1 if asyncio.run(main(n, first)) else 0)
