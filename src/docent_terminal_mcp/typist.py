"""Typing a line as a person would: one key at a time, at a human speed
with jitter, pausing at the end of a phrase, and now and then making a
mistake that is noticed a few keys later, erased and corrected.

For demos and for sessions a human watches.  The plan (`plan`) is pure and
seeded, so a demo can be repeated exactly; `type_out` performs it.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

# Keys next to each other on a QWERTY keyboard, for the occasional slip.
ROWS = ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"]
NEAR: dict[str, str] = {}
for r, row in enumerate(ROWS):
    for c, ch in enumerate(row):
        near = [row[i] for i in (c - 1, c + 1) if 0 <= i < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS):
                near += [ROWS[rr][i] for i in (c - 1, c, c + 1) if 0 <= i < len(ROWS[rr])]
        NEAR[ch] = "".join(n for n in near if n.isalnum())

# Misspellings people make, and the word as meant.
MISSPELLINGS = {
    "the": "teh", "and": "adn", "with": "wiht", "that": "taht", "from": "form", "this": "tihs",
    "which": "whcih", "their": "thier", "receive": "recieve", "because": "becuase",
    "make": "amke", "just": "jsut", "print": "pirnt", "echo": "ecoh", "grep": "gerp",
    "cd": "dc", "ls": "sl", "git": "gti", "python3": "pyhton3",
}
PHRASE_ENDS = ".,;:!?"


@dataclass
class Key:
    """One step: `text` typed (one character), or BSpace; after `delay` s."""
    text: str | None
    delay: float

    @property
    def backspace(self) -> bool:
        return self.text is None


def plan(line: str, wps: float = 2.0, seed: int | None = None, mistakes: float = 0.1) -> list[Key]:
    """The keystrokes for typing `line` at `wps` words per second (a word
    being five letters and a space), with a mistake in about `mistakes` of
    the words: a transposition or a misspelling most often, a neighbouring
    key sometimes; each noticed 0-3 keys later, erased and retyped."""
    rng = random.Random(seed)
    per_key = 1 / max(wps * 6, 0.1)
    keys: list[Key] = []

    def gap(ch: str, prev: str) -> float:
        d = per_key * rng.lognormvariate(0, 0.35)
        if prev and prev in PHRASE_ENDS and ch == " ":
            d += rng.uniform(0.3, 1.0)  # the end of a phrase: thinking about the next
        if ch in "|&;" and prev == " ":
            d += rng.uniform(0.2, 0.8)  # before the next stage of a pipeline
        return d

    def typed(text: str, prev: str) -> str:
        for ch in text:
            keys.append(Key(ch, gap(ch, prev)))
            prev = ch
        return prev

    def erase(n: int) -> None:
        keys.append(Key(None, rng.uniform(0.15, 0.5)))  # the moment of noticing
        for _ in range(n - 1):
            keys.append(Key(None, per_key * rng.uniform(0.5, 1.0)))

    prev = ""
    i = 0
    while i < len(line):
        # The next word (letters and digits) or single other character.
        j = i
        while j < len(line) and (line[j].isalnum() or line[j] in "_-"):
            j += 1
        word = line[i:j] if j > i else line[i]
        after = line[j:j + rng.randint(0, 3)] if j > i else ""
        wrong = None
        if j > i and len(word) >= 2 and rng.random() < mistakes:
            kind = rng.random()
            if word.lower() in MISSPELLINGS and kind < 0.5:
                wrong = MISSPELLINGS[word.lower()]
            elif kind < 0.8 and len(word) >= 3:
                k = rng.randrange(len(word) - 1)
                wrong = word[:k] + word[k + 1] + word[k] + word[k + 2:]  # transposed
            else:
                k = rng.randrange(len(word))
                slip = rng.choice(NEAR.get(word[k].lower(), "") or word[k])
                wrong = word[:k] + (slip.upper() if word[k].isupper() else slip) + word[k + 1:]
            if wrong == word:
                wrong = None
        if wrong is not None:
            # Typed wrong, typed on a little (the next few characters of the
            # line), noticed: erase back to the first wrong key, retype.
            first = next(k for k in range(len(word)) if wrong[k] != word[k])
            prev = typed(wrong, prev)
            prev = typed(after, prev)
            erase(len(wrong) - first + len(after))
            prev = typed(word[first:] + after, word[first - 1] if first else prev)
            i = j + len(after)
        else:
            prev = typed(word, prev)
            i += len(word)
    return keys


def replay(keys: list[Key]) -> str:
    """The line the keys leave (backspace erasing): for checking a plan."""
    out: list[str] = []
    for k in keys:
        if k.backspace:
            if out:
                out.pop()
        elif k.text is not None:
            out.append(k.text)
    return "".join(out)


async def type_out(tmux, name: str, line: str, wps: float = 2.0, seed: int | None = None,
                   mistakes: float = 0.1) -> float:
    """Type `line` into session `name` as planned; returns the seconds it took."""
    total = 0.0
    for k in plan(line, wps, seed, mistakes):
        await asyncio.sleep(k.delay)
        total += k.delay
        if k.backspace:
            await tmux.send_keys(name, ["BSpace"])
        else:
            await tmux.send_text(name, k.text, literal_tabs=True)
    return total
