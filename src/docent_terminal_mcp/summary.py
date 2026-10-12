"""Summary lines: what a test run or a build says about how it went, picked
out of long output without the caller choosing a `show` pattern.

Models handed a long `make check` either piped it through `| tail` or
redirected it to a file, and then read the results some other way (models
large and small alike, even after the reply notes); they
asked for the pass/fail lines in the reply without having to pick a filter.
A reply whose output is clipped leads with these lines when they are not
among the lines it shows anyway.

Only well-known shapes, each anchored at the start of a line: a pattern that
fires on ordinary output would make the block noise.
"""
from __future__ import annotations

import re
from typing import Iterable

# A failure or problem named one per line.
FAILURES = re.compile(
    r"^(?:"
    r"(?:FAIL|ERROR|UNRESOLVED|XPASS|KPASS):\s"      # DejaGnu (GDB, binutils, GCC)
    r"|FAILED\s\S"                                  # pytest's short summary
    r"|--- FAIL:\s"                                 # go test
    r"|not ok\s+\d+"                                # TAP: pg_regress, prove
    r"|\S+:\d+(?::\d+)?:\s+(?:fatal\s+)?error:\s"   # gcc/clang diagnostics
    r"|\d+:\d\d:\d\d load avg: .*\] \S+ failed\b"     # CPython's regrtest, per test
    r"|test \S+ failed -- "                         # regrtest, with its traceback
    r")"
)
# A line that sums a run up.
TOTALS = re.compile(
    r"^(?:"
    r"=+ .*\b(?:passed|failed|errors?|no tests ran)\b.* in [\d.]+s.*=+$"  # pytest
    r"|test result: "                               # cargo test
    r"|(?:FAIL|ok)\s+\S+\s+[\d.]+s$"                # go test, per package
    r"|\d+% tests passed, \d+ tests? failed"       # ctest
    r"|# (?:All \d+ tests passed|\d+ of \d+ tests failed)"  # pg_regress
    r"|All \d+ tests passed\.|\d+ of \d+ tests failed\."
    r"|== Tests result: "                           # CPython's regrtest
    r"|\d+ tests? failed:$"                         # regrtest, before the list of names
    r"|Ran \d+ tests? in |FAILED \((?:failures|errors)="  # unittest
    r"|make(?:\[\d+\])?: \*\*\* "                   # make giving up
    r")"
)
# DejaGnu prints a summary per test file in a parallel run, then the whole
# run's at the end: only the last block counts (a full GDB `make -j check`
# printed 2,688 of them).
DEJAGNU_HEADER = re.compile(r"^\s*=== .* Summary ===\s*$")
DEJAGNU_COUNT = re.compile(r"^# of ")
REGRTEST_FAILED = re.compile(r"^\d+ tests? failed:$")

MAX_LINES = 20  # a GDB run's final summary alone is ten
MAKE_ERRORS_KEPT = 1  # nested makes repeat the failure once per level: the outermost says it


def summary_lines(lines: Iterable[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(row, text) of the summary lines in `lines`, in order, at most
    MAX_LINES -- totals first in priority, then failures -- and the
    failure lines left out, those most worth seeing first."""
    failures: list[tuple[int, str]] = []
    totals: list[tuple[int, str]] = []
    make_errors: list[tuple[int, str]] = []
    block: list[tuple[int, str]] = []  # the latest DejaGnu summary block
    in_block = False
    in_failed_list = False  # regrtest's "2 tests failed:" and the names under it
    for row, text in lines:
        if in_failed_list:
            if text.startswith("    ") and text.strip():
                totals.append((row, text))
                continue
            in_failed_list = False
        if REGRTEST_FAILED.match(text):
            totals.append((row, text))
            in_failed_list = True
            continue
        if DEJAGNU_HEADER.match(text):
            block, in_block = [(row, text)], True
            continue
        if in_block:
            if DEJAGNU_COUNT.match(text):
                block.append((row, text))
                continue
            if text.strip():
                in_block = False
        if text.startswith("make") and TOTALS.match(text) and "***" in text:
            make_errors.append((row, text))
        elif TOTALS.match(text):
            totals.append((row, text))
        elif FAILURES.match(text):
            failures.append((row, text))
    totals += (block if len(block) > 1 else []) + make_errors[-MAKE_ERRORS_KEPT:]
    # A failed test before a harness's complaints: DejaGnu's ERROR: comes in
    # multi-line tcl dumps that would crowd the FAILs out.
    failures.sort(key=lambda x: x[1].startswith(("ERROR:", "UNRESOLVED:")))
    room = max(MAX_LINES - len(totals), 0)
    kept = failures[:room]
    return sorted(totals[:MAX_LINES] + kept), failures[len(kept):]


def search_pattern(lines: list[tuple[int, str]]) -> str:
    """A `search` pattern for the failure lines left out."""
    kinds = []
    for _, text in lines:
        if re.match(r"\S+:\d+(?::\d+)?:\s+(?:fatal\s+)?error:\s", text):
            kind = r": (fatal )?error: "
        elif (m := re.match(r"(?:FAIL|ERROR|UNRESOLVED|XPASS|KPASS):|FAILED |--- FAIL:|not ok ", text)):
            kind = "^" + re.escape(m.group(0)).replace("\\ ", " ")
        else:
            continue
        if kind not in kinds:
            kinds.append(kind)
    return "|".join(kinds) or FAILURES.pattern


def is_test_result(text: str) -> bool:
    """A line that reports a test run (not a build's error or make's)."""
    if text.startswith("make") or re.match(r"\S+:\d+(?::\d+)?:\s+(?:fatal\s+)?error:\s", text):
        return False
    return bool(DEJAGNU_HEADER.match(text) or DEJAGNU_COUNT.match(text) or TOTALS.match(text)
                or FAILURES.match(text))
