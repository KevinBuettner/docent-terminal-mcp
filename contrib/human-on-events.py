#!/usr/bin/env python3
"""A simulated human who acts on what happens in a docent-terminal-mcp session,
not on a timer: it watches the session's region index and, at the moment
each event first happens, does one thing through a real attached tmux
client (contrib/simulated-human.py's Human).

    human-on-events.py SESSION STATE_DIR LOG [TIMEOUT_S]

STATE_DIR is the server's state directory (~/.local/state/docent-terminal-mcp-HOST).
Each rule fires once:
  gdb-half-line   the first time gdb is at its prompt after the agent's
                  input: type "print n" without Enter, leave it 45 s, erase it
  gdb-command     the next time: type "info breakpoints" + Enter
  suite-ctrl-c    8 s into a run-tests command: ctrl+c
  program-report  the agent's program at its own "stock> " prompt under gdb:
                  type "report" + Enter
Every action goes to LOG with a timestamp, and the session's events too.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("simulated_human", Path(__file__).with_name("simulated-human.py"))
simulated_human = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulated_human)


def regions(index: Path) -> dict[int, dict]:
    latest = {}
    try:
        for line in index.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            latest[r["id"]] = r
    except FileNotFoundError:
        pass
    return latest


def main():
    session, state, log = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    deadline = time.time() + (float(sys.argv[4]) if len(sys.argv) > 4 else 3600)
    index = state / "sessions" / session / "regions.jsonl"
    while not index.exists() and time.time() < deadline:
        time.sleep(1)
    human = simulated_human.Human(session, log, 120, 40)
    fired: set[str] = set()
    seen_gdb_prompts: set[int] = set()
    suite_started: dict[int, float] = {}

    def type_slowly(text):
        human.do(f"type {text}")

    try:
        while time.time() < deadline and len(fired) < 4:
            rs = regions(index)
            for rid, r in sorted(rs.items()):
                label = (r.get("command") or "").strip()
                mine = r.get("source") == "human"
                # gdb back at its prompt after the agent's input.
                if (r["kind"] == "input" and not mine and r["status"] == "ready" and rid not in seen_gdb_prompts
                        and rid == max(rs) and label.startswith("(gdb)")):
                    seen_gdb_prompts.add(rid)
                    if "gdb-half-line" not in fired:
                        fired.add("gdb-half-line")
                        human.note(f"EVENT region {rid} ready at (gdb): leaving a half-typed line")
                        time.sleep(1)
                        type_slowly("print n")
                        time.sleep(45)
                        human.do("backspace 7")
                    elif "gdb-command" not in fired:
                        fired.add("gdb-command")
                        human.note(f"EVENT region {rid} ready at (gdb): typing a gdb command")
                        time.sleep(1)
                        type_slowly("info breakpoints")
                        human.do("enter")
                # The agent's program at its own prompt under gdb.
                if ("program-report" not in fired and r["kind"] == "input" and not mine
                        and r["status"] == "ready" and rid == max(rs) and label.startswith("stock>")):
                    fired.add("program-report")
                    human.note(f"EVENT region {rid} at stock>: typing report")
                    time.sleep(1)
                    type_slowly("report")
                    human.do("enter")
                # The suite running.
                if r["kind"] == "command" and r["status"] == "running" and "run-tests" in label:
                    suite_started.setdefault(rid, time.time())
                    if "suite-ctrl-c" not in fired and time.time() - suite_started[rid] >= 8:
                        fired.add("suite-ctrl-c")
                        human.note(f"EVENT region {rid} (suite) running 8 s: ctrl+c")
                        human.do("ctrlc")
            time.sleep(0.3)
    finally:
        human.note(f"done; fired: {sorted(fired)}")
        human.detach()


if __name__ == "__main__":
    main()
