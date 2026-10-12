"""How long a closed session's logs are kept.

A session's directory goes when it is closed (no tmux pane, so no person
attached either), nothing in it was written or read for KEEP_CLOSED_DAYS,
and it is not pinned.  No reference counting: clients that would hold
references cannot be trusted to let go, so the dates decide.  Marker files in the directory, so that people and other
programs can set them too:

  keep             pinned: the sweep leaves it alone
  delete-on-close  its logs go the moment it is closed (or found closed)
  last-used        touched by every read, search or regions call: reading
                   changes no other file, and reads count as use

A deleted session leaves a line in removed.jsonl (its tombstone, kept
TOMBSTONE_DAYS), so that asking for it says when and why it went.

Manager (server.py) does the deleting; this module holds the rules that
need no tmux.
"""
import json
import os
import shutil
import socket as so
import sys
import time
from pathlib import Path

KEEP_MARK = "keep"
ON_CLOSE_MARK = "delete-on-close"
LAST_USED = "last-used"
TOMBSTONES = "removed.jsonl"  # in the state directory, beside sessions/
TOMBSTONE_DAYS = 30
DEFAULT_KEEP_DAYS = 3.0
SWEEP_EVERY_S = 3600  # the full sweep: at start-up, then hourly
ON_CLOSE_EVERY_S = 60  # sessions marked delete-on-close that closed by themselves
DEFAULT_SOCKET = "docent-terminal-mcp"
HOST = so.gethostname()


def keep_days(value: str | float | None = None) -> float:
    """Days a closed session's logs are kept after their last change; 0
    keeps them forever.  From `value` (the serve option), else the
    environment's DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS, else 3."""
    if value is None:
        value = os.environ.get("DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS")
    if value is None or value == "":
        return DEFAULT_KEEP_DAYS
    try:
        days = float(value)
    except ValueError:
        print(f"docent-terminal-mcp: DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS={value!r} is not a number; keeping "
              f"closed sessions {DEFAULT_KEEP_DAYS:g} days", file=sys.stderr)
        return DEFAULT_KEEP_DAYS
    return max(days, 0.0)


def marked(directory: Path, mark: str) -> bool:
    return (directory / mark).exists()


def set_mark(directory: Path, mark: str, on: bool) -> None:
    if on:
        (directory / mark).touch()
    else:
        (directory / mark).unlink(missing_ok=True)


def newest_change(directory: Path) -> float:
    """The newest mtime of anything in the directory (the directory too:
    a file removed counts as a change)."""
    newest = directory.stat().st_mtime
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            try:
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def made_here(meta: dict, home: Path, socket: str) -> bool:
    """Was this session made by a server like this one -- this host, this
    tmux socket?  Only then does "no pane in our tmux" mean closed.  (A home
    on NFS is seen by every host; a server on another host, or on another
    tmux socket, sees none of these panes.)  Sessions from before the host
    was recorded count as this host's only in a per-host state directory
    (docent-terminal-mcp-HOST, as the systemd unit makes)."""
    host = meta.get("host")
    if host is not None:
        here = host == HOST
    else:
        short = HOST.split(".")[0]
        here = home.name.endswith(f"-{short}") or home.name.endswith(f"-{HOST}")
    return here and meta.get("tmux_socket", DEFAULT_SOCKET) == socket


def expires(directory: Path, days: float) -> float | None:
    """When a closed session's logs become due for deletion (epoch s), or
    None when they are kept (pinned, or retention off)."""
    if days <= 0 or marked(directory, KEEP_MARK):
        return None
    return newest_change(directory) + days * 86400


def used(directory: Path) -> None:
    """A read, search or regions call: the logs are in use."""
    try:
        (directory / LAST_USED).touch()
    except OSError:
        pass


def remove(directory: Path, why: str, days: float) -> None:
    """Delete a session's directory, leaving its tombstone."""
    shutil.rmtree(directory)
    entry = {"name": directory.name, "removed": time.time(), "why": why, "keep_days": days}
    path = directory.parent.parent / TOMBSTONES
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def tombstone(home: Path, name: str) -> dict | None:
    """The latest tombstone of a session of this name, if any."""
    found = None
    try:
        for line in (home / TOMBSTONES).read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("name") == name:
                found = entry
    except OSError:
        pass
    return found


def gone_text(entry: dict) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["removed"]))
    return f"its logs were removed on {when} ({entry.get('why', 'deleted')})"


def prune_tombstones(home: Path) -> None:
    """Forget tombstones older than TOMBSTONE_DAYS."""
    path = home / TOMBSTONES
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    cutoff = time.time() - TOMBSTONE_DAYS * 86400
    keep = []
    for line in lines:
        try:
            if json.loads(line).get("removed", 0) >= cutoff:
                keep.append(line)
        except ValueError:
            pass
    if len(keep) != len(lines):
        path.write_text("".join(f"{line}\n" for line in keep))


def log(message: str) -> None:
    print(f"docent-terminal-mcp: {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", file=sys.stderr, flush=True)
