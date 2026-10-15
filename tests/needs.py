"""What a test needs beyond tmux and bash, and whether this machine has it.

    @needs("gdb", "cc")          a decorator: skipped, naming what is missing
    missing("ssh localhost")     the unmet ones, for a test that decides itself

A need is a program on the PATH sessions get (the server's, without its
venv: `uv run` puts the venv's python3 first), or one of the conditions in
CONDITIONS.  The README's "Tests" section lists them all; keep it in step.
"""

import functools
import os
import shutil
import subprocess

import pytest

from docent_terminal_mcp.tmux import session_environ


@functools.cache
def _ssh_localhost() -> bool:
    """`ssh localhost` logs in with no password and no host key question."""
    if shutil.which("ssh") is None:
        return False
    try:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.cache
def _proc_syscall() -> bool:
    """The server can read /proc/PID/task/TID/syscall and /proc/PID/mem of a
    program in a session.  The tmux server detaches, so those programs are
    not the server's descendants: Yama's ptrace_scope 1 (Ubuntu's default)
    allows it to root only, 3 to nobody."""
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            scope = f.read().strip()
    except OSError:
        return True  # no Yama
    return scope == "0" or (scope in ("1", "2") and os.geteuid() == 0)


CONDITIONS = {
    "ssh localhost": _ssh_localhost,
    "/proc syscall": _proc_syscall,
}


def missing(*wanted: str) -> list[str]:
    path = session_environ().get("PATH")
    return [w for w in wanted if not (CONDITIONS[w]() if w in CONDITIONS else shutil.which(w, path=path))]


def needs(*wanted: str):
    """Skip unless every one of `wanted` is there (checked once, at collection)."""
    unmet = missing(*wanted)
    return pytest.mark.skipif(bool(unmet), reason=f"needs {', '.join(unmet)}")
