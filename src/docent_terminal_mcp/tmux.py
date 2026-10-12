"""Thin async wrapper over the tmux command line.

All sessions live on a dedicated tmux server (`tmux -L <socket>`), so they
never mix with the user's own tmux sessions.  A human watches or joins one
with `tmux -L <socket> attach -t <name>`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shlex
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("docent_terminal_mcp")


class TmuxError(Exception):
    pass


SEND_CHUNK = 2048  # characters per send-keys

# The server's own environment that must not reach the sessions: its venv
# (a `pip install` there would change the running server; `python3` would
# be the server's), its service's bookkeeping, its own settings.
SERVER_ONLY_ENV = ("VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT",
                   "INVOCATION_ID", "JOURNAL_STREAM", "MEMORY_PRESSURE_WATCH", "MEMORY_PRESSURE_WRITE",
                   "SYSTEMD_EXEC_PID", "NOTIFY_SOCKET", "MANAGERPID", "DOCENT_TERMINAL_MCP_HOME",
                   "DOCENT_TERMINAL_MCP_SOCKET", "DOCENT_TERMINAL_MCP_SOCKET_PATH")


def cell_width(ch: str) -> int:
    if unicodedata.combining(ch) or ch in "\u200b\u200c\u200d\ufe0f":
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def expand_tabs(row: str, width: int) -> str:
    """A captured screen row with its tabs as the cells they cover: to the
    next multiple of 8, never past the last column (a terminal's tab
    stops), counting wide characters as two cells."""
    if "\t" not in row:
        return row
    out, col = [], 0
    for ch in row:
        if ch == "\t":
            stop = min((col // 8 + 1) * 8, width - 1)
            out.append(" " * max(stop - col, 0))
            col = max(stop, col)
        else:
            out.append(ch)
            col += cell_width(ch)
    return "".join(out)


def session_environ() -> dict[str, str]:
    """The environment tmux (and so every session) is given: this
    process's, without SERVER_ONLY_ENV and without the venv on PATH."""
    env = {k: v for k, v in os.environ.items() if k not in SERVER_ONLY_ENV}
    venv_bins = {os.path.dirname(sys.executable)}
    if venv := os.environ.get("VIRTUAL_ENV"):
        venv_bins.add(os.path.join(venv, "bin"))
    if "PATH" in env:
        env["PATH"] = os.pathsep.join(d for d in env["PATH"].split(os.pathsep) if d not in venv_bins)
    return env


@dataclass
class PaneInfo:
    dead: bool
    dead_status: int | None
    width: int
    height: int
    alternate_on: bool
    current_command: str
    cursor_x: int
    cursor_y: int
    tty: str
    pid: int  # the pane's own process: the session's shell


# tmux names for the keys `send` accepts, keyed by the lower-case spelling a
# model is likely to use.  Anything not here is rejected rather than typed as
# literal text, which is what tmux would otherwise do with an unknown name.
KEYS = {
    "enter": "Enter", "return": "Enter", "tab": "Tab", "btab": "BTab",
    "escape": "Escape", "esc": "Escape", "space": "Space",
    "backspace": "BSpace", "bspace": "BSpace", "delete": "DC", "insert": "IC",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "PPage", "pagedown": "NPage",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}


def normalize_key(key: str) -> str:
    """Map spellings like "ctrl+c", "C-c", "^C", "alt+x", "Up" to tmux names."""
    k = key if key == " " else key.strip()
    if len(k) == 1 and k.isprintable():
        return k  # a single character is its own tmux key name
    low = k.lower()
    if low in KEYS:
        return KEYS[low]
    for prefix in ("ctrl+", "ctrl-", "control+", "c-"):
        if low.startswith(prefix) and len(low) == len(prefix) + 1:
            return "C-" + low[-1]
    if len(low) == 2 and low[0] == "^":
        return "C-" + low[1]
    for prefix in ("alt+", "alt-", "meta+", "m-"):
        if low.startswith(prefix):
            rest = k[len(prefix):]
            if len(rest) == 1:
                return "M-" + rest
            if rest.lower() in KEYS:
                return "M-" + KEYS[rest.lower()]
    raise TmuxError(
        f"unknown key {key!r}; use names like ctrl+c, ctrl+d, enter, tab, escape, "
        "backspace, up, down, left, right, home, end, pageup, pagedown, f1-f12, alt+x, "
        "or a single character"
    )


PANE_FORMAT = "\t".join([
    "#{pane_dead}", "#{pane_dead_status}", "#{pane_width}", "#{pane_height}",
    "#{alternate_on}", "#{cursor_x}", "#{cursor_y}", "#{pane_tty}", "#{pane_pid}",
    "#{pane_current_command}",
])


def parse_pane_info(out: str, what: str) -> "PaneInfo":
    dead, status, width, height, alt, cx, cy, tty, pid, cmd = (out.split("\t", 9) + [""] * 10)[:10]
    if not (width.isdigit() and height.isdigit() and cx.isdigit() and cy.isdigit()):
        raise TmuxError(f"tmux display-message answered oddly for session {what!r}: {out!r}")
    return PaneInfo(
        dead=dead == "1",
        dead_status=int(status) if status.lstrip("-").isdigit() else None,
        width=int(width), height=int(height), alternate_on=alt == "1",
        current_command=cmd, cursor_x=int(cx), cursor_y=int(cy), tty=tty,
        pid=int(pid) if pid.isdigit() else 0,
    )


@dataclass
class PaneRow:
    """One pane as `list-panes -a` reports it."""
    pane_id: str
    window: str  # the tmux session holding it
    index: int
    active: bool
    piped: bool
    dead: bool


class Tmux:
    def __init__(self, socket: str, config: Path):
        self.socket = socket
        self.config = config
        # A session is one pane.  Its calls target the pane by id (%N): the
        # window's *active* pane is whichever a split or a human's click made
        # active, and typing into it would be typing into another pane.
        self.panes: dict[str, str] = {}  # session name -> pane id
        self.windows: dict[str, str] = {}  # session name -> tmux session holding its pane

    def register(self, name: str, pane_id: str | None, window: str | None = None) -> None:
        if pane_id:
            self.panes[name] = pane_id
        self.windows[name] = window or name

    def forget(self, name: str) -> None:
        self.panes.pop(name, None)
        self.windows.pop(name, None)

    def window(self, name: str) -> str:
        """The tmux session holding this session's pane."""
        return self.windows.get(name, name)

    def pane(self, name: str) -> str:
        """The session's own pane as a tmux target."""
        return self.panes.get(name) or self.target(self.window(name))

    def wtarget(self, name: str) -> str:
        """The window holding the session's pane, for window-wide calls
        (size, clients)."""
        return self.target(self.window(name))

    def attach_target(self, name: str) -> str:
        """The target a human gives `tmux attach`: the window's session, by
        bare name unless a dot in it would be read as window.pane (tmux
        3.7c)."""
        window = self.window(name)
        return self.target(window) if "." in window else window

    def attach_argv(self, name: str) -> list[str]:
        """The command a human runs to watch or join a session.  When the
        server runs where they cannot run tmux directly -- in a container,
        an OpenShell sandbox -- DOCENT_TERMINAL_MCP_ATTACH_PREFIX holds the words
        that get them there: "openshell sandbox exec -n NAME --tty --",
        "podman exec -it NAME", "ssh -t HOST"."""
        prefix = shlex.split(os.environ.get("DOCENT_TERMINAL_MCP_ATTACH_PREFIX", ""))
        return prefix + ["tmux", "-L", self.socket, "attach", "-t", self.attach_target(name)]

    def attach_command(self, name: str) -> str:
        return shlex.join(self.attach_argv(name))

    def expected_socket_path(self) -> str:
        """Where tmux puts this socket, worked out as tmux does ($TMUX_TMPDIR,
        else /tmp; tmux-UID; the name) -- also when no server runs."""
        base = os.path.realpath(os.path.join(os.environ.get("TMUX_TMPDIR") or "/tmp", f"tmux-{os.getuid()}"))
        return os.path.join(base, self.socket)

    async def socket_path(self) -> str | None:
        """Where the tmux server's socket is: under TMUX_TMPDIR (or /tmp),
        which a client in another mount namespace (a private /tmp) or with
        another TMUX_TMPDIR does not see at the same path."""
        out = await self.run("display-message", "-p", "#{socket_path}", check=False)
        return out.strip() or None

    @staticmethod
    def target(name: str) -> str:
        """The session as an unambiguous tmux target: exactly this session
        (=), its current window (:).  A bare name with a dot in it is read
        as window.pane by tmux 3.7c ("can't find pane"), and by then only
        for some commands."""
        return f"={name}:"

    async def run(self, *args: str, check: bool = True, input: bytes | None = None) -> str:
        proc = await asyncio.create_subprocess_exec(
            "tmux", "-L", self.socket, "-f", str(self.config), *args,
            env=session_environ(),
            stdin=asyncio.subprocess.DEVNULL if input is None else asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(input)
        if check and proc.returncode != 0:
            raise TmuxError(f"tmux {args[0]}: {err.decode(errors='replace').strip()}")
        return out.decode(errors="replace")

    async def server_running(self) -> bool:
        proc = await asyncio.create_subprocess_exec(
            "tmux", "-L", self.socket, "list-sessions",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        return await proc.wait() == 0

    async def ensure_server(self) -> None:
        """Start the tmux server if none runs, and, under a systemd service,
        start it in a transient scope of its own: stopping or restarting a
        service kills every process in its control group, and the sessions
        must outlive the server (a plain server process is simply outlived
        by the daemon tmux forks).  Elsewhere the first new-session starts
        the server.  The scope's server is told to stay when its last session
        closes, so that new sessions keep landing in it."""
        if await self.server_running():
            return
        if not (os.environ.get("INVOCATION_ID") and shutil.which("systemd-run")):
            return
        unit = "docent-terminal-mcp-tmux-" + re.sub(r"[^\w.-]", "_", self.socket)
        proc = await asyncio.create_subprocess_exec(
            "systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
            "--description=tmux server holding docent-terminal-mcp sessions",
            "tmux", "-L", self.socket, "-f", str(self.config),
            "start-server", ";", "set-option", "-s", "exit-empty", "off",
            env=session_environ(),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            log.warning("tmux server not started in its own systemd scope (%s): "
                        "sessions will not survive a restart of this service",
                        err.decode(errors="replace").strip())

    async def new_session(
        self, name: str, argv: list[str], cwd: str, env: dict[str, str],
        cols: int, rows: int, raw_log: Path,
    ) -> str:
        """Start the session; returns its pane id."""
        # The pane blocks on a tmux channel until pipe-pane is attached, so no
        # output -- not even the first prompt -- escapes the log.
        channel = f"docent-terminal-mcp-ready-{name}"
        args = ["new-session", "-d", "-P", "-F", "#{pane_id}", "-s", name, "-x", str(cols), "-y", str(rows),
                "-c", cwd]
        await self.ensure_server()
        await self.scrub_global_environment()
        pane_id = (await self.run(*args, *self._gated(channel, env, argv))).strip()
        self.register(name, pane_id)
        await self.start_logging(pane_id, raw_log, channel)
        return pane_id

    async def split(
        self, name: str, parent: str, argv: list[str], cwd: str, env: dict[str, str],
        below: bool, size: str | None, raw_log: Path,
    ) -> str:
        """A new pane in `parent`'s window, next to its pane, logged from its
        first byte like a new session's once the caller has called
        start_logging with the channel; the parent's pane stays active.
        Returns the new pane's id."""
        channel = f"docent-terminal-mcp-ready-{name}"
        args = ["split-window", "-d", "-P", "-F", "#{pane_id}", "-v" if below else "-h", "-t", self.pane(parent),
                "-c", cwd]
        if size:
            args += ["-l", size]
        pane_id = (await self.run(*args, *self._gated(channel, env, argv))).strip()
        self.register(name, pane_id, self.window(parent))
        return pane_id  # gated: the caller starts its logging (start_logging)

    @staticmethod
    def _gated(channel: str, env: dict[str, str], argv: list[str]) -> list[str]:
        args = []
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        gate = f"tmux wait-for {shlex.quote(channel)}; exec \"$@\""
        # tmux puts its own PATH in place of one given with -e (3.7b): a
        # session_create with env PATH=/x/bin and command "psql" died with
        # "exec: psql: not found".  `env`
        # sets every variable again for the command itself.
        assignments = [f"{k}={v}" for k, v in env.items()]
        return args + ["--", "sh", "-c", gate, "sh", *(["env", *assignments] if assignments else []), *argv]

    async def start_logging(self, pane_id: str, raw_log: Path, channel: str | None = None) -> None:
        await self.run("pipe-pane", "-O", "-t", pane_id, f"exec cat >> {shlex.quote(str(raw_log))}")
        if channel:
            await self.run("wait-for", "-S", channel)

    async def adopt_pane(self, name: str, pane_id: str, window: str, raw_log: Path) -> None:
        """Log a pane that appeared without us (a split made by a human or by
        `tmux split-window`): what it shows now, its history included, goes
        into the log first, then its output from here on.  The pane's state,
        its capture and the start of pipe-pane are one tmux command, so no
        output falls between them (tmux runs a command list without
        processing pane output in between); the pipe then holds its output
        back until the capture is in the log."""
        ready = raw_log.with_name("adopt.ready")
        gate = (f"while [ ! -e {shlex.quote(str(ready))} ]; do sleep 0.02; done; rm -f {shlex.quote(str(ready))}; "
                f"exec cat >> {shlex.quote(str(raw_log))}")
        out = await self.run("display-message", "-p", "-t", pane_id, PANE_FORMAT, ";",
                             "capture-pane", "-p", "-e", "-S", "-", "-t", pane_id, ";",
                             "pipe-pane", "-O", "-t", pane_id, gate)
        head, _, text = out.partition("\n")
        info = parse_pane_info(head, pane_id)
        rows = text.rstrip("\n").split("\n")
        # Rows up to the cursor's; below it the screen is empty.
        keep = len(rows) - info.height + info.cursor_y + 1 if len(rows) >= info.height else info.cursor_y + 1
        rows = rows[:max(1, keep)]
        data = "\r\n".join(rows) + "\r" + (f"\x1b[{info.cursor_x}C" if info.cursor_x else "")
        if info.alternate_on:
            data = "\x1b[?1049h" + data
        with open(raw_log, "ab") as f:
            f.write(data.encode())
        ready.touch()
        self.register(name, pane_id, window)

    async def server_pid(self) -> int | None:
        """The tmux server's pid: pane ids restart at %0 with a new server,
        so a recorded one is only good for the server it came from."""
        out = await self.run("display-message", "-p", "#{pid}", check=False)
        return int(out.strip()) if out.strip().isdigit() else None

    async def pane_path(self, pane_id: str) -> str:
        return (await self.run("display-message", "-p", "-t", pane_id, "#{pane_current_path}", check=False)).strip()

    async def list_panes(self) -> list[PaneRow]:
        out = await self.run("list-panes", "-a", "-F",
                             "#{pane_id}\t#{session_name}\t#{pane_index}\t#{pane_active}\t#{pane_pipe}\t#{pane_dead}",
                             check=False)
        rows = []
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) == 6 and parts[2].isdigit():
                rows.append(PaneRow(parts[0], parts[1], int(parts[2]), parts[3] == "1", parts[4] == "1",
                                    parts[5] == "1"))
        return rows

    async def pane_table(self) -> dict[str, tuple[PaneInfo, bool]]:
        """Every pane's PaneInfo and whether its window's size is pinned, in
        one call: a listing asked each pane on its own, several tmux
        processes per live session (1.1 s for 20 of them on one test machine)."""
        out = await self.run("list-panes", "-a", "-F", "#{pane_id}\t#{window-size}\t" + PANE_FORMAT, check=False)
        table = {}
        for line in out.splitlines():
            pane_id, size, rest = (line.split("\t", 2) + ["", ""])[:3]
            try:
                table[pane_id] = (parse_pane_info(rest, pane_id), size == "manual")
            except TmuxError:
                continue  # answered oddly: the per-pane calls will say so
        return table

    async def client_table(self) -> dict[str, tuple[int, int | None]]:
        """Per tmux session (a window here): how many clients are attached,
        and the epoch seconds of their latest input."""
        out = await self.run("list-clients", "-F", "#{session_name}\t#{client_activity}", check=False)
        table: dict[str, tuple[int, int | None]] = {}
        for line in out.splitlines():
            window, _, activity = line.partition("\t")
            n, last = table.get(window, (0, None))
            if activity.isdigit() and (last is None or int(activity) > last):
                last = int(activity)
            table[window] = (n + 1, last)
        return table

    async def scrub_global_environment(self) -> None:
        """A tmux server started by an older server process may hold that
        process's environment (its venv): clear it, so new sessions do not
        inherit it."""
        out = await self.run("show-environment", "-g", check=False)
        for line in out.splitlines():
            key, _, value = line.partition("=")
            if key in SERVER_ONLY_ENV:
                await self.run("set-environment", "-g", "-u", key, check=False)
            elif key == "PATH" and value != session_environ().get("PATH", value):
                await self.run("set-environment", "-g", "PATH", session_environ()["PATH"], check=False)

    async def has_session(self, name: str) -> bool:
        """Does the session's pane (or, not yet registered, the tmux session
        of that name) exist?"""
        if (pane_id := self.panes.get(name)) is not None:
            out = await self.run("display-message", "-p", "-t", pane_id, "#{pane_id}", check=False)
            return out.strip() == pane_id
        proc = await asyncio.create_subprocess_exec(
            "tmux", "-L", self.socket, "has-session", "-t", self.target(name),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        return await proc.wait() == 0

    async def list_sessions(self) -> list[str]:
        out = await self.run("list-sessions", "-F", "#{session_name}", check=False)
        return [s for s in out.splitlines() if s]

    async def send_text(self, name: str, text: str, literal_tabs: bool = False) -> None:
        """Type `text`.  With `literal_tabs` (a line to run, not keystrokes),
        a tab goes in as a character (ctrl+v first: readline's quoted insert,
        the terminal's literal-next), not as completion."""
        pieces = text.split("\t") if literal_tabs else [text]
        for n, piece in enumerate(pieces):
            if n:
                await self.run("send-keys", "-t", self.pane(name), "C-v", "Tab")
            await self._type(name, piece)

    async def _type(self, name: str, text: str) -> None:
        # tmux refuses over-long commands, so type long text in pieces.
        for i in range(0, len(text), SEND_CHUNK):
            chunk = text[i : i + SEND_CHUNK]
            # An argument ending in ";" is tmux's own command separator
            # (and "\;" an escaped one), even after -l --: the semicolons
            # would be lost.  They go as key codes instead.
            body = chunk.rstrip(";")
            if body:
                await self.run("send-keys", "-t", self.pane(name), "-l", "--", body)
            if len(body) < len(chunk):
                await self.run("send-keys", "-t", self.pane(name), "-H", *["3b"] * (len(chunk) - len(body)))

    async def paste(self, name: str, text: str) -> None:
        """Paste `text` as a terminal would, marked as a bracketed paste
        when the program asked for that, so it is taken as one block rather
        than typed keystrokes (no auto-indentation, no line-by-line
        execution)."""
        buffer = f"docent-terminal-mcp-{name}"
        await self.run("load-buffer", "-b", buffer, "-", input=text.encode())
        await self.run("paste-buffer", "-p", "-d", "-b", buffer, "-t", self.pane(name))

    async def send_keys(self, name: str, keys: list[str]) -> None:
        # ";" as a key would be tmux's command separator: sent as its code.
        while keys:
            if keys[0] == ";":
                await self.run("send-keys", "-t", self.pane(name), "-H", "3b")
                keys = keys[1:]
                continue
            n = keys.index(";") if ";" in keys else len(keys)
            await self.run("send-keys", "-t", self.pane(name), "--", *keys[:n])
            keys = keys[n:]

    async def capture(self, name: str) -> list[str]:
        """The visible screen, one string per screen row, exactly as shown:
        a wrapped line stays split across rows."""
        out = await self.run("capture-pane", "-p", "-t", self.pane(name))
        # tmux keeps the tab a program moved the cursor with: cells, here.
        width = (await self.pane_info(name)).width
        return [expand_tabs(line, width).rstrip() for line in out.removesuffix("\n").split("\n")]

    async def capture_row(self, name: str, row: int) -> str:
        """One screen row, as shown."""
        out = await self.run("capture-pane", "-p", "-t", self.pane(name), "-S", str(row), "-E", str(row), check=False)
        return expand_tabs(out.rstrip("\n"), (await self.pane_info(name)).width)

    async def pane_info(self, name: str) -> PaneInfo:
        return await self.pane_info_of(self.pane(name), name)

    async def pane_info_of(self, target: str, name: str | None = None) -> PaneInfo:
        out = (await self.run("display-message", "-p", "-t", target, PANE_FORMAT)).rstrip("\n")
        return parse_pane_info(out, name or target)

    async def last_client_activity(self, name: str) -> float | None:
        """Epoch seconds of the most recent input from any attached client."""
        out = await self.run("list-clients", "-t", self.wtarget(name), "-F", "#{client_activity}", check=False)
        times = [int(t) for t in out.split() if t.isdigit()]
        return max(times) if times else None

    async def set_size(self, name: str, cols: int | None, rows: int | None, fixed: bool) -> None:
        """Pin the window at cols x rows (attaching clients no longer resize
        it), or let it follow the attached clients again."""
        if fixed:
            await self.run("set-option", "-w", "-t", self.wtarget(name), "window-size", "manual")
            args = ["resize-window", "-t", self.wtarget(name)]
            if cols:
                args += ["-x", str(cols)]
            if rows:
                args += ["-y", str(rows)]
            await self.run(*args)
        else:
            # resize-window sets window-size to manual itself, so it goes
            # first and the unset last.
            await self.run("resize-window", "-A", "-t", self.wtarget(name), check=False)
            await self.run("set-option", "-w", "-u", "-t", self.wtarget(name), "window-size")

    async def size_fixed(self, name: str) -> bool:
        out = await self.run("show-options", "-w", "-v", "-t", self.wtarget(name), "window-size", check=False)
        return out.strip() == "manual"

    async def client_count(self, name: str) -> int:
        out = await self.run("list-clients", "-t", self.wtarget(name), "-F", "x", check=False)
        return len(out.split())

    async def in_mode(self, name: str) -> bool:
        """Is the pane in a mode (copy mode: a human scrolled back)?  Keys
        sent then go to the mode, not the program."""
        out = await self.run("display-message", "-p", "-t", self.pane(name), "#{pane_in_mode}", check=False)
        return out.strip() == "1"

    async def leave_mode(self, name: str) -> None:
        await self.run("send-keys", "-t", self.pane(name), "-X", "cancel", check=False)

    async def kill_session(self, name: str) -> None:
        """End the session: its pane only, when its window has others; else
        the tmux session."""
        others = [p for p in await self.list_panes()
                  if p.window == self.window(name) and p.pane_id != self.panes.get(name)]
        if others and name in self.panes:
            await self.run("kill-pane", "-t", self.panes[name], check=False)
        else:
            await self.run("kill-session", "-t", self.wtarget(name), check=False)
        self.forget(name)


# Where a sleeping process waits, as ps reports it (wchan).
TTY_READ_WAITS = ("wait_woken", "n_tty_read")  # blocked reading the terminal
POLL_WAITS = ("poll_schedule_timeout", "do_sys_poll", "do_select", "core_sys_select",
              "ep_poll", "do_epoll_wait")  # an input loop (readline, gdb, python)
# Waiting on another process: a shell for its child, a pipeline's reader
# or writer for the other end (`gdb ... | tail`: tail decides nothing).
NEUTRAL_WAITS = ("do_wait", "anon_pipe_read", "anon_pipe_write", "pipe_read", "pipe_write", "pipe_wait")

# The syscalls an input loop waits in, by machine: their fd sets say whether
# the terminal is among what it waits for.
_WAIT_SYSCALLS = {
    "x86_64": {7: "poll", 271: "poll", 23: "select", 270: "select", 232: "epoll", 281: "epoll", 441: "epoll",
               0: "read", 17: "read", 19: "read", 45: "read", 47: "read"},  # read pread64 readv recvfrom recvmsg
    "aarch64": {73: "poll", 72: "select", 22: "epoll", 441: "epoll",
                63: "read", 67: "read", 65: "read", 207: "read", 212: "read"},
}
# restart_syscall: an interrupted call resumed (gdb under gdb, after ctrl+c
# and `continue`, still in its poll) -- the number says nothing of what it
# waits in: no evidence either way.
_RESTART_SYSCALL = {"x86_64": 219, "aarch64": 128}
_MAX_FDS = 65536


def _read_mem(pid: str, addr: int, size: int) -> bytes | None:
    try:
        with open(f"/proc/{pid}/mem", "rb") as f:
            f.seek(addr)
            return f.read(size)
    except (OSError, ValueError, OverflowError):
        return None


def _epoll_targets(pid: str, fd: int) -> set[int] | None:
    try:
        with open(f"/proc/{pid}/fdinfo/{fd}") as f:
            return {int(m.group(1)) for m in re.finditer(r"^tfd:\s*(\d+)", f.read(), re.M)}
    except OSError:
        return None


def waits_on_tty(pid: str, tty: str) -> bool | None:
    """Is process `pid`, sleeping in an input loop or a read, waiting for
    the terminal `tty`?  Read from the fd set of the call each of its threads waits in
    (/proc/PID/task/TID/syscall, the set from its memory; an epoll set from
    fdinfo).  A redirected `make -j check`: make waits in pselect for its
    jobs, DejaGnu's expect in epoll for its gdb's pty -- neither on the
    terminal.  None when it cannot be told (another machine, memory not
    readable: ptrace_scope)."""
    try:
        links = {int(fd): os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
        tasks = os.listdir(f"/proc/{pid}/task")
    except (OSError, ValueError):
        return None
    tty_fds = {fd for fd, link in links.items() if link in (tty, "/dev/tty")}
    epolls = {fd for fd, link in links.items() if link == "anon_inode:[eventpoll]"}
    if not tty_fds and not epolls:
        return False  # no way to the terminal at all (stdin from /dev/null, output to a file)
    calls = _WAIT_SYSCALLS.get(platform.machine())
    if calls is None:
        return None
    seen = False
    elsewhere = False  # a thread blocked in some other call: connect(), a futex, nanosleep
    unknown = False  # a thread whose call says nothing (restart_syscall): no evidence either way
    for tid in tasks:
        try:
            with open(f"/proc/{pid}/task/{tid}/syscall") as f:
                fields = f.read().split()
        except OSError:
            return None
        if not fields or not fields[0].isdigit():
            continue  # running
        if int(fields[0]) == _RESTART_SYSCALL.get(platform.machine()):
            unknown = True
            continue
        if (kind := calls.get(int(fields[0]))) is None:
            # Blocked in a call that reads nothing: connect() to an unroutable
            # host sleeps in wait_woken as a terminal read does, and read as
            # "waiting for input" (a test wedged in connect, called fine by
            # the reply).
            elsewhere = True
            continue
        args = [int(a, 16) for a in fields[1:7]]
        count = args[1] if kind == "poll" else args[0]  # poll(fds, nfds), select(nfds, readfds, ...)
        if kind in ("poll", "select") and not 0 < count <= _MAX_FDS:
            continue
        if kind == "read":  # read(fd, ...): a pty master read sleeps as a terminal read does
            fds = {args[0]}
        elif kind == "epoll":
            fds = _epoll_targets(pid, args[0])
        elif kind == "poll":  # struct pollfd {int fd; short events; short revents;}
            raw = _read_mem(pid, args[0], 8 * args[1])
            fds = None if raw is None else {
                int.from_bytes(raw[i:i + 4], "little", signed=True) for i in range(0, len(raw) - 7, 8)}
        else:  # select: the read set, a bitmap of nfds bits
            raw = _read_mem(pid, args[1], (args[0] + 7) // 8) if args[1] else b""
            fds = None if raw is None else {i for i in range(args[0]) if raw[i // 8] >> (i % 8) & 1}
        if fds is None:
            return None
        seen = True
        for fd in fds:
            if fd in tty_fds:
                return True
            if fd in epolls and (inner := _epoll_targets(pid, fd)) is not None and inner & tty_fds:
                return True
    # (A thread on the terminal answered True above.)  Not on the terminal
    # only when every waiting thread is known to wait on something else.
    if unknown:
        return None
    return False if seen or elsewhere else None
SHELLS = ("bash", "sh", "dash", "zsh", "ksh", "fish")  # a shell waiting for its child: the child decides


# Programs whose own processes are elsewhere: on another host, or (openshell
# sandbox exec/connect) in a sandbox whose processes cannot be seen from here.
REMOTE_CLIENTS = ("ssh", "mosh-client", "telnet", "sshpass", "openshell")
# Container clients are such programs only when they hold a terminal on a
# container (podman exec -it box): `podman build` works here, in view.
CONTAINER_CLIENTS = ("podman", "docker", "nerdctl", "kubectl")
CARRIER_NAMES = REMOTE_CLIENTS + CONTAINER_CLIENTS
_CONTAINER_ENTRIES = {"exec", "attach", "run", "start", "debug"}


def container_entry(argv: list[str]) -> str | None:
    """The subcommand of a container client's command line that puts this
    terminal on a container -- `exec -it`, `run -it`, `attach` -- else None."""
    args = argv[1:]
    sub = next((a for a in args if a in _CONTAINER_ENTRIES), None)
    if sub is None:
        return None
    if sub == "attach":
        return sub
    options = [a for a in args[args.index(sub) + 1:] if a.startswith("-")]
    interactive = any(a in ("--interactive", "--tty", "--stdin") or (not a.startswith("--") and set(a[1:]) & set("it"))
                      for a in options)
    return sub if interactive else None


def stdin_is_terminal(pid: int) -> bool:
    """Is the process's standard input a terminal?  (True when /proc cannot
    tell: the old answer.)"""
    try:
        target = os.readlink(f"/proc/{pid}/fd/0")
    except OSError:
        return True
    return target.startswith("/dev/pts/") or target.startswith("/dev/tty")


def ssh_one_shot(argv: list[str]) -> bool:
    """Is this `ssh [options] host command...` without -t: one remote
    command, run and done -- not a login?  It changes no "where" (the next
    line still runs here), and while it runs it is a command at work, not
    a remote shell (an `ssh -o BatchMode=yes host 'set -e;
    ...'` read "this session is on host" and "waiting for input")."""
    if not argv:
        return False
    prog = argv[0].rsplit("/", 1)[-1]
    if prog in ("bash", "sh", "dash", "zsh") and len(argv) > 1 and not argv[1].startswith("-"):
        return ssh_one_shot(argv[1:])  # a wrapper script named ssh
    if prog not in ("ssh", "slogin"):
        return False
    args, i, tty = argv[1:], 0, False
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if not a.startswith("-") or a == "-":
            break
        letters = a[1:]
        for j, c in enumerate(letters):
            if c == "t":
                tty = True
            if c in _SSH_VALUE_OPTS:
                if not letters[j + 1:]:
                    i += 1
                break
        i += 1
    return i + 1 < len(args) and not tty


def is_carrier(pid: int | str, comm: str) -> bool:
    """Is this foreground process only the near end of a program that runs
    elsewhere (another host, a sandbox, a container), out of ps's sight?"""
    comm = comm.strip()
    if comm in ("ssh", "slogin") and (info := proc_info(int(pid))) is not None and ssh_one_shot(info.argv):
        return False
    if comm in REMOTE_CLIENTS:
        # Only one holding the terminal: an ssh a script runs with its
        # input piped (git, rsync, a harness: `ssh host cmd`) carries no
        # terminal, and its waits are its own -- taken for a remote shell,
        # it made a busy script "waiting for input".
        return stdin_is_terminal(int(pid))
    if comm in CONTAINER_CLIENTS:
        info = proc_info(int(pid))
        return info is not None and container_entry(info.argv) is not None
    return False


async def foreground_state(tty: str, shell_pid: int = 0) -> str | None:
    """What the terminal's foreground processes are doing: "read" (one is
    blocked reading the terminal -- a question), "poll" (one waits in an
    input loop), "shell" (the session's own shell is at its prompt: whatever
    ran has finished), "busy" (none is waiting for input: running, sleeping
    on a timer as a build or `sleep` does, or between processes), or None
    when ps cannot tell.  The session's own shell in readline, at its prompt, is
    not "waiting for input" in this sense: the command it ran has finished,
    and its markers say so.  (The shell blocked in `read` is: a question.)"""
    if not tty.startswith("/dev/"):
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-t", tty.removeprefix("/dev/"), "-o", "pid=,stat=,wchan:40=,comm=",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError:
        return None
    state = None
    sure = None  # as `state`, from processes known to wait on this terminal
    elsewhere = False  # one waits on other fds: a child's pty, a socket, its jobs
    for line in out.decode(errors="replace").splitlines():
        pid, stat, wchan, comm = (line.split(None, 3) + ["", "", ""])[:4]
        if "+" not in stat:
            continue  # not in the foreground process group
        # (A shell that exec'd a program -- `exec python3` -- is that program.)
        if pid == str(shell_pid) and wchan.startswith(POLL_WAITS) and comm.strip() in SHELLS:
            return "shell"  # the session's own shell at its prompt (readline): not a question
        if comm.strip() in ("ssh", "slogin") and (one := proc_info(int(pid))) is not None \
                and ssh_one_shot(one.argv):
            # One remote command: at work -- unless ssh itself asks
            # something on the terminal (a password, a host key).
            if wchan.startswith(TTY_READ_WAITS):
                state = sure = "read"
            else:
                elsewhere = True
            continue
        if is_carrier(pid, comm):
            # A carrier: what it waits in says nothing of the program at
            # the other end (ssh polls; openshell's threads sleep on a
            # futex, "busy" for ever).  An input loop, as far as can be
            # told: the text and the pauses decide.
            state = state or "poll"
            continue
        if stat[0] in "RD":
            return "busy"
        if wchan.startswith(TTY_READ_WAITS) or wchan.startswith(POLL_WAITS):
            kind = "read" if wchan.startswith(TTY_READ_WAITS) else "poll"
            on_tty = waits_on_tty(pid, tty)
            if on_tty is False:
                # Reading or polling something else: its child's pty
                # (DejaGnu's expect), a socket, its jobs (make).
                elsewhere = True
                continue
            state = "read" if kind == "read" else state or "poll"
            if on_tty:
                sure = "read" if kind == "read" else sure or "poll"
        elif not wchan.startswith(NEUTRAL_WAITS) and wchan not in ("", "-", "0"):
            # Sleeping on something else: a timer, a lock, its child.  Busy
            # -- unless another process of the job waits on the terminal:
            # Claude Code with an MCP server started through `uv run`,
            # whose uv sleeps on a futex while its child serves.
            elsewhere = True
    if proc.returncode not in (0, 1):  # 1: no processes matched
        return None
    if elsewhere:
        # Working (a testsuite, a build) -- unless another process of the
        # job is known to wait on the terminal: a program at its input box
        # whose helpers wait on sockets or sleep (Claude Code and its MCP
        # servers).
        return sure or "busy"
    # Nothing waiting for input -- including a moment with no foreground
    # process at all, while the shell takes the terminal back.
    return state or "busy"


TMUX_CONFIG = """\
# Generated by docent-terminal-mcp.  The user's own configuration is read first so an
# attached human keeps their key bindings.
source-file -q ~/.tmux.conf
set -g history-limit 50000
set -g remain-on-exit on
"""


@dataclass
class Process:
    pid: int
    pgid: int
    command: str


@dataclass
class ProcInfo:
    pid: int
    ppid: int
    comm: str
    uid: int
    argv: list[str]


def proc_info(pid: int) -> ProcInfo | None:
    """A process's parent, name, real uid and arguments, from /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        with open(f"/proc/{pid}/status") as f:
            uid = next(int(line.split()[1]) for line in f if line.startswith("Uid:"))
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except (OSError, StopIteration, ValueError):
        return None
    # comm is in parentheses and may itself contain ") ": split at the last one.
    head, _, rest = stat.rpartition(") ")
    fields = rest.split()
    return ProcInfo(pid, int(fields[1]), head.partition("(")[2], uid, argv) if len(fields) > 1 else None


def process_chain(pid: int, stop: int, limit: int = 64) -> list[ProcInfo]:
    """`pid` and its ancestors, up to but not including `stop` (the
    session's own shell or program): innermost first."""
    chain = []
    while pid > 1 and pid != stop and len(chain) < limit:
        info = proc_info(pid)
        if info is None:
            break
        chain.append(info)
        pid = info.ppid
    return chain


# ssh's options that take a value (the destination is the first argument
# that is neither an option nor one of these values).
_SSH_VALUE_OPTS = set("BbcDEeFIiJLlmOoPpQRSWw")


def remote_destination(argv: list[str]) -> tuple[str, str | None] | None:
    """(host, user) from an ssh, telnet or mosh command line."""
    if not argv:
        return None
    prog = argv[0].rsplit("/", 1)[-1]
    args = argv[1:]
    user = None
    if prog in ("bash", "sh", "dash", "zsh", "perl") or prog.startswith("python"):
        # A wrapper script named ssh: "/bin/bash /usr/local/bin/ssh host".
        return remote_destination(args) if args and not args[0].startswith("-") else None
    if prog == "sshpass":  # sshpass [-p x | -f file | -e] ssh ...
        while args and args[0].startswith("-"):
            args = args[2:] if args[0] in ("-p", "-f", "-d") else args[1:]
        return remote_destination(args)
    if prog == "openshell":
        # openshell sandbox exec -n NAME ... / openshell sandbox connect NAME
        for i, a in enumerate(args):
            if a in ("-n", "--name") and i + 1 < len(args):
                return args[i + 1], None
            if a.startswith("--name="):
                return a.split("=", 1)[1], None
        if "connect" in args:
            rest = [a for a in args[args.index("connect") + 1:] if not a.startswith("-")]
            return (rest[0], None) if rest else None
        return None
    if prog in CONTAINER_CLIENTS:
        # podman exec [options] CONTAINER cmd...; podman run --name NAME ... IMAGE
        sub = container_entry(argv)
        if sub is None:
            return None
        rest = args[args.index(sub) + 1:]
        for i, a in enumerate(rest):
            if a == "--name" and i + 1 < len(rest):
                return rest[i + 1], None
            if a.startswith("--name="):
                return a.split("=", 1)[1], None
        valued = {"-e", "--env", "-u", "--user", "-w", "--workdir", "--env-file", "--detach-keys", "-c", "--container", "-n", "--namespace"}
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--":
                i += 1
                break
            if not a.startswith("-"):
                break
            i += 2 if a in valued else 1
        return (rest[i], None) if i < len(rest) and sub in ("exec", "attach", "start") else ("container", None)
    if prog in ("ssh", "slogin"):
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                i += 1
                break
            if not a.startswith("-") or a == "-":
                break
            letters = a[1:]
            for j, c in enumerate(letters):
                if c in _SSH_VALUE_OPTS:
                    value = letters[j + 1:] or (args[i + 1] if i + 1 < len(args) else "")
                    if not letters[j + 1:]:
                        i += 1
                    if c == "l":
                        user = value
                    elif c == "o" and value.lower().startswith("user="):
                        user = value.split("=", 1)[1]
                    break
            i += 1
        if i >= len(args):
            return None
        dest = args[i]
        if dest.startswith("ssh://"):
            dest = dest[6:].split("/", 1)[0]
            dest = dest.rsplit(":", 1)[0] if dest.count(":") == 1 else dest
        if "@" in dest:
            user, dest = dest.rsplit("@", 1)
        return (dest, user) if dest else None
    if prog in ("telnet", "mosh-client"):
        rest = [a for a in args if not a.startswith("-")]
        return (rest[0], None) if rest else None
    return None


def process_tree_summary(roots: list[int]) -> tuple[int, tuple[int, float, str]] | None:
    """How many processes run under `roots` (them included), and the newest
    of them: (count, (pid, seconds since it started, its command line)).
    For a long job's "still running": a testsuite moving on starts new
    processes; a hang shows its newest one minutes old.  None when /proc
    cannot tell."""
    try:
        hz = os.sysconf("SC_CLK_TCK")
        uptime = float(open("/proc/uptime").read().split()[0])
        children: dict[int, list[int]] = {}
        started: dict[int, float] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                stat = open(f"/proc/{entry}/stat").read()
            except OSError:
                continue
            fields = stat[stat.rindex(")") + 2:].split()
            pid = int(entry)
            children.setdefault(int(fields[1]), []).append(pid)
            started[pid] = int(fields[19]) / hz
    except (OSError, ValueError):
        return None
    seen, todo = set(), [r for r in roots if r in started]
    while todo:
        pid = todo.pop()
        if pid not in seen:
            seen.add(pid)
            todo.extend(children.get(pid, []))
    if not seen:
        return None
    newest = max(seen, key=lambda q: (started[q], q))  # (one clock tick apart: the later pid)
    try:
        argv = open(f"/proc/{newest}/cmdline", "rb").read().split(b"\0")
        command = " ".join(" ".join(a.decode(errors="replace") for a in argv if a).split())
    except OSError:
        command = "?"
    return len(seen), (newest, max(uptime - started[newest], 0.0), command)


async def background_processes(tty: str, shell_pid: int) -> list[Process]:
    """The job leaders on the terminal outside its foreground group, but
    not the session's own shell: jobs started with `&` (`nohup make &`)."""
    if not tty.startswith("/dev/"):
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-t", tty.removeprefix("/dev/"), "-o", "pid=,pgid=,tpgid=,comm=",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError:
        return []
    found = []
    for line in out.decode(errors="replace").splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and all(f.lstrip("-").isdigit() for f in fields[:3]):
            pid, pgid, tpgid = map(int, fields[:3])
            if pid == pgid and pgid != tpgid and pid != shell_pid:
                found.append(Process(pid, pgid, fields[3].strip()))
    return sorted(found, key=lambda p: p.pid)


async def foreground_table() -> dict[str, list[Process]]:
    """foreground_processes for every terminal at once (one ps), keyed by
    tty path: for a listing of many sessions."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-e", "-o", "tty=,pid=,pgid=,tpgid=,comm=",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError:
        return {}
    table: dict[str, list[Process]] = {}
    for line in out.decode(errors="replace").splitlines():
        fields = line.split(None, 4)
        if (len(fields) == 5 and fields[0] != "?" and all(f.lstrip("-").isdigit() for f in fields[1:4])
                and fields[2] == fields[3]):
            table.setdefault("/dev/" + fields[0], []).append(Process(int(fields[1]), int(fields[2]), fields[4].strip()))
    return {tty: sorted(procs, key=lambda p: p.pid) for tty, procs in table.items()}


async def foreground_processes(tty: str) -> list[Process]:
    """The processes of the terminal's foreground process group (whatever
    keys typed now would reach), by pid; empty when ps cannot tell."""
    if not tty.startswith("/dev/"):
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-t", tty.removeprefix("/dev/"), "-o", "pid=,pgid=,tpgid=,comm=",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError:
        return []
    found = []
    for line in out.decode(errors="replace").splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and all(f.lstrip("-").isdigit() for f in fields[:3]) and fields[1] == fields[2]:
            found.append(Process(int(fields[0]), int(fields[1]), fields[3].strip()))
    return sorted(found, key=lambda p: p.pid)
