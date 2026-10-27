"""MCP server exposing interactive terminal sessions backed by tmux."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import difflib
import functools
import inspect
import json
import os
import re
import secrets
import shlex
import shutil
import signal as signal_module
import stat as stat_module
import sys
import termios
import time
import typing
from typing import Annotated
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .guide import INDEX, INSTRUCTIONS, TOPICS
from .names import NAME, SHORT
from .session import ACTOR, DEFAULT_SETTLE_MS, QUESTION_ENDS, UntilWatch, LOCAL_HOST, LOCAL_USER, MAX_LINE_CHARS, Place, Region, Rule, Session, clip_line, integration_line, place_of_processes, prompt_regex, short_host
from . import retention, summary
from .tmux import SHELLS, TMUX_CONFIG, is_carrier, process_chain, process_tree_summary, Process, Tmux, TmuxError, background_processes, foreground_processes, foreground_state, foreground_table, normalize_key



class Typing(BaseModel):
    """Type the input as a person would, for a human watching (a demo)."""

    wps: float = Field(2.0, description="words per second (a word: five letters and a space)")
    mistakes: float = Field(0.1, description="share of words mistyped, then noticed and corrected")
    seed: int | None = Field(None, description="the same seed types the same way every time")


class Respond(BaseModel):
    """One expect rule: when `pattern` appears, do one of line / text / keys
    / stop -- or one of the first three with stop: answer, then stop."""

    pattern: str = Field(description="regex searched for in output not yet answered: what appears "
                         "from now on, and a question already waiting at the cursor")
    line: str | None = Field(None, description="type this and press Enter")
    text: str | None = Field(None, description="type this, no Enter")
    keys: list[str] | None = Field(None, description="press these keys, as in `send`")
    stop: bool = Field(False, description="end the script here and hand back (e.g. a password prompt); with "
                       "line, text or keys: answer first, then end")
    times: int = Field(1, description="how many times this rule may fire")
    secret: bool = Field(False, description="leave the sent value out of the transcript")


EXPECT_OUTCOMES = {
    "done": "the shell command finished",
    "ready": "the program is at its prompt",
    "until": "`until` matched",
    "finished": "the input was taken, and the program it went to exited",
    "stopped": "a stop rule matched; over to you",
    "unanswered": "output stopped at a question no rule answers; answer it with the `run` tool (the answer as input), or the `expect` tool again with a rule for it",
    "fullscreen": "a full-screen program took over; drive it with the `send` tool",
    "timeout": "timed out; the program may still be working (the `wait` tool, the `read` tool) or waiting for input",
    "exited": "the session's own program exited",
}


NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]{0,63}$")  # "+": panes of a window, "gdb+1"
NAME_CHAR = re.compile(r"[A-Za-z0-9_.+-]")
LIVE_LISTED = 12  # an unknown session's error lists the live ones up to this many, else only the caller's
BASHRC = Path(__file__).parent / "shell" / "bashrc"
ZSH_DIR = Path(__file__).parent / "shell" / "zsh"


class Manager:
    def __init__(self, home: Path, socket: str):
        self.home = home
        self.sessions_dir = home / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        config = home / "tmux.conf"
        config.write_text(TMUX_CONFIG)
        # The session's own commands (`NAME ssh HOST`), first on its PATH.
        self.bin_dir = home / "bin"
        self.bin_dir.mkdir(exist_ok=True)
        for command in (NAME, SHORT):
            shim = self.bin_dir / command
            shim.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -m docent_terminal_mcp.server \"$@\"\n")
            shim.chmod(0o755)
        self.tmux = Tmux(socket, config)
        self.sessions: dict[str, Session] = {}
        # name -> (pane id, window, tmux server pid) for every session on
        # disk, read once and kept up to date by create/split/adopt.
        self.index: dict[str, tuple[str | None, str, int | None]] | None = None
        self.pane_lock = asyncio.Lock()
        # Retention (retention.py): days a closed session's logs are kept
        # after they were last written or read.
        self.keep_days = retention.keep_days()
        self.closed_logs: dict[str, Session] = {}  # closed sessions read (get_logs)
        self.housekeeping: asyncio.Task | None = None
        # Sessions being made (create, split, adopt): their directory exists
        # before their pane is known, and they are not closed.
        self.creating: set[str] = set()

    def _read_index(self) -> dict:
        index = {}
        for d in self.sessions_dir.iterdir():
            try:
                meta = json.loads((d / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            index[d.name] = (meta.get("pane_id"), meta.get("window") or d.name, meta.get("tmux_pid"))
        return index

    def _started_by(self, name: str) -> str | None:
        if (s := self.sessions.get(name)) is not None:
            return s.meta.get("started_by")
        try:
            return json.loads((self.sessions_dir / name / "meta.json").read_text()).get("started_by")
        except (OSError, ValueError):
            return None

    def _record_pane(self, name: str, pane_id: str, window: str, tmux_pid: int | None) -> None:
        """Note a session's pane in its meta.json and the index."""
        path = self.sessions_dir / name / "meta.json"
        meta = json.loads(path.read_text())
        meta.update(pane_id=pane_id, window=window, tmux_pid=tmux_pid)
        path.write_text(json.dumps(meta, indent=1))
        if (s := self.sessions.get(name)) is not None:
            s.meta.update(pane_id=pane_id, window=window, tmux_pid=tmux_pid)
        self.index[name] = (pane_id, window, tmux_pid)

    async def sync_panes(self, window: str | None = None) -> list[str]:
        """Register each live session's pane (by id: see Tmux.panes) and
        adopt the panes that appeared in a session's window without us -- a
        human's split, or `tmux split-window` typed in the session -- in
        `window`, or in every window.  Returns the names of the adopted."""
        async with self.pane_lock:
            if self.index is None:
                self.index = self._read_index()
            rows = await self.tmux.list_panes()
            if not rows:
                return []
            spid = await self.tmux.server_pid()
            by_id = {r.pane_id: r for r in rows}
            windows: dict[str, list] = {}
            for r in rows:
                windows.setdefault(r.window, []).append(r)
            claimed = set()
            for name, (pane_id, win, tpid) in list(self.index.items()):
                r = by_id.get(pane_id) if pane_id else None
                if r is not None and tpid == spid:
                    # Followed by id wherever it went: a pane moved to another
                    # window (a client's break-pane into a session of
                    # its own) is the same session, with the same log.
                    claimed.add(pane_id)
                    if r.window != win:
                        self._record_pane(name, pane_id, r.window, spid)
                    self.tmux.register(name, pane_id, r.window)
            # Sessions from before pane ids were recorded: the logged pane of
            # the tmux session of their name.
            for name, (pane_id, win, tpid) in list(self.index.items()):
                if name in windows and not (pane_id and pane_id in claimed):
                    free = sorted((r for r in windows[name] if r.pane_id not in claimed),
                                  key=lambda r: (not r.piped, r.index))
                    if free and free[0].piped:
                        claimed.add(free[0].pane_id)
                        self._record_pane(name, free[0].pane_id, name, spid)
                        self.tmux.register(name, free[0].pane_id, name)
            adopted = []
            for win, panes in windows.items():
                if (window is not None and win != window) or win not in self.index:
                    continue  # not a window of ours
                for r in sorted(panes, key=lambda r: r.index):
                    if r.pane_id not in claimed and not r.dead:
                        adopted.append(await self._adopt(win, r.pane_id, spid))
                        claimed.add(r.pane_id)
            return adopted

    def _child_name(self, window: str) -> str:
        n = 1
        while (self.sessions_dir / f"{window}+{n}").exists():
            n += 1
        return f"{window}+{n}"

    async def _adopt(self, window: str, pane_id: str, spid: int | None) -> str:
        name = self._child_name(window)
        info = await self.tmux.pane_info_of(pane_id)
        directory = self.sessions_dir / name
        directory.mkdir(parents=True)
        meta = {
            "name": name, "command": info.current_command, "argv": None, "cwd": await self.tmux.pane_path(pane_id),
            "cols": info.width, "rows": info.height, "shell_integration": False, "prompt_pattern": None,
            "created": time.time(), "started_by": None, "adopted": True, "nonce": secrets.token_hex(6),
            "pane_id": pane_id, "window": window, "parent": window, "tmux_pid": spid,
            "host": retention.HOST, "tmux_socket": self.tmux.socket,
            "tmux_socket_path": self.tmux.expected_socket_path(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=1))
        raw = directory / "raw.log"
        raw.touch()
        (directory / "widths.jsonl").write_text(json.dumps({"offset": 0, "width": info.width}) + "\n")
        await self.tmux.adopt_pane(name, pane_id, window, raw)
        self.index[name] = (pane_id, window, spid)
        return name

    async def close_sessions(self, session: str, window: bool = False, force: bool = False) -> str:
        """session_close, and POST /api/sessions/NAME/close (where `force`
        lives: a close a person asked for -- a client's Close button, the
        CLI -- not one a model can choose: given `force`, a model in testing
        forced a watched session closed in 3 of 3 runs)."""
        await self.get(session)
        names = self.window_sessions(session) if window else [session]
        if session not in names:
            names.append(session)
        if not force:
            # Not a terminal someone is using: a person
            # attached, or another's command running.  Checked for every pane
            # before any is closed.
            refusals = []
            for name in names:
                s = await self.get(name)
                try:
                    if (await self.tmux.pane_info(name)).dead:
                        continue  # nothing runs in an exited pane
                except TmuxError:
                    continue
                if (humans := await self.tmux.client_count(name)) > 0:
                    refusals.append(f"session {name!r} has {humans} person{'s' if humans > 1 else ''} attached "
                                    "(watching or typing in it)")
                    continue
                by = s.meta.get("started_by")
                if by == ACTOR.get():
                    continue
                s.sync()
                running = [r for r in s.regions if r.kind == "command" and not r.done]
                jobs = await foreground_jobs(s)
                if running or jobs:
                    what = (f"`{(running[-1].command or '').partition(chr(10))[0][:60]}` (region {running[-1].id})"
                            if running else named(jobs))
                    whose = "a human's" if by in (None, "human") else f"{by}'s"
                    refusals.append(f"session {name!r} is {whose} and {what} is running there")
            if refusals:
                raise ToolError("not closed: " + "; ".join(refusals) + ".  Someone is using it: leave it to them, "
                                "or ask the user")
        out = []
        for name in names:
            s = await self.get(name)
            try:
                info = await self.tmux.pane_info(name)
                if info.dead:  # keep the exit status for session_list
                    s.meta["exit_status"] = info.dead_status
                    (s.dir / "meta.json").write_text(json.dumps(s.meta, indent=1))
            except TmuxError:
                pass
            s.sync()
            running = [r for r in s.regions if (r.kind == "command" and not r.done)
                       or (r.kind == "input" and r.status in ("running", "timeout", "until"))]
            started_by = s.meta.get("started_by")
            await self.tmux.kill_session(name)
            s.meta["closed"] = time.time()
            (s.dir / "meta.json").write_text(json.dumps(s.meta, indent=1))
            self.sessions.pop(name, None)
            s.close()
            whose = (f" (started by {started_by}, not you)" if started_by and started_by != ACTOR.get()
                     else " (a pane nobody started: adopted)" if s.meta.get("adopted") else "")
            ended = ("; it ended what was running: " + ", ".join(
                f"`{(r.command or '').partition(chr(10))[0][:60]}` (region {r.id})" for r in running[-3:])) if running else ""
            if retention.marked(s.dir, retention.ON_CLOSE_MARK):
                self.delete_logs(s.dir, "closed, marked delete-on-close")
                out.append(f"closed {name!r}{whose}{ended}; its logs were deleted (it was marked delete-on-close)")
            else:
                out.append(f"closed {name!r}{whose}{ended}; logs remain in {s.dir}")
        return "\n".join(out)

    # -- retention ---------------------------------------------------------

    LOG_ACTIONS = ("keep", "default", "delete_on_close", "delete")

    async def logs_action(self, session: str, action: str) -> str:
        """session_logs (and POST /api/sessions/NAME/logs): pin a session's
        logs, put them back under the usual rule, mark them to go on close,
        or delete them now."""
        if action not in self.LOG_ACTIONS:
            raise ToolError(f"action must be one of {', '.join(self.LOG_ACTIONS)}")
        directory = self.sessions_dir / session
        if not NAME_RE.match(session) or not (directory / "meta.json").exists():
            raise ToolError(f"no logs of a session named {session!r} (session_list names the sessions and closed "
                            "ones' NAME.N)")
        live = await self.live_names()  # (first: a live session of an older meta gets its host here)
        meta = json.loads((directory / "meta.json").read_text())
        here = retention.made_here(meta, self.home, self.tmux.socket)
        is_live = (live is None or session in live or await self.tmux.has_session(session)) if here else None
        whose = meta.get("started_by")
        whose = "" if whose in (None, ACTOR.get()) else f" (started by {whose})"
        if action == "keep":
            retention.set_mark(directory, retention.ON_CLOSE_MARK, False)
            retention.set_mark(directory, retention.KEEP_MARK, True)
            return f"the logs of {session!r}{whose} are kept: never deleted by the usual rule ({directory})"
        if action == "default":
            retention.set_mark(directory, retention.ON_CLOSE_MARK, False)
            retention.set_mark(directory, retention.KEEP_MARK, False)
            rule = (f"deleted once closed and unchanged for {self.keep_days:g} days" if self.keep_days > 0
                    else "kept (this server keeps every closed session)")
            return f"the logs of {session!r}{whose} follow the usual rule again: {rule}"
        if not here:
            raise ToolError(f"session {session!r} was made by a server on {meta.get('host', 'another host')} (tmux "
                            f"socket {meta.get('tmux_socket', retention.DEFAULT_SOCKET)!r}): only there can it be told "
                            "closed; delete its logs there")
        if action == "delete_on_close":
            retention.set_mark(directory, retention.KEEP_MARK, False)
            retention.set_mark(directory, retention.ON_CLOSE_MARK, True)
            if not is_live:
                self.delete_logs(directory, "closed, marked delete-on-close")
                return f"session {session!r}{whose} is closed: its logs were deleted"
            return f"the logs of {session!r}{whose} will be deleted when it closes"
        if is_live:
            raise ToolError(f"session {session!r} is live: its logs are still being written.  session_close it first, "
                            "or action=\"delete_on_close\"")
        self.delete_logs(directory, "deleted on request")
        return f"the logs of {session!r}{whose} were deleted ({directory})"


    async def live_names(self) -> set[str] | None:
        """The sessions with a pane in tmux now -- none when no tmux server
        answers on our socket; None when tmux answers oddly, and so nothing
        may be called closed."""
        if not await self.tmux.server_running():
            return set()
        await self.sync_panes()
        try:
            ids = set((await self.tmux.run("list-panes", "-a", "-F", "#{pane_id}")).split())
        except TmuxError:
            # A server up with no session left: tmux 3.7 says "no current
            # target" to list-panes -a (unless run inside tmux, whose $TMUX
            # gives it one) -- no pane, not an odd answer.  As None, nothing
            # was ever swept there.
            if not (await self.tmux.run("list-sessions", "-F", "#{session_name}", check=False)).strip():
                return set()
            return None
        live = {name for name, pane_id in self.tmux.panes.items() if pane_id in ids}
        self._stamp_legacy(live)
        return live

    def _stamp_legacy(self, live: set[str]) -> None:
        """Sessions made before 2026-10-05 have no "host" or "tmux_socket" in
        their meta: unless the state directory names the host, they counted
        as another server's -- their logs never swept, session_logs refused
        (f44-mesa-1: 31 of 57).  One live in this server's tmux is this
        server's: say so in its meta, which then holds when it closes."""
        for name in live:
            path = self.sessions_dir / name / "meta.json"
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if "host" in meta or "tmux_socket" in meta:
                continue
            meta.update({"host": retention.HOST, "tmux_socket": self.tmux.socket,
                         "tmux_socket_path": self.tmux.expected_socket_path()})
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=1))
            tmp.replace(path)

    def _due(self, directory: Path, live: set[str], marked_only: bool) -> str | None:
        """Why this directory's logs may go now, or None to keep them."""
        try:
            meta = json.loads((directory / "meta.json").read_text())
        except (OSError, ValueError):
            return None  # not a session's directory, or one half made
        if directory.name in live or not retention.made_here(meta, self.home, self.tmux.socket):
            return None
        if directory.name in self.creating or (
                meta.get("pane_id") is None and time.time() - (meta.get("created") or 0) < CREATING_GRACE_S):
            return None  # being made (here, or by another server process just now)
        if not live and meta.get("tmux_socket_path") != self.tmux.expected_socket_path():
            # No pane at all: no tmux server answers here.  Closed for sure
            # only if the session was made on this very socket path -- not
            # under another TMUX_TMPDIR or /tmp, whose server we cannot see.
            return None
        if retention.marked(directory, retention.ON_CLOSE_MARK):
            return "closed, marked delete-on-close"
        if marked_only or self.keep_days <= 0 or retention.marked(directory, retention.KEEP_MARK):
            return None
        age = time.time() - retention.newest_change(directory)
        if age < self.keep_days * 86400:
            return None
        return f"closed and unused -- neither written nor read -- for {age / 86400:.1f} days"

    def delete_logs(self, directory: Path, why: str) -> None:
        name = directory.name
        for cache in (self.sessions, self.closed_logs):
            if (loaded := cache.pop(name, None)) is not None:
                loaded.close()
        if self.index is not None:
            self.index.pop(name, None)
        self.tmux.forget(name)
        retention.remove(directory, why, self.keep_days)
        retention.log(f"deleted the logs of session {name!r} ({why}): {directory}")

    async def sweep(self, marked_only: bool = False, dry_run: bool = False) -> list[tuple[str, str]]:
        """Delete the logs that are due (all of them, or only those marked
        delete-on-close); returns (name, why) for each.  Each is checked
        again just before it goes -- its pane back, its name reused, a pin --
        with no await between that check and the deletion."""
        live = await self.live_names()
        if live is None:
            return []
        if not marked_only:
            retention.prune_tombstones(self.home)
        gone = []
        for directory in sorted(self.sessions_dir.iterdir()):
            if (why := self._due(directory, live, marked_only)) is None:
                continue
            if dry_run:
                gone.append((directory.name, why))
                continue
            live = await self.live_names()
            if live is None:
                return gone
            if await self.tmux.has_session(directory.name):
                continue
            if (why := self._due(directory, live, marked_only)) is not None:
                self.delete_logs(directory, why)
                gone.append((directory.name, why))
        return gone

    def start_housekeeping(self) -> None:
        """The sweep: at once, then hourly, and every minute for sessions
        marked delete-on-close that closed by themselves (their program
        exited).  Started by the first call, or the shared server's start."""
        if self.housekeeping is not None:
            return

        async def loop():
            last_full = 0.0
            while True:
                full = time.time() - last_full >= retention.SWEEP_EVERY_S
                try:
                    await self.sweep(marked_only=not full)
                except Exception as e:  # never let a sweep take the server down
                    retention.log(f"sweep failed: {e!r}")
                if full:
                    last_full = time.time()
                for s in list(self.sessions.values()):  # save_to files of commands still running
                    if s.saves:
                        try:
                            s.sync()
                            s.write_saves()
                        except Exception as e:
                            retention.log(f"save_to refresh failed for {s.name}: {e!r}")
                await asyncio.sleep(retention.ON_CLOSE_EVERY_S)

        self.housekeeping = asyncio.get_running_loop().create_task(loop())

    async def describe_sessions(self) -> list[dict]:
        """Every session this server knows, live or closed, as data: for the
        session_list tool and the /api/sessions endpoint."""
        await self.sync_panes()
        rows = {r.pane_id: r for r in await self.tmux.list_panes()}
        live = {name for name, pane_id in self.tmux.panes.items() if pane_id in rows}
        # Every live pane's details and every client in two calls, not
        # several per pane (the listing's cost was those, not the walk).
        panes, clients = (await self.tmux.pane_table(), await self.tmux.client_table()) if live else ({}, {})
        foreground = await foreground_table() if live else {}
        out = []
        for d in sorted(self.sessions_dir.iterdir()):
            if not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            raw = d / "raw.log"
            entry = {
                "name": d.name, "command": meta["command"], "cwd": meta.get("cwd"),
                "created": meta.get("created"), "started_by": meta.get("started_by"), "log_dir": str(d),
                "last_output": raw.stat().st_mtime if raw.exists() else None,
                "state": "live" if d.name in live else "closed",
                "exit_status": meta.get("exit_status"), "adopted": bool(meta.get("adopted")),
                "integrated": bool(meta.get("shell_integration")),
                "parent": meta.get("parent"),
                "agents_typed": self._agents_typed(d),
                "keep": retention.marked(d, retention.KEEP_MARK),
                "delete_on_close": retention.marked(d, retention.ON_CLOSE_MARK),
            }
            if entry["state"] == "closed":
                # (None: kept -- pinned, retention off, or another server's.)
                entry["expires"] = (retention.expires(d, self.keep_days)
                                    if retention.made_here(meta, self.home, self.tmux.socket) else None)
            if entry["state"] == "live":
                r = rows[self.tmux.panes[d.name]]
                entry.update(pane_id=r.pane_id, window=r.window, parent=meta.get("parent"), index=r.index,
                             active=r.active)
                try:
                    await self._describe_live(entry, panes.get(r.pane_id), clients.get(r.window, (0, None)), foreground)
                except (TmuxError, ValueError) as e:
                    # One pane answering oddly must not take the listing down.
                    entry.update(state="error", error=str(e))
            out.append(entry)
        return out

    def _agents_typed(self, directory: Path) -> dict[str, float]:
        """When each agent last typed into the session (epoch s): from the
        loaded session, else from its state.json."""
        if (loaded := self.sessions.get(directory.name)) is not None:
            return dict(loaded.last_typed)
        try:
            return dict(json.loads((directory / "state.json").read_text()).get("last_typed", {}))
        except (OSError, ValueError):
            return {}

    async def _describe_live(self, entry: dict, pane: tuple | None, clients: tuple[int, int | None],
                             foreground: dict) -> None:
        name = entry["name"]
        info, fixed = pane if pane is not None else (await self.tmux.pane_info(name), await self.tmux.size_fixed(name))
        if info.dead:
            entry.update(state="exited", exit_status=info.dead_status)
            return
        try:
            loaded = self.sessions.get(name)
            place = (await loaded.where(info, foreground) if loaded
                     else (await place_of_processes(info.tty, info.pid, foreground))[0])
        except (TmuxError, OSError):
            place = None
        if place is not None:
            entry["place"] = {"host": place.host, "user": place.user, "via": place.via, "label": place.label()}
        loaded = self.sessions.get(name)
        # (Only a session this server has loaded -- one an agent used since it
        # started -- can say whether its shell is integrated; loading one just
        # to list it upset a pane being adopted.)
        if loaded is not None:
            loaded.sync()
            if loaded.at_shell_prompt:
                entry["shell"] = "integrated"
            elif loaded.at_plain_shell_prompt():
                entry["shell"] = "plain"
        if loaded is not None and loaded.holder is not None:
            entry["typing_now"] = {"agent": loaded.holder[0], "tool": loaded.holder[1]}
        entry.update(
            human_input=clients[1],
            running=info.current_command, width=info.width, height=info.height,
            fixed=fixed,
            humans=clients[0],
            attach={"socket": self.tmux.socket,
                    "argv": self.tmux.attach_argv(name),
                    "command": self.tmux.attach_command(name)},
        )

    async def get(self, name: str) -> Session:
        directory = self.sessions_dir / name
        if not NAME_RE.match(name) or not (directory / "meta.json").exists():
            if NAME_RE.match(name) and (gone := retention.tombstone(self.home, name)) is not None:
                raise ToolError(f"no session named {name!r} any more: {retention.gone_text(gone)}")
            await self.sync_panes()
            live = sorted([n for n in list(self.tmux.panes) if await self.tmux.has_session(n)])
            if not live:
                which = "there are no live sessions"
            elif len(live) <= LIVE_LISTED:
                which = f"the live sessions are {', '.join(live)}"
            else:
                mine = [n for n in live if self._started_by(n) == ACTOR.get()]
                which = (f"of the {len(live)} live sessions, yours are {', '.join(mine)}" if mine else
                         f"none of the {len(live)} live sessions is yours") + " (session_list lists them all)"
            if name.isdigit():
                raise ToolError(
                    f"no session named {name!r}: a number like that is a process's ID in another tool (a shell "
                    f"tool's session ID, a PID), not a session here, and nothing here reaches that process: go on "
                    f"with the tool that started it, or start the program here (session_create, then run); {which}")
            # A live name with debris after it that no name has (qwen: 'tests\n</parameterparameter>').
            if near := [n for n in live if name.startswith(n) and not NAME_RE.fullmatch(name[:len(n) + 1])]:
                raise ToolError(f"no session named {name!r} -- did you mean {near[0]!r}? (what follows it cannot "
                                f"be part of a session's name)")
            # ... or before it (qwen under codex: '=build').
            if near := [n for n in live if name.endswith(n) and len(name) > len(n)
                        and not NAME_CHAR.match(name[-len(n) - 1])]:
                raise ToolError(f"no session named {name!r} -- did you mean {near[0]!r}? (what comes before it "
                                f"cannot be part of a session's name)")
            # A slip of the keys (qwen: 'gdbgdb' for 'gdbdbg'); the live ones are still named.
            if near := difflib.get_close_matches(name, live, 1, 0.8):
                raise ToolError(f"no session named {name!r} -- did you mean {near[0]!r}? {which}")
            raise ToolError(f"no session named {name!r}: {which} (session_create makes one)")
        if self.index is None or name not in self.index:
            self.index = self._read_index()
        adopted = await self.sync_panes(self.index[name][1])
        if adopted and (notes := REPLY_NOTES.get()) is not None:
            notes.append(f"[a new pane appeared in this session's window (a split) and is now session "
                         f"{', '.join(map(repr, adopted))}, logged from now on -- the `run` tool there, by that session name (a bash "
                         f"idle at its prompt gets the shell integration first); `session_split` makes such a pane "
                         f"integrated from the start]")
        session = self.sessions.get(name)
        if session is not None:
            await self._note_width(session)
            return session
        if not await self.tmux.has_session(name):
            meta = json.loads((directory / "meta.json").read_text())
            if meta.get("closed"):
                when = time.strftime("%H:%M:%S", time.localtime(meta["closed"]))
                raise ToolError(f"session {name!r} was closed at {when}: nothing can be typed there (session_create "
                                f"a new one); the `read`, `search` and `regions` tools still read its logs, in {directory}")
            if not await self.tmux.server_running():
                raise ToolError(
                    f"session {name!r} is not reachable: no tmux server answers on socket {self.tmux.socket!r} here. "
                    f"If this server runs with another /tmp than the one that started the session (a new sudo or "
                    f"login session with a private /tmp, a container), its tmux server is invisible from here: "
                    f"set TMUX_TMPDIR to a directory both see.  Its logs remain in {directory}")
            raise ToolError(f"session {name!r} no longer exists in tmux (its program exited and the pane was "
                            f"closed, or someone killed it); its logs remain in {directory}")
        session = Session(name, directory, self.tmux, json.loads((directory / "meta.json").read_text()))
        self.sessions[name] = session
        await self._note_width(session)
        return session

    async def get_logs(self, name: str) -> Session:
        """For read, search and regions: a live session, or a closed one's
        logs (read from its directory; nothing there needs tmux).  Either
        way the logs are in use: see retention.used."""
        directory = self.sessions_dir / name
        try:
            session = await self.get(name)
        except ToolError:
            if not NAME_RE.match(name) or not (directory / "meta.json").exists():
                raise
            live = await self.live_names()
            if live is None or name in live or await self.tmux.has_session(name):
                raise  # not closed: get() said why
            session = self.closed_logs.get(name)
            if session is None or session.dir != directory or not directory.exists():
                session = Session(name, directory, self.tmux, json.loads((directory / "meta.json").read_text()))
                self.closed_logs[name] = session
        retention.used(directory)
        return session

    async def _note_width(self, session: Session) -> None:
        """The pane's width now, recorded before the call reads the output
        that came since the last: a program redraws for a new width at once
        (readline after a resize), and that redraw read at the old width
        puts its cursor moves -- up a row, erase it -- on the wrong line."""
        try:
            info = await self.tmux.pane_info(session.name)
        except TmuxError:
            return
        if not info.dead:
            session.set_width(info.width)

    async def create(
        self, name: str | None, command: str | None, cwd: str | None,
        cols: int, rows: int, env: dict[str, str] | None, prompt_pattern: str | None,
        delete_logs_on_close: bool = False,
    ) -> Session:
        if name is None:
            n = 1
            while (self.sessions_dir / f"t{n}").exists():
                n += 1
            name = f"t{n}"
        if not NAME_RE.match(name):
            raise ToolError("session names may contain letters, digits, '_', '.', '-'")
        # Not closed while it is being made -- nor its earlier logs, about
        # to be archived (the sweep runs while this awaits tmux).
        self.creating.add(name)
        try:
            return await self._create_named(name, command, cwd, cols, rows, env, prompt_pattern,
                                            delete_logs_on_close)
        finally:
            self.creating.discard(name)

    async def _create_named(self, name, command, cwd, cols, rows, env, prompt_pattern,
                            delete_logs_on_close) -> Session:
        directory = self.sessions_dir / name
        archived = None
        if directory.exists():
            if await self.tmux.has_session(name):
                meta = json.loads((directory / "meta.json").read_text()) if (directory / "meta.json").exists() else {}
                by = meta.get("started_by")
                whose = "yours" if by == ACTOR.get() else f"started by {by}" if by else "open"
                raise ToolError(f"session {name!r} already exists ({whose}): to work in it, use it -- "
                                f"run/read/... with session={name!r}; for a session of your own, pick another name")
            # A closed session's name may be reused; keep its logs -- unless
            # they were to go when it closed.
            if (old := self.sessions.pop(name, None)) is not None:
                old.close()
            if retention.marked(directory, retention.ON_CLOSE_MARK):
                self.delete_logs(directory, "closed, marked delete-on-close (its name reused)")
            else:
                n = 1
                while (self.sessions_dir / f"{name}.{n}").exists():
                    n += 1
                archived = self.sessions_dir / f"{name}.{n}"
                directory.rename(archived)
        if prompt_pattern is not None:
            compile_pattern(prompt_pattern)
        shell = integrated_shell(command)
        integrated = shell is not None
        argv, shell_env = shell[1:] if shell else (shlex.split(command), {})
        if argv and argv[0] in (NAME, SHORT):
            # `NAME ssh host`, `NAME into ...` as the session's own program:
            # sessions' shells find NAME through their PATH; this one has none.
            argv = [str(self.bin_dir / NAME), *argv[1:]]
        directory.mkdir(parents=True)
        meta = {
            "name": name, "command": shell[0] if shell else command, "argv": argv,
            "cwd": cwd or os.getcwd(), "cols": cols, "rows": rows,
            "shell_integration": integrated, "prompt_pattern": prompt_pattern,
            "created": time.time(), "started_by": ACTOR.get(),
            # Carried by this session's shell-integration markers: markers
            # without it are program output (another session's log catted).
            "nonce": secrets.token_hex(6),
            # Whose tmux holds it: only that server may call it closed.
            "host": retention.HOST, "tmux_socket": self.tmux.socket,
            "tmux_socket_path": self.tmux.expected_socket_path(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=1))
        if delete_logs_on_close:  # before the first byte is logged
            retention.set_mark(directory, retention.ON_CLOSE_MARK, True)
        raw = directory / "raw.log"
        raw.touch()
        (directory / "widths.jsonl").write_text(json.dumps({"offset": 0, "width": cols}) + "\n")
        try:
            pane_id = await self.tmux.new_session(name, argv, meta["cwd"],
                                                  {**shell_env, "__DTMCP_BIN": str(self.bin_dir), **(env or {}), "__DTMCP_NONCE": meta["nonce"]}, cols, rows, raw)
        except TmuxError as e:
            raise ToolError(str(e)) from e
        if self.index is None:
            self.index = self._read_index()
        self._record_pane(name, pane_id, name, await self.tmux.server_pid())
        meta = json.loads((directory / "meta.json").read_text())
        session = Session(name, directory, self.tmux, meta)
        session.archived = archived
        self.sessions[name] = session
        return session


    async def split(self, parent: str, name: str | None, command: str | None, below: bool, size: str | None,
                    env: dict[str, str] | None, prompt_pattern: str | None) -> Session:
        """A new pane in `parent`'s window, as a session of its own."""
        await self.get(parent)
        window = self.tmux.window(parent)
        if not await self.tmux.has_session(parent):
            raise ToolError(f"session {parent!r} has no live pane to split")
        async with self.pane_lock:
            name = name or self._child_name(window)
            if not NAME_RE.match(name):
                raise ToolError("session names may contain letters, digits, '_', '.', '+', '-'")
            directory = self.sessions_dir / name
            if directory.exists():
                raise ToolError(f"a session named {name!r} exists or existed; pick another name (or none)")
            if prompt_pattern is not None:
                compile_pattern(prompt_pattern)
            shell = integrated_shell(command)
            integrated = shell is not None
            argv, shell_env = shell[1:] if shell else (shlex.split(command), {})
            parent_pane = self.tmux.pane(parent)
            spid = await self.tmux.server_pid()
            meta = {
                "name": name, "command": shell[0] if shell else command, "argv": argv,
                "cwd": await self.tmux.pane_path(parent_pane) or os.getcwd(), "cols": None, "rows": None,
                "shell_integration": integrated, "prompt_pattern": prompt_pattern,
                "created": time.time(), "started_by": ACTOR.get(), "nonce": secrets.token_hex(6),
                "pane_id": None, "window": window, "parent": window, "tmux_pid": spid,
                "host": retention.HOST, "tmux_socket": self.tmux.socket,
            "tmux_socket_path": self.tmux.expected_socket_path(),
            }
            directory.mkdir(parents=True)
            raw = directory / "raw.log"
            raw.touch()
            try:
                pane_id = await self.tmux.split(name, parent, argv, meta["cwd"],
                                                {**shell_env, "__DTMCP_BIN": str(self.bin_dir), **(env or {}), "__DTMCP_NONCE": meta["nonce"]}, below, size, raw)
            except TmuxError as e:
                (directory / "meta.json").write_text(json.dumps({**meta, "failed": str(e)}, indent=1))
                raise ToolError(str(e)) from e
            info = await self.tmux.pane_info_of(pane_id)
            meta.update(pane_id=pane_id, cols=info.width, rows=info.height)
            (directory / "meta.json").write_text(json.dumps(meta, indent=1))
            (directory / "widths.jsonl").write_text(json.dumps({"offset": 0, "width": info.width}) + "\n")
            self.index[name] = (pane_id, window, spid)
            await self.tmux.start_logging(pane_id, raw, f"docent-terminal-mcp-ready-{name}")
        session = Session(name, directory, self.tmux, meta)
        self.sessions[name] = session
        return session

    def window_sessions(self, name: str) -> list[str]:
        """The live sessions sharing `name`'s window, `name` included."""
        window = self.tmux.window(name)
        return sorted(n for n, w in self.tmux.windows.items() if w == window and n in self.tmux.panes)


PLAIN_BASH_OPTIONS = {"-i", "-l", "--login"}


def integrated_shell(command: str | None) -> tuple[str, list[str], dict] | None:
    """The shell a session gets with the integration: ("bash" | "zsh",
    argv, extra environment), or None for another command.  No command,
    or a plain interactive bash or zsh ("zsh -l"), asks for a shell."""
    if command is None or plain_bash(command):
        return "bash", ["bash", "--rcfile", str(BASHRC), "-i"], {}
    if plain_shell(command, "zsh"):
        # zsh reads its startup files from ZDOTDIR: ours read the user's.
        user = os.environ.get("ZDOTDIR", "")
        return "zsh", ["zsh", "-l", "-i"], {"ZDOTDIR": str(ZSH_DIR), "__DTMCP_USER_ZDOTDIR": user}
    return None


def plain_shell(command: str, shell: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return bool(argv) and os.path.basename(argv[0]) == shell and set(argv[1:]) <= PLAIN_BASH_OPTIONS


def plain_bash(command: str) -> bool:
    """Is `command` just an interactive bash ("bash", "/bin/bash -l")?  Then
    the session gets the integrated bash, as with no command: asking for
    "bash" is asking for a shell, and without the integration it would have
    no command boundaries, no exit statuses, and only prompt matching to go
    by.  Other options ("bash --norc") keep a plain bash."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return bool(argv) and os.path.basename(argv[0]) == "bash" and set(argv[1:]) <= PLAIN_BASH_OPTIONS


async def check_human(s: Session, force: bool, tool: str = "run") -> None:
    """Refuse to type where it could collide with a human, unless forced:
    over their unsubmitted text, into a pane they have scrolled back (copy
    mode would eat the keys), or right after their input when a program is
    running and not at a clean prompt."""
    if await s.tmux.in_mode(s.name):
        if not force:
            raise ToolError(
                "a human is viewing this session's scrollback (tmux copy mode), which would swallow "
                "typed keys; wait and retry, or ask the user.  force=true leaves copy mode first "
                "(their view jumps back to the bottom)."
            )
        await s.tmux.leave_mode(s.name)
    if force:
        return
    foreign = s.foreign_input()
    if foreign and s.typed_by(foreign) == "human" and not s.at_shell_prompt \
            and await s.tmux.client_count(s.name) == 0:
        # No human is attached to have typed it, and no agent did: at a
        # program's prompt it is the program's output (">>> step 1").
        foreign = ""
    if foreign:
        who = s.typed_by(foreign)
        whom = "a human is probably" if who == "human" else f"agent {who!r} typed it and is probably"
        raise ToolError(
            f"the prompt holds unsubmitted input you did not type: {foreign!r} -- {whom} "
            "mid-command. Wait and retry, or ask the user. force=true only if the line "
            "is stale: "
            + ("the `run` tool then discards it before typing." if tool == "run"
               else "the `send` tool then types after it; the `run` tool would discard it first.")
        )
    pattern = s.prompt_pattern
    if not s.at_clean_prompt(pattern) and await s.human_active():
        raise ToolError(
            "an attached terminal sent input (a key, click, scroll or focus change) within the last "
            "2 s while a program was running; wait and retry, or ask the user"
        )




PROBE = " echo dtmcp-probe b=$BASH_VERSION z=$ZSH_VERSION f=$FISH_VERSION"
# (fish drops a word whose variable is unset: "dtmcp-probe f=4.6.0")
PROBE_ANSWER = re.compile(r"^dtmcp-probe(?: b=(\S*))?(?: z=(\S*))?(?: f=(\S*))?\s*$", re.M)


async def probe_shell(s: Session, timeout_s: float) -> str | None:
    """Which shell is at the prompt: one line bash, zsh and fish all read
    the same way (the leading space keeps it out of their history)."""
    start = s.content_end()
    await s.tmux.send_text(s.name, PROBE)
    await s.tmux.send_keys(s.name, ["Enter"])
    s.note_sent()  # (typed by the caller's call, not a human)
    found: list[re.Match] = []

    def answered() -> bool:
        text = s.span_text(start, s.content_end())
        found[:] = [m for m in PROBE_ANSWER.finditer(text) if "$" not in m.group(0)]
        return bool(found)

    await s.wait(answered, timeout_s)
    await s.wait(lambda: False, 1, settle_s=0.3)  # its prompt, before the next line is typed
    if not found:
        return None
    bash, zsh, fish = found[-1].groups()
    return "bash" if bash else "zsh" if zsh else "fish" if fish else None


def readable_by_all(path: Path) -> bool:
    """Whether any user of this host can read `path` (a directory: and
    enter it), going by the modes of it and of every directory above."""
    try:
        return all(q.stat().st_mode & (0o005 if q.is_dir() else 0o004) == (0o005 if q.is_dir() else 0o004)
                   for q in (path, *path.parents))
    except OSError:
        return False


async def integration_source(s: Session) -> tuple[str | None, Place | None]:
    """Where the shell at the prompt reads the integration from: this
    server's own files (None) on this host, when that shell's user can read
    them; else the directory `NAME install HOST` writes, under the home of
    whoever is at the prompt."""
    from .remote import INSTALL_DIR, SHELL_DIR
    try:
        place = await s.where()
    except (TmuxError, OSError):
        place = None
    if place is None:
        return None, None
    here = place.host in (short_host(LOCAL_HOST), "localhost", "127.0.0.1", "::1")
    if here and (place.user in (None, LOCAL_USER, "root") or readable_by_all(SHELL_DIR)):
        return None, place
    return "~/" + INSTALL_DIR, place


# What bash and zsh say of a file they cannot read.
NOT_READ = re.compile(r"no such file or directory|permission denied|not a directory", re.I)


def not_installed(directory: str | None, shell: str, place: Place | None) -> str:
    """The reply when the integration file was not there to read."""
    from .remote import SHELL_DIR
    file = f"{directory or SHELL_DIR}/integration.{shell}"
    rest = "The `run` tool still works here, without exact exit statuses"
    if directory is None or place is None:
        return f"[no shell integration: the shell at the prompt could not read {file}. {rest}]"
    hop = (f" -- or connect with `{NAME} ssh {place.label()}`, which brings it along"
           if place.via in ("ssh", "sshpass") else "")
    return (f"[the shell integration is not installed for {place.label()} ({file} is not there): the user "
            f"installs it with `{NAME} install {place.label()}`, once per host (ask them; do not run it "
            f"yourself){hop}. {rest}]")


async def refuse_unless_shell_prompt(s: Session) -> None:
    """shell_integrate types a line for bash or zsh: only at a shell's
    prompt, never into a program (python) or a question (a password)."""
    at = s.term.cursor_text().strip()
    jobs = [p for p in await foreground_jobs(s) if not is_carrier(p.pid, p.command) and p.command not in SHELLS]
    m = DEFAULT_PROMPT.search(s.term.cursor_text())
    kind = s._prompt_kind(m) if m else None
    if jobs or kind != "sh" or at.endswith(QUESTION_ENDS):
        where = f"the cursor is at {at[-50:]!r}" if at else "no prompt is showing"
        running = f"; in the foreground: {named(jobs)}" if jobs else ""
        raise ToolError(f"shell_integrate types a line at a shell's prompt, but {where}{running}. "
                        "Get back to the shell's prompt first; force=true types it anyway")


async def integrate_adopted(s: Session) -> str | None:
    """An adopted pane's plain bash gets the shell integration on the first
    `run` there: the caller is about to type into it
    anyway, while a pane a human split for their own use is never typed
    into.  Only the pane's own bash, idle at its prompt; tried once."""
    if not s.meta.get("adopted") or s.integrated or s.meta.get("integration_tried") or s.at_shell_prompt:
        return None
    info = await s.tmux.pane_info(s.name)
    if info.current_command != "bash" or s.term.full_screen or s.foreign_input():
        return None
    if not s.term.cursor_text().strip():
        # Just split: bash is still reading its startup files.
        await s.wait(lambda: bool(s.term.cursor_text().strip()), ADOPTED_PROMPT_WAIT_S)
    if s.term.cursor_text().strip():
        try:
            await refuse_unless_shell_prompt(s)
        except ToolError:
            return None
    else:
        # Its prompt was printed before the pane was adopted, so before
        # logging began: the screen shows it, the log does not.
        info = await s.tmux.pane_info(s.name)
        rows = await s.tmux.capture(s.name)
        line = rows[min(info.cursor_y, len(rows) - 1)] if rows else ""
        line += " " if line.endswith(("$", "#", "%", ">")) else ""  # (the capture drops trailing spaces)
        m = DEFAULT_PROMPT.search(line)
        if m is None or s._prompt_kind(m) != "sh" or await foreground_jobs(s):
            return None
    s.meta["integration_tried"] = True
    before = s.offset
    s.shell_state = "unknown"  # wait for the newly marked prompt
    directory, _ = await integration_source(s)
    await s.tmux.send_text(s.name, integration_line(s.meta.get("nonce"), directory=directory))
    await s.tmux.send_keys(s.name, ["Enter"])
    why = await s.wait(lambda: s.shell_state == "input" and s.offset > before, 10)
    s.note_model_input()
    if why == "done":
        s.meta["shell_integration"] = True  # the pane's own shell: exit checks may use its pid
    (s.dir / "meta.json").write_text(json.dumps(s.meta, indent=1))
    if why == "done":
        return "[shell integration added to this adopted pane's bash first: exact exit statuses from here on]"
    return f"[shell integration did not take in this adopted pane ({why}); it stays a plain shell]"


def report_foreign_input(s: Session) -> None:
    """Input someone else typed at a program's prompt since the caller's
    last call becomes a region, and the reply starts by saying so."""
    region = s.claim_foreign_input()
    if region is not None and region.source != ACTOR.get() and (notes := REPLY_NOTES.get()) is not None:
        who = "a human" if region.source == "human" else f"agent {region.source!r}"
        notes.append(f"[since your last call, {who} typed at the prompt: {region.command!r} -- "
                     f"{describe(s, region)[1:-1]}]")


async def wait_out_foreign_line(s: Session, pending: str, timeout_s: float, max_lines: int) -> str:
    """Someone's unsubmitted text is at the prompt (run and send refuse to
    type over it): wait until it is gone, and say whether it was erased or
    submitted -- the latter becomes their region, shown."""
    who = s.typed_by(pending)
    whose = "a human's" if who == "human" else f"agent {who!r}'s"
    why = await s.wait(lambda: not s.foreign_input(), timeout_s)
    if why != "done":
        return (f"[{whose} unsubmitted {pending!r} is still at the prompt after {timeout_s:g} s; "
                "the `run` and `send` tools will not type over it -- call the `wait` tool again, or ask the user]")
    # Submitted, or erased?  A submitted line moves the cursor on; give the
    # program a moment to answer and show its prompt again.
    await s.wait(lambda: False, 2, settle_s=0.5, prompt=s.prompt_pattern)
    region = s.claim_foreign_input()
    if region is None:
        return f"[{whose} unsubmitted {pending!r} was erased; the prompt is free]"
    return (f"[{whose} {pending!r} was submitted:]\n" + render_region(s, region, max_lines))


def signal_job(job: list[Process], signum: int) -> None:
    """Signal the foreground job: its process group, or -- where signalling
    a group is not permitted though its processes are the caller's own (an
    OpenShell sandbox denies kill(-pgid)) -- each process of it."""
    try:
        os.killpg(job[0].pgid, signum)
        return
    except PermissionError:
        pass
    reached, gone = 0, 0
    for p in job:
        try:
            os.kill(p.pid, signum)
            reached += 1
        except ProcessLookupError:
            gone += 1
        except PermissionError:
            pass
    if not reached:
        raise ProcessLookupError if gone == len(job) else PermissionError


def exit_said(status, form: str = " (status {})") -> str:
    """The status in `form`, or nothing when tmux kept none -- a pane's
    program killed by a signal shows no status there ("has exited (status
    None)" in an eval's reply)."""
    return form.format(status) if status is not None else ""


async def refuse_if_exited(s: Session) -> None:
    """Nothing can be typed into a session whose program has exited."""
    info = await s.tmux.pane_info(s.name)
    s.note_pane(info)  # which program is in the foreground: its prompt is known by it
    if info.dead:
        s.pane_exited(info)
        raise ToolError(f"the program in session {s.name!r} has exited{exit_said(info.dead_status)}; its log "
                        f"remains (read, search).  session_close ends the tmux session; session_create "
                        f"starts a new one")


async def busy_command(s: Session, pattern: re.Pattern) -> Region | None:
    """The shell command that is running, busy, and not asking anything,
    if there is one."""
    command = s.open_command
    if command is None or s.term.full_screen or s.prompt_here(pattern):
        return None
    remote = s.remote_command()
    if remote is not None and s.term.col == 0:
        # Its shell's markers say it is still running over there, and nothing
        # on the line asks anything: the local ssh looks idle, the command
        # is not.
        return remote
    info = await s.tmux.pane_info(s.name)
    return command if await foreground_state(info.tty, info.pid if s.integrated else 0) == "busy" else None


INTERRUPT_WAIT_S = 5


async def interrupt_first(s: Session, pattern: re.Pattern) -> str:
    """run's interrupt=true: a busy command holding the session gets a
    ctrl+c, and the line is typed once a prompt is back.  One call where
    the busy refusal's remedy took three -- each an approval card for a
    user in ask-first mode (adjutant's glm, a `tail -f`)."""
    command = await busy_command(s, pattern)
    if command is None:
        return ""
    open_ids = {r.id for r in s.regions if not r.done}
    before = await foreground_jobs(s)
    s.note_sent()
    await s.tmux.send_keys(s.name, ["C-c"])
    await s.wait(lambda: s.at_shell_prompt or bool(s.prompt_here(pattern)), INTERRUPT_WAIT_S)
    s.note_model_input()
    s.close_stale_input(interrupted=True)
    ended_regions = [r for r in s.regions if r.id in open_ids and r.done]
    s.open_at_reply = s.open_at_reply - {r.id for r in ended_regions}  # reported here, not again at the end
    ended = [describe(s, r) for r in ended_regions]
    if not (s.at_shell_prompt or s.prompt_here(pattern)):
        stuck = await foreground_jobs(s)
        who = named([p for p in stuck if p.pid in {b.pid for b in before}] or stuck) if stuck else "it"
        raise ToolError(f"ctrl+c sent to region {command.id} ($ {(command.command or '')[:60]}), but no prompt "
                        f"came back within {INTERRUPT_WAIT_S} s ({who} still in the foreground), so nothing was "
                        f"typed.  `wait` waits for it; `send` signal=\"TERM\" ends a job that ignores ctrl+c")
    return "".join(f"[interrupted with ctrl+c: {d[1:]}\n" for d in ended) or \
        f"[interrupted with ctrl+c: region {command.id}]\n"


async def refuse_type_ahead(s: Session, pattern: re.Pattern, line: str = "") -> None:
    """A shell command is running, busy, and not asking anything: a line
    typed now would be read by it, or run by the shell after it finishes,
    unreported.  Refuse (force=true types ahead anyway), and offer the
    line as a `send`: right when the program reads keys itself (a TUI that
    paints in place was not seen as one)."""
    command = await busy_command(s, pattern)
    if command is None:
        return
    as_send = ""
    if line and "\n" not in line and len(line) <= 200:
        as_send = (f"  Only if that program reads keys itself (a TUI, an editor) and the line is meant for "
                   f"it: `send` text={json.dumps(line, ensure_ascii=False)} keys=[\"enter\"].")
    # (No mention of force here: models given it retried with it, typing ahead
    # into the running command, in probes with local models.)
    raise ToolError(
        f"region {command.id} ($ {(command.command or '')[:60]}) is still running and not waiting for input, "
        "so this session is busy: a line typed now would be read by it, or run by the shell after it ends, with "
        "no reply for it.  To run something else now, use another session (session_create, or session_split "
        "beside this one); the `wait` tool waits for this command; `send` keys=['ctrl+c'] interrupts it, or this "
        "line with interrupt=true interrupts it and then types the line."
        + as_send)


def discard_pending(s: Session, force: bool = False) -> bool:
    """Should typing start by clearing the prompt?  Yes for a human's line
    (only reached with force) and for text the model left half-typed; not
    for the model's own incomplete command, which its next line completes,
    nor at a program's prompt for text nobody typed (its output)."""
    pending = s.pending_input()
    if not pending:
        return False
    if not s.at_shell_prompt:
        return pending == (s.model_typed or "") or force
    return bool(s.foreign_input()) or not s.continuation


async def clear_prompt(s: Session) -> str:
    """Discard unsubmitted input at the shell prompt -- a human's stale line
    (force=true), or text the model itself left there -- and say what was
    discarded: it leaves no other trace."""
    whose = "" if s.foreign_input() else "your own "
    discarded = s.pending_input()
    if s.continuation:
        await s.tmux.send_keys(s.name, ["C-c"])  # abandon the incomplete command
        # The last line may be only the continuation prompt: nothing typed.
        typed = discarded.split("\n")
        while len(typed) > 1 and typed[-1].strip() in ("", ">"):
            typed.pop()
        first, *more = typed
        what = f"abandoned {whose}incomplete command: {first!r}" + (
            f" (+{len(more)} line{'s' if len(more) > 1 else ''})" if more else "")
    else:
        await s.tmux.send_keys(s.name, ["C-e", "C-u"])
        what = f"discarded {whose}unsubmitted {'text' if whose else 'line'}: {discarded!r}"
    await s.wait(lambda: not s.continuation and not s.pending_input(), 3)
    s.note_model_input()
    return f"[force: {what}]" if not whose else f"[{what}]"


def compile_pattern(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regular expression {pattern!r}: {e}") from e


def call_prompt(s: Session, prompt_pattern: str | None) -> re.Pattern:
    """A call's prompt pattern: its own (checked, with continuation prompts
    added), else the session's."""
    if prompt_pattern is None:
        return s.prompt_pattern
    compile_pattern(prompt_pattern)
    return prompt_regex(prompt_pattern)


def numbered(session: Session, rows: range, span: tuple | None = None) -> str:
    """Numbered lines; with `span` (start, end), cut at its bounds (a
    region ending mid-row, output without a final newline, does not take
    the next prompt with it)."""
    width = len(str(rows.stop))
    lines = session.iter_span(*span, rows) if span else session.iter_lines(rows)
    return "\n".join(f"{r + 1:>{width}}\t{clip_line(line.rstrip())}" for r, line in lines)


def region_place(region: Region) -> str | None:
    """Where a region ran, if not here as this user: "build-host", "root@devbox"."""
    host = short_host(region.host) if region.host else short_host(LOCAL_HOST)
    user = region.user if region.user and region.user != LOCAL_USER else None
    if host == short_host(LOCAL_HOST) and user is None:
        return None
    return f"{user}@{host}" if user else host


def describe(session: Session, region: Region) -> str:
    s = session.summary(region)
    bits = [f"region {s['id']}"]
    if region.command:
        first, *more = region.command.split("\n")
        # A program's input keeps its end (the input, after the program's
        # prompt), a command its start -- a plain shell's too, its prompt
        # already cut to its sign ("$ make", Session.short_shell_label).
        tail = region.kind == "input" and not re.match(r"[$#%>] ", first)
        cmd = first if len(first) <= 80 else ("..." + first[-77:] if tail else first[:77] + "...")
        if more:
            cmd += f" (+{len(more)} line{'s' if len(more) > 1 else ''})"
        bits.append(f"$ {cmd}" if region.kind == "command" else cmd)
    else:
        bits.append(region.kind)
    if where := region_place(region):
        bits.append(f"on {where}")
    if s["source"] != ACTOR.get():
        bits.append(f"started by {s['source']}")
    if region.echo_cut:
        bits.append(f"echo cut short: {region.echo_cut[0]} of {region.echo_cut[1]} characters")
    if region.status == "answered":
        bits.append(f"answered (region {region.answered_by})")
    elif region.status != "done":
        bits.append(region.status)
    stages = region.pipestatus or []
    # 141 = SIGPIPE, in a stage with a later one: that one stopped reading
    # (`| head`), which is no failure -- the header said "but stage 1
    # exited 141" and a model stopped to check for a crash (pg-default-prune).
    failed = next(((i, v) for i, v in enumerate(stages) if v and not (v == 141 and i + 1 < len(stages))), None)
    sigpipe = next((i for i, v in enumerate(stages) if v == 141 and i + 1 < len(stages)), None)
    if s["exit_status"] == 0 and failed is not None:
        # A failed build behind `| tee f | tail`: the pipeline's 0 must not
        # read as success (seen on a PostgreSQL build).
        bits.append(f"exit 0, but stage {failed[0] + 1} exited {failed[1]}")
    elif s["exit_status"] == 0 and sigpipe is not None:
        bits.append(f"exit 0 (stage {sigpipe + 1}: 141, SIGPIPE -- a later stage stopped reading)")
    elif s["exit_status"] is not None:
        bits.append(f"exit {s['exit_status']}")
    elif region.kind == "command" and region.status == "done" and (region.command or "").split()[:1] in (["exit"], ["logout"]):
        # (Its shell ended with it: no status to report -- "not seen" read
        # as a failure after leaving an ssh hop.)
        bits.append("ended its shell")
    elif region.kind == "command" and region.status == "done":
        bits.append("exit status not seen")
    if region.pipestatus:
        bits.append("pipestatus " + " ".join(map(str, region.pipestatus)))
    if s["lines"]:
        bits.append(f"lines {s['lines']}")
    elif region.done:
        bits.append("no output")
    bits.append(s["file"])
    return "[" + " · ".join(bits) + "]"


def open_command_ids(session: Session) -> set[int]:
    return {r.id for r in session.regions if r.kind == "command" and not r.done}


# Notes a call's reply starts with (see call_logger): set per call.
REPLY_NOTES: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("reply_notes", default=None)
# The regions a call's reply showed, and what to do once it is written (a
# save_to): set per call.
REPLY_REGIONS: contextvars.ContextVar[list[Region] | None] = contextvars.ContextVar("reply_regions", default=None)
REPLY_AFTER: contextvars.ContextVar[list | None] = contextvars.ContextVar("reply_after", default=None)
HELD_UP_S = 1.0  # a typing call held up longer than this says by whom


def ago(t: float | None, now: float) -> str:
    """'40s ago', '12m ago', '3h ago', '2d ago'."""
    s = max(0, now - t)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            return f"{int(s // size)}{unit} ago"
    return f"{int(s)}s ago"


def presence(d: dict) -> str:
    """session_list's activity summary: how old the session is, its last
    output, who typed last and when (agents by name, "you" for the caller,
    attached humans' terminals), and a typing call in progress."""
    now = time.time()
    parts = []
    if d.get("created"):
        parts.append(f"started {ago(d['created'], now)}")
    if d.get("last_output"):
        parts.append(f"output {ago(d['last_output'], now)}")
    me = ACTOR.get()
    typed = sorted((d.get("agents_typed") or {}).items(), key=lambda kv: -kv[1])
    if typed:
        parts.append("typed: " + ", ".join(f"{'you' if a == me else a} {ago(t, now)}" for a, t in typed[:3])
                     + (f", +{len(typed) - 3} more" if len(typed) > 3 else ""))
    if d.get("human_input"):
        parts.append(f"human terminal input {ago(d['human_input'], now)}")
    if (busy := d.get("typing_now")) and busy["agent"] != me:
        parts.append(f"now: {busy['agent']}'s {busy['tool']} in progress")
    return "; ".join(parts)


@contextlib.asynccontextmanager
async def typing_turn(session: Session, tool: str):
    """Hold the session for a call that types: typing calls of all agents
    take turns.  One held up behind another agent's call says so -- it may
    have waited the whole of that call's timeout."""
    holder = session.holder
    t0 = time.monotonic()
    async with session.lock:
        waited = time.monotonic() - t0
        if waited >= HELD_UP_S and holder is not None and (notes := REPLY_NOTES.get()) is not None:
            who, what = holder
            whose = "your own" if who == ACTOR.get() else f"{who}'s"
            notes.append(f"[waited {waited:.1f} s for {whose} {what} in this session]")
        session.holder = (ACTOR.get(), tool)
        try:
            yield
        finally:
            session.holder = None
            session.note_typed()


def reading(session: Session) -> contextlib.AbstractContextManager:
    """For a call that only reads (read, search, regions): no lock, so it
    never waits behind another agent's run.  Nothing it does can interleave
    with an update: the server is one event loop, and sync() and the reads
    after it do not yield."""
    return contextlib.nullcontext()


@contextlib.asynccontextmanager
async def replying(session: Session, tool: str | None = None):
    """For a call that types (`tool` names it: see typing_turn) or only
    waits and watches (no lock: waiting must not keep another agent from
    typing -- the input that ends what is waited for may be theirs), and
    afterwards note which commands were still running, so the next call
    can report those that finished in between."""
    async with (typing_turn(session, tool) if tool else contextlib.nullcontext()):
        try:
            yield
            await note_place(session)
            if (after := REPLY_AFTER.get()) is not None:
                after.append(lambda: deleted_cwd_note(session))
                after.append(lambda: fedora_rpms_note(session))
        finally:
            session.typing = None  # a `run`'s human-like typing is for that call only
            session.open_at_reply = open_command_ids(session)


UNSEEN = object()


def is_carrier_command(command: str) -> bool:
    """Is this command line one that took the session elsewhere -- ssh,
    `docent-terminal-mcp ssh`/`into`, a container's exec -it?"""
    from .tmux import CONTAINER_CLIENTS, REMOTE_CLIENTS, container_entry
    argv = command.split()
    if not argv:
        return False
    prog = os.path.basename(argv[0])
    if prog == NAME:
        return argv[1:2] in (["ssh"], ["into"], ["su"])
    return prog in REMOTE_CLIENTS or (prog in CONTAINER_CLIENTS and container_entry(argv) is not None)


MISSING_FOR_S = 12 * 3600  # how long "no integration installed there" is remembered


def _missing_path(s: Session) -> Path:
    return s.dir.parent.parent / "integration-missing.json"  # beside sessions/: one for the server


def note_integration_missing(s: Session, label: str, found: bool = False) -> None:
    """shell_integrate found no installed integration for `label`
    (user@host): every session's notes for that host say so from now on."""
    path = _missing_path(s)
    try:
        known = json.loads(path.read_text())
    except (OSError, ValueError):
        known = {}
    if found:
        if known.pop(label, None) is None:
            return  # (nothing was remembered)
    else:
        known[label] = time.time()
    path.write_text(json.dumps(known))


def integration_missing_since(s: Session, label: str) -> str | None:
    """When shell_integrate last found no integration for `label`, as HH:MM,
    if that is recent (the user may install it since)."""
    try:
        t = json.loads(_missing_path(s).read_text()).get(label)
    except (OSError, ValueError):
        return None
    if t is None or time.time() - t > MISSING_FOR_S:
        return None
    return time.strftime("%H:%M", time.localtime(t))


async def note_place(s: Session) -> None:
    """Say so when the host (or user) the session's next line runs on
    changed since this agent's last reply: after `ssh host`, `sudo -i`,
    and when they end -- a dropped ssh (exit 255) lands the next line here
    again.  The first reply to an agent says where a session is only if
    that is not here."""
    try:
        info = await s.tmux.pane_info(s.name)
        if info.dead:
            return  # the session's own program ended: nowhere for a next line to run
        place = await s.where(info)
    except (TmuxError, OSError):
        return
    before = s.place_noted.get(ACTOR.get(), UNSEEN)
    if before is UNSEEN or not ((place is None and before is None) or (place is not None and place.same(before))):
        s.note_place(place)
    first = before is UNSEEN  # this agent has not been told before: "is on", not "is now on"
    if before is UNSEEN:
        before = None
    if (place is None and before is None) or (place is not None and place.same(before)):
        return
    if (notes := REPLY_NOTES.get()) is None:
        return
    here = f"{LOCAL_USER}@{short_host(LOCAL_HOST)}"
    # A shell over there without the integration (ssh, sudo -i do not pass
    # it on): say what to do, now -- only then are its exit statuses exact
    # and a further hop from it seen.
    plain = place is not None and s.shell_state not in ("prompt", "input") and s.at_plain_shell_prompt()
    integrate = ""
    unmarked = place is not None and place.via != "shell"  # (no markers from the shell over there)
    if unmarked and (when := integration_missing_since(s, place.label())):
        # shell_integrate found no integration installed there (in any session).
        integrate = (f"; shell_integrate found no integration installed on {place.label()} at {when} "
                     f"(the user installs it: `{NAME} install {place.label()}`)")
        if place.via in ("ssh", "sshpass"):
            integrate += f" -- or connect with `{NAME} ssh {place.label()}`, which brings it along"
    elif plain:
        integrate = (f"; if the shell there is bash or zsh, `shell_integrate` session=\"{s.name}\" makes its exit "
                     "statuses exact and keeps track of further hops")
        if place.via in ("ssh", "sshpass"):
            integrate += f" -- or connect with `{NAME} ssh {place.label()}`, which brings it along"
    if place is None:
        left = f"the {before.via} to {before.label()}" if before.via in ("ssh", "telnet", "mosh-client", "sshpass") \
            else f"{before.label()}'s shell"
        carrier = next((r for r in reversed(s.regions) if r.kind == "command" and r.done and r.command
                        and is_carrier_command(r.command)), None)
        if carrier is not None and before.via == "shell":  # (markers said where: say how it was reached)
            status = f", exit {carrier.exit_status}" if carrier.exit_status is not None else ""
            last = next((r for r in reversed(s.regions) if r.kind == "command" and r.id > carrier.id), None)
            if carrier.exit_status and last is not None and (last.command or "").strip() in ("exit", "logout"):
                # A bare `exit` passes on the far shell's last status: ssh
                # did not fail (`ls /nonexistent`, then exit: "exit 2").
                status += ": the far shell's last command's, which a bare `exit` passes on"
            left = f"the connection to {before.label()} ({carrier.command.split()[0]}{status})"
        notes.append(f"[this session is back on {here}: {left} ended -- the next line runs here]")
    elif place.host == short_host(LOCAL_HOST):
        notes.append(f"[this session's shell {'runs' if first else 'now runs'} as {place.user} on {place.host} ({place.via}) -- "
                     f"the next line runs as {place.user}{integrate}]")
    else:
        via = f"via {place.via}" if place.via != "shell" else "its shell's markers say so"
        notes.append(f"[this session is {'on' if first else 'now on'} {place.label()} ({via}) -- the next line runs there{integrate}]")


def outer_finished(session: Session, were_open: set[int], shown: list[Region]) -> list[str]:
    """Commands open before a call that finished during it -- a nested
    shell, su or ssh left by `exit` -- reported after the call's own."""
    ids = {r.id for r in shown}
    done = [r for r in session.regions if r.id in were_open and r.done and r.id not in ids]
    return ["[the command finished:]\n" + describe(session, r) for r in reversed(done)]


SHOW_TAIL = 3
REWRITE_ROWS = 60  # rows above the cursor `send` watches for in-place rewrites  # last lines shown after a `show` filter's matches


def filtered(session: Session, start, end, show: re.Pattern, max_lines: int, where: str,
             region_id: int | None = None, after: int = 0, last: int | None = None) -> str:
    """The lines of a span matching `show`, numbered, then its last few
    lines: what a `cmd | grep` would give, without filtering the command.
    Every match in order while they fit; beyond that identical matching
    lines are shown once with a count, and if still too many, the first
    and last are kept and the rest named for `search`.  `after` adds that
    many lines after each match (grep -A); `last` sets how many last lines
    follow."""
    rows = session.span_rows(start, end)
    where = where.replace("{start}", str(rows.start + 1)).replace("{limit}", str(len(rows)))
    hits = [(r, t) for r, t in session.iter_span(start, end) if show.search(t)]
    tail_n = SHOW_TAIL if last is None else max(last, 0)
    if after > 0:
        return with_context(session, start, end, rows, show, hits, max_lines, where, region_id, after, tail_n)
    budget = max(max_lines - tail_n - 2, 2)
    if len(hits) <= budget:
        # Every match, in order, repeats included: in a trace the repeats
        # and their order are the information (an infrun trace shown as
        # "15 distinct" sent a model to grep in its own shell tool).
        entries = [(r, t, [r]) for r, t in hits]
    else:
        groups: dict[str, list[int]] = {}
        for r, t in hits:
            groups.setdefault(t, []).append(r)
        entries = [(rs[0], t, rs) for t, rs in groups.items()]
    head = tail = entries
    if len(entries) > budget:
        head, tail = entries[: budget // 2], entries[len(entries) - (budget - budget // 2):]
    else:
        tail = []
    distinct = f", {len(entries)} distinct" if len(entries) < len(hits) else ""
    out = [f"[{len(hits)} of {len(rows)} lines match {show.pattern!r}{distinct}; {where}]"]

    def line(entry):
        first, text, rs = entry
        more = f"  [x{len(rs)}, last at line {rs[-1] + 1}]" if len(rs) > 1 else ""
        return f"{first + 1}\t{clip_line(text)}{more}"

    out += [line(e) for e in head]
    if tail:
        skipped = len(entries) - len(head) - len(tail)
        first_skipped = entries[len(head)][0]
        skip = sum(1 for r, _ in hits if r < first_skipped)
        target = f"region={region_id} " if region_id is not None else ""
        out.append(f"[... {skipped} more distinct matching lines; search pattern={show.pattern!r} "
                   f"{target}skip={skip} to page through them ...]")
        out += [line(e) for e in tail]
    return "\n".join(out + last_lines(session, start, end, rows, {e[0] for e in head + tail}, tail_n))


def last_lines(session: Session, start, end, rows: range, shown: set[int], n: int) -> list[str]:
    """The span's last `n` lines not already shown, after its matches."""
    last = [(r, t) for r, t in session.iter_span(start, end, range(max(rows.start, rows.stop - n), rows.stop))
            if r not in shown]
    return (["[last lines, not matches:]"] + [f"{r + 1}\t{clip_line(t)}" for r, t in last]) if last else []


def with_context(session: Session, start, end, rows: range, show: re.Pattern, hits: list[tuple[int, str]],
                 max_lines: int, where: str, region_id: int | None, after: int, tail_n: int) -> str:
    """`filtered` with `after` lines after each match, as grep -A shows
    them: a match numbered "N:", a line after it "N-", "--" between
    groups.  Repeats are not merged (the lines after them differ); when
    the groups do not fit, the first and last are kept."""
    groups: list[list[int]] = []
    for r, _ in hits:
        stop = min(r + after + 1, rows.stop)
        if groups and r <= groups[-1][-1] + 1:
            groups[-1].extend(range(groups[-1][-1] + 1, stop))
        else:
            groups.append(list(range(r, stop)))
    budget = max(max_lines - tail_n - 2, 2)
    head, tail = groups, []
    if sum(len(g) + 1 for g in groups) > budget:
        head, tail, used = [], [], 0
        for g in groups:  # the first half of the budget from the front ...
            if used + len(g) + 1 > budget // 2 and head:
                break
            head.append(g)
            used += len(g) + 1
        for g in reversed(groups[len(head):]):  # ... the rest from the back
            if used + len(g) + 1 > budget and tail:
                break
            tail.insert(0, g)
            used += len(g) + 1
    matched = {r for r, _ in hits}
    wanted = {r for g in head + tail for r in g}
    text = dict(session.iter_span(start, end, range(min(wanted, default=0), max(wanted, default=-1) + 1)))

    def block(g):
        return [f"{r + 1}{':' if r in matched else '-'}\t{clip_line(text.get(r, ''))}" for r in g]

    out = [f"[{len(hits)} of {len(rows)} lines match {show.pattern!r}, each shown with up to {after} "
           f"line{'s' if after > 1 else ''} after it (\"N:\" a match, \"N-\" a line after one); {where}]"]
    for i, g in enumerate(head):
        out += (["--"] if i else []) + block(g)
    if tail:
        skipped = len(groups) - len(head) - len(tail)
        if skipped:
            skip = sum(1 for r, _ in hits if r < groups[len(head)][0])
            target = f"region={region_id} " if region_id is not None else ""
            out.append(f"[... {skipped} more matches with the lines after them; search pattern={show.pattern!r} "
                       f"{target}skip={skip} context={after} to page through them ...]")
        else:
            out.append("--")
        for i, g in enumerate(tail):
            out += (["--"] if i else []) + block(g)
    ending = [] if rows.stop - 1 in wanted else last_lines(session, start, end, rows, wanted, tail_n)
    return "\n".join(out + ending)


def render_progress(session: Session, region: Region, max_lines: int, show: re.Pattern | None = None,
                    after: int = 0, last: int | None = None) -> str:
    """A region shown again while (or after) it runs: only what was not
    shown before, so polling a build shows its new tail each time."""
    shown_in_reply(region)
    end = session.region_end(region)
    seen = session.shown_upto.get(region.id)
    where = f"`read` region={region.id} start={{start}} limit={{limit}} shows them, or the `search` tool"
    if seen is None or seen <= region.start[0]:
        out = render_region(session, region, max_lines, show, after, last)
    else:
        start = (seen, 0)
        body = (filtered(session, start, end, show, max_lines, where, region.id, after, last) if show
                else clip_with_summary(session, start, end, max_lines, where, region.id, last))
        out = (describe(session, region) + f"\n[lines {region.start[0] + 1}-{seen} shown before]"
               + ("\n" + body if body else ""))
    note_shown(session, region)
    return out


def note_shown(session: Session, region: Region) -> None:
    """Remember how far a running region has been shown (its last row may
    still grow, so it is shown again)."""
    end = session.region_end(region)[0]
    if region.done:
        session.shown_upto.pop(region.id, None)
    else:
        session.shown_upto[region.id] = end
    # The commands still open around it (an ssh, a nested bash) hold these
    # lines too: shown now, not again as theirs.
    for outer in session.open_commands.values():
        if outer is not region and not outer.done and outer.start <= region.start:
            session.shown_upto[outer.id] = max(session.shown_upto.get(outer.id, 0), end)
    session._save_state()


def render_region(session: Session, region: Region, max_lines: int, show: re.Pattern | None = None,
                  after: int = 0, last: int | None = None) -> str:
    shown_in_reply(region)
    where = f"`read` region={region.id} start={{start}} limit={{limit}} shows them, or the `search` tool"
    end = session.region_end(region)
    body = (filtered(session, region.start, end, show, max_lines, where, region.id, after, last) if show
            else clip_with_summary(session, region.start, end, max_lines, where, region.id, last))
    out = describe(session, region) + ("\n" + body if body else "")
    if region.echo_cut:
        out += (f"\n[the program echoed only the first {region.echo_cut[0]} of the {region.echo_cut[1]} characters "
                "typed, so it most likely ran only those: it cut the line (a line-length limit, as busybox ash's). "
                "Split the command, or put it in a script]")
    return out


SUMMARY_ON = os.environ.get("DOCENT_TERMINAL_MCP_SUMMARY", "1") != "0"


def clip_with_summary(session: Session, start, end, max_lines: int, where: str, region_id: int,
                      last: int | None = None) -> str:
    """Output clipped to max_lines, led by its summary lines (test results,
    make's failure; see summary.py) where the clip would leave them out --
    unless the caller chose its own lines with `last_lines` (or `show`)."""
    rows = session.span_rows(start, end)
    if last is not None or not SUMMARY_ON or len(rows) <= max_lines:
        return session.clip_span(start, end, max_lines, where, last=last)
    found, more = summary.summary_lines(session.iter_span(start, end))
    if not found:
        return session.clip_span(start, end, max_lines, where)
    # Only what the clipped output would not show anyway (a run's last lines
    # often hold its summary); the room left goes to more failure lines.
    budget = max(max_lines - len(found) - 3, 10)
    shown = session.clip_rows(start, end, budget)
    missing = [(r, t) for r, t in found if r not in shown]
    more = [(r, t) for r, t in more if r not in shown]
    room = max(summary.MAX_LINES - len(missing), 0)
    missing, more = sorted(missing + more[:room]), more[room:]
    if not missing:
        return session.clip_span(start, end, max_lines, where)
    budget = max(max_lines - len(missing) - 3, 10)
    out = ["[summary lines picked out of the whole output (test results, make's failure; `show` picks your own):]"]
    out += [f"{r + 1}\t{clip_line(t)}" for r, t in missing]
    if more:
        out.append(f"[... {len(more)} more failure lines: `search` pattern={summary.search_pattern(more)!r} "
                   f"region={region_id}]")
    out.append("[the output itself:]")
    return "\n".join(out + [session.clip_span(start, end, budget, where)])


def shown_in_reply(region: Region) -> None:
    if (shown := REPLY_REGIONS.get()) is not None and region not in shown:
        shown.append(region)


async def save_target(s: Session, path: str) -> str:
    """A save_to path made absolute -- a relative one is in the directory
    of what runs in front in the session (its shell, at a shell prompt) --
    and checked writable before anything is typed."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        info = await s.tmux.pane_info(s.name)
        front = await foreground_processes(info.tty)
        pid = next((p.pid for p in front if p.pid == p.pgid), info.pid)
        try:
            base = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            raise ToolError(f"save_to {path!r}: cannot tell the session's directory; give an absolute path")
        path = os.path.join(base, path)
    parent = os.path.dirname(path) or "/"
    if os.path.isdir(path):
        raise ToolError(f"save_to {path!r} is a directory; name a file")
    if not os.path.isdir(parent):
        raise ToolError(f"save_to {path!r}: no directory {parent!r} (on {LOCAL_HOST}, where this server runs)")
    if not os.access(path if os.path.exists(path) else parent, os.W_OK):
        raise ToolError(f"save_to {path!r}: not writable")
    return path


def check_output_options(last_lines: int | None, show_after: int, show: str | None) -> None:
    if last_lines is not None and last_lines < 0:
        raise ToolError("last_lines must be 0 or more")
    if show_after < 0:
        raise ToolError("show_after must be 0 or more")
    if show_after and not show:
        raise ToolError("show_after counts lines after each `show` match: give `show` too")


def save_reply(s: Session, path: str) -> str:
    """After the reply: write the output of the regions it showed to
    `path`, and keep it up to date while they run."""
    regions = REPLY_REGIONS.get() or []
    if not regions:
        return f"[save_to: no output to save; {path} not written]"
    try:
        s.save_regions(path, regions)
    except OSError as e:
        return f"[save_to: could not write {path}: {e.strerror or e}]"
    running = any(not r.done for r in regions)
    return (f"[output saved to {path} (on {LOCAL_HOST})"
            + ("; it is kept up to date while the command runs, and written in full when it ends]" if running
               else "]"))


FULL_REGIONS = 3  # regions rendered in full when a call ran several commands
LISTED_REGIONS = 10


def render_regions(session: Session, regions: list[Region], max_lines: int,
                   show: re.Pattern | None = None, after: int = 0, last: int | None = None) -> list[str]:
    """Several regions from one call: up to FULL_REGIONS in full; beyond
    that, headers for the earlier ones (a few) and the last in full."""
    for r in regions:
        shown_in_reply(r)
    if len(regions) <= FULL_REGIONS:
        return [render_region(session, r, max_lines, show, after, last) for r in regions]
    earlier = regions[:-1]
    out = [describe(session, r) for r in earlier[:LISTED_REGIONS]]
    if len(earlier) > LISTED_REGIONS:
        out.append(f"[... {len(earlier) - LISTED_REGIONS} more commands; the `regions` tool lists them]")
    return out + [render_region(session, regions[-1], max_lines, show, after, last)]


def screen_rows(rows: list[str], indices: list[int]) -> list[str]:
    """Numbered screen rows.  Runs of two or more blank rows, and of three or
    more identical rows (vim's and less's "~" filler), are summarised."""
    out: list[str] = []
    width = len(str(indices[-1] + 1)) if indices else 1
    i = 0
    while i < len(indices):
        r = indices[i]
        j = i
        while (j + 1 < len(indices) and indices[j + 1] == indices[j] + 1
               and rows[indices[j + 1]] == rows[r]):
            j += 1
        if j > i and not rows[r]:
            out.append(f"[rows {r + 1}-{indices[j] + 1} blank]")
        elif j - i >= 2:
            out.append(f"[rows {r + 1}-{indices[j] + 1}: {rows[r]!r}]")
        else:
            out.append(f"{r + 1:>{width}}\t{rows[r]}")
            j = i
        i = j + 1
    return out


SCREEN_MODES = ("full", "changes", "cursor")


async def render_screen(
    tmux: Tmux, session: Session, start: int | None = None, limit: int | None = None,
    mode: str = "full", max_rows: int | None = None,
) -> str:
    """The pane as a human sees it: numbered screen rows exactly as shown (a
    wrapped line spans two rows) with the size and cursor position.

    mode="full": rows `start`.. (1-based) up to `limit` of them; with only
    `limit`, the last rows in use.  mode="changes": only rows that differ
    from the screen this session last returned.  mode="cursor": the cursor
    row and two rows either side.  `max_rows` caps the rows shown (the
    cursor row is kept); the rest are named for read view="screen".
    """
    info = await tmux.pane_info(session.name)
    rows = await tmux.capture(session.name)
    previous, session.last_screen = session.last_screen, rows
    cy = min(info.cursor_y, len(rows) - 1)
    header = (
        # tmux reports the column past the edge after a write to the last
        # cell (a pending wrap); the cursor is on that last cell.
        f"[screen {info.width}x{info.height} · cursor row {info.cursor_y + 1}, "
        f"col {min(info.cursor_x, info.width - 1) + 1}"
        + (" · full-screen program" if info.alternate_on or session.term.full_screen else "")
    )
    if not (info.alternate_on or session.term.full_screen):
        # Where this screen sits in the log, and the region to read for what
        # scrolled off: `read`/`search` take both.
        header += f" · log line {session.term.row + 1}"
        if session.regions:
            last = session.regions[-1]
            header += f" · last region {last.id} ({last.status if not last.done else 'done'})"
    above: list[str] = []
    notes: list[str] = []
    comparable = previous is not None and len(previous) == len(rows)
    if mode == "cursor" and cy >= info.height - 1 and (info.alternate_on or session.term.full_screen):
        # A pager's or editor's command line: the rows around it say nothing
        # of where the view moved, and a note pointing elsewhere was ignored
        # (a model lost 15 calls in less).  Hand back the useful view.
        mode = "changes" if comparable else "full"
        notes.append("[the cursor is on the bottom line (a status or command line), so "
                     + ("the rows that changed are shown" if comparable else "the whole screen is shown")
                     + ' instead of screen="cursor"]')
    if mode == "changes" and comparable:
        indices = [i for i in range(len(rows)) if rows[i] != previous[i]]
        header += f" · {len(indices)} row{'s' if len(indices) != 1 else ''} changed"
        if not indices:
            notes.append("[no rows changed]")
    elif mode == "cursor":
        indices = list(range(max(0, cy - 2), min(len(rows), cy + 3)))
    else:
        if mode == "changes":
            header += " · whole screen (nothing to compare with)"
        used = max([i for i, r in enumerate(rows) if r] + [cy])
        if start is None and limit is not None:
            first, stop = max(0, used + 1 - limit), used + 1
            if first > 0:
                above.append(f"[rows 1-{first} above; start=1 to see them]")
        else:
            first = max(0, (start or 1) - 1)
            stop = used + 1 if limit is None else min(used + 1, first + limit)
        indices = list(range(first, stop))
        if stop <= used:
            notes.append(f"[rows {stop + 1}-{used + 1} not shown; start={stop + 1}]")
        if max(stop, used + 1) < len(rows):
            notes.append(f"[rows {max(stop, used + 1) + 1}-{len(rows)} blank]")
    if max_rows is not None and len(screen_rows(rows, indices)) > max_rows:
        # Drop rows from the end (keeping the cursor row) until the reply,
        # with summaries counted as one line each, fits.
        keep = list(indices)
        while len(keep) > 1 and len(screen_rows(rows, keep)) > max(max_rows - 1, 1):
            drop = next(i for i in reversed(keep) if i != cy)
            keep.remove(drop)
        dropped = [i for i in indices if i not in keep]
        notes.insert(0, f"[{len(dropped)} rows not shown (max_lines): rows {dropped[0] + 1}-{dropped[-1] + 1}; "
                        "read view=screen for all]")
        indices = keep
    return "\n".join([header + "]"] + above + screen_rows(rows, indices) + notes)


def escape_raw(data: bytes) -> str:
    """Raw output made readable: escape and control characters spelled out
    (\\e, \\r, \\xNN), with a real line break after each \\n."""
    out = []
    for ch in data.decode("utf-8", "backslashreplace"):
        if ch == "\x1b":
            out.append("\\e")
        elif ch == "\n":
            out.append("\\n\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch < " " or ch == "\x7f":
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


STATUS_HINTS = {
    "timeout": "still running; the `wait` tool keeps waiting, `send` keys=['ctrl+c'] interrupts",
    "settled": "no output for a while and no prompt seen -- the program may still be working; call the `wait` or `read` tool later",
    "fullscreen": "the program is full-screen; drive it with the `send` tool",
    "waiting": "output stopped mid-line: a question to answer (the `run` tool with the answer as input, or the `expect` tool), or a prompt "
    "the pattern does not know (if it is the program's prompt, pass prompt_pattern -- here or in session_create)",
    "exited": "the program in this session has exited",
    "until": "`until` matched; the command may still be running, and while it does this session is busy: run other "
    "commands in another session (session_create, or session_split beside this one).  The `wait` tool keeps waiting "
    "for it, the `read` or `search` tool shows more, `send` keys=['ctrl+c'] interrupts",
    "continuation": "the shell is showing its continuation prompt: the command is incomplete "
    "(unclosed quote, heredoc or block). Send the rest with the `run` tool, or `send` keys=['ctrl+c'] to abandon it",
}


# A yes/no question and its default: "(y or n)", "(y or [n])", "[Y/n]",
# "(yes/no)".  Answering it with Enter alone takes the default -- for gdb's
# pending-breakpoint question, no: the breakpoint is silently not set.
GDB_YES_NO = re.compile(r"\((?:\[y\]|y) or (?:\[n\]|n)\)\s*$")  # gdb's own style: "(y or n)", "(y or [n])"
YES_NO = re.compile(r"[(\[]\s*(\[?)(y|yes)(\]?)\s*(?:or|/)\s*(\[?)(n|no)(\]?)\s*[)\]]\s*[?:]?\s*$", re.IGNORECASE)
YES_NO_ANSWER = re.compile(r"y|yes|n|no", re.IGNORECASE)


def yes_no_hint(text: str) -> str | None:
    """For a cursor line ending in a yes/no question: how to answer it."""
    m = YES_NO.search(text.rstrip() + " ")
    if not m:
        return None
    yes_default = bool(m.group(1)) or m.group(2)[0] == "Y"
    no_default = bool(m.group(4)) or m.group(5)[0] == "N"
    default = "yes" if yes_default and not no_default else "no" if no_default and not yes_default else None
    enter = f"; Enter alone answers {default}" if default else ""
    hint = (f"[a yes/no question: answer it with `run` input=\"y\" or \"n\"{enter} -- or, when a command "
            f"may ask, the `expect` tool answers in the same call]")
    if GDB_YES_NO.search(text.rstrip() + " "):
        # How to stop gdb asking, usable at once at its prompt: each of these
        # cost a model a round trip in testing.
        if "debuginfod" in text:
            hint += ("\n[\"n\" turns debuginfod off for this gdb; a gdb started with "
                     "-iex 'set debuginfod enabled off' never asks]")
        else:
            hint += "\n[`set confirm off` at gdb's prompt stops such questions in this gdb]"
    return hint


# Pagers stop output mid-line like a question; each gets its own answer,
# never the offer of a prompt_pattern (the eval: gdb's pager was a model's
# main friction in gdb-under-gdb).  gdb's reads a whole line; more's a key.
PAGER_HINTS = (
    (re.compile(r"--Type <RET> for more, q to quit, c to continue without paging--\s*$"),
     "[gdb's pager, not a question: the `run` tool with input=\"c\" shows the rest without stopping, "
     "input=\"q\" ends this listing, input=\"\" one more page.  gdb started with "
     "-iex \"set pagination off\" (or `set pagination off` typed at its prompt) never stops so]"),
    (re.compile(r"---Type <return> to continue, or q <return> to quit---\s*$"),
     "[gdb's pager, not a question: the `run` tool with input=\"\" shows one more page, input=\"q\" "
     "ends this listing.  `set pagination off` at gdb's prompt turns it off]"),
    (re.compile(r"--More--(\(\d+%\))?\s*$"),
     "[a pager (more), not a question: `send` keys=[\" \"] shows the next page, `send` keys=[\"q\"] ends it]"),
    # less kept off the alternate screen (-X: systemctl's, git's LESS=FRX)
    # is no full-screen program; its prompt with -M, or at the end.
    (re.compile(r"(^lines \d+-\d+(/\d+)?( \d+%)?( \(END\))?|\(END\))\s*$"),
     "[a pager (less), not a question: `send` keys=[\" \"] shows the next page, `send` keys=[\"q\"] ends it.  "
     "--no-pager (systemctl, journalctl, git) or PAGER=cat before the command keeps it from stopping]"),
)


# What says a program stopped somewhere else than its prompt was left at:
# gdb's signal and exit reports (the inferior's, or a called function's).
STOP_EVENT = re.compile(r"(?:Program|Thread \d+ \"[^\"]*\") received signal \w+|"
                        r"The program being debugged was signaled|Program terminated with signal \w+|"
                        r"\[Inferior \d+ \(process \d+\) exited[^\]]*\]")


# Programs with a prompt of their own, which can run one another (gdb
# under gdb, python under gdb).
INTERACTIVE = ("gdb", "lldb", "python", "pdb", "ipython", "node", "guile", "sqlite3", "psql")


async def nesting_note(s: Session) -> str | None:
    """Which program reads the next line, when it runs under another one of
    its kind -- prompt text cannot tell gdb from gdb under gdb: an inner
    gdb stopped at its own "(gdb)" was taken for the outer one, and got
    the outer one's commands.  From the processes:
    the terminal's foreground program and the chain up to the shell."""
    try:
        info = await s.tmux.pane_info(s.name)
        procs = await foreground_processes(info.tty)
    except (TmuxError, OSError):
        return None
    if not procs or info.dead:
        return None
    chain = process_chain(min(p.pgid for p in procs), info.pid)
    programs = [p for p in chain if p.comm.startswith(INTERACTIVE)]
    if len(programs) < 2:
        return None
    inner, outer = programs[0], programs[1]

    def who(p) -> str:
        exe = verified_exe(p.pid)
        return f"{p.comm} (pid {p.pid}" + (f", {exe})" if exe else ")")

    note = (f"[the program at this prompt is {who(inner)}, running under {who(outer)} -- the next line goes "
            f"to the inner one, pid {inner.pid}]")
    if (inner.comm.startswith("gdb") and outer.comm.startswith("gdb") and s.term.cursor_text().strip() == "(gdb)"
            and "(top-gdb)" not in s.term.text_between((max(0, s.term.row - 400), 0), s.content_end())):
        # What people do: give the outer gdb a prompt of its own, as
        # gdb's build tree does with gdb-gdb.gdb -- then the prompt itself
        # tells them apart.
        note += ("\n[both gdbs print \"(gdb)\": give the outer one a prompt of its own -- `set prompt (top-gdb) ` "
                 "at its prompt, or -ex 'set prompt (top-gdb) ' when starting it, as gdb's build tree does with "
                 "gdb-gdb.gdb -- and the prompt itself tells them apart]")
    return note


FEDORA_RPMS = re.compile(r"^Missing rpms, try: dnf ", re.M)


def fedora_rpms_note(s: Session) -> str | None:
    """Fedora's gdb prints "Missing rpms, try: dnf ... install <dozens of
    packages>" -- some 2,000 characters -- at every stop, pushing what the
    reply was for into its clipped middle.
    Once per session for each agent: the setting that silences it."""
    key = f"{ACTOR.get()}:fedora_rpms"
    if key in s.habits_noted:
        return None
    if not any(FEDORA_RPMS.search(s.region_text(r)) for r in REPLY_REGIONS.get() or []):
        return None
    s.habits_noted.add(key)
    s._save_state()
    return ("[Fedora gdb's missing-package hint (\"Missing rpms, try: dnf ...\") comes at every stop: "
            "`set rpm-suggestion enabled off` at the prompt of the gdb that printed it silences it (or -iex "
            "'set rpm-suggestion enabled off' when starting that gdb; a gdb built from upstream sources has no "
            "such setting)]")


async def deleted_cwd_note(s: Session) -> str | None:
    """The program in front, or the session's shell, still in a directory
    that has since been deleted: its errors then name something else (gdb
    started in a testsuite output directory that the next `make check`
    deleted died with a std::logic_error from
    current_directory, chased as a gdb bug).  Once per process and
    directory, for each agent."""
    try:
        info = await s.tmux.pane_info(s.name)
        procs = [] if info.dead else await foreground_processes(info.tty)
    except (TmuxError, OSError):
        return None
    if info.dead:
        return None
    pids = [p.pid for p in procs if p.pid == p.pgid][:1] + [info.pid]
    notes = []
    for pid in dict.fromkeys(pids):
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            continue  # gone, or not ours to look at
        if not cwd.endswith(" (deleted)"):
            continue
        path = cwd.removesuffix(" (deleted)")
        key = f"{ACTOR.get()}:{pid}:{path}"
        if key in s.cwd_noted:
            continue
        s.cwd_noted.add(key)
        s._save_state()
        how = ('`cd "$PWD"` in it' if comm in SHELLS
               else "gdb's own `cd` command" if comm.startswith("gdb") else "`cd` (if it has one)")
        notes.append(f"[{comm} (pid {pid}) is in a directory that has been deleted since it went there: {path}.  "
                     f"Errors about the current directory come from that, not from {comm}.  If the directory was "
                     f"made again (a testsuite remakes its output directories), {how} goes to the new one; "
                     "otherwise move to a directory that stays]")
    return "\n".join(notes) or None


def verified_exe(pid: int) -> str | None:
    """The binary a process runs, only where that is certain: our own user's
    process, seen in our own filesystem view (not inside a container or
    sandbox, where the path means something else).  "(deleted)" stays: the
    running program is not the one on disk any more."""
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return None
        if os.readlink(f"/proc/{pid}/ns/mnt") != os.readlink("/proc/self/ns/mnt"):
            return None
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


def prompt_change_note(s: Session, region: Region) -> str | None:
    """When the program came back to another prompt than the one the input
    was typed at -- gdb under gdb: typed at (top-gdb), the inner gdb's
    (gdb) is the one waiting now -- say so: replies show no prompt line, so
    which program reads the next line was the model's guess."""
    if region.kind != "input" or not region.command:
        return None
    before = s.prompt_start(region.command)
    now = s.prompt_here(s.prompt_pattern)
    if before is None or before.start() != 0 or now is None:
        return None
    was, now_text = before.group(0).strip(), now.group(0).strip()
    # A bare sign (a continuation prompt, a label shortened to "$ cmd")
    # says nothing of which program it was.
    signs = (">", "...", "...:", "$", "#", "%")
    if not was or not now_text or was == now_text or now_text in signs or was in signs:
        return None
    if s._prompt_kind(before) == s._prompt_kind(now) == "sh":
        return None  # a shell's prompt changes with its directory
    return (f"[now at a different prompt: {now_text!r}, not {was!r} -- the next line goes to the program "
            f"at {now_text!r}]")


def question_hint(text: str) -> str | None:
    """How to answer what the cursor line asks: a yes/no question, or a pager."""
    for rx, hint in PAGER_HINTS:
        if rx.search(text):
            return hint
    return yes_no_hint(text)


DEFAULT_PROMPT = prompt_regex(None)

SIGNALS = {name: getattr(signal_module, f"SIG{name}") for name in ("TERM", "KILL", "INT", "HUP", "QUIT")}
SIGNALS_BY_NUMBER = {v: k for k, v in SIGNALS.items()}
INTERRUPT_KEYS = ("C-c",)  # (ctrl+\ is as often a program's escape key: kermit's, telnet's)


async def foreground_jobs(s: Session) -> list[Process]:
    """What typed keys reach now, if not the session's own shell or program
    (the pane's process): a command it runs, or a program's inferior."""
    info = await s.tmux.pane_info(s.name)
    if info.dead:
        return []
    return [p for p in await foreground_processes(info.tty) if p.pid != info.pid and p.pgid != info.pid]


def named(procs: list[Process], limit: int = 4) -> str:
    shown = ", ".join(f"{p.command} (pid {p.pid})" for p in procs[:limit])
    return shown + (f" and {len(procs) - limit} more" if len(procs) > limit else "")


# A program that handles SIGINT itself (gdb, python) and is busy acts on
# ctrl+c only when it next checks -- gdb printing a large array, about 50 s
# later (a field report).  ctrl+d or `quit` typed now would be
# read at its next prompt and end it; TERM would end it mid-command.
DEFERRED_INTERRUPT = (
    "[{start} {who} is still busy, not at a prompt: a program that handles ctrl+c itself "
    "(gdb, python) acts on it when it next checks, which can be much later.  The `wait` tool shows it "
    "stop.  Don't type ahead: ctrl+d or quit would be read at its next prompt and end it.  "
    "`send` signal=\"TERM\" only if losing the program's state is acceptable]")


async def foreground_busy(s: Session) -> bool:
    """Whether the foreground job is working (running, or sleeping on
    something other than terminal input), as ps sees it."""
    info = await s.tmux.pane_info(s.name)
    return await foreground_state(info.tty, 0) == "busy"


async def foreground_notes(s: Session, before: list[Process], interrupt: list[str],
                           signum: int | None, quiet: bool) -> list[str]:
    """What is in the foreground after `send`, when the reply would not
    show it: a job that survived an interrupt or a signal (at its own
    prompt, gdb takes ctrl+c as "Quit" and stays), or, with no new output,
    whatever the input went to."""
    after = await foreground_jobs(s)
    # ssh staying says nothing about the remote program ctrl+c went to.
    survivors = [p for p in after if p.pid in {b.pid for b in before} and not is_carrier(p.pid, p.command)]
    if survivors and signum is not None:
        more = "; signal=\"KILL\" ends it" if signum != SIGNALS["KILL"] else ""
        return [f"[still running after SIG{SIGNALS_BY_NUMBER[signum]}: {named(survivors)}{more}]"]
    if survivors and interrupt and not s.prompt_here(s.prompt_pattern):
        s.note_interrupt_survivors({p.pid for p in survivors})
        if await foreground_busy(s):
            return [DEFERRED_INTERRUPT.format(who=named(survivors), start="ctrl+c delivered, but")]
        return [f"[still in the foreground after ctrl+c: {named(survivors)} -- keys and text sent now "
                f"go to it.  It is waiting for input and handled ctrl+c itself: gdb at its own prompt "
                f"takes it as \"Quit\" and stays, unseen if its output goes to a pipe.  ctrl+d or its "
                f"quit command ends such a program; otherwise `send` signal=\"TERM\"]"]
    if quiet and after and not all(is_carrier(p.pid, p.command) for p in after):
        return [f"[no new output; in the foreground: {named(after)} -- what was sent went to it]"]
    return []


# One-shot shell habits that a session makes unnecessary, remarked on in the
# reply the first time an agent uses each in a session: the reply is the one
# text a model reads at the moment the habit costs something (in an
# evaluation the same advice in the instructions moved nothing, a pattern
# offered in a reply took prompt_pattern misuse from 107 calls to 7).
GDB_LAUNCH = re.compile(r"(?:^|[;&|(]\s*)(?:\w+=\S*\s+)*(?:[\w.~/$-]*/)?gdb(?=\s|$)")
REDIRECTS = {">", ">>", "<", ">&", "&>", "&>>", ">|", "<<<"}
COMMAND_ENDS = {";", "&&", "||", "|", "&", "|&", "(", ")"}
BATCH_FLAGS = {"-batch", "--batch", "-batch-silent", "--batch-silent"}
GDB_INFO_ONLY = {"--version", "-version", "--configuration", "-configuration", "--help", "-help"}


def gdb_launch(line: str) -> tuple[list[str], int, bool, int] | None:
    """The tokens of a shell line that starts gdb, the index of the gdb word,
    whether a pipe feeds it (`printf ... | gdb`), and where its command
    starts (VAR=value words before it)."""
    if not GDB_LAUNCH.search(line):
        return None
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return None
    for i, t in enumerate(toks):
        if os.path.basename(t) == "gdb" and (i == 0 or toks[i - 1] in COMMAND_ENDS or "=" in toks[i - 1]):
            j = i - 1
            while j >= 0 and "=" in toks[j] and toks[j] not in COMMAND_ENDS:
                j -= 1  # VAR=value words before it
            return toks, i, j >= 0 and toks[j] in ("|", "|&"), j + 1
    return None


def live_gdb_line(line: str) -> str | None:
    """`line`'s gdb command as it would run live in a session: without
    -batch, without what pipes into it or takes its output (a pipe, a
    redirect), commands fed on its stdin from a file given with -x."""
    found = gdb_launch(line)
    if found is None:
        return None
    toks, i, _, first = found
    out: list[str] = toks[first:i]
    k = i
    while k < len(toks) and toks[k] not in COMMAND_ENDS:
        t = toks[k]
        if t in REDIRECTS:
            target = toks[k + 1] if k + 1 < len(toks) else ""
            if out and out[-1].isdigit():
                out.pop()  # the fd of 2>&1, 2>file
            if t == "<" and target:
                out += ["-x", target]
            k += 2
            continue
        if t not in BATCH_FLAGS:
            out.append(t)
        k += 1
    return shlex.join(out)


def gdb_one_shot(line: str) -> bool:
    """Does `line` start gdb to run once and exit: -batch, or commands
    piped or redirected into it?  (Not --version and the like.)"""
    found = gdb_launch(line)
    if found is None:
        return False
    toks, i, piped_in, _ = found
    words = toks[i:]
    if end := next((n for n, t in enumerate(words) if t in COMMAND_ENDS), None):
        words = words[:end]
    if GDB_INFO_ONLY & set(words):
        return False
    return piped_in or "<" in words or bool(BATCH_FLAGS & set(words))


def gdb_one_shot_note(line: str) -> str:
    live = live_gdb_line(line)
    # A trade-off, not a rule: for one question -batch is right; for a
    # series, a live gdb.  (Qwen, told to go live, added a prompt_pattern
    # it thought it needed: 2 of b4e96f5's 5 trials that passed one.)
    return ("[note: -batch suits a single question; for a series about the same program, start it once live"
            + (f" -- `{live}` --" if live else "")
            + " and ask each at its (gdb) prompt with the `run` tool, no prompt_pattern needed (its prompt is "
            "known): the session keeps it up between calls]")


REDIRECT = re.compile(r"(?<![<\d])>>?\s*(?P<a>[\w./~$-]+).*2>&1|&>>?\s*(?P<b>[^\s;|&)]+)")


def redirect_unread(line: str) -> bool:
    """Output sent to a file, and not read back on the same line: a
    `> out 2>&1; grep x out` reads it at once, and its grep reaches the
    log (the note on each of those was noise)."""
    m = REDIRECT.search(line)
    if not m:
        return False
    target = (m["a"] or m["b"]).strip("\"'")
    return not re.search(rf"(?<![\w./~$-]){re.escape(target)}(?![\w./~$-])", line[m.end():])


HABITS = [
    ("gdb_batch", gdb_one_shot, gdb_one_shot_note),
    ("pipe_tail",
     lambda line: re.search(r"\|\s*(tail|head)(\s|$)", line),
     # Stated as the trade-off, not a rule: pipes have their good uses, and
     # too much re-education costs goodwill.
     "[note: with `| tail`/`| head` only those lines reach the log (and a human watching); without it a reply "
     "still shows the first and last lines of long output, and the log keeps everything for the `read` and "
     "`search` tools and `show=REGEX`.  `last_lines=N` shows the same lines as `| tail -N` and keeps the rest]"),
    ("redirect_log",
     # (`| tee FILE` keeps the output on the terminal too: nothing to say.)
     redirect_unread,
     # A file of one's own for later is a fine choice (people redirect to
     # grep or diff later; a model kept baseline and fixed runs to compare)
     # -- say what the region keeps, and the way to have both.  The fact
     # alone, no advice (a `| tee` suggestion still pulled against a model's
     # deliberate choice).  Naming save_to is still no advice against the
     # file (models knew save_to and redirected anyway): it gives both.
     "[note: this output goes to the file, not this session's log -- the `read` and `search` tools won't "
     "find it there.  `save_to=FILE` gives both, the file and the log]"),
    # A job put in the background inside a session: `wait` cannot wait for
    # it, and the model went on to poll from Bash with `sleep 580`, six
    # times.  Said as the job starts.
    ("background_job",
     lambda line: re.search(r"(?<![&|>])&\s*(?:#.*)?$", line.strip()),
     "[note: a job started with `&` cannot be waited for -- the `wait` tool waits for the foreground.  For its "
     "output and exit status in this session, run it without `&` and with timeout_s up to 1500: the reply comes "
     "when it ends (a client that moves long calls to the background, as Claude Code does, tells you then), and "
     "other work goes on in another session (session_create, session_split)]"),
    # Not a one-shot habit but a trap models fell into: typed at the session's shell, `pgrep -f P`
    # does not see itself, but a wrapper keeps the whole line in its own
    # arguments, where P matches too -- pkill then kills the wrapper,
    # pgrep counts it.
    ("pkill_self",
     lambda line: re.search(r"\bp(?:kill|grep)\b[^;&|]*\s(?:-[a-zA-Z]*f[a-zA-Z]*|--full)(?=\s)", line)
     # (Wrappers that stay as the parent; env, nice, nohup, stdbuf exec it.)
     and re.search(r"(?:^|[\s;&|(])(?:(?:sudo|timeout|watch|flock)\s[^;&|]*"
                   r"|(?:ba|z|da|k)?sh\s+(?:-\w+\s+)*-\w*c\s.*)\bp(?:kill|grep)\b", line),
     "[note: `pkill -f`/`pgrep -f` match whole command lines, and the wrapper in front (sudo, timeout, sh -c ...) "
     "holds this very line in its arguments, so the pattern matches it too: pkill can kill its own wrapper, "
     "pgrep counts it.  Match the name exactly (-x), or write the pattern so it cannot match itself ('[m]yserver')]"),
]


SEND_LINES_NOTE = (
    "[note: the `send` tool types all its lines at once, without waiting: a line typed while the program is busy "
    "goes to whatever reads next -- the program it started, or a (y or n) question.  The `run` tool with several lines "
    "sends them one at a time, each after the prompt comes back, and stops at a question]")
PATTERN_AT_LAUNCH_NOTE = (
    "[note: prompt_pattern was not needed to start the program: the default recognises (gdb), (top-gdb), "
    ">>>, IPython's In [n]:, (Pdb), (lldb) and prompts ending in $ # % >.  Pass one only after a reply came back \"waiting\" at a "
    "prompt it did not know -- that reply offers the pattern]")


# Prompts the default recognises: a prompt_pattern that matches one of them
# adds nothing, wherever the reply ends -- at that prompt, or at a question
# before it (gdb's "Quit anyway? (y or n)" after -ex quit: an eval's Qwen
# sent `\(gdb\) $` twice and heard nothing, the note tied to the prompt).
KNOWN_PROMPTS = ("(gdb) ", "(top-gdb) ", ">>> ", "In [1]: ", "(Pdb) ", "(lldb) ")


def pattern_known(pattern: str | None) -> bool:
    """Does `pattern` describe a prompt the default already recognises?"""
    if not pattern:
        return False
    try:
        rx = re.compile(pattern)
    except re.error:
        return False
    return any(rx.search(p) and DEFAULT_PROMPT.search(p) for p in KNOWN_PROMPTS)


PATTERN_INTO_KNOWN_NOTE = (
    "[note: prompt_pattern was not needed for this line: the program's prompt is one the default recognises, "
    "so `run` already waits for it.  Pass one only after a reply came back \"waiting\" at a prompt it did not "
    "know -- that reply offers the pattern]")


HABITS_NOTED: set[str] = set()  # "sessions dir:agent:habit" noted in any session of that server


def note_once(s: Session, habit: str, text: str) -> list[str]:
    """`text`, the first time this agent shows `habit` in this session."""
    key = f"{ACTOR.get()}:{habit}"
    if key in s.habits_noted:
        return []
    s.habits_noted.add(key)
    s._save_state()
    return [text]


# A `| tail`/`| head` stage: at the end of the line, or of one command of a
# `;`/`&&`/`||` list (`cp ...; make check | tail -3; cp ...`).
PIPE_STAGE = re.compile(r"\|\s*(tail|head)\b([^|;&]*)(?=$|;|&&|\|\|)")
# Wrappers before the command that does the work (`time make check`).
WRAPPERS = {"time", "nice", "env", "nohup", "stdbuf", "ionice", "xvfb-run", "command", "exec", "sudo"}


def runs_tests(command: str) -> bool:
    """Does this one command (no pipes or lists) run a test suite?  By its
    command word, so `pip install pytest` is not one:
    make/gmake with a check or test target, pytest, py.test, ctest,
    runtest, tox, nox, `cargo test`, `go test`, `meson test`,
    `python -m pytest|test|unittest`."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    while words and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or os.path.basename(words[0]) in WRAPPERS
                     or (words[0] == "timeout" and len(words) > 1)):
        if words[0] == "timeout":
            words = words[1:]
            while words and (words[0].startswith("-") or re.match(r"^\d+[smhd]?$", words[0])):
                words = words[1:]
            continue
        words = words[1:]
        while words and words[0].startswith("-"):  # env -i, nice -n 5 (its number goes next)
            words = words[1:]
            if words and re.match(r"^-?\d+$", words[0]):
                words = words[1:]
    if not words:
        return False
    prog, rest = os.path.basename(words[0]), words[1:]
    if prog in ("make", "gmake"):
        return any(w in ("check", "test", "tests") or w.startswith(("check-", "test-")) for w in rest
                   if not w.startswith("-") and "=" not in w)
    if prog in ("pytest", "py.test", "ctest", "runtest", "tox", "nox"):
        return True
    if prog in ("cargo", "go", "meson"):
        return rest[:1] == ["test"]
    if re.match(r"^python[\d.]*$|^pypy[\d.]*$", prog) or prog == "python":
        return any(a == "-m" and b in ("pytest", "test", "unittest") for a, b in zip(rest, rest[1:]))
    return False


def piped_stage(s: Session, r: Region) -> tuple[re.Match, bool] | None:
    """The `| tail`/`| head` stage of a finished command of the caller's that
    cut a test run's output, and that a reply without it would have shown
    (the stage straight after the test command -- after a `| grep`, the
    grep chose the lines), with True; or another such stage, with False.
    None when there is none."""
    line = (r.command or "").split("\n", 1)[0].strip()
    if not (r.kind == "command" and r.done and r.source == ACTOR.get()):
        return None
    other = None
    for m in PIPE_STAGE.finditer(line):
        before = re.split(r";|&&|\|\||\bdo\b|\bthen\b", line[:m.start()])[-1]
        stages = before.split("|")
        if len(stages) > 1:
            continue  # `cmd | grep x | head`: not what the pipe alone took away
        if runs_tests(re.sub(r"\d?>&\d|\d?>\s*/dev/null", "", stages[0]).strip()):
            return m, True
        other = other or m
    if other is None:
        return None
    # A test runner the list above does not know (a script, a harness of the
    # project's own): its output says so -- when the line has this one stage.
    if len(list(PIPE_STAGE.finditer(line))) == 1 and any(
            summary.is_test_result(t) for _, t in s.iter_span(r.start, s.region_end(r))):
        return other, True
    return other, False


def pipe_count(m: re.Match) -> str:
    n = re.search(r"-n\s*\+?(\d+)|-(\d+)\b", m.group(2))
    return (n.group(1) or n.group(2)) if n else "10"


def pipe_hid_matches(s: Session, regions: list[Region], show: re.Pattern | None) -> list[str]:
    """`show` behind a `| tail`/`| head`: the pipe let only those lines reach
    the session, so `show` never saw the rest -- said when it matched nothing
    (`make check | tail -2` with show='^(FAIL|ERROR):': "0 of 2 lines
    match", then gdb.sum grepped from another shell), and on a test run even
    when it matched the totals ("# of unexpected failures 2", the FAIL lines
    cut).  Said each time: it
    is a loss, not a habit."""
    out = []
    for r in regions:
        if not show or (found := piped_stage(s, r)) is None:
            continue
        m, tests = found
        matched = any(show.search(t) for _, t in s.iter_span(r.start, s.region_end(r)))
        if matched and not tests:
            continue
        count, pipe = pipe_count(m), m.group(0).strip()
        lines = f"{count} line{'' if count == '1' else 's'}"
        if m.group(1) == "tail":
            instead = (f"With `last_lines={count}` in place of the pipe, the reply shows the same lines and "
                       "`show` sees all of the output")
        else:
            instead = "Without the pipe, the reply keeps the first and last lines and `show` sees all of the output"
        what = (f"`show` matched nothing in region {r.id}, but" if not matched
                else f"`show` saw only part of this test run in region {r.id}:")
        out.append(f"[{what} `{pipe}` let only {lines} of the output reach the session -- the rest, its FAIL lines "
                   f"among it, never got here.  {instead}]" if matched else
                   f"[{what} `{pipe}` let only {lines} of the output reach the session -- the rest never got here.  "
                   f"{instead}]")
    return out


def pipe_notes(s: Session, regions: list[Region], show: re.Pattern | None, last_lines: int | None) -> list[str]:
    """What a `| tail`/`| head` at the end of a finished command cost this
    reply -- from `run`, and from the `wait` that sees it end (a long
    `make check | tail -5` started with timeout_s=60 ended in a `wait`,
    which said nothing)."""
    return pipe_hid_matches(s, regions, show) + (
        pipe_hid_summary(s, regions) if show is None and last_lines is None else [])


def pipe_hid_summary(s: Session, regions: list[Region]) -> list[str]:
    """A test run piped through `| tail`/`| head`: what the reply would
    have shown without the pipe.  Said each time, as concretely as that
    (a model skimmed the general `| tail` note and piped every make
    check, so the summary block never fired; asked, it said "if it had said
    concretely 'without the pipe this reply would have led with the FAIL
    lines', I'd have dropped the pipe")."""
    out = []
    for r in regions:
        if (found := piped_stage(s, r)) is None or not found[1]:
            continue
        pipe = found[0].group(0).strip()
        out.append(f"[without `{pipe}`, this reply would have led with the run's FAIL lines and totals, picked "
                   "out of the whole output -- and the log would keep all of it for `search`]")
    return out


def habit_notes(s: Session, command: str, regions: list[Region]) -> list[str]:
    """The notes for the one-shot habits in a shell command, each once per
    agent.  (A repeat of the gdb note at every 3rd one-shot run was tried
    and dropped: not a lever for the local models, noise to a
    model running -batch A/B on purpose.)"""
    if not regions:
        return []
    line = command.split("\n", 1)[0]
    out = []
    for name, matches, text in HABITS:
        key = f"{ACTOR.get()}:{name}"
        # Once per agent, in any session: a model piping on purpose saw the
        # same note again in each new session.
        server_key = f"{s.dir.parent}:{key}"  # (per state directory: one server's sessions)
        if key not in s.habits_noted and server_key not in HABITS_NOTED and matches(line):
            s.habits_noted.add(key)
            HABITS_NOTED.add(server_key)
            out.append(text(line) if callable(text) else text)
    return out


# Second guesses: a line refused once, before anything is typed, with the
# call that serves better -- the same line sent again runs as it is, and
# so does any later line of that habit from that agent.  The notes
# above come after the command and did not change a habit within a run
# (four tailcall runs: "did it anyway out of shell habit"); a refusal
# comes before, and costs at most one call per habit.  Kevin's idea, from
# Claude Code's refusal of `sleep N; cat`.
SECOND_GUESS_ON = os.environ.get("DOCENT_TERMINAL_MCP_SECOND_GUESS", "1") != "0"
GUESS_ASKED: set[str] = set()  # "sessions dir:agent:habit" refused once: not again
# The note a habit's refusal stands in for (said there; not again after).
GUESS_NOTES = {"pipe_tail": "pipe_tail", "redirect": "redirect_log", "background": "background_job"}

def line_segments(line: str) -> list[tuple[int, int]]:
    """The spans of a line's commands between top-level `;`, `&&` and `||`
    (not inside quotes, $( ), ( ) or { }): where a pipe or redirect ends one
    that is not the last."""
    spans, start, depth, quote, i = [], 0, 0, "", 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 1
            elif c == quote:
                quote = ""
        elif c == "\\":
            i += 1
        elif c in "'\"":
            quote = c
        elif c in "({":
            depth += 1
        elif c in ")}":
            depth = max(depth - 1, 0)
        elif depth == 0 and (c == ";" or line[i:i + 2] in ("&&", "||")):
            spans.append((start, i))
            i += 1 if c == ";" else 2
            start = i
            continue
        i += 1
    spans.append((start, len(line)))
    return spans


# `| tail`/`| head` ending a command that is not the line's last (`make | tail
# -5; echo done`): an eval's Qwen, 18 times in d2a4031's rounds.
MID_PIPE = re.compile(r"\s*\|\s*(?P<what>tail|head)(?:\s+(?:-n\s*|-)\d+)?\s*$")

# `| tail [-n] N` ending the line, after an optional `| tee FILE`.
TAIL_END = re.compile(r"^(?P<cmd>.*?\S)(?:\s*\|\s*tee\s+(?P<tee>[^\s|;&<>-][^\s|;&<>]*))?"
                      r"\s*\|\s*tail(?:\s+(?:-n\s*|-)(?P<n>\d+))?\s*$")
# `> FILE 2>&1` or `&> FILE` ending the line (not >>: save_to rewrites the file),
# a `;` after it too (an eval's `gdb ... -ex q > /tmp/sysgdb.txt 2>&1;` ran unasked).
REDIRECT_END = re.compile(r"^(?P<cmd>.*?\S)\s*(?:(?<![<>&\d])>\s*(?P<a>[^\s;|&<>]+)\s+2>&1|(?<!&)&>\s*(?P<b>[^\s;|&<>]+))\s*;?\s*$")
BACKGROUND_END = re.compile(r"(?<![&|>])&\s*$")
# `sleep N; cat log`: polling for something to happen.
SLEEP_POLL = re.compile(r"^\s*sleep\s+(?P<n>\d+(?:\.\d+)?)(?P<unit>[smh]?)\s*(?:;|&&)\s*(?P<rest>\S.*)$")
# `set -e` (or -o errexit) at the line's top level, not inside ( ) or bash -c.
SET_ERREXIT = re.compile(r"^\s*(?:[^()'\"]*[;&|]\s*)?set\s+(?:-[a-zA-Z]*e[a-zA-Z]*|-o\s+errexit)\b")
POLL_READERS = {"cat", "tail", "head", "grep", "egrep", "rg", "ls", "wc", "ps", "pgrep", "stat", "du", "jobs",
                "curl", "ss", "netstat", "test", "[", "less", "tac", "file", "find", "nvidia-smi", "squeue"}


def second_guess(s: Session, habit: str, line: str, why: str, instead: str) -> None:
    """Refuse `line` with why and what instead, the first time this agent
    shows `habit` on this server; after that, the habit's lines run -- the
    refused one sent again, and any other.  (Until 5c3dd1b only the same
    line sent again ended it, and each new variant was refused anew: an
    eval's Qwen got six redirect refusals in one trial, each "this happens
    once".)"""
    key = f"{s.dir.parent}:{ACTOR.get()}:{habit}"
    if not SECOND_GUESS_ON or key in GUESS_ASKED:
        return
    GUESS_ASKED.add(key)
    if habit in GUESS_NOTES:
        HABITS_NOTED.add(f"{s.dir.parent}:{ACTOR.get()}:{GUESS_NOTES[habit]}")
    raise ToolError(
        # (Was "not typed -- a second guess, made once: ...", the way out last:
        # Kevin found it open to reading, and in the evals no model resent a line.)
        f"Refused, not run (this happens once; sending the same line again runs it): {why}  {instead}")


def as_run(s: Session, **args) -> str:
    """A `run` call to copy: `run` session="s" input="..." last_lines=5."""
    return "`run` " + " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}"
                               for k, v in {"session": s.name, **args}.items())


async def own_command(s: Session) -> str | None:
    """How the shell at the prompt runs this server's own command (`dtmcp`
    when that shell's PATH finds this server's shim, else the shim's full
    path) -- only for a shell on this host, as this user: one elsewhere, or
    of another user, may not have it or not be able to run it.  The PATH
    is the shell's as it started (/proc/PID/environ)."""
    shim = s.dir.parent.parent / "bin" / SHORT
    try:
        info = await s.tmux.pane_info(s.name)
        shells = [p for p in await foreground_processes(info.tty) if os.path.basename(p.command.lstrip("-")) in SHELLS]
        if len(shells) != 1 or os.stat(f"/proc/{shells[0].pid}").st_uid != os.getuid() or not os.access(shim, os.X_OK):
            return None
        environ = Path(f"/proc/{shells[0].pid}/environ").read_bytes().split(b"\0")
    except (TmuxError, OSError):
        return None
    path = next((e[5:].decode(errors="replace") for e in environ if e.startswith(b"PATH=")), "")
    found = shutil.which(SHORT, path=path)
    try:
        return SHORT if found and os.path.samefile(found, shim) else shlex.quote(str(shim))
    except OSError:
        return shlex.quote(str(shim))


SSH_NO_SHELL = set("NfWOTGVQ")  # ssh options with no interactive shell at the far end
SU_VALUE_OPTS = {"-s", "--shell", "-g", "--group", "-G", "--supp-group", "-w", "--whitelist-environment"}


def hop_without_integration(line: str) -> str | None:
    """"ssh" or "su" when `line` is a plain interactive ssh or su -- which
    leaves a shell without the integration at the far end."""
    from .remote import SU_COMMAND_OPTS, split_destination
    if not re.fullmatch(r"[\w@.:/=+,~%-]+(?: +[\w@.:/=+,~%-]+)*\s*", line.strip()):
        return None  # (a list, a pipe, quoting, a variable: not the plain hop)
    argv = line.split()
    if argv[0] == "ssh":
        parts = split_destination(argv[1:])
        if parts is None or parts[2] or any(a.startswith("-") and not a.startswith("--") and SSH_NO_SHELL & set(a[1:])
                                            for a in parts[0]):
            return None
        return "ssh"
    if argv[0] == "su":
        args, users = argv[1:], []
        while args:
            a = args.pop(0)
            if a in SU_COMMAND_OPTS or a.startswith("--command") or a.startswith("--session-command"):
                return None
            if a in SU_VALUE_OPTS:
                args = args[1:]
            elif a == "-" or not a.startswith("-"):
                users += [a] if a != "-" else []
        return "su" if len(users) <= 1 else None
    return None


async def second_guesses(s: Session, line: str, last_lines: int | None, save_to: str | None,
                         tool: str = "run") -> None:
    """The second guesses for a line typed at a shell's prompt.  For
    `expect`'s input, those about where the line runs and the shell it
    leaves -- a plain ssh or su, a shell without the integration, `set -e`:
    the others offer `run`'s arguments (last_lines, save_to), and an
    expect's line is mostly a program that asks.  (A bare `ssh coyote.lan`
    went through expect unasked: an adjutant assistant's check.)"""
    if "\n" in line:
        return
    if (hop := hop_without_integration(line)) and (cmd := await own_command(s)):
        what = "the host" if hop == "ssh" else "the user"
        instead = (as_run(s, input=cmd + " " + line.strip()) if tool == "run" else
                   f"`expect` session={json.dumps(s.name)} input={json.dumps(cmd + ' ' + line.strip())} with the "
                   f"same respond rules")
        second_guess(
            s, f"hop_{hop}", line,
            f"a plain `{hop}` leaves a shell without the shell integration on {what}'s side: replies there "
            "cannot tell when a command ends or how it ended (no exit status; a quiet command comes back "
            "\"settled\" as if done), and further hops go unseen.",
            f"`{NAME} {hop}` takes the same arguments and carries the integration along for this {hop} (nothing "
            f"is installed): {instead}.")
    if not s.at_shell_prompt and s.shell_in_front() and s.at_plain_shell_prompt():
        # A shell without the integration (su, sudo -i, ssh): a reply cannot
        # tell when a command ends -- a quiet `make check` came back "settled"
        # at once, and only `wait` with an until= found its end.
        place = await s.where()
        if place is None or place.via == "shell" or not integration_missing_since(s, place.label()):
            where = f" ({place.label()})" if place is not None else ""
            second_guess(
                s, "plain_shell", line,
                f"the shell at this prompt{where} has no shell integration, so a reply cannot tell when a command "
                "ends or how: no exit status, and a command quiet for a few seconds comes back \"settled\" as if "
                "it were done.",
                f"`shell_integrate` session={json.dumps(s.name)} adds it in one call (bash or zsh); then send this "
                "line.")
    if SET_ERREXIT.search(line):
        second_guess(
            s, "errexit", line,
            "`set -e` typed at this shell's prompt stays on in it: the first command that fails later -- a "
            "test run, a grep that finds nothing -- exits the shell, and the session with it (an eval build's "
            "`set -e; ...` left one so, and a later pytest's exit 4 closed it).",
            f"For this line only: {as_run(s, input='( ' + line.strip() + ' )')} -- the subshell takes the set -e "
            "with it.")
    if tool != "run":
        return
    if m := BACKGROUND_END.search(line):
        second_guess(
            s, "background", line,
            "a job started with `&` cannot be waited for -- the `wait` tool waits for the foreground, and its "
            "output and exit status come with no reply.",
            f"In the foreground, with the reply when it ends: {as_run(s, input=line[:m.start()].rstrip(), timeout_s=1500)} "
            "(a client that moves long calls to the background, as Claude Code does, tells you then); other work "
            "meanwhile goes in another session (session_create, or session_split beside this one).")
    if last_lines is None and (m := TAIL_END.match(line)):
        n = int(m["n"] or 10)
        args = {"input": m["cmd"], "last_lines": n}
        tee = m["tee"] and save_to is None and await s.where() is None
        if m["tee"] and not tee:
            args["input"] = line[:line.rindex("|")].rstrip()
        elif tee:
            args["save_to"] = m["tee"]
        second_guess(
            s, "pipe_tail", line,
            f"`| tail` lets only {'its last line' if n == 1 else f'its last {n} lines'} reach the session: the log, the `read` and "
            "`search` tools and a human watching never see the rest, and a reply already keeps the first and "
            "last lines of long output (a test run's FAIL lines and totals first).",
            f"The same {'line' if n == 1 else f'{n} lines'} in the reply, all of the output kept: {as_run(s, **args)}."
            + ("  (save_to writes the file, as the tee would.)" if tee else ""))
    if save_to is None and (m := REDIRECT_END.match(line)) and (target := m["a"] or m["b"]) != "/dev/null" \
            and await s.where() is None:
        second_guess(
            s, "redirect", line,
            "this output goes to the file only: not to this session's log, so the `read` and `search` tools, "
            "`show` and a human watching never see it.",
            f"Both, the file and the log: {as_run(s, input=m['cmd'], save_to=target.strip(chr(39) + chr(34)))}.")
    # The same, ending a command that is not the line's last: `make > log
    # 2>&1; echo EXIT=$?` (an eval's gdb, its prompt sent to the file, and a
    # run timed out typing into it).  A command's exit status comes with the
    # reply: the echo is not needed either.
    for a, b in line_segments(line)[:-1]:
        part = line[a:b]
        if m := MID_PIPE.search(part):
            rest = (line[:a] + part[:m.start()] + line[b:]).strip()
            second_guess(
                s, "pipe_tail", line,
                f"`| {m['what']}` here lets only part of this command's output reach the session: the log, the "
                "`read` and `search` tools and a human watching never see the rest, and a reply already keeps "
                "the first and last lines of long output.",
                f"All of it, the reply clipped as usual: {as_run(s, input=rest)} -- with show=REGEX for the lines "
                "you want.")
        if (m := REDIRECT_END.match(part)) and (target := m["a"] or m["b"]) != "/dev/null" \
                and await s.where() is None:
            rest = (line[:a] + m["cmd"] + line[b:]).strip()
            second_guess(
                s, "redirect", line,
                "this command's output goes to the file only: not to this session's log, so the `read` and "
                "`search` tools, `show` and a human watching never see it -- and a program that asks (gdb at "
                "its prompt) asks into the file, unseen.",
                f"Both, the file and the log: {as_run(s, input=rest, save_to=target.strip(chr(39) + chr(34)))}.")
    if (m := SLEEP_POLL.match(line)) and float(m["n"]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m["unit"]] >= 5 \
            and os.path.basename(m["rest"].split()[0]) in POLL_READERS:
        second_guess(
            s, "sleep_poll", line,
            "a fixed sleep before looking waits the whole time, however soon the thing is ready, and it may not "
            "be ready by then either.",
            "For a command running in a session: the `wait` tool on it, with until=REGEX (it returns the moment "
            "the output matches) or a timeout_s up to 1500.  For a file or a condition: "
            f"{json.dumps('until COND; do sleep 1; done; ' + m['rest'])} returns as soon as it holds.")


def typed_not_submitted(s: Session, text: str, keys: list[str], finished: list) -> str | None:
    """A word `send` typed with no Enter, left on a program's input line:
    say so.  An eval's gdb got send text="quit" -- "no new output", no word
    that it was never submitted; a ctrl+d then only deleted on that line,
    and the next `run` typed its line after the "quit".  (Tab completion, a
    one-key answer, an editor's keys: no note.)"""
    word = text.strip()
    if (len(word) < 2 or "\r" in text or "\t" in text or keys or finished or s.term.full_screen
            or s.at_shell_prompt or not s.term.cursor_text().rstrip().endswith(word)):
        return None
    return (f"[{word[-40:]!r} is on the input line, not submitted: `send` presses no Enter -- keys=[\"enter\"] "
            "submits it (or ctrl+u clears it); `run` input= types a line, presses Enter and waits for the answer]")


async def interrupt_notes(s: Session, regions: list[Region]) -> list[str]:
    """For a command of the caller's that ended by SIGINT (130) when an
    attached human's terminal had just sent input: say that the ctrl+c was
    probably theirs.  (Keys the server sends do not count as a client's
    input, so the caller's own ctrl+c never looks like this.)"""
    mine = [r for r in regions if r.kind == "command" and r.source == ACTOR.get() and r.ended is not None
            and 130 in (r.pipestatus or [r.exit_status])]
    if not mine:
        return []
    t = await s.tmux.last_client_activity(s.name)
    notes = []
    for r in mine:
        # tmux reports activity in whole seconds.
        if t is not None and r.started - 1 <= t and r.ended - 3 <= t <= r.ended + 1:
            at = time.strftime("%H:%M:%S", time.localtime(t))
            notes.append(f"[region {r.id} was interrupted (exit 130, ctrl+c), and an attached human's terminal "
                         f"sent input at {at}, just before -- the interrupt was probably theirs, not yours]")
    return notes


def saved_terminal(pid: int, tty: str) -> bool:
    """Has the process the terminal open past fds 0-2: a copy of its
    original output, kept while fd 1 goes elsewhere?"""
    try:
        return any(int(fd) > 2 and os.readlink(f"/proc/{pid}/fd/{fd}") in (tty, "/dev/tty")
                   for fd in os.listdir(f"/proc/{pid}/fd"))
    except (OSError, ValueError):
        return False


def output_nowhere(jobs: list[Process], tty: str) -> str | None:
    """Why nothing of a running job reaches the terminal, when nothing does:
    no foreground process writes to it, and the end of the pipeline writes
    to something other than a pipe or a regular file (/dev/null, a socket).
    (A regular file has its own note, with its growth.)"""
    outs = []
    for p in jobs:
        try:
            outs.append((p, os.readlink(f"/proc/{p.pid}/fd/1")))
        except OSError:
            continue
    if not outs or any(t == tty or t.startswith(("/dev/pts/", "/dev/tty")) for _, t in outs):
        return None
    if any(saved_terminal(p.pid, tty) for p, _ in outs):
        # pytest's capture: while a test runs, fd 1 is its temporary file,
        # and its dots go to the terminal through a copy of the original.
        return None
    for p, target in outs:
        if target.startswith("pipe:"):
            continue
        if target.endswith(" (deleted)"):
            return None  # a temporary file, unlinked: a file all the same
        try:
            if stat_module.S_ISREG(os.stat(target).st_mode):
                return None
        except OSError:
            pass
        return (f"nothing it prints reaches the terminal: {p.command}'s output goes to {target} -- `until`, "
                "`show` and this session's log see none of it")
    return None


def output_file(jobs: list[Process]) -> tuple[str, int, float] | None:
    """A foreground job writing to a regular file: (path, size, seconds since
    it was last written)."""
    for p in jobs:
        try:
            target = os.readlink(f"/proc/{p.pid}/fd/1")
            st = os.stat(target)
        except OSError:
            continue
        if stat_module.S_ISREG(st.st_mode):
            return target, st.st_size, max(time.time() - st.st_mtime, 0.0)
    return None


def human_size(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} bytes"


def human_age(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def output_elsewhere(jobs: list[Process], tty: str) -> str | None:
    """Where the foreground job's output goes, if not the terminal: a live
    gdb with its output in a file or pipe waits at a prompt no one sees."""
    for p in jobs:
        try:
            target = os.readlink(f"/proc/{p.pid}/fd/1")
        except OSError:
            continue
        if target != tty and not target.startswith("/dev/pts/"):
            return f"{p.command}'s output goes to {target}, not the terminal"
    return None


ADOPTED_PROMPT_WAIT_S = 3  # a pane just split: its bash's first prompt
ANSWER_QUIET_S = 0.3  # expect: a shell this quiet at its prompt ran no stray answer
TEXT_CAP = 2048  # Claude Code cuts server instructions and each tool description here
MAX_TIMEOUT_S = 1500  # clients drop a call idle this long (Claude Code: 1800 s without a reply)


def cap_timeout(timeout_s: float) -> float:
    """A timeout the client will wait out: longer ones are cut, and the
    reply says so -- a call the client aborts loses the reply and all the
    time waited."""
    if timeout_s <= MAX_TIMEOUT_S:
        return timeout_s
    if (notes := REPLY_NOTES.get()) is not None:
        notes.append(f"[timeout_s {timeout_s:g} cut to {MAX_TIMEOUT_S} s: clients abort a call with no reply "
                     "for 1800 s; the `wait` tool again keeps waiting]")
    return MAX_TIMEOUT_S


MATCH_WINDOW = 200  # characters shown either side of a match on a long line
CONTEXT_LINE_CHARS = 400  # of a context line around a search hit


def around_match(line: str, rx: re.Pattern) -> str:
    """A search hit's line, or on a long one (minified JSON, base64) the
    part around the match, with its column: the whole line can be megabytes."""
    if len(line) <= MAX_LINE_CHARS:
        return line
    m = rx.search(line)
    if m is None:
        return clip_line(line)
    lo, hi = max(0, m.start() - MATCH_WINDOW), min(len(line), m.end() + MATCH_WINDOW)
    return (f"[col {m.start() + 1} of {len(line)}] " + ("..." if lo else "") + line[lo:hi]
            + ("..." if hi < len(line) else ""))


SECRET_PROMPT = re.compile(r"(pass(word|phrase)?|passcode|\bPIN\b|secret|token|otp|verification code)[^:\n]*:\s*$", re.I)


async def secret_prompt_hint(s: Session) -> str | None:
    """A prompt for a password or passphrase: the terminal's echo is off
    with line input on (getpass, sudo, ssh's password: what a person types
    is not shown), or -- over ssh, where the local terminal is raw -- the
    prompt says so.  What `run` types becomes the region's label and is in
    the transcript; the reply says so before anyone types a secret there."""
    at = s.term.cursor_text().rstrip()
    if not at.strip():
        return None
    try:
        info = await s.tmux.pane_info(s.name)
        fd = os.open(info.tty, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            lflag = termios.tcgetattr(fd)[3]
        finally:
            os.close(fd)
        echo_off = not lflag & termios.ECHO and lflag & termios.ICANON
    except (OSError, termios.error, TmuxError):
        echo_off = False
    if not echo_off and not SECRET_PROMPT.search(at):
        return None
    why = "the terminal is not echoing input" if echo_off else "it asks for one"
    return (f"[{at.strip()[-50:]!r} is a password or passphrase prompt ({why}): ask the human to type it in the "
            f"session (they can attach), or answer with the `expect` tool and secret=true -- a line given to the "
            f"`run` tool becomes the region's label and is shown in replies]")


async def status_hint(s: Session, status: str, region: Region | None = None) -> str:
    """The hint for a call that ended without an answer, naming what is in
    the foreground when that is not the session's own program: the job the
    next keys would reach, which a pipeline's text does not tell.  For
    `region`, a command someone else started, no advice to interrupt it."""
    hint = f"[{STATUS_HINTS.get(status, status)}]"
    if status == "settled" and s.handed_off:
        hint = ("[nothing new printed, and what runs in the foreground is waiting for input: a program the "
                "last line resumed or started is probably at a prompt it printed earlier (gdb under gdb after "
                "`continue`) -- the `run` tool types its next command there]")
    if status == "waiting" and (secret := await secret_prompt_hint(s)):
        return secret
    if status == "waiting" and (text := s.unknown_prompt()):
        hint = (f"[output stopped at {text!r}, which looks like a shell prompt but not this shell's "
                f"({s.prompt_signature()!r}): the command may still be running and printed it (the `wait` "
                f"tool keeps waiting for the real prompt), or the prompt changed (su, a new PS1: the next "
                f"call learns it; shell_integrate makes it exact)]")
    owner = region.source if region is not None else ACTOR.get()
    if status == "timeout" and owner != ACTOR.get():
        whose = "a human's" if owner == "human" else f"{owner}'s"
        hint = (f"[still running -- {whose} command, not yours: the `wait` tool keeps waiting for it, the `regions` "
                f"tool lists the session without waiting; interrupting it would cut {whose} work short]")
    if status in ("timeout", "settled", "waiting"):
        jobs = [p for p in await foreground_jobs(s) if not is_carrier(p.pid, p.command)]  # ssh tells nothing
        if jobs:
            stuck = [p for p in jobs if p.pid in s.interrupt_survivors]
            if stuck and status == "timeout" and owner == ACTOR.get():
                if await foreground_busy(s):
                    hint = DEFERRED_INTERRUPT.format(who=named(stuck), start="still running; ctrl+c was delivered, but")
                else:
                    hint = (f"[still running; ctrl+c did not stop {named(stuck)} before -- ctrl+d or its quit "
                            f"command ends such a program, or `send` signal=\"TERM\"; the `wait` tool keeps waiting]")
            hint += f"\n[in the foreground: {named(jobs)}]"
            if status == "timeout" and (tree := process_tree_summary([p.pid for p in jobs])) is not None:
                # What is under it, and the newest process: a testsuite
                # moving on starts new ones, a hang shows an old one -- the
                # model judged progress from `ps -eo etime` in its own shell
                # tool.
                count, (pid, age, command) = tree
                if count > 1:
                    hint += (f"\n[under it: {count} processes; the newest started {human_age(age)} ago: "
                             f"{command[:100]} (pid {pid})]")
            hint += await redirect_note(s, jobs)
    if status == "timeout" and (tag := elapsed_tag(s, region)):
        hint = hint.replace("[still running", "[still running " + tag, 1)
    return hint


def elapsed_tag(s: Session, region: Region | None = None) -> str:
    """"(34 min so far; no output for 12 min)" for the command running: a
    wedged job looked like a slow one, and a `wait` reply without it left
    "still going" and "stuck" a guess."""
    running = region if region is not None else s.open_command
    # (An input region ends at its timeout, while what it started runs on.)
    if running is None or not running.started or (running.done and running.status != "timeout"):
        return ""
    quiet = time.monotonic() - s.last_output
    return (f"({human_age(time.time() - running.started)} so far"
            + (f"; no output for {human_age(quiet)}" if quiet >= 60 else "") + ")")


async def steadily_waiting(tty: str, looks: int = 4, apart_s: float = 0.25) -> bool:
    """Waiting for input on each of a few looks, not on one: a busy job has
    moments that look like it (a process exiting mid-check, a brief poll)."""
    for i in range(looks):
        if i:
            await asyncio.sleep(apart_s)
        if await foreground_state(tty, 0) not in ("read", "poll"):
            return False
    return True


async def redirect_note(s: Session, jobs: list[Process] | None = None) -> str:
    """The note for a foreground job whose output is not the terminal."""
    if jobs is None:
        jobs = [p for p in await foreground_jobs(s) if not is_carrier(p.pid, p.command)]
    info = await s.tmux.pane_info(s.name)
    if (where := output_elsewhere(jobs, info.tty)) and await steadily_waiting(info.tty):
        # Only a program waiting for input: a build or testsuite redirected
        # to a log is simply running (the note told a model "`wait` would
        # wait for nothing", and it went back to polling from Bash).
        # Say what to do instead, not only what is wrong: told only that,
        # a model concluded "redirect to files and use gdb -batch" and spent
        # six hours relaunching (in an evaluation).
        region = s.open_command
        live = live_gdb_line(region.command) if region is not None and region.command else None
        return (f"\n[{where}, and it is waiting for input: at a prompt (gdb, python) nothing shows it.  End it "
                "with `send` keys=['ctrl+d'] or its quit command (or signal=\"TERM\"), then start it again live, "
                "with no pipe or redirect" + (f" -- `{live}`" if live else "") + ": the region keeps all its output, "
                "`show=REGEX` picks out lines as they come, and the program stays up for the next question.  "
                "-batch or an output file would make every question a fresh start]")
    if (silent := output_nowhere(jobs, info.tty)) is not None:
        # The whole job's output ends somewhere else: `until`, `show` and the
        # log see none of it -- an `until` that timed out "for a reason the
        # tool couldn't explain" (`| tee F | tail -3000 > /dev/null` on a full
        # make check).
        return f"\n[{silent}]"
    if (written := output_file(jobs)) is not None:
        # A build or testsuite redirected to a file: each `wait` showed
        # nothing new, for 25 minutes at a time.
        path, size, age = written
        return (f"\n[its output goes to the file {path}, not the terminal: {human_size(size)} so far, "
                f"last written {human_age(age)} ago (a build may write there only as steps finish: the processes "
                f"under it say what runs now) -- nothing of it shows here; read that file to see how far it is]")
    return ""


CALL_LOG_MAX = 5 << 20


def call_logger(path: Path, identify, manager: "Manager | None" = None):
    """Wrap a tool so each call runs as its agent (`identify(ctx)`, see
    ACTOR) and is logged: when, which agent, tool and session, how long,
    and whether it failed -- never its input or output.  Enough to tell
    whether a slow or missing reply is the server's doing."""

    def wrap(fn):
        @functools.wraps(fn)
        async def logged(*args, ctx: Context, **kwargs):
            start = time.time()
            outcome = "ok"
            actor = identify(ctx)
            token = ACTOR.set(actor)
            if manager is not None:
                manager.start_housekeeping()
            notes: list[str] = []
            notes_token = REPLY_NOTES.set(notes)
            regions_token = REPLY_REGIONS.set([])
            after: list = []
            after_token = REPLY_AFTER.set(after)
            try:
                try:
                    result = await fn(*args, **kwargs)
                    if isinstance(result, str):
                        extra = []
                        for f in after:
                            n = f()
                            if inspect.isawaitable(n):
                                n = await n
                            if n:
                                extra.append(n)
                        result = "\n".join([result] + extra)
                    return "\n".join(notes + [result]) if notes and isinstance(result, str) else result
                except TmuxError as e:
                    raise ToolError(str(e)) from e
            except BaseException as e:
                outcome = "error: " + (str(e).splitlines() or [type(e).__name__])[0][:200]
                raise
            finally:
                ACTOR.reset(token)
                REPLY_NOTES.reset(notes_token)
                REPLY_REGIONS.reset(regions_token)
                REPLY_AFTER.reset(after_token)
                entry = {
                    "time": round(start, 3), "actor": actor, "tool": fn.__name__,
                    "session": kwargs.get("session"),
                    "ms": round((time.time() - start) * 1000), "outcome": outcome,
                }
                try:
                    if path.exists() and path.stat().st_size > CALL_LOG_MAX:
                        path.replace(path.with_suffix(".log.1"))
                    with open(path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except OSError:
                    pass

        # Declare the Context parameter so the SDK passes it in; the tool
        # itself does not see it.
        # (Resolved now: this module's annotations are strings.)
        sig = inspect.signature(fn, eval_str=True)
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context)
        logged.__signature__ = sig.replace(parameters=[*sig.parameters.values(), ctx_param])
        logged.__annotations__ = {**typing.get_type_hints(fn, include_extras=True), "ctx": Context}
        return logged

    return wrap


class Actors:
    """Names the agents of a shared server.  The X-Terminal-Actor header a
    client is configured to send is used as it is: a stable name across
    reconnects.  Without it, the name the client gives itself, made unique
    per connection ("claude-code", "claude-code#2").  A single-client
    server has one agent, "model"."""

    def __init__(self, shared: bool):
        self.shared = shared
        self.by_connection: dict[str, str] = {}

    def __call__(self, ctx: Context) -> str:
        if not self.shared:
            return "model"
        request = getattr(ctx.request_context, "request", None)
        headers = getattr(request, "headers", None) or {}
        named = headers.get("x-terminal-actor")
        if named:
            # Set in the client's configuration: a stable identity, kept
            # across reconnects (and shared by clients configured alike).
            return re.sub(r"[^A-Za-z0-9_.@-]", "-", named)[:40] or "agent"
        params = ctx.session.client_params
        info = getattr(params, "client_info", None) if params else None
        name = getattr(info, "name", None) or "agent"
        name = re.sub(r"[^A-Za-z0-9_.@-]", "-", name)[:40] or "agent"
        # An HTTP connection is known by its session id, so two clients with
        # one name are told apart; without one (in-process), the name is it.
        conn = headers.get("mcp-session-id") or f"name:{name}"
        if conn in self.by_connection:
            return self.by_connection[conn]
        taken = set(self.by_connection.values())
        unique, n = name, 1
        while unique in taken or unique == "human":
            n += 1
            unique = f"{name}#{n}"
        self.by_connection[conn] = unique
        return unique


# Arguments of the shell tools models know best (codex's exec_command and
# write_stdin), carried over to ours: what they are here.
OTHER_TOOLS_ARGUMENTS = {
    "yield_time_ms": [("timeout_s", ", in seconds")],
    "max_output_tokens": [("max_lines", "")],
    "workdir": [("cwd", "")],
    "session_id": [("session", ", a name")],
    "chars": [("text", ""), ("input", "")],
}


# Our own tools' arguments mixed up between them: what each is in this one
# (an eval's Claude Code x Qwen: send session="debug" input="q").
OWN_TOOLS_ARGUMENTS = {
    ("send", "input"): "'input' (that is `run`'s: here text=, typed with no Enter -- add keys=[\"enter\"], "
                       "or `run` input= types the line and waits for its result)",
    ("send", "timeout_s"): "'timeout_s' (that is `run`'s: here wait_ms=, in milliseconds -- `send` waits only "
                           "for output to settle; `wait` waits for a command)",
    ("run", "text"): "'text' (that is `send`'s: here input=, the line, Enter added)",
    ("run", "keys"): "'keys' (that is `send`'s: keystrokes such as ctrl+c go through `send`; "
                     "interrupt=true here stops a running command before the line)",
}


def unknown_arguments(tool: str, unknown: list[str], known: list[str]) -> str:
    """The refusal of a call with arguments the tool does not have."""
    def abbreviates(k: str, n: str) -> bool:  # "cmd" for "command", not "cwd"
        letters = iter(n)
        return k[:1] == n[:1] and all(c in letters for c in k)

    def guess(k: str) -> str:
        if own := OWN_TOOLS_ARGUMENTS.get((tool, k)):
            return own
        if alias := next(((n, unit) for n, unit in OTHER_TOOLS_ARGUMENTS.get(k, ()) if n in known), None):
            return f"{k!r} (another shell tool's argument: here {alias[0]}{alias[1]})"
        if k == "workdir" and "cwd" not in known:
            return f"{k!r} (a session keeps its directory: `cd` there, or session_create cwd=)"
        near = ([n for n in known if abbreviates(k, n) or k.startswith(n)]
                or difflib.get_close_matches(k, known, 1))
        return f"{k!r} (did you mean {near[0]!r}?)" if near else repr(k)
    names = ", ".join(guess(k) for k in unknown)
    return (f"{tool} has no argument{'s' if len(unknown) > 1 else ''} {names}; nothing was done.  "
            f"Its arguments: {', '.join(known)}")


class Server(MCPServer):
    """An argument a tool does not have is refused, not dropped: codex's
    models carried their own shell tool's names over -- session_create
    cmd="gdb -q ./avg" started bash, silently, and `run` got
    yield_time_ms and max_output_tokens."""

    async def call_tool(self, name, arguments, context=None):
        tool = self._tool_manager.get_tool(name)
        if tool is not None:
            known = list(tool.parameters.get("properties", {}))
            if unknown := [k for k in arguments if k not in known]:
                raise ToolError(unknown_arguments(name, unknown, known))
        # A session's name with blanks around it is that name: qwen sent session="\nvim-ini".
        arguments = {k: v.strip() if k in ("session", "name") and isinstance(v, str) else v
                     for k, v in arguments.items()}
        return await super().call_tool(name, arguments, context)


def build_server(home: Path | None = None, socket: str | None = None, shared: bool = False) -> MCPServer:
    home = home or Path(os.environ.get("DOCENT_TERMINAL_MCP_HOME", Path.home() / ".local/state/docent-terminal-mcp"))
    socket = socket or os.environ.get("DOCENT_TERMINAL_MCP_SOCKET", "docent-terminal-mcp")
    mgr = Manager(home, socket)
    mcp = Server("docent-terminal-mcp", instructions=INSTRUCTIONS)
    mcp.terminal_manager = mgr  # for the HTTP endpoints of the shared server
    logged = call_logger(home / "calls.log", Actors(shared), mgr)

    def tool(**kw):
        """Register a tool, with its calls logged.  `meta` carries Claude
        Code's hints: alwaysLoad (never deferred behind its tool search)
        and searchHint; `annotations` marks the read-only tools."""
        return lambda fn: mcp.tool(**kw)(logged(fn))

    @tool(annotations=ToolAnnotations(read_only_hint=True), meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'terminal usage guidance, patterns, gotchas'})
    async def guide(
        topic: Annotated[str | None, Field(description="one of patterns, programs, output, fullscreen, sharing; none lists them")] = None,
    ) -> str:
        """Usage guidance, worth one call before the first real use.  Topics:
        patterns (one-shot shell habits -- pipes, redirects, -batch, sleep-and-
        poll, nohup -- and their interactive form, plus the gotchas), programs
        (gdb, python, REPLs, prompts, questions), fullscreen (vim, less, send
        order, screen view), output (timeouts, large output, views, the log
        files), sharing (working alongside a human or another agent)."""
        if topic is None:
            return INDEX
        if topic not in TOPICS:
            raise ToolError(f"unknown topic {topic!r}. {INDEX}")
        return TOPICS[topic]

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'start a terminal session, tmux, shell, gdb, ssh, REPL'})
    async def session_create(
        name: Annotated[str | None, Field(description="a name for the session (default: t1, t2, ...); the name of a closed session archives its old logs")] = None,
        command: Annotated[str | None, Field(description="the program to run instead of bash with shell integration, e.g. \"gdb -q ./prog\" or \"ssh host\"; \"bash\" or \"zsh\" (with -i/-l) is that shell with the integration, \"bash --norc\" or \"zsh -f\" a plain one")] = None,
        cwd: Annotated[str | None, Field(description="working directory (default: the server process's own)")] = None,
        cols: Annotated[int, Field(description="terminal width in columns")] = 160,
        rows: Annotated[int, Field(description="terminal height in rows")] = 48,
        env: Annotated[dict[str, str] | None, Field(description="environment variables set in the session, on top of the server's environment")] = None,
        prompt_pattern: Annotated[str | None, Field(description="the session's default prompt regex for `run`/`expect` into a program; not for bash, and rarely needed (see `run`)")] = None,
        fixed_size: Annotated[bool, Field(description="keep cols x rows even when a human attaches (the default lets the pane follow their window)")] = False,
        delete_logs_on_close: Annotated[bool, Field(description="delete the session's logs when it closes, for one that may show something sensitive (see session_logs)")] = False,
    ) -> str:
        """Start a terminal session: a tmux pane a human can watch or type into,
        whose output is logged in full and kept as numbered regions.  Use one
        instead of a one-shot shell call for anything that runs a program with
        a prompt (gdb, python, ssh, any REPL), takes more than a few seconds (a
        build, a testsuite), prints more than a reply holds, may ask a question
        (y/n), or that the user may want to watch or take over; a one-shot call
        stays right for reading and editing files and for quick commands whose
        whole output you want.  Open one at the start of a task and do the
        task's shell work in it with the `run` tool:
        sessions are cheap, keep running between calls and across your own
        restarts (`session_list` finds them), and keep their cwd, environment
        and running programs -- cd once, start gdb once.

        With no `command` (or "bash"), the session is bash with shell
        integration: exact command boundaries and exit statuses, and programs
        are started from it with the `run` tool (make, gdb, python, ssh, ...).  `command`
        runs another program as the session itself instead, e.g. "gdb -q -ex
        'break main' -ex run ./prog" or "ssh host"; when it exits the session is
        over.  One session per job is the usual shape: a shell for the build
        and the tests, a second one for gdb while the first is busy.

        Tell the user the session's name and attach command (in the reply) on
        a line of its own, so they can watch or step in.  First time here:
        load the `terminal-sessions` skill if one is listed, else `guide`.

        By default the pane follows the window of a human who attaches, so
        `cols`/`rows` apply only until then; `fixed_size` keeps them (the human
        then sees the pane clipped or padded).  `session_resize` changes either
        later."""
        session = await mgr.create(name, command, cwd, cols, rows, env, prompt_pattern, delete_logs_on_close)
        if fixed_size:
            await mgr.tmux.set_size(session.name, cols, rows, fixed=True)
        async with session.lock:
            if session.integrated:
                await session.wait(lambda: session.shell_state == "input", 10)
            else:
                # First output (an ssh login takes longer than any settle
                # time), then quiet -- or a shell it brought along at its prompt.
                start = session.offset
                await session.wait(lambda: session.offset > start, 10)
                await session.wait(lambda: session.shell_state == "input", 10, settle_s=0.5)
                session.learn_prompt(session.prompt_pattern)
        screen = session.term.text_between((max(0, session.term.row - 20), 0), session.content_end())
        socket_path = await mgr.tmux.socket_path()
        return (
            f"session {session.name!r} started ({session.meta['command'] + ' with shell integration' if session.integrated else session.meta['command']})\n"
            f"human can watch/join with: {mgr.tmux.attach_command(session.name)}"
            + (f"  (tmux socket {socket_path})" if socket_path else "") + "\n"
            f"log directory: {session.dir}"
            + (" (deleted when the session closes)" if delete_logs_on_close else "") + "\n"
            + (f"logs of the earlier session {session.name!r} moved to {session.archived}\n" if session.archived else "")
            + (f"---\n{screen.strip()}" if screen.strip() else "")
        )

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'split a terminal session window into panes, side by side'})
    async def session_split(
        session: Annotated[str, Field(description="the session whose window to split (its pane is the one split)")],
        below: Annotated[bool, Field(description="true: the new pane goes below; false: to the right")] = True,
        size: Annotated[str | None, Field(description="the new pane's size: rows (below) or columns (right), or a "
                                                      "percentage such as \"50%\"; default half")] = None,
        name: Annotated[str | None, Field(description="a name for the new session (default: <window>+1, +2, ...)")] = None,
        command: Annotated[str | None, Field(description="the program to run instead of bash with shell integration, "
                                                         "as for session_create")] = None,
        env: Annotated[dict[str, str] | None, Field(description="environment variables set in the new pane")] = None,
        prompt_pattern: Annotated[str | None, Field(description="the new session's default prompt regex, as for "
                                                                "session_create; rarely needed")] = None,
    ) -> str:
        """Split a session's window: a new pane beside it, shown together to a
        human attached to that window, and a session of its own -- logged
        from its first byte, with shell integration (exact exit statuses) by
        default.  Use it instead of typing `tmux split-window`, whose pane has
        neither until the server adopts it.  For two things to watch at
        once: a testsuite running in one pane, gdb in the other.  Both panes
        are addressed by session name; typing never follows the active
        pane."""
        s = await mgr.split(session, name, command, below, size, env, prompt_pattern)
        async with s.lock:
            if s.integrated:
                await s.wait(lambda: s.shell_state == "input", 10)
            else:
                await s.wait(lambda: False, 10, settle_s=0.5)
                s.learn_prompt(s.prompt_pattern)
        screen = s.term.text_between((max(0, s.term.row - 20), 0), s.content_end())
        return (
            f"session {s.name!r} started in a new pane of {mgr.tmux.window(s.name)!r}'s window "
            f"({s.meta['command'] + ' with shell integration' if s.integrated else s.meta['command']})\n"
            f"human can watch/join (both panes) with: {mgr.tmux.attach_command(s.name)}\n"
            f"log directory: {s.dir}\n" + (f"---\n{screen.strip()}" if screen.strip() else "")
        )

    @tool(annotations=ToolAnnotations(read_only_hint=True), meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'list terminal sessions'})
    async def session_list(
        name: Annotated[str | None, Field(description="only sessions whose name this regex matches (searched, as `grep` does)")] = None,
        attached: Annotated[bool, Field(description="only sessions a human is attached to")] = False,
    ) -> str:
        """List terminal sessions: state, what is running, size, who started each
        (you, a human, another agent), humans attached, the attach command, and
        presence: how old each session is, its last output, which agents typed
        into it and when, a human terminal's last input, a typing call in
        progress -- to judge whether a session you find is in use.
        Sessions started by earlier servers or by others are used by name like
        your own; a closed session's logs stay on disk for a few days (the
        listing says until when; session_logs keeps them).  Worth a call at the
        start of a conversation: a session from an earlier one may still be
        running, its log intact."""
        out = []
        described = await mgr.describe_sessions()
        # (65 rows on a host an editor shares, for the one pane wanted: an
        # adjutant session's report.)
        if name:
            name_rx = compile_pattern(name)
            described = [d for d in described if name_rx.search(d["name"])]
        if attached:
            described = [d for d in described if d["state"] == "live" and d.get("humans")]
        closed = sorted((d for d in described if d["state"] == "closed"), key=lambda d: d.get("created") or 0)
        older = closed[:-CLOSED_LISTED] if len(closed) > CLOSED_LISTED else []
        described.sort(key=lambda d: d["state"] == "closed")  # live (and exited) first; stable within each
        for d in described:
            if d in older:
                continue
            state = d["state"]
            if state == "closed" and d.get("exit_status") is not None:
                state += f", exited{exit_said(d['exit_status'], ' ({})')}"
            elif state == "exited":
                state = f"exited{exit_said(d['exit_status'], ' ({})')}; session_close ends the tmux session, logs are kept"
            elif state == "error":
                state = f"error: {d['error']}"
            elif state == "live":
                n = d["humans"]
                via = d["place"]["via"] if d.get("place") else ""
                on = f", on {d['place']['label']} ({'its shell says so' if via == 'shell' else via})" if via else ""
                on += {"integrated": ", shell integrated (exact exit statuses)",
                       "plain": ", plain shell (no exit statuses: shell_integrate, or `docent-terminal-mcp ssh`)"}.get(d.get("shell"), "")
                fixed = ""
                if d["fixed"]:
                    fixed = " fixed" if len(mgr.window_sessions(d["name"])) <= 1 else " (its window's size is fixed; this pane is one of its panes)"
                state += (
                    f", running {d['running']}{on}, {d['width']}x{d['height']}{fixed}, "
                    f"{n or 'no'} human{'s' if n != 1 else ''} attached\t{d['attach']['command']}"
                )
            if d.get("keep"):
                state += ", logs kept"
            elif d.get("delete_on_close"):
                state += ", logs deleted when it closes"
            elif d.get("expires"):
                state += f", logs deleted after {time.strftime('%Y-%m-%d %H:%M', time.localtime(d['expires']))}"
            by = d.get("started_by")
            who = f" (started by {by})" if by and by != ACTOR.get() else ""
            if state.startswith("live") and (d.get("parent") or len(mgr.window_sessions(d["name"])) > 1):
                state += f"\tpane {d.get('pane_id')} of window {d.get('window')!r}{', active' if d.get('active') else ''}"
            if d.get("adopted") and state.startswith("live"):
                state += (" (adopted: a plain shell until the first `run` there adds exit statuses)"
                          if not d.get("integrated") else " (adopted)")
            out.append(f"{d['name']}\t{d['command']}{who}\t{state}\t{presence(d)}")
        if older:
            out.append(f"[{len(older)} older closed session{'s' if len(older) > 1 else ''} not listed; "
                       f"their logs remain under {mgr.sessions_dir}]")
        return "\n".join(out) or ("no sessions match" if name or attached else "no sessions")

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'resize a terminal session'})
    async def session_resize(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        cols: Annotated[int | None, Field(description="new width in columns (default: unchanged)")] = None,
        rows: Annotated[int | None, Field(description="new height in rows (default: unchanged)")] = None,
        fixed: Annotated[bool, Field(description="true pins the size against attaching humans; false lets it follow their window again")] = True,
    ) -> str:
        """Set the pane size and pin it (`fixed`, the default: attaching
        humans no longer resize it), or with fixed=false let it follow the
        attached human's window again."""
        s = await mgr.get(session)
        await mgr.tmux.set_size(session, cols, rows, fixed)
        info = await mgr.tmux.pane_info(session)
        how = "fixed" if fixed else "following the attached window"
        out = f"[session {s.name!r} is {info.width}x{info.height}, {how}]"
        others = [n for n in mgr.window_sessions(session) if n != session]
        if (cols and info.width != cols) or (rows and info.height != rows):
            if others:
                size = (await mgr.tmux.run("display-message", "-p", "-t", mgr.tmux.wtarget(session),
                                           "#{window_width}x#{window_height}", check=False)).strip()
                out += (f"\n[asked {cols or info.width}x{rows or info.height}: the window is {size}, shared with "
                        f"{', '.join(others)}, so this pane is smaller -- session_close the others, or "
                        f"session_resize the window's other panes]")
            else:
                out += f"\n[asked {cols or info.width}x{rows or info.height}; tmux gave {info.width}x{info.height}]"
        return out

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'close kill a terminal session'})
    async def session_close(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        window: Annotated[bool, Field(description="close every session sharing this session's window (its split "
                                                  "panes) too, not only this one")] = False,
    ) -> str:
        """Kill a session and whatever runs in it.  Its logs and region files are
        kept on disk for a few days (session_logs keeps them for good, or
        deletes them).  Refused for a session a person is attached to, and
        for another's while a command runs there: leave those to their
        owners, or ask the user to close it.  Close what you are done with; a session another agent or
        a human started is theirs to close.  A session that shares a window
        with others (a split) closes its own pane only; `window` closes them
        all."""
        return await mgr.close_sessions(session, window, force=False)

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'keep or delete a terminal session\'s logs'})
    async def session_logs(
        session: Annotated[str, Field(description="the session's name, live or closed (an earlier session of a reused name is NAME.N, as session_list shows)")],
        action: Annotated[typing.Literal["keep", "default", "delete_on_close", "delete"], Field(description=(
            "\"keep\": never delete them (when the user asks for that); \"default\": back to the usual rule; \"delete_on_close\": delete them "
            "when the session closes (at once if it is closed); \"delete\": delete them now (closed sessions only)"))],
    ) -> str:
        """Keep or delete a session's logs (its log directory: the full output,
        regions, raw bytes).  The usual rule: a closed session's logs are
        deleted once they have been neither written nor read for a few days
        (DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS, default 3).  Reading a closed
        session's logs needs none of this -- and counts as use: the `read`,
        `search` and `regions` tools work on them as they are.  Change how
        a session's logs are kept only when the user asks: "keep" exempts a
        session they want to refer back to; "delete_on_close" and "delete" are for logs that showed
        something sensitive.  Deleting cannot be undone: ask the user first,
        and never delete another agent's or a human's logs unasked."""
        return await mgr.logs_action(session, action)

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'run a shell command or program input in a terminal session, build, test, gdb'})
    async def run(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        input: Annotated[str, Field(description="the line to type (Enter is added); several lines are sent one at a time, each after the previous is back at a prompt")],
        timeout_s: Annotated[float, Field(description="seconds to wait before giving up (the command keeps running; `wait` resumes); at most 1500")] = 60,
        prompt_pattern: Annotated[str | None, Field(description="regex that marks the program's prompt at the end of the cursor line; rarely needed, the default knows shell, gdb, pdb, lldb and python prompts and any ending in $ # % >; pass one only after a reply came back \"waiting\" at a prompt it did not know (it offers the pattern); make it cover a part that changes")] = None,
        settle_ms: Annotated[int | None, Field(description="into a program with no recognised prompt: return once output has been quiet this long and nothing is visibly working (default 3000)")] = None,
        max_lines: Annotated[int, Field(description="lines of output shown in the reply; longer output is clipped in the middle, led by its summary lines (test results, compiler errors, make's failure) when the middle holds them; all of it stays readable with `read`/`search`")] = 80,
        force: Annotated[bool, Field(description="type even over someone else's unsubmitted line at the prompt, discarding it first; only for a line you know is stale")] = False,
        show: Annotated[str | None, Field(description="regex: reply with only the output lines matching it, in order, plus the last few (like `| grep`, but the command runs unfiltered and the log keeps everything), e.g. \"error:|warning:|^FAIL\" for a build -- or a program's answer: gdb's `step` under `set debug infrun 1`")] = None,
        until: Annotated[str | None, Field(description="regex: return as soon as the output matches it, the command still running (status \"until\"; `wait` resumes); not a prompt pattern")] = None,
        typing: Annotated[Typing | None, Field(description="type the input as a person would, for a human watching: {wps, mistakes, seed}")] = None,
        save_to: Annotated[str | None, Field(description="a file to copy the command's output into, as `> FILE 2>&1` would, while the command itself runs unredirected (the reply and the log are as usual): written at once, kept up to date while it runs, written in full when it ends.  A relative path is in the session's current directory; the file is on the machine this server runs on, even for a session over ssh.  For a later diff or grep by other tools; `read` and `search` need no file")] = None,
        last_lines: Annotated[int | None, Field(description="reply with only the last N lines of the output (like `| tail -N`, but the command runs unfiltered and the log keeps everything; the reply says where the rest is); with `show`, how many last lines follow the matches (default 3)")] = None,
        show_after: Annotated[int, Field(description="with `show`: also this many lines after each match (like `grep -A N`); a match is numbered \"N:\", a line after one \"N-\"")] = 0,
        interrupt: Annotated[bool, Field(description="if a command is running and holds the session (a `tail -f`, a server), interrupt it with ctrl+c first, then type the line once the prompt is back; the reply names what was interrupted.  Only for a command you mean to stop")] = False,
    ) -> str:
        """Type a line into the session, press Enter, and wait for the result:
        the session's form of a shell tool call, and of a line typed at gdb's
        or python's prompt.

        At a bash prompt (shell integration) it waits for the command to exit
        and returns its output with the exit status (pipestatus for a pipeline;
        no `; echo $?` needed); for a build or a testsuite give `timeout_s` the
        whole run (up to 1500 s) and get the exit status in one reply.  Inside
        a program (gdb, python, psql, ...) it waits for the program's next
        prompt; starting one from bash returns at its first prompt, or its
        first screen (vim, less; continue with `send`).  The session persists:
        cwd, variables and a running program stay between calls.

        Output longer than `max_lines` keeps its first and last lines with a
        marker naming what was left out; nothing is lost: `read` pages it,
        `search` finds lines in it.  So no `| tail`, `| head`, `2>&1 > log`
        -- and never a pipe or redirect on an interactive program (gdb
        without -batch, python): it then waits at a prompt you cannot see.
        To watch for particular lines as they come, `show`=REGEX replies with
        the matching lines and the last few while the command runs
        unfiltered, exit status its own.

        At `timeout_s` -- a short one gives your turn back soon -- the
        command keeps running and its region stays open: `wait` keeps
        waiting, `send keys=["ctrl+c"]` interrupts, `send signal="TERM"` if
        it survives.  `until`=REGEX returns as soon as the output matches.

        Several lines go one at a time, each once the previous is back at the
        prompt; a line that asks a question or times out stops the rest.  Into Python or IPython they go
        as one paste (an exception drops the lines after it).  Predictable
        questions: `expect`.  Keystrokes (ctrl+c, tab, arrows) are `send`'s
        `keys`, not an argument here.

        `prompt_pattern` is rarely needed: never pass one to start a program
        (shell, gdb, pdb, lldb, python and any prompt ending in $ # % > are
        known); only after a reply came back "waiting" at a prompt it did not
        know, which then offers the pattern."""
        s = await mgr.get(session)
        timeout_s = cap_timeout(timeout_s)
        show_rx = compile_pattern(show) if show else None
        until_rx = compile_pattern(f"(?m:{until})") if until else None
        check_output_options(last_lines, show_after, show)
        if save_to:
            target = await save_target(s, save_to)
            REPLY_AFTER.get().append(lambda: save_reply(s, target))
        async with replying(s, "run"):
            await refuse_if_exited(s)
            s.typing = typing.model_dump() if typing else None
            s.sync()
            if s.term.full_screen:
                raise ToolError(
                    "a full-screen program is running; use the `send` tool for keys and `read` view='screen'"
                )
            await check_human(s, force)
            forced = ""
            if discard_pending(s, force):
                forced = await clear_prompt(s) + "\n"
            if integrated_note := await integrate_adopted(s):
                forced += integrated_note + "\n"
            pattern = call_prompt(s, prompt_pattern)
            if interrupt:
                forced += await interrupt_first(s, pattern)
            if s.at_shell_prompt or (s.shell_in_front() and s.at_plain_shell_prompt()):
                await second_guesses(s, input, last_lines, save_to)
            were_open = open_command_ids(s) | s.open_at_reply
            # A ctrl+c seen not to stop the last command says nothing about
            # this one (it may well stop it).
            s.note_interrupt_survivors(set())
            if s.at_shell_prompt:
                # A shell command even with a prompt_pattern (the prompt of
                # the program it starts): one region, exact, and the caller's.
                regions, why, unsent = await s.run_shell_command(input, pattern, timeout_s, until_rx)
                stopped = [f"[stopped: {unsent} more line{'s' if unsent > 1 else ''} not sent, since the one "
                           f"before did not come back to the prompt]"] if unsent else []
                if not regions:
                    new = s.clip_span(s.input_start or (s.term.row, 0), s.content_end(), max_lines)
                    if why == "continuation":
                        return f"{forced}[{STATUS_HINTS['continuation']}]\n{new}"
                    if why == "done":
                        return f"{forced}[no command ran (a comment or an empty line); the shell is back at its prompt]"
                    return f"{forced}[no command started: {why}; the shell may want more input]\n{new}"
                parts = ([forced.rstrip()] if forced else []) + render_regions(s, regions, max_lines, show_rx, show_after, last_lines)
                parts += stopped
                parts += await interrupt_notes(s, regions)
                parts += habit_notes(s, input, regions)
                parts += pipe_notes(s, regions, show_rx, last_lines)
                if prompt_pattern is not None and any(r.status in ("running", "ready") for r in regions[-1:]) \
                        and not regions[-1].done and (DEFAULT_PROMPT.search(s.term.cursor_text())
                                                      or pattern_known(prompt_pattern)):
                    parts += note_once(s, "pattern_at_launch", PATTERN_AT_LAUNCH_NOTE)
                for r in regions:
                    note_shown(s, r)
                if why == "done" and not regions[-1].done and s.term.full_screen:
                    parts.append("[the command is running as a full-screen program; drive it with the `send` tool]")
                    parts.append(await render_screen(mgr.tmux, s, max_rows=max_lines))
                elif why == "done" and not regions[-1].done:
                    at = s.term.cursor_text().strip()
                    where = (f"at {at[-60:]!r}" if at
                             else "(it is reading the terminal, with no prompt printed)" if s.foreground == "read"
                             else "(nothing printed for a while: or it is idle -- a server, say)")
                    tag = elapsed_tag(s, regions[-1])
                    parts.append(
                        f"[the command is still running{' ' + tag if tag else ''} and appears to be waiting for "
                        f"input {where}; further calls of the `run` tool are sent to it]" + await redirect_note(s)
                    )
                    if nested := await nesting_note(s):
                        parts.append(nested)
                    if secret := await secret_prompt_hint(s):
                        parts.append(secret)
                    elif yn := question_hint(at):
                        parts.append(yn)
                elif why != "done":
                    parts.append(await status_hint(s, why))
                    if until and why == "timeout":
                        parts.append(f"[`until` {until!r} did not match within {timeout_s:g} s]")
                parts += outer_finished(s, were_open, regions)
                return "\n".join(parts)
            if settle_ms is None and prompt_pattern is None:
                settle_ms = DEFAULT_SETTLE_MS
            settle = settle_ms / 1000 if settle_ms else None
            # A program that takes typed lines answers each before reading
            # the next (gdb), so several lines are sent one at a time, each
            # with its own region: the reply reads like the transcript.
            lines = [input] if s.will_paste(input) else input.removesuffix("\n").split("\n")
            parts = [forced.rstrip()] if forced else []
            report_foreign_input(s)
            if not force:
                await refuse_type_ahead(s, pattern, input)
            # Typed at a question rather than a prompt -- `read -p`, gdb's
            # "(y or n)" -- the line is its answer, whatever it was meant as.
            last = s.regions[-1] if s.regions else None
            # (Shaped like one: "Continue?", "Password:", "(y or n)" -- not a
            # program's prompt the call's pattern does not know, "stock> ".)
            question = s.term.col > 0 and not s.prompt_here(pattern) and (
                s.open_command is not None or (last is not None and last.status == "waiting")
            ) and s.term.cursor_text().rstrip().endswith(QUESTION_ENDS) and not DEFAULT_PROMPT.search(
                s.term.cursor_text())  # "(gdb) " ends in ")" but is gdb's prompt
            if question and s.shell_in_front():
                # A shell at its own prompt waits in readline; one asking
                # (`read -p`) is blocked reading the terminal.  "[12:00 #3]
                # ready: " is a prompt, whatever it ends with.
                info = await s.tmux.pane_info(s.name)
                question = await foreground_state(info.tty, 0) == "read"
            shell_after = None  # the lines after a program that ended, run by its shell
            for i, line in enumerate(lines):
                region, enclosing = await s.run_program_input(line, pattern, timeout_s, settle, until_rx)
                parts.append(render_region(s, region, max_lines, show_rx, show_after, last_lines))
                if i == 0 and question:  # the question itself is in the region's label
                    parts.append(f"[{line!r} was sent as the answer to that question, not as a command]")
                if region.status in ("timeout", "settled", "until"):  # `wait` may resume it
                    s.shown_upto[region.id] = s.region_end(region)[0]
                if until and region.status == "timeout":
                    parts.append(f"[`until` {until!r} did not match within {timeout_s:g} s]")
                if region.status == "finished" and i + 1 < len(lines) and s.at_shell_prompt:
                    # The program ended -- `quit`, or its "Quit anyway?" answered
                    # -- and its shell is back at its prompt: the lines after it
                    # are that shell's (quit, y, and a new gdb line took three
                    # calls).
                    shell_after = await s.run_shell_command("\n".join(lines[i + 1:]), s.prompt_pattern,
                                                            timeout_s, until_rx)
                    break
                if region.status != "ready" and i + 1 < len(lines):
                    left = len(lines) - i - 1
                    rest = lines[i + 1:]
                    asked = s.term.cursor_text().strip() if region.status == "waiting" else ""
                    if asked and YES_NO.search(asked + " ") and YES_NO_ANSWER.fullmatch(rest[0].strip()):
                        # The next line is that question's answer, written
                        # ahead ("quit", "y"): send it.  Held back, the reply
                        # said to send "y\ny" -- the second y into the shell.
                        continue
                    if asked and YES_NO.search(asked + " "):
                        # Copy-ready: the answer first, then the rest, in one call.
                        again = "\\n".join(["y"] + rest).replace('"', '\\"')
                        parts.append(f"[stopped: {left} more line{'s' if left > 1 else ''} not sent -- it asks "
                                     f"{asked[-60:]!r}.  To answer and go on in one call: the `run` tool with "
                                     f"input=\"{again}\" (or \"n\" in place of \"y\")]")
                    else:
                        parts.append(f"[stopped: {left} more line{'s' if left > 1 else ''} not sent, "
                                     f"since this one did not come back to the prompt]")
                    break
                if i + 1 < len(lines) and (event := STOP_EVENT.search(s.region_text(region))):
                    # Back at the prompt, but somewhere else: a signal (an
                    # expression's call aborted), the program stopped or
                    # exited -- the lines after it were meant for where it
                    # was (five `p tp->...` in a frame without tp).
                    left = len(lines) - i - 1
                    parts.append(f"[stopped: {left} more line{'s' if left > 1 else ''} not sent -- this one "
                                 f"ended with {event.group(0).strip()!r}: the program may be somewhere else now; "
                                 "look before going on]")
                    break
            out = "\n".join(parts)
            if prompt_pattern is not None and (region.status == "ready" and DEFAULT_PROMPT.search(s.term.cursor_text())
                                               or pattern_known(prompt_pattern)):
                # Into a program already at a prompt the default knows (an
                # eval's Qwen passed `\(gdb\) $` with 9 lines into a running
                # gdb, "y" and "quit\ny" too): the launch note never fired.
                for note in note_once(s, "pattern_into_known", PATTERN_INTO_KNOWN_NOTE):
                    out += "\n" + note
            if region.status == "finished" and enclosing is not None:
                out += "\n[the command finished:]\n" + describe(s, enclosing)
            if shell_after is not None:
                regions, why, unsent = shell_after
                out += "\n[the lines after it went to the shell:]"
                if regions:
                    out += "\n" + "\n".join(render_regions(s, regions, max_lines, show_rx, show_after, last_lines))
                    for r in regions:
                        note_shown(s, r)
                if unsent:
                    out += (f"\n[stopped: {unsent} more line{'s' if unsent > 1 else ''} not sent, since the one "
                            "before did not come back to the prompt]")
                if why != "done":
                    out += "\n" + await status_hint(s, why)
                elif regions and not regions[-1].done:
                    at = s.term.cursor_text().strip()
                    out += (f"\n[the command is still running and appears to be waiting for input"
                            + (f" at {at[-60:]!r}" if at else "") + "; further calls of the `run` tool are sent to it]")
                return out
            elif region.status == "exited":
                info = await mgr.tmux.pane_info(session)
                out += f"\n[the program in this session exited{exit_said(info.dead_status, ' with status {}')}]"
            elif region.status == "fullscreen":
                out += f"\n[{STATUS_HINTS['fullscreen']}]\n" + await render_screen(mgr.tmux, s, max_rows=max_lines)
            elif region.status == "waiting" and prompt_pattern is not None and (
                    at := s.term.cursor_text().strip()) and s.prompt_here(prompt_regex(None)):
                out += (f"\n[output stopped at {at[-40:]!r}, which looks like a prompt your prompt_pattern does "
                        f"not match -- the program's own (gdb's, say, once the program it ran has exited).  "
                        f"Leave prompt_pattern out (the default knows it), or match it]")
            elif region.status == "waiting" and prompt_pattern is not None and (at := s.term.cursor_text()).strip():
                offered = re.sub(r"\d+", r"\\d+", re.escape(at.lstrip()).replace("\\ ", " ")) + "$"
                out += (f"\n[your prompt_pattern `{prompt_pattern}` did not match where the output stopped: "
                        f"{at.strip()!r}.  If that is the prompt, does your pattern cover the parts that "
                        f"change (a time, a counter, the directory)?  From this one: `{offered}`]")
            elif region.status == "waiting" and (secret := await secret_prompt_hint(s)):
                out += "\n" + secret
            elif region.status == "waiting" and (yn := question_hint(s.term.cursor_text())):
                out += "\n" + yn
            elif region.status in STATUS_HINTS:
                # (This input's time, not the command it went to: a pytest run
                # typed into a shell under `su` was "62 min so far", su's age.)
                out += "\n" + await status_hint(s, region.status, region)
                if region.status == "waiting" and (at := s.term.cursor_text()).strip():
                    # Digits become \d+: counters, times and dates in a prompt
                    # change every time ("In [7]:", "[18:02:11 #7] ready: ").
                    offered = re.sub(r"\d+", r"\\d+", re.escape(at.lstrip()).replace("\\ ", " ")) + "$"
                    out += (f"\n[if {at.strip()!r} is the program's prompt, not a question: pass "
                            f"prompt_pattern `{offered}` (the regex exactly as between the backticks; here "
                            "and to the `wait` tool; digits generalised -- check it against the next prompt)]")
            elif region.status == "ready" and s.term.cursor_text().strip() == ">" and s.shell_prompt_seen():
                out += f"\n[{STATUS_HINTS['continuation']}]"
            elif region.status == "ready" and (note := prompt_change_note(s, region)):
                out += "\n" + note
            if region.status in ("ready", "waiting") and (nested := await nesting_note(s)):
                out += "\n" + nested
            return out

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'add bash shell integration over ssh or su'})
    async def shell_integrate(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        timeout_s: Annotated[float, Field(description="seconds to wait for the integrated prompt to appear")] = 10,
        force: Annotated[bool, Field(description="type even over someone else's unsubmitted line, and without checking that the cursor is at a shell prompt")] = False,
    ) -> str:
        """Add shell integration to the bash now waiting at the prompt: exact
        command boundaries and exit statuses for `run` there.

        Sessions started without a `command` have it already, and so does any
        bash started inside them.  A shell that could not inherit it -- one
        reached through ssh or `su -` -- falls back to prompt matching, with no
        exit statuses; call this at its prompt, typically right after `run ssh
        host` came back at the remote prompt.  Types one short line that reads
        the integration file: this server's own on this host; on another host
        the copy the user installed there with `docent-terminal-mcp install HOST`
        (without it the reply says so; `docent-terminal-mcp ssh HOST` needs none).
        For bash and zsh, and only at a prompt."""
        s = await mgr.get(session)
        async with typing_turn(s, "shell_integrate"):
            s.sync()
            if s.term.full_screen:
                raise ToolError("a full-screen program is running; get back to a bash prompt first")
            await check_human(s, force)
            if not force:
                await refuse_unless_shell_prompt(s)
            if s.at_shell_prompt:
                # Markers already come from the shell at the prompt (`docent-terminal-mcp
                # ssh` brought them along): the installed-file line would only
                # fail there, on screen.
                return "[shell integration already active here: nothing typed]"
            shell = await probe_shell(s, min(timeout_s, 5))
            if shell not in ("bash", "zsh"):
                what = f"it is {shell}" if shell else "no bash, zsh or fish answered"
                return (f"[not bash or zsh ({what}): no integration -- the `run` tool still works there, "
                        "without exact exit statuses]")
            directory, place = await integration_source(s)
            line = integration_line(s.meta.get("nonce"), shell, directory)
            before = s.offset
            start = s.content_end()
            s.shell_state = "unknown"  # wait for the newly marked prompt
            await mgr.tmux.send_text(session, line)
            await mgr.tmux.send_keys(session, ["Enter"])
            s.note_sent()
            # (the echoed line names the file too: only what follows it counts)
            not_read = lambda: bool(NOT_READ.search(s.span_text(start, s.content_end()).replace(line.strip(), "")))
            why = await s.wait(lambda: (s.shell_state == "input" and s.offset > before) or not_read(), timeout_s)
            s.note_model_input()
            if not_read() and not (s.shell_state == "input" and s.offset > before):
                if directory is not None and place is not None:
                    note_integration_missing(s, place.label())
                return not_installed(directory, shell, place)
            if why != "done":
                screen = await render_screen(mgr.tmux, s, limit=5)
                return f"[no marked prompt appeared ({why}) -- is the program at the prompt bash?]\n{screen}"
            if place is not None:
                note_integration_missing(s, place.label(), found=True)
            await note_place(s)
            return "[shell integration active: commands here now report exact output and exit status]"

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'answer prompts and questions, y/n, password, interactive script'})
    async def expect(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        respond: Annotated[list[Respond], Field(description="the rules: each fires when its `pattern` appears, doing its one action (line, text, keys or stop); a question no rule covers ends the call as \"unanswered\"")],
        input: Annotated[str | None, Field(description="a line to type first (Enter added), e.g. the command that will ask the questions; without it the rules are applied to a question already waiting at the cursor")] = None,
        until: Annotated[str | None, Field(description="regex: return as soon as the output matches it, the command still running (status \"until\"; `wait` resumes); not a prompt pattern")] = None,
        timeout_s: Annotated[float, Field(description="seconds to wait before giving up (the command keeps running; `wait` resumes); at most 1500")] = 60,
        prompt_pattern: Annotated[str | None, Field(description="regex that marks the program's prompt at the end of the cursor line; rarely needed, the default knows shell, gdb, pdb, lldb and python prompts and any ending in $ # % >; pass one only after a reply came back \"waiting\" at a prompt it did not know (it offers the pattern); make it cover a part that changes")] = None,
        max_lines: Annotated[int, Field(description="lines of output shown in the reply; longer output is clipped in the middle, led by its summary lines (test results, compiler errors, make's failure) when the middle holds them; all of it stays readable with `read`/`search`")] = 80,
        force: Annotated[bool, Field(description="type even over someone else's unsubmitted line at the prompt, discarding it first; only for a line you know is stale")] = False,
    ) -> str:
        """Script predictable interactions in one call: optionally type `input`
        (+ Enter), then answer each prompt that matches a `respond` rule, in
        whatever order the prompts appear.  The interactive form of `yes |`,
        `echo y |`, gdb's -batch and an expect script: the program gets a real
        terminal, and only the questions you foresaw are answered.  Rules that
        never match are fine -- use them for questions only sometimes asked.

        Ends when the command finishes (bash), the program is back at its
        prompt (`prompt_pattern`, default as for `run`), `until` (a regex)
        matches, a `stop` rule matches, or -- the important case -- output
        stops at a question no rule answers: that returns at once as
        "unanswered" instead of waiting for the timeout, so a surprise comes
        back to you rather than being guessed at.

        Example, gdb's optional questions:
          input="gdb -q -ex 'break lib_fn' -ex run ./prog",
          respond=[{"pattern": "Make breakpoint pending", "line": "y"},
                   {"pattern": "Enable debuginfod", "line": "n"}]
        Example, ssh, handing any password prompt to the human:
          input="ssh host", respond=[{"pattern": "continue connecting", "line": "yes"},
                                     {"pattern": "[Pp]assword:", "stop": true}]
        A plain ssh like this one is refused once (no exit statuses on the
        far side): "docent-terminal-mcp ssh host" carries the shell
        integration along; sent again unchanged, the plain one runs.
        Without `input`, the rules apply to a question already waiting at the
        cursor (one a `run` came back "waiting" at).

        Returns a transcript (what matched, what was sent), rules that never
        fired, and the resulting region(s) as `run` does."""
        s = await mgr.get(session)
        timeout_s = cap_timeout(timeout_s)
        rules = []
        key_names: dict[int, list[str]] = {}
        for i, r in enumerate(respond):
            actions = [a for a in ("line", "text", "keys") if getattr(r, a) is not None]
            if len(actions) > 1 or not (actions or r.stop):
                raise ToolError(f"respond[{i}] must have one of line, text, keys -- or stop alone, or one of "
                                "them with stop (answer, then stop)")
            then_stop = bool(actions) and r.stop
            actions = actions or ["stop"]
            if r.keys is not None:
                try:
                    key_names[i] = [normalize_key(k) for k in r.keys]
                except TmuxError as e:
                    raise ToolError(f"respond[{i}]: {e}") from e
            value = {"line": r.line, "text": r.text, "keys": r.keys, "stop": None}[actions[0]]
            rules.append(Rule(compile_pattern(f"(?m:{r.pattern})"), actions[0], value,
                              max(1, r.times), r.secret, source=r.pattern, then_stop=then_stop))
        until_rx = compile_pattern(f"(?m:{until})") if until else None
        prompt_rx = call_prompt(s, prompt_pattern)
        async with replying(s, "expect"):
            await refuse_if_exited(s)
            s.sync()
            if s.term.full_screen:
                raise ToolError("a full-screen program is running; use the `send` tool")
            forced = ""
            if input is not None:
                await check_human(s, force)
                if discard_pending(s, force):
                    forced = await clear_prompt(s) + "\n"
                if s.at_shell_prompt or (s.shell_in_front() and s.at_plain_shell_prompt()):
                    await second_guesses(s, input, None, None, tool="expect")
            if not s.at_shell_prompt:
                report_foreign_input(s)
            were_open = open_command_ids(s) | s.open_at_reply
            known = len(s.regions)
            if input is not None:
                s.note_interrupt_survivors(set())  # a new command (see `run`)
            result = await s.run_expect(input, rules, until_rx, prompt_rx, timeout_s, key_names)
            n = len(result.transcript)
            answers = {r.value for r in rules if r.action == "line" and r.fired and not r.secret}
            strayed = [r for r in s.regions[known:] if r.kind == "command" and r.command in answers]
            if not strayed and answers and result.outcome == "done" and s.integrated:
                # The program may have exited without reading its answer: the
                # shell runs it as a command once back at its prompt -- under
                # load, after this reply would have gone.  Wait for that, or
                # for the shell to sit quietly at its prompt.
                def ran_or_idle() -> bool:
                    return any(r.kind == "command" and r.command in answers for r in s.regions[known:]) or (
                        s.shell_state == "input" and time.monotonic() - s.last_output >= ANSWER_QUIET_S)
                await s.wait(ran_or_idle, 2)
                strayed = [r for r in s.regions[known:] if r.kind == "command" and r.command in answers]
            if any(not r.done for r in strayed):
                # An answer typed at the shell is a command of a few words; report
                # its exit status rather than "running" (a race under load).
                await s.wait(lambda: all(r.done for r in strayed), 2)
            outcome = EXPECT_OUTCOMES[result.outcome]
            if result.outcome == "exited":
                outcome += exit_said((await mgr.tmux.pane_info(session)).dead_status, " with status {}")
            asked_hint = None
            if result.outcome == "unanswered" and (at := s.term.cursor_text().strip()):
                # Say which: "a question no rule answers" right after a rule
                # that fired read as a contradiction, when what stopped it
                # was gdb's pager at the end of a long `bt`.
                asked_hint = question_hint(at)
                if any(rx.search(at) for rx, _ in PAGER_HINTS):
                    outcome = f"output stopped at a pager, {at[-70:]!r} -- not a question; see below"
                else:
                    outcome = outcome.replace("a question no rule answers",
                                              f"a question no rule answers, {at[-70:]!r}", 1)
            parts = [f"{forced}[expect: {outcome} · {n} response{'s' if n != 1 else ''}]"]
            if asked_hint:
                parts.append("  " + asked_hint)
            if result.outcome == "timeout" and (jobs := await foreground_jobs(s)):
                parts.append(f"  in the foreground: {named(jobs)}")
            parts += [f"  {t}" for t in result.transcript]
            if until and result.outcome == "ready":
                parts.append(f"  `until` {until!r} did not match before the program was back at its prompt")
            if result.unused:
                parts.append("  not triggered: " + ", ".join(f"'{u}'" for u in result.unused))
            echoed = [r for r in rules if r.secret and r.fired and isinstance(r.value, str) and len(r.value) >= 3
                      and r.value in s.span_text(result.text_start, s.content_end())]
            if echoed:
                parts.append("  [the terminal echoed a secret=true answer: the program did not hide it, so it is in "
                             "the session's log and output (secret=true only keeps it out of this transcript)]")
            for r in strayed:
                parts.append(f"  the answer {r.command!r} ran at the shell as a command (region {r.id}"
                             f"{f', exit {r.exit_status}' if r.exit_status is not None else ''}): "
                             "the program that asked had exited before the answer reached it")
            for r in [*result.regions, *([s.open_command] if s.open_command else [])]:
                note_shown(s, r)  # (a `wait` after an `until` shows only what is new)
            if result.regions:
                parts += render_regions(s, result.regions, max_lines)
                parts += result.notes
            else:
                new = s.clip_span(result.text_start, s.content_end(), max_lines, "the `read` tool shows more")
                if new:
                    parts.append(new)
            if result.outcome == "fullscreen":
                parts.append(await render_screen(mgr.tmux, s, max_rows=max_lines))
            parts += outer_finished(s, were_open, result.regions)
            return "\n".join(parts)

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'send keystrokes, ctrl+c, interrupt, signal, full-screen program, vim, less'})
    async def send(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        text: Annotated[str, Field(description="typed literally, first; a newline in it is Enter")] = "",
        keys: Annotated[list[str] | None, Field(description="key names pressed after the text, in order: ctrl+c, ctrl+d, enter, tab, escape, backspace, up, down, left, right, home, end, pageup, pagedown, f1-f12, alt+x, or a single character")] = None,
        wait_ms: Annotated[int, Field(description="the most to wait for output to settle before replying (milliseconds)")] = 2000,
        settle_ms: Annotated[int, Field(description="reply once output has paused this long (milliseconds)")] = 300,
        max_lines: Annotated[int, Field(description="lines of output, or rows of the screen, shown in the reply")] = 80,
        screen: Annotated[str, Field(description="with a full-screen program: \"full\" (every row), \"changes\" (rows changed since the last screen you saw; cheapest when editing), or \"cursor\" (the rows around the cursor)")] = "full",
        force: Annotated[bool, Field(description="type even over someone else's unsubmitted line at the prompt, discarding it first (only for a line you know is stale); with `signal`: send it although the foreground job is ssh, which it would end")] = False,
        signal: Annotated[str | None, Field(description="instead of text and keys: \"TERM\", \"KILL\", \"INT\", \"HUP\" or \"QUIT\", sent to the foreground job (never the session's own shell or program) when ctrl+c and ctrl+d fail; TERM first")] = None,
    ) -> str:
        """Send keystrokes without waiting for a command's result (that is
        `run`).  ORDER: all of `text` is typed first (literally; a newline in it
        is Enter), THEN the `keys` are pressed in list order.  To press a key
        before typing (escape, then an ex command), make two calls.

        For: interrupting (ctrl+c; ctrl+d ends input, or gdb/python at their
        prompt; `q` leaves a pager), answering a one-character prompt, tab
        completion, history and arrows in a REPL, and driving full-screen
        programs (vim, less, top).  Command lines go through `run`, which waits
        for their result; `send` only waits until output pauses for `settle_ms`
        (at most `wait_ms`) and returns what appeared -- or, while a full-screen
        program runs, the rendered screen with the cursor position: `screen`=
        "full" (every row), "changes" (only rows changed since the screen last
        returned: cheapest for editing), or "cursor" (the rows around the
        cursor).  With nothing to send, it just reads what is new.

        Keys are names such as ctrl+c, ctrl+d, enter, tab, escape, backspace,
        up, down, left, right, home, end, pageup, pagedown, f1-f12, alt+x, or
        any single character.  Shell commands that finish during the call are
        reported with their exit status.  Refuses, like `run`, to type over a
        human's unsubmitted line unless `force`.  When nothing new appears, or
        a program is still in the foreground after ctrl+c, the reply names what
        is running there.

        `signal` ("TERM", "KILL", "INT", "HUP", "QUIT"), instead of text and
        keys, sends that signal to the terminal's foreground job -- never the
        session's own shell or program: for a job that ignores ctrl+c and
        ctrl+d.  TERM first; KILL if it survives.  Prefer it to `pkill` from
        outside, which can hit the session's own shell."""
        s = await mgr.get(session)
        try:
            tmux_keys = [normalize_key(k) for k in keys or []]
        except TmuxError as e:
            raise ToolError(str(e)) from e
        if screen not in SCREEN_MODES:
            raise ToolError(f"screen must be one of {', '.join(SCREEN_MODES)}")
        signum = None
        if signal is not None:
            if text or tmux_keys:
                raise ToolError("`signal` goes alone: send text and keys in a call of their own")
            signum = SIGNALS.get(signal.upper().removeprefix("SIG"))
            if signum is None:
                raise ToolError(f"signal must be one of {', '.join(SIGNALS)}")
        interrupt = [k for k in tmux_keys if k in INTERRUPT_KEYS]
        # Only ctrl+c, or only a signal: at once, as a person's would -- not
        # after a run in flight (another agent's, or the caller's own that its
        # client moved to the background), which it is usually meant to end.
        urgent = signum is not None or (bool(tmux_keys) and not text and len(interrupt) == len(tmux_keys))
        async with replying(s, None if urgent else "send" if text or tmux_keys else None):
            s.sync()
            if text or tmux_keys or signum:
                await refuse_if_exited(s)
            if text or tmux_keys:
                await check_human(s, force, "send")
            start = (s.term.row, 0)
            # Rows above the cursor a program may rewrite in place (an inline
            # menu, a dialog drawn with cursor-up): shown again if changed.
            paintings, was_full = s.term.paintings, s.term.full_screen
            above = range(max(0, s.term.row - REWRITE_ROWS), s.term.row)
            before_rows = [s.term.line(r) for r in above]
            open_ids = {r.id for r in s.regions if not r.done} | s.open_at_reply
            known = len(s.regions)
            before = await foreground_jobs(s) if interrupt or signum else []
            head, notes = [], []
            if signum is not None:
                if not force and (remote := [p for p in before if is_carrier(p.pid, p.command)]):
                    raise ToolError(
                        f"the foreground job is {named(remote)}: a signal would end the connection, not the "
                        "remote command, whose processes are on the other host.  Send keys instead (ctrl+c, "
                        "ctrl+d) -- they reach the remote side.  force=true sends the signal anyway")
                if not before:
                    raise ToolError("nothing to signal: nothing runs in the foreground but the session's own "
                                    "shell or program, which is left alone (a finished command, or one that "
                                    "never started?  the `regions` tool shows).  Only to end the session itself: "
                                    "session_close")
                try:
                    signal_job(before, signum)
                except ProcessLookupError:
                    raise ToolError(f"the foreground job ({named(before)}) ended before the signal") from None
                except PermissionError:
                    raise ToolError(f"not permitted to signal {named(before)} (another user's process: "
                                    "sudo, su?); ctrl+c, or ask the user") from None
                head.append(f"[sent SIG{SIGNALS_BY_NUMBER[signum]} to the foreground job: {named(before)}]")
            driving = s.model_driving  # a run in flight keeps its own
            if not urgent:
                s.model_driving = ACTOR.get() if (text or tmux_keys) else None
            several = text.rstrip("\n").count("\n") >= 1 and not s.at_shell_prompt
            text = text.replace("\r\n", "\n").replace("\n", "\r")  # Enter, as a keyboard sends it
            if text or tmux_keys:
                if not s.at_shell_prompt:
                    report_foreign_input(s)
                s.note_sent()
            try:
                await mgr.tmux.send_text(session, text)
                await mgr.tmux.send_keys(session, tmux_keys)
                sent = time.monotonic()
                await s.wait(
                    lambda: time.monotonic() - sent >= wait_ms / 1000, wait_ms / 1000 + 1,
                    settle_s=settle_ms / 1000,
                )
            finally:
                s.model_driving = driving if urgent else None
            s.note_model_input()
            closed = s.close_fullscreen_region(s.prompt_pattern)
            ended = [r for i, r in enumerate(s.regions)
                     if r is closed or (r.kind == "command" and r.done and (r.id in open_ids or i >= known))]
            finished = [describe(s, r) for r in ended]
            # (After the caller's own ctrl+c no note is needed.)
            interrupted = [] if interrupt else await interrupt_notes(s, ended)
            if s.term.full_screen or s.term.paintings > paintings or (was_full and not finished and not s.at_shell_prompt):
                # (Painted and ended within the call -- a popup, cursor shown
                # again: the screen, not the text, shows it.  So too a program
                # that painted without the alternate screen and shows the
                # cursor for a prompt of its own: top's `O` asks "add filter
                # #1" on its screen, and the reply said "no new output".)
                out = await render_screen(mgr.tmux, s, mode=screen, max_rows=max_lines)
            else:
                changed = [r for r, old in zip(above, before_rows)
                           if r < len(s.term.lines) and s.term.line(r) != old]
                if changed:
                    start = (changed[0], 0)
                out = s.clip_span(start, s.output_end(), max_lines, "the `read` tool shows more")
                out = out.rstrip("\n") if out.strip() else ""  # an inline program's blank rows below
                # Trailing blanks are trimmed from the text, but a space
                # before the cursor matters (did completion add one?).
                before_cursor = s.term.cursor_text()
                if s.term.row == len(s.term.lines) - 1 and len(before_cursor) > len(before_cursor.rstrip()):
                    gap = len(before_cursor) - len(before_cursor.rstrip())
                    out += f"\n[cursor is {gap} space{'s' if gap > 1 else ''} past the end of the last line]"
                if (text or tmux_keys or signum) and not finished:
                    notes += await foreground_notes(s, before, interrupt, signum, quiet=not out.strip())
            s.close_stale_input(interrupted=bool(interrupt))
            if several and not s.term.full_screen:  # (an editor in insert mode: just typing)
                notes += note_once(s, "send_lines", SEND_LINES_NOTE)
            if unsubmitted := typed_not_submitted(s, text, tmux_keys, finished):
                notes.append(unsubmitted)
            return "\n".join(head + ([out] if out else []) + notes + [f"[finished: {d[1:]}" for d in finished]
                             + interrupted)

    @tool(meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'wait for a running command in a terminal session, poll'})
    async def wait(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        timeout_s: Annotated[float, Field(description="seconds to keep waiting (the command keeps running after; `wait` again); at most 1500")] = 60,
        max_lines: Annotated[int, Field(description="lines of output shown in the reply; longer output is clipped in the middle, led by its summary lines (test results, compiler errors, make's failure) when the middle holds them; all of it stays readable with `read`/`search`")] = 80,
        show: Annotated[str | None, Field(description="regex: reply with only the output lines matching it, in order, plus the last few (like `| grep`, but the command runs unfiltered and the log keeps everything), e.g. \"error:|warning:|^FAIL\" for a build -- or a program's answer: gdb's `step` under `set debug infrun 1`")] = None,
        until: Annotated[str | None, Field(description="regex: return as soon as output not shown to you before matches it, the command still running (as for `run`); one that never matches is a plain wait")] = None,
        save_to: Annotated[str | None, Field(description="a file to copy the command's output into, as `> FILE 2>&1` would, while the command itself runs unredirected (the reply and the log are as usual): written at once, kept up to date while it runs, written in full when it ends.  A relative path is in the session's current directory; the file is on the machine this server runs on, even for a session over ssh.  For a later diff or grep by other tools; `read` and `search` need no file")] = None,
        last_lines: Annotated[int | None, Field(description="reply with only the last N lines of the output (like `| tail -N`, but the command runs unfiltered and the log keeps everything; the reply says where the rest is); with `show`, how many last lines follow the matches (default 3)")] = None,
        show_after: Annotated[int, Field(description="with `show`: also this many lines after each match (like `grep -A N`); a match is numbered \"N:\", a line after one \"N-\"")] = 0,
    ) -> str:
        """Keep waiting for what is running in the session -- what your last call
        left running (a `run` that timed out, settled, or returned at `until`),
        or else a command a human or another agent started -- and return its
        region: only the lines not shown before.  The interactive form of
        `sleep 30; tail log`: call it as often as needed, with `timeout_s` up to
        the client's patience; each reply is new output only, and the command's
        exit status when it ends.  Commands of others that finished since your
        last call are reported at once (`regions` lists the session without
        waiting).  When a human's (or another agent's) unsubmitted text holds
        the prompt, waits for it to be erased or submitted, and says which.
        After a `run` into a program (gdb, python), waits for the program's
        prompt; after a shell command, for it to finish.  `show` filters the
        lines as for `run`; `until` returns at a pattern, `last_lines` and
        `save_to` work as for `run`."""
        s = await mgr.get(session)
        timeout_s = cap_timeout(timeout_s)
        show_rx = compile_pattern(show) if show else None
        until_rx = compile_pattern(until) if until else None
        check_output_options(last_lines, show_after, show)
        if save_to:
            target = await save_target(s, save_to)
            REPLY_AFTER.get().append(lambda: save_reply(s, target))
        async with replying(s):
            s.sync()
            info = await mgr.tmux.pane_info(session)
            if info.dead:
                s.pane_exited(info)  # its shell's `exit N` is the status of what it ran last
            if (pending := s.foreign_input()) and not s.term.full_screen:
                return await wait_out_foreign_line(s, pending, timeout_s, max_lines)
            last = s.regions[-1] if s.regions else None
            if last is not None and last.kind == "input" and last.status in ("timeout", "settled", "waiting", "until"):
                pattern = re.compile(last.prompt_pattern) if last.prompt_pattern else s.prompt_pattern
                seen = s.shown_upto.get(last.id)
                status = await s.resume_input(last, pattern, timeout_s, until_rx,
                                              (seen, 0) if seen and seen > last.start[0] else None)
                out = render_progress(s, last, max_lines, show_rx, show_after, last_lines)
                if status == "ready" and (note := prompt_change_note(s, last)):
                    out += "\n" + note
                return out + ("\n" + await status_hint(s, status, last) if status in STATUS_HINTS else "")
            region = s.open_command
            if region is not None and s.at_shell_prompt:
                # The integrated shell nearest the cursor is at its prompt: what
                # stays open is the ssh (or bash) it runs in, not a command.
                region = None
            since = s.finished_unseen()
            if region is None and since:
                return "\n".join(["[finished since your last call:]"]
                                  + [render_progress(s, r, max_lines, show_rx, show_after, last_lines) for r in since]
                                  + pipe_notes(s, since, show_rx, last_lines)
                                  + await interrupt_notes(s, since))
            if region is None:
                # No command region: a session whose program is not a shell
                # (session_create command="python3 crawl.py"), or a shell at
                # its prompt.  `until` still ends the wait: it sat out its whole
                # timeout_s while the program printed the pattern.
                start = (s.term.row, 0)
                watch = UntilWatch(s, until_rx)
                why = await s.wait(lambda: watch(start), timeout_s,
                                   settle_s=None if until_rx else DEFAULT_SETTLE_MS / 1000)
                head = ("[`until` matched; what runs in the session goes on running]" if watch.hit
                        else f"[no command running; {why}]")
                out = head + "\n" + s.clip_span(start, s.content_end(), max_lines, "the `read` tool shows more")
                if watch.hit:
                    return out
                info = await mgr.tmux.pane_info(session)
                if jobs := await background_processes(info.tty, info.pid):
                    out += (f"\n[no command in the foreground, but in the background: {named(jobs)} -- started with "
                            "`&`, which the `wait` tool cannot wait for.  Next time run it without `&` and with "
                            "timeout_s up to 1500: the reply comes when it ends, with its exit status (a client that "
                            "moves long calls to the background, as Claude Code does, tells you then), and other "
                            "work goes on in another session (session_create, session_split).  This one's output goes "
                            "where it was sent (the "
                            "session's log, or its log file); `jobs` or `ps` shows whether it still runs]")
                return out
            pattern = s.prompt_pattern
            asking: list[str] = []
            # The open command is the ssh (or container entry) itself: the
            # shell on its far side is a plain one -- an integrated one's
            # command would be the open region -- so nothing but the text
            # says when what runs there stops.  Whoever started the ssh, a
            # prompt or question over there is the answer, not the ssh's
            # whole life (`wait` sat out its timeout at "[sudo] password
            # for alice: ").
            carrier = region.kind == "command" and is_carrier_command(region.command or "")
            watch = UntilWatch(s, until_rx)
            seen = s.shown_upto.get(region.id)
            unseen = (seen, 0) if seen and seen > region.start[0] else region.start
            via = os.path.basename(region.command.split()[0]) if carrier else ""

            def done_or_asking() -> bool:
                # At a program's prompt (>>>, (gdb)), or at the shell prompt
                # known for it -- over ssh, "user@host:~$ " printed by a
                # command still running is not one.
                # (Only your own: another agent's command, you wait out.)
                if not region.done and (region.source == ACTOR.get() or carrier) and s.awaiting_input(pattern) and (
                        (m := DEFAULT_PROMPT.search(s.term.cursor_text())) is None
                        or s._prompt_kind(m) != "sh" or carrier
                        or ((sig := s.prompt_signature()) is not None and sig in s.term.cursor_text())):
                    asking.append(s.term.cursor_text())
                return region.done or bool(asking) or watch(unseen)

            why = await s.wait(done_or_asking, timeout_s)
            out = render_progress(s, region, max_lines, show_rx, show_after, last_lines)
            out = "\n".join([out] + pipe_notes(s, [region], show_rx, last_lines))
            if asking and not region.done:
                at = asking[-1].strip()
                if carrier and (m := DEFAULT_PROMPT.search(asking[-1])) and s._prompt_kind(m) == "sh":
                    return out + (f"\n[the shell at the far end of this {via} is at its prompt "
                                  f"{at[-60:]!r}: nothing runs there now.  It is a plain shell (no exit statuses); "
                                  "the `run` tool types the next command into it]")
                where = f"at {at[-60:]!r}" if at else "(reading, with no prompt printed)"
                if tag := elapsed_tag(s, region):
                    where += " " + tag
                if carrier:
                    owner = "a human's" if region.source == "human" else f"{region.source or 'someone'}'s"
                    whose = "" if region.source == ACTOR.get() else f" (in {owner} {via})"
                    out += f"\n[what runs at the far end{whose} is waiting for input {where}]"
                else:
                    out += f"\n[the command is waiting for input {where}; the `run` tool sends it the next line]"
                if secret := await secret_prompt_hint(s):
                    return out + "\n" + secret
                return out + (f"\n{hint}" if (hint := question_hint(at)) else "")
            if region.done:
                out += "".join("\n" + n for n in await interrupt_notes(s, [region]))
            elif watch.hit:
                return out + "\n" + await status_hint(s, "until", region)
            if why != "done":
                out += "\n" + await status_hint(s, why, region)
            return out

    @tool(annotations=ToolAnnotations(read_only_hint=True), meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'read terminal output, log, region, screen'})
    async def read(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        region: Annotated[int | None, Field(description="read only this region (its id from a reply header or `regions`); default: the whole session log")] = None,
        start: Annotated[int | None, Field(description="cooked: the line number to start from (1-based, the numbers replies and `search` show; the same in a region and in the log); screen: the first row; raw: the byte offset")] = None,
        limit: Annotated[int, Field(description="lines (or screen rows) to return; without `start`, the last that many")] = 100,
        view: Annotated[str, Field(description="\"cooked\" (the log as a plain-text file, one entry per real line), \"screen\" (the terminal as displayed now, rows as tmux renders them; for full-screen programs), or \"raw\" (the bytes written, escape codes spelled out)")] = "cooked",
        max_bytes: Annotated[int, Field(description="raw view only: the most bytes to return")] = 4000,
    ) -> str:
        """Read session output: the log or one region, by line number -- what
        `tail -n`, `sed -n` or `head` would do on a log file you had redirected
        to, except that nothing had to be redirected.  A reply's "[... N lines
        omitted (lines A-B) ...]" marker gives the range to read.

        view="cooked" (default): the whole-session log as plain text with line
        numbers, real line breaks only.  With `region`, reads that region;
        `start` is a line number (1-based, as shown) to page from.  Without
        `start`, shows the last `limit` lines.
        view="screen": what the terminal shows right now, row by row as tmux
        renders it (a long line wraps onto several rows), with the size and
        cursor position.  `start`/`limit` select screen rows.  The view to use
        for full-screen programs.  Note that such programs clip long lines at
        the window edge; check file contents another way.
        view="raw": the bytes the program wrote, escape codes included, with
        control characters spelled out (\\e, \\r, \\xNN).  For debugging odd
        rendering.  `start` is a byte offset; `max_bytes` bounds the result.
        With `region`, reads that region's bytes (bounds may include a little
        output from either side)."""
        s = await mgr.get_logs(session)
        if view == "screen" and mgr.sessions.get(session) is not s:
            raise ToolError(f"session {session!r} is closed: it has no screen.  view=\"cooked\" (its log) or "
                            "\"raw\" read what it showed")
        if view == "screen":
            with reading(s):
                return await render_screen(mgr.tmux, s, start, limit)
        if view == "raw":
            with reading(s):
                s.sync()
                lo, hi = 0, s.offset
                if region is not None:
                    try:
                        r = s.region(region)
                    except KeyError as e:
                        raise ToolError(str(e)) from e
                    if r.raw_start is None:
                        raise ToolError(f"region {region} predates raw offsets; read without `region`")
                    lo, hi = r.raw_start, r.raw_end if r.raw_end is not None else s.offset
                first = max(lo, start) if start is not None else max(lo, hi - max_bytes)
                last = min(hi, first + max_bytes)
                with open(s.raw_path, "rb") as f:
                    f.seek(first)
                    data = f.read(last - first)
                header = f"[{s.raw_path} bytes {first}-{last}"
                header += f" of region bytes {lo}-{hi}" if region is not None else f" of {hi}"
                header += f"; next start={last}]" if last < hi else "]"
                return header + "\n" + escape_raw(data)
        if view != "cooked":
            raise ToolError("view must be 'cooked', 'screen' or 'raw'")
        with reading(s):
            s.sync()
            if region is not None:
                try:
                    r = s.region(region)
                except KeyError as e:
                    raise ToolError(str(e)) from e
                s.region_file(r)  # being worked with: have its file on hand
                rows = s.region_rows(r)
                header = describe(s, r) + "\n"
            else:
                rows = range(0, len(s.term.lines))
                header = ""
            note = ""
            if region is not None and not rows:
                return header + f"[region {region} has no output{' yet' if not r.done else ''}]"
            if start is None:
                first = max(rows.start, rows.stop - limit)
            else:
                first = max(rows.start, start - 1)
                where = f"region {region}" if region is not None else "the log"
                if start - 1 < rows.start:
                    note = f"[start {start} is before {where}; starting at {rows.start + 1}]\n"
                elif start - 1 >= rows.stop:
                    return header + f"[start {start} is past the end of {where} (last line {rows.stop})]"
            shown = range(first, min(rows.stop, first + limit))
            header += note
            footer = ""
            if shown.stop < rows.stop:
                footer = f"\n[{rows.stop - shown.stop} more lines; next start={shown.stop + 1}]"
            elif shown.start > rows.start and start is None:
                footer = f"\n[{shown.start - rows.start} earlier lines; start={rows.start + 1} for the beginning]"
            span = (r.start, s.region_end(r)) if region is not None else None
            return header + numbered(s, shown, span) + footer

    @tool(annotations=ToolAnnotations(read_only_hint=True), meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'search grep terminal output log'})
    async def search(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        pattern: Annotated[str, Field(description="Python regex, tried against each line of the cooked log on its own (^ and $ anchor a line); the reply gives the total number of matching lines, then the hits in log order with their line numbers (usable as `read start=`) and regions")],
        region: Annotated[int | None, Field(description="search only this region's output; default: the whole session log")] = None,
        context: Annotated[int, Field(description="lines shown before and after each hit (0: the hit alone)")] = 2,
        max_results: Annotated[int, Field(description="hits to show per call; the reply says how many there are in all")] = 20,
        skip: Annotated[int, Field(description="hits to skip first, for paging (the reply's \"[more: skip=N]\")")] = 0,
        ignore_case: Annotated[bool, Field(description="match case-insensitively")] = False,
        literal: Annotated[bool, Field(description="take `pattern` as plain text, not a regex")] = False,
    ) -> str:
        """Search session output (regex by default) without reading it all: what
        `grep` would do on a log file you had redirected to.  Use it for a
        build's errors, a testsuite's FAIL lines, a value in a long trace --
        after the command, or while it still runs.

        Reports the total number of matching lines (so you know whether you
        saw them all), then up to `max_results` of them (after skipping `skip`),
        each with `context` lines around it, its line number (the number `read
        start=` takes) and the region it belongs to.  Restrict to one region
        with `region`."""
        s = await mgr.get_logs(session)
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(re.escape(pattern), flags) if literal else compile_pattern(f"(?{'i' if ignore_case else ''}:{pattern})")
        with reading(s):
            s.sync()
            if region is not None:
                try:
                    rows = s.region_rows(s.region(region))
                except KeyError as e:
                    raise ToolError(str(e)) from e
            else:
                rows = range(0, len(s.term.lines))
            hits = [r for r, line in s.iter_lines(rows) if rx.search(line)]
            total = len(hits)
            page = hits[skip : skip + max_results]
            out = [f"{total} matching line{'s' if total != 1 else ''}"
                   + (f"; showing {skip + 1}-{skip + len(page)}" if total > len(page) else "")]
            last_reg = -1
            for r in page:
                reg = s.region_at(r)
                if reg is None:
                    label = "outside any region (prompt/command line)"
                elif reg.id == last_reg:
                    label = f"region {reg.id}"  # named in full above
                else:
                    cmd = reg.command.splitlines()[0] if reg.command else reg.kind
                    label = f"region {reg.id} ({cmd if len(cmd) <= 60 else cmd[:57] + '...'})"
                last_reg = reg.id if reg else -1
                if context == 0:
                    # One line per hit; the region named only when it changes.
                    if reg is None or label != f"region {reg.id}":
                        out.append(f"-- {label}")
                    out.append(f"{r + 1}\t{around_match(s.term.line(r).rstrip(), rx)}")
                    continue
                lo, hi = max(rows.start, r - context), min(rows.stop, r + context + 1)
                width = len(str(hi))
                block = [f"-- line {r + 1}, {label}"]
                for i, line in s.iter_lines(range(lo, hi)):
                    mark = ">" if i == r else " "
                    text = around_match(line.rstrip(), rx) if i == r else clip_line(line.rstrip(), CONTEXT_LINE_CHARS)
                    block.append(f"{mark}{i + 1:>{width}}\t{text}")
                out.append("\n".join(block))
            if skip + len(page) < total:
                out.append(f"[more: skip={skip + len(page)}]")
            return "\n".join(out)

    @tool(annotations=ToolAnnotations(read_only_hint=True), meta={"anthropic/alwaysLoad": True, "anthropic/searchHint": 'list commands and their output regions in a terminal session'})
    async def regions(
        session: Annotated[str, Field(description="the session's name (from session_create or session_list)")],
        last: Annotated[int, Field(description="how many of the most recent regions to list")] = 20,
    ) -> str:
        """List the most recent regions: each command or program input with its
        status (running, done, waiting at a question, ...), exit status, line
        range and file.  The session's history at a glance: to find a region id
        for `read`/`search`, to check on a command left running without waiting
        for it, or to see what a human or another agent ran."""
        if last < 1:
            raise ToolError("last must be at least 1")
        s = await mgr.get_logs(session)
        with reading(s):
            s.sync()
            listed = s.regions[-last:]
            earlier = len(s.regions) - len(listed)
            head = [f"[{earlier} earlier region{'s' if earlier != 1 else ''} not listed; last={len(s.regions)} lists all]"] if earlier else []
            return "\n".join(head + [describe(s, r) for r in listed]) or "no regions yet"

    return mcp


DEFAULT_PORT = 8766
CLOSED_LISTED = 8  # closed sessions shown by session_list (the most recent)
CREATING_GRACE_S = 120  # a session directory with no pane yet, this new, is being made


def default_home() -> Path:
    return Path(os.environ.get("DOCENT_TERMINAL_MCP_HOME", Path.home() / ".local/state/docent-terminal-mcp"))


def hold_lock(home: Path):
    """One server per state directory: two would both write the same logs
    and indexes.  Returns the open lock file (kept for the process's life)."""
    import fcntl

    home.mkdir(parents=True, exist_ok=True)
    lock = open(home / "server.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        owner = lock.read().strip() or "another process"
        sys.exit(f"docent-terminal-mcp: {home} is in use by {owner}.  With a shared server running, "
                 f"connect to it (`docent-terminal-mcp connect`, or over HTTP with `docent-terminal-mcp token`) "
                 f"instead of starting another.")
    import socket as so

    lock.seek(0)
    lock.truncate()
    lock.write(f"pid {os.getpid()} on {so.gethostname()} ({' '.join(sys.argv[1:]) or 'stdio'})\n")
    lock.flush()
    return lock


def read_token(home: Path) -> str:
    """The shared server's bearer token, created (mode 0600) on first use:
    other local users can reach a localhost port, but not this file."""
    path = home / "token"
    if not path.exists():
        import secrets

        home.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    return path.read_text().strip()


INSTANCE = secrets.token_hex(8)  # this server process: the bridge sees a restart by it


def identity() -> dict:
    """What a client (one holding several servers, say) checks before
    sending a token: that this is docent-terminal-mcp, and whose, on which host;
    and which process (`instance`, new at every start)."""
    import getpass
    import socket as so

    return {"service": NAME, "host": so.gethostname(), "user": getpass.getuser(), "instance": INSTANCE}


def default_socket_path() -> Path:
    """Where the shared server listens: a Unix socket in the user's runtime
    directory -- local to the host, one per user, admitting only that user
    -- so several users on a host need no port numbers and no tokens.  Not
    the state directory, which may be on NFS, where a socket file would be
    stale on every other host."""
    if path := os.environ.get("DOCENT_TERMINAL_MCP_SOCKET_PATH"):
        return Path(path)
    run = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(run) / "docent-terminal-mcp" / "server.sock"


def bind_unix_socket(path: Path):
    """Bind the server's Unix socket (directory 0700, socket 0600).  A
    socket file that still answers belongs to another server: refuse; one
    that does not is stale and is replaced."""
    import socket as so

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.exists():
        probe = so.socket(so.AF_UNIX, so.SOCK_STREAM)
        try:
            probe.connect(str(path))
        except OSError:
            path.unlink()  # nothing listens: left by a server that did not exit cleanly
        else:
            sys.exit(f"docent-terminal-mcp: {path} is in use by another server.  Connect to it "
                     f"(`docent-terminal-mcp connect`) instead of starting another.")
        finally:
            probe.close()
    sock = so.socket(so.AF_UNIX, so.SOCK_STREAM)
    old = os.umask(0o077)
    try:
        sock.bind(str(path))
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    sock.listen(128)
    return sock


def front(app, manager: Manager, socket_path: Path | None, token: str | None):
    """ASGI front of the shared server.  /health answers anyone (no token;
    it says only what and whose this is).  Everything else -- MCP at /mcp,
    the session listing at /api/sessions -- is open on the Unix socket,
    which only its user can reach, and needs `Authorization: Bearer` over
    TCP, where any local user can reach the port."""
    import hmac

    expected = f"Bearer {token}".encode() if token else None
    socket_name = str(socket_path) if socket_path else None

    async def read_body(receive) -> bytes:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                return body

    async def reply(send, status, body, ctype=b"application/json"):
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", ctype)]})
        await send({"type": "http.response.body", "body": body})

    async def front(scope, receive, send):
        if scope["type"] == "lifespan":
            manager.start_housekeeping()  # the sweep at start-up, then hourly
        if scope["type"] != "http":
            return await app(scope, receive, send)
        path = scope.get("path", "")
        if path == "/health":
            return await reply(send, 200, json.dumps(identity()).encode())
        # uvicorn reports a Unix socket's local address as (path, None).
        server = scope.get("server")
        over_socket = socket_name is not None and server is not None and tuple(server) == (socket_name, None)
        if not over_socket:
            given = dict(scope.get("headers") or []).get(b"authorization", b"")
            if expected is None or not hmac.compare_digest(given, expected):
                return await reply(send, 401, b"docent-terminal-mcp: missing or wrong token\n", b"text/plain")
        else:
            # The app's DNS-rebinding guard wants a localhost Host header; on
            # the socket the header means nothing (the path says whose server
            # this is), so clients may use any URL.
            headers = [(k, v) for k, v in scope.get("headers") or [] if k != b"host"]
            scope = {**scope, "headers": headers + [(b"host", b"localhost:0")]}
        if path == "/api/sessions":
            body = {**identity(), "sessions": await manager.describe_sessions()}
            return await reply(send, 200, json.dumps(body).encode())
        if scope.get("method") == "POST" and (m := re.fullmatch(r"/api/sessions/([^/]+)/logs", path)):
            # {"action": "keep" | "default" | "delete_on_close" | "delete"}: as
            # the session_logs tool (a client's own UI, `NAME logs`).
            try:
                action = json.loads(await read_body(receive) or b"{}").get("action")
                answer = await manager.logs_action(m.group(1), action)
            except (ToolError, ValueError, AttributeError) as e:
                return await reply(send, 400, json.dumps({"error": str(e)}).encode())
            return await reply(send, 200, json.dumps({"ok": True, "message": answer}).encode())
        if scope.get("method") == "POST" and (m := re.fullmatch(r"/api/sessions/([^/]+)/close", path)):
            # {"window": bool, "force": bool}: a close a person asked for
            # (a client's Close button, `NAME close`).  Only here can a session a
            # person is attached to, or another's busy one, be closed: the
            # session_close tool has no force.
            try:
                args = json.loads(await read_body(receive) or b"{}")
                token = ACTOR.set("human")
                try:
                    answer = await manager.close_sessions(m.group(1), bool(args.get("window")), bool(args.get("force")))
                finally:
                    ACTOR.reset(token)
            except (ToolError, TmuxError, ValueError, AttributeError) as e:
                return await reply(send, 400, json.dumps({"error": str(e)}).encode())
            return await reply(send, 200, json.dumps({"ok": True, "message": answer}).encode())
        if scope.get("method") == "POST" and path == "/api/sweep":
            # The sweep now; ?dry_run=1 names what it would delete.
            dry = b"dry_run=1" in scope.get("query_string", b"")
            gone = await manager.sweep(dry_run=dry)
            body = {"dry_run": dry, "keep_closed_days": manager.keep_days,
                    "deleted" if not dry else "due": [{"session": n, "why": w} for n, w in gone]}
            return await reply(send, 200, json.dumps(body).encode())
        await app(scope, receive, send)

    return front


def serve(socket_path: Path, tcp: tuple[str, int] | None) -> None:
    """The shared server: one process for every agent, on the user's Unix
    socket (no token) and, with `tcp`, on a loopback port with the token."""
    import socket as so

    import uvicorn

    home = default_home()
    lock = hold_lock(home)  # noqa: F841 -- held while serving
    mcp = build_server(home, shared=True)
    sockets = [bind_unix_socket(socket_path)]
    token = None
    where = [f"unix socket {socket_path}"]
    if tcp is not None:
        host, port = tcp
        token = read_token(home)
        sockets.append(so.create_server((host, port)))
        where.append(f"http://{host}:{port}/mcp (token in {home / 'token'})")
    app = front(mcp.streamable_http_app(host=tcp[0] if tcp else "localhost"), mcp.terminal_manager,
                socket_path, token)
    print(f"docent-terminal-mcp: serving on {' and '.join(where)}", file=sys.stderr)
    try:
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=sockets)
    finally:
        socket_path.unlink(missing_ok=True)


def connect(socket_path: Path, actor: str | None) -> None:
    """A stdio bridge to the shared server's Unix socket, for clients that
    start an MCP server as a command (Claude Code, Hermes): no port, no
    token.  Relays messages both ways until the client closes stdin or the
    server goes away (the client then reconnects by starting a new one)."""
    import anyio
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.stdio import stdio_server

    if not socket_path.exists():
        sys.exit(f"docent-terminal-mcp: no server socket at {socket_path}: is the service running "
                 f"(`systemctl --user status docent-terminal-mcp`)?")
    headers = {"X-Terminal-Actor": actor} if actor else {}

    from mcp.shared.message import SessionMessage
    from mcp_types import JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse

    # Errors the client transport makes up when the server is gone or restarted
    # (its session ids are new): answer them by reconnecting, not by relaying.
    GONE = {"Session not found", "Session terminated", "SSE stream ended without a response"}
    REINIT_ID = "docent-terminal-mcp-connect-reinit"

    def http_client():
        return httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=str(socket_path)),
                                  headers=headers, timeout=httpx2.Timeout(30.0, read=300.0))

    # After a restart, a call that was in flight is sent again -- if that is
    # safe.  One that types (run, expect) must not type again: it becomes a
    # `wait` on its session, which reports the command it had started.  One
    # that only acts (send, shell_integrate, session_create/close) cannot
    # be repeated or waited for: the client is told to look.
    TYPING_WAITED = {"run", "expect"}
    NOT_REPEATABLE = {"send", "shell_integrate", "session_create", "session_split", "session_close", "session_resize"}
    # (Whether the old server typed it before going is unknown: typed at most once.)
    RESTARTED = ("[the server restarted during this call; its input was not typed again, and may not have "
                 "been typed at all -- what follows is the `wait` tool's reply for the session: if no command of yours shows "
                 "there, send it again]")
    def restarted_note(original, result: dict) -> str:
        """RESTARTED, or -- when the `wait` reply shows a region labelled
        with the call's input -- that the command did reach the shell."""
        try:
            args = original.message.params.get("arguments") or {}
            first = (args.get("input") or "").partition(chr(10))[0].strip()
            text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
            m = re.search(r"\[region (\d+) · [^\n]*?" + re.escape(first[:60]), text) if first else None
        except (AttributeError, IndexError, TypeError):
            m = None
        if m is None:
            return RESTARTED
        return (f"[the server restarted during this call; your input did reach the shell before it (region "
                f"{m.group(1)}) and was not typed again -- what follows is the `wait` tool's reply for the session]")

    NOT_REPEATED = ("[the server restarted during this call, which was not repeated (it may have acted "
                    "already): the `read` or `regions` tool shows what happened]")

    def with_note(result: dict, note: str) -> dict:
        """A tool result with `note` put first, in the text and in the
        structured result (which some clients show instead)."""
        texts = [c for c in result.get("content", []) if c.get("type") == "text"]
        joined = "\n".join([note] + [c["text"] for c in texts])
        out = {**result, "content": [{"type": "text", "text": joined}]}
        if isinstance(result.get("structuredContent"), dict) and "result" in result["structuredContent"]:
            out["structuredContent"] = {**result["structuredContent"], "result": joined}
        return out

    def replay(message):
        """The message to send again after a restart, or None."""
        msg = message.message
        if not (isinstance(msg, JSONRPCRequest) and msg.method == "tools/call"):
            return message
        name = (msg.params or {}).get("name")
        args = (msg.params or {}).get("arguments") or {}
        if name in NOT_REPEATABLE:
            return None
        if name in TYPING_WAITED:
            # What is left of the call's own timeout: a 30-minute build's call
            # must not turn into a wait for the whole build.
            elapsed = time.monotonic() - sent_at.get(msg.id, time.monotonic())
            left = max(1.0, float(args.get("timeout_s", 60)) - elapsed)
            waited = {"session": args.get("session"), "timeout_s": round(left, 1),
                      "max_lines": args.get("max_lines", 80)}
            return SessionMessage(JSONRPCRequest(jsonrpc="2.0", id=msg.id, method="tools/call",
                                                 params={**msg.params, "name": "wait", "arguments": waited}))
        return message

    DOWN = "down"
    sent_at: dict = {}  # request id -> when it was sent (monotonic)

    async def server_instance(http) -> str | None:
        """The server process answering on the socket now: its instance id,
        DOWN when nothing accepts connections there, None when it cannot be
        told (an older server)."""
        try:
            return (await http.get("http://docent-terminal-mcp/health")).json().get("instance")
        except (httpx2.ConnectError, FileNotFoundError, ConnectionRefusedError):
            return DOWN
        except Exception:  # noqa: BLE001
            return None

    async def bridge():
        init_request: JSONRPCRequest | None = None  # the client's initialize, replayed after a restart
        pending: dict = {}  # id -> SessionMessage of requests not yet answered
        rewritten: set = set()  # ids re-sent as `wait`: their answer says so
        held: list = []  # calls not yet sent when a restart was seen: sent afresh once reconnected
        closed = False  # the client closed its side: done

        async def reinitialize(up_read, up_write):
            """A restarted server knows no session: initialize one for the
            client (with its own parameters) and re-send what it is waiting on."""
            await up_write.send(SessionMessage(JSONRPCRequest(
                jsonrpc="2.0", id=REINIT_ID, method="initialize", params=init_request.params)))
            async for message in up_read:
                if isinstance(message, Exception):
                    raise message
                if isinstance(message.message, (JSONRPCResponse, JSONRPCError)) and message.message.id == REINIT_ID:
                    break
            await up_write.send(SessionMessage(JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")))
            for mid, message in list(pending.items()):
                resend = replay(message)
                if resend is None:
                    # Not safe to repeat: answer it here instead.
                    del pending[mid]
                    await down_write.send(SessionMessage(JSONRPCResponse(jsonrpc="2.0", id=mid, result={
                        "content": [{"type": "text", "text": NOT_REPEATED}],
                        "structuredContent": {"result": NOT_REPEATED}, "isError": False})))
                    continue
                if resend is not message:
                    rewritten.add(mid)
                await up_write.send(resend)

        async with stdio_server() as (down_read, down_write):
            attempts = refused = 0
            while not closed:
                if not socket_path.exists():
                    if attempts >= 60:
                        sys.exit(f"docent-terminal-mcp connect: no server at {socket_path} for 30 s; giving up")
                    attempts += 1
                    await anyio.sleep(0.5)
                    continue
                attempts = 0
                reconnect = False
                try:
                    async with http_client() as http, \
                            streamable_http_client("http://docent-terminal-mcp/mcp", http_client=http) as up:
                        up_read, up_write = up[0], up[1]
                        instance = await server_instance(http)
                        if init_request is not None:
                            await reinitialize(up_read, up_write)
                        for message in held:  # never sent before: nothing to be careful about
                            pending[message.message.id] = message
                            await up_write.send(message)
                        held.clear()
                        async with anyio.create_task_group() as tg:
                            async def client_to_server():
                                nonlocal init_request, closed, reconnect
                                try:
                                    async for message in down_read:
                                        if isinstance(message, Exception):
                                            break
                                        msg = message.message
                                        if isinstance(msg, JSONRPCRequest):
                                            if msg.method == "initialize":
                                                init_request = msg
                                            elif msg.method == "tools/call" and init_request is not None \
                                                    and await server_instance(http) not in (instance, None):
                                                # Restarted (or restarting) while the client was
                                                # idle: this call reached no server -- send it to
                                                # the new one as it is, not as a call caught in
                                                # the restart.
                                                held.append(message)
                                                reconnect = True
                                                return
                                            pending[msg.id] = message
                                            sent_at[msg.id] = time.monotonic()
                                        await up_write.send(message)
                                    closed = True
                                finally:
                                    tg.cancel_scope.cancel()

                            async def server_to_client():
                                nonlocal reconnect
                                try:
                                    async for message in up_read:
                                        if isinstance(message, Exception):
                                            print(f"docent-terminal-mcp connect: server: {message}", file=sys.stderr)
                                            reconnect = True
                                            break
                                        msg = message.message
                                        if isinstance(msg, JSONRPCError) and msg.error.message in GONE \
                                                and msg.id in pending:
                                            reconnect = True  # answer it after reconnecting
                                            break
                                        if isinstance(msg, (JSONRPCResponse, JSONRPCError)):
                                            original = pending.pop(msg.id, None)
                                            sent_at.pop(msg.id, None)
                                            if isinstance(msg, JSONRPCResponse) and msg.id in rewritten:
                                                rewritten.discard(msg.id)
                                                message = SessionMessage(JSONRPCResponse(
                                                    jsonrpc="2.0", id=msg.id,
                                                    result=with_note(msg.result, restarted_note(original, msg.result))))
                                        await down_write.send(message)
                                    else:
                                        reconnect = True  # the server closed the stream
                                finally:
                                    tg.cancel_scope.cancel()

                            tg.start_soon(client_to_server)
                            tg.start_soon(server_to_client)
                except Exception as e:  # noqa: BLE001 -- the server went away mid-call
                    if closed:
                        break
                    # A socket file nobody answers on (a server that failed to
                    # start): give up like a missing one, rather than forever.
                    refused = refused + 1 if "refused" in repr(e).lower() else 0
                    if refused >= 60:
                        sys.exit(f"docent-terminal-mcp connect: {socket_path} refuses connections for 30 s; giving up")
                    print(f"docent-terminal-mcp connect: {type(e).__name__}: {e}; reconnecting", file=sys.stderr)
                    reconnect = True
                if reconnect and not closed:
                    if init_request is None:
                        sys.exit("docent-terminal-mcp connect: the server went away before the client initialized")
                    print("docent-terminal-mcp connect: the server restarted; reconnecting", file=sys.stderr)
                    await anyio.sleep(0.5)
            # The client closed stdin: close our side too, or stdio_server
            # waits for its stdout writer forever (a bridge left running).
            await down_write.aclose()

    anyio.run(bridge)


def wrapper_nonce(tool: str) -> str | None:
    """The session's nonce, for `ssh`/`into`/`su` to carry to the shell at
    the far end.  A pane whose shell an older server started (before the
    rename of 2026-10-12) has it as __TMCP_NONCE -- adjutant's pane, 19
    days old, sent none, and the hop ran without exit statuses.  None:
    said on stderr, since nothing else would say why."""
    nonce = os.environ.get("__DTMCP_NONCE") or os.environ.get("__TMCP_NONCE")
    if not nonce:
        print(f"{NAME} {tool}: no session nonce in the environment (__DTMCP_NONCE): this shell was not started "
              f"by the terminal server, so the far shell's exit statuses will not be tracked", file=sys.stderr)
    return nonce


def ssh_main(args: list[str]) -> None:
    """`NAME ssh [--print] [ssh options] [user@]host` (see remote.py)."""
    from .remote import bootstrap, carried, ssh_argv
    show = args[:1] == ["--print"]
    nonce = wrapper_nonce("ssh")
    carry = carried(dict(os.environ))
    try:
        argv = ssh_argv(args[1:] if show else args, nonce, os.environ.get("DOCENT_TERMINAL_MCP_SSH", "ssh"), carry)
    except ValueError as e:
        sys.exit(str(e))
    if show:
        print(shlex.join(argv[:-1]) + " 'exec sh -c <the script below>'\n")
        print(bootstrap(nonce, readable=True, carry=carry), end="")
        return
    os.execvp(argv[0], argv)


def into_main(args: list[str]) -> None:
    """`NAME into [--print] COMMAND...` (see remote.into_argv)."""
    from .remote import bootstrap, carried, into_argv
    show = args[:1] == ["--print"]
    nonce = wrapper_nonce("into")
    carry = carried(dict(os.environ))
    try:
        argv = into_argv(args[1:] if show else args, nonce, carry)
    except ValueError as e:
        sys.exit(str(e))
    if show:
        print(shlex.join(argv[:-1]) + " '<the script below>'\n")
        print(bootstrap(nonce, readable=True, carry=carry), end="")
        return
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        sys.exit(f"{NAME} into: {argv[0]}: {e.strerror}")


def su_main(args: list[str]) -> None:
    """`NAME su [--print] [su options] [-] [user]` (see remote.su_argv)."""
    from .remote import bootstrap, carried, su_argv
    show = args[:1] == ["--print"]
    nonce = wrapper_nonce("su")
    carry = carried(dict(os.environ))
    try:
        argv = su_argv(args[1:] if show else args, nonce, carry)
    except ValueError as e:
        sys.exit(str(e))
    if show:
        print(shlex.join(argv[:2]) + " 'exec sh -c <the script below>' " + shlex.join(argv[3:]) + "\n")
        print(bootstrap(nonce, readable=True, carry=carry), end="")
        return
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        sys.exit(f"{NAME} su: {argv[0]}: {e.strerror}")


def install_main(args: list[str]) -> None:
    """`NAME install [--print] [--remove] [ssh options] [user@]host`: the
    integration files into the host's ~/.local/share (see remote.py)."""
    import subprocess
    from .remote import INSTALL_DIR, INSTALL_PARENT, INSTALLED, SHELL_DIR, install_argv, install_script, install_tar
    flags = set()
    while args and args[0] in ("--print", "--remove"):
        flags.add(args.pop(0))
    remove = "--remove" in flags
    try:
        argv = install_argv(args, remove, os.environ.get("DOCENT_TERMINAL_MCP_SSH", "ssh"))
    except ValueError as e:
        sys.exit(str(e))
    if "--print" in flags:
        print(shlex.join(argv[:-1]) + " 'exec sh -c <the script below>'"
              + ("" if remove else "  < a tar archive of the files below") + "\n")
        print(install_script(remove) + ("" if remove else "\n"), end="")
        for name in () if remove else INSTALLED:
            print(f"{SHELL_DIR / name}  ->  ~/{INSTALL_DIR}/{name}")
        return
    if remove:
        print(f"{NAME}: removing ~/{INSTALL_PARENT}/v* on {argv[-2]}", flush=True)
    else:
        print(f"{NAME}: copying {', '.join(INSTALLED)} to ~/{INSTALL_DIR} on {argv[-2]}", flush=True)
    sys.exit(subprocess.run(argv, input=None if remove else install_tar()).returncode)


def post_main(socket_path: Path, path: str, body: dict) -> dict:
    """POST to the shared server over its Unix socket (no token); prints
    the answer's message, or exits with its error."""
    import http.client
    import socket as so

    class UnixConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = so.socket(so.AF_UNIX, so.SOCK_STREAM)
            self.sock.connect(str(socket_path))

    try:
        conn = UnixConnection("localhost")
        conn.request("POST", path, json.dumps(body).encode() if body else b"", {"Content-Type": "application/json"})
        answer = json.loads(conn.getresponse().read() or b"{}")
    except OSError as e:
        sys.exit(f"{NAME}: no shared server answers on {socket_path} ({e})")
    if "error" in answer:
        sys.exit(f"{NAME}: {answer['error']}")
    if "message" in answer:
        print(answer["message"])
    return answer


def logs_main(socket_path: Path, action: str, session: str | None) -> None:
    """`NAME logs ...`: the shared server's POST /api/... over its Unix
    socket (no token), so that its own checks -- is the session closed --
    and its log of deletions apply."""
    import http.client
    import socket as so

    class UnixConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = so.socket(so.AF_UNIX, so.SOCK_STREAM)
            self.sock.connect(str(socket_path))

    if action in ("due", "sweep"):
        path, body = "/api/sweep" + ("?dry_run=1" if action == "due" else ""), b""
    else:
        if not session:
            sys.exit(f"{NAME} logs {action}: which session?")
        path = f"/api/sessions/{session}/logs"
        body = json.dumps({"action": action.replace("-", "_")}).encode()
    try:
        conn = UnixConnection("localhost")
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        answer = json.loads(conn.getresponse().read() or b"{}")
    except OSError as e:
        sys.exit(f"{NAME} logs: no shared server answers on {socket_path} ({e})")
    if "error" in answer:
        sys.exit(f"{NAME} logs: {answer['error']}")
    if "message" in answer:
        print(answer["message"])
        return
    listed = answer.get("due", answer.get("deleted", []))
    days = answer.get("keep_closed_days")
    print(f"{len(listed)} session{'s' if len(listed) != 1 else ''} "
          + ("due for deletion" if action == "due" else "deleted")
          + (f" (closed logs are kept {days:g} days)" if days else " (this server keeps closed sessions)") + ":")
    for item in listed:
        print(f"  {item['session']}\t{item['why']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog=NAME, description=(
        "Interactive terminal sessions over MCP.  With no command, serves one client over stdio."))
    sub = parser.add_subparsers(dest="cmd")
    p_serve = sub.add_parser("serve", help=(
        "serve every agent from one process: on the user's Unix socket, and with --tcp on a port"))
    p_serve.add_argument("--socket", type=Path, default=None, metavar="PATH", help=(
        "the Unix socket (default: $DOCENT_TERMINAL_MCP_SOCKET_PATH, else $XDG_RUNTIME_DIR/docent-terminal-mcp/server.sock)"))
    p_serve.add_argument("--tcp", type=int, nargs="?", const=DEFAULT_PORT, default=None, metavar="PORT", help=(
        f"also listen on a loopback port (default {DEFAULT_PORT}), with the bearer token, "
        "for clients that can only be given a URL"))
    p_serve.add_argument("--host", default="127.0.0.1", help="the address for --tcp")
    p_serve.add_argument("--keep-closed-days", default=None, metavar="DAYS", help=(
        "delete a closed session's logs once nothing in them changed for DAYS days (default: "
        "$DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS, else 3); 0 keeps them forever"))
    p_logs = sub.add_parser("logs", help=(
        f"keep or delete sessions' logs on the running shared server: {NAME} logs keep|default|"
        f"delete-on-close|delete SESSION, {NAME} logs due (what the next sweep would delete), "
        f"{NAME} logs sweep (run it now)"))
    p_logs.add_argument("action", choices=["keep", "default", "delete-on-close", "delete", "due", "sweep"])
    p_logs.add_argument("session", nargs="?", help="the session's name (NAME.N for an earlier one of that name)")
    p_logs.add_argument("--socket", type=Path, default=None, metavar="PATH")
    p_close = sub.add_parser("close", help=(
        f"close a session on the running shared server, as a person: {NAME} close [--window] [--force] SESSION "
        "(--force: even with someone attached, or another's command running)"))
    p_close.add_argument("session")
    p_close.add_argument("--window", action="store_true", help="every pane of its window")
    p_close.add_argument("--force", action="store_true", help="even if someone is using it")
    p_close.add_argument("--socket", type=Path, default=None, metavar="PATH")
    p_connect = sub.add_parser("connect", help=(
        "stdio bridge to the shared server's Unix socket (no port, no token): "
        "claude mcp add terminal -- docent-terminal-mcp connect --actor claude-code"))
    p_connect.add_argument("--socket", type=Path, default=None, metavar="PATH")
    p_connect.add_argument("--actor", default=None, metavar="NAME", help="the agent's name (X-Terminal-Actor)")
    sub.add_parser("token", help="print the shared server's bearer token (for --tcp clients)")
    p_skill = sub.add_parser("skill", help=(
        "write the Claude Code skill for these tools (the guide, in the form Claude Code loads on "
        "demand) into a skills directory: docent-terminal-mcp skill ~/.claude/skills"))
    p_skill.add_argument("dir", type=Path, help="~/.claude/skills, a project's .claude/skills, or a plugin directory")
    p_skill.add_argument("--plugin", action="store_true", help=(
        "write a Claude Code plugin there instead: .claude-plugin/plugin.json, an .mcp.json that "
        "starts this server, and the skill under skills/"))
    p_ssh = sub.add_parser("ssh", help=(
        f"ssh to a host with the shell integration carried along for this connection (nothing is "
        f"installed there; plain ssh is unaffected): {NAME} ssh [ssh options] [user@]host"))
    p_ssh.add_argument("--print", action="store_true", help="print the ssh command line and its script, do not run it")
    p_ssh.add_argument("ssh_args", nargs=argparse.REMAINDER, help="ssh's options and the destination")
    p_install = sub.add_parser("install", help=(
        f"copy the shell integration files to a host (~/.local/share/{NAME}), so that shell_integrate "
        f"can read them in a shell reached by plain ssh or sudo -i there: "
        f"{NAME} install [--print] [--remove] [ssh options] [user@]host"))
    p_install.add_argument("--print", action="store_true", help="print the ssh command, its script and the files; do not run it")
    p_install.add_argument("--remove", action="store_true", help="remove the files from the host instead")
    p_install.add_argument("ssh_args", nargs=argparse.REMAINDER, help="ssh's options and the destination")
    if sys.argv[1:2] == ["ssh"]:
        return ssh_main(sys.argv[2:])  # ssh's own options pass through untouched
    p_into = sub.add_parser("into", help=(
        f"run a command that opens a shell somewhere else (a container, a sandbox) with the shell "
        f"integration carried along, as `{NAME} ssh` does over ssh: "
        f"{NAME} into podman exec -it box    {NAME} into openshell sandbox exec -n box --tty --"))
    p_into.add_argument("--print", action="store_true", help="print the command line and its script, do not run it")
    p_into.add_argument("command", nargs=argparse.REMAINDER, help="the command, up to where the program to run would go")
    p_su = sub.add_parser("su", help=(
        f"su to another user with the shell integration carried along, as `{NAME} ssh` does over ssh: "
        f"{NAME} su [su options] [-] [user]"))
    p_su.add_argument("--print", action="store_true", help="print the su command line and its script, do not run it")
    p_su.add_argument("su_args", nargs=argparse.REMAINDER, help="su's options and the user")
    if sys.argv[1:2] == ["su"]:
        return su_main(sys.argv[2:])
    if sys.argv[1:2] == ["install"]:
        return install_main(sys.argv[2:])
    if sys.argv[1:2] == ["into"]:
        return into_main(sys.argv[2:])
    args = parser.parse_args()
    if args.cmd == "logs":
        return logs_main(args.socket or default_socket_path(), args.action, args.session)
    if args.cmd == "close":
        return post_main(args.socket or default_socket_path(), f"/api/sessions/{args.session}/close",
                         {"window": args.window, "force": args.force})
    if args.cmd == "serve":
        if args.keep_closed_days is not None:
            os.environ["DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS"] = args.keep_closed_days
        serve(args.socket or default_socket_path(), (args.host, args.tcp) if args.tcp else None)
    elif args.cmd == "connect":
        connect(args.socket or default_socket_path(), args.actor)
    elif args.cmd == "token":
        print(read_token(default_home()))
    elif args.cmd == "skill":
        from .skill import write_skill
        print(write_skill(args.dir, args.plugin))
    elif args.cmd == "ssh":
        ssh_main(args.ssh_args)
    else:
        lock = hold_lock(default_home())  # noqa: F841
        build_server().run("stdio")


if __name__ == "__main__":
    main()
