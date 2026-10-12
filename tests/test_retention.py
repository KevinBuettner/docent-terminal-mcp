"""Retention of closed sessions' logs: the sweep, the pin, delete now and
delete on close, and what keeps a session's logs."""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from mcp import Client

from docent_terminal_mcp import retention
from docent_terminal_mcp.server import build_server, front

DAY = 86400


@pytest.fixture
def env(tmp_path):
    socket = f"dtmcp-ret-{os.getpid()}-{tmp_path.name}"
    yield tmp_path, socket
    subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
    tmpdir = os.environ.get("TMUX_TMPDIR", "/tmp")
    (Path(tmpdir) / f"tmux-{os.getuid()}" / socket).unlink(missing_ok=True)


def plain_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".bash_profile").write_text("PS1='$ '\n")
    return str(home)


def age(directory: Path, days: float) -> None:
    """Make everything in a session's directory `days` old."""
    t = time.time() - days * DAY
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            os.utime(os.path.join(root, name), (t, t))
    os.utime(directory, (t, t))


class Server:
    """One server for a whole test (its settings changed in place)."""

    def __init__(self, home, socket):
        self.mcp = build_server(home, socket)
        self.mgr = self.mcp.terminal_manager
        self.dir = home / "sessions"

    async def __aenter__(self):
        self.client = Client(self.mcp)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.client.__aexit__(*exc)

    async def __call__(self, tool, /, **args):
        r = await self.client.call_tool(tool, args)
        text = "\n".join(x.text for x in r.content)
        if r.is_error:
            raise AssertionError(text)
        return text

    async def closed(self, name, home):
        """A session made and closed, its logs then aged 4 days."""
        await self("session_create", name=name, cwd=str(home), env={"HOME": plain_home(home)})
        await self("run", session=name, input=f"echo {name}-output")
        await self("session_close", session=name)
        age(self.dir / name, 4)


async def test_the_sweep_deletes_only_what_is_due(env):
    home, socket = env
    async with Server(home, socket) as t:
        t.mgr.keep_days = 3
        for name in ("old", "pinned", "recent", "other-host"):
            await t.closed(name, home)
        age(t.dir / "recent", 1)
        retention.set_mark(t.dir / "pinned", retention.KEEP_MARK, True)
        age(t.dir / "pinned", 4)
        meta = json.loads((t.dir / "other-host" / "meta.json").read_text())
        (t.dir / "other-host" / "meta.json").write_text(json.dumps({**meta, "host": "elsewhere"}))
        age(t.dir / "other-host", 4)
        await t("session_create", name="live", cwd=str(home), env={"HOME": plain_home(home)})
        age(t.dir / "live", 4)  # an idle live session: never

        due = await t.mgr.sweep(dry_run=True)
        assert [n for n, _ in due] == ["old"], due
        assert (t.dir / "old").exists()  # a dry run deletes nothing
        gone = await t.mgr.sweep()
        assert gone == [("old", "closed and unused -- neither written nor read -- for 4.0 days")], gone
        assert not (t.dir / "old").exists()
        for kept in ("pinned", "recent", "other-host", "live"):
            assert (t.dir / kept).exists(), kept


async def test_without_a_tmux_server_only_sessions_of_our_socket_are_closed(env):
    # No tmux server answering on our socket: the last session closed (no
    # systemd scope keeps a server), or a reboot -- every session made on
    # this socket path is closed.  One made under another TMUX_TMPDIR or
    # /tmp may live in a server we cannot see: kept, as are sessions from
    # before the path was recorded.
    home, socket = env
    async with Server(home, socket) as t:
        t.mgr.keep_days = 3
        for name in ("ours", "elsewhere", "older"):
            await t.closed(name, home)
        for name, path in (("elsewhere", "/some/other/tmp/tmux-1/x"), ("older", None)):
            meta = json.loads((t.dir / name / "meta.json").read_text())
            meta.pop("tmux_socket_path")
            if path:
                meta["tmux_socket_path"] = path
            (t.dir / name / "meta.json").write_text(json.dumps(meta))
            age(t.dir / name, 4)
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
        assert [n for n, _ in await t.mgr.sweep()] == ["ours"]
        assert (t.dir / "elsewhere").exists() and (t.dir / "older").exists()


async def test_zero_days_keeps_everything_and_archives_follow_the_rule(env):
    home, socket = env
    async with Server(home, socket) as t:
        await t.closed("x", home)
        await t("session_create", name="x", cwd=str(home), env={"HOME": plain_home(home)})  # x -> x.1
        age(t.dir / "x.1", 4)
        t.mgr.keep_days = 0
        assert await t.mgr.sweep() == []
        t.mgr.keep_days = 3
        assert [n for n, _ in await t.mgr.sweep()] == ["x.1"]
        assert (t.dir / "x").exists()


async def test_delete_on_close(env):
    # (Also the race the first version had: the first call starts the
    # housekeeping, whose sweep of marked sessions ran while session_create
    # awaited tmux -- and deleted the session being made.)
    home, socket = env
    async with Server(home, socket) as t:
        out = await t("session_create", name="secret", cwd=str(home), env={"HOME": plain_home(home)},
                      delete_logs_on_close=True)
        assert "(deleted when the session closes)" in out, out
        await t("run", session="secret", input="echo shown-over-ssh")
        out = await t("session_list")
        assert "logs deleted when it closes" in out, out
        out = await t("session_close", session="secret")
        assert "its logs were deleted (it was marked delete-on-close)" in out, out
        assert not (t.dir / "secret").exists()
        # Marked later, and closed by something else (its pane killed): the
        # next sweep of marked sessions takes it, however recent.
        await t("session_create", name="later", cwd=str(home), env={"HOME": plain_home(home)})
        out = await t("session_logs", session="later", action="delete_on_close")
        assert "will be deleted when it closes" in out, out
        pane = json.loads((t.dir / "later" / "meta.json").read_text())["pane_id"]
        subprocess.run(["tmux", "-L", socket, "kill-pane", "-t", pane], check=True)
        assert [n for n, _ in await t.mgr.sweep(marked_only=True)] == ["later"]


async def test_a_reused_name_marked_delete_on_close_is_not_archived(env):
    home, socket = env
    async with Server(home, socket) as t:
        await t("session_create", name="n", cwd=str(home), env={"HOME": plain_home(home)})
        retention.set_mark(t.dir / "n", retention.ON_CLOSE_MARK, True)
        pane = json.loads((t.dir / "n" / "meta.json").read_text())["pane_id"]
        subprocess.run(["tmux", "-L", socket, "kill-pane", "-t", pane], check=True)
        out = await t("session_create", name="n", cwd=str(home), env={"HOME": plain_home(home)})
        assert "moved to" not in out and not (t.dir / "n.1").exists(), out


async def test_session_logs_keep_default_delete(env):
    home, socket = env
    async with Server(home, socket) as t:
        await t("session_create", name="s", cwd=str(home), env={"HOME": plain_home(home)})
        with pytest.raises(AssertionError, match="is live: its logs are still being written"):
            await t("session_logs", session="s", action="delete")
        await t("session_close", session="s")
        out = await t("session_list")
        assert "logs deleted after " in out, out
        out = await t("session_logs", session="s", action="keep")
        assert "are kept" in out and (t.dir / "s" / "keep").exists(), out
        assert "logs kept" in await t("session_list")
        out = await t("session_logs", session="s", action="default")
        assert "usual rule again" in out and not (t.dir / "s" / "keep").exists(), out
        out = await t("session_logs", session="s", action="delete")
        assert "were deleted" in out and not (t.dir / "s").exists(), out
        with pytest.raises(AssertionError, match="no logs of a session named 's'"):
            await t("session_logs", session="s", action="keep")


async def test_another_hosts_session_cannot_be_deleted_here(env):
    home, socket = env
    async with Server(home, socket) as t:
        await t.closed("theirs", home)
        meta = json.loads((t.dir / "theirs" / "meta.json").read_text())
        (t.dir / "theirs" / "meta.json").write_text(json.dumps({**meta, "host": "elsewhere"}))
        with pytest.raises(AssertionError, match="made by a server on elsewhere"):
            await t("session_logs", session="theirs", action="delete")
        assert (t.dir / "theirs").exists()


def test_made_here_for_sessions_from_before_the_host_was_recorded(tmp_path):
    old = {"name": "x"}  # no host, no tmux_socket
    assert not retention.made_here(old, tmp_path / "docent-terminal-mcp", "docent-terminal-mcp")
    short = retention.HOST.split(".")[0]
    assert retention.made_here(old, tmp_path / f"docent-terminal-mcp-{short}", "docent-terminal-mcp")
    assert not retention.made_here(old, tmp_path / f"docent-terminal-mcp-{short}", "another-socket")
    assert retention.made_here({"host": retention.HOST, "tmux_socket": "s"}, tmp_path, "s")


def test_keep_days_setting(monkeypatch):
    monkeypatch.delenv("DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS", raising=False)
    assert retention.keep_days() == 3
    monkeypatch.setenv("DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS", "0")
    assert retention.keep_days() == 0
    assert retention.keep_days("7.5") == 7.5
    assert retention.keep_days("soon") == 3


async def test_the_http_endpoints(env):
    home, socket = env
    async with Server(home, socket) as t:
        t.mgr.keep_days = 3
        await t.closed("old", home)
        app = front(None, t.mgr, None, None)  # (no socket: over TCP, so a token is wanted)
        sent = []

        async def post(path, body=b"", query=b"", token=True):
            sent.clear()
            messages = iter([{"type": "http.request", "body": body, "more_body": False}])
            headers = [(b"authorization", b"Bearer t")] if token else []
            await front(None, t.mgr, None, "t")(
                {"type": "http", "method": "POST", "path": path, "query_string": query, "headers": headers},
                lambda: _next(messages), lambda m: _append(sent, m))
            return sent[0]["status"], json.loads(sent[1]["body"])

        assert app is not None
        status, body = await post("/api/sweep", query=b"dry_run=1")
        assert status == 200 and body["due"] == [
            {"session": "old", "why": "closed and unused -- neither written nor read -- for 4.0 days"}], body
        status, body = await post("/api/sessions/old/logs", json.dumps({"action": "keep"}).encode())
        assert status == 200 and "are kept" in body["message"], body
        status, body = await post("/api/sessions/old/logs", json.dumps({"action": "explode"}).encode())
        assert status == 400 and "action must be one of" in body["error"], body
        status, body = await post("/api/sweep")
        assert status == 200 and body["deleted"] == [], body  # pinned


async def _next(it):
    return next(it)


async def _append(sent, message):
    sent.append(message)


async def test_a_closed_sessions_logs_are_read_and_reading_counts_as_use(env):
    # No references -- the dates decide, and a read is a use.  (The
    # read/search/regions tools refused a closed session before.)
    home, socket = env
    async with Server(home, socket) as t:
        t.mgr.keep_days = 3
        await t.closed("past", home)
        out = await t("read", session="past")
        assert "past-output" in out, out
        out = await t("search", session="past", pattern="past-output")
        assert "matching line" in out, out
        out = await t("regions", session="past")
        assert "echo past-output" in out, out
        with pytest.raises(AssertionError, match="is closed: it has no screen"):
            await t("read", session="past", view="screen")
        with pytest.raises(AssertionError, match="nothing can be typed there"):
            await t("run", session="past", input="echo no")
        assert (t.dir / "past" / "last-used").exists()
        assert await t.mgr.sweep() == []  # just read: in use
        age(t.dir / "past", 4)
        assert [n for n, _ in await t.mgr.sweep()] == ["past"]


async def test_a_removed_session_says_when_and_why(env):
    home, socket = env
    async with Server(home, socket) as t:
        t.mgr.keep_days = 3
        await t.closed("gone", home)
        await t.mgr.sweep()
        when = time.strftime("%Y-%m-%d")
        for tool in ("read", "run"):
            args = {"input": "x"} if tool == "run" else {}
            with pytest.raises(AssertionError, match=(
                    f"no session named 'gone' any more: its logs were removed on {when} [0-9:]+ "
                    r"\(closed and unused -- neither written nor read -- for 4.0 days\)")):
                await t(tool, session="gone", **args)
        # Tombstones go after 30 days.
        lines = (home / "removed.jsonl").read_text().splitlines()
        old = {**json.loads(lines[0]), "removed": time.time() - 31 * DAY}
        (home / "removed.jsonl").write_text(json.dumps(old) + "\n")
        retention.prune_tombstones(home)
        assert (home / "removed.jsonl").read_text() == ""
