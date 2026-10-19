"""End-to-end tests: the real MCP server driving real tmux sessions."""

import asyncio
import re
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from needs import needs
from pathlib import Path
from mcp import Client

from docent_terminal_mcp.server import build_server

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="needs tmux")


@pytest.fixture
def env(tmp_path):
    socket = f"dtmcp-test-{os.getpid()}-{tmp_path.name}"
    yield tmp_path, socket
    subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
    # kill-server leaves the socket file behind.
    tmpdir = os.environ.get("TMUX_TMPDIR", "/tmp")
    (Path(tmpdir) / f"tmux-{os.getuid()}" / socket).unlink(missing_ok=True)


async def call(home, socket, tool, /, **args):
    """Call one tool on a freshly built server, as a new process would."""
    async with Client(build_server(home, socket)) as c:
        r = await c.call_tool(tool, args)
    text = "\n".join(x.text for x in r.content)
    if r.is_error:
        raise AssertionError(text)
    return text


class Term:
    def __init__(self, home, socket):
        self.home, self.socket = home, socket

    async def __call__(self, tool, /, **args):
        return await call(self.home, self.socket, tool, **args)


def plain_home(tmp_path):
    """A HOME whose login profile sets a short, fixed prompt, so tests do not
    depend on the host's startup files."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".bash_profile").write_text("PS1='$ '\n")
    return str(home)


# The core shell tests run under zsh too (with it installed): the same
# markers, from zsh's own integration.
ALSO_ZSH = {
    "test_command_regions_and_exit_status", "test_output_is_clipped_and_searchable", "test_read_pages_a_region",
    "test_interactive_program_inside_bash", "test_interrupt_long_command", "test_pipestatus_is_reported",
    "test_wait_resumes_a_timed_out_command", "test_continuation_prompt_returns_at_once",
    "test_multi_line_heredoc_still_runs_whole", "test_send_reports_commands_that_finish",
    "test_regions_lists_each_command_once", "test_force_reports_what_it_discarded",
    "test_human_input_is_respected_and_attributed", "test_multi_line_input_runs_every_line",
    "test_leaving_a_nested_shell_reports_it", "test_very_long_input_is_typed_in_pieces",
    "test_expect_answers_prompts_in_one_call", "test_no_echo_input_is_never_logged",
    "test_answering_a_question_reports_the_command", "test_send_refuses_to_type_over_a_human",
}


def pytest_generate_tests(metafunc):
    if "term" in metafunc.fixturenames and metafunc.function.__name__ in ALSO_ZSH and shutil.which("zsh"):
        metafunc.parametrize("term", ["bash", "zsh"], indirect=True)


@pytest.fixture
async def term(env, request):
    t = Term(*env)
    if getattr(request, "param", "bash") == "zsh":
        home = Path(env[0]) / "zhome"
        home.mkdir(exist_ok=True)
        (home / ".zshrc").write_text("PS1='$ '\n")
        await t("session_create", name="s", cwd=str(env[0]), command="zsh", env={"HOME": str(home)})
    else:
        await t("session_create", name="s", cwd=str(env[0]), env={"HOME": plain_home(env[0])})
    return t


async def test_command_regions_and_exit_status(term):
    out = await term("run", session="s", input="echo hi; false")
    assert "exit 1" in out and out.splitlines()[1] == "hi"
    out = await term("run", session="s", input="printf 'a\\nb'")
    assert out.splitlines()[1:] == ["a", "b"]


async def test_output_is_clipped_and_searchable(term):
    out = await term("run", session="s", input="seq 1 1000", max_lines=20)
    assert "lines omitted" in out and out.rstrip().endswith("1000")
    out = await term("search", session="s", pattern=r"^77\d$", context=0, max_results=3)
    assert out.startswith("10 matching lines; showing 1-3")
    assert "region 1 (seq 1 1000)" in out
    assert "[more: skip=3]" in out


async def test_read_pages_a_region(term):
    await term("run", session="s", input="seq 1 50")
    out = await term("read", session="s", region=1, start=3, limit=2)
    lines = out.splitlines()
    assert lines[1].split("\t") == ["3", "2"]  # session line 3 holds "2"
    assert lines[2].split("\t") == ["4", "3"]
    assert "next start=5" in lines[3]


async def test_region_files_are_plain_text(term):
    await term("run", session="s", input="printf '\\033[31mred\\033[0m\\n'")
    text = (term.home / "sessions/s/regions/0001.txt").read_text()
    assert text == "red\n"


async def test_region_files_are_read_only(term):
    await term("run", session="s", input="echo one")
    path = term.home / "sessions/s/regions/0001.txt"
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(PermissionError):
        path.write_text("clobbered\n")
    # A dropped region's file is written again (over the read-only mode) when read.
    for i in range(2, 12):
        await term("run", session="s", input=f"echo {i}")
    assert not path.exists()
    out = await term("read", session="s", region=1)
    assert out.splitlines()[1].split("\t")[1] == "one" and path.read_text() == "one\n"
    assert path.stat().st_mode & 0o777 == 0o444


async def test_the_text_fits_claude_codes_caps(env):
    """Claude Code cuts a server's instructions, and each tool's description,
    at 2048 characters (CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH), silently
    for the model.  Models once never saw the last third of the
    instructions or of `run`'s description."""
    from docent_terminal_mcp.server import INSTRUCTIONS, TEXT_CAP
    assert len(INSTRUCTIONS) <= TEXT_CAP, len(INSTRUCTIONS)
    async with Client(build_server(*env)) as c:
        tools = (await c.list_tools()).tools
    over = {t.name: len(t.description or "") for t in tools if len(t.description or "") > TEXT_CAP}
    assert not over, over
    assert "patterns" in await call(*env, "guide")
    assert "| tail" in await call(*env, "guide", topic="patterns")


async def test_tools_carry_claude_codes_hints(env):
    """Every tool is marked always-loaded (never deferred behind Claude
    Code's tool search) with a search hint; the read-only ones say so."""
    async with Client(build_server(*env)) as c:
        tools = (await c.list_tools()).tools
    for t in tools:
        assert t.meta and t.meta.get("anthropic/alwaysLoad") is True and t.meta.get("anthropic/searchHint"), t.name
    read_only = {t.name for t in tools if t.annotations and t.annotations.read_only_hint}
    assert read_only == {"guide", "session_list", "read", "search", "regions"}


def test_the_skill_is_written_from_the_guide(tmp_path):
    """`docent-terminal-mcp skill DIR` writes a Claude Code skill: a description
    within Claude Code's 1536-character cap, the patterns in SKILL.md, one
    reference file per other guide topic; --plugin adds the plugin files."""
    from docent_terminal_mcp.skill import SKILL_DESCRIPTION, write_skill
    assert len(SKILL_DESCRIPTION) <= 1536
    d = write_skill(tmp_path / "skills")
    text = (d / "SKILL.md").read_text()
    assert text.startswith("---\nname: terminal-sessions\ndescription: ")
    assert "| tail" in text and "[programs.md](programs.md)" in text
    assert (d / "programs.md").read_text().startswith("# Programs with their own prompt")
    plugin = tmp_path / "plugin"
    d = write_skill(plugin, plugin=True)
    assert d == plugin / "skills" / "terminal-sessions" and (d / "SKILL.md").exists()
    assert json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["terminal"]["args"][0] == "connect"
    assert json.loads((plugin / ".claude-plugin/plugin.json").read_text())["name"] == "docent-terminal-mcp"


@needs("gdb")
async def test_shell_habits_are_remarked_on_once(term):
    """A one-shot habit in a session's shell command gets a note in the reply
    the first time: gdb -batch with -ex, | tail, capturing output in a file."""
    out = await term("run", session="s", input="gdb --version")
    assert "[note:" not in out, out  # gdb started to say something about itself
    out = await term("run", session="s", input="gdb -q -nx -batch -ex 'print 41+1'")
    assert "$1 = 42" in out and "-batch suits a single question; for a series about the same program" in out, out
    assert "start it once live -- `gdb -q -nx -ex 'print 41+1'` -- and ask each at its (gdb) prompt" in out, out
    out = await term("run", session="s", input="gdb -q -nx -batch -ex 'print 2'")
    assert "[note:" not in out  # once per session
    out = await term("run", session="s", input="seq 1 5 | tail -2")
    assert out.splitlines()[1:3] == ["4", "5"] and "with `| tail`/`| head` only those lines reach the log" in out
    assert "`last_lines=N` shows the same lines as `| tail -N` and keeps the rest" in out, out
    assert "not needed" not in out  # the trade-off, not a rule
    out = await term("run", session="s", input="seq 1 3 | tee kept.log")
    assert "[note:" not in out  # tee keeps the output in the log: nothing to say
    out = await term("run", session="s", input="seq 1 3 > out.log 2>&1; grep -c . out.log")
    assert "[note:" not in out, out  # read back at once
    out = await term("run", session="s", input="seq 1 3 &> 'out2.log'; wc -l < out2.log")
    assert "[note:" not in out, out
    out = await term("run", session="s", input="seq 1 3 > out.log 2>&1; wc -l < out.log.old")
    assert ("[note: this output goes to the file, not this session's log -- the `read` and `search` tools "
            "won't find it there.  `save_to=FILE` gives both, the file and the log]") in out, out
    assert "tee" not in out, out  # no advice against the file; save_to gives both
    out = await term("run", session="s", input="echo 'gdb -batch | tail > x 2>&1' >/dev/null; echo plain")
    assert "[note:" not in out  # all three already noted; a plain command never gets one


@needs("gdb")
async def test_gdb_fed_its_commands_on_stdin_is_one_shot_too(term):  # a model's `printf ... | gdb`
    out = await term("run", session="s", input="printf 'print 6*7\\n' | gdb -q -nx")
    assert "$1 = 42" in out and "-batch suits a single question; for a series about the same program" in out, out
    assert "`gdb -q -nx`" in out, out
    (term.home / "cmds").write_text("print 2\n")
    out = await term("run", session="s", input="gdb -q -nx -batch < cmds")
    assert "[note:" not in out, out  # once per session


async def test_one_shot_gdb_is_remarked_on_once(term):  # no every-3rd repeat
    # The repeat changed no model's habit and was noise to a model
    # running -batch A/B on purpose.
    run = "gdb -q -nx -batch -ex 'print 1' /bin/true"
    notes = []
    for _ in range(4):
        out = await term("run", session="s", input=run)
        notes.append([line for line in out.splitlines() if line.startswith("[note:")])
    assert "-batch suits a single question; for a series about the same program" in notes[0][0], notes
    assert notes[1:] == [[], [], []], notes


async def test_a_wrapped_pgrep_f_is_told_it_matches_itself(term):
    # Typed at the shell, `pgrep -f P` does not see itself; under a wrapper
    # that stays as its parent (timeout, sudo, sh -c), the wrapper's
    # arguments hold P and it is counted -- or killed, with pkill.  (P is
    # made in the session, so no other process's arguments hold it.)
    await term("run", session="s", input="p=dtmcpself-$$-$RANDOM")
    out = await term("run", session="s", input='pgrep -fa "$p"; true')
    assert "[note:" not in out and "dtmcpself" not in out.partition("\n")[2], out
    out = await term("run", session="s", input='timeout 5 pgrep -fa "$p"')
    assert "timeout 5 pgrep -fa dtmcpself-" in out.partition("\n")[2], out  # the wrapper, found
    assert "pkill can kill its own wrapper" in out and "('[m]yserver')" in out, out


async def test_every_parameter_has_a_description(env):
    """Models fill arguments from the parameter descriptions, not the tool
    text."""
    async with Client(build_server(*env)) as c:
        tools = (await c.list_tools()).tools
    missing = []
    for t in tools:
        schema = t.input_schema
        for name, prop in schema["properties"].items():
            if not prop.get("description"):
                missing.append(f"{t.name}.{name}")
        for model, d in schema.get("$defs", {}).items():
            for name, prop in d["properties"].items():
                if not prop.get("description"):
                    missing.append(f"{t.name}.{model}.{name}")
    assert not missing, missing
    assert len(tools) == 15
    close = next(t for t in tools if t.name == "session_close")
    assert "force" not in close.input_schema["properties"]  # a person's choice, not a model's


async def test_no_echo_input_is_never_logged(term):
    out = await term("run", session="s", input="printf 'Password: '; read -s pw; echo; echo len=${#pw}")
    assert "waiting for input" in out
    out = await term("send", session="s", text="sw0rdfish", keys=["enter"])
    assert "len=9" in out
    for path in (term.home / "sessions/s").rglob("*"):
        if path.is_file():
            assert b"sw0rdfish" not in path.read_bytes(), path


async def test_interactive_program_inside_bash(term):
    out = await term("run", session="s", input=f"{sys.executable} -q")
    assert "waiting for input" in out
    out = await term("run", session="s", input="print(6 * 7)")
    assert "[region 2 · >>> print(6 * 7) ·" in out and out.splitlines()[-1] == "42"
    out = await term("run", session="s", input="exit()")
    out = await term("run", session="s", input="echo back")
    assert "$ echo back" in out and out.splitlines()[-1] == "back"


async def test_interrupt_long_command(term):
    out = await term("run", session="s", input="sleep 30", timeout_s=0.5)
    assert "still running" in out
    await term("send", session="s", keys=["ctrl+c"])
    out = await term("regions", session="s")
    assert "exit 130" in out


async def test_human_input_is_respected_and_attributed(term):
    subprocess.run(["tmux", "-L", term.socket, "send-keys", "-t", "s", "-l", "echo from human"], check=True)
    with pytest.raises(AssertionError, match="unsubmitted input you did not type: 'echo from human'"):
        await term("run", session="s", input="echo from model")
    subprocess.run(["tmux", "-L", term.socket, "send-keys", "-t", "s", "Enter"], check=True)
    await term("run", session="s", input="echo from model")
    out = await term("regions", session="s")
    assert "started by human" in out.splitlines()[0]
    assert "started by" not in out.splitlines()[1]


@needs("less")
async def test_full_screen_program_returns_the_screen(term):
    out = await term("send", session="s", text="less /etc/passwd", keys=["enter"])
    assert out.startswith("[screen 160x48 · cursor row") and "full-screen program]" in out
    assert "root" in out
    await term("send", session="s", keys=["q"])
    out = await term("run", session="s", input="echo done")
    assert out.splitlines()[-1] == "done"


async def test_bad_key_is_rejected(term):
    with pytest.raises(AssertionError, match="unknown key"):
        await term("send", session="s", keys=["ctrl-alt-banana"])


@needs("vi")
async def test_run_returns_when_a_full_screen_program_starts(term):
    (term.home / "f.txt").write_text("alpha\nbeta\n")
    out = await term("run", session="s", input="vi f.txt", timeout_s=20)
    assert "full-screen program; drive it with the `send` tool" in out
    assert "alpha" in out and "cursor row 1, col 1" in out
    await term("send", session="s", text=":q!", keys=["enter"])
    out = await term("run", session="s", input="echo out")
    assert out.splitlines()[-1] == "out"


async def test_screen_view_is_literal_rows_with_cursor(env):
    t = Term(*env)
    await t("session_create", name="n", cols=40, rows=10, env={"HOME": plain_home(env[0])})
    await t("run", session="n", input="echo " + "x" * 60)
    out = await t("read", session="n", view="screen")
    lines = out.splitlines()
    # (the log has no wraps: the cursor's row 5 is log line 3; the region
    # names where the output went)
    assert lines[0] == "[screen 40x10 · cursor row 5, col 3 · log line 3 · last region 1 (done)]"
    assert lines[1:5] == [
        "1\t$ echo " + "x" * 33,  # the command line, wrapped...
        "2\t" + "x" * 27,          # ...onto the next row
        "3\t" + "x" * 40,          # the output line, wrapped...
        "4\t" + "x" * 20,          # ...likewise
    ]
    assert lines[5:] == ["5\t$", "[rows 6-10 blank]"]
    out = await t("read", session="n", view="screen", start=3, limit=1)
    assert out.splitlines()[1:] == ["3\t" + "x" * 40, "[rows 4-5 not shown; start=4]", "[rows 6-10 blank]"]


async def test_raw_view_shows_escape_codes(term):
    await term("run", session="s", input="printf '\\033[1mbold\\033[0m\\r\\n'")
    out = await term("read", session="s", view="raw", region=1)
    assert "\\e[1mbold\\e[0m\\r\\r\\n" in out  # the tty's onlcr adds a \\r
    assert out.startswith(f"[{term.home}/sessions/s/raw.log bytes ")


async def test_pipestatus_is_reported(term):
    out = await term("run", session="s", input="false | true")
    assert "exit 0, but stage 1 exited 1 · pipestatus 1 0" in out  # (a failed stage leads)
    out = await term("run", session="s", input="true")
    assert "· pipestatus" not in out


async def test_login_startup_files_are_read(env, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text("export FROM_PROFILE=yes\n. ~/.bashrc\n")
    (home / ".bashrc").write_text("alias hello='echo hi from alias'\n")
    t = Term(*env)
    await t("session_create", name="l", env={"HOME": str(home)})
    out = await t("run", session="l", input="echo $FROM_PROFILE; hello")
    assert out.splitlines()[1:] == ["yes", "hi from alias"]


async def test_wait_resumes_a_timed_out_command(term):
    out = await term("run", session="s", input="sleep 1.5; echo finished", timeout_s=0.3)
    assert "[region 1 · $ sleep 1.5; echo finished · running · file when done]" in out
    out = await term("wait", session="s", timeout_s=10)
    assert "[region 1 · $ sleep 1.5; echo finished · exit 0" in out
    assert out.splitlines()[-1] == "finished"


async def test_wait_takes_progress_dots_for_output_not_a_prompt(term):
    # 2026-09-27: pytest's "....F.." after a timed-out run was "a human's
    # unsubmitted 'F..'" at Python's "..." prompt; wait waited on that.
    out = await term("run", session="s", input="printf '...F....'; sleep 2; echo; echo finished", timeout_s=0.5)
    assert "running" in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "unsubmitted" not in out and "exit 0" in out and out.splitlines()[-1] == "finished", out


async def test_guide(env):
    t = Term(*env)
    assert "programs, fullscreen, output, sharing" in await t("guide")
    assert "send keys=[\"escape\"]" in await t("guide", topic="fullscreen")
    with pytest.raises(AssertionError, match="unknown topic"):
        await t("guide", topic="nope")



# -- regressions --------------------------------------------------------------

def human_types(term, *keys, literal=False):
    args = ["tmux", "-L", term.socket, "send-keys", "-t", "s"] + (["-l"] if literal else []) + list(keys)
    subprocess.run(args, check=True)


async def test_refusal_does_not_recommend_force(term):
    human_types(term, "echo mid-command", literal=True)
    with pytest.raises(AssertionError, match="Wait and retry, or ask the user") as e:
        await term("run", session="s", input="date")
    assert "retry with force" not in str(e.value)


async def test_send_refuses_to_type_over_a_human(term):
    human_types(term, "echo human line", literal=True)
    with pytest.raises(AssertionError, match="did not type: 'echo human line'"):
        await term("send", session="s", keys=["ctrl+u"])
    screen = await term("read", session="s", view="screen")
    assert "$ echo human line" in screen  # nothing was sent


async def test_model_may_continue_its_own_partial_input(term):
    await term("send", session="s", text="echo own")
    out = await term("send", session="s", text=" words", keys=["enter"])
    assert "[finished: region 1 · $ echo own words · exit 0" in out


async def test_force_replaces_a_stale_line(term):
    human_types(term, "echo stale", literal=True)
    out = await term("run", session="s", input="echo fresh", force=True)
    assert "$ echo fresh ·" in out and out.splitlines()[-1] == "fresh"


async def test_continuation_prompt_returns_at_once(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="echo 'unclosed", timeout_s=20)
    assert asyncio.get_running_loop().time() - t0 < 5
    assert "continuation prompt" in out
    out = await term("run", session="s", input="now closed'")
    assert "exit 0" in out and out.splitlines()[1:] == ["unclosed", "now closed"]
    assert "$ echo 'unclosed (+1 line)" in out  # 17


async def test_multi_line_heredoc_still_runs_whole(term):
    out = await term("run", session="s", input="cat <<END\none\ntwo\nEND")
    assert out.splitlines()[1:] == ["one", "two"]
    assert "(+3 lines)" in out.splitlines()[0]


async def test_send_reports_commands_that_finish(term):
    await term("run", session="s", input="sleep 30", timeout_s=0.3)
    out = await term("send", session="s", keys=["ctrl+c"])
    assert "[finished: region 1 · $ sleep 30 · exit 130" in out


async def test_answering_a_question_reports_the_command(term):
    out = await term("run", session="s", input="printf 'Continue? '; read ans; echo got $ans")
    assert "waiting for input" in out
    out = await term("run", session="s", input="yes")
    assert "[the command finished:]" in out
    assert "$ printf 'Continue? '; read ans; echo got $ans · exit 0" in out
    assert "got yes" in out
    first = out.splitlines()[0]
    assert "Continue? yes · finished" in first
    assert out.splitlines()[2] == "['yes' was sent as the answer to that question, not as a command]", out


async def test_program_regions_hold_output_only(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", input="print('a'); print('b')")
    lines = out.splitlines()
    assert lines[0].startswith(">>> print('a'); print('b') · ready") is False  # header begins "[region"
    assert "[region 2 · >>> print('a'); print('b') · ready · lines" in lines[0]
    assert lines[1:] == ["a", "b"]
    await term("run", session="s", input="exit()")


async def test_regions_lists_each_command_once(term):
    await term("run", session="s", input="echo once")
    assert (await term("regions", session="s")).count("echo once") == 1


async def test_clip_marker_names_the_omitted_lines(term):
    out = await term("run", session="s", input="seq 1 100", max_lines=10)
    assert "[... 90 lines omitted (lines 5-94); `read` region=1 start=5 limit=90 shows them, or the `search` tool ...]" in out
    out = await term("read", session="s", region=1, start=5, limit=1)
    assert out.splitlines()[1] == "5\t4"


async def test_read_start_outside_region_is_explained(term):
    await term("run", session="s", input="echo x")
    await term("run", session="s", input="seq 1 5")
    out = await term("read", session="s", region=2, start=1)
    assert "[start 1 is before region 2; starting at 4]" in out
    out = await term("read", session="s", region=2, start=99)
    assert "past the end of region 2 (last line 8)" in out


async def test_raw_header_gives_totals(term):
    await term("run", session="s", input="echo x")
    out = await term("read", session="s", view="raw", region=1)
    assert " of region bytes " in out.splitlines()[0]
    size = (term.home / "sessions/s/raw.log").stat().st_size
    out = await term("read", session="s", view="raw", max_bytes=10)
    assert out.splitlines()[0].endswith(f"bytes {size - 10}-{size} of {size}]")


async def test_screen_limit_shows_newest_rows(term):
    await term("run", session="s", input="seq 1 30")
    out = await term("read", session="s", view="screen", limit=3)
    lines = out.splitlines()
    assert lines[1] == "[rows 1-29 above; start=1 to see them]"
    assert [l.split("\t")[1] for l in lines[2:5]] == ["29", "30", "$"]


async def test_blank_rows_summarised_anywhere(term):
    await term("run", session="s", input="printf 'top\\n\\n\\n\\n\\nbottom\\n'")
    out = await term("read", session="s", view="screen")
    assert "[rows 3-6 blank]" in out and "bottom" in out


@needs("vi")
async def test_screen_changes_mode(env):
    # "changes" compares with the screen held in memory, so this test keeps
    # one server for all its calls instead of one per call.
    home, socket = env
    (home / "f.txt").write_text("".join(f"line {i}\n" for i in range(20)))
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text

        await call("session_create", name="s", cwd=str(home), env={"HOME": plain_home(home)})
        out = await call("run", session="s", input="vi f.txt")
        assert "full-screen program" in out
        out = await call("send", session="s", keys=["j"], screen="changes")
        lines = out.splitlines()
        assert "cursor row 2, col 1" in lines[0] and "changed]" in lines[0]
        assert len(lines) < 6  # just the changed rows, not all 48
        out = await call("send", session="s", keys=["l"], screen="changes")
        assert "cursor row 2, col 2" in out
        out = await call("send", session="s", keys=["j"], screen="cursor")
        assert len(out.splitlines()) == 6  # header + five rows
        await call("send", session="s", text=":q!", keys=["enter"])


@needs("less")
async def test_cursor_mode_on_a_pagers_bottom_line_shows_what_moved(env):
    # A pager's cursor sits on its command line, whose neighbours say nothing
    # of where the view went: the reply hands back the changed rows (one
    # server, so the previous screen is there to compare with).
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text

        await call("session_create", name="s", cwd=str(home), env={"HOME": plain_home(home)})
        await call("run", session="s", input="seq 1 300 | less", timeout_s=5)
        out = await call("send", session="s", text="/^250", keys=["enter"], screen="cursor")
        assert "the rows that changed are shown" in out and "\t250" in out, out
        assert "changed]" in out.splitlines()[0], out
        await call("send", session="s", keys=["q"])


async def test_session_list_details(term):
    out = await term("session_list")
    assert "no humans attached\ttmux -L" in out
    await term("send", session="s", text="exit", keys=["enter"])
    out = await term("session_list")
    assert "exited (0); session_close ends the tmux session, logs are kept" in out
    assert "attach" not in out
    await term("session_close", session="s")
    assert "closed, exited (0)" in await term("session_list")


# -- regressions --------------------------------------------------------------

async def test_closed_session_name_can_be_reused(term):
    await term("run", session="s", input="echo first life")
    await term("session_close", session="s")
    out = await term("session_create", name="s", env={"HOME": plain_home(term.home)})
    assert f"logs of the earlier session 's' moved to {term.home}/sessions/s.1" in out
    assert "first life" in (term.home / "sessions/s.1/session.txt").read_text()
    out = await term("run", session="s", input="echo second life")
    assert "[region 1 · $ echo second life" in out
    with pytest.raises(AssertionError, match="already exists"):
        await term("session_create", name="s")


async def test_force_reports_what_it_discarded(term):
    human_types(term, "echo typed by a human", literal=True)
    out = await term("run", session="s", input="date", force=True)
    assert out.startswith("[force: discarded unsubmitted line: 'echo typed by a human']")
    human_types(term, "echo 'abc", literal=True)
    human_types(term, "Enter")
    out = await term("run", session="s", input="date", force=True)
    assert out.startswith("[force: abandoned incomplete command: \"echo 'abc\"")


@pytest.mark.parametrize("ps1", ["$ ", "alice@host-1:tmp$ "])
@pytest.mark.parametrize("sizes", [(80, 60, 100), (100, 60)])
async def test_own_wrapped_line_is_discarded_after_a_resize(env, sizes, ps1):
    # readline redraws a wrapped line after a resize; read at the old
    # width, its erase took the line above from the log, and `run` then
    # typed after the caller's own unsubmitted text: the two ran as one.
    t = Term(*env)
    home = Path(plain_home(env[0]))
    (home / ".bash_profile").write_text(f"PS1='{ps1}'\n")
    first, *later = sizes
    await t("session_create", name="s", cols=first, rows=30, env={"HOME": str(home)})
    await t("send", session="s", text="echo short")
    out = await t("run", session="s", input="echo first")  # a discard before
    assert out.startswith("[discarded your own unsubmitted text: 'echo short']"), out
    # Two rows at 60 columns, one at 80 or 100.
    await t("send", session="s", text="echo pending $(printf 'item%02d ' $(seq 1 25)) tail-of-pending-line")
    for cols in later:
        await t("session_resize", session="s", cols=cols, rows=30)
    await asyncio.sleep(1)  # readline's redraw has arrived
    out = await t("run", session="s", input="echo second")
    assert out.startswith("[discarded your own unsubmitted text: ") and out.splitlines()[-1] == "second", out
    log = await t("read", session="s")
    assert "\tfirst\n" in log, log  # the redraw's erase did not land on the line above


async def test_send_refusal_describes_send_force(term):
    human_types(term, "echo mine", literal=True)
    with pytest.raises(AssertionError, match="the `send` tool then types after it"):
        await term("send", session="s", text="x")


async def test_send_shows_space_before_cursor(term):
    out = await term("send", session="s", text="echo ")
    assert out.endswith("[cursor is 1 space past the end of the last line]")
    out = await term("send", session="s", text="x")
    assert "past the end" not in out


async def test_answered_question_is_marked(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", input="input('Sure? ')")
    assert "· waiting ·" in out
    await term("run", session="s", input="yes")
    regions = (await term("regions", session="s")).splitlines()
    assert "answered (region 3)" in regions[1]
    await term("run", session="s", input="exit()")


async def test_until_ends_a_wait_in_a_session_whose_program_is_not_a_shell(env):
    # session_create command="python3 crawl.py", then wait until="five errors so far":
    # the wait sat out its whole timeout_s while the crawler printed it.
    t = Term(*env)
    out = await t("session_create", name="p", command="sh -c 'echo started; sleep 14; echo five errors so far; sleep 60'")
    assert "five errors" not in out.split("---", 1)[1], out
    began = time.monotonic()
    out = await t("wait", session="p", timeout_s=40, until="five errors")
    assert time.monotonic() - began < 20, out
    assert out.startswith("[`until` matched") and "five errors so far" in out, out


@pytest.mark.skipif(sys.version_info < (3, 13), reason="the REPL indents by itself from Python 3.13")
async def test_a_block_sent_after_its_header_alone_keeps_its_own_indentation(env):
    # glm sent `class Matrix:` alone, then the body with its own indentation:
    # the REPL's own indent went in front of it (IndentationError).
    t = Term(*env)
    await t("session_create", name="p", command=f"{sys.executable} -q")
    await t("run", session="p", input="class M:")
    out = await t("run", session="p", input="    def f(self):\n        return 7")
    assert "Error" not in out, out
    out = await t("run", session="p", input="M().f()")
    assert out.splitlines()[-1] == "7", out


async def test_session_program_exit_status(env):
    t = Term(*env)
    await t("session_create", name="p", command=f"{sys.executable} -q")
    out = await t("run", session="p", input="import sys; sys.exit(3)")
    assert "[the program in this session exited with status 3]" in out


# -- expect -------------------------------------------------------------------

ASK = "printf 'Continue? (y/n) '; read a; echo got $a; printf 'Name: '; read n; echo hi $n"


async def test_expect_answers_prompts_in_one_call(term):
    out = await term("expect", session="s", input=ASK, respond=[
        {"pattern": r"Name: $", "line": "Alice"},  # rule order need not match prompt order
        {"pattern": r"Continue\? \(y/n\)", "line": "y"},
    ])
    lines = out.splitlines()
    assert lines[0] == "[expect: the shell command finished · 2 responses]"
    assert lines[1].endswith("'Continue? (y/n)' -> sent line 'y'")
    assert lines[2].endswith("'Name:' -> sent line 'Alice'")
    assert "exit 0" in lines[3]
    assert lines[4:] == ["Continue? (y/n) y", "got y", "Name: Alice", "hi Alice"]


async def test_expect_hands_back_an_unanticipated_question(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("expect", session="s", input=ASK, timeout_s=30, respond=[
        {"pattern": r"Continue\?", "line": "y"},
        {"pattern": r"never asked", "line": "x"},
    ])
    assert asyncio.get_running_loop().time() - t0 < 10  # not the 30s timeout
    assert out.startswith("[expect: output stopped at a question no rule answers, 'Name:'"), out  # which one
    assert "not triggered: 'never asked'" in out
    out = await term("run", session="s", input="Ann")
    assert "hi Ann" in out and "[the command finished:]" in out


async def test_expect_stop_rule_and_secret(term):
    out = await term("expect", session="s", respond=[{"pattern": "Password:", "stop": True}],
                     input="read -s -p 'Password: ' p; echo; echo len=${#p}")
    assert out.startswith("[expect: a stop rule matched; over to you · 1 response]")
    out = await term("send", session="s", text="hunter22", keys=["enter"])
    assert "len=8" in out
    out = await term("expect", session="s", input="read -s -p 'Pin: ' p; echo; echo len=${#p}",
                     respond=[{"pattern": "Pin:", "line": "4321", "secret": True}])
    assert "sent line (secret)" in out and "4321" not in out and "len=4" in out


@needs("ssh localhost")
async def test_a_timeout_in_an_inner_shell_counts_from_its_own_input(term):
    # A line typed into a shell on another host (ssh, then su there): the
    # time so far is that line's, not the ssh's.
    await term("run", session="s", input="ssh -t localhost bash --norc --noprofile -i")
    await term("run", session="s", input="sleep 4")
    out = await term("run", session="s", input="sleep 10; echo x", timeout_s=2)
    m = re.search(r"still running \((\d+) s so far", out)
    assert m and int(m.group(1)) <= 3, out
    await term("send", session="s", keys=["ctrl+c"])
    await term("run", session="s", input="exit")


def test_command_place_looks_at_each_part_of_a_list():
    from docent_terminal_mcp.session import command_place
    place = command_place("uname -r; su - dtest", "h", "root")  # (missed: the session stayed "root@h")
    assert (place.host, place.user, place.via) == ("h", "dtest", "su")
    assert command_place("cd /x && sudo -i", "h", "u").user == "root"
    assert command_place("true || ssh b", "h", "u").host == "b"
    assert command_place("echo 'a;b'; su", "h", "u").user == "root"
    for line in ("ssh b uptime", "echo pw | su - x", "su -c id", "ls; ls", "echo 'su - x'"):
        assert command_place(line, "h", "u") is None, line


@needs("less")
async def test_less_off_the_alternate_screen_is_named_a_pager(term):
    # systemctl's pager: LESS=FRSXMK keeps less on the normal screen, and
    # "lines 1-3/3 (END)" was offered a prompt_pattern, as a question.
    out = await term("run", session="s", input="seq 3 | LESS=RSXMK less", timeout_s=10)
    assert "a pager (less), not a question" in out and "prompt_pattern" not in out, out
    out = await term("send", session="s", keys=["q"])
    assert "exit 0" in out, out


async def test_expect_repeating_rule_and_program_prompt(term):
    out = await term("expect", session="s", input=f"{sys.executable} -q", respond=[])
    assert out.startswith("[expect: the program is at its prompt")
    out = await term("expect", session="s",
                     input="print(sum(int(input('n? ')) for _ in range(3)))",
                     respond=[{"pattern": r"n\? $", "line": "5", "times": 3}])
    assert out.startswith("[expect: the program is at its prompt · 3 responses]")
    assert out.splitlines()[-1] == "15"
    assert "· ready ·" in out
    # Input that ends the program: said of the program, not "the shell command".
    out = await term("expect", session="s", input="exit()", respond=[])
    assert out.startswith("[expect: the input was taken, and the program it went to exited"), out
    assert "exit 0" in out, out


async def test_expect_until(term):
    out = await term("expect", session="s", input="for i in 1 2 3 4 5; do echo step $i; sleep 0.2; done",
                     until=r"^step 3$", respond=[])
    assert out.startswith("[expect: `until` matched · 0 responses]")
    await term("wait", session="s")


async def test_expect_rejects_ambiguous_rules(term):
    with pytest.raises(AssertionError, match="must have one of line, text, keys -- or stop alone"):
        await term("expect", session="s", input="true", respond=[{"pattern": "x", "line": "a", "text": "b"}])


async def test_expect_reports_session_program_exit(env):
    t = Term(*env)
    await t("session_create", name="p", command=f"{sys.executable} -q")
    out = await t("expect", session="p", input="import sys; sys.exit(int(input('code? ')))",
                  respond=[{"pattern": r"code\? $", "line": "4"}])
    assert out.startswith("[expect: the session's own program exited with status 4 · 1 response]")


async def test_committed_rows_are_read_back_from_disk(term):
    # 3000 rows is well past the 500-row edit window, so most of this output
    # lives only in session.txt; regions, read, search and restart must agree.
    out = await term("run", session="s", input="seq 1 3000", max_lines=6)
    assert "(lines 5-2998)" in out  # session line numbers; the region starts at line 2 and out.splitlines()[-1] == "3000"
    out = await term("read", session="s", region=1, start=1001, limit=2)
    assert out.splitlines()[1:3] == ["1001\t1000", "1002\t1001"]
    out = await term("search", session="s", pattern="^1234$", context=1)
    assert out.splitlines()[0] == "1 matching line"
    assert out.splitlines()[2:5] == [" 1234\t1233", ">1235\t1234", " 1236\t1235"]
    region_file = (term.home / "sessions/s/regions/0001.txt").read_text().split("\n")
    assert region_file[:2] == ["1", "2"] and region_file[-2:] == ["3000", ""]
    session_txt = (term.home / "sessions/s/session.txt").read_text().split("\n")
    assert session_txt[1234] == "1234"


# -- nested shells ------------------------------------------------------------

async def test_nested_bash_inherits_integration(term):
    out = await term("run", session="s", input="bash")
    assert "[region 1 · $ bash · running" in out
    out = await term("run", session="s", input="false")
    assert "[region 2 · $ false · exit 1" in out  # a real command region
    out = await term("run", session="s", input="echo inner")
    assert out.splitlines()[-1] == "inner"
    out = await term("run", session="s", input="exit 3")
    assert "[region 4 · $ exit 3 · exit status not seen · lines" in out  # ended with the shell
    regions = (await term("regions", session="s")).splitlines()
    assert "$ bash · exit 3" in regions[0]  # the outer command, closed by the outer shell
    out = await term("run", session="s", input="echo outer")
    assert out.splitlines()[-1] == "outer"


async def test_exec_bash_keeps_integration(term):
    out = await term("run", session="s", input="exec bash")
    regions = await term("regions", session="s")
    assert "$ exec bash ·" in regions and "running" not in regions
    out = await term("run", session="s", input="false")
    assert "$ false · exit 1" in out


async def test_shell_integrate_a_plain_bash(env):
    t = Term(*env)
    await t("session_create", name="p", command="bash --norc --noprofile -i")
    out = await t("run", session="p", input="false")
    assert "exit" not in out.splitlines()[0]  # no markers: prompt matching only
    out = await t("shell_integrate", session="p")
    assert out.startswith("[shell integration active")
    out = await t("run", session="p", input="false")
    assert "$ false · exit 1" in out
    out = await t("run", session="p", input="printf 'x\\ny\\n'")
    assert out.splitlines()[1:] == ["x", "y"]


async def test_shell_integrate_types_nothing_where_already_integrated(term):
    # A remote shell that `docent-terminal-mcp ssh` integrated got the installed-file
    # line anyway, which failed on screen.
    await term("run", session="s", input="true")
    out = await term("shell_integrate", session="s")
    assert out == "[shell integration already active here: nothing typed]"
    assert "integration.bash" not in await term("read", session="s")


async def test_shell_integrate_refuses_non_bash(env):
    t = Term(*env)
    await t("session_create", name="p", command=f"{sys.executable} -q")
    # Refused before typing anything: the cursor is at Python's prompt.
    with pytest.raises(AssertionError, match="shell_integrate types a line at a shell's prompt"):
        await t("shell_integrate", session="p", timeout_s=2)


# -- pane size ----------------------------------------------------------------

class Client_:
    """A real tmux client attached from a pty of a given size, as a human's
    terminal would be."""

    def __init__(self, socket, name, cols, rows):
        import fcntl, pty, struct, termios, threading
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.proc = subprocess.Popen(
            ["tmux", "-L", socket, "attach", "-t", name],
            stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
        )
        os.close(slave)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        try:
            while os.read(self.master, 65536):
                pass
        except OSError:
            pass

    def kill(self):
        self.proc.kill()
        self.proc.wait()
        os.close(self.master)


async def test_a_client_attaching_is_not_human_input(term):
    # adjutant attaches a reader to each pane at once: nine sessions showed
    # "human terminal input 12m ago", the moment they were attached.
    await term("run", session="s", input="true")
    c = Client_(term.socket, "s", 100, 30)
    try:
        assert await wait_for(lambda: asyncio.sleep(0, bool(tmux_cmd(term, "list-clients").strip())))
        await asyncio.sleep(1.2)  # client_activity counts seconds
        out = await term("session_list")
        assert "1 human attached" in out and "human terminal input" not in out, out
        os.write(c.master, b"x")  # a key typed in that terminal
        await asyncio.sleep(0.5)
        out = await term("session_list")
        assert "human terminal input" in out, out
    finally:
        c.kill()


async def wait_for(check, timeout=5):
    for _ in range(int(timeout / 0.1)):
        if await check():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_attaching_resizes_unless_fixed(env):
    t = Term(*env)
    home = plain_home(env[0])
    await t("session_create", name="a", cols=100, rows=30, env={"HOME": home})
    await t("session_create", name="f", cols=100, rows=30, env={"HOME": home}, fixed_size=True)
    clients = [Client_(env[1], n, 70, 20) for n in ("a", "f")]
    try:
        async def attached():
            out = await t("session_list")
            return out.count("1 human attached") == 2
        assert await wait_for(attached)
        out = await t("session_list")
        a_line = next(l for l in out.splitlines() if l.startswith("a\t"))
        f_line = next(l for l in out.splitlines() if l.startswith("f\t"))
        assert "70x" in a_line  # followed the human's window (tmux may keep a status row)
        assert "100x30 fixed" in f_line
        out = await t("session_resize", session="f", cols=90, rows=25)
        assert out == "[session 'f' is 90x25, fixed]"
        out = await t("session_resize", session="f", fixed=False)
        assert "following the attached window" in out and "70x" in out
    finally:
        for c in clients:
            c.kill()


async def test_human_activity_blocks_only_where_it_could_collide(term):
    human = Client_(term.socket, "s", 80, 24)
    try:
        async def attached():
            return "1 human attached" in await term("session_list")
        assert await wait_for(attached)
        # At an empty prompt a keypress, click or focus change cannot collide.
        os.write(human.master, b"\x0c")
        out = await term("run", session="s", input="echo at-prompt")
        assert out.splitlines()[-1] == "at-prompt"
        # While a program runs off a prompt, recent input blocks typing.
        await term("run", session="s", input="cat", timeout_s=1)
        os.write(human.master, b"\x0c")
        await asyncio.sleep(0.3)
        with pytest.raises(AssertionError, match="sent input \\(a key, click, scroll or focus change\\)"):
            await term("send", session="s", text="typed into cat")
        await asyncio.sleep(3.1)  # tmux reports activity in whole seconds
        await term("send", session="s", keys=["ctrl+c"])
    finally:
        human.kill()


async def test_scrolled_back_human_blocks_typing(term):
    subprocess.run(["tmux", "-L", term.socket, "copy-mode", "-t", "s"], check=True)
    with pytest.raises(AssertionError, match="viewing this session's scrollback"):
        await term("run", session="s", input="echo x")
    out = await term("run", session="s", input="echo forced", force=True)
    assert out.splitlines()[-1] == "forced"


@needs("gdb")
async def test_gdb_continuation_prompt(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", input="python\nx = 6 * 7\nprint(x)\nend")
    assert "stopped" not in out and out.splitlines()[-1] == "42"


async def test_two_humans_at_different_sizes(term):
    humans = [Client_(term.socket, "s", 90, 30), Client_(term.socket, "s", 60, 20)]
    try:
        async def both():
            return "2 humans attached" in await term("session_list")
        assert await wait_for(both)
        # tmux sizes the window for the most recently active client.
        os.write(humans[0].master, b"\x0c")
        async def sized(width):
            return f"[screen {width}x" in await term("read", session="s", view="screen", limit=1)
        assert await wait_for(lambda: sized(90))
        os.write(humans[1].master, b"\x0c")
        assert await wait_for(lambda: sized(60))
    finally:
        for h in humans:
            h.kill()


async def test_editing_a_line_with_wide_characters(env):
    t = Term(*env)
    await t("session_create", name="w", env={"HOME": plain_home(env[0]), "LANG": "C.UTF-8"})
    await t("send", session="w", text="echo 日本語テスト", keys=["backspace", "backspace", "left", "left"])
    out = await t("send", session="w", text="X", keys=["enter"])
    assert "[finished: region 1 · $ echo 日本X語テ · exit 0" in out
    assert "日本X語テ" in out.splitlines()


@needs("top")
async def test_top_is_treated_as_full_screen(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="top -d 1", timeout_s=20)
    assert asyncio.get_running_loop().time() - t0 < 10  # returned with the screen, no timeout
    assert "full-screen program; drive it with the `send` tool" in out and "Tasks:" in out
    out = await term("send", session="s", keys=["q"])
    assert "[finished: region 1 · $ top -d 1 · exit 0" in out
    session_txt = (term.home / "sessions/s/session.txt").read_text()
    assert "Tasks:" not in session_txt and "[full-screen program output omitted" in session_txt


@needs("top")
async def test_a_prompt_of_tops_own_comes_back_as_its_screen(term):
    # top paints without the alternate screen and shows the cursor for `O`'s
    # "add filter #1" prompt: the reply said "[no new output ...]".
    await term("run", session="s", input="top -d 1", timeout_s=20)
    out = await term("send", session="s", keys=["O"])
    assert "add filter #1" in out and "no new output" not in out, out
    await term("send", session="s", keys=["enter"])  # an empty filter: none
    out = await term("send", session="s", keys=["q"])
    assert "exit 0" in out, out


async def test_redraw_loop_of_clears_stays_text(term):
    # clear + print in a loop never hides the cursor: its output is kept
    # (it used to become the placeholder).
    out = await term("run", session="s", timeout_s=20,
                     input="for i in 1 2 3; do printf '\\033[H\\033[2Jframe %d\\n' $i; sleep 0.2; done")
    assert out.splitlines()[1:] == ["frame 1", "frame 2", "frame 3"]
    out = await term("run", session="s", input="echo normal again")
    assert out.splitlines()[-1] == "normal again"


# -- prompts ------------------------------------------------------------------

async def test_pdb_with_the_default_prompt(term):
    (term.home / "prog.py").write_text("x = 1\ny = x + 1\nprint('y is', y)\n")
    out = await term("run", session="s", input=f"{sys.executable} -m pdb prog.py")
    assert "waiting for input" in out and "(Pdb)" in out
    out = await term("run", session="s", input="next")
    assert "· ready ·" in out
    out = await term("run", session="s", input="p x")
    assert "· ready ·" in out and out.splitlines()[-1] == "1"
    out = await term("run", session="s", input="continue")
    assert "y is 2" in out
    await term("run", session="s", input="quit")


REPL = """\
import sys
while True:
    try:
        line = input("ready: ")
    except EOFError:
        break
    print("got", line.upper())
"""


async def test_custom_prompt_pattern_per_call_and_default(env):
    home = env[0]
    (home / "repl.py").write_text(REPL)
    t = Term(*env)
    # "ready: " is not a prompt the default pattern knows.
    await t("session_create", name="r", command=f"{sys.executable} {home}/repl.py")
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="r", input="abc", prompt_pattern=r"ready: $")
    assert asyncio.get_running_loop().time() - t0 < 2  # no settle wait
    assert "· ready ·" in out and out.splitlines()[-1] == "got ABC"
    # Without it, run only knows output has stopped mid-line.
    out = await t("run", session="r", input="def")
    assert "· waiting ·" in out
    # As the session default.
    await t("session_create", name="d", command=f"{sys.executable} {home}/repl.py",
            prompt_pattern=r"ready: $")
    out = await t("run", session="d", input="xyz")
    assert "· ready ·" in out and out.splitlines()[-1] == "got XYZ"


async def test_calls_are_logged_without_content(term):
    await term("run", session="s", input="echo secret-ish")
    with pytest.raises(AssertionError):
        await term("read", session="s", view="bogus")
    entries = [json.loads(l) for l in (term.home / "calls.log").read_text().splitlines()]
    assert [e["tool"] for e in entries] == ["session_create", "run", "read"]
    assert entries[1]["session"] == "s" and entries[1]["outcome"] == "ok" and entries[1]["ms"] >= 0
    assert entries[2]["outcome"].startswith("error: view must be")
    assert "secret-ish" not in (term.home / "calls.log").read_text()


# -- input size, multi-line input, index format --------------------------------

async def test_very_long_input_is_typed_in_pieces(term):
    word = "x" * 40000  # tmux refuses a single send-keys this long
    out = await term("run", session="s", input=f"echo {word} | wc -c")
    assert out.splitlines()[-1].strip() == "40001"


async def test_multi_line_input_runs_every_line(term):
    lines = [f"echo line{i}" for i in range(200)]
    out = await term("run", session="s", input="\n".join(lines), max_lines=4)
    regions = (await term("regions", session="s", last=500)).splitlines()
    assert len(regions) == 200 and "$ echo line199 ·" in regions[-1]
    assert not any("started by human" in r for r in regions)
    # The reply lists a few earlier commands and shows the last in full.
    assert "[... 189 more commands; the `regions` tool lists them]" in out
    assert len(out.splitlines()) < 20 and out.splitlines()[-1] == "line199"


async def test_old_index_format_is_converted(term):
    await term("run", session="s", input="echo kept")
    d = term.home / "sessions/s"
    regions = [json.loads(l) for l in (d / "regions.jsonl").read_text().splitlines()]
    (d / "regions.json").write_text(json.dumps(regions))  # as an older server wrote it
    (d / "regions.jsonl").unlink()
    out = await term("regions", session="s")
    assert "$ echo kept · exit 0" in out
    assert not (d / "regions.json").exists() and (d / "regions.jsonl").exists()


# -- regressions --------------------------------------------------------------

async def test_editing_a_wrapped_command_back_onto_one_row(env):
    t = Term(*env)
    await t("session_create", name="u", cols=40, rows=12, env={"HOME": plain_home(env[0])})
    await t("run", session="u", input="echo line-above")
    await t("send", session="u", text="echo " + "a" * 39)  # wraps onto a second row
    out = await t("send", session="u", keys=["backspace"] * 6)
    assert out == "$ echo " + "a" * 33  # R5: the edited line comes back
    assert await t("send", session="u", keys=["ctrl+u"]) == "$"
    out = await t("run", session="u", input="echo X")
    assert out.splitlines()[0].startswith("[region 2 · $ echo X · exit 0")
    out = await t("read", session="u")
    assert [l.split("\t")[1] for l in out.splitlines()[:4]] == ["$ echo line-above", "line-above", "$ echo X", "X"]


async def test_width_changes_are_replayed(env):
    t = Term(*env)
    await t("session_create", name="w", cols=40, rows=12, env={"HOME": plain_home(env[0])})
    await t("session_resize", session="w", cols=30, rows=12)
    await t("run", session="w", input="true")  # a wait loop notices the new width
    await t("send", session="w", text="echo " + "b" * 30)  # wraps at 30, not 40
    await t("send", session="w", keys=["backspace"] * 6)
    await t("send", session="w", keys=["ctrl+u"])
    await t("run", session="w", input="echo Y")
    widths = (env[0] / "sessions/w/widths.jsonl").read_text()
    assert '"width": 30' in widths
    out = await t("read", session="w")  # a fresh server replays from raw.log
    assert [l.split("\t")[1] for l in out.splitlines()[-3:]] == ["$ echo Y", "Y", "$"]


async def test_nested_shell_with_unexported_login_hook(env):
    # As Fedora's /etc/profile.d does: an unexported function added to
    # PROMPT_COMMAND by the login profile -- here twice.
    home = env[0] / "home7"
    home.mkdir()
    (home / ".bash_profile").write_text(
        "PS1='$ '\n"
        "count_prompt() { echo x >> ~/prompts; }\n"
        "PROMPT_COMMAND+=(count_prompt)\n"
        "PROMPT_COMMAND+=(count_prompt)\n"
    )
    t = Term(*env)
    await t("session_create", name="n", env={"HOME": str(home)})
    before = (home / "prompts").read_text().count("x")
    await t("run", session="n", input="true")
    assert (home / "prompts").read_text().count("x") == before + 1  # once, not twice
    out = await t("run", session="n", input="bash")
    out = await t("run", session="n", input="false")
    assert "$ false · exit 1" in out  # the nested shell is integrated
    await t("run", session="n", input="exit")
    log = (env[0] / "sessions/n/session.txt").read_text()
    assert "command not found" not in log


PY_BLOCK = "def f(n):\n    total = 0\n    for i in range(n):\n        total += i * i\n    return total\n\nprint(f(10), f(1000))"


async def test_multi_line_python_in_the_repl(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", input=PY_BLOCK)
    header, *body = out.splitlines()
    assert header.startswith("[region 2 · >>> def f(n): (+") and "· ready ·" in header
    assert body == ["285 332833500"]  # only the output: the echo is the label
    out = await term("run", session="s", input="print('after')")
    assert out.splitlines()[1:] == ["after"]
    out = await term("expect", session="s", input="for i in range(2):\n    print(i)", respond=[])
    assert out.splitlines()[-2:] == ["0", "1"]
    await term("run", session="s", input="exit()")


async def test_leaving_a_nested_shell_reports_it(term):
    await term("run", session="s", input="bash")
    out = await term("run", session="s", input="exit 3")
    assert "[the command finished:]\n[region 1 · $ bash · exit 3" in out


async def test_following_size_is_listed_as_such(env):
    t = Term(*env)
    await t("session_create", name="f", cols=100, rows=30, fixed_size=True, env={"HOME": plain_home(env[0])})
    assert "100x30 fixed" in await t("session_list")
    await t("session_resize", session="f", fixed=False)
    assert " fixed" not in await t("session_list")


@needs("vi")
async def test_screen_replies_honour_max_lines(term):
    (term.home / "f.txt").write_text("one\ntwo\n")
    out = await term("run", session="s", input="vi f.txt", max_lines=5)
    lines = out.splitlines()
    screen = lines[lines.index(next(l for l in lines if l.startswith("[screen"))):]
    assert len(screen) <= 6  # header + at most max_lines rows/summaries
    out = await term("send", session="s", keys=["j"])
    assert "[rows 3-47: '~']" in out  # vim's filler rows, summarised
    await term("send", session="s", text=":q!", keys=["enter"])


async def test_shell_integrate_types_a_readable_line(env):  # nothing opaque typed
    t = Term(*env)
    await t("session_create", name="p", command="bash --norc --noprofile -i")
    await t("shell_integrate", session="p")
    out = await t("read", session="p")
    assert "/shell/integration.bash && export __DTMCP_NONCE=" in out and "base64" not in out, out
    typed = [line for line in out.splitlines() if "integration.bash" in line]
    assert len(typed) == 1 and len(typed[0]) < 200, typed


async def test_force_on_empty_continuation_line(term):
    human_types(term, "echo 'abc", literal=True)
    human_types(term, "Enter")
    out = await term("run", session="s", input="date", force=True)
    assert out.startswith("[force: abandoned incomplete command: \"echo 'abc\"]")


# -- region files: most recently used only ------------------------------------

async def test_only_recent_region_files_are_kept(term):
    for i in range(1, 13):
        await term("run", session="s", input=f"echo output-{i}")
    files = sorted(p.name for p in (term.home / "sessions/s/regions").iterdir())
    assert files == [f"{i:04d}.txt" for i in range(5, 13)]  # the last 8
    regions = (await term("regions", session="s", last=12)).splitlines()
    assert f"{term.home}/sessions/s/session.txt lines 2-2]" in regions[0]  # region 1: a line range
    assert regions[-1].endswith("regions/0012.txt]")
    session_txt = (term.home / "sessions/s/session.txt").read_text().splitlines()
    assert session_txt[1] == "output-1"  # where that range points
    # Reading an old region brings its file back, dropping the least recently used.
    out = await term("read", session="s", region=1)
    assert out.splitlines()[0].endswith("regions/0001.txt]")
    assert (term.home / "sessions/s/regions/0001.txt").read_text() == "output-1\n"
    assert not (term.home / "sessions/s/regions/0005.txt").exists()
    # Listing and searching do not bring files back.
    await term("search", session="s", pattern="output-2")
    assert not (term.home / "sessions/s/regions/0002.txt").exists()


# -- regressions --------------------------------------------------------------

async def test_replay_reads_early_output_at_the_creation_width(env):
    t = Term(*env)
    await t("session_create", name="rp", cols=60, rows=20, fixed_size=True, env={"HOME": plain_home(env[0])})
    await t("run", session="rp", input="echo line-above-the-edit")
    await t("send", session="rp", text="echo replay-width-" + "f" * 50)  # wraps at 60
    await t("send", session="rp", keys=["backspace"] * 12)
    await t("send", session="rp", keys=["ctrl+u"])
    await t("run", session="rp", input="echo after-edit-at-60")
    await t("session_resize", session="rp", cols=100, rows=20)
    await t("run", session="rp", input="echo now-at-100")
    out = await t("read", session="rp")  # every call is a fresh server: a replay
    assert [l.split("\t")[1] for l in out.splitlines()[:6]] == [
        "$ echo line-above-the-edit", "line-above-the-edit",
        "$ echo after-edit-at-60", "after-edit-at-60", "$ echo now-at-100", "now-at-100",
    ]
    out = await t("read", session="rp", region=3)
    assert out.splitlines()[1].endswith("\tnow-at-100")


@needs("gdb")
async def test_several_gdb_commands_in_one_run(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", input="p 111\np 222\np 333")
    headers = [l for l in out.splitlines() if l.startswith("[region")]
    assert [h.split(" · ")[1] for h in headers] == ["(gdb) p 111", "(gdb) p 222", "(gdb) p 333"]
    assert [l for l in out.splitlines() if l.startswith("$")] == ["$1 = 111", "$2 = 222", "$3 = 333"]
    # A line that does not come back to the prompt stops the rest.
    out = await t("run", session="g", input='shell read -p "ok? " x\np 444', settle_ms=1500)
    assert "[stopped: 1 more line not sent" in out
    await t("run", session="g", input="y")


async def test_python_block_label_has_no_trailing_prompt(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", input="for i in range(2):\n    print(i)")
    assert out.splitlines()[0].startswith("[region 2 · >>> for i in range(2): (+1 line) ·")
    await term("run", session="s", input="exit()")


@needs("top")
async def test_run_top_with_a_slow_refresh(term):
    out = await term("run", session="s", input="top -d 5", timeout_s=20)
    assert "full-screen program; drive it with the `send` tool" in out and "[screen " in out
    await term("send", session="s", keys=["q"])


async def test_two_clears_keep_the_output(term):
    out = await term("run", session="s", input="clear; echo first; clear; echo second-final")
    assert out.splitlines()[-1] == "second-final" and "full-screen" not in out
    assert "second-final" in (term.home / "sessions/s/session.txt").read_text()


async def test_wait_reports_across_calls_same_server(env):
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text
        await call("session_create", name="s", env={"HOME": plain_home(home)})
        out = await call("run", session="s", input="sleep 1; echo slept", timeout_s=0.3)
        assert "running" in out.splitlines()[0]
        await asyncio.sleep(1.5)
        out = await call("wait", session="s")
        assert out.startswith("[finished since your last call:]")
        assert "$ sleep 1; echo slept · exit 0" in out and out.splitlines()[-1] == "slept"
        # Reported once: a second wait has nothing new.
        assert (await call("wait", session="s", timeout_s=1)).startswith("[no command running")


async def test_run_discards_own_unsubmitted_text(term):
    await term("send", session="s", text="echo half-typed")
    out = await term("run", session="s", input="echo second-run")
    assert out.splitlines()[0] == "[discarded your own unsubmitted text: 'echo half-typed']"
    assert "$ echo second-run · exit 0" in out and out.splitlines()[-1] == "second-run"


async def test_exit_status_not_seen_is_said(term):
    out = await term("run", session="s", input="printf '\\033]0;unterminated'; echo t3")
    await term("run", session="s", input="true")
    assert "exit status not seen" in (await term("regions", session="s")).splitlines()[0]


@needs("less")
async def test_screen_budget_counts_summaries_once(term):
    out = await term("run", session="s", input="printf 'a\\nb\\nc\\n' | less", max_lines=20)
    screen = out[out.index("[screen "):].splitlines()
    assert any(l.endswith("\ta") for l in screen) and len(screen) <= 21
    await term("send", session="s", keys=["q"])


async def test_search_names_a_region_once(term):
    long = "echo " + "x" * 100 + "; printf 'hit\\nhit\\n'"
    await term("run", session="s", input=long)
    out = await term("search", session="s", pattern="^hit$", context=0)
    assert out.count("region 1 (") == 1 and "...)" in out


# -- regressions --------------------------------------------------------------

async def test_shells_on_other_terminals_are_left_alone(term):
    # A shell a program starts on its own pipe or pty (as expect does for
    # GDB's testsuite) must print its prompts unchanged.
    out = await term("run", session="s", input="printf 'PS1=\"x$ \"\\necho inner\\nexit\\n' | bash -i 2>&1 | cat -v")
    assert "inner" in out and "133" not in out
    # ...while a nested shell on the session's own terminal keeps the markers.
    await term("run", session="s", input="bash")
    out = await term("run", session="s", input="false")
    assert "$ false · exit 1" in out
    await term("run", session="s", input="exit")


PTY_PROBE = r"""
import os, pty
pid, fd = pty.fork()
if pid == 0:
    os.execvp("bash", ["bash", "-i"])
os.write(fd, b'PS1="sub$ "; echo marker-check; exit\n')
out = b""
while True:
    try:
        chunk = os.read(fd, 4096)
    except OSError:
        break
    if not chunk:
        break
    out += chunk
print("LEAK" if b"133;" in out else "CLEAN")
"""


async def test_shell_on_a_new_pty_is_left_alone(term):
    (term.home / "ptyprobe.py").write_text(PTY_PROBE)
    out = await term("run", session="s", input=f"{sys.executable} ptyprobe.py")
    assert out.splitlines()[-1] == "CLEAN"


async def test_a_build_pausing_mid_line_is_not_a_question(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="printf 'checking for _Bool... '; sleep 4; echo yes; echo done")
    assert asyncio.get_running_loop().time() - t0 >= 4
    assert "exit 0" in out.splitlines()[0] and out.splitlines()[-1] == "done"


async def test_a_real_question_still_returns_quickly(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="read -p 'Proceed? ' x; echo got $x", timeout_s=20)
    assert asyncio.get_running_loop().time() - t0 < 3
    assert "waiting for input at 'Proceed?'" in out
    await term("run", session="s", input="y")


async def test_wait_shows_only_the_new_tail(env):
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text
        await call("session_create", name="s", env={"HOME": plain_home(home)})
        out = await call("run", session="s", input="for i in 1 2 3 4 5 6; do echo step$i; sleep 0.4; done",
                         timeout_s=1)
        assert "step1" in out
        out = await call("wait", session="s", timeout_s=10)
        assert "shown before]" in out and "step1" not in out and out.splitlines()[-1] == "step6"


async def test_show_filters_the_reply_not_the_command(term):
    out = await term("run", session="s", input="for i in $(seq 1 200); do echo line$i; [ $((i % 50)) = 0 ] && echo FAIL: t$i; done; false",
                     show="^FAIL", max_lines=20)
    lines = out.splitlines()
    assert "exit 1" in lines[0]  # the command's own status
    assert lines[1].startswith("[4 of 204 lines match '^FAIL'")
    assert [l.split("\t")[1] for l in lines[2:6]] == ["FAIL: t50", "FAIL: t100", "FAIL: t150", "FAIL: t200"]
    assert "[last lines, not matches:]" in lines
    session_txt = (term.home / "sessions/s/session.txt").read_text()
    assert "line137" in session_txt  # nothing was filtered from the log


@needs("gdb")
async def test_expect_multi_line_label(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("expect", session="g", input="p 1\np 2", respond=[])
    # One region per line, sent one at a time.
    assert "[region 1 · (gdb) p 1 · ready ·" in out and "[region 2 · (gdb) p 2 · ready ·" in out, out
    assert out.splitlines()[-1] == "$2 = 2"


@needs("gdb")
async def test_expect_sees_the_prompt_of_a_session_program(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    out = await t("expect", session="g", input="p 5", respond=[], timeout_s=20)
    assert asyncio.get_running_loop().time() - t0 < 5
    assert out.startswith("[expect: the program is at its prompt")


# -- regressions --------------------------------------------------------------

HUMANS: list = []  # clients attached by type_as_human, detached at the end of each test


def type_as_human(socket, name, text):
    """A human types `text` through an attached terminal (and stays): the
    server tells a human's typing from a program's output by that."""
    human = Client_(socket, name, 100, 30)
    HUMANS.append(human)
    import time as t
    for _ in range(50):  # attached?
        if subprocess.run(["tmux", "-L", socket, "list-clients", "-t", f"={name}"],
                          capture_output=True, text=True).stdout.strip():
            break
        t.sleep(0.05)
    t.sleep(0.2)
    os.write(human.master, text.encode())


@pytest.fixture(autouse=True)
def _detach_humans():
    yield
    while HUMANS:
        HUMANS.pop().kill()


@needs("gdb")
async def test_human_line_at_a_program_prompt_is_protected(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    type_as_human(env[1], "g", "print 111")
    await asyncio.sleep(0.3)
    with pytest.raises(AssertionError, match="you did not type: 'print 111'"):
        await t("run", session="g", input="print 222")
    with pytest.raises(AssertionError, match="you did not type: 'print 111'"):
        await t("send", session="g", text="x")
    out = await t("run", session="g", input="print 222", force=True)
    assert out.startswith("[force: discarded unsubmitted line: 'print 111']")
    assert out.splitlines()[-1] == "$1 = 222"


@needs("gdb")
async def test_human_line_in_gdb_under_bash_is_protected(term):
    await term("run", session="s", input="gdb -q -nx")
    type_as_human(term.socket, "s", "print 444")
    await asyncio.sleep(0.3)
    with pytest.raises(AssertionError, match="you did not type: 'print 444'"):
        await term("run", session="s", input="print 555")
    await term("run", session="s", input="quit", force=True)


@needs("gdb")
async def test_own_half_typed_text_at_a_program_prompt(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    await t("send", session="g", text="print 9")
    out = await t("send", session="g", text="9", keys=["enter"])  # continuing its own text
    assert "$1 = 99" in out
    await t("send", session="g", text="print 12")
    out = await t("run", session="g", input="print 7")
    assert out.splitlines()[0] == "[discarded your own unsubmitted text: 'print 12']"
    assert out.splitlines()[-1] == "$2 = 7"


@needs("vi")
async def test_custom_prompt_session_details(env):
    home = env[0]
    (home / "f.txt").write_text("x\n")
    t = Term(*env)
    await t("session_create", name="p", cwd=str(home), prompt_pattern=r"my-shell: $",
            command="bash --norc --noprofile -i", env={"HOME": plain_home(home), "PS1": "my-shell: "})
    await t("run", session="p", input="PS1='my-shell: '")
    out = await t("run", session="p", input="printf 'partial line no newline'")
    assert out.splitlines()[-1] == "partial line no newline"
    out = await t("run", session="p", input="echo next")
    assert out.splitlines()[0].startswith("[region 3 · my-shell: echo next ·")
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="p", input="vi f.txt")
    assert asyncio.get_running_loop().time() - t0 < 2.5
    assert "full-screen; drive it with the `send` tool" in out and "[screen " in out
    await t("send", session="p", text=":q!", keys=["enter"])


async def test_show_collapses_repeats_and_keeps_head_and_tail(term):
    cmd = ("for i in $(seq 1 300); do echo 'WARNING: no global config'; "
           "[ $((i % 30)) = 0 ] && echo FAIL: t$i; done")
    out = await term("run", session="s", input=cmd, show="WARNING|FAIL", max_lines=12)
    lines = out.splitlines()
    assert "[310 of 310 lines match 'WARNING|FAIL', 11 distinct;" in lines[1]
    assert any("WARNING: no global config  [x300, last at line" in l for l in lines)
    assert "FAIL: t300" in out and "more distinct matching lines; search" in out


# -- regressions --------------------------------------------------------------

async def test_progress_line_is_not_a_prompt_once_the_prompt_is_known(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", timeout_s=10,
                     input="import sys, time; sys.stdout.write('45% [===> '); sys.stdout.flush(); time.sleep(2); print('\\ndone-py')")
    assert asyncio.get_running_loop().time() - t0 >= 2
    assert "· ready ·" in out and out.splitlines()[-1] == "done-py"
    await term("run", session="s", input="exit()")


@needs("gdb")
async def test_gdb_progress_line_is_not_a_prompt(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", timeout_s=10,
                  input="shell printf '45%% [===>   ] $ '; sleep 2; echo; echo done-g")
    assert "· ready ·" in out and out.splitlines()[-1] == "done-g"


@needs("gdb")
async def test_wait_resumes_a_timed_out_program_input(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", input="shell sleep 3; echo late-g", timeout_s=1, prompt_pattern=r"\(gdb\) $")
    assert "· timeout ·" in out
    out = await t("wait", session="g", timeout_s=10)
    assert "· ready ·" in out and out.splitlines()[-1] == "late-g"


@needs("gdb")
async def test_expect_until_does_not_outlast_the_prompt(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    out = await t("expect", session="g", input="print 5", respond=[], until="never-matches", timeout_s=15)
    assert asyncio.get_running_loop().time() - t0 < 5
    assert "`until` 'never-matches' did not match" in out and "$1 = 5" in out


async def test_expect_does_not_answer_after_the_program_exits(term):
    out = await term("expect", session="s", respond=[{"pattern": "Continue\\?", "line": "y"}],
                     input="sh -c 'printf \"Continue? \"; exit 3'")
    assert "not sent: the program had exited" in out
    regions = await term("regions", session="s")
    assert "$ y ·" not in regions  # the answer did not run as a shell command


async def test_label_of_input_typed_after_output(env):
    t = Term(*env)
    await t("session_create", name="p", prompt_pattern=r"my-shell: $",
            command="bash --norc --noprofile -i", env={"HOME": plain_home(env[0])})
    await t("run", session="p", input="PS1='my-shell: '")
    await t("run", session="p", input="printf '45%% [===> '; sleep 2; echo", timeout_s=0.5)
    out = await t("run", session="p", input="echo queued", timeout_s=6)
    assert "45%" not in out.splitlines()[0]


async def test_program_sleeping_mid_line_is_not_a_question(term):
    """Quiet mid-line with the process sleeping on a timer: not a question,
    and not "settled" either -- the call keeps waiting for the prompt (up
    to timeout_s), as it would for a shell command."""
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", timeout_s=10, settle_ms=1000,
                     # (A child `sleep`: Python's own time.sleep sometimes shows
                     # as waiting in poll(), like a program waiting for input.)
                     input="import sys, os; sys.stdout.write('45% [===> '); sys.stdout.flush(); os.system('sleep 3'); print('\\ndone-py')")
    # Never a question.  (Under heavy load Python can sit in poll() part-way
    # through the line, looking idle for the 1 s settle: then "settled",
    # which `wait` resumes -- an honest answer, not a wrong one.)
    assert "· waiting ·" not in out.splitlines()[0], out
    if "· settled ·" in out.splitlines()[0]:
        out = await term("wait", session="s", timeout_s=10)
    assert "· ready ·" in out and out.splitlines()[-1] == "done-py", out
    out = await term("run", session="s", timeout_s=1.5, settle_ms=500,
                     input="import time; time.sleep(3); print('late')")
    assert "· timeout ·" in out.splitlines()[0], out  # still busy at timeout_s: not settled
    out = await term("wait", session="s", timeout_s=10)
    assert "· ready ·" in out and out.splitlines()[-1] == "late", out
    await term("run", session="s", input="exit()")


@needs("vi")
async def test_fullscreen_region_closes_in_a_plain_shell(env):
    home, socket = env
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", env={"HOME": plain_home(home)})
    await t("run", session="p", input="true")
    out = await t("run", session="p", input="vi -u NONE")
    assert "· fullscreen ·" in out.splitlines()[0], out
    out = await t("send", session="p", text=":q", keys=["enter"])
    assert "[finished: region 2 · " in out and "· ready ·" in out, out
    assert "· fullscreen ·" not in await t("regions", session="p")


async def test_session_list_caps_closed_sessions(term):
    for i in range(10):
        await term("session_create", name=f"c{i}", env={"HOME": plain_home(term.home)})
        await term("session_close", session=f"c{i}")
    out = await term("session_list")
    assert "c0\t" not in out and "c1\t" not in out and "c9\tbash\tclosed" in out, out
    assert "[2 older closed sessions not listed" in out


async def test_expect_at_the_shell_sends_lines_one_at_a_time(term):
    # All lines went at once: the question of the first took the second as
    # its answer, and the rule's answer was never sent.
    out = await term("expect", session="s", timeout_s=15,
                     input="read -p 'Proceed? [y/N] ' a; echo \"answer=$a\"\nfor i in 1 2\ndo echo loop$i; done\necho last",
                     respond=[{"pattern": r"Proceed\?", "line": "y"}])
    assert "answer=y" in out and "loop2" in out and out.rstrip().endswith("last"), out
    out = await term("expect", session="s", timeout_s=15, respond=[],
                     input="read -p 'Name? ' n\necho never")
    assert "[stopped: 1 more line not sent" in out and "never" not in out, out
    await term("send", session="s", text="x\n")


@needs("gdb")
async def test_gdb_blocks_go_in_one_run(env):
    # Inside `while` gdb's continuation prompt is " >" (indented by depth):
    # a multi-line `run` stopped there, "N more lines not sent".
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", timeout_s=15, input="set $i = 0\nwhile $i < 2\nif $i == 1\nprint 100 + $i\nend\n"
                  "set $i = $i + 1\nend\nprint $i")
    assert "not sent" not in out and "$1 = 101" in out and "$2 = 2" in out, out


@needs("gdb")
async def test_gdbs_pager_gets_its_own_hint(env):
    # "--Type <RET> for more" was offered as a prompt_pattern to pass.
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    await t("run", session="g", input="set height 6", timeout_s=10)
    out = await t("run", session="g", input="help all", timeout_s=10)
    assert "[gdb's pager, not a question: the `run` tool with input=\"c\"" in out and "prompt_pattern" not in out, out
    out = await t("run", session="g", input="q", timeout_s=10)
    assert "· ready ·" in out, out
    await t("run", session="g", input="help all", timeout_s=10)
    out = await t("run", session="g", input="c", timeout_s=10, max_lines=4)
    assert "· ready ·" in out and "pager, not a question" not in out, out


@needs("gdb")
async def test_gdb_under_gdb_resumed_after_ctrl_c_is_waiting_not_working(term):
    # `continue` at (top-gdb) resumes the inner gdb at the (gdb) it printed
    # before the ctrl+c: nothing new appears, and the `run` sat out its timeout.
    await term("run", session="s", timeout_s=30,
               input="gdb -q -nx -iex 'set debuginfod enabled off' -iex 'set prompt (top-gdb) ' --args gdb -q -nx")
    out = await term("run", session="s", input="run", timeout_s=30)
    assert "· ready ·" in out, out
    await term("send", session="s", keys=["ctrl+c"], wait_ms=3000)
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="continue", timeout_s=30)
    took = asyncio.get_running_loop().time() - t0
    assert took < 15 and "probably at a prompt it printed earlier" in out, (took, out)
    out = await term("run", session="s", input="print 6*7", timeout_s=10)
    assert "$1 = 42" in out, out


@needs("gdb")
async def test_expect_sends_lines_one_at_a_time(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("expect", session="g", input="break nosuchfile.c:20\nprint 7", respond=[], timeout_s=10)
    assert "a question no rule answers" in out.splitlines()[0] and "· 0 responses" in out.splitlines()[0], out
    assert "[stopped: 1 more line not sent" in out and "$1 = 7" not in out, out
    out = await t("expect", session="g", input="n", respond=[], timeout_s=10)
    out = await t("expect", session="g", input="print 7\nprint 8", respond=[], timeout_s=10)
    assert "$1 = 7" in out and "$2 = 8" in out, out


async def test_expect_reports_an_answer_that_ran_at_the_shell(term):
    """The program asks, then exits before the answer arrives: the shell
    runs the answer as a command.  The reply says so."""
    out = await term("expect", session="s", timeout_s=10,
                     input="sh -c 'printf \"Continue? \"; sleep 0.3; exit 3'",
                     respond=[{"pattern": "Continue\\?", "line": "false"}])
    # (An answer that is no command would run Fedora's command-not-found
    # handler, which asks PackageKit and can take seconds under load.)
    assert "the answer 'false' ran at the shell as a command (region 2, exit 1)" in out, out


async def test_connect_survives_a_server_restart(served):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, port, environ = served
    params = StdioServerParameters(command=sys.executable, env=environ,
                                   args=["-m", "docent_terminal_mcp.server", "connect", "--actor", "bridged"])
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            r = await session.call_tool("session_create", {"name": "b", "env": {"HOME": plain_home(home)}})
            assert not r.is_error, r.content[0].text
            r = await session.call_tool("run", {"session": "b", "input": "sleep 2; echo survived", "timeout_s": 0.5})
            assert "running" in r.content[0].text.splitlines()[0]
            # The service restarts: a new process, new MCP session ids.  (As
            # systemd does it: the new one once the old has exited.)
            subprocess.run(["pkill", "-f", f"docent_terminal_mcp.server serve --tcp {port}"], check=True)
            for _ in range(100):
                if subprocess.run(["pgrep", "-f", f"docent_terminal_mcp.server serve --tcp {port}"],
                                  capture_output=True).returncode != 0:
                    break
                await asyncio.sleep(0.1)
            new = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=environ,
                                   stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if (home / "s.sock").exists():
                        break
                    await asyncio.sleep(0.1)
                r = await session.call_tool("wait", {"session": "b", "timeout_s": 10})
                assert not r.is_error, r.content[0].text
                assert "$ sleep 2; echo survived · exit 0" in r.content[0].text, r.content[0].text
                assert r.content[0].text.splitlines()[-1] == "survived"
                r = await session.call_tool("regions", {"session": "b"})
                assert "sleep 2; echo survived" in r.content[0].text
            finally:
                new.terminate()
                new.wait(timeout=10)


async def test_connect_sends_the_first_call_after_a_restart(served):
    # 2026-09-27: a restart between calls, and the next `run` and `send` of
    # each client were answered "the server restarted during this call" and
    # never reached the new server -- though the old one never saw them.
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, port, environ = served
    params = StdioServerParameters(command=sys.executable, env=environ,
                                   args=["-m", "docent_terminal_mcp.server", "connect", "--actor", "bridged"])
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            r = await session.call_tool("session_create", {"name": "b", "env": {"HOME": plain_home(home)}})
            assert not r.is_error, r.content[0].text
            subprocess.run(["pkill", "-f", f"docent_terminal_mcp.server serve --tcp {port}"], check=True)
            for _ in range(100):
                if subprocess.run(["pgrep", "-f", f"docent_terminal_mcp.server serve --tcp {port}"],
                                  capture_output=True).returncode != 0:
                    break
                await asyncio.sleep(0.1)
            new = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=environ,
                                   stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if (home / "s.sock").exists():
                        break
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)  # idle: the restart falls between calls
                r = await session.call_tool("run", {"session": "b", "input": "echo once", "timeout_s": 10})
                text = r.content[0].text
                assert not r.is_error and "restarted" not in text and text.splitlines()[-1] == "once", text
                r = await session.call_tool("regions", {"session": "b"})
                assert r.content[0].text.count("echo once") == 1, r.content[0].text
            finally:
                new.terminate()
                new.wait(timeout=10)


async def test_the_servers_own_shell_integration_stays_out_of_its_sessions(env, monkeypatch):
    # A server started from an integrated shell (pytest in a session, on
    # Ubuntu): its exported PROMPT_COMMAND reached `bash --norc`, but not
    # the functions it names -- dash, as sh, drops BASH_FUNC_name%% on its
    # way to the command -- and every prompt said "__dtmcp_prompt: command
    # not found".  The user's own exported functions (`module`) must
    # still get through.
    home, socket = env
    monkeypatch.setenv("PROMPT_COMMAND", "__dtmcp_prompt")
    monkeypatch.setenv("BASH_FUNC___dtmcp_prompt%%", "() {  :\n}")
    monkeypatch.setenv("__DTMCP_TTY", "/dev/pts/999")
    monkeypatch.setenv("BASH_FUNC_userfn%%", "() {  echo user-function\n}")
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", env={"HOME": plain_home(home)})
    await t("run", session="p", input="true")
    out = await t("run", session="p", timeout_s=10,
                  input='echo "pc=[$PROMPT_COMMAND] tty=[$__DTMCP_TTY]"; userfn')
    assert "pc=[] tty=[]" in out and "user-function" in out and "not found" not in out, out


async def test_a_long_command_in_a_plain_shell_is_labelled_by_its_start(env):
    # "region 5 · ...e mcp add -s user terminal -- ...; cd": a plain shell's
    # line kept its end, as a program's input does after its prompt.
    home, socket = env
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", env={"HOME": plain_home(home)})
    await t("run", session="p", input="true")
    line = "echo start-of-it && echo " + "x" * 80 + "; true"
    out = await t("run", session="p", input=line, timeout_s=10)
    assert "· $ echo start-of-it && echo xxx" in out.splitlines()[0], out


async def test_painting_ends_at_the_next_prompt_in_a_plain_shell(env):
    """A command that paints the screen (cursor addressing, as a pager or a
    status tree does) and exits without showing the cursor: in a shell
    without integration the next prompt must end the full-screen state,
    or every later `run` is refused."""
    home, socket = env
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", env={"HOME": plain_home(home)})
    await t("run", session="p", input="true")
    out = await t("run", session="p", input="printf '\\e[?25l\\e[3;1Hpainted\\n'", timeout_s=10)
    assert "full-screen" not in out.splitlines()[-1] or "· ready ·" in out, out
    out = await t("run", session="p", input="echo after-painting", timeout_s=10)
    assert out.splitlines()[-1] == "after-painting", out


async def test_silent_command_in_a_plain_shell_waits_for_the_prompt(env):
    """A shell without integration (ssh): a command that prints nothing
    used to come back "settled" after the settle time; its prompt is
    known, so the call waits for it."""
    home, socket = env
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", env={"HOME": plain_home(home)})
    await t("run", session="p", input="true")  # the prompt kind is learned here
    out = await t("run", session="p", input="sleep 4; echo after-silence", timeout_s=10, settle_ms=1000)
    assert "· ready ·" in out.splitlines()[0] and out.splitlines()[-1] == "after-silence", out




async def test_wait_resumes_a_waiting_region(term):
    await term("run", session="s", input=f"{sys.executable} -q")
    out = await term("run", session="s", timeout_s=10, settle_ms=1000,
                     input="import select, sys; sys.stdout.write('pick> '); sys.stdout.flush(); select.select([sys.stdin], [], [], 3); print('\\nafter')")
    assert "· waiting ·" in out.splitlines()[0], out  # in an input loop, mid-line: looks like a question
    out = await term("wait", session="s", timeout_s=10)  # the model decides it is not one
    assert "· ready ·" in out and out.splitlines()[-1] == "after", out
    await term("run", session="s", input="exit()")


async def test_tmux_server_gets_its_own_scope_under_systemd(env, tmp_path, monkeypatch):
    """Under a systemd service (INVOCATION_ID set), the tmux server is
    started through systemd-run --scope, once, and told to outlive its
    last session.  A fake systemd-run records its arguments and runs the
    command."""
    home, socket = env
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "systemd-run.log"
    fake = bin_dir / "systemd-run"
    fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" >> {record}; echo -- >> {record}\n"
                    'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("INVOCATION_ID", "0123456789abcdef")
    t = Term(home, socket)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    await t("session_create", name="s2", env={"HOME": plain_home(home)})
    calls = record.read_text().split("--\n")[:-1]
    assert len(calls) == 1, calls  # the second session found the server running
    args = calls[0].splitlines()
    assert args[:4] == ["--user", "--scope", "--quiet", "--collect"] and f"--unit=docent-terminal-mcp-tmux-{socket}" in args
    assert args[-8:] == ["-f", str(home / "tmux.conf"), "start-server", ";", "set-option", "-s", "exit-empty", "off"]
    out = subprocess.run(["tmux", "-L", socket, "show-options", "-s", "exit-empty"], capture_output=True, text=True).stdout
    assert out.strip() == "exit-empty off"
    assert "s2\tbash\tlive" in await t("session_list")


async def test_heredoc_into_a_plain_shell_sends_every_line(env):
    """A shell without integration (ssh, su -) takes typed lines one at a
    time; its "> " continuation prompt must count as its prompt, or the
    lines after the first are never sent."""
    home, socket = env
    t = Term(home, socket)
    await t("session_create", name="p", command="bash --norc --noprofile", cwd=str(home),
            env={"HOME": plain_home(home)})
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="p", timeout_s=10,
                  input="cat > unit.txt <<'EOF'\n[Unit]\nDescription=x\nEOF\necho written && cat unit.txt")
    assert asyncio.get_running_loop().time() - t0 < 5, out
    assert "not sent" not in out and out.splitlines()[-2:] == ["[Unit]", "Description=x"], out
    assert (home / "unit.txt").read_text() == "[Unit]\nDescription=x\n"
    # The whole "bash-5.3$ " is the prompt: not in the output; the label
    # keeps only its sign (a prompt's text may be wrong).
    assert out.splitlines()[0].startswith("[region 1 · $ cat > unit.txt <<'EOF' · ready ·"), out
    assert "[region 2 · > [Unit] · ready · " in out and "bash-5.3\n" not in out


async def test_screen_prompt_of_another_kind_does_not_delay_settling(term):
    """The screen check before "settled" (a prompt the log has not caught up
    with) must apply the prompt-family filter too, or a program printing a
    bare "> " and pausing keeps a call waiting until its timeout."""
    await term("run", session="s", input=f"{sys.executable} -q")
    t0 = asyncio.get_running_loop().time()
    # In an input loop (select), so the process is not "busy": the call
    # settles (mid-line: "waiting") once quiet, unless the screen check
    # wrongly holds it for the bare ">" until the timeout.
    out = await term("run", session="s", timeout_s=10, settle_ms=1000,
                     input="import select, sys; sys.stdout.write('> '); sys.stdout.flush(); select.select([sys.stdin], [], [], 3); print('\\nafter')")
    assert asyncio.get_running_loop().time() - t0 < 2.5, out
    assert "· waiting ·" in out.splitlines()[0], out
    out = await term("wait", session="s", timeout_s=10)
    assert "· ready ·" in out and out.splitlines()[-1] == "after", out
    await term("run", session="s", input="exit()")


async def test_session_name_with_a_dot(env):  # tmux 3.7c
    """tmux reads a bare "a.b" target as window a, pane b (3.7c: has-session
    fails, and a listing that crashed on one such session answered 500).
    Sessions are targeted exactly, so a dotted name works throughout."""
    home, socket = env
    t = Term(home, socket)
    out = await t("session_create", name="tekhex.c-db154a-term", env={"HOME": plain_home(home)})
    assert "attach -t =tekhex.c-db154a-term:" in out  # a bare dotted name is window.pane to tmux 3.7c
    out = await t("run", session="tekhex.c-db154a-term", input="echo dotted")
    assert out.splitlines()[-1] == "dotted"
    out = await t("session_list")
    assert "tekhex.c-db154a-term\tbash\tlive, running bash" in out, out
    await t("session_resize", session="tekhex.c-db154a-term", cols=70, rows=20)
    assert "70x20 fixed" in await t("session_list")
    out = await t("send", session="tekhex.c-db154a-term", text="echo again", keys=["enter"])
    assert "again" in out
    assert "closed 'tekhex.c-db154a-term'" in await t("session_close", session="tekhex.c-db154a-term")
    assert "tekhex.c-db154a-term\tbash\tclosed" in await t("session_list")


async def test_wait_reports_a_humans_command_finished_between_calls(env):
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text
        await call("session_create", name="s", env={"HOME": plain_home(home)})
        await call("run", session="s", input="echo mine")
        human_types(Term(home, socket), "echo by-human", literal=True)
        human_types(Term(home, socket), "Enter")
        await asyncio.sleep(1)
        out = await call("wait", session="s", timeout_s=1)
        assert out.startswith("[finished since your last call:]"), out
        assert "$ echo by-human · started by human · exit 0" in out and out.splitlines()[-1] == "by-human"
        assert (await call("wait", session="s", timeout_s=1)).startswith("[no command running")


async def test_search_context_zero_is_compact(term):
    await term("run", session="s", input="printf 'hit\\nx\\nhit\\n'")
    out = await term("search", session="s", pattern="^hit$", context=0)
    assert out.splitlines()[1].startswith("-- region 1 (")
    assert [l.split("\t")[1] for l in out.splitlines()[2:4]] == ["hit", "hit"]


# -- the shared server: several agents ----------------------------------------

from mcp_types import Implementation  # noqa: E402


async def test_two_agents_share_a_session(env):
    home, socket = env
    server = build_server(home, socket, shared=True)
    async with Client(server, client_info=Implementation(name="agent-a", version="1")) as a, \
               Client(server, client_info=Implementation(name="agent-b", version="1")) as b:
        async def call(c, tool, **args):
            r = await c.call_tool(tool, args)
            return r.is_error, r.content[0].text

        err, _ = await call(a, "session_create", name="s", env={"HOME": plain_home(home)})
        assert not err
        err, out = await call(a, "run", session="s", input="echo from-a")
        assert not err and "started by" not in out.splitlines()[0]  # a's own
        err, out = await call(b, "regions", session="s")
        assert "$ echo from-a · started by agent-a ·" in out  # b sees whose it is
        # a half-types; b is refused, and told who typed it.
        await call(a, "send", session="s", text="echo half-by-a")
        err, out = await call(b, "run", session="s", input="echo from-b")
        assert err and "agent 'agent-a' typed it" in out
        # a finishes its own line.
        err, out = await call(a, "send", session="s", text="!", keys=["enter"])
        assert not err and "[finished: region 2 · $ echo half-by-a!" in out
        err, out = await call(b, "run", session="s", input="echo from-b")
        assert not err and out.splitlines()[-1] == "from-b"
        err, out = await call(a, "regions", session="s")
        assert "$ echo from-b · started by agent-b" in out
        # session_list names the agent that started a session, to the others.
        err, out = await call(b, "session_list")
        assert "s\tbash (started by agent-a)\tlive" in out, out
        err, out = await call(a, "session_list")
        assert "started by" not in out


async def test_wait_reports_another_agents_command(env):
    home, socket = env
    server = build_server(home, socket, shared=True)
    async with Client(server, client_info=Implementation(name="agent-a", version="1")) as a, \
               Client(server, client_info=Implementation(name="agent-b", version="1")) as b:
        async def call(c, tool, **args):
            r = await c.call_tool(tool, args)
            assert not r.is_error, r.content[0].text
            return r.content[0].text

        await call(a, "session_create", name="s", env={"HOME": plain_home(home)})
        await call(a, "run", session="s", input="echo from-a")
        await call(b, "run", session="s", input="echo from-b")
        out = await call(a, "wait", session="s", timeout_s=1)
        assert out.startswith("[finished since your last call:]"), out
        assert "$ echo from-b · started by agent-b · exit 0" in out and out.splitlines()[-1] == "from-b"
        assert (await call(a, "wait", session="s", timeout_s=1)).startswith("[no command running")
        # b already saw its own command in its reply: nothing new for b.
        assert (await call(b, "wait", session="s", timeout_s=1)).startswith("[no command running")


def free_port():
    import socket as so
    with so.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


@pytest.fixture
def served(env):
    home, socket = env
    port = free_port()
    environ = {**os.environ, "DOCENT_TERMINAL_MCP_HOME": str(home), "DOCENT_TERMINAL_MCP_SOCKET": socket,
               "DOCENT_TERMINAL_MCP_SOCKET_PATH": str(home / "s.sock")}
    proc = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve", "--tcp", str(port)],
                            env=environ, stderr=subprocess.PIPE)
    import socket as so
    for _ in range(100):
        try:
            so.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            import time as t
            t.sleep(0.1)
    yield home, port, environ
    proc.terminate()
    proc.wait(5)


async def test_shared_server_over_http(served):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    import urllib.error
    import urllib.request

    home, port, _ = served
    url = f"http://127.0.0.1:{port}/mcp"
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"))
    assert e.value.code == 401
    token = (home / "token").read_text().strip()
    assert oct((home / "token").stat().st_mode & 0o777) == "0o600"
    headers = {"Authorization": f"Bearer {token}", "X-Terminal-Actor": "web-ui"}
    async with streamable_http_client(url, http_client=create_mcp_http_client(headers=headers)) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            r = await session.call_tool("session_create", {"name": "h", "env": {"HOME": plain_home(home)}})
            assert not r.is_error, r.content[0].text
            r = await session.call_tool("run", {"session": "h", "input": "echo over-http"})
            assert r.content[0].text.splitlines()[-1] == "over-http"
    log = [json.loads(l) for l in (home / "calls.log").read_text().splitlines()]
    assert {e["actor"] for e in log} == {"web-ui"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/sessions",
                                 headers={"Authorization": f"Bearer {token}"})
    (entry,) = json.loads(urllib.request.urlopen(req).read())["sessions"]
    assert entry["name"] == "h" and entry["state"] == "live" and entry["running"] == "bash"
    assert entry["attach"]["argv"][:3] == ["tmux", "-L", entry["attach"]["socket"]]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/sessions",
                                 headers={"Authorization": f"Bearer {token}"})
    (entry,) = json.loads(urllib.request.urlopen(req).read())["sessions"]
    assert entry["name"] == "h" and entry["state"] == "live" and entry["running"] == "bash"
    assert entry["attach"]["argv"][:3] == ["tmux", "-L", entry["attach"]["socket"]]


async def test_shared_server_over_its_unix_socket(served):
    """On the Unix socket -- the user's own, 0700/0600 -- no token is needed,
    for /api/sessions or for MCP; the TCP port still wants it."""
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    home, port, _ = served
    sock = home / "s.sock"
    assert oct(sock.stat().st_mode & 0o777) == "0o600" and oct(home.stat().st_mode & 0o777) == "0o700"
    http = httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=str(sock)),
                              headers={"X-Terminal-Actor": "over-socket"}, timeout=httpx2.Timeout(30, read=300))
    async with http:
        assert (await http.get("http://x/health")).json()["service"] == "docent-terminal-mcp"
        assert (await http.get("http://x/api/sessions")).json()["sessions"] == []
        async with streamable_http_client("http://x/mcp", http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                r = await session.call_tool("session_create", {"name": "u", "env": {"HOME": plain_home(home)}})
                assert not r.is_error, r.content[0].text
                r = await session.call_tool("run", {"session": "u", "input": "echo over-socket"})
                assert r.content[0].text.splitlines()[-1] == "over-socket"
        (entry,) = (await http.get("http://x/api/sessions")).json()["sessions"]
        assert entry["name"] == "u" and entry["started_by"] == "over-socket"
    async with httpx2.AsyncClient() as tcp:
        assert (await tcp.get(f"http://127.0.0.1:{port}/api/sessions")).status_code == 401


async def test_connect_bridges_stdio_to_the_socket(served):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, port, environ = served
    params = StdioServerParameters(command=sys.executable, env=environ,
                                   args=["-m", "docent_terminal_mcp.server", "connect", "--actor", "bridged"])
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            assert "run" in {t.name for t in (await session.list_tools()).tools}
            r = await session.call_tool("session_create", {"name": "b", "env": {"HOME": plain_home(home)}})
            assert not r.is_error, r.content[0].text
            r = await session.call_tool("run", {"session": "b", "input": "echo bridged"})
            assert r.content[0].text.splitlines()[-1] == "bridged"
    log = [json.loads(l) for l in (home / "calls.log").read_text().splitlines()]
    assert {e["actor"] for e in log} == {"bridged"}
    assert json.loads((home / "sessions" / "b" / "meta.json").read_text())["started_by"] == "bridged"


def test_connect_exits_when_the_client_closes_stdin(served):
    """A client that only closes stdin (rather than killing the bridge)
    must not leave the bridge running."""
    home, port, environ = served
    proc = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "connect"], env=environ,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "c", "version": "0"}}}) + "\n")
    proc.stdin.flush()
    assert '"id":1' in proc.stdout.readline()
    proc.stdin.close()
    try:
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill()


def test_connect_without_a_server_says_so(env):
    home, socket = env
    environ = {**os.environ, "DOCENT_TERMINAL_MCP_SOCKET_PATH": str(home / "none.sock")}
    r = subprocess.run([sys.executable, "-m", "docent_terminal_mcp.server", "connect"], env=environ,
                       input="", capture_output=True, text=True, timeout=20)
    assert r.returncode != 0 and "no server socket at" in r.stderr


def test_second_server_on_the_same_socket_refuses(served, tmp_path):
    """Another state directory (another lock), the same socket path: refused
    while the first server answers there."""
    home, port, environ = served
    other = {**environ, "DOCENT_TERMINAL_MCP_HOME": str(tmp_path / "other-home")}
    r = subprocess.run([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=other,
                       capture_output=True, text=True, timeout=20)
    assert r.returncode != 0 and "is in use by another server" in r.stderr


def test_second_server_on_the_same_state_refuses(served):
    home, port, environ = served
    r = subprocess.run([sys.executable, "-m", "docent_terminal_mcp.server"], env=environ,
                       input="", capture_output=True, text=True, timeout=20)
    assert r.returncode != 0 and "is in use by pid" in r.stderr


async def test_http_agents_without_a_configured_name_are_told_apart(served):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    home, port, _ = served
    url = f"http://127.0.0.1:{port}/mcp"
    auth = {"Authorization": f"Bearer {(home / 'token').read_text().strip()}"}
    async with streamable_http_client(url, http_client=create_mcp_http_client(headers=auth)) as s1, \
               streamable_http_client(url, http_client=create_mcp_http_client(headers=auth)) as s2:
        async with ClientSession(s1[0], s1[1], client_info=Implementation(name="claude-code", version="1")) as a, \
                   ClientSession(s2[0], s2[1], client_info=Implementation(name="claude-code", version="1")) as b:
            await a.initialize()
            await b.initialize()
            await a.call_tool("session_create", {"name": "h", "env": {"HOME": plain_home(home)}})
            await a.call_tool("run", {"session": "h", "input": "echo by-first"})
            r = await b.call_tool("regions", {"session": "h"})
            assert "started by claude-code ·" in r.content[0].text
            await b.call_tool("run", {"session": "h", "input": "echo by-second"})
            r = await a.call_tool("regions", {"session": "h"})
            assert "$ echo by-second · started by claude-code#2" in r.content[0].text


def test_health_and_session_listing_endpoints(served):
    import urllib.error
    import urllib.request

    home, port, _ = served
    base = f"http://127.0.0.1:{port}"
    health = json.loads(urllib.request.urlopen(base + "/health").read())
    assert health["service"] == "docent-terminal-mcp" and health["host"] and health["user"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/api/sessions")
    assert e.value.code == 401
    token = (home / "token").read_text().strip()
    req = urllib.request.Request(base + "/api/sessions", headers={"Authorization": f"Bearer {token}"})
    listing = json.loads(urllib.request.urlopen(req).read())
    assert listing["service"] == "docent-terminal-mcp" and listing["sessions"] == []


# Defects models ran into in a first evaluation (E1-E7).

PLAIN_BASH = "bash --norc --noprofile -i"


async def test_command_bash_gets_the_integration(env):
    t = Term(*env)
    out = await t("session_create", name="b", command="bash", env={"HOME": plain_home(env[0])})
    assert "started (bash with shell integration)" in out and "(tmux socket /" in out
    out = await t("run", session="b", input="false")
    assert "exit 1" in out
    out = await t("session_create", name="p", command=PLAIN_BASH, env={"HOME": plain_home(env[0])})
    assert f"started ({PLAIN_BASH})" in out


@pytest.mark.parametrize("profile", ["PROMPT_COMMAND+=(:)\n", 'PROMPT_COMMAND="history -a; $PROMPT_COMMAND"\n'])
async def test_a_server_started_inside_a_session_gets_working_shells(env, monkeypatch, profile):
    # 2026-09-27: the suite run inside a session failed 171 tests.  The
    # session exports PROMPT_COMMAND=__dtmcp_prompt; a login profile that
    # adds to it (Fedora's systemd osc-context hook) hid it from the
    # "already integrated" check, so the hook ran itself until bash died.
    monkeypatch.setenv("PROMPT_COMMAND", "__dtmcp_prompt")
    home = Path(plain_home(env[0]))
    (home / ".bash_profile").write_text("PS1='$ '\n" + profile)
    t = Term(*env)
    await t("session_create", name="s", env={"HOME": str(home)})
    out = await t("run", session="s", input="echo ok; false", timeout_s=10)
    assert "exit 1" in out and "\nok" in out, out
    out = await t("run", session="s", input="bash -i", timeout_s=10)  # and one nested in it
    out = await t("run", session="s", input="echo nested", timeout_s=10)
    assert "exit 0" in out and "\nnested" in out, out


@needs("ipython3")
async def test_ipython_prompts_are_known(env):
    t = Term(*env)
    await t("session_create", name="s", env={"HOME": plain_home(env[0])})
    for line, want in (("ipython3 --no-banner --colors=NoColor --no-confirm-exit", "In [1]:"),
                       ("x = 6*7; x", "Out[1]: 42"),
                       ("def f(n):\n    return n * 2\n\nf(x)", "Out[2]: 84"),  # one paste (one cell), like Python's
                       ("for i in range(2): print(i)", "\n1"),  # the empty line that runs it
                       ("exit", "exit 0")):
        t0 = asyncio.get_running_loop().time()
        out = await t("run", session="s", input=line, timeout_s=20)
        took = asyncio.get_running_loop().time() - t0
        assert want in out and "· waiting ·" not in out and took < 8, (line, took, out)
        assert "\n\n\n" not in out, out  # not the rows prompt_toolkit reserves below its prompt


async def test_programs_started_from_a_plain_bash_are_ready_at_their_prompts(env):
    t = Term(*env)
    await t("session_create", name="p", command=PLAIN_BASH, env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    for line in ("gdb -q -nx", "print 6*7", "set prompt (top-gdb) ", "print 2", "quit", "echo back",
                 f"{sys.executable} -q", "6*7", "exit()"):
        out = await t("run", session="p", input=line, timeout_s=20)
        assert "· ready ·" in out, (line, out)
    assert asyncio.get_running_loop().time() - t0 < 10  # no settle waits


async def test_prompt_pattern_keeps_continuation_prompts(env):
    t = Term(*env)
    await t("session_create", name="p", command=PLAIN_BASH, cwd=str(env[0]), env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="p", input="cat > hd <<'EOF'\none\ntwo\nEOF", prompt_pattern=r"\$ $",
                  timeout_s=30)
    assert asyncio.get_running_loop().time() - t0 < 10, out  # not the timeout at "> "
    assert "not sent" not in out and (env[0] / "hd").read_text() == "one\ntwo\n"


@needs("gdb")
async def test_run_with_a_prompt_pattern_at_the_shell_is_the_callers_command(term):
    out = await term("run", session="s", input="gdb -q -nx", prompt_pattern=r"\(gdb\) $")
    assert "[region 1 · $ gdb -q -nx · running" in out and "waiting for input at '(gdb)'" in out
    await term("run", session="s", input="quit", prompt_pattern=r"\(gdb\) $")
    out = await term("regions", session="s")
    assert "human" not in out and len(out.splitlines()) == 2, out


async def test_expect_answers_a_question_already_on_screen(term):
    await term("run", session="s", input="read -p 'Quit anyway? (y or n) ' a; echo GOT=$a")
    t0 = asyncio.get_running_loop().time()
    out = await term("expect", session="s", respond=[{"pattern": "Quit anyway", "line": "y"}], timeout_s=20)
    assert asyncio.get_running_loop().time() - t0 < 5
    assert "sent line 'y'" in out and "GOT=y" in out


@needs("gdb")
async def test_a_job_surviving_ctrl_c_is_named(term):
    out = await term("run", session="s", input="gdb -q -nx 2>&1 | grep Breakpoint", timeout_s=2)
    assert "[in the foreground: gdb (pid " in out and "grep (pid " in out
    out = await term("send", session="s", keys=["ctrl+c"])
    assert out.startswith("[still in the foreground after ctrl+c: gdb (pid "), out
    out = await term("send", session="s", text="echo hi", keys=["enter"])
    assert "[no new output; in the foreground: gdb (pid " in out
    # Later timeouts no longer suggest the ctrl+c that failed.
    out = await term("wait", session="s", timeout_s=1)
    assert "ctrl+c did not stop gdb (pid " in out and "keys=['ctrl+c'] interrupts" not in out, out
    out = await term("send", session="s", keys=["ctrl+d"])
    assert "[finished: region 1" in out


@needs("gdb", "python3")
async def test_a_busy_job_that_defers_ctrl_c_is_not_told_to_quit(term):
    """gdb printing a large array takes ctrl+c only when it next checks
    (about 50 s later).  While it is busy, the hints must not suggest ctrl+d
    or quit (read at its next prompt, they would end it) and must say to
    wait; here a Python loop stands in for gdb."""
    prog = ("python3 -c $'import signal, time\\nf = []\\nsignal.signal(2, lambda *a: f.append(1))\\n"
            "t = time.time()\\nwhile time.time() - t < 6: pass\\nprint(\"interrupted\" if f else \"not\")'")
    out = await term("run", session="s", input=prog, timeout_s=1)
    assert "[in the foreground: python3 (pid " in out, out
    out = await term("send", session="s", keys=["ctrl+c"])
    assert "\n[ctrl+c delivered, but python3 (pid " in out and "still busy, not at a prompt" in out, out
    assert "The `wait` tool shows it stop" in out and "ends such a program" not in out, out
    out = await term("wait", session="s", timeout_s=1)
    assert "ctrl+c was delivered, but python3 (pid " in out, out
    assert "ctrl+c did not stop" not in out and "ends such a program" not in out, out
    out = await term("wait", session="s", timeout_s=20)
    assert "interrupted" in out and "exit 0" in out, out


async def test_a_ctrl_c_is_remembered_for_its_command_only(term):
    """The same program (gdb, a REPL) runs the next command: a ctrl+c that
    did not stop the last one says nothing about this one."""
    await term("session_create", name="py")  # started from bash, as gdb was in the report
    await term("run", session="py", input="python3 -q")
    await term("run", session="py", input="import signal, time; f = []; signal.signal(2, lambda *a: f.append(1))")
    busy = "exec('t = time.time()\\nwhile time.time() - t < 4: pass')"
    await term("run", session="py", input=busy, timeout_s=1)
    out = await term("send", session="py", keys=["ctrl+c"])
    assert "ctrl+c delivered, but python3" in out, out
    out = await term("wait", session="py", timeout_s=20)
    assert " · ready" in out, out
    out = await term("run", session="py", input=busy, timeout_s=1)
    assert "delivered" not in out and "did not stop" not in out, out
    assert "keys=['ctrl+c'] interrupts" in out, out


async def test_a_command_reading_without_a_prompt_is_waiting(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="cat", timeout_s=30)
    assert asyncio.get_running_loop().time() - t0 < 6, out
    assert "waiting for input (it is reading the terminal, with no prompt printed)" in out, out
    out = await term("run", session="s", input="hello")
    assert "hello" in out
    out = await term("send", session="s", keys=["ctrl+d"])
    assert "$ cat · exit 0" in out, out
    # Silent but busy is not waiting.
    out = await term("run", session="s", input="sleep 3; echo slept", timeout_s=30)
    assert "exit 0" in out and out.splitlines()[-1] == "slept", out


@needs("/proc syscall")
async def test_a_program_splicing_from_the_terminal_is_waiting(term):
    # Ubuntu 26.04's cat (uutils) reads its input with splice(0, ...), not
    # read(): "cat" ran to its 30 s timeout, never "waiting for input".
    code = "import os; r, w = os.pipe(); n = os.splice(0, w, 4096); print('spliced', n)"
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=30)
    assert asyncio.get_running_loop().time() - t0 < 6, out
    assert "waiting for input (it is reading the terminal" in out, out
    out = await term("run", session="s", input="hello")
    assert "spliced 6" in out, out


async def test_a_program_reading_without_a_prompt_in_a_plain_bash(env):
    t = Term(*env)
    await t("session_create", name="p", command=PLAIN_BASH, env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="p", input="cat", prompt_pattern=r"\$ $", timeout_s=30)
    assert asyncio.get_running_loop().time() - t0 < 6 and "· waiting ·" in out, out
    await t("send", session="p", keys=["ctrl+d"])


async def test_send_signal_ends_a_stuck_job(term):
    await term("run", session="s", input="bash -c \"trap '' INT TERM; sleep 30; true\"", timeout_s=1)
    out = await term("send", session="s", signal="TERM")
    assert out.startswith("[sent SIGTERM to the foreground job: bash (pid "), out
    assert "[still running after SIGTERM: bash (pid " in out and 'signal="KILL" ends it' in out
    out = await term("send", session="s", signal="KILL")
    assert "exit 137" in out
    with pytest.raises(AssertionError, match="nothing to signal: nothing runs in the foreground"):
        await term("send", session="s", signal="TERM")
    with pytest.raises(AssertionError, match="`signal` goes alone"):
        await term("send", session="s", signal="TERM", keys=["enter"])


@needs("gdb")
async def test_a_question_under_an_explicit_prompt_pattern_returns(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    t0 = asyncio.get_running_loop().time()
    out = await t("run", session="g", input="break nosuchfunc\ninfo breakpoints", prompt_pattern=r"\(gdb\) $",
                  timeout_s=30)
    assert asyncio.get_running_loop().time() - t0 < 10, out
    assert "· waiting ·" in out and "(y or [n])" in out and "[stopped: 1 more line not sent" in out
    out = await t("run", session="g", input="n\np 1+1", prompt_pattern=r"\(gdb\) $")
    lines = out.splitlines()
    assert "[region 3 · (gdb) p 1+1 · ready" in lines[2] and lines[-1] == "$1 = 2", out
    assert "· ready ·" in lines[0] and "(y or [n]) n" in lines[0]
    assert lines[1] == "['n' was sent as the answer to that question, not as a command]", out


async def test_wait_resumes_with_the_calls_prompt_pattern(env):
    home = env[0]
    (home / "repl.py").write_text(REPL.replace('print("got"', '__import__("time").sleep(2); print("got"'))
    t = Term(*env)
    await t("session_create", name="r", command=f"{sys.executable} {home}/repl.py")
    out = await t("run", session="r", input="sleep", prompt_pattern=r"ready: $", timeout_s=0.5)
    assert "· timeout ·" in out
    out = await t("wait", session="r", timeout_s=10)
    assert "· ready ·" in out, out


async def test_read_of_a_region_without_output(term):
    await term("run", session="s", input="sleep 3", timeout_s=0.5)
    out = await term("read", session="s", region=1, start=1)
    assert out.endswith("[region 1 has no output yet]"), out


async def two_agents(served):
    """Two named agents on one shared server (an async context manager
    yielding their ClientSessions)."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    home, port, _ = served
    url = f"http://127.0.0.1:{port}/mcp"
    token = (home / "token").read_text().strip()
    stack = contextlib.AsyncExitStack()
    agents = []
    for name in ("ann", "bob"):
        headers = {"Authorization": f"Bearer {token}", "X-Terminal-Actor": name}
        streams = await stack.enter_async_context(
            streamable_http_client(url, http_client=create_mcp_http_client(headers=headers)))
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        agents.append(session)
    return stack, agents


async def timed(coro):
    t0 = asyncio.get_running_loop().time()
    r = await coro
    return r.content[0].text, asyncio.get_running_loop().time() - t0


async def test_one_agents_wait_does_not_block_another(served):
    """Ann waits on Bob's program; Bob's input that ends it must get
    through at once (it waited out Ann's wait: a near-deadlock), and
    reading must never wait for another agent's call."""
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        await bob.call_tool("run", {"session": "p", "input": "cat"})
        waiting = asyncio.ensure_future(timed(ann.call_tool("wait", {"session": "p", "timeout_s": 20})))
        await asyncio.sleep(1)
        text, took = await timed(bob.call_tool("read", {"session": "p"}))
        assert took < 2, text
        text, took = await timed(bob.call_tool("send", {"session": "p", "keys": ["ctrl+d"]}))
        assert took < 5 and "[finished: region 1 · $ cat" in text, (took, text)
        text, took = await waiting
        assert took < 10 and "$ cat · started by bob · exit 0" in text, (took, text)


async def test_a_typing_call_held_up_by_another_agent_says_so(served):
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        running = asyncio.ensure_future(ann.call_tool("run", {"session": "p", "input": "sleep 3"}))
        await asyncio.sleep(0.5)
        text, took = await timed(bob.call_tool("run", {"session": "p", "input": "echo after"}))
        await running
        assert took >= 2 and text.startswith("[waited 2."), text
        assert "for ann's run in this session]" in text.splitlines()[0], text
        assert text.splitlines()[-1] == "after"


async def test_session_list_puts_live_sessions_first_and_says_if_exit_statuses_are_exact(served):
    # Closed sessions mixed into the list had to be
    # scanned past; whether a shell has the integration had to be guessed.
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "a-closed", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("session_close", {"session": "a-closed"})
        await ann.call_tool("session_create", {"name": "b-live", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("session_create", {"name": "c-plain", "command": "bash --norc --noprofile -i"})
        await ann.call_tool("run", {"session": "b-live", "input": "true"})
        await ann.call_tool("run", {"session": "c-plain", "input": "true"})
        out, _ = await timed(bob.call_tool("session_list", {}))
        lines = out.splitlines()
        assert [l.split("\t")[0] for l in lines] == ["b-live", "c-plain", "a-closed"], out
        assert "shell integrated (exact exit statuses)" in lines[0], out
        assert "plain shell (no exit statuses" in lines[1], out


@needs("less")
async def test_field_test_wordings(term):
    out = await term("run", session="s", input="true")
    assert "· exit 0 · no output ·" in out.splitlines()[0], out
    out = await term("run", session="s", input="read -p 'Continue? ' a; echo got=$a", timeout_s=3)
    out = await term("run", session="s", input="y")
    assert "the program quit" not in out, out
    await term("run", session="s", input="seq 1 300 | less", timeout_s=5)
    # On a pager's bottom line, screen="cursor" hands back what moved (the
    # match, at the top), not the rows around the command line.
    out = await term("send", session="s", text="/^250", keys=["enter"], screen="cursor")
    assert 'instead of screen="cursor"' in out and "\t250" in out, out
    await term("send", session="s", keys=["q"])


async def test_an_interrupted_repl_input_does_not_stay_timed_out(term):
    await term("run", session="s", input=f"{sys.executable} -q", timeout_s=10)
    out = await term("run", session="s", input="import time; time.sleep(30)", timeout_s=1)
    assert "timeout" in out.splitlines()[0], out
    await term("send", session="s", keys=["ctrl+c"])
    out = await term("regions", session="s")
    assert "time.sleep(30) · interrupted" in out and "timeout" not in out, out
    await term("send", session="s", keys=["ctrl+d"])


@needs("python3")
async def test_a_search_hit_on_a_long_line_shows_the_part_around_it(term):
    # A hit on a 400,000-character line returned all of it.
    out = await term("run", session="s", input="python3 -c \"print('x' * 300000 + 'NEEDLE' + 'y' * 100000)\"; echo END")
    assert "more characters on this line" in out, out[-300:]
    for context in (0, 1):
        out = await term("search", session="s", pattern="NEEDLE", context=context)
        assert len(out) < 2000 and "NEEDLE" in out and "[col 300001 of 400006]" in out, out[:300]


async def test_a_password_prompt_is_named_as_one(term):
    # getpass: echo off with line input on.  The reply offered a
    # prompt_pattern; a password given to `run` would become a region label.
    out = await term("run", session="s", input=f"{sys.executable} -c \"import getpass; getpass.getpass('Code? ')\"",
                     timeout_s=5)
    assert "a password or passphrase prompt (the terminal is not echoing input)" in out, out
    assert "prompt_pattern" not in out, out
    await term("send", session="s", keys=["ctrl+c"])


async def test_a_pane_adopted_after_its_prompt_is_integrated_on_the_first_run(term):
    # Split, prompt printed, then adopted: logging began after the prompt,
    # so the log showed none, and the first run went in unintegrated.
    tmux_cmd(term, "split-window", "-t", "=s:")
    await asyncio.sleep(1.5)
    await term("session_list")
    out = await term("run", session="s+1", input="echo first; false")
    assert out.startswith("[shell integration added to this adopted pane's bash first"), out
    assert "exit 1" in out, out


async def test_a_wait_after_expect_until_shows_only_what_is_new(served):
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        out, _ = await timed(ann.call_tool("expect", {
            "session": "p", "input": "read -p 'Go? ' a; for i in 1 2 3 4 5 6; do echo n$i; sleep 0.5; done",
            "respond": [{"pattern": "Go\\?", "line": "y"}], "until": "n3", "timeout_s": 10}))
        assert "n3" in out, out
        out, _ = await timed(ann.call_tool("wait", {"session": "p", "timeout_s": 10}))
        assert "n6" in out and "n2" not in out, out


async def test_an_echoed_secret_answer_is_said_to_be_in_the_log(term):
    out = await term("expect", session="s", input="read -p 'Password: ' p; echo done",
                     respond=[{"pattern": "Password:", "line": "hunter2", "secret": True}], timeout_s=10)
    assert "the terminal echoed a secret=true answer" in out, out


async def test_round_7_wordings(served):
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        out, _ = await timed(ann.call_tool("run", {"session": "p", "input": "for i in 1 2 3; do echo x$i; sleep 1; done",
                                                   "until": "NEVER", "timeout_s": 1}))
        assert "[`until` 'NEVER' did not match within 1 s]" in out, out
        await ann.call_tool("wait", {"session": "p", "timeout_s": 10})
        await ann.call_tool("run", {"session": "p", "input": "sleep 900", "timeout_s": 1})
        # Another's busy session is not closed; its owner
        # may, and the reply says what it ended.  (No force for agents.)
        out, _ = await timed(bob.call_tool("session_close", {"session": "p"}))
        assert "not closed: session 'p' is ann's and `sleep 900` (region 2) is running there" in out, out
        out, _ = await timed(ann.call_tool("session_close", {"session": "p"}))
        assert "closed 'p'; it ended what was running: `sleep 900` (region 2)" in out, out
        await ann.call_tool("session_create", {"name": "q", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("session_split", {"session": "q", "name": "q2"})
        out, _ = await timed(ann.call_tool("session_resize", {"session": "q", "cols": 120, "rows": 30}))
        assert "shared with q2, so this pane is smaller" in out, out


CUTTING_SHELL = r"""
import os, sys, termios, tty
fd = sys.stdin.fileno()
tty.setraw(fd)
w = sys.stdout.buffer
def out(b): w.write(b); w.flush()
while True:
    out(b"cut$ ")
    line, col = b"", 5
    while True:
        c = os.read(fd, 1)
        if c in (b"\r", b"\n"):
            break
        if len(line) >= 100:
            continue  # a line buffer of 100: the rest is dropped, not echoed
        line += c
        out(c)
        col += 1
        if col == 40:
            out(b"\r\r\n")  # busybox ash breaks a long echo with real newlines
            col = 0
    out(b"\r\n")
    if line == b"exit":
        break
    out(b"ran %d characters\r\n" % len(line))
"""


async def test_an_echo_cut_short_is_flagged(term):  # busybox ash on OpenWrt
    script = term.home / "cut.py"
    script.write_text(CUTTING_SHELL)
    out = await term("run", session="s", input=f"{sys.executable} {script}", timeout_s=10)
    line = "echo " + "x" * 150
    out = await term("run", session="s", input=line, timeout_s=10)
    assert "echo cut short: 100 of 155 characters" in out and "ran 100 characters" in out, out
    assert "the program echoed only the first 100 of the 155 characters" in out, out
    assert "(+" not in out.splitlines()[0] and out.splitlines()[1] == "ran 100 characters", out
    short = "echo " + "y" * 60  # wrapped by the program, not cut: the label is the whole line
    out = await term("run", session="s", input=short, timeout_s=10)
    assert "cut short" not in out and "ran 65 characters" in out, out
    assert out.splitlines()[0].count("y") >= 30, out.splitlines()[0]
    await term("send", session="s", text="exit", keys=["enter"])


async def test_what_was_shown_survives_a_restart(term):
    # `term` starts a new server for each call, as a restart between them
    # would: a `wait` after it re-showed lines a timed-out `run` had shown.
    out = await term("run", session="s", input="echo first-part; sleep 3; echo second-part", timeout_s=1)
    assert "first-part" in out and "still running" in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "second-part" in out and "shown before" in out and "first-part" not in out.splitlines()[1:], out


async def test_session_list_says_who_typed_when_and_who_is_typing(served):
    # Presence: how old a session is, its last output, which agents typed
    # into it and when, and a typing call in progress -- so an agent can
    # judge whether a session it finds is in use, and by whom.
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("run", {"session": "p", "input": "echo from-ann"})
        await bob.call_tool("run", {"session": "p", "input": "echo from-bob"})
        text, _ = await timed(ann.call_tool("session_list", {}))
        line = next(l for l in text.splitlines() if l.startswith("p\t"))
        assert "started " in line and "output " in line, line
        assert "typed: bob " in line and ", you " in line, line  # the latest first; the caller is "you"
        running = asyncio.ensure_future(bob.call_tool("run", {"session": "p", "input": "sleep 3"}))
        await asyncio.sleep(0.5)
        text, _ = await timed(ann.call_tool("session_list", {}))
        assert "now: bob's run in progress" in text, text
        await running


async def test_ctrl_c_and_signals_do_not_wait_for_a_run_in_flight(served):
    # 2026-09-27: a ctrl+c waited 762 s for another agent's run and then
    # reached an idle prompt; the caller's own, 72 s (Claude Code moves a
    # long call to the background, so the caller's run can be in flight).
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        for who, how in ((bob, {"keys": ["ctrl+c"]}), (ann, {"keys": ["ctrl+c"]}), (bob, {"signal": "TERM"})):
            running = asyncio.ensure_future(timed(ann.call_tool("run", {"session": "p", "input": "sleep 60",
                                                                        "timeout_s": 30})))
            await asyncio.sleep(1)
            text, took = await timed(who.call_tool("send", {"session": "p", **how}))
            assert took < 5 and "waited" not in text, (how, took, text)
            text, took = await running
            assert took < 8 and "$ sleep 60 · exit 1" in text, (how, took, text)  # 130 or 143


async def test_the_timeout_hint_leaves_another_agents_command_alone(served):
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        await bob.call_tool("run", {"session": "p", "input": "sleep 30", "timeout_s": 0.5})
        text, _ = await timed(ann.call_tool("wait", {"session": "p", "timeout_s": 1}))
        assert "ctrl+c" not in text and "bob's command" in text, text
        await bob.call_tool("send", {"session": "p", "keys": ["ctrl+c"]})


async def test_session_create_on_a_live_name_says_whose_it_is(served):
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        r = await bob.call_tool("session_create", {"name": "p"})
        assert r.is_error
        text = r.content[0].text
        assert "started by ann" in text and "use it" in text and "close it first" not in text, text


async def test_a_humans_ctrl_c_is_named(term):
    """A human attached to the session presses ctrl+c in the agent's
    command: the reply says the interrupt was probably theirs."""
    human = Client_(term.socket, "s", 80, 24)
    try:
        async def attached():
            return "1 human attached" in await term("session_list")
        assert await wait_for(attached)
        running = asyncio.ensure_future(term("run", session="s", input="sleep 20", timeout_s=30))
        await asyncio.sleep(1.5)
        os.write(human.master, b"\x03")
        out = await running
        assert "exit 130" in out and "the interrupt was probably theirs, not yours" in out, out
        # The agent's own ctrl+c is not attributed to the human.
        await asyncio.sleep(4)  # past the human's activity
        await term("run", session="s", input="sleep 20", timeout_s=1)
        out = await term("send", session="s", keys=["ctrl+c"])
        assert "exit 130" in out and "probably theirs" not in out, out
    finally:
        human.kill()


async def test_a_run_in_flight_across_a_restart_is_not_typed_again(served):
    """The service restarts while `run` waits for its command: the bridge
    must not send the `run` again (it would type the command a second
    time); it waits for the command on the new server instead."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, port, environ = served
    params = StdioServerParameters(command=sys.executable, env=environ,
                                   args=["-m", "docent_terminal_mcp.server", "connect", "--actor", "bridged"])
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            await session.call_tool("session_create", {"name": "b", "env": {"HOME": plain_home(home)}})
            running = asyncio.ensure_future(session.call_tool(
                "run", {"session": "b", "input": "sleep 4; echo once-$((6*7))", "timeout_s": 30}))
            await asyncio.sleep(1.5)
            # As systemd restarts it: stop the old server (it cancels the call
            # in flight), and start the new one once the old has exited.
            subprocess.run(["pkill", "-f", f"docent_terminal_mcp.server serve --tcp {port}"], check=True)
            for _ in range(100):
                if subprocess.run(["pgrep", "-f", f"docent_terminal_mcp.server serve --tcp {port}"],
                                  capture_output=True).returncode != 0:
                    break
                await asyncio.sleep(0.1)
            new = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=environ,
                                   stderr=subprocess.PIPE)
            try:
                r = await asyncio.wait_for(running, 40)
                text = r.content[0].text if len(r.content) == 1 else "\n".join(c.text for c in r.content)
                assert not r.is_error, text
                assert "the server restarted" in text and "once-42" in text, text
                # (the region shows the command ran, so the note says so
                # instead of "may not have been typed")
                assert "your input did reach the shell before it (region 1)" in text, text
                r = await session.call_tool("regions", {"session": "b"})
                assert r.content[0].text.count("sleep 4; echo once") == 1, r.content[0].text
                # The call's own timeout holds across the restart: what is left of it.
                t0 = asyncio.get_running_loop().time()
                running = asyncio.ensure_future(session.call_tool(
                    "run", {"session": "b", "input": "sleep 30", "timeout_s": 4}))
                await asyncio.sleep(1.5)
                new.terminate()  # (as systemd restarts it)
                new.wait(timeout=10)
                new = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=environ,
                                       stderr=subprocess.PIPE)
                r = await asyncio.wait_for(running, 40)
                took = asyncio.get_running_loop().time() - t0
                assert took < 12, (took, r.content[0].text)
                await session.call_tool("send", {"session": "b", "keys": ["ctrl+c"]})
            finally:
                new.terminate()
                new.wait(timeout=10)


async def test_a_send_in_flight_across_a_restart_is_not_repeated(served):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, port, environ = served
    params = StdioServerParameters(command=sys.executable, env=environ,
                                   args=["-m", "docent_terminal_mcp.server", "connect", "--actor", "bridged"])
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            await session.call_tool("session_create", {"name": "b", "env": {"HOME": plain_home(home)}})
            sending = asyncio.ensure_future(session.call_tool(
                "send", {"session": "b", "text": "echo sent-once", "keys": ["enter"], "wait_ms": 6000,
                         "settle_ms": 6000}))
            await asyncio.sleep(1.5)
            subprocess.run(["pkill", "-f", f"docent_terminal_mcp.server serve --tcp {port}"], check=True)
            for _ in range(100):
                if subprocess.run(["pgrep", "-f", f"docent_terminal_mcp.server serve --tcp {port}"],
                                  capture_output=True).returncode != 0:
                    break
                await asyncio.sleep(0.1)
            new = subprocess.Popen([sys.executable, "-m", "docent_terminal_mcp.server", "serve"], env=environ,
                                   stderr=subprocess.PIPE)
            try:
                r = await asyncio.wait_for(sending, 40)
                assert "which was not repeated" in r.content[0].text, r.content[0].text
                r = await session.call_tool("regions", {"session": "b"})
                assert r.content[0].text.count("echo sent-once") == 1, r.content[0].text
            finally:
                new.terminate()
                new.wait(timeout=10)


@needs("gdb")
async def test_a_humans_input_at_a_program_prompt_is_reported(env):
    """Inside gdb there are no shell markers: a command a human typed at
    the (gdb) prompt between the agent's calls must still become a region
    of theirs, and the agent's next reply must say so."""
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    await t("run", session="g", input="print 1")
    human = Client_(t.socket, "g", 100, 30)
    try:
        await asyncio.sleep(0.5)
        for ch in b"print 22\r":
            os.write(human.master, bytes([ch]))
            await asyncio.sleep(0.02)
        await asyncio.sleep(1)
        out = await t("run", session="g", input="print 3")
        assert out.startswith("[since your last call, a human typed at the prompt: '(gdb) print 22'"), out
        assert out.splitlines()[-1] == "$3 = 3"
        out = await t("regions", session="g")
        assert "(gdb) print 22 · started by human" in out, out
        # The agent's own `send` at the prompt is not taken for a human's.
        await t("send", session="g", text="print 4", keys=["enter"])
        out = await t("run", session="g", input="print 5")
        assert "a human typed" not in out, out
    finally:
        human.kill()


@needs("gdb")
async def test_wait_waits_out_a_humans_unsubmitted_line(env):
    """A human's half-typed line at gdb's prompt: `run` is refused; `wait`
    waits for the line to go (erased or submitted) and says which."""
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    await t("run", session="g", input="print 1")
    human = Client_(t.socket, "g", 100, 30)
    try:
        await asyncio.sleep(0.5)
        os.write(human.master, b"print 22")
        await asyncio.sleep(0.5)

        async def erase_later():
            await asyncio.sleep(2)
            os.write(human.master, b"\x7f" * 8)
        eraser = asyncio.ensure_future(erase_later())
        t0 = asyncio.get_running_loop().time()
        out = await t("wait", session="g", timeout_s=20)
        await eraser
        assert asyncio.get_running_loop().time() - t0 < 8, out
        assert "'print 22' was erased" in out, out
        # Submitted instead: the next wait reports it as the human's command.
        os.write(human.master, b"print 33")
        await asyncio.sleep(0.5)

        async def submit_later():
            await asyncio.sleep(2)
            os.write(human.master, b"\r")
        submitter = asyncio.ensure_future(submit_later())
        out = await t("wait", session="g", timeout_s=20)
        await submitter
        assert "'print 33' was submitted" in out and "(gdb) print 33 · started by human" in out, out
    finally:
        human.kill()


async def test_run_can_type_like_a_person(term):
    t0 = asyncio.get_running_loop().time()
    out = await term("run", session="s", input="echo the quick brown fox; echo jumps over",
                     typing={"wps": 12, "mistakes": 0.6, "seed": 3})
    took = asyncio.get_running_loop().time() - t0
    assert "$ echo the quick brown fox; echo jumps over · exit 0" in out, out
    assert out.splitlines()[-2:] == ["the quick brown fox", "jumps over"], out
    assert took > 0.5  # a key at a time
    # The next call types at once again.
    t0 = asyncio.get_running_loop().time()
    await term("run", session="s", input="echo fast")
    assert asyncio.get_running_loop().time() - t0 < 2


async def test_an_offered_prompt_pattern_works_when_passed_back(env):
    """The pattern a "waiting" reply offers is shown exactly as it is to be
    passed: copied back verbatim, it matches the next (changed) prompt."""
    import re as re_
    t = Term(*env)
    await t("session_create", name="p", command="bash --norc --noprofile -i", env={"HOME": plain_home(env[0])})
    out = await t("run", session="p", input="PS1='[\\t #\\#] ready: '")
    offered = re_.search(r"pass prompt_pattern `([^`]+)`", out).group(1)
    assert "\\\\" not in offered, offered  # no doubled backslashes
    await asyncio.sleep(1.1)  # the time in the prompt changes
    out = await t("run", session="p", input="echo back", prompt_pattern=offered)
    assert "· ready ·" in out and "back" in out, out


# Split windows: a person splits a session's window.
def tmux_cmd(term, *args):
    return subprocess.run(["tmux", "-L", term.socket, *args], capture_output=True, text=True).stdout


async def test_a_split_does_not_redirect_the_session_and_is_adopted(term):
    """`tmux split-window` makes the new pane active; the tools used to
    follow the active pane, so the next `run` typed into an unlogged pane
    and reported a timeout.  A session is its pane, by id; the new pane
    becomes a session of its own, its screen so far in its log."""
    await term("run", session="s", input="tmux split-window")
    tmux_cmd(term, "send-keys", "-t", "=s:.1", "echo before-adoption", "Enter")
    await asyncio.sleep(0.5)
    out = await term("run", session="s", input="echo still-here")
    assert "exit 0" in out and "still-here" in out, out
    assert "is now session 's+1'" in out, out
    out = await term("session_list")
    line = next(l for l in out.splitlines() if l.startswith("s+1\t"))
    assert "pane %" in line and "window 's'" in line and "adopted" in line, line
    out = await term("run", session="s+1", input="echo in-the-other-pane")
    assert "in-the-other-pane" in out, out
    out = await term("search", session="s+1", pattern="before-adoption|in-the-other-pane", context=0)
    # What the pane showed before adoption is in its log, then what came after.
    assert "\tbefore-adoption\n" in out and out.rstrip().endswith("\tin-the-other-pane"), out
    out = await term("search", session="s", pattern="in-the-other-pane", context=0)
    assert out.startswith("0 matching lines"), out


async def test_session_split_is_logged_and_integrated_and_closes_alone(term):
    out = await term("session_split", session="s", below=False)
    assert "session 's+1' started in a new pane of 's''s window (bash with shell integration)" in out, out
    out = await term("run", session="s+1", input="echo right; false")
    assert "exit 1" in out and "right" in out, out
    out = await term("run", session="s", input="echo left")
    assert "exit 0" in out and "left" in out, out
    assert tmux_cmd(term, "display-message", "-p", "-t", "=s:", "#{window_panes}").strip() == "2"
    out = await term("session_close", session="s+1")
    assert "closed 's+1'" in out
    assert tmux_cmd(term, "display-message", "-p", "-t", "=s:", "#{window_panes}").strip() == "1"
    out = await term("run", session="s", input="echo alone")
    assert "exit 0" in out and "alone" in out, out


async def test_session_close_window_closes_every_pane(term):
    await term("session_split", session="s")
    await term("session_split", session="s", name="gdbpane")
    out = await term("session_close", session="gdbpane", window=True)
    assert all(f"closed {n!r}" in out for n in ("s", "s+1", "gdbpane")), out
    assert tmux_cmd(term, "has-session", "-t", "=s") == ""
    assert subprocess.run(["tmux", "-L", term.socket, "has-session", "-t", "=s"]).returncode != 0


async def test_a_session_from_before_pane_ids_finds_its_pane(term):
    """Sessions created before pane ids were recorded: the logged pane of
    their tmux session is theirs, even when a split made another active."""
    meta_path = term.home / "sessions/s/meta.json"
    import json as _json
    meta = _json.loads(meta_path.read_text())
    for k in ("pane_id", "window", "tmux_pid"):
        meta.pop(k, None)
    meta_path.write_text(_json.dumps(meta))
    tmux_cmd(term, "split-window", "-t", "=s:")  # active now, not piped
    out = await term("run", session="s", input="echo mine")
    assert "exit 0" in out and "mine" in out, out
    assert _json.loads(meta_path.read_text())["pane_id"].startswith("%")


# What models tripped on in an evaluation.
@needs("gdb")
async def test_a_yes_no_question_is_answered_not_offered_a_prompt_pattern(env):
    t = Term(*env)
    await t("session_create", name="g", command="gdb -q -nx", env={"HOME": plain_home(env[0])})
    out = await t("run", session="g", input="break no_such_function")
    assert "Enter alone answers no" in out and "pass prompt_pattern" not in out, out
    await t("run", session="g", input="n")


async def test_a_yes_no_question_at_the_shell_names_its_default(term):
    out = await term("run", session="s", input="read -p 'Continue? [Y/n] ' a; echo got=$a")
    assert "Enter alone answers yes" in out, out
    await term("run", session="s", input="")


@needs("python3")
async def test_send_with_several_lines_to_a_program_is_noted_once(term):
    await term("run", session="s", input="python3 -q")
    out = await term("send", session="s", text="1+1\n2+2\n")
    assert "the `send` tool types all its lines at once" in out, out
    out = await term("send", session="s", text="3+3\n4+4\n")
    assert "types all its lines at once" not in out, out


@needs("python3")
async def test_a_prompt_pattern_at_launch_is_noted(term):
    out = await term("run", session="s", input="python3 -q", prompt_pattern=r">>> $")
    assert "prompt_pattern was not needed to start the program" in out, out


async def test_an_expect_rule_can_answer_then_stop(term):
    out = await term("expect", session="s", input="read -p 'Name? ' a; read -p 'Again? ' b; echo got=$a,$b",
                     respond=[{"pattern": "Name\\?", "line": "ann", "stop": True}], timeout_s=10)
    assert out.startswith("[expect: a stop rule matched") and "sent line 'ann', then stopped" in out, out
    await term("run", session="s", input="bob")


async def test_session_errors_say_what_to_do(term):
    await term("run", session="s", input="true")
    with pytest.raises(AssertionError, match="no session named 'build': the live sessions are s "):
        await term("run", session="build", input="true")
    with pytest.raises(AssertionError, match=r"no region 9 in session s: its regions are 1-1"):
        await term("read", session="s", region=9)
    await term("session_close", session="s")
    with pytest.raises(AssertionError, match="session 's' was closed at"):
        await term("run", session="s", input="true")


async def test_a_number_for_a_session_is_taken_for_another_tools_process(term, monkeypatch):
    # codex's own shell answers "Process running with session ID 4440"; models then ask us for '4440'.
    from docent_terminal_mcp import server as server_module
    await term("run", session="s", input="true")
    with pytest.raises(AssertionError, match=r"no session named '4440': a number like that is a process's ID in "
                                             r"another tool .*the live sessions are s"):
        await term("wait", session="4440")
    monkeypatch.setattr(server_module, "LIVE_LISTED", 0)  # a shared server: dozens of sessions, most not yours
    with pytest.raises(AssertionError, match=r"no session named 'build': of the 1 live sessions, yours are s "
                                             r"\(session_list lists them all\)"):
        await term("run", session="build", input="true")


async def test_a_session_whose_tmux_server_is_gone_says_so(term):
    tmux_cmd(term, "kill-server")
    with pytest.raises(AssertionError, match="no tmux server answers.*TMUX_TMPDIR"):
        await term("run", session="s", input="true")


async def test_an_adopted_bash_is_integrated_on_the_first_run_there(term):
    tmux_cmd(term, "split-window", "-t", "=s:")
    out = await term("session_list")  # adopts, types nothing
    assert "s+1\t" in out and "until the first `run` there" in out, out
    out = await term("run", session="s+1", input="false")
    assert out.startswith("[shell integration added to this adopted pane's bash first"), out
    assert "exit 1" in out, out
    out = await term("run", session="s+1", input="echo two; true")
    assert "exit 0" in out and "integration added" not in out, out
    out = await term("session_list")
    line = next(l for l in out.splitlines() if l.startswith("s+1\t"))
    assert line.split("\t")[-2].endswith("(adopted)"), line


async def test_a_pane_moved_to_a_session_of_its_own_is_followed(term):
    """A front end moves an adopted pane out of its window (new-session named
    as the pane's session, break-pane, kill the placeholder).  The pane is
    followed by id: same session, same log, its window now its own; the
    parent's window closes without it."""
    tmux_cmd(term, "split-window", "-t", "=s:")
    await term("session_list")
    pane = next(l for l in (await term("session_list")).splitlines() if l.startswith("s+1\t")).split("pane ")[1].split()[0]
    tmux_cmd(term, "new-session", "-d", "-s", "moved-away")
    tmux_cmd(term, "rename-session", "-t", "=moved-away", "s+1")
    tmux_cmd(term, "break-pane", "-d", "-s", pane, "-t", "=s+1:")
    tmux_cmd(term, "kill-window", "-t", "=s+1:0")
    out = await term("session_list")
    assert "s+1+1" not in out, out  # not adopted a second time
    out = await term("run", session="s+1", input="echo still-mine")
    assert "still-mine" in out, out
    out = await term("session_close", session="s", window=True)
    assert "closed 's'" in out and "s+1" not in out, out
    out = await term("run", session="s+1", input="echo alive")
    assert "alive" in out, out


async def test_a_pane_moved_into_any_window_keeps_its_session(term):
    tmux_cmd(term, "split-window", "-t", "=s:")
    listing = await term("session_list")
    pane = next(l for l in listing.splitlines() if l.startswith("s+1\t")).split("pane ")[1].split()[0]
    tmux_cmd(term, "break-pane", "-d", "-s", pane, "-n", "x")  # a new window of tmux session 's'
    tmux_cmd(term, "new-session", "-d", "-s", "elsewhere")
    tmux_cmd(term, "join-pane", "-d", "-s", pane, "-t", "=elsewhere:")
    out = await term("session_list")
    assert "s+2" not in out and "elsewhere+" not in out, out
    line = next(l for l in out.splitlines() if l.startswith("s+1\t"))
    assert "window 'elsewhere'" in line, line
    out = await term("run", session="s+1", input="echo followed")
    assert "followed" in out, out


EPOLL_ON = ("import select, os, time; r, w = os.pipe(); e = select.epoll(); e.register({fd}, select.EPOLLIN); "
            "t = time.time(); e.poll(6); print('slept', round(time.time() - t))")


@needs("/proc syscall")
async def test_a_job_polling_other_fds_is_not_waiting_for_input(term):
    # A redirected `make check`: DejaGnu's expect waits in epoll on the ptys
    # of the gdb it runs, not on the terminal.  Called "waiting for input",
    # and `wait` invited typing into the running suite, in 7 of 11 trials.
    code = EPOLL_ON.format(fd="r")
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}" > {term.home}/log.txt', timeout_s=4)
    assert "still running" in out and "waiting for input" not in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "waiting for input" not in out and "exit 0" in out, out
    # Its counterpart: an epoll loop on the terminal (node's REPL, say) is.
    code = EPOLL_ON.format(fd="0")
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=4)
    assert "waiting for input" in out, out
    await term("wait", session="s", timeout_s=10)


SELECT_ON = ("import select, os, time; r, w = os.pipe(); t = time.time(); "
             "select.select([{fd}], [], [], 6); print('slept', round(time.time() - t))")


@needs("/proc syscall")
async def test_a_job_selecting_on_other_fds_is_not_waiting_for_input(term):
    # `make -j check < /dev/null > log`: the parent make waits in pselect for
    # its jobs, which looked like an input loop on the terminal.
    code = SELECT_ON.format(fd="r")
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}" > {term.home}/log.txt', timeout_s=4)
    assert "still running" in out and "waiting for input" not in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "waiting for input" not in out and "exit 0" in out, out
    code = SELECT_ON.format(fd="0")  # the counterpart: select on the terminal is
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=4)
    assert "waiting for input" in out, out
    await term("wait", session="s", timeout_s=10)


@needs("/proc syscall")
async def test_a_job_reading_another_pty_is_not_reading_the_terminal(term):
    # A plain `make -j check`: DejaGnu's expect reads its gdb's pty master,
    # the same kernel wait as reading the terminal -- "waiting for input
    # (reading, with no prompt printed)", 6 to 26 times a trial.
    code = ("import os, select; m, s = os.openpty(); t = os.times().elapsed; "
            "select.select([], [], [], 0); os.set_blocking(m, True); "
            "import threading; threading.Timer(6, lambda: os.write(s, b'x')).start(); os.read(m, 1); print('read it')")
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=4)
    assert "still running" in out and "waiting for input" not in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "waiting for input" not in out and "read it" in out, out


@needs("/proc syscall")
async def test_helpers_waiting_elsewhere_do_not_hide_a_program_at_the_terminal(term):
    # Claude Code idle at its input box, its MCP helper processes (same
    # foreground group) waiting on sockets: the helpers made it "busy", and
    # a `run` of "/model" was refused twice -- with advice to send ctrl+c.
    helper = "import select, os; r, w = os.pipe(); select.select([r], [], [], 30)"
    sleeper = "import time; time.sleep(30)"  # (2026-10-01: an MCP server under `uv run`: uv sleeps on a futex)
    code = (f"import select, subprocess, sys; p = subprocess.Popen([sys.executable, '-c', '{helper}']); q = subprocess.Popen([sys.executable, '-c', '{sleeper}']); "
            "print('tui up', flush=True); select.select([0], [], [], 30); print('got', sys.stdin.readline().strip()); "
            "p.kill(); q.kill()")
    out = await term("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=6)
    assert "tui up" in out and "waiting for input" in out, out
    out = await term("run", session="s", input="/model", timeout_s=10)
    assert "got /model" in out, out


async def test_a_refused_line_is_offered_as_a_send(term):
    # The refusal named `wait` and ctrl+c only; for a program that reads
    # keys itself the call that works is `send`, with the caller's own line.
    out = await term("run", session="s", input="sleep 5", timeout_s=1)
    with pytest.raises(AssertionError) as e:
        await term("run", session="s", input="/model", timeout_s=2)
    assert "`send` text=\"/model\" keys=[\"enter\"]" in str(e.value), str(e.value)
    # Local models given "force=true types ahead anyway" retried with it; the
    # right move for a new command is another session (probes, 2026-10-01).
    assert "use another session (session_create" in str(e.value) and "force" not in str(e.value), str(e.value)
    await term("wait", session="s", timeout_s=10)


async def test_the_attach_command_can_carry_a_prefix(env, monkeypatch):
    # A server inside a sandbox or container: `tmux attach` alone is no use
    # to a person outside it.
    monkeypatch.setenv("DOCENT_TERMINAL_MCP_ATTACH_PREFIX", "openshell sandbox exec -n 'my box' --tty --")
    t = Term(*env)
    out = await t("session_create", name="p")
    assert f"watch/join with: openshell sandbox exec -n 'my box' --tty -- tmux -L {env[1]} attach -t p" in out, out
    out = await t("session_list")
    assert "\topenshell sandbox exec -n 'my box' --tty -- tmux -L " in out, out


async def test_a_signal_reaches_the_job_where_groups_cannot_be_signalled(term, monkeypatch):
    # An OpenShell sandbox denies kill(-pgid) though the processes are the
    # caller's own: "not permitted to signal bash, sleep (another user's
    # process ...)".  Each process of the job is signalled instead.
    def denied(pgid, sig):
        raise PermissionError
    monkeypatch.setattr(os, "killpg", denied)
    await term("run", session="s", input="sleep 30", timeout_s=1)
    out = await term("send", session="s", signal="TERM")
    assert "[sent SIGTERM to the foreground job: sleep" in out, out
    assert await wait_for(lambda: regions_has(term, "exit 143")), await term("regions", session="s")


async def regions_has(term, text):
    return text in await term("regions", session="s")


async def test_wait_names_background_jobs(term):
    # `nohup make ... &` in the session: `wait` said "no command running"
    # while make ran on.
    await term("run", session="s", input=f"nohup sleep 20 > {term.home}/bg.log 2>&1 &")
    out = await term("wait", session="s", timeout_s=3)
    assert "[no command in the foreground, but in the background: sleep (pid " in out, out
    await term("run", session="s", input="kill %1")


async def test_a_moment_of_polling_in_a_busy_job_is_not_waiting(term):
    # Busy builds were "waiting for input" minutes in: one check in which a
    # process looked like an input loop (or could not be told) ended the call.
    code = ("import select, sys, time\nwhile True:\n select.select([sys.stdin], [], [], 0.1)\n time.sleep(0.9)")
    out = await term("run", session="s", input=f"{sys.executable} -c '{code}' > {term.home}/log.txt", timeout_s=8)
    assert "waiting for input" not in out and "still running" in out, out
    await term("send", session="s", keys=["ctrl+c"])


def fake_ssh(home):
    """A command named ssh that starts a plain shell (no integration, as
    over a real ssh), for tests without a network."""
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND PS1='remote$ ' bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    return bin_dir


def install_integration(home):
    """What `docent-terminal-mcp install HOST` leaves in the remote user's home."""
    from docent_terminal_mcp.remote import install_script, install_tar
    subprocess.run(["sh", "-c", install_script()], input=install_tar(), env={**os.environ, "HOME": str(home)},
                   check=True, capture_output=True)


async def test_install_puts_the_integration_on_a_host_for_shell_integrate(env):
    # `docent-terminal-mcp install HOST` (the user runs it, once): the files go to
    # ~/.local/share/docent-terminal-mcp/vN there, and shell_integrate types one
    # line that reads them.  Without them it says how to install them.
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    remote_home = Path(home) / "remote-home"
    remote_home.mkdir()
    # ssh with a remote command runs it; without, a plain shell "over there"
    (bin_dir / "ssh").write_text(
        f"#!/bin/bash\nexport HOME={remote_home}\ncase \"${{@: -1}}\" in exec*) exec bash -c \"${{@: -1}}\";; esac\n"
        "env -u PROMPT_COMMAND PS1='remote$ ' bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home), "DOCENT_TERMINAL_MCP_SSH": str(bin_dir / "ssh")})
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    out = await t("shell_integrate", session="s")
    assert "not installed for remote-fake" in out and "`docent-terminal-mcp install remote-fake`" in out, out
    assert "`docent-terminal-mcp ssh remote-fake`" in out, out
    out = await t("run", session="s", input="false", timeout_s=10)
    assert "exit 1" not in out, out  # still a plain shell
    await t("run", session="s", input="exit", timeout_s=10)
    # Another session to that host is told at once, not sent to shell_integrate again.
    await t("session_create", name="s2", env={"HOME": plain_home(home)})
    out = await t("run", session="s2", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    assert "shell_integrate found no integration installed on remote-fake at " in out, out
    assert "`shell_integrate` session=" not in out, out
    await t("run", session="s2", input="exit", timeout_s=10)
    # ...also when the first note comes with a command still running, and
    # a close names that command.
    await t("session_create", name="s3", command=f"{bin_dir}/ssh remote-fake", env={"HOME": plain_home(home)})
    out = await t("run", session="s3", input="sleep 30", timeout_s=2)
    assert "shell_integrate found no integration installed on remote-fake" in out, out
    out = await t("session_close", session="s3")
    assert "it ended what was running: `$ sleep 30` (region 1)" in out, out
    out = await t("run", session="s", input="docent-terminal-mcp install --print remote-fake", timeout_s=10)
    assert "integration.bash  ->  ~/.local/share/docent-terminal-mcp/v" in out and "tar -C" in out, out
    assert not (remote_home / ".local").exists(), out  # --print: nothing done
    out = await t("run", session="s", input="docent-terminal-mcp install remote-fake", timeout_s=15)
    assert "integration installed in" in out and "exit 0" in out, out
    from docent_terminal_mcp.remote import INSTALL_DIR, INSTALLED, SHELL_DIR
    for name in INSTALLED:
        assert (remote_home / INSTALL_DIR / name).read_bytes() == (SHELL_DIR / name).read_bytes()
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    out = await t("shell_integrate", session="s")
    assert "shell integration active" in out, out
    out = await t("run", session="s", input="false", timeout_s=10)
    assert "exit 1" in out, out
    out = await t("read", session="s")
    assert " . ~/.local/share/docent-terminal-mcp/v" in out and "base64" not in out, out
    await t("run", session="s", input="exit", timeout_s=10)
    out = await t("run", session="s", input="docent-terminal-mcp install --remove remote-fake", timeout_s=15)
    assert "integration removed" in out and not (remote_home / ".local/share/docent-terminal-mcp").exists(), out


def test_the_installed_integration_has_a_version():
    # A host keeps the files `docent-terminal-mcp install` put there.  When the
    # server comes to need something older files lack (a new marker field),
    # bump remote.INTEGRATION_VERSION so they read as not installed; either
    # way, record the files' new hash here.
    import hashlib
    from docent_terminal_mcp.remote import INSTALLED, INTEGRATION_VERSION, SHELL_DIR
    digest = hashlib.sha256(b"".join((SHELL_DIR / name).read_bytes() for name in INSTALLED)).hexdigest()
    assert (INTEGRATION_VERSION, digest[:16]) == (1, "c1d22cbee6e49dee"), digest[:16]


async def test_a_session_says_which_host_it_is_on(env):
    # After `ssh host` nothing said where the next line would land, nor
    # that a dropped ssh (exit 255) put it back here.
    home, socket = env
    t = Term(home, socket)
    bin_dir = fake_ssh(home)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    out = await t("run", session="s", input=f"{bin_dir}/ssh -l drgn remote-fake", timeout_s=10)
    assert "[this session is on drgn@remote-fake (via ssh) -- the next line runs there; if the shell there is bash " \
           "or zsh, `shell_integrate` session=\"s\" makes its exit statuses exact" in out, out
    assert "or connect with `docent-terminal-mcp ssh drgn@remote-fake`, which brings it along]" in out, out
    out = await t("run", session="s", input="echo over-there", timeout_s=10)
    assert "· on drgn@remote-fake ·" in out.splitlines()[0] and "now on" not in out, out  # said once, then tagged
    out = await t("session_list")
    assert "on drgn@remote-fake (ssh)" in out, out
    out = await t("run", session="s", input="exit 255", timeout_s=10)
    assert "[this session is back on " in out and "the ssh to drgn@remote-fake ended" in out, out
    out = await t("run", session="s", input="echo home-again", timeout_s=10)
    assert " on " not in out.splitlines()[0] and "back on" not in out, out


async def test_wait_over_someone_elses_plain_ssh_stops_at_the_far_side(env):
    # A human's plain ssh is the only open command: `wait` sat out its whole
    # timeout although the cursor stood at "[sudo] password for alice: ".
    # The far side's question, or its shell back at its prompt, is the answer.
    home, socket = env
    t = Term(home, socket)
    bin_dir = fake_ssh(home)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    human_types(t, f"{bin_dir}/ssh remote-fake", "Enter")
    await asyncio.sleep(1.5)
    human_types(t, "read -sp '[sudo] password for alice: ' pw; echo; echo got-it", "Enter")
    await asyncio.sleep(0.5)
    began = time.monotonic()
    out = await t("wait", session="s", timeout_s=20)
    assert time.monotonic() - began < 12, out
    assert "is waiting for input at '[sudo] password for alice:" in out and "(in a human's ssh)" in out, out
    assert "is a password or passphrase prompt" in out, out
    human_types(t, "hunter2", "Enter")
    await asyncio.sleep(0.5)
    began = time.monotonic()
    out = await t("wait", session="s", timeout_s=20)
    assert time.monotonic() - began < 12, out
    assert "got-it" in out and "the shell at the far end of this ssh is at its prompt 'remote$'" in out, out


async def test_a_session_whose_program_is_ssh_says_where(env):
    # `session_create command="ssh host"`: the ssh is the session's own
    # process, which the chain stopped at.
    home, socket = env
    t = Term(home, socket)
    bin_dir = fake_ssh(home)
    out = await t("session_create", name="r", command=f"{bin_dir}/ssh remote-fake", env={"HOME": plain_home(home)})
    out = await t("session_list")
    assert "on remote-fake (ssh)" in out, out
    out = await t("run", session="r", input="echo over-there", timeout_s=10)
    assert "· on remote-fake ·" in out.splitlines()[-2] or "· on remote-fake ·" in out, out


@pytest.mark.parametrize("login_shell", ["bash", "zsh"])
async def test_ssh_subcommand_brings_the_integration_along(env, login_shell):
    # `docent-terminal-mcp ssh HOST`: the integration goes as ssh's remote command,
    # nothing typed at a prompt, nothing left on the host.  (A fake ssh runs
    # the remote command as a login shell elsewhere would.)
    if shutil.which(login_shell) is None:
        pytest.skip(f"needs {login_shell}")
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    remote_tmp = Path(home) / "remote-tmp"
    remote_tmp.mkdir()
    zhome = Path(home) / "zhome"
    zhome.mkdir()
    # (both shells take the real hostname from their login files: the
    # remote user's own files override it, as on another host)
    (zhome / ".zshrc").write_text("HOST=remote-fake\nPS1='%m%# '\n")
    (zhome / ".bash_profile").write_text("HOSTNAME=remote-fake\nPS1='\\h$ '\n")
    (bin_dir / "ssh").write_text(
        "#!/bin/bash\nenv -u PROMPT_COMMAND -u __DTMCP_BIN -u ZDOTDIR HOSTNAME=remote-fake HOST=remote-fake "
        f"TMPDIR={remote_tmp} SHELL=$(command -v {login_shell}) HOME={zhome} bash -c \"${{@: -1}}\"\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home), "DOCENT_TERMINAL_MCP_SSH": str(bin_dir / "ssh")})
    out = await t("run", session="s", input="docent-terminal-mcp ssh remote-fake", timeout_s=15)
    assert "is on remote-fake" in out and "shell_integrate" not in out, out  # integrated on arrival
    out = await t("run", session="s", input="echo there; false", timeout_s=10)
    assert "exit 1" in out and "· on remote-fake ·" in out.splitlines()[0], out
    if login_shell == "bash":
        assert list(remote_tmp.iterdir()) == [], list(remote_tmp.iterdir())  # removed once loaded
    out = await t("run", session="s", input="exit", timeout_s=10)
    assert "back on " in out, out
    assert list(remote_tmp.iterdir()) == [], list(remote_tmp.iterdir())


async def test_into_carries_the_integration_into_a_container(env):
    # `docent-terminal-mcp into COMMAND...`: COMMAND runs a program somewhere else
    # with a terminal (podman exec -it box, openshell sandbox exec ... --);
    # the bootstrap goes as that program.  (A stand-in runs it in a clean
    # environment under another host name.)
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    box_tmp = Path(home) / "box-tmp"
    box_tmp.mkdir()
    box_home = Path(home) / "box-home"
    box_home.mkdir()
    (box_home / ".bash_profile").write_text("HOSTNAME=box-fake\nPS1='\\h$ '\n")
    (bin_dir / "enter").write_text(
        "#!/bin/bash\n[ \"$1\" = exec ] && [ \"$2\" = -it ] && shift 3\n"
        f"exec env -u PROMPT_COMMAND -u __DTMCP_BIN -u ZDOTDIR HOSTNAME=box-fake TMPDIR={box_tmp} "
        f"SHELL=/bin/bash HOME={box_home} \"$@\"\n")
    (bin_dir / "enter").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    out = await t("run", session="s", input=f"docent-terminal-mcp into {bin_dir}/enter exec -it box", timeout_s=15)
    out = await t("run", session="s", input="echo inside; false", timeout_s=10)
    assert "exit 1" in out and "· on box-fake ·" in out.splitlines()[0], out
    assert list(box_tmp.iterdir()) == [], list(box_tmp.iterdir())  # removed once loaded
    out = await t("run", session="s", input="exit", timeout_s=10)
    out = await t("run", session="s", input="echo back; true", timeout_s=10)
    assert "exit 0" in out and " on " not in out.splitlines()[0], out


def test_container_clients_are_carriers_only_with_a_terminal_on_a_container():
    from docent_terminal_mcp.tmux import container_entry, remote_destination
    assert container_entry("podman exec -it box bash".split()) == "exec"
    assert container_entry("docker exec --interactive --tty box sh".split()) == "exec"
    assert container_entry("podman run --rm -ti --name scratch fedora bash".split()) == "run"
    assert container_entry("podman attach box".split()) == "attach"
    assert container_entry("podman build -t x .".split()) is None      # works here, in view
    assert container_entry("podman exec box make -j8".split()) is None  # no terminal: nothing to type into
    assert container_entry("podman run --rm fedora make".split()) is None
    assert remote_destination("podman exec -it -u root -w /src box bash".split()) == ("box", None)
    assert remote_destination("podman run -it --name scratch fedora".split()) == ("scratch", None)
    assert remote_destination("podman build .".split()) is None


async def test_a_session_whose_program_is_the_servers_own_command_finds_it(env):
    # session_create command="docent-terminal-mcp ssh host": "exec: docent-terminal-mcp:
    # not found" -- only sessions' shells had it on their PATH.
    t = Term(*env)
    out = await t("session_create", name="p", command="docent-terminal-mcp into --print true")
    out = await t("read", session="p")
    assert "<the script below>" in out and "not found" not in out, out


def test_an_openshell_command_line_names_its_sandbox():
    from docent_terminal_mcp.tmux import remote_destination
    assert remote_destination("openshell sandbox exec -n box --tty -- bash".split()) == ("box", None)
    assert remote_destination("openshell sandbox connect box".split()) == ("box", None)
    assert remote_destination("openshell sandbox list".split()) is None


async def test_a_remote_shells_markers_say_where_it_is_until_it_is_gone(env):
    # After `sudo -i` over ssh and shell_integrate, the markers say root@host;
    # leaving that shell for one without markers left "root@" on every
    # region.  (A shell whose markers claim another host and user fakes it.)
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME=remote-fake USER=fakeroot "
                                 "PS1='fakeroot@remote-fake# ' bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    install_integration(plain_home(home))
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    out = await t("shell_integrate", session="s")
    out += await t("run", session="s", input="echo marked", timeout_s=10)
    assert "now on fakeroot@remote-fake (its shell's markers say so)" in out and "· on fakeroot@remote-fake ·" in out, out
    out = await t("run", session="s", timeout_s=10,
                  input="env -u PROMPT_COMMAND PS1='plain@remote-fake$ ' bash --norc --noprofile -i")
    out = await t("run", session="s", input="echo plain", timeout_s=10)
    assert "fakeroot@" not in out.splitlines()[0] and "· on remote-fake ·" in out, out


async def test_a_silent_command_on_another_host_is_not_waiting_for_input(env):
    # `podman build -q` in a shell reached with `docent-terminal-mcp ssh`: silent
    # for minutes, and the local ssh sits in its input loop all the while --
    # "waiting for input (nothing printed for a while ...)" after 3 s of a
    # 900 s timeout.  (The stand-in for that ssh: a command that itself
    # polls the terminal while it works.)
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME=remote-fake PS1='remote-fake$ ' "
                                 "bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    install_integration(plain_home(home))
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    await t("shell_integrate", session="s")
    code = "import select, sys; select.select([sys.stdin], [], [], 6); print('built')"
    out = await t("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=15)
    assert "built" in out and "exit 0" in out and "waiting for input" not in out, out


async def test_a_pause_mid_line_on_another_host_is_waiting_only_at_a_question(env):
    # pytest -q over `docent-terminal-mcp ssh`: a slow test paused its line
    # of dots, and run and wait both came back "waiting for input" at
    # '........' within seconds of a 1500 s timeout -- and at a row of
    # exactly '...', which read as Python's continuation prompt.
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME=remote-fake PS1='remote-fake$ ' "
                                 "bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    install_integration(plain_home(home))
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    await t("shell_integrate", session="s")
    code = ("import select, sys; print('....', end='', flush=True); select.select([sys.stdin], [], [], 5); "
            "print('.. passed')")
    out = await t("run", session="s", input=f'{sys.executable} -c "{code}"', timeout_s=15)
    assert "passed" in out and "exit 0" in out and "waiting for input" not in out, out
    out = await t("run", session="s", input=f'{sys.executable} -c "{code.replace("....", "...")}"', timeout_s=15)
    assert "passed" in out and "exit 0" in out and "waiting for input" not in out, out
    out = await t("run", session="s", input="read -p 'Continue? [y/N] ' a; echo got=$a", timeout_s=15)
    assert "waiting for input" in out, out
    out = await t("run", session="s", input="y", timeout_s=10)
    assert "got=y" in out, out


async def test_a_command_running_over_ssh_is_not_typed_ahead_into(env):
    # B: after `until` matched, a run typed into the remote `sleep` (the
    # local ssh looked idle), and bash ran it later as a "human's" command.
    # C: a `wait` then returned the outer ssh region, re-sending its output.
    # A: shell_integrate's probe line came back as a human's input.
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME=remote-fake PS1='remote-fake$ ' "
                                 "bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    install_integration(plain_home(home))
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    await t("shell_integrate", session="s")
    out = await t("run", session="s", input="echo go; sleep 4; echo slept", until="^go", timeout_s=10)
    assert "until" in out, out
    with pytest.raises(AssertionError, match="still running and not waiting for input"):
        await t("run", session="s", input="echo second")
    out = await t("wait", session="s", timeout_s=10)
    assert "slept" in out and "exit 0" in out, out

    out = await t("wait", session="s", timeout_s=2)
    assert out.startswith("[no command running") and "ssh remote-fake" not in out, out
    out = await t("regions", session="s")
    assert "started by human" not in out, out
    # The run's region was made before its shell's marker named the host:
    # the host is saved too, so a restarted server still knows it.
    saved = {}
    for line in (Path(home) / "sessions/s/regions.jsonl").read_text().splitlines():
        r = json.loads(line)
        saved[r["id"]] = r
    assert any(r["command"].startswith("echo go") and r["host"] == "remote-fake" for r in saved.values()), saved


async def test_a_dropped_connection_does_not_show_seen_lines_again(served):
    # After the ssh dropped, `wait` showed the ssh region whole, repeating
    # the inner command's lines already shown (the inner one said so).
    home = served[0]
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME=remote-fake PS1='remote-fake$ ' "
                                 "bash --norc --noprofile -i\n")
    (bin_dir / "ssh").chmod(0o755)
    install_integration(plain_home(home))
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "s", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("run", {"session": "s", "input": f"{bin_dir}/ssh remote-fake", "timeout_s": 10})
        await ann.call_tool("shell_integrate", {"session": "s"})
        out, _ = await timed(ann.call_tool("run", {"session": "s", "input": "echo seen-once; sleep 30", "timeout_s": 2}))
        assert "seen-once" in out, out
        subprocess.run(["pkill", "-KILL", "-f", f"{bin_dir}/ssh remote-fake"])  # the connection drops
        out, _ = await timed(ann.call_tool("wait", {"session": "s", "timeout_s": 10}))
        assert "seen-once" not in out.splitlines() and "back on" in out, out  # (its label names it; its output not again)


async def test_a_pane_whose_shell_exits_reports_its_status(served):
    # A split pane running `...; exit 3`: `wait` said "exit status not seen"
    # while the server knew the pane had exited with 3 (a long-lived
    # server reaps the gone shell's command before anything asks the pane).
    home = served[0]
    stack, (ann, bob) = await two_agents(served)
    async with stack:
        await ann.call_tool("session_create", {"name": "p", "env": {"HOME": plain_home(home)}})
        await ann.call_tool("session_split", {"session": "p", "name": "q"})
        await ann.call_tool("run", {"session": "q", "input": "for i in 1 2; do echo B$i; sleep 1; done; exit 3",
                                    "timeout_s": 1})
        await asyncio.sleep(3)
        out, _ = await timed(ann.call_tool("wait", {"session": "q", "timeout_s": 5}))
        assert "exit 3" in out and "exit status not seen" not in out, out


async def test_a_second_hop_from_an_integrated_shell_is_seen(env):
    # ssh A, shell_integrate there, then `ssh B` from A: the local processes
    # show only the ssh to A; A's markers say it runs `ssh B`.
    home, socket = env
    t = Term(home, socket)
    bin_dir = Path(home) / "fakebin"
    bin_dir.mkdir()
    for name, host in (("ssh", "host-a"), ("ssh2", "host-b")):
        (bin_dir / name).write_text(f"#!/bin/bash\nenv -u PROMPT_COMMAND HOSTNAME={host} PS1='{host}$ ' "
                                    "bash --norc --noprofile -i\n")
        (bin_dir / name).chmod(0o755)
    (bin_dir / "hop").mkdir()
    (bin_dir / "hop" / "ssh").symlink_to(bin_dir / "ssh2")  # a second "ssh", named so
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    install_integration(plain_home(home))
    await t("run", session="s", input=f"{bin_dir}/ssh host-a", timeout_s=10)
    await t("shell_integrate", session="s")
    out = await t("run", session="s", input=f"{bin_dir}/hop/ssh host-b", timeout_s=10)
    assert "[this session is now on host-b (via ssh)" in out and "shell_integrate" in out, out
    out = await t("run", session="s", input="echo on-b", timeout_s=10)
    assert "· on host-b ·" in out.splitlines()[0], out
    out = await t("run", session="s", input="exit", timeout_s=10)
    assert "now on host-a" in out, out  # back on A: its markers again


@needs("zsh")
async def test_shell_integrate_integrates_zsh(term):
    await term("run", session="s", input="zsh -f", timeout_s=10)
    out = await term("shell_integrate", session="s", timeout_s=10)
    assert "shell integration active" in out, out
    out = await term("run", session="s", input="echo in-zsh; false", timeout_s=10)
    assert "exit 1" in out and "in-zsh" in out, out
    out = await term("run", session="s", input="true | false | true", timeout_s=10)
    assert "pipestatus 0 1 0" in out, out
    await term("run", session="s", input="exit", timeout_s=10)


@needs("fish")
async def test_shell_integrate_names_a_shell_it_cannot_integrate(term):
    await term("run", session="s", input="fish --no-config", timeout_s=10)
    out = await term("shell_integrate", session="s", force=True, timeout_s=10)
    assert "not bash or zsh (it is fish)" in out, out
    await term("send", session="s", text="exit\n")


@needs("zsh")
async def test_session_create_zsh_is_integrated(env):
    home, socket = env
    t = Term(home, socket)
    h = Path(home) / "zhome"
    h.mkdir()
    (h / ".zshrc").write_text("PS1='%n@%m %# '\nHISTFILE=$ZDOTDIR/.zhist\n")
    out = await t("session_create", name="z", command="zsh", env={"HOME": str(h)})
    assert "started (zsh with shell integration)" in out, out
    out = await t("run", session="z", input="echo hi; false", timeout_s=10)
    assert "exit 1" in out and out.splitlines()[-1] == "hi", out
    out = await t("run", session="z", input="for i in 1 2\ndo echo n$i\ndone", timeout_s=10)
    assert "exit 0" in out and "n2" in out, out
    out = await t("run", session="z", input="print -r -- $HISTFILE; zsh -c 'echo nested'", timeout_s=10)
    assert f"{h}/.zhist" in out, out  # the user's own ZDOTDIR while their .zshrc ran
    out = await t("run", session="z", input="zsh -i", timeout_s=10)  # a nested zsh is integrated too
    out = await t("run", session="z", input="false", timeout_s=10)
    assert "exit 1" in out, out


def ubuntu_global_compinit() -> bool:
    """Does /etc/zsh/zshrc run compinit here (Ubuntu's, unless skipped)?"""
    try:
        return ("skip_global_compinit" in Path("/etc/zsh/zshrc").read_text()
                and re.search(r"^ID.*=.*ubuntu", Path("/etc/os-release").read_text(), re.M) is not None)
    except OSError:
        return False


@needs("zsh")
@pytest.mark.skipif(not ubuntu_global_compinit(), reason="needs Ubuntu's /etc/zsh/zshrc (its compinit)")
async def test_ubuntus_compinit_dumps_in_the_users_home_not_the_package(env):
    # /etc/zsh/zshrc's compinit wrote $ZDOTDIR/.zcompdump -- with ZDOTDIR
    # the package's shell/zsh, an untracked file in the checkout.
    from docent_terminal_mcp.remote import SHELL_DIR
    dump = SHELL_DIR / "zsh" / ".zcompdump"
    dump.unlink(missing_ok=True)
    home, socket = env
    t = Term(home, socket)
    h = Path(home) / "zhome"
    h.mkdir()
    (h / ".zshrc").write_text("PS1='%n@%m %# '\n")
    await t("session_create", name="z", command="zsh", env={"HOME": str(h)})
    out = await t("run", session="z", timeout_s=10,
                  input="print -r -- compdef=$+functions[compdef] skip=${skip_global_compinit-unset}")
    assert out.splitlines()[-1] == "compdef=1 skip=unset", out
    assert (h / ".zcompdump").exists() and not dump.exists()


@needs("gdb")
async def test_a_gdb_piped_to_tail_is_diagnosed_at_once(term):
    # tail blocked on the pipe made the job "busy": the note for a gdb
    # waiting at a prompt nobody can see came only after a ctrl+c killed tail.
    out = await term("run", session="s", timeout_s=6,
                     input="gdb -q -nx -iex 'set debuginfod enabled off' 2>&1 | tail -5")
    assert "gdb's output goes to pipe:[" in out and "waiting for input" in out, out
    # ... and what to do instead (a model told only what was wrong went to
    # -batch and output files): the same gdb, live, ready to copy.
    assert "start it again live, with no pipe or redirect -- `gdb -q -nx -iex 'set debuginfod enabled off'`" in out, out
    assert "-batch or an output file would make every question a fresh start" in out, out
    out = await term("run", session="s", input="print 1", timeout_s=6)
    assert "probably at a prompt it printed earlier" not in out, out  # it never changed hands
    await term("send", session="s", signal="TERM")


async def test_a_gdb_redirected_to_a_file_is_offered_the_live_line(term):
    out = await term("run", session="s", timeout_s=6, input=f"gdb -q -nx > {term.home}/g.out 2>&1")
    assert "not the terminal, and it is waiting for input" in out, out
    assert "with no pipe or redirect -- `gdb -q -nx`: the region keeps all its output" in out, out
    await term("send", session="s", signal="TERM")


async def test_a_busy_job_redirected_to_a_file_gets_no_prompt_warning(term):
    """A build or testsuite redirected to a log is simply running: the note
    for a program waiting at an unseen prompt ("`wait` would wait for
    nothing") sent a model back to polling from Bash."""
    out = await term("run", session="s", input=f"sleep 4 > {term.home}/log.txt", timeout_s=1)
    assert "still running" in out and "waiting for input" not in out, out
    # ... but where its output goes, and how far it got (four 25-minute
    # waits on `make check > file` once showed nothing).
    assert f"[its output goes to the file {term.home}/log.txt, not the terminal: 0 bytes so far, last written" in out, out
    out = await term("wait", session="s", timeout_s=10)
    assert "exit 0" in out, out


async def test_session_list_does_not_ask_each_pane(env, monkeypatch):
    # A listing asked tmux and ps about every live pane on its own: 1.1 s for
    # 20 live sessions.  Now one call each, however many live.
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        for i in range(6):
            await c.call_tool("session_create", {"name": f"l{i}", "cwd": str(home)})
        await c.call_tool("session_list", {})
        spawned = []
        real = asyncio.create_subprocess_exec

        async def counting(*argv, **kw):
            spawned.append(argv[0])
            return await real(*argv, **kw)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", counting)
        r = await c.call_tool("session_list", {})
        monkeypatch.setattr(asyncio, "create_subprocess_exec", real)
        assert r.content[0].text.count("live, running bash") == 6, r.content[0].text
        assert len(spawned) <= 8, spawned


async def test_logs_subcommand_against_the_shared_server(served):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    home, port, environ = served
    token = (home / "token").read_text().strip()
    headers = {"Authorization": f"Bearer {token}", "X-Terminal-Actor": "cli-test"}
    async with streamable_http_client(f"http://127.0.0.1:{port}/mcp",
                                      http_client=create_mcp_http_client(headers=headers)) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            for tool, args in (("session_create", {"name": "c", "env": {"HOME": plain_home(home)}}),
                               ("session_create", {"name": "keepalive", "env": {"HOME": plain_home(home)}}),
                               ("session_close", {"session": "c"})):
                r = await session.call_tool(tool, args)
                assert not r.is_error, r.content[0].text

    def logs(*args):
        p = subprocess.run([sys.executable, "-m", "docent_terminal_mcp.server", "logs", *args], env=environ,
                           capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout + p.stderr

    rc, out = logs("keep", "c")
    assert rc == 0 and "the logs of 'c' (started by cli-test) are kept" in out, out
    rc, out = logs("delete", "keepalive")
    assert rc != 0 and "is live" in out, out
    rc, out = logs("due")
    assert rc == 0 and out.startswith("0 sessions due for deletion (closed logs are kept 3 days)"), out
    rc, out = logs("delete", "c")
    assert rc == 0 and "were deleted" in out and not (home / "sessions" / "c").exists(), out
    # `NAME close`: a person's close; --force even when someone else's
    # command runs there (here cli-test's, not the CLI user's).
    p = subprocess.run([sys.executable, "-m", "docent_terminal_mcp.server", "close", "--force", "keepalive"],
                       env=environ, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and "closed 'keepalive' (started by cli-test, not you)" in p.stdout, p.stdout + p.stderr


async def test_input_into_a_connection_that_dropped_before_wait_is_finished(env):
    # Killing the sshd of a plain ssh session on a remote host, then calling
    # `wait`: the remote command's region read "ready" (the local prompt
    # after the dropped ssh), where a wait already running said "finished".
    home, socket = env
    t = Term(home, socket)
    bin_dir = fake_ssh(home)
    await t("session_create", name="s", env={"HOME": plain_home(home)})
    await t("run", session="s", input=f"{bin_dir}/ssh remote-fake", timeout_s=10)
    out = await t("run", session="s", input="sleep 600", timeout_s=1)
    assert "timeout" in out.splitlines()[0], out
    shell = json.loads(tmux_cmd(t, "display-message", "-p", "-t", "=s:", "{\"pid\": #{pane_pid}}"))["pid"]
    subprocess.run(["pkill", "-KILL", "-P", str(shell)], check=True)  # the fake ssh, as a dropped link ends it
    await asyncio.sleep(1)
    out = await t("wait", session="s", timeout_s=10)
    assert "sleep 600 · on remote-fake · finished" in out, out


async def test_close_leaves_a_terminal_someone_is_using(env):
    # Not closed: a session a person is attached to (anyone's, own too), and
    # another's session while a command runs there.  Closed: one's own busy
    # session, another's idle one; anything with force (a front end's Close,
    # asked for by the person).  The refusal does not offer force.
    t = Term(*env)
    home = plain_home(env[0])
    for name in ("mine", "theirs", "idle", "watched"):
        await t("session_create", name=name, env={"HOME": home})
    for name in ("theirs", "idle"):
        path = env[0] / "sessions" / name / "meta.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "started_by": "viewer"}))
    for name in ("mine", "theirs"):
        await t("run", session=name, input="sleep 600", timeout_s=1)
    with pytest.raises(AssertionError) as e:
        await t("session_close", session="theirs")
    assert "not closed: session 'theirs' is viewer's and `sleep 600` (region 1) is running there" in str(e.value)
    assert "force" not in str(e.value)
    assert "closed 'idle' (started by viewer, not you)" in await t("session_close", session="idle")
    assert "it ended what was running: `sleep 600`" in await t("session_close", session="mine")
    client = Client_(env[1], "watched", 80, 24)
    try:
        async def attached():
            return "1 human attached" in await t("session_list")
        assert await wait_for(attached)
        with pytest.raises(AssertionError, match="session 'watched' has 1 person attached"):
            await t("session_close", session="watched")
        # The person may close it (a front end's Close, `NAME close --force`):
        # force is the endpoint's, not the tool's.
        mgr = build_server(*env).terminal_manager
        assert "closed 'watched'" in await mgr.close_sessions("watched", force=True)
    finally:
        client.kill()
    assert "closed 'theirs'" in await build_server(*env).terminal_manager.close_sessions("theirs", force=True)


async def test_wait_until(term):  # symmetry with run until=
    # A pattern in output not yet shown returns at once, the command still
    # running; one that never matches is a plain wait; output shown before
    # does not count.
    await term("run", session="s", input="echo early-READY; sleep 2; echo late-READY; sleep 30", timeout_s=1)
    began = time.monotonic()
    out = await term("wait", session="s", until="READY", timeout_s=20)
    assert 3 > time.monotonic() - began > 0.5, out  # not the early one, shown already
    assert "late-READY" in out and "`until` matched" in out, out
    await term("send", session="s", keys=["ctrl+c"])
    await term("run", session="s", input="sleep 2; echo done", timeout_s=1)
    out = await term("wait", session="s", until="NEVER-PRINTED", timeout_s=20)
    assert "exit 0" in out.splitlines()[0] and "done" in out and "`until` matched" not in out, out


async def test_wait_until_on_program_input(term):
    await term("run", session="s", input=f"{sys.executable} -q", timeout_s=10)
    out = await term("run", session="s", timeout_s=1,
                     input="import time; [print(f'n{i}', flush=True) or time.sleep(0.2) for i in range(40)]")
    assert "timeout" in out.splitlines()[0], out
    out = await term("wait", session="s", until="n25", timeout_s=20)
    assert "n25" in out and "n39" not in out and "until" in out.splitlines()[0], out
    out = await term("wait", session="s", timeout_s=20)
    assert "ready" in out.splitlines()[0] and "n39" in out, out
    await term("send", session="s", keys=["ctrl+d"])


@needs("/proc syscall")
async def test_an_ssh_with_its_input_piped_is_not_a_remote_shell(term):
    # A script running `ssh host cmd` with its input piped (git, rsync, a
    # test harness) carries no terminal; taken for an interactive ssh, its
    # quiet poll read "waiting for input" -- with the new pipe warning's
    # advice to end it and start it again.
    bin_dir = term.home / "pipebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").symlink_to(os.path.realpath(sys.executable))  # a process named ssh, polling a pipe
    out = await term("run", session="s", timeout_s=6, input=(
        f"{bin_dir}/ssh -c 'import os, select; r, w = os.pipe(); select.select([r], [], [], 60)' < /dev/null | cat"))
    assert "waiting for input" not in out and "still running" in out, out
    await term("send", session="s", keys=["ctrl+c"])


async def test_a_prompt_followed_by_output_is_back_at_the_prompt(term):
    # gdb with `set debug infrun 1` prints "(gdb) [infrun] ...: exit" after
    # its prompt: no prompt ends a line, and each `step` sat out its 60 s.
    (term.home / "fakegdb.sh").write_text(
        "printf '(gdb) '\n"
        "while read -r l; do echo \"did $l\"; printf '(gdb) [infrun] fetch_inferior_event: exit\\n'; done\n")
    out = await term("run", session="s", input=f"bash {term.home}/fakegdb.sh", timeout_s=10)
    assert "waiting for input at '(gdb)'" in out, out
    began = time.monotonic()
    out = await term("run", session="s", input="step", timeout_s=30)
    assert time.monotonic() - began < 20, out
    assert "(gdb) step · ready" in out.splitlines()[0] and "did step" in out, out
    out = await term("run", session="s", input="next", timeout_s=30)
    assert "· ready" in out.splitlines()[0] and "did next" in out, out
    await term("send", session="s", keys=["ctrl+d"])


FAKE_NESTED_GDB = r"""p='(top-gdb) '
printf '%s' "$p"
while read -r l; do
  case "$l" in
    continue) echo 'Continuing.'; echo '28        /* foo:3 */'; p='(gdb) ' ;;
    step) echo 'Thread 1 "gdb" hit Breakpoint 1, print_stop_location ()'; p='(top-gdb) ' ;;
    'p boom') echo 'Thread 1 "gdb" received signal SIGABRT, Aborted.' ;;
    *) echo "\$1 = $l" ;;
  esac
  printf '%s' "$p"
done
"""


async def test_a_reply_says_when_the_prompt_changed(term):
    # gdb under gdb: typed at (top-gdb), the inner gdb's (gdb) came back --
    # replies show no prompt line, so which one reads the next line was a
    # guess.  And a line that ends in a signal stops the lines after it.
    (term.home / "nested.sh").write_text(FAKE_NESTED_GDB)
    await term("run", session="s", input=f"bash {term.home}/nested.sh", timeout_s=10)
    out = await term("run", session="s", input="continue", timeout_s=20)
    assert "(top-gdb) continue · ready" in out and ("[now at a different prompt: '(gdb)', not '(top-gdb)' -- "
                                                    "the next line goes to the program at '(gdb)']") in out, out
    out = await term("run", session="s", input="step", timeout_s=20)
    assert "(gdb) step · ready" in out and "now at a different prompt: '(top-gdb)', not '(gdb)'" in out, out
    out = await term("run", session="s", input="p 1", timeout_s=20)
    assert "different prompt" not in out, out  # same prompt: nothing to say
    out = await term("run", session="s", input="p 2\np boom\np 3\np 4", timeout_s=20)
    assert "$1 = p 2" in out and "received signal SIGABRT" in out and "$1 = p 3" not in out, out
    assert ("[stopped: 2 more lines not sent -- this one ended with 'Thread 1 \"gdb\" received signal SIGABRT': "
            "the program may be somewhere else now; look before going on]") in out, out
    await term("send", session="s", keys=["ctrl+d"])


async def test_show_lists_every_match_in_order_when_they_fit(term):
    # An infrun trace's repeated lines were folded ("15 distinct", "[x2, last
    # at line N]"): in a trace the repeats and their order are the point.
    out = await term("run", session="s", show="stop_pc|range", input=(
        "printf 'stop_pc=1\\nnoise\\nrange A\\nstop_pc=2\\nrange A\\nnoise\\nstop_pc=3\\n'"))
    body = [line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line]
    assert body[:5] == ["stop_pc=1", "range A", "stop_pc=2", "range A", "stop_pc=3"], out
    assert "distinct" not in out and "[x2" not in out, out


@needs("gdb", "gcc")
async def test_gdb_with_debug_infrun_comes_back_ready(term):
    src = term.home / "s.c"
    src.write_text("int g;\nint main(void) {\n  g = 1;\n  g = 2;\n  g = 3;\n  return g;\n}\n")
    subprocess.run(["gcc", "-g", "-O0", "-o", str(term.home / "s"), str(src)], check=True)
    await term("run", session="s", timeout_s=30,
               input=f"gdb -q -nx -iex 'set debuginfod enabled off' -ex 'break main' -ex run {term.home}/s")
    await term("run", session="s", input="set debug infrun 1", timeout_s=20)
    began = time.monotonic()
    out = await term("run", session="s", input="next", timeout_s=40)
    assert time.monotonic() - began < 15 and "· ready" in out.splitlines()[0], out
    assert "[infrun]" in out, out
    await term("run", session="s", input="set debug infrun 0", timeout_s=20)
    await term("run", session="s", input="kill", timeout_s=20)
    await term("send", session="s", keys=["ctrl+d"])



async def test_a_long_job_says_how_long_and_what_runs_under_it(term):
    # "still running" told nothing of progress: a wedged suite looked like a
    # slow one; the model fell back to `ps -eo etime` in Bash.
    out = await term("run", session="s", timeout_s=3, input="sleep 100")
    assert re.search(r"\[still running \([34] s so far\)", out), out
    await term("send", session="s", keys=["ctrl+c"])
    out = await term("run", session="s", timeout_s=3, input="bash -c 'sleep 100; true'")
    assert "under it: 2 processes; the newest started" in out and ": sleep 100 (pid " in out, out
    await term("send", session="s", keys=["ctrl+c"])


async def test_a_job_put_in_the_background_gets_the_foreground_way(term):
    out = await term("run", session="s", input="sleep 30 &")
    assert "[note: a job started with `&` cannot be waited for" in out and "timeout_s up to 1500" in out, out
    out = await term("wait", session="s", timeout_s=2)
    assert "Next time run it without `&`" in out, out
    out = await term("run", session="s", input="sleep 31 &")
    assert "[note:" not in out, out  # once per agent


async def test_habit_notes_are_once_per_agent_not_per_session(env):
    home, socket = env
    async with Client(build_server(home, socket)) as c:
        async def call(tool, **args):
            r = await c.call_tool(tool, args)
            return r.content[0].text
        for name in ("a", "b"):
            await call("session_create", name=name, cwd=str(home), env={"HOME": plain_home(home)})
        out = await call("run", session="a", input="seq 1 5 | tail -2")
        assert "with `| tail`/`| head` only those lines reach the log" in out, out
        out = await call("run", session="b", input="seq 1 5 | tail -2")
        assert "[note:" not in out, out


@needs("gdb")
async def test_gdb_under_gdb_names_which_gdb_reads_the_next_line(term):
    # An inner gdb at its own "(gdb)" was taken for the outer one: prompt
    # text cannot tell them apart; the processes can.
    gdb = shutil.which("gdb")
    out = await term("run", session="s", timeout_s=60, input=(
        f"gdb -q -nx -iex 'set debuginfod enabled off' -ex 'set prompt (top-gdb) ' -ex run "
        f"--args {gdb} -q -nx -iex 'set debuginfod enabled off'"))
    assert "waiting for input at '(gdb)'" in out, out
    real = os.path.realpath(gdb)
    assert f"[the program at this prompt is gdb (pid " in out and f", {real}), running under gdb (pid " in out, out
    assert "give the outer one a prompt of its own" not in out, out  # it has one: (top-gdb)
    out = await term("run", session="s", input="print 6*7", timeout_s=20)
    assert "$1 = 42" in out and "running under gdb" in out, out
    await term("send", session="s", keys=["ctrl+c"])
    await term("run", session="s", input="kill", timeout_s=20)
    await term("run", session="s", input="quit", timeout_s=20)



@needs("gdb")
async def test_two_gdbs_both_at_gdb_get_the_set_prompt_hint(term):
    gdb = shutil.which("gdb")
    out = await term("run", session="s", timeout_s=60, input=(
        f"gdb -q -nx -iex 'set debuginfod enabled off' -ex run --args {gdb} -q -nx -iex 'set debuginfod enabled off'"))
    assert "running under gdb (pid " in out, out
    assert ("give the outer one a prompt of its own -- `set prompt (top-gdb) ` at its prompt, or -ex "
            "'set prompt (top-gdb) ' when starting it") in out, out
    await term("send", session="s", keys=["ctrl+c"])
    await term("run", session="s", input="kill", timeout_s=20)
    await term("run", session="s", input="quit", timeout_s=20)


def test_a_binary_is_named_only_when_verified(tmp_path):  # accuracy first
    from docent_terminal_mcp.server import verified_exe
    assert verified_exe(os.getpid()) == os.path.realpath(f"/proc/{os.getpid()}/exe")
    assert verified_exe(1) is None  # another user's (root's)
    if shutil.which("bwrap") is None:
        return
    # In a sandbox's own filesystem view the path means something else.
    p = subprocess.Popen(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "sleep", "30"])
    try:
        for _ in range(50):
            kids = subprocess.run(["pgrep", "-x", "sleep", "-P", str(p.pid)], capture_output=True, text=True).stdout.split()
            if kids:
                break
            time.sleep(0.1)
        assert kids, "no sleep under bwrap"
        assert verified_exe(int(kids[0])) is None
    finally:
        p.kill()
        p.wait()


async def test_an_ssh_running_one_command_is_not_a_login(term):
    # `ssh -o BatchMode=yes host 'set -e; ...'` read "this session is on
    # host -- the next line runs there" and "waiting for input" while the
    # remote command worked: one command, not a login.
    bin_dir = term.home / "oneshot"
    bin_dir.mkdir()
    (bin_dir / "ssh").symlink_to(os.path.realpath(sys.executable))  # polls its stdin, the terminal, as ssh does
    out = await term("run", session="s", timeout_s=6, input=(
        f"{bin_dir}/ssh -c 'import select, sys; select.select([sys.stdin], [], [], 60)' remote-fake 'make install'"))
    assert "still running" in out and "waiting for input" not in out, out
    assert "this session is on" not in out, out
    await term("send", session="s", keys=["ctrl+c"])



@needs("gdb", "gcc")
async def test_quit_answered_then_the_shell_gets_the_rest(term):
    # `quit` + a new command line: "Quit anyway?" stopped the batch, an
    # `expect` call typed y, a third call ran the line.  Now the reply offers
    # the one call, and after the answer ends gdb the shell gets the rest.
    src = term.home / "q.c"
    src.write_text("int main(void) { return 0; }\n")
    subprocess.run(["gcc", "-g", "-o", str(term.home / "q"), str(src)], check=True)
    await term("run", session="s", timeout_s=30,
               input=f"gdb -q -nx -iex 'set debuginfod enabled off' -ex 'break main' -ex run {term.home}/q")
    out = await term("run", session="s", input="quit\necho after-quit", timeout_s=20)
    assert "Quit anyway?" in out, out
    assert ("To answer and go on in one call: the `run` tool with input=\"y\\necho after-quit\"" in out), out
    assert "`set confirm off` at gdb's prompt stops such questions in this gdb" in out, out
    out = await term("run", session="s", input="y\necho after-quit", timeout_s=20)
    assert "[the lines after it went to the shell:]" in out and "after-quit" in out.split("went to the shell")[1], out
    assert "$ echo after-quit · exit 0" in out, out


@needs("gdb", "gcc")
async def test_an_answer_written_ahead_is_sent_to_its_question(term):
    # DeepSeek under codex sent "quit\ny": the y was held back, and the
    # reply said to send "y\ny" -- the second y for the shell.
    src = term.home / "q.c"
    src.write_text("int main(void) { return 0; }\n")
    subprocess.run(["gcc", "-g", "-o", str(term.home / "q"), str(src)], check=True)
    await term("run", session="s", timeout_s=30,
               input=f"gdb -q -nx -iex 'set debuginfod enabled off' -ex 'break main' -ex run {term.home}/q")
    out = await term("run", session="s", input="quit\ny\necho after-quit", timeout_s=20)
    assert "not sent" not in out, out
    assert "Quit anyway? (y or n) y · finished" in out, out
    assert "$ echo after-quit · exit 0" in out.split("went to the shell")[1], out


async def test_an_argument_the_tool_does_not_have_is_refused(term):
    # codex's models used their own shell tool's names: session_create
    # cmd="gdb -q ./avg" started bash, and nothing said so.
    with pytest.raises(AssertionError) as e:
        await term("session_create", name="g", cmd="sleep 100")
    assert "session_create has no argument 'cmd' (did you mean 'command'?); nothing was done." in str(e.value)
    assert "Its arguments: name, command, cwd," in str(e.value)
    with pytest.raises(AssertionError) as e:
        await term("run", session="s", input="echo hi", yield_time_ms=1000, max_output_tokens=2000)
    assert ("run has no arguments 'yield_time_ms' (another shell tool's argument: here timeout_s, in seconds), "
            "'max_output_tokens' (another shell tool's argument: here max_lines); nothing was done.") in str(e.value)
    with pytest.raises(AssertionError) as e:
        await term("run", session="s", input="ls", workdir="/tmp")
    assert "'workdir' (a session keeps its directory: `cd` there, or session_create cwd=)" in str(e.value)
    # run's and send's arguments mixed up (an eval's Claude Code x Qwen:
    # send input="q", and the list of send's arguments was not enough).
    with pytest.raises(AssertionError) as e:
        await term("send", session="s", input="q")
    assert "send has no argument 'input' (that is `run`'s: here text=, typed with no Enter" in str(e.value)
    with pytest.raises(AssertionError) as e:
        await term("run", session="s", text="ls")
    assert "run has no argument 'text' (that is `send`'s: here input=" in str(e.value)
    out = await term("session_list")
    assert "\ng\t" not in "\n" + out, out


async def test_blanks_around_a_session_name_are_not_part_of_it(term):
    # qwen sent session="\nvim-ini" for its session vim-ini.
    out = await term("run", session="\ns ", input="echo hi")
    assert "exit 0" in out and "hi" in out, out
    # Tool-call markup after it (an eval's qwen: 'tests\n</parameterparameter>') is asked about.
    with pytest.raises(AssertionError, match=r"no session named 's\\n</parameterparameter>' -- did you mean 's'\?"):
        await term("run", session="s\n</parameterparameter>", input="echo hi")
    # ... and before it (an eval's qwen under codex: '=build').
    with pytest.raises(AssertionError, match=r"no session named '=s' -- did you mean 's'\? \(what comes before"):
        await term("run", session="=s", input="echo hi")


async def test_interrupt_ends_the_command_holding_the_session_then_types_the_line(term):
    # adjutant's glm: a `tail -f` held the session, and the busy refusal's
    # remedy (send ctrl+c, run again) took three calls -- three approval
    # cards in ask-first mode.
    await term("run", session="s", input="tail -f /dev/null", timeout_s=1)
    with pytest.raises(AssertionError, match="interrupt=true interrupts it and then types the line"):
        await term("run", session="s", input="echo hi")
    out = await term("run", session="s", input="echo hi", interrupt=True)
    assert "[interrupted with ctrl+c: region" in out and "tail -f /dev/null" in out and "130" in out, out
    assert out.endswith("\nhi") and "exit 0" in out and out.count("tail -f") == 1, out
    # Nothing holds the session: nothing is interrupted.
    out = await term("run", session="s", input="echo again", interrupt=True)
    assert "interrupted" not in out and "again" in out, out
    # A job that ignores ctrl+c: nothing is typed.
    await term("run", session="s", input="sh -c 'trap \"\" INT; sleep 30'", timeout_s=1)
    with pytest.raises(AssertionError, match="no prompt came back within 5 s .*so nothing was typed"):
        await term("run", session="s", input="echo never", interrupt=True)
    await term("send", session="s", signal="TERM")


async def test_a_session_name_mistyped_is_asked_about(term):
    # An eval's qwen sent 'gdbgdb' for its session gdbdbg.
    await term("session_create", name="gdbdbg")
    with pytest.raises(AssertionError, match=r"no session named 'gdbgdb' -- did you mean 'gdbdbg'\? the live sessions"):
        await term("send", session="gdbgdb", keys=["enter"])
    with pytest.raises(AssertionError, match=r"no session named 'zzz': the live sessions"):
        await term("send", session="zzz", keys=["enter"])
    await term("session_close", session="gdbdbg")


async def test_a_job_whose_output_reaches_nothing_says_so(term):
    # `make check 2>&1 | tee F | tail -3000 > /dev/null`: nothing reached the
    # terminal, and `until` timed out with no word why.
    out = await term("run", session="s", timeout_s=3, until="Running",
                     input="sh -c 'echo Running x; sleep 30' | tail -3 > /dev/null")
    assert "`until` 'Running' did not match within 3 s" in out, out
    assert ("nothing it prints reaches the terminal: tail's output goes to /dev/null -- `until`, `show` and this "
            "session's log see none of it") in out, out
    await term("send", session="s", keys=["ctrl+c"])


async def test_a_moment_that_looks_like_an_input_loop_is_not_waiting(term, monkeypatch):
    # pytest's dots during a slow test: its epoll_wait, seen once with its
    # thread between calls, read as an input loop, and run returned
    # "waiting for input" at a row of dots, 12 s into a 19-minute run.
    from docent_terminal_mcp import session as session_module
    real, n = session_module.foreground_state, 0

    async def flickering(tty, shell_pid=0):
        nonlocal n
        n += 1
        state = await real(tty, shell_pid)
        return "poll" if state == "busy" and n % 5 == 0 else state
    monkeypatch.setattr(session_module, "foreground_state", flickering)
    out = await term("run", session="s", timeout_s=6, input="printf '....'; sleep 30")
    assert "waiting for input" not in out and "still running (6 s" in out, out
    await term("send", session="s", keys=["ctrl+c"])


async def test_a_job_printing_through_a_saved_copy_of_the_terminal_reaches_it(term):
    # pytest's capture: fd 1 is an unlinked temporary file while a test runs,
    # and the dots go to the terminal through a copy of the original fd 1.
    # The reply said "nothing it prints reaches the terminal".
    script = term.home / "capture.py"
    script.write_text(
        "import os, sys, tempfile, time\n"
        "saved = os.dup(1)\n"
        "capture = tempfile.TemporaryFile()\n"
        "os.dup2(capture.fileno(), 1)\n"
        "while True:\n"
        "    os.write(saved, b'.'); time.sleep(0.5)\n")
    out = await term("run", session="s", timeout_s=3, input=f"{sys.executable} {script}")
    assert "still running" in out, out
    assert "reaches the terminal" not in out and "goes to" not in out, out
    await term("send", session="s", keys=["ctrl+c"])



async def test_env_path_is_in_effect_for_the_sessions_command(env):
    # tmux put its own PATH in place of env's: `psql` in env PATH's directory
    # was "not found".
    home, socket = env
    t = Term(home, socket)
    bin_dir = home / "only-here"
    bin_dir.mkdir()
    (bin_dir / "hello-env").write_text("#!/bin/sh\necho found-me; exec sleep 30\n")
    (bin_dir / "hello-env").chmod(0o755)
    out = await t("session_create", name="p", command="hello-env",
                  env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert "found-me" in out and "not found" not in out, out


BLOCKING_CONNECT = """import socket, time
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(0)
port = srv.getsockname()[1]
held = []
for i in range(4):
    c = socket.socket(); c.setblocking(False)
    try: c.connect(("127.0.0.1", port))
    except BlockingIOError: pass
    held.append(c)
time.sleep(0.5)
c = socket.socket()
print("connecting", flush=True)
c.connect(("127.0.0.1", port))
"""


@needs("/proc syscall")
async def test_a_process_blocked_in_connect_is_not_waiting_for_input(term):
    # connect() sleeps in wait_woken, as a terminal read does: a test wedged
    # on an unroutable host read "waiting for input (it is reading the
    # terminal)", and the skill then said "not a hang, wait again".
    (term.home / "bc.py").write_text(BLOCKING_CONNECT)
    out = await term("run", session="s", timeout_s=6, input=f"{sys.executable} {term.home}/bc.py")
    assert "connecting" in out and "waiting for input" not in out, out
    assert re.search(r"\[still running \(\d+ s so far\)", out), out
    await term("send", session="s", keys=["ctrl+c"])



async def test_an_input_that_finished_while_nobody_waited_is_closed(term):
    # A statement left blocked with a short timeout_s finished later (the
    # other session committed); the next input was typed -- and the first
    # stayed "timeout" for good: session_close said it "ended" it.
    await term("run", session="s", input=f"{sys.executable} -q", timeout_s=10)
    out = await term("run", session="s", input="import time; time.sleep(3); print('merged')", timeout_s=1)
    assert "timeout" in out.splitlines()[0], out
    await asyncio.sleep(4)
    out = await term("run", session="s", input="print(7)", timeout_s=10)
    assert "7" in out, out
    out = await term("regions", session="s")
    assert "time.sleep(3); print('merged') · ready" in out and "timeout" not in out, out
    out = await term("session_close", session="s")
    assert "time.sleep" not in out, out  # (the REPL itself is running, and is ended)



async def test_sigpipe_from_head_is_not_a_failed_stage(term):
    out = await term("run", session="s", input="seq 1 100000 | head -2")
    first = out.splitlines()[0]
    assert "exit 0 (stage 1: 141, SIGPIPE -- a later stage stopped reading)" in first and "but stage" not in first, out
    out = await term("run", session="s", input="false | true")
    assert "exit 0, but stage 1 exited 1" in out, out


async def test_last_lines_replies_with_the_tail_only(term):
    out = await term("run", session="s", input="seq 1 50", last_lines=5)
    body = out.splitlines()[1:]
    assert body[-5:] == ["46", "47", "48", "49", "50"], out
    assert "45 lines omitted" in out and "\n1\n" not in out, out
    # With show, it is how many last lines follow the matches.
    out = await term("run", session="s", input="seq 1 30", show="^7$", last_lines=1)
    assert "\t7" in out and "\t30" in out and "\t29" not in out, out


async def test_show_after_adds_the_lines_after_each_match(term):
    out = await term("run", session="s", input="printf 'a\\nERR x\\nctx1\\nctx2\\nb\\nERR y\\nc\\n'",
                     show="^ERR", show_after=1)
    lines = out.splitlines()
    i = next(n for n, l in enumerate(lines) if re.match(r"\d+:\tERR x$", l))
    assert re.match(r"\d+-\tctx1$", lines[i + 1]) and lines[i + 2] == "--", out
    assert re.match(r"\d+:\tERR y$", lines[i + 3]) and re.match(r"\d+-\tc$", lines[i + 4]), out
    assert not any(l.endswith("\tctx2") for l in lines), out
    with pytest.raises(AssertionError, match="give `show` too"):
        await term("run", session="s", input="true", show_after=2)


async def test_save_to_copies_the_output_to_a_file(term, env):
    home = env[0]
    out = await term("run", session="s", input="seq 1 3", save_to="out.txt")
    assert (home / "out.txt").read_text() == "1\n2\n3\n", out
    assert f"[output saved to {home / 'out.txt'}" in out, out
    # A command still running: written now, in full once a later call sees it end.
    path = home / "slow.txt"
    out = await term("run", session="s", input="echo start; sleep 2; echo end", timeout_s=0.5, save_to=str(path))
    assert path.read_text() == "start\n" and "kept up to date" in out, out
    await term("wait", session="s", timeout_s=10)
    assert path.read_text() == "start\nend\n"
    # A path that cannot be written is refused before anything is typed.
    with pytest.raises(AssertionError, match="no directory"):
        await term("run", session="s", input="echo never", save_to=str(home / "nowhere" / "x.txt"))
    assert "never" not in await term("read", session="s")


async def test_show_behind_a_tail_pipe_says_what_the_pipe_hid(term):
    out = await term("run", session="s", input="printf 'FAIL: a\\nok\\nok\\n' | tail -2", show="^FAIL")
    assert "`| tail -2` let only 2 lines" in out and "`last_lines=2` in place of the pipe" in out, out
    # Said each time, not once.
    out = await term("run", session="s", input="printf 'FAIL: b\\nok\\n' | tail -n 1", show="^FAIL")
    assert "let only 1 line of" in out, out
    # Not when show found something, nor without a pipe.
    out = await term("run", session="s", input="printf 'FAIL: c\\nok\\n' | tail -2", show="^FAIL")
    assert "matched nothing" not in out, out
    out = await term("run", session="s", input="printf 'ok\\n'", show="^FAIL")
    assert "matched nothing" not in out, out


@needs("python3")
async def test_a_deleted_working_directory_is_reported(term, env):
    (env[0] / "out").mkdir()
    await term("run", session="s", input="cd out")
    (env[0] / "out").rmdir()  # as the next `make check` removes a test's output directory
    out = await term("run", session="s", input="true")
    assert "bash (pid " in out and f"is in a directory that has been deleted since it went there: {env[0] / 'out'}" in out
    assert '`cd "$PWD"` in it goes to the new one' in out, out
    out = await term("run", session="s", input="true")
    assert "has been deleted" not in out, out  # once per process and directory
    # The program in front, too.
    (env[0] / "py").mkdir()
    await term("run", session="s", input=f"cd {env[0] / 'py'} && python3 -q", timeout_s=10)
    (env[0] / "py").rmdir()
    out = await term("run", session="s", input="print(6*7)")
    assert "42" in out and "python3 (pid " in out and "has been deleted" in out, out
    await term("run", session="s", input="exit()")


async def test_long_output_leads_with_its_summary_lines(term, env):
    # A parallel DejaGnu run: a summary per test file, FAILs among them, the
    # run's own summary at the end, make giving up.
    lines = []
    for i in range(300):
        lines += [f"Running t{i}.exp ...", "", "\t\t=== gdb Summary ===", "", "# of expected passes\t\t3"]
        if i in (40, 200):
            lines.append(f"FAIL: gdb.base/t{i}.exp: step {i}")
    lines += ["", "\t\t=== gdb Summary ===", "", "# of expected passes\t\t900", "# of unexpected failures\t2",
              "make[1]: *** [Makefile:162: check] Error 2", "make: *** [Makefile:2165: check] Error 2", "", "real 1m"]
    (env[0] / "check.out").write_text("\n".join(lines) + "\n")
    out = await term("run", session="s", input="cat check.out", max_lines=30)
    block = out.split("[the output itself:]")[0]
    assert "[summary lines picked out of the whole output" in block, out
    assert re.search(r"\d+\tFAIL: gdb.base/t40.exp: step 40\n\d+\tFAIL: gdb.base/t200.exp", block), out
    # The run's summary and make's failure are in the last lines shown: not again.
    assert "Summary" not in block and "make:" not in block, out
    assert re.search(r"# of expected passes\s+900", out) and "make: *** [Makefile:2165" in out, out
    # A long tail after them: then the block carries them, the last DejaGnu block only.
    (env[0] / "check.out").write_text("\n".join(lines + [f"cleanup {i}" for i in range(100)]) + "\n")
    out = await term("run", session="s", input="cat check.out", max_lines=30)
    block = out.split("[the output itself:]")[0]
    assert block.count("=== gdb Summary ===") == 1 and re.search(r"# of expected passes\s+900", block), out
    assert "make: *** [Makefile:2165" in block and "make[1]" not in block, out
    assert len(out.splitlines()) <= 30 + 4, out
    # Not when it all fits, nor when the caller chose its lines.
    assert "summary lines" not in await term("run", session="s", input="tail -5 check.out")
    assert "summary lines" not in await term("run", session="s", input="cat check.out", max_lines=30, last_lines=5)
    assert "summary lines" not in await term("run", session="s", input="cat check.out", max_lines=30, show="^FAIL")
    # Too many failures: the rest named for `search`.
    (env[0] / "many.out").write_text("".join(f"FAIL: t{i}\n" for i in range(60)) + "x\n" * 100)
    out = await term("run", session="s", input="cat many.out", max_lines=30)
    assert "[... 37 more failure lines: `search` pattern='^FAIL:' region=" in out, out  # t0-t2 are in the first lines


async def test_a_piped_test_run_says_what_the_reply_would_have_led_with(term):
    out = await term("run", session="s", input="printf 'FAIL: gdb.base/a.exp: x\\n# of expected passes 3\\n' | tail -1")
    assert "[without `| tail -1`, this reply would have led with the run's FAIL lines and totals" in out, out
    out = await term("run", session="s", input="printf 'FAIL: b\\n# of expected passes 3\\n' | tail -1")
    assert "this reply would have led with" in out, out  # each time
    assert "would have led" not in await term("run", session="s", input="seq 1 5 | tail -2")  # no test results
    out = await term("run", session="s", input="printf 'FAIL: c\\n' | tail -1", show="^FAIL")
    assert "would have led" not in out, out  # the caller chose its lines


async def test_fedora_gdbs_missing_rpms_hint_is_named_once(term):
    line = "printf 'Missing rpms, try: dnf --enablerepo=*debug* install glibc-debuginfo-2.42 zlib-debuginfo\\n'"
    out = await term("run", session="s", input=line)
    assert "`set rpm-suggestion enabled off` at the prompt of the gdb that printed it silences it" in out, out
    assert "rpm-suggestion" not in await term("run", session="s", input=line)  # once


async def test_a_piped_test_run_that_ends_in_a_wait_gets_the_note_too(term):
    cmd = "sleep 2; printf 'FAIL: gdb.base/a.exp: x\\n# of expected passes 3\\n' | tail -1"
    out = await term("run", session="s", input=cmd, timeout_s=0.5)
    assert "would have led" not in out, out  # still running
    out = await term("wait", session="s", timeout_s=10)
    assert "[without `| tail -1`, this reply would have led with the run's FAIL lines and totals" in out, out


@needs("make")
async def test_pipe_notes_find_a_test_run_mid_list_and_behind_show(term):
    # A test run by its command, its tail showing only make's lines, mid-list.
    (term.home / "Makefile").write_text("check:\n\t@echo 'FAIL: gdb.base/a.exp: x'\n\t@echo 'make: *** [Makefile:1: check] Error 2'\n")
    out = await term("run", session="s", input="true; make -s check 2>&1 | tail -1; echo after")
    assert "[without `| tail -1`, this reply would have led with the run's FAIL lines and totals" in out, out
    # show= that matched the totals but not the FAILs the pipe cut.
    out = await term("run", session="s", show="^# of",
                     input="printf 'FAIL: gdb.base/b.exp: y\\n# of expected passes 3\\n' | tail -1")
    assert "`show` saw only part of this test run in region" in out and "its FAIL lines among it" in out, out
    # An ordinary pipe with a matching show: nothing to say.
    out = await term("run", session="s", show="^4", input="seq 1 5 | tail -2")
    assert "saw only part" not in out and "would have led" not in out, out


async def test_cpython_regrtest_is_a_test_run_too(term):
    out_lines = ["Using random seed: 1"] + [f"0:00:0{i % 10} load avg: 0.4 [{i}/60/1] test_m{i} passed" for i in range(60)]
    out_lines += ["0:00:09 load avg: 4.4 mem: 27 MiB [60/60/1] test_dict failed (1 error)", "",
                  "== Tests result: FAILURE ==", "", "1 test failed:", "    test_dict", "", "59 tests OK."]
    out_lines += [f"cleanup {i}" for i in range(40)]
    (term.home / "regr.out").write_text("\n".join(out_lines) + "\n")
    out = await term("run", session="s", input="cat regr.out", max_lines=30)
    block = out.split("[the output itself:]")[0]
    assert "test_dict failed (1 error)" in block and "== Tests result: FAILURE ==" in block, out
    assert "1 test failed:" in block and "    test_dict" in block, out
    # `./python -m test` piped: known as a test run by its command.
    (term.home / "python").write_text("#!/bin/sh\ncat regr.out\n")
    (term.home / "python").chmod(0o755)
    out = await term("run", session="s", input="./python -m test test_dict 2>&1 | tail -2")
    assert "[without `| tail -2`, this reply would have led with the run's FAIL lines and totals" in out, out


@needs("make")
async def test_pipe_notes_only_for_a_test_command_piped_straight_into_tail(term):
    from docent_terminal_mcp.server import runs_tests
    assert runs_tests("PYTHONASYNCIODEBUG=1 /v/bin/pytest tests/a.py -x 2>&1")
    assert runs_tests("time make -j8 check TESTS=gdb.base/a.exp") and runs_tests("./python -m test test_x -v")
    assert runs_tests("timeout 600 env -i FOO=1 ctest -j4") and runs_tests("cargo test --release")
    assert not runs_tests("/v/bin/pip install --no-index pytest setuptools") and not runs_tests("make all-gdb")
    assert not runs_tests("python repro.py") and not runs_tests("git log --grep=pytest")
    (term.home / "Makefile").write_text("check:\n\t@echo 'FAIL: gdb.base/a.exp: x'\n\t@echo '# of expected passes 3'\n"
                                        "all:\n\t@echo built\n")
    # A grep between: the grep chose the lines -- no note.
    out = await term("run", session="s", input="make -s check 2>&1 | grep -E 'FAIL|# of' | head -5")
    assert "would have led" not in out, out
    # Two pipes, the second the test run's: the note names that one.
    out = await term("run", session="s", input="make -s all | tail -1; make -s check 2>&1 | tail -1")
    assert "[without `| tail -1`, this reply would have led" in out and out.count("would have led") == 1, out
    out = await term("run", session="s", input="echo pip install pytest | tail -1")
    assert "would have led" not in out, out


async def test_expect_says_a_pager_is_not_a_question(term):
    pager = "--Type <RET> for more, q to quit, c to continue without paging--"
    out = await term("expect", session="s", timeout_s=20, respond=[{"pattern": "Enable debuginfod", "line": "n"}],
                     input=f"printf 'Enable debuginfod? (y or [n]) '; read a; printf '#0 main\\n{pager}'; read b; echo done")
    assert out.startswith(f"[expect: output stopped at a pager, {pager!r} -- not a question; see below"), out
    assert "[gdb's pager, not a question: the `run` tool with input=\"c\"" in out, out
    await term("run", session="s", input="c")


async def test_a_prompt_followed_by_several_debug_lines_is_back_at_the_prompt(term):
    # Killing the program under `set debug infrun 1`: "(gdb) [infrun] ..." and
    # then five more indented [infrun] lines; the answer sat out its timeout.
    (term.home / "fakegdb.sh").write_text(
        "printf '(gdb) '\n"
        "while read -r l; do echo \"did $l\"; printf '(gdb) [infrun] fetch_inferior_event: enter\\n'\n"
        "  for x in 1 2 3 4 5; do printf '  [infrun] reset: reason=%s\\n' $x; done; done\n")
    await term("run", session="s", input=f"bash {term.home}/fakegdb.sh", timeout_s=10)
    began = time.monotonic()
    out = await term("run", session="s", input="kill", timeout_s=30)
    assert time.monotonic() - began < 20, out
    assert "· ready" in out.splitlines()[0] and "did kill" in out, out
    # Ordinary output after an old prompt line is not mistaken for it.
    (term.home / "fake2.sh").write_text(
        "printf '(gdb) '\nwhile read -r l; do printf '(gdb) noted\\n'; for x in 1 2 3 4 5; do echo plain $x; done; done\n")
    await term("send", session="s", keys=["ctrl+d"])
    await term("run", session="s", input=f"bash {term.home}/fake2.sh", timeout_s=10)
    out = await term("run", session="s", input="go", timeout_s=6)
    assert "· ready" not in out.splitlines()[0], out
    await term("send", session="s", keys=["ctrl+d"])


def test_ssh_and_into_carry_the_variables_the_session_names():
    # DOCENT_TERMINAL_MCP_SSH_ENV: an editor that created the session names
    # its variables, and `ssh`/`into` set them on the far side (values of
    # any kind, base64 in the script), the list itself included.
    import subprocess
    from docent_terminal_mcp.remote import bootstrap, carried
    environ = {"DOCENT_TERMINAL_MCP_SSH_ENV": "EDITOR_ID, ODD  bad-name UNSET",
               "EDITOR_ID": "review", "ODD": "it's \"quoted\" $HOME\nand two lines", "bad-name": "x"}
    carry = carried(environ)
    assert carry == {"DOCENT_TERMINAL_MCP_SSH_ENV": environ["DOCENT_TERMINAL_MCP_SSH_ENV"],
                     "EDITOR_ID": "review", "ODD": environ["ODD"]}
    assert carried({}) == {}
    script = bootstrap(None, carry=carry)
    assert "'" not in script
    lines = script.splitlines()
    setup = [line for line in lines if line.startswith("dec()") or "| dec); export" in line]
    out = subprocess.run(["sh", "-c", "\n".join(setup) + '\nprintf "%s|%s|%s" "$DOCENT_TERMINAL_MCP_SSH_ENV" "$EDITOR_ID" "$ODD"'],
                         capture_output=True, text=True, check=True).stdout
    assert out == "|".join([environ["DOCENT_TERMINAL_MCP_SSH_ENV"], "review", environ["ODD"]]), out


@pytest.mark.second_guess
async def test_a_habit_is_second_guessed_once_and_runs_when_sent_again(term):
    with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
        await term("run", session="s", input="seq 1 5 | tail -2")
    assert '`run` session="s" input="seq 1 5" last_lines=2' in str(e.value), e.value
    assert "sending the same line again runs it" in str(e.value)
    out = await term("regions", session="s")
    assert "seq 1 5" not in out, out  # nothing was typed
    out = await term("run", session="s", input="seq 1 5 | tail -2")  # the same line again: it runs
    assert out.splitlines()[1:3] == ["4", "5"], out
    assert "[note:" not in out, out  # the refusal said it: no note on top
    out = await term("run", session="s", input="seq 1 9 | tail -3")
    assert out.splitlines()[1:4] == ["7", "8", "9"], out  # declined: not asked again
    out = await term("run", session="s", input="seq 1 3", last_lines=2)
    assert out.splitlines()[-2:] == ["2", "3"], out


@pytest.mark.second_guess
async def test_each_habit_gets_its_own_second_guess(term):
    async def refused(line, *wanted):
        with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
            await term("run", session="s", input=line)
        for w in wanted:
            assert w in str(e.value), e.value
    await refused("seq 1 3 2>&1 | tee kept.log | tail -1", 'input="seq 1 3 2>&1" last_lines=1 save_to="kept.log"')
    await refused("seq 1 3 > out.log 2>&1", 'input="seq 1 3" save_to="out.log"')
    await refused("sleep 30 &", 'input="sleep 30" timeout_s=1500', "session_split")
    await refused("sleep 30; cat out.log", "the `wait` tool", '"until COND; do sleep 1; done; cat out.log"')
    await refused("set -e; false", 'input="( set -e; false )"', "exits the shell")
    # Lines that are none of these run at once.
    for line in ("seq 1 3 > /dev/null 2>&1", "sleep 1; cat /dev/null", "seq 1 3 >> out.log 2>&1", "echo a | tail -f -",
                 "( set -e; true )", "bash -ec 'set -e; true'"):
        out = await term("run", session="s", input=line, timeout_s=3)
        assert "Refused, not run" not in out, out


@pytest.mark.second_guess
async def test_a_redirect_before_a_trailing_semicolon_gets_its_second_guess(term):
    # An eval's `gdb --version | head -1; gdb ... -ex q > /tmp/sysgdb.txt 2>&1;` ran unasked.
    with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
        await term("run", session="s", input="echo a; seq 1 3 > out.log 2>&1;")
    assert 'input="echo a; seq 1 3" save_to="out.log"' in str(e.value), e.value


@pytest.mark.second_guess
async def test_a_plain_ssh_or_su_is_offered_the_hop_that_carries_the_integration(term, env):
    # The session's bashrc puts the shim on PATH after the shell starts, out
    # of /proc/PID/environ's sight: the full path, which is there for certain.
    shim = env[0] / "bin" / "dtmcp"
    with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
        await term("run", session="s", input="ssh -l root nowhere.invalid")
    assert f'`run` session="s" input="{shim} ssh -l root nowhere.invalid"' in str(e.value), e.value
    assert "`docent-terminal-mcp ssh` takes the same arguments" in str(e.value)
    with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
        await term("run", session="s", input="su - nobody")
    assert f'input="{shim} su - nobody"' in str(e.value), e.value
    # On the PATH the shell started with: the short name.
    await term("session_create", name="p", cwd=str(env[0]),
               env={"HOME": plain_home(env[0]), "PATH": f"{shim.parent}:/usr/bin:/bin"})
    with pytest.raises(AssertionError, match=r"Refused, not run \(this happens once; sending the same line again runs it\)") as e:
        await term("run", session="p", input="ssh nowhere.invalid")
    assert 'input="dtmcp ssh nowhere.invalid"' in str(e.value), e.value


@pytest.mark.second_guess
async def test_a_shell_without_the_integration_is_second_guessed(term):
    out = await term("run", session="s", input="env -i PS1='plain$ ' bash --norc --noprofile -i", timeout_s=5)
    assert "plain$" in out, out
    with pytest.raises(AssertionError, match="has no shell integration") as e:
        await term("run", session="s", input="echo hi")
    assert '`shell_integrate` session="s"' in str(e.value), e.value
    out = await term("run", session="s", input="echo hi")  # sent again: it runs
    assert "hi" in out.splitlines()[1:], out
    out = await term("run", session="s", input="exit")


def test_su_carries_the_bootstrap_as_its_session_command():
    from docent_terminal_mcp.remote import su_argv
    argv = su_argv(["-", "dtest"], "abc123")
    assert argv[:2] == ["su", "--session-command"] and argv[3:] == ["-", "dtest"]
    assert argv[2].startswith("exec sh -c '") and "export __DTMCP_NONCE=abc123" in argv[2]
    for bad in (["-c", "id"], ["--command=id"], ["--session-command", "id"]):
        with pytest.raises(ValueError):
            su_argv(bad, None)


def test_which_lines_are_plain_hops():
    from docent_terminal_mcp.server import hop_without_integration as hop
    assert [hop(l) for l in ("ssh h", "ssh -p 22 -l root h", "su", "su - dtest", "su -s /bin/zsh dtest")] == \
        ["ssh", "ssh", "su", "su", "su"]
    assert {hop(l) for l in ("ssh h ls", "ssh -N -L 1:a:2 h", "ssh -fN h", "su -c id", "su - a b", "sudo -i",
                             "ssh h; ls", "dtmcp ssh h", "ssh \"$H\"")} == {None}
