"""A Claude Code skill built from the guide, so the long form of the
guidance reaches a model through the channel made for it.

Claude Code cuts a server's instructions and each tool description at
2048 characters; a skill has no such cap, its description (up to 1536
characters) sits in the model's context every turn, and its body loads
when the model decides it applies.  `docent-terminal-mcp skill DIR` writes
DIR/terminal-sessions/SKILL.md plus one reference file per guide topic;
`--plugin` wraps that as a Claude Code plugin that also starts the server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .guide import ESSENTIALS, TOPICS, WHEN

SKILL_NAME = "terminal-sessions"

# What Claude Code shows the model every turn; the body loads on demand.
SKILL_DESCRIPTION = (
    "Read before the first call to the terminal MCP tools (session_create, run, expect, send, "
    "wait, read, search), and before starting a build, a testsuite, gdb, ssh, python or another "
    "REPL from a one-shot shell command while those tools are available.  A terminal session is "
    "the better instrument for anything that runs a program at a prompt, runs longer than a few "
    "seconds, prints more than a reply holds, or may ask a question: this says when to use it "
    "instead of the shell, what the shell habits become there (| tail, | grep, > log, gdb -batch, "
    "sleep-and-poll, nohup, timeout, yes |, ssh per command), and the gotchas.  Skipping it "
    "costs relaunches, lost output and stuck programs."
)

# The body: the instructions' two parts (so the skill and the server say the
# same thing), the announce rule, the patterns topic, and the references.
SKILL_BODY = """\
# Terminal sessions

{when}
The tools are `session_create`, `run`, `expect`, `send`, `wait`, `read`,
`search`, `regions`, `session_list`, `shell_integrate`, `session_resize`,
`session_split`, `session_close`, and `guide` (the same text as here, served by the server).

## The essentials

{essentials}- Announce each new session's attach command to the user on a line of its
  own, so they can watch or step in.

## One-shot habits and their interactive form

{patterns}

## More

- [programs.md](programs.md): gdb, python, REPLs, prompts, questions, `expect`.
- [output.md](output.md): timeouts, large output, `show`, views, the log files.
- [fullscreen.md](fullscreen.md): vim, less, top; `send` order; the screen view.
- [sharing.md](sharing.md): working alongside a human or another agent.
"""


def skill_files() -> dict[str, str]:
    """The skill's files, path -> text (relative to the skill directory)."""
    assert len(SKILL_DESCRIPTION) <= 1536, len(SKILL_DESCRIPTION)
    patterns = TOPICS["patterns"].split("\n", 2)[2]  # drop the title line
    files = {
        "SKILL.md": (
            "---\n"
            f"name: {SKILL_NAME}\n"
            f"description: {json.dumps(SKILL_DESCRIPTION)}\n"
            "---\n\n" + SKILL_BODY.replace("{when}", WHEN).replace("{essentials}", ESSENTIALS)
            .replace("{patterns}", patterns.rstrip("\n"))
        ),
    }
    for topic in ("programs", "output", "fullscreen", "sharing"):
        title, _, body = TOPICS[topic].partition("\n")
        files[f"{topic}.md"] = f"# {title}\n{body}"
    return files


def write_skill(out: Path, plugin: bool = False) -> Path:
    """Write the skill under `out` (a skills directory: ~/.claude/skills,
    a project's .claude/skills); with `plugin`, write a plugin under `out`
    whose .mcp.json starts this installation's server and whose skills/
    holds the skill.  Returns the skill directory."""
    skill_dir = out / ("skills" if plugin else "") / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    for name, text in skill_files().items():
        (skill_dir / name).write_text(text)
    if plugin:
        (out / ".claude-plugin").mkdir(exist_ok=True)
        (out / ".claude-plugin" / "plugin.json").write_text(json.dumps({
            "name": "docent-terminal-mcp",
            "description": "Interactive terminal sessions a human can watch, with the skill for using them",
            "version": "0.1.0",
        }, indent=1) + "\n")
        exe = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0].endswith("docent-terminal-mcp") \
            else Path(sys.executable).parent / "docent-terminal-mcp"
        (out / ".mcp.json").write_text(json.dumps({"mcpServers": {"terminal": {
            "command": str(exe), "args": ["connect", "--actor", "claude-code"],
        }}}, indent=1) + "\n")
    return skill_dir
