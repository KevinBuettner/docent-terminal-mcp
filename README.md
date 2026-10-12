# docent-terminal-mcp

An MCP server that gives an AI model interactive terminal sessions that
a human can watch and type into at the same time.  Sessions are tmux
panes.  Every byte of output is logged, split into per-command *regions*,
and saved to disk.  Regions may be searched, either with tools provided
by this MCP server or by standard filesystem tools favored by the AI
model.  Output may scroll off the screen a human is watching, but the log
keeps it all.  The AI model reads back just the lines it needs, so bulk
output never has to enter its context.

It is equally a way to give a model a terminal of its own for software
development and debugging, watched or not.  There the server acts as a
*docent*: each reply carries the output that matters -- the exit status,
a build's errors, the end of a long run -- together with advice on using
the tool at hand, given as it is being used: that a program is waiting
at a prompt, for example, or that gdb is still there and needs no
relaunching, or that the `| tail` is unnecessary.  A model meeting an
unfamiliar tool is guided through it.  With a human in the session as
well, either one may act as docent to the other, as the situation calls
for.

The command is `docent-terminal-mcp`, or `dtmcp` for short.  The examples
register the server under the name `terminal`: a client shows its tools
under that name (`mcp__terminal__run` in Claude Code, `terminal_run` in
Open WebUI with the ID `terminal`), and the tools' own names are short and
unprefixed (`session_create`, `run`, `read`, ...).

## Use

Requires Linux (parts of the process inspection read `/proc`).  Install
beforehand: tmux (developed and tested with 3.7b), bash 4.4 or newer or
zsh for the shell integration, ssh for `docent-terminal-mcp ssh`, and
[uv](https://docs.astral.sh/uv/).  uv provides the rest.  `uv sync`
installs the Python dependencies into the checkout's `.venv`, and
downloads Python 3.14 first if the system has none.

In a clone of this repository, once:

```sh
uv sync
```

Then register the server with the agent's harness, which starts it as a
command and talks to it over stdio.  (To let several agents share
sessions, run one server for all of them instead: see below.)

Claude Code:

```sh
claude mcp add terminal -- uv run --project "$PWD" docent-terminal-mcp
```

Codex:

```sh
codex mcp add terminal -- uv run --project "$PWD" docent-terminal-mcp
```

Hermes Agent, in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  terminal:
    command: uv
    args: [run, --project, /path/to/docent-terminal-mcp, docent-terminal-mcp]
```

Open WebUI cannot start a command.  It reaches a shared server over HTTP
(`docent-terminal-mcp serve --tcp`, below).

## One server for several agents

`docent-terminal-mcp` with no arguments serves one client over stdio.  To
let several agents -- Claude Code sessions, an assistant's gateway, ... --
share sessions (and each other's work), run one shared server instead:

```sh
uv run docent-terminal-mcp serve            # or install contrib/docent-terminal-mcp.service (systemd --user)
claude mcp add terminal -- uv run --project /path/to/docent-terminal-mcp \
    docent-terminal-mcp connect --actor claude-code
```

The server listens on a **Unix socket**,
`$XDG_RUNTIME_DIR/docent-terminal-mcp/server.sock` (`/run/user/<uid>/...`).
`DOCENT_TERMINAL_MCP_SOCKET_PATH` or `docent-terminal-mcp serve --socket`
overrides it.  The directory is 0700 and the socket 0600, so only that
user (and root) can reach it.  No token, no port numbers to hand out, and
any number of users on a host need no configuration -- each one's server
is found by path.  The socket lives in the runtime directory, not the
state directory, which may be on NFS.

`docent-terminal-mcp connect [--actor NAME]` is a stdio bridge to the
socket, for clients that start an MCP server as a command (Claude Code
and Codex above, Hermes's `mcp_servers` with a `command:`).  One light
process per client session.  When the service restarts, the bridge
reconnects by itself -- a new MCP session with the client's own
initialize parameters -- and answers the call that was in flight; the
client notices nothing.  Only if no server comes back within 30 s does
it exit.

`docent-terminal-mcp serve --tcp [PORT]` also listens on a TCP port
(default 8766) for a client that can only be given a URL.  It binds to
loopback unless `--host` says otherwise; `--host 0.0.0.0` lets Open WebUI
on another machine reach it, with `http://HOST:8766/mcp` and connection
type MCP.  Requests need the bearer token from
`$DOCENT_TERMINAL_MCP_HOME/token` (mode 0600, created on first use;
`docent-terminal-mcp token` prints it).  A port is open to every user who
can reach it; the token file is not.

```sh
claude mcp add --transport http terminal http://127.0.0.1:8766/mcp \
    --header "Authorization: Bearer $(uv run --project /path/to/docent-terminal-mcp docent-terminal-mcp token)" \
    --header "X-Terminal-Actor: claude-code"
```

Each client is an *agent*.  `X-Terminal-Actor` (`connect --actor`) names
it, stably across reconnects; without it the client's own name is used,
made unique per connection (`claude-code#2`).  Regions another agent
started name it.  Its unsubmitted text is protected like a human's, and
per-agent state is kept apart: own half-typed text, "finished since your
last call", screen diffs.  `calls.log` records the agent of every call.

One server per state directory: a second (stdio or shared) refuses to
start while one runs, naming the host that runs it.  A home on NFS is
shared by every host the user logs in on, and so is the enabled unit in
`~/.config/systemd/user`.  `contrib/docent-terminal-mcp.service` therefore
gives each host its own state directory and venv (`%l`, the short host
name: `~/.local/state/docent-terminal-mcp-HOST`,
`~/.local/share/docent-terminal-mcp-venv-HOST`).  One service per host
just works, and no host rebuilds the checkout's `.venv` for its own
Python.  A second server on the same socket path refuses too.

Besides MCP at `/mcp`, the server answers `GET /health` (no token:
`{"service": "docent-terminal-mcp", "host", "user", "instance"}`) and
`GET /api/sessions` (every session's state; on the socket without a
token, on the port with it).  `/health` tells you what a port is before
sending a token.  `instance` is new at every start, so the bridge sees a
restart before it sends a call.  `/api/sessions` reports the exit status,
command, cwd, size, the agent that started it, attached humans, last
output time, log directory, the tmux argv to attach to it, the pane
(`pane_id`, `window`, `parent`, `index`, `active`, `adopted`,
`integrated`; `place`, the host and user its next line runs on when not
this one), and presence: `agents_typed` (agent to epoch seconds of its
last typing call), `human_input` (an attached client's last input, from
tmux), `typing_now` (the agent and tool holding the session).  For
example:

```sh
curl --unix-socket $XDG_RUNTIME_DIR/docent-terminal-mcp/server.sock http://x/api/sessions
```

After updating the checkout (a `git pull`, say), restart the service: a
running server keeps the code it started with.

```sh
systemctl --user restart docent-terminal-mcp
```

Clients on the `connect` bridge reconnect by themselves.  A call in
flight comes back with a note; a `run` is never typed twice.  Claude Code
sees tools added by the new code only after `/mcp`.  Sessions and logs
are unaffected: under systemd the tmux server holding the sessions runs
in a transient scope of its own
(`docent-terminal-mcp-tmux-<socket>.scope`), outside the service's
control group, which a restart or stop would otherwise kill.

## Watching, logs and environment

Watch or join a session the model started:

```sh
tmux -L docent-terminal-mcp ls                # the sessions
tmux -L docent-terminal-mcp attach -t t1      # watch or join one
```

The model gets the same attach command in the replies of `session_create`
and `session_list`.  The tool descriptions and the skill ask it to pass
the command on to the user when it starts a session; whether it does is
up to the model.

Every tool call is recorded in `$DOCENT_TERMINAL_MCP_HOME/calls.log`
(JSON lines: time, tool, session, duration, ok/error).  Never its input
or output.  So a slow or missing reply can be pinned on the server or
the client.

### Environment variables

- `DOCENT_TERMINAL_MCP_ATTACH_PREFIX` -- words put before the attach
  command in replies and `/api/sessions`, for a server that runs where a
  person cannot run tmux directly: `openshell sandbox exec -n NAME --tty
  --`, `podman exec -it NAME`, `ssh -t HOST`.
- `DOCENT_TERMINAL_MCP_HOME` -- default `~/.local/state/docent-terminal-mcp`.
- `DOCENT_TERMINAL_MCP_SOCKET` -- tmux socket name, default
  `docent-terminal-mcp`.
- `DOCENT_TERMINAL_MCP_SOCKET_PATH` -- the shared server's Unix socket,
  default `$XDG_RUNTIME_DIR/docent-terminal-mcp/server.sock`.

## Tools

| tool | purpose |
|---|---|
| `session_create` | bash with shell integration (default, or `command="bash"`), zsh with it (`command="zsh"`), or any command |
| `run` | type a line + Enter; wait for the command to exit, the program's next prompt, a full-screen program's first screen, or `until` (a regex in the output); several lines go one at a time; `show` (a regex, with `show_after` lines after each match) and `last_lines` shape the reply, `save_to` copies the output to a file; `typing` types it as a person would (speed, pauses, corrected mistakes, a seed) for a human watching |
| `expect` | optionally type a line (several go one at a time, as with `run`), then answer prompts by rules (`{pattern, line/text/keys/stop}`) until the command finishes, the program's prompt, `until`, a stop rule, or an unanswered question |
| `send` | type `text`, *then* press `keys` (`ctrl+c`, `up`, `tab`, `q`, ...); returns what appeared, or the screen (`screen="full"`, `"changes"`, `"cursor"`); names a job still in the foreground after ctrl+c; `signal="TERM"`/`"KILL"` ends a foreground job that ignores keys |
| `wait` | keep waiting for a running command (e.g. after `run` timed out); `until`, `show`, `show_after`, `last_lines`, `save_to` as for `run` |
| `read` | `view="cooked"` (log/region, line numbers), `"screen"` (literal rows, size, cursor), `"raw"` (escape codes spelled out) |
| `search` | regex over the log or a region: total count, hits with context and region ids |
| `regions` | list recent commands/inputs with status, exit code, lines, file |
| `shell_integrate` | add the shell integration to a bash or zsh that could not inherit it (ssh, `su -`, `sudo -i`): asks which shell is at the prompt, then types one line that reads the integration file (on another host: the one `docent-terminal-mcp install HOST` put there); types nothing where the shell is already integrated |
| `session_resize` | set the pane size and pin it against attaching clients, or let it follow them again |
| `session_split` | a new pane in a session's window, a session of its own (`<window>+N`), logged and integrated from its first prompt; a pane split any other way is adopted as one |
| `session_logs` | keep a session's logs for good, put them back under the usual rule, delete them when the session closes, or delete a closed session's logs now (see Retention) |
| `session_list`, `session_close` | per session: what runs, which host it is on (over ssh), pane id, window, parent, index, active, and presence (age, last output, who typed when, who is typing); `session_close window=true` closes every pane of the window; it will not close a terminal someone is using (a person attached, or another's command running) -- a person closes those with `docent-terminal-mcp close --force NAME` or `POST /api/sessions/NAME/close` `{"force": true}` (a client's Close button) |
| `guide` | longer usage guidance by topic: patterns (shell habits and their interactive form), programs, output, fullscreen, sharing |

A clipped reply leads with the output's summary lines when the clipped
middle holds them: FAIL lines and a DejaGnu run's final summary, pytest,
cargo, go test, ctest and TAP results, compiler errors, make giving up
(`summary.py`; `DOCENT_TERMINAL_MCP_SUMMARY=0` turns it off, for an A/B).

## What the model is told, and the skill

Three channels carry the guidance.  All are sized for Claude Code's
limits: it cuts a server's instructions, and each tool's description, at
2048 characters -- `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` -- silently
for the model.

- the server **instructions** (under 2048 characters, `guide.py`): the
  rule for choosing a session over a one-shot shell call, "open one at
  the start of a task", and the five things that go wrong most;
- each **tool description** (under 2048) and every **parameter's
  description** in the schema: use cases, the one-shot habit each tool
  replaces, the gotchas;
- the **`guide` tool** (topics `patterns`, `programs`, `fullscreen`,
  `output`, `sharing`) for the long form.  Every tool is marked
  `anthropic/alwaysLoad` so Claude Code never defers it behind its tool
  search, and the read-only tools carry `readOnlyHint`.

The same long form is also a **Claude Code skill**, which loads on demand
without any cap and stays in context once loaded:

```sh
uv run docent-terminal-mcp skill ~/.claude/skills          # or a project's .claude/skills
uv run docent-terminal-mcp skill /path/to/plugin --plugin  # a plugin: .mcp.json starts the server, skills/ holds the skill
```

This writes `terminal-sessions/SKILL.md` (the instructions' two parts and
the patterns) with one reference file per other topic.  Both are
generated from `guide.py`, so the skill and the server say the same
thing.

## How it works

**Capture**: `tmux pipe-pane -O` appends the pane's raw output to
`raw.log`.  The pane blocks on a tmux channel until the pipe is
attached, so nothing is lost.  Input is never logged; text reaches the
log only if the program echoes it, so no-echo password entry leaves no
trace.

**Cooked view** (`cooked.py`): the raw stream interpreted into unbounded
lines.  A line break is always a real one, never a wrap.  Carriage
returns, backspace, erase and cursor motion are applied, so readline
edits and progress bars read as a human saw them.  Columns are screen
cells, so wide (CJK, emoji) characters edit correctly.  Full-screen
programs -- on the alternate screen (vim, less), or painting the normal
one by re-homing the cursor (top, redraw loops) -- become a placeholder
line; read them with `view="screen"`.

**Regions**: in an integrated shell, from the markers the shell prints
(see [Shell integration](#shell-integration)): exact command boundaries
and exit codes.  In a plain shell, and for lines sent to a program, from
prompt matching.

**Where a session is**: the process chain from the foreground job to
the session's shell (an `ssh`'s command line, a shell of another user
after `sudo -i`) and the markers of integrated shells.  A reply says
when it changes, and regions that ran elsewhere are tagged `· on HOST`.
Lines sent to a foreground program (gdb, python) become *input* regions
ending at the next prompt (regex), when output settles, or at a timeout.

**Memory**: rows that can no longer change live only in `session.txt`,
read back through a byte-offset table.  3 million lines (a 25 MB log)
peak at about 150 MB for the whole server process.

**Recovery**: session state is rebuilt from `raw.log` + `regions.jsonl`,
so a new server process picks up sessions a previous one started,
including anything typed while no server was running.

On disk, per session: `meta.json`, `raw.log`, `session.txt` (cooked whole
session; its line numbers are the ones tools report), `regions.jsonl`
(the index, one line per change), `regions/NNNN.txt` (the 8 most recently
used regions; any other is a line range of `session.txt`, and
`read region=N` writes its file again), `state.json` (per agent: the
prompt text it left unsubmitted, so a restarted server can tell it from
a human's, what finished since its last reply, notes already given, where
it was last told the session is).

A region holds only output.  Its command -- for bash, the command line;
for a program, its prompt plus the input -- labels the region instead.

## Shell integration

### What it is, and why it is needed

A terminal gives a program nothing but a stream of bytes.  Nothing in it
says where one command's output ends and the next prompt begins, or
whether the command succeeded.  A person sees that at a glance.  A
program reading the stream has to guess.  The shell integration removes
the guessing: a short script loaded into the session's bash or zsh that
makes the shell mark those points with invisible escape sequences -- the
OSC 133 "semantic prompt" markers that terminals such as iTerm2, kitty
and WezTerm use for the same purpose:

- prompt start (and continuation prompts, such as inside a heredoc or an
  unclosed quote);
- end of the prompt, where the typed command starts;
- start of the command's output;
- command finished, with its exit status (and, for a pipeline, every
  stage's).

Each marker also carries the shell's pid, so nested shells are told
apart; the host and user, so the server knows when a command runs over
ssh or after `su`; and the session's *nonce*, a random value known only
to the session's shell.  Markers without it -- some program's output,
another session's raw log catted, say -- are ignored.

### What it gives

- `run` returns the moment the command finishes, with its exit status
  (`exit 1`; a pipeline's statuses too), never early and never by guess.
- Each command's output is a region of its own with exact boundaries:
  neither the prompt nor the echoed command line is mixed into it.
- A continuation prompt is known for what it is, not taken for the
  command having finished.
- Shells inside shells: a bash started in a session inherits the
  integration, so its commands get regions and statuses too, and the
  outer command closes with its own status when it exits.  Across ssh
  (with the integration brought along, below), replies say which host a
  session is on and tag the regions that ran there.

### Without it

A shell without the integration -- one reached by a plain `ssh`, `su -`
or `sudo -i`, a `bash --norc`, a shell other than bash or zsh -- still
works.  Its output is logged, cooked and searchable like that of any
other shell.  `send`, `expect`, `read`, the screen view and a human
attaching are unaffected.  What is lost is knowing where commands begin
and end.  The server falls back to *prompt matching*, as it does for
programs like gdb or python:

- **No exit statuses.**  A reply cannot say whether a command
  succeeded; the model has to check (`echo $?`) itself.
- **Boundaries are inferred.**  A command counts as finished when the
  last line looks like a prompt: it ends in `$ `, `# `, `% ` or `> `, or
  matches the session's `prompt_pattern`.  An output line that happens to
  look like one can end a region early.  A prompt that does not (a
  customised `PS1`) leaves `run` waiting until the output has been quiet
  for a while, or until its timeout; `prompt_pattern` fixes that.
- **Further hops are not seen.**  An ssh started from a plain shell on
  another host is invisible to the server, which inspects only its own
  host's processes, so the host a reply names can be out of date.

Replies say when a session is at a plain shell, and how to integrate it
there (`shell_integrate`, or `docent-terminal-mcp ssh`).

### What it changes in the shell

- bash is started with `shell/bashrc`.  It first reads the same startup
  files a login shell would (as a normal tmux pane does), then loads
  `shell/integration.bash`.  zsh gets `shell/integration.zsh` through a
  `ZDOTDIR` of docent-terminal-mcp's (`shell/zsh/`), whose files read the
  user's own.
- `PROMPT_COMMAND` becomes a single hook that runs the shell's own
  prompt commands.  `PS1`, `PS2` and `PS0` get the markers wrapped around
  them (and again if a prompt framework rebuilds them).  The hook
  functions and the `__DTMCP_*` variables are exported, so a bash
  started inside inherits them -- and programs run there can see them in
  their environment.
- Markers are printed only when the prompt goes to the session's own
  terminal: a program that runs a shell on a pty of its own (expect,
  pexpect, GDB's testsuite) sees its prompts unchanged.

### Getting it into a shell

A session started with no command (or `bash`) is an integrated bash.
`command="zsh"` is an integrated zsh, and a zsh started inside it is
integrated too.  A shell reached over ssh, `su -` or `sudo -i` does not
inherit it.  There `shell_integrate` first asks which shell is at the
prompt and then types one readable line:
`. DIR/integration.bash && export __DTMCP_NONCE=...` (`.zsh` for zsh).
DIR is the server's own `shell/` directory on its host (when the user at
the prompt can read it), and `~/.local/share/docent-terminal-mcp/vN`
anywhere else -- see `docent-terminal-mcp install` below.  The server
never copies the integration to a host by itself.

**Remote hosts.**  A plain `ssh` is left as it is: its shell works by
prompt matching.  `docent-terminal-mcp ssh [ssh options] [user@]host`, run
in a session in place of `ssh`, connects with the integration.  Like
kitty's `kitten ssh`, it passes a small POSIX sh script as ssh's remote
command.  The script unpacks the integration into a temporary directory,
starts the user's shell (zsh if that is their login shell, else bash)
with it, and removes the directory (bash once loaded, zsh when it exits).
Nothing is typed at a prompt and nothing stays on the host; running it
is the choice to do this.  `docent-terminal-mcp ssh --print host` shows
the ssh command and the script.  Sessions have the command on their PATH
(a shim in `$DOCENT_TERMINAL_MCP_HOME/bin`).  Replies say which host a
session is on, and when that changes.

`DOCENT_TERMINAL_MCP_SSH_ENV`, a list of variable names set in the
session (by the client that created it, through `session_create`'s
`env`), has those variables set on the far side too -- an editor's
`ADJUTANT_EDITOR`, say, so a program over there still knows which editor
it runs in.  The list itself goes along, for the next hop.
`docent-terminal-mcp into` carries them the same way.

`docent-terminal-mcp install [ssh options] [user@]host` is for the shells
`docent-terminal-mcp ssh` cannot reach -- a plain ssh that is already
open, `sudo -i` there (install for `root@host`), a second hop.  The user
runs it, once per host and user.  It copies `integration.bash` and
`integration.zsh` to `~/.local/share/docent-terminal-mcp/vN` on the host
(a tar archive on the stdin of one ssh; older versions' directories are
removed), and from then on `shell_integrate` works there.  `--print`
shows the ssh command, the script and the files without running anything;
`--remove` takes the files away again.  N is the integration's version
(`remote.INTEGRATION_VERSION`).  After an upgrade that changes it, a host
reads as not installed until the command is run again.  On a host without
the files `shell_integrate` types the same line, the shell says the file
is not there, and the reply names the install command for the user.  That
answer is remembered for the host, server-wide, for 12 hours
(`$DOCENT_TERMINAL_MCP_HOME/integration-missing.json`; a later success
clears it).

`docent-terminal-mcp into COMMAND...` is the same for anything that
opens a shell somewhere else with a terminal -- a container, a sandbox:
`docent-terminal-mcp into podman exec -it box`,
`docent-terminal-mcp into openshell sandbox exec -n box --tty --`,
`docent-terminal-mcp into bwrap ... --`.  The bootstrap goes as the
program COMMAND runs.  Use it as a session's own program
(`session_create command="docent-terminal-mcp into ..."`) when the point
is confinement: typed into a host shell, a failed entry leaves the next
lines running on the host.  `openshell` is treated as `ssh` is: a
carrier whose own processes say nothing about the program at the other
end.  So are `podman`, `docker`, `nerdctl` and `kubectl` when their
command line holds a terminal on a container (`exec -it`, `run -it`,
`attach`).

Not yet: fish (`shell_integrate` names it and leaves it alone; it works
as a plain shell).

## Retention

A closed session's directory is deleted once all of these hold.  It is
closed (no tmux pane, so no person attached either; an exited pane still
kept is not closed).  Nothing in it was written or read for
`DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS` days (`docent-terminal-mcp serve
--keep-closed-days`; default 3, `0` keeps everything) -- reading changes
no file, so `read`, `search` and `regions`, which work on a closed
session's logs too, touch `last-used` there.  And it is not pinned.  No
reference counting: clients forget to let go, as people do, so the dates
decide.  Archived generations (`NAME.N`) follow the same rule.  The
sweep runs at start-up and hourly; sessions marked delete-on-close,
every minute.  Each candidate is checked again just before it goes, and
each deletion is one line in the server's log and one in `removed.jsonl`
(a tombstone, kept 30 days), so that a later request for it says when
and why it went.

Marker files in a session's directory, which anyone may set: `keep`
(pinned) and `delete-on-close` (its logs go the moment it closes;
`session_create delete_logs_on_close=true` sets it before the first byte
is logged).  The `session_logs` tool, `POST /api/sessions/NAME/logs`
(`{"action": "keep" | "default" | "delete_on_close" | "delete"}`) and
`docent-terminal-mcp logs keep|default|delete-on-close|delete NAME` set
and clear them, or delete a closed session's logs now.
`docent-terminal-mcp logs due` (`POST /api/sweep?dry_run=1`) lists what
the next sweep would delete; `docent-terminal-mcp logs sweep` runs it.
`session_list` says until when a closed session's logs are kept.
`/api/sessions` has `keep`, `delete_on_close` and `expires`.

Only the server that made a session may call it closed: `meta.json`
records the host, the tmux socket and its path.  A home on NFS is seen
from every host, and a server under another `TMUX_TMPDIR` cannot see this
one's tmux; such sessions are left alone.  With no tmux server answering,
only sessions made on this very socket path count as closed.

## Security

docent-terminal-mcp has no security boundary of its own: what a model
types runs with the rights of the user who started the server.  The
intended answer is a sandbox made for the purpose, with docent-terminal-mcp
as the terminal in or in front of it:

- the session's shell alone in a sandbox: `docent-terminal-mcp into
  COMMAND...` runs the shell integration through the command that enters
  the sandbox (`podman exec -it ...`, `bwrap ...`), so the server stays
  outside and keeps seeing the processes;
- or the whole server inside one, reached through its Unix socket in a
  mounted directory.

Recipes for rootless podman (with a seccomp profile that lets gdb trace),
bubblewrap and NVIDIA OpenShell have been tried; they will be added to
`contrib/` once re-verified.

## Tests

```sh
uv run pytest
```

The end-to-end tests drive real tmux sessions on private sockets; some
attach real tmux clients from Python ptys to stand in for a human.

## License

MIT; see `LICENSE`.  Written by Kevin Buettner with Claude (Anthropic's
Claude Code).
