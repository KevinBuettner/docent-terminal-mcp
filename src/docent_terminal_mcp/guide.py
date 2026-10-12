"""What the model is told: the server instructions (sent with every
session, and cut by Claude Code at 2048 characters), and the longer
guidance served on demand by the `guide` tool.  The skill is built from
the same parts, so the three never drift."""

# The instructions.  WHEN says when a session is the better instrument and
# the one rule that decides it (the eval's rounds 3-4: models that never
# opened a session never used the server; those that opened one at the
# start of the task did their work in it).  ESSENTIALS is what goes wrong
# most.  The skill body reuses both.
from .names import NAME

WHEN = """\
Interactive terminal sessions (tmux): a shell kept open between calls that
a human can watch and type into.  All output is logged in full, split into
per-command regions, searchable.

Use a session instead of a one-shot shell call for anything that runs a
program with a prompt (gdb, python, ssh, any REPL), takes more than a few
seconds (a build, a testsuite), prints more than a reply holds, may ask a
question, or that the user may want to watch or take over.  Open one
at the start of a task (`session_create`, no arguments: bash with exact
exit statuses) and do the task's shell work in it: cwd, environment and
programs persist.  A plain shell call stays right for
reading and editing files and for quick commands whose whole output you
want; a shell job that notifies you when it ends, if you have one, for a
build you need nothing from before then.
"""

ESSENTIALS = """\
- `run` types a line + Enter and waits for the result: the exit status at
  bash, or the next prompt of the program you are talking to.  `send`
  presses keys (ctrl+c, q, arrows) without waiting for a result.  `expect`
  is for a command that may ask (gdb's debuginfod offer, "Quit anyway?"):
  it answers in the same call.
- gdb: `gdb -q -ex 'break foo' -ex run -ex bt ./prog` without -batch
  answers like a batch run, then waits at (gdb): ask the next question
  there instead of relaunching.  Never pipe or redirect a live program's
  output: it then waits at a prompt you cannot see.
- `| tail`, `| grep`, `> log 2>&1` are seldom needed: a reply keeps the
  first and last lines and the exit status, `show=REGEX` picks out lines,
  `search` and `read` find the rest; `save_to` keeps a file copy.
- At `timeout_s` (up to 1500: one call for a whole build) the command
  keeps running: `wait` resumes it, `send keys=["ctrl+c"]` interrupts,
  `send signal="TERM"` if that fails.
- prompt_pattern is rarely needed: only after a reply came back "waiting"
  at a prompt it did not know -- it offers the pattern.
"""

POINTER = """\
- Details: the `terminal-sessions` skill if one is listed, else `guide`.
"""
INSTRUCTIONS = WHEN + "\n" + ESSENTIALS + POINTER

TOPICS = {
    "patterns": """\
One-shot shell habits and their interactive form

What a shell tool taught you to do, and what it becomes in a session.
The reason is the same each time: output is kept in full and in order,
the program stays alive between calls, and a human may be watching.
Open one session at the start of a task and keep it: the choice is then
made once, and each command after that costs one call, as with the
shell tool -- with the exit status, and without the time limit a shell
tool puts on one call (often 10 minutes).

  cmd | tail -50, cmd | head -20
    -> `run cmd`.  Output longer than max_lines keeps its first and last
    lines, with a marker naming what was left out; `read region=N start=L`
    pages the middle.  Only the end: `last_lines=50`.  (A pipe into head
    can also kill cmd with SIGPIPE.)
  cmd 2>&1 | grep -E 'error|FAIL'
    -> `run cmd show="error|FAIL"`: the matching lines plus the last few,
    while the command runs unfiltered, exit status its own, the human
    seeing everything.  stderr is on the terminal already.  Never pipe an
    interactive program: it waits at a prompt you cannot see.
  cmd | grep -A3 'error'
    -> `run cmd show="error" show_after=3`: each match and the 3 lines
    after it.
  cmd > build.log 2>&1, then grep/sed/tail on build.log
    -> nothing to redirect: every command's output is logged.  Afterwards
    `search pattern=... region=N` (total count, hits with context and line
    numbers) and `read region=N start=L`; or read the region's file with
    your own tools (guide output).  When a file is the point -- to diff
    two runs, or for a tool that reads it -- `run cmd save_to=build.log`:
    the command runs unredirected, the reply is as usual, and build.log
    gets a copy, kept up to date while it runs.
  make -j8 > build.log 2>&1 &  ...  sleep 30; tail build.log  (poll a job)
    -> `run "make -j8" timeout_s=1500`: one call, the exit status in the
    reply, the first and last lines shown, all of it kept (a client that
    backgrounds long calls -- Claude Code, after about 2 minutes -- still
    gets that reply, later).  Or a short
    timeout_s, other work meanwhile, then `wait` as often as needed: each
    reply is new output only, and the exit status when it ends.
    `until=REGEX` returns the moment a line appears.  `regions` shows the
    state without waiting.  Other sessions and your other tools stay free.
  nohup cmd &, setsid, screen/tmux by hand
    -> a session outlives your call and your own restart; what runs in it
    keeps running; `session_list` finds it again, with its whole log.
  timeout 60 cmd
    -> timeout_s.  The command is not killed at the deadline: `wait` for
    it, or `send keys=["ctrl+c"]`; `send signal="TERM"` if that fails.
  gdb -batch -ex 'break f' -ex run -ex bt ./prog
    -> the same line without -batch: the same output, then you are at
    (gdb) to ask the follow-up (`run input="p *ptr"`) without paying to
    reach that point again.  -batch answers gdb's questions silently with
    their defaults (a pending breakpoint is dropped); without it, `expect`
    answers the ones you foresee and the rest come back to you.
  echo 'cmds' | gdb, printf ... | python3, yes | cmd, an expect script
    -> `run` with several lines (sent one at a time, each waiting for the
    prompt), or `expect` with one rule per question.  A REPL keeps its
    state between calls: define now, use later.
  python3 -c '...' again and again
    -> `run python3` once, then each snippet as a `run`; a multi-line
    block is pasted whole.  The same for psql, sqlite3, node, irb.
  ssh host cmd, one ssh per command
    -> `run ssh host` once (`expect` for the host-key question, a stop
    rule for a password), then plain `run`s there.  For exact exit
    statuses there: `run {NAME} ssh host` instead, which brings the shell
    integration along for that connection (nothing installed there, plain
    ssh unchanged) -- with a user to ask, ask how they want to connect.
  cd dir && cmd, every time
    -> cd once; the session keeps its cwd, variables and environment.
  cmd; echo "exit=$?"
    -> the exit status is in every reply header (pipestatus for a
    pipeline); the echo would make the region's status echo's.
  kill %1, pkill prog
    -> `send keys=["ctrl+c"]`, ctrl+d, or `send signal="TERM"` for the
    foreground job.  pkill by name from outside can hit the session's
    own shell.
  less, man, vim, top, htop, a curses installer
    -> fine in a session: `run` returns the first screen, `send` drives
    them and returns the screen (guide fullscreen).
  script, tee, to keep a record
    -> the log is the record, and the human sees it live.
  sudo, su -, ssh with a password
    -> `expect` with a stop rule at the password prompt; the human types
    it (keystrokes are never logged).  A reply names such a prompt.  If you
    must answer it, `expect` with secret=true -- which keeps the value out
    of the reply only: at a prompt that echoes, it is in the log anyway.

Where the plain shell tool stays right: reading and editing files, a
one-line command whose whole output you want, and work with nothing to
show anyone.  A shell job that notifies you when it ends, if your shell
tool has one, is fine for a build you need nothing from before then; the
testsuite and the debugging belong in the session, where the output stays
and gdb stays alive at the point of interest.

Gotchas:
  - `keys` belongs to `send`; `run` takes `input` (a line, Enter added).
  - A program started with its output piped or redirected waits where you
    cannot see it: if a reply names a foreground job you did not expect,
    ctrl+d ends gdb or python at their prompt.
  - A multi-line `run` stops at the first line that asks a question or
    times out; the reply says how many lines were not sent.
  - `set confirm off` in gdb answers every question with its default,
    which for a pending breakpoint is "no".
  - `force` discards someone else's typed line; only for text you know
    is stale.  Refusals name whose line it is.
  - A full-screen program blocks `run`; `send` drives it (`q`, `:q`).
  - `run` refuses while a command is still running and not waiting for
    input (a build, or one `until` returned from early): the line would be
    read by it, or run by the shell after it ends, unreported.  `wait`,
    ctrl+c, or for a program that reads keys itself, `send`.  force=true
    there means typing ahead.
  - timeout_s above 1500 is cut: clients abort a call idle for 30 min.
    Some clients (Claude Code) move a call that runs longer than about 2
    minutes to the background and go on; its reply arrives later.  To keep
    working meanwhile, a timeout_s under 2 minutes and then `wait` is the
    same thing in steps.
  - `pgrep -f`/`pkill -f PATTERN` also match a `bash -c` or `ssh host '...'`
    whose command line holds the pattern, the asking one included;
    `pgrep -f '[h]ermes gateway'` matches the program, not that line.
  - The session's files are read-only for you; never write them.
""",
    "programs": """\
Programs with their own prompt (gdb, python, pdb, a REPL)

Either start one from bash with `run` -- it returns when the program waits at
its prompt -- or give it a session of its own (a session for bash itself
needs no command: session_create without one, or with command="bash", is
the bash with shell integration):

    session_create command="gdb -q ./prog"
    run input="break main"
    run input="run"

Each line you `run` into a program becomes an input region.  Like a shell
command's region, it holds only the output: the prompt and your input line
are its label, e.g. "[region 5 · (gdb) bt · ready · ...]".  Status "ready"
means the program is back at its prompt.  "finished" means the shell
command it was talking to ended (the program quit, or a `read -p` got its
answer); that command is reported with its exit status.  "exited" means
the session's own program quit, with its exit status.  A question you
answered shows as "answered (region N)".  A region ends
when the cursor line ends in a prompt.  The default prompt_pattern recognises
(gdb), (top-gdb), (Pdb), (lldb), >>>, ... and lines ending in $ # % >.  Pass
your own regex (matched against the end of the cursor line) for anything
else, either per call or as the session default in session_create.  A
continuation prompt ("> ": a heredoc, gdb's `commands`) counts as well.

Without a recognised prompt, `run` returns when output has been quiet for
settle_ms (default 3000) and the program is not visibly working (a process
running, or sleeping on a timer, keeps the call waiting for the prompt, up
to timeout_s): status "settled" means nothing shows it still working, but
it may be (`wait` or `read` later); "waiting" means output stopped mid-line
with the program waiting for input, which is nearly always a question such
as "(y or n)" -- answer it with `run`, or `wait` if it is not one.  The
same holds for a shell command that asks something (read -p): answer with
`run`, and the reply reports the command's exit status when it finishes.

Predictable questions cost a round trip each with `run`.  `expect` answers
them in one call -- rules fire on whatever appears, in any order, and a
rule that never fires is fine.  Without `input`, the rules also see a
question already waiting at the cursor (one `run` came back "waiting" at):

    expect input="run" respond=[{"pattern": "Enable debuginfod", "line": "n"}]
    expect input="break foo" respond=[{"pattern": "Make breakpoint pending", "line": "y"}]
    expect input="quit" respond=[{"pattern": "Quit anyway", "line": "y"}]
    expect input="ssh host" respond=[
        {"pattern": "continue connecting", "line": "yes"},
        {"pattern": "[Pp]assword:", "stop": true}]     <- hand over to the human

A question no rule covers ends the call at once ("unanswered"), so a
surprise comes back to you instead of being guessed at.  Rules match the
output as shown, trailing spaces included ("Name: $"); "times" lets a rule
fire repeatedly; "secret" keeps a sent value out of the transcript.

Several lines in one `run` (or `expect`) to a program such as gdb are sent
one at a time, each with its own region, so the reply reads like the
transcript (bt / frame 1 / p x in one call).  If a line does not come back
to the prompt, the rest are not sent, and the reply says so; with `expect`,
a question raised by one line is answered by the rules, never by the lines
after it.  For Python's REPL (and IPython) a multi-line block is pasted
whole instead, as one input: an exception in it drops the lines after it,
and a traceback counts lines from the block's start.

gdb's setup on its command line, without -batch: `gdb -q -ex 'break foo'
-ex run -ex bt ./prog`, started with `run` in a session (or as the
session's command), answers in one call like a batch run -- and then
leaves you at the (gdb) prompt at the point of interest.  Ask the
follow-up question there, set more breakpoints, continue: no paying again
to reach that point (minutes or hours, for some bugs), and no assuming a
rerun reaches the same state (programs are not always deterministic).
Stay at that prompt for the follow-ups: a relaunch pays again to get there.
Don't redirect or pipe a live gdb's output (`> trace.out`, `| grep`): it
then waits at a prompt you cannot see; the session keeps the output, so
`search` it.  When you are done, `quit`.  -batch is for gdb run by a tool
that cannot answer back (a plain shell command, a script).

Every hit of a breakpoint in one reply, rather than a theory built at the
first: one `run` with the lines `break foo`, `commands`, `bt`, `continue`,
`end`, `run`.  gdb prints each stop with its backtrace and goes on; the
reply holds them all (`silent` as the first line after `commands` drops
the stop banner).

Without -batch, gdb asks its questions for real, where -batch would have
answered them: "Make breakpoint pending on future shared library load?
(y or [n])" for a breakpoint in a library not loaded yet stops the -ex
commands after it until answered.  Answer it with `expect` (input=the gdb
command line, respond=[{"pattern": "Make breakpoint pending", "line":
"y"}]), or avoid it with -iex 'set breakpoint pending on'.  Not with
`set confirm off`: that answers each question's default, and here the
default is no -- the breakpoint is silently not set, and the program runs
past the point you wanted.

Nested shells: a bash started inside a session (bash, exec bash, a
subshell script run interactively) inherits the shell integration, so its
commands get exact regions and exit statuses too; the outer command stays
open around them and closes with its own status.  A shell that cannot
inherit it -- over ssh, or after `su -` -- falls back to prompt matching
(no exit statuses).  `{NAME} ssh host` (run in the session, in place of
ssh) connects with the integration: it goes along as ssh's remote command
for that connection only -- nothing is typed at a prompt, nothing stays on
the host, and a plain ssh is unchanged.  Choosing it is the user's call
(it runs a setup script on their host): when working with a person, ask
how they want to connect.  At a shell already open over there (a plain
ssh, `sudo -i`), `shell_integrate` types one line that reads the
integration file -- this server's own on this host; on another host the
copy the user put there with `{NAME} install host` (they run it, once per
host; without it the reply says so and the shell stays plain).  That
answer is remembered for the host, for every session, for 12 hours (or
until shell_integrate works there): later notes say when it was found
missing instead of offering shell_integrate again.  Replies say which
host a session is on ("on build-host"), and when that changes.

Programs that notice their output is not a terminal behave differently:
gdb run with output piped answers its own questions ("[answered N; input
not from terminal]").  In a session they are asked for real.
""",
    "fullscreen": """\
Full-screen programs (vim, less, man, top, htop)

`run` starting one returns as soon as it has drawn its screen, and includes
the screen.  From then on, drive it with `send`; each `send` returns the
rendered screen.

The screen view shows rows exactly as displayed: a long line wraps onto
the next row; rows are numbered; runs of blank rows are summarised.  The header
gives the size and the cursor position as 1-based row and column, e.g.
"[screen 85x55 · cursor row 4, col 1 · full-screen program]".

`send` types all of `text` first, THEN presses `keys` in order.  To press a
key before typing, use two calls.  Example, an ex command in vim from an
unknown mode:

    send keys=["escape"]
    send text=":wq" keys=["enter"]

Putting "escape" after the text would cancel the command.

Output size: every `send` returns the whole screen by default.  While
editing, pass screen="changes" to get only the rows that changed since the
last screen returned, or screen="cursor" for the rows around the cursor.
Use read view="screen" for a full look when needed.  In pagers (less,
man) the cursor sits on the status line, so screen="cursor" hands back the
changed rows there instead (or the whole screen when there is nothing to
compare with).  `max_lines` also caps the rows of a screen reply.

Full-screen programs clip long lines at the window edge; verify file
contents from the shell afterwards (or `:set wrap` in vim).  In the cooked
log, a full-screen program appears as one placeholder line.
""",
    "output": """\
Long-running commands and large output

Timeouts: when `run` reaches timeout_s the command keeps running and its
region stays open, and the reply names the processes in the foreground
(what keys typed now would reach).  Call `wait` to keep waiting for it, or
`send keys=["ctrl+c"]` to interrupt it (exit status 130).  If something
survives ctrl+c the reply says so: gdb at its own prompt takes ctrl+c as
"Quit" and stays, unseen when its output goes to a pipe -- ctrl+d or `quit`
ends it.  A job that ignores all of that: `send signal="TERM"`, then
"KILL" -- the foreground job only, never the session's own shell.
timeout_s is capped at 1500 s (clients abort a call with no reply for 30
min, losing it); for a long build, `wait` in several calls -- each reply
is new output only.  `wait` also reports commands that finished since
your last call.

No need to redirect output to a file to keep replies small: replies are
already clipped to max_lines (set it low for a noisy build or testsuite),
and the full output is logged and searchable -- and stays visible to a
human watching.  Polling a running command with `wait` shows only the lines
not shown before.

To see only some lines as they come -- errors in a build, FAIL lines in a
testsuite -- pass `show` (a regex) to `run` and `wait` instead of piping
through grep: the reply lists the matching lines and the last few, while
the command runs unfiltered, so the human sees everything and the exit
status is the command's own.  Every matching line is shown, in order,
while they fit max_lines; beyond that identical ones are shown once with
a count.  `show_after`=N adds the N lines after each match, as grep -A
does (a match numbered "N:", a line after it "N-").  `last_lines`=N
replies with only the last N lines -- with `show`, it is how many last
lines follow the matches.  `save_to`=FILE copies the output to a file as
well (a relative path is in the session's current directory, on the
machine this server runs on): written at once, kept up to date while the
command runs, in full when it ends.  It works on a program's answer too: `run` input="step"
show="stop_pc|stepped" at gdb's prompt, under `set debug infrun 1`.  `show` sees only what the command prints: some test harnesses
write their results to a file alone.  (GDB's parallel `make check` prints
each FAIL line whole but among other tests' output, a summary per test
file, and the whole run's summary last; gdb.sum has them merged.)  To tell
whether a change added a failure, compare the failing tests by name
between the two runs, never their counts: "1 unexpected failure" before
and after can be two different tests (keep the first run's gdb.sum, then
diff the sorted FAIL lines of both).  The header reports
`pipestatus` whenever a
pipeline's stages differ, so there is no need to echo PIPESTATUS -- and
appending `; echo ...` would make the region's exit status echo's.

Large output: `run` keeps the first and last parts of output longer than
max_lines and replaces the middle with a line naming what was left out,
"[... 480 lines omitted (lines 127-606); ...]".  When the middle holds
lines that say how a run went -- test results (FAIL:, a DejaGnu run's
final summary, pytest's, cargo's, go test's, ctest's, TAP's not ok),
compiler errors, make giving up -- the reply leads with them, numbered,
under "[summary lines picked out of the whole output ...]"; `show` or
`last_lines` choose your own lines instead.  That suits build logs;
for a list you need whole (a grep of failures), run it on its own or raise
max_lines.  Nothing is lost:
  - `read region=N start=L` pages through it (line numbers as shown),
  - `search pattern=... region=N` finds lines without reading them all;
    it reports the total number of matches, so you know if you saw them all,
  - a line longer than 2000 characters (minified JSON, base64) is cut in
    every reply, "[... N more characters on this line; ...]"; `search`
    then shows the part around a match, "[col C of N] ...", 200
    characters either side,
  - the most recently used regions (8) are also plain text files,
    <log dir>/regions/NNNN.txt, which your own file tools (Read, Grep) can
    use directly, often more cheaply.  An older region's header gives its
    lines of session.txt instead (use them as a Read offset); `read
    region=N` writes its file again.

Statuses in a region's header: "running" (still going; for a command
whose region was shown, `wait` continues it), "exit N" (a shell command
finished; "pipestatus A B" for a pipeline; "exit status not seen" when its
shell ended without saying, e.g. `exit` itself), "ready" (a program's
input is back at its prompt), "waiting" (output stopped at a question or
an unknown prompt), "settled" (quiet, nothing visibly working -- it may
still be), "timeout" (timeout_s passed, still working; `wait` resumes),
"until" (your `until` pattern appeared, still running), "interrupted" (a
ctrl+c brought the program back to its prompt), "finished" (the shell
command a program input went to ended), "exited" (the session's own
program quit), "answered (region N)" (a question, answered in region N),
"fullscreen" (a full-screen program; read view="screen").  "no output"
means the region printed nothing.

Views: read view="cooked" (default) is the log as a file -- real line
breaks only, escape codes applied, and control characters a terminal
shows as nothing (a stray \\x05 in a string) left out, as on screen.  view="screen" is the terminal as shown
now.  view="raw" is the bytes the program wrote, escape codes spelled out,
for debugging odd rendering -- or output that may hold such bytes.

Files, per session under the log directory session_create names
(<state>/sessions/<name>/ on the server's host):
  session.txt      the whole cooked log; its line numbers are the ones
                   replies, `read start=` and `search` use, in a region too
  regions/NNNN.txt the text of a finished region (its id zero-padded to 4),
                   for the 8 most recently used regions only; an older
                   region's header gives its lines of session.txt instead,
                   and `read region=N` writes the file again
  regions.jsonl    the index: one JSON line per region, the last per id
                   wins (its format is the server's own)
  raw.log          every byte the pane wrote, escape codes included
The server owns these.  Read them with any tool (grep, sed, Read; grep -r
across a session works); never write, move or delete them -- region files
are made read-only for that reason, and the rest is what every `read` and
`search` is built from.  Nothing is dropped for the session's life.  A
closed session's logs stay readable with `read`, `search` and `regions`,
and are deleted once neither written nor read for a few days
(DOCENT_TERMINAL_MCP_KEEP_CLOSED_DAYS, default 3; session_list says when):
`session_logs`
keeps a session's logs for good, or deletes them now or when it closes
(ask the user first; session_create delete_logs_on_close=true for a
session that may show something sensitive).
""",
    "sharing": """\
Sharing a session with a human

Announce every new session prominently, before using it -- on its own
line, not in passing -- with the attach command session_create printed:

    New terminal session "gdb": tmux -L docent-terminal-mcp attach -t gdb

session_list shows how many humans are attached to each session.

Regions a human started are marked "started by human"; unmarked regions
are yours.  Output of a human's command is logged like any other.

On a shared server, other agents can use the same sessions.  session_list
names the agent that started each session; regions they started name them
("started by claude-code#2"); unsubmitted text they typed is protected like a
human's (the refusal names the agent).  `wait`, when nothing of yours is
running, reports the commands a human or another agent ran since your
last call -- both those still running then and those started and finished
in between -- and, if one is still running, waits for it.  `regions`
lists the session at once, without waiting.

Only calls that type (run, expect, send, shell_integrate) take turns: one
of yours waits while another agent's is typing or waiting for its result,
and the reply then starts "[waited N s for <agent>'s run in this
session]".  `wait`, `read`, `search` and `regions` never hold anyone up.
A command someone else started is theirs: the reply says so rather than
suggest interrupting it.  session_close will not close a terminal
someone is using -- one a person is attached to, or another's while a
command runs there: leave it, or ask the user to close it (they can, from
their side).  Otherwise it says whose session it was and which running
command it ended -- ask before closing one that is not yours.

`run` and `send` refuse to type when the prompt holds text you did not
type (the human is mid-command; `wait` waits for them to finish or erase
it, and says which), or when an attached human pressed a key
within the last 2 seconds.  Do not reach for force=true: wait, or ask the
user.  force is for a line you know is stale; `run` with force discards
that line (or abandons an incomplete multi-line command) before typing.

When a human attaches, tmux resizes the pane to their window (with several
humans, the most recently active one's), so the cols and rows given to
session_create stop applying; the screen view header always reports the
current size.  To keep a size -- output formatted for a known width --
create the session with fixed_size=true or call session_resize; the human
then sees the pane at that size, clipped or padded.  session_list marks
such sessions "fixed".

Split windows: `session_split` puts a new pane beside a session's, in
the same tmux window, as a session of its own (named <window>+1, +2, ...
unless you name it), logged and with shell integration from its first
prompt: a human attached to the window sees both, for instance a
testsuite in one pane and gdb in the other.  A pane made any other way
-- a human's split, or `tmux split-window` typed in a session -- is
adopted as <window>+N the next time a tool touches that window or
session_list runs: its screen so far goes into its log.  It is a plain
shell until the first `run` there, which adds the shell integration
first when a bash sits idle at its prompt (exact exit statuses from
then on); nothing is typed into it before you use it.  Every tool addresses a session by
name; which pane is active on screen does not matter.  session_close
closes one pane; window=true closes the whole window.

Passwords: keystrokes are never recorded -- the log holds only what the
terminal displays.  Typed commands appear because the terminal echoes
them; a password typed at a prompt that turns echo off appears nowhere.
A reply names a password or passphrase prompt as one (the terminal has
echo off, or over ssh the prompt asks for one).  Let the human type it.
If you must answer it, use `expect` with secret=true, never `run`: a line
given to `run` becomes the region's label.  secret=true keeps the value
out of the reply's transcript only; at a prompt that echoes, the value is
in the log anyway, and the reply says so.
""",
}
# The program's name is filled in: it may change (names.py).
TOPICS = {k: v.replace("{NAME}", NAME) for k, v in TOPICS.items()}

INDEX = "Topics: " + ", ".join(TOPICS) + ".  Call guide with one of them."
