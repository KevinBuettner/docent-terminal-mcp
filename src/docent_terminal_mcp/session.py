"""A terminal session: a tmux pane, its output log, and the regions within it.

On-disk layout, one directory per session:

    meta.json        how the session was created
    raw.log          every byte the pane wrote, via `tmux pipe-pane -O`
    session.txt      the cooked (file-like) view of the whole log
    regions.jsonl    the region index: a line per change, the last per id wins
    regions/NNNN.txt cooked text of the REGION_FILES most recently used
                     finished regions (the rest are line ranges of
                     session.txt, and come back as files when read)
    widths.jsonl     the pane width from each raw.log offset on, so replay
                     interprets wrapped-line editing as it was first seen

Only pane *output* is logged; input never is.  Anything a person or the
model types reaches the log only if the program echoes it, so a password
typed at a no-echo prompt is never recorded.

A region is a span of the cooked view with a known origin:

  command  one shell command, delimited exactly by OSC 133 markers from the
           bash integration, with its exit status.
  input    what followed one model-sent line to a program with no shell
           markers (gdb, python, a shell without integration), ending at
           the next prompt, when output settles, or at the timeout.

Input regions nest inside the command region of the program they talk to.
Line numbers are 1-based and match session.txt.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
import os
from array import array
from collections.abc import Iterator
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cooked import CookedTerminal, Pos
from .tmux import CARRIER_NAMES, REMOTE_CLIENTS, SHELLS, Tmux, is_carrier, ssh_one_shot, foreground_processes, foreground_state, proc_info, process_chain, remote_destination
from .typist import type_out

INTEGRATION = Path(__file__).parent / "shell" / "integration.bash"
INTEGRATION_ZSH = Path(__file__).parent / "shell" / "integration.zsh"
LOCAL_HOST = __import__("socket").gethostname()
LOCAL_USER = __import__("getpass").getuser()


async def place_of_processes(tty: str, shell_pid: int,
                             table: dict[str, list] | None = None) -> tuple[Place | None, int | None]:
    """Where the foreground job runs, from the process chain between it
    and the session's own shell (`shell_pid`): an ssh on it -- that host,
    from its command line (and the ssh's pid); a shell of another user
    (sudo -i, su -).  (None, None): here, as this user.  `table`: the
    foreground processes of every terminal, read once for a listing."""
    procs = table.get(tty, []) if table is not None else await foreground_processes(tty)
    leader = min((p.pgid for p in procs), default=0)
    chain = process_chain(leader, shell_pid) if leader and leader != shell_pid else []
    if (own := proc_info(shell_pid)) is not None and own.comm not in SHELLS:
        chain.append(own)  # the session's own program: `session_create command="ssh host"`
    remote = next((p for p in chain if is_carrier(p.pid, p.comm)), None)
    if remote is not None:
        dest = remote_destination(remote.argv)
        host, user = dest if dest else ("?", None)
        return Place(short_host(host), user, remote.comm), remote.pid
    shell = next((p for p in chain if p.comm in SHELLS), None)
    if shell is not None and (user := user_name(shell.uid)) != LOCAL_USER:
        via = next((p.comm for p in chain if p.comm in ("sudo", "su", "doas", "runuser")), "shell")
        return Place(short_host(LOCAL_HOST), user, via), None
    return None, None


def command_place(command: str, host: str, user: str | None) -> Place | None:
    """Where a shell command typed on `host` (as `user`) goes on to: `ssh
    B`, `ssh -J j B`, `sudo -i`, `sudo -u U -s`, `su -`, `su - U` -- also
    as a later part of a list (`uname -r; su - dtest`, `cd /x && sudo -i`):
    the first part that goes somewhere is where the next line runs while
    it does.  Not a pipeline's: its input is not the terminal."""
    lexer = shlex.shlex(command.split("\n", 1)[0], posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    parts: list[list[str]] = [[]]
    for token in tokens:
        if token in (";", "&&", "||", "&"):
            parts.append([])
        else:
            parts[-1].append(token)
    for argv in parts:
        if "|" in argv or "|&" in argv:
            continue
        if (place := _argv_place(argv, host)) is not None:
            return place
    return None


def _argv_place(argv: list[str], host: str) -> Place | None:
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        argv = argv[1:]  # VAR=value prefixes
    if not argv:
        return None
    prog = argv[0].rsplit("/", 1)[-1]
    if prog in ("ssh", "slogin", "sshpass", "telnet"):
        if ssh_one_shot(argv[2:] if prog == "sshpass" else argv):
            return None  # one remote command: the next line still runs here
        dest = remote_destination(argv)
        return Place(short_host(dest[0]), dest[1], prog) if dest else None
    if prog == "sudo":
        opts, target, shell = argv[1:], "root", False
        i = 0
        while i < len(opts) and opts[i].startswith("-"):
            o = opts[i]
            if o in ("-u", "--user") and i + 1 < len(opts):
                target, i = opts[i + 1], i + 1
            elif o.startswith("--user="):
                target = o.split("=", 1)[1]
            elif o in ("-i", "-s", "--login", "--shell"):
                shell = True
            i += 1
        rest = opts[i:]
        if shell and not rest or rest and rest[0].rsplit("/", 1)[-1] in SHELLS:
            return Place(host, target, "sudo")
        return None
    if prog == "su":
        names = [a for a in argv[1:] if not a.startswith("-")]
        if "-c" in argv or "--command" in argv:
            return None
        return Place(host, names[0] if names else "root", "su")
    return None


def user_name(uid: int) -> str:
    try:
        return __import__("pwd").getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def short_host(host: str) -> str:
    return host.split(".", 1)[0]


@dataclass
class Place:
    """Where a session's next line would run, when not this host as this
    user: the host (short name), the user if another, and how that was
    seen -- "ssh" (its command line), "shell" (an integrated shell's
    markers), "sudo"/"su"/"shell" for a local shell of another user."""
    host: str
    user: str | None
    via: str
    at: tuple = (0, 0)  # (for markers: the text position they came at)

    def label(self) -> str:
        who = f"{self.user}@" if self.user and self.user != LOCAL_USER else ""
        return who + self.host

    def same(self, other: "Place | None") -> bool:
        return other is not None and (self.host, self.user or LOCAL_USER) == (other.host, other.user or LOCAL_USER)


# Who is making the current tool call: an agent's name on a shared server,
# "model" on a single-client (stdio) one.  Per-agent state is kept by it.
ACTOR: contextvars.ContextVar[str] = contextvars.ContextVar("actor", default="model")


def integration_line(nonce: str | None = None, shell: str = "bash", directory: str | None = None) -> str:
    """What shell_integrate types: one readable line that reads the
    integration file from `directory` (default: this server's own; on
    another host, where `NAME install HOST` put it) and sets the session's
    marker nonce.  The leading space keeps it out of history where
    HISTCONTROL (bash) or HIST_IGNORE_SPACE (zsh) allows."""
    if directory is None:
        directory = shlex.quote(str(INTEGRATION.parent))
    return (f" . {directory}/integration.{'zsh' if shell == 'zsh' else 'bash'}"
            + (f" && export __DTMCP_NONCE={nonce}" if nonce else ""))


# Earlier versions typed the integration whole, base64-encoded; in the
# cooked view of logs from those, a note stands for it.
LEGACY_TYPED = ('[ -n "$BASH_VERSION" ] && eval "$(printf %s ', 'eval "$(printf %s IyB')
INTEGRATION_NOTE = "[shell integration typed by shell_integrate]"

# A prompt at the end of the cursor line, before the cursor: Python's at the
# start of the line only (a "checking for x..." line is not one), a shell's
# or REPL's sign followed by a space.
# gdb's own continuation prompt (python/commands/define bodies) is a bare
# ">" on its own.
# A shell prompt is the whole line up to its "$ " (user@host:dir$ ), so the
# label of a command in a shell without integration names the host, and
# the prompt's text does not leak into the previous command's output.
# zsh's continuation prompt names what is open ("for> ", "dquote> ",
# "if then> "): these words, and no others, so "build> " printed by a
# program is not taken for one.
ZSH_PS2 = ("for|while|until|if|then|else|elif|do|foreach|quote|dquote|bquote|cmdsubst|subsh|cursh"
           "|heredoc|heredocd|pipe|cmdand|cmdor|case|select|function|repeat|braceparam|mathsubst|math|always")
# The terminal's echo of a ctrl+c ("^C") can sit before a prompt printed
# without a line break: Python's REPL back at ">>> " after a command that
# handled the interrupt itself.  IPython's prompts ("In [3]: ", "   ...: ")
# are Python's kind, followed by a space or the line's end: pytest's
# progress line ("....F.. [ 35%]") starts with no "..." prompt -- and
# Python's own "..." always has its space ("... "): a progress line paused
# at three dots is not one.  gdb
# indents its continuation prompt by block depth (" >" inside a while,
# "  >" nested).
DEFAULT_PROMPT_PATTERN = (
    r"((?P<py>^(?:\^C)*(>>>|\.\.\.(?=\s)|In \[\d+\]:| +\.\.\.:)(?=\s|$)\s*)|(?P<cont>^(?:\^C)* *(?:(?:" + ZSH_PS2 + r") )*(?:" + ZSH_PS2 + r")?> ?)"
    r"|(?P<dbg>(\((gdb|top-gdb|Pdb|lldb)\)|irb\([^)]*\):\d+[>*]|C-Kermit>)\s*)|(?P<sh>^.*?[$#%>] +))$"
)
DEFAULT_PROMPT_RE = re.compile(DEFAULT_PROMPT_PATTERN)
PY_MAIN = re.compile(r"^(?:\^C)*(>>>|In \[\d+\]:)")  # Python's (or IPython's) main prompt
PY_CONT = re.compile(r"(\.\.\.|\.\.\.:)$")  # and its continuation prompt, at the end of a line
PY_AT_END = re.compile(r"(>>>|\.\.\.|In \[\d+\]:|\.\.\.:)$")  # either, output before it or not
# Once a program has shown a prompt of one kind, only that family counts as
# its prompt: after "(gdb) " or ">>> ", a progress line ending in "> " or
# "$ " is output, not a prompt.  A bare "> " continuation goes with gdb's
# prompt and with a shell's (bash's PS2, after an unfinished line).
# The program is told by the shell command it runs in (with the shell
# integration) and by the terminal's foreground process (tmux's
# pane_current_command): gdb started from a plain bash is a program of its
# own.  Where neither tells (a remote gdb, over ssh), a debugger's prompt
# still counts where a shell's was seen: "(gdb) " is no progress line.
PROMPT_FAMILIES = {"py": {"py"}, "cont": {"cont", "dbg", "sh", "py"}, "dbg": {"dbg", "cont"}, "sh": {"sh", "cont"}}
PROMPT_OVERRIDES = {("sh", "dbg")}  # (known, seen): the seen kind counts anyway
CONTINUATION = r"(?P<cont>^ *> ?)"  # gdb indents it by block depth: " >" inside a while


def with_continuation(source: str) -> str:
    """A prompt pattern given by the model, with the continuation prompt
    ("> ": bash's PS2 in a heredoc, gdb's in a `commands` block) added: a
    pattern naming the program's own prompt is not meant to make every
    continued command sit out its timeout."""
    if "(?P<cont>" in source:
        return source
    if source.endswith("$") and not source.endswith("\\$"):
        combined = f"(?:{source[:-1]}|{CONTINUATION})$"
    else:
        combined = f"(?:{source})|{CONTINUATION}$"
    try:
        re.compile(combined)
    except re.error:  # e.g. the pattern's own global flags, or group names
        return source
    return combined


def prompt_regex(source: str | None) -> re.Pattern:
    """The prompt pattern to use: the model's (with continuations), or the default."""
    return re.compile(with_continuation(source) if source else DEFAULT_PROMPT_PATTERN)


DEFAULT_SETTLE_MS = 3000
POLL_S = 0.05
HUMAN_IDLE_S = 2.0
PROMPT_QUIET_S = 0.3
DEBUG_TAIL_ROWS = 40  # how far up a prompt followed by a program's debug lines is looked for
DEBUG_LINE = re.compile(r"^\s*\[[a-z][\w -]*\] ")  # gdb's "set debug" output: "[infrun] ...", "  [remote] ..."
PARTIAL_LINE_QUIET_S = 1.5
QUESTION_ENDS = ("?", ":", ")", "]")  # how questions end ("(y or n)", "[Y/n]"); prompts end in > $ #
FOREGROUND_STABLE_S = 1.0  # at the start of a line, "waiting for input" needs a state this old
EXPECT_OVERLAP_ROWS = 50
MULTILINE_QUIET_S = 2.0
REGION_FILES = 8  # region files kept, most recently used first
REGION_FILE_MODE = 0o444  # the server owns them; other tools only read
SYNC_CHUNK = 1 << 20


@dataclass
class Region:
    id: int
    kind: str  # "command" | "input"
    source: str  # the agent that started it ("model" for a single client), or "human"
    start: Pos
    end: Pos | None = None
    command: str | None = None
    # running | done (command) | ready (input: back at a prompt) | settled |
    # waiting | answered (a waiting question, answered by region answered_by) |
    # finished (the command it talked to ended) | timeout | exited | abandoned
    status: str = "running"
    exit_status: int | None = None
    pipestatus: list[int] | None = None
    started: float = 0.0
    ended: float | None = None
    # Byte range of raw.log holding the region.  Marker-delimited regions
    # get the bounds of the read chunks containing their markers, so the
    # range may include a little output from either side.
    raw_start: int | None = None
    raw_end: int | None = None
    answered_by: int | None = None
    # An input region's prompt pattern, when the call gave its own: `wait`
    # resumes with it.
    prompt_pattern: str | None = None
    host: str | None = None  # where it ran: a command's shell's host (h= in the marker), or
    user: str | None = None  # the remote host of an input region (Session.where); and as whom
    shell_pid: str | None = None  # a command's shell's pid
    echo_cut: list | None = None  # [echoed, typed] characters when the echo stopped short of the input

    @classmethod
    def from_json(cls, d: dict) -> Region:
        d = dict(d)
        d["start"] = tuple(d["start"])
        d["end"] = tuple(d["end"]) if d["end"] is not None else None
        return cls(**d)

    @property
    def done(self) -> bool:
        return self.status != "running"


@dataclass
class Rule:
    """Answer output matching `pattern` with `action`: "line" (text + Enter),
    "text", "keys" (tmux key names), or "stop" (end the script there)."""

    pattern: re.Pattern
    action: str
    value: str | list[str] | None = None
    times: int = 1
    secret: bool = False
    source: str = ""  # the pattern as written, for reporting
    fired: int = 0
    then_stop: bool = False  # answer, then end the script (line/text/keys with stop)

    def describe_action(self, keys_shown: list[str] | None = None) -> str:
        if self.action == "stop":
            return "stopped"
        if self.then_stop:
            return Rule(self.pattern, self.action, self.value, secret=self.secret).describe_action(keys_shown) \
                + ", then stopped"
        if self.action == "keys":
            return "pressed " + " ".join(keys_shown or [])
        shown = "(secret)" if self.secret else repr(self.value)
        return f"sent {'line' if self.action == 'line' else 'text'} {shown}"


MAX_LINE_CHARS = 2000  # of one line in a reply: a minified or base64 line can be megabytes


def clip_line(text: str, limit: int = MAX_LINE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (text[:limit] + f" [... {len(text) - limit} more characters on this line; "
            "the `search` tool shows the part around a match, the region's file has it whole]")


class UntilWatch:
    """`run`'s `until`: has the pattern appeared in the output since
    `start` (given with each call: known only once the output begins)?
    Only rows not searched before are searched again, with an overlap for
    patterns spanning lines, as expect does."""

    def __init__(self, session: Session, pattern: re.Pattern | None):
        self.session, self.pattern = session, pattern
        self.scan_from: Pos | None = None
        self.hit = False

    def __call__(self, start: Pos) -> bool:
        if self.pattern is None or self.hit:
            return self.hit
        term = self.session.term
        text = self.session.unconsumed_text(max(start, self.scan_from or start))
        if self.pattern.search(text):
            self.hit = True
        else:
            self.scan_from = (max(0, term.row - EXPECT_OVERLAP_ROWS), 0)
        return self.hit


@dataclass
class ExpectResult:
    outcome: str  # done | ready | until | finished | stopped | unanswered | fullscreen | timeout | exited
    transcript: list[str]
    unused: list[str]  # patterns of rules that never fired
    regions: list[Region]  # shell command regions, or the one input region
    text_start: Pos  # where new output began, for showing it without regions
    notes: list[str] = field(default_factory=list)  # e.g. lines not sent


class Session:
    def __init__(self, name: str, directory: Path, tmux: Tmux, meta: dict):
        self.name = name
        self.dir = directory
        self.tmux = tmux
        self.meta = meta
        self.raw_path = directory / "raw.log"
        self.text_path = directory / "session.txt"
        self.regions_dir = directory / "regions"
        self.regions_dir.mkdir(exist_ok=True)
        self.lock = asyncio.Lock()  # typing calls take turns (server.typing_turn)
        self.holder: tuple[str, str] | None = None  # (agent, tool) of the call holding it
        self.last_typed: dict[str, float] = {}  # agent -> when its last typing call ended (epoch s)
        self.interrupt_survivors: set[int] = set()  # pids ctrl+c was seen not to stop
        self.habits_noted: set[str] = set()  # "agent:habit" pairs already remarked on (server.habit_notes)

        self.term = CookedTerminal(on_osc=self._on_osc)
        self.term.elisions = [(needle, INTEGRATION_NOTE) for needle in LEGACY_TYPED]
        self.offset = 0
        self._chunk = (0, 0)  # raw.log range being fed, for region raw bounds
        self.regions: list[Region] = []
        self.last_output = time.monotonic()

        # Shell-marker state.
        self.shell_state = "unknown"  # unknown | prompt | input | running | finished
        self.input_start: Pos | None = None
        self.open_commands: dict[str, Region] = {}  # by shell pid
        self.model_driving: str | None = None  # the agent whose call is typing: commands are its
        self.continuation = False  # the shell is showing PS2
        self.continuations = 0  # PS2 prompts seen, to notice new ones
        self.prompts = 0  # PS1 prompts seen: a line that ran no command (a comment) still brings one
        self.reaped = 0  # commands closed because their shell was gone (reap_gone_shells)
        self.handed_off = False  # the terminal changed hands since, and its holder waits for input
        self.input_holders: set[int] | None = None  # foreground process groups when a line went to a program
        self.in_python = False  # a ">>>" was seen where a shell's prompt was learned: "..." is Python's too
        self.sent: tuple[str, int] | None = None  # (agent, raw offset) of the last `send` that typed
        self.typing: dict | None = None  # the current call's human-like typing (typist.plan's arguments)
        # Per agent (see ACTOR): prompt text it left unsubmitted, the screen
        # it last got, the commands running at its last reply, and how far
        # it has seen each running region.
        self._typed: dict[str, str] = {}
        self._last_screen: dict[str, list[str]] = {}
        self._open_at_reply: dict[str, set[int]] = {}
        self._working, self._working_checked = False, 0.0
        self._last_id_at_reply: dict[str, int] = {}
        self._shown_upto: dict[str, dict[int, int]] = {}
        # Deleted working directories already reported (server.deleted_cwd_note).
        self.cwd_noted: set[str] = set()
        # save_to files still to write: path -> the regions they hold.
        self.saves: dict[str, list[int]] = {}
        self.foreground: str | None = None  # see tmux.foreground_state
        self.prompt_place: Place | None = None  # where the shell at the last prompt is (its marker)
        self.place_noted: dict = {}  # agent -> the Place (or None: here) its last reply was at
        self._remote: tuple | None = None  # (pid, text position first seen) of the ssh in use
        self.foreground_since = 0.0  # since when it has been that
        # program -> prompt kind; the program is its command's id, or with no
        # command open, "0:" + the foreground process's name (see _program_key)
        self.prompt_kinds: dict[int | str, str] = {}
        self.pane_command = ""  # the foreground process, as tmux last named it
        # program -> the shell prompt it last showed, where no markers tell
        # (see prompt_signature)
        self.prompt_texts: dict[int | str, str] = {}
        self._prompt_candidate: tuple[int | str, str] | None = None  # see note_unknown_prompt

        self.archived: Path | None = None  # earlier logs moved aside on name reuse
        self._commands_seen = 0

        # Committed rows live in session.txt, not in memory: `_row_offsets`
        # holds where each starts, and the terminal reads them back from there.
        self._flushed_rows = 0
        self._flushed_bytes = 0
        self._row_offsets = array("Q")
        self.term.committed_line = self._stored_line

        # Width changes, by raw.log offset (see set_width).
        self.widths_path = directory / "widths.jsonl"
        self._widths: list[tuple[int, int]] = []
        if self.widths_path.exists():
            for line in self.widths_path.read_text().splitlines():
                with contextlib.suppress(ValueError, KeyError):
                    d = json.loads(line)
                    self._widths.append((d["offset"], d["width"]))
        # Output before the first recorded change was read at the creation
        # size.  (Older sessions did not record it.)
        if meta.get("cols") and not any(offset == 0 for offset, _ in self._widths):
            self._widths.insert(0, (0, meta["cols"]))
        self.term.width = self._widths[0][1] if self._widths else None

        # Replay: rebuild the cooked view from raw.log, reusing persisted
        # regions rather than re-creating them.
        self.index_path = directory / "regions.jsonl"
        self.regions = self._load_index()
        # Region ids with a file, least recently used first.
        files = sorted(self.regions_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
        self._files = [int(p.stem) for p in files if p.stem.isdigit()]
        self._persisted = [r for r in self.regions if r.kind == "command"]
        self.text_path.write_text("")
        self._reader = open(self.text_path, "rb")
        self.sync()
        state = directory / "state.json"
        if state.exists():
            saved = json.loads(state.read_text())
            typed = saved.get("model_typed") or {}
            self._typed = typed if isinstance(typed, dict) else {"model": typed}
            self.prompt_kinds = {(int(k) if k.isdigit() else k): v for k, v in saved.get("prompt_kinds", {}).items()}
            self.prompt_texts = {(int(k) if k.isdigit() else k): v for k, v in saved.get("prompt_texts", {}).items()}
            # So a `wait` after a server restart still reports what finished
            # since the agent's last reply.
            self._open_at_reply = {a: set(ids) for a, ids in saved.get("open_at_reply", {}).items()}
            self._last_id_at_reply = dict(saved.get("last_id_at_reply", {}))
            self.interrupt_survivors = set(saved.get("interrupt_survivors", []))
            self.habits_noted = set(saved.get("habits_noted", []))
            self.sent = tuple(saved["sent"]) if saved.get("sent") else None
            self.place_noted = {a: (Place(*v) if v else None) for a, v in saved.get("place_noted", {}).items()}
            self.last_typed = dict(saved.get("last_typed", {}))
            # How far each agent has seen each open region: a `wait` after a
            # restart shows only what is new, as before it.
            self._shown_upto = {a: {int(k): v for k, v in seen.items()}
                                for a, seen in saved.get("shown_upto", {}).items()}
            self.saves = dict(saved.get("saves", {}))
            self.cwd_noted = set(saved.get("cwd_noted", []))
        self.trim_region_files()
        # An input region still running when the last server stopped was
        # never finished; close it where the output now ends.
        for r in self.regions:
            if r.kind == "input" and not r.done:
                self._finish(r, self.content_end(), "abandoned")

    # -- reading the log -------------------------------------------------------

    def sync(self) -> bool:
        """Consume new output from raw.log.  Returns True if there was any.

        Works in chunks, flushing committed rows to disk after each, so a
        large backlog (a replay, a burst of output) never sits in memory.
        """
        got = False
        try:
            f = open(self.raw_path, "rb")
        except FileNotFoundError:
            return False
        with f:
            f.seek(self.offset)
            while data := f.read(SYNC_CHUNK):
                self._chunk = (self.offset, self.offset + len(data))
                self._feed_at_widths(data)
                self._flush_text()
                got = True
        if got:
            self.last_output = time.monotonic()
        return got

    def _feed_at_widths(self, data: bytes) -> None:
        """Feed `data` (starting at self.offset), switching the terminal's
        width at the offsets where it changed."""
        start = self.offset
        for offset, width in self._widths:
            if start < offset < start + len(data):
                cut = offset - self.offset
                self.term.feed(data[:cut])
                self.offset += cut
                data = data[cut:]
            if offset <= self.offset:
                self.term.width = width
        self.term.feed(data)
        self.offset += len(data)

    def set_width(self, width: int) -> None:
        """The pane is now `width` columns: output from here on is read at
        that width, and the change is recorded for replay."""
        if width == self.term.width:
            return
        self.term.width = width
        self._widths.append((self.offset, width))
        with open(self.widths_path, "a") as f:
            f.write(json.dumps({"offset": self.offset, "width": width}) + "\n")

    def _flush_text(self) -> None:
        term = self.term
        with open(self.text_path, "r+b") as f:
            f.seek(self._flushed_bytes)
            f.truncate()
            if term.floor > self._flushed_rows:
                parts = []
                offset = self._flushed_bytes
                for r in range(self._flushed_rows, term.floor):
                    data = (term.line(r).rstrip() + "\n").encode()
                    parts.append(data)
                    self._row_offsets.append(offset)
                    offset += len(data)
                f.write(b"".join(parts))
                for r in range(self._flushed_rows, term.floor):
                    term.lines[r] = None  # now read back from the file
                self._flushed_bytes = offset
                self._flushed_rows = term.floor
            tail = "\n".join(term.line(r).rstrip() for r in range(term.floor, len(term.lines)))
            f.write(tail.encode())

    def _stored_line(self, row: int) -> str:
        start = self._row_offsets[row]
        stop = self._row_offsets[row + 1] if row + 1 < len(self._row_offsets) else self._flushed_bytes
        return os.pread(self._reader.fileno(), stop - start - 1, start).decode(errors="replace")

    def iter_lines(self, rows: range) -> Iterator[tuple[int, str]]:
        """(row, text) for `rows`, reading committed rows sequentially from
        session.txt -- the efficient way to scan many rows."""
        r = rows.start
        if r < self._flushed_rows:
            # A handle of its own, so interleaved scans cannot disturb each other.
            with open(self.text_path, "rb") as f:
                f.seek(self._row_offsets[r])
                stop = min(rows.stop, self._flushed_rows)
                while r < stop:
                    yield r, f.readline()[:-1].decode(errors="replace")
                    r += 1
        for r in range(max(r, rows.start), rows.stop):
            yield r, self.term.line(r)

    def close(self) -> None:
        self._reader.close()

    # -- spans of text ---------------------------------------------------------

    @staticmethod
    def span_rows(start: Pos, end: Pos) -> range:
        """Rows holding text between two positions (an end at the start of a
        row does not include that row)."""
        (r1, _), (r2, c2) = start, end
        return range(r1, r2 + 1 if c2 > 0 else r2)

    def iter_span(self, start: Pos, end: Pos, rows: range | None = None) -> Iterator[tuple[int, str]]:
        """(row, text) for the span from `start` to `end`, the first and last
        rows cut at the positions; `rows` may narrow it."""
        full = self.span_rows(start, end)
        rows = full if rows is None else range(max(rows.start, full.start), min(rows.stop, full.stop))
        for r, line in self.iter_lines(rows):
            lo = start[1] if r == start[0] else 0
            hi = end[1] if r == end[0] else None
            yield r, line[lo:hi].rstrip()

    def span_text(self, start: Pos, end: Pos) -> str:
        return "\n".join(text for _, text in self.iter_span(start, end))

    def clip_rows(self, start: Pos, end: Pos, max_lines: int) -> set[int]:
        """The rows `clip_span` shows of the span (all of them if it fits)."""
        rows = self.span_rows(start, end)
        max_lines = max(1, max_lines)
        if len(rows) <= max_lines:
            return set(rows)
        head = min(max(3, max_lines // 5), max_lines // 2)
        return set(rows[:head]) | set(rows[len(rows) - (max_lines - head):])

    def clip_span(self, start: Pos, end: Pos, max_lines: int, where: str = "", numbers: bool = True,
                  last: int | None = None) -> str:
        """The span's text, keeping the head and tail when longer than
        `max_lines` -- or only its `last` lines, as `tail -n` would; the
        marker names the omitted lines.  Only the rows shown are read."""
        rows = self.span_rows(start, end)
        n = len(rows)
        max_lines = max(1, max_lines if last is None else last)
        if n <= max_lines:
            return "\n".join(clip_line(t) for _, t in self.iter_span(start, end))
        head = 0 if last is not None else min(max(3, max_lines // 5), max_lines // 2)
        tail = max_lines - head
        omitted = n - head - tail
        first_omitted = rows.start + head
        what = f"{omitted} lines omitted"
        if numbers:
            what += f" (lines {first_omitted + 1}-{first_omitted + omitted})"
        top = [clip_line(t) for _, t in self.iter_span(start, end, range(rows.start, first_omitted))]
        bottom = [clip_line(t) for _, t in self.iter_span(start, end, range(rows.stop - tail, rows.stop))]
        where = where.replace("{start}", str(first_omitted + 1)).replace("{limit}", str(min(omitted, 200)))
        where = where.replace("shows them", "pages through them") if omitted > 200 else where
        return "\n".join(top + [f"[... {what}{'; ' + where if where else ''} ...]"] + bottom)

    # -- shell markers ---------------------------------------------------------

    def _on_osc(self, text: str, pos: Pos) -> None:
        if not text.startswith("133;"):
            return
        kind, *params = text[4:].split(";")
        opts = dict(p.partition("=")[::2] for p in params if "=" in p)
        plain = [p for p in params if "=" not in p]
        nonce = self.meta.get("nonce")
        if nonce and opts.get("n") != nonce:
            return  # printed by a program (another session's log catted), not this session's shell
        shell = opts.get("pid", "")  # which shell: nested shells each mark
        if kind == "A":
            if opts.get("k") == "s":  # secondary (continuation) prompt
                self.continuation = True
                self.continuations += 1
                self.shell_state = "prompt"
                return
            # A prompt from a shell with a command still open means it never
            # reported the end (e.g. `exec bash` replaced it).
            region = self.open_commands.pop(shell, None)
            if region is not None:
                self._finish(region, pos, "done")
            self.continuation = False
            self.shell_state = "prompt"
            self.prompts += 1
            # Where this shell is (older shells' markers do not say).
            self.prompt_place = (Place(short_host(opts["h"]), opts.get("u") or None, "shell", pos)
                                 if opts.get("h") else None)
            # Back at the prompt: whatever drew on a grid has finished, even
            # if it left the alternate screen on.
            if self.term.full_screen:
                self.term.leave_full_screen()
        elif kind == "B":
            if not self.continuation:  # a continued command starts at PS1
                self.input_start = pos
            self.shell_state = "input"
        elif kind == "C":
            command = None
            if self.input_start is not None:
                command = self.typed_between(self.input_start, pos, submitted=True).strip()
            self._commands_seen += 1
            region = self._persisted_command(self._commands_seen)
            if region is None:
                region = self._new_region(
                    "command", self.model_driving or "human",
                    pos, command
                )
                region.raw_start = self._chunk[0]
                region.host, region.shell_pid = opts.get("h"), shell or None
                region.user = opts.get("u")
            elif region.host is None and opts.get("h"):
                # Made by `run` before its shell's output marker came: where
                # it runs is known only now (and must outlive a restart).
                region.host, region.user = opts.get("h"), opts.get("u")
                region.shell_pid = region.shell_pid or shell or None
                self._save_region(region)
            if not region.done:
                self.open_commands[shell] = region
            self.shell_state = "running"
        elif kind == "D":
            region = self.open_commands.pop(shell, None)
            if region is not None and not region.done:
                status = plain[0] if plain else ""
                region.exit_status = int(status) if status.lstrip("-").isdigit() else None
                if "pipestatus" in opts:
                    region.pipestatus = [int(v) for v in opts["pipestatus"].split() if v.lstrip("-").isdigit()]
                region.raw_end = self._chunk[1]
                self._finish(region, pos, "done")
                # Commands of shells nested inside it that never reported an
                # end (the shell exited while running `exit`) end with it.
                for inner, r in list(self.open_commands.items()):
                    if r.id > region.id:
                        del self.open_commands[inner]
                        self._finish(r, pos, "done")
            self.shell_state = "finished"

    @property
    def open_command(self) -> Region | None:
        """The innermost running shell command (the latest started)."""
        running = [r for r in self.open_commands.values() if not r.done]
        return max(running, key=lambda r: r.id) if running else None

    def _persisted_command(self, n: int) -> Region | None:
        return self._persisted[n - 1] if n <= len(self._persisted) else None

    # -- regions ---------------------------------------------------------------

    def _new_region(self, kind: str, source: str, start: Pos, command: str | None) -> Region:
        region = Region(
            id=(self.regions[-1].id + 1) if self.regions else 1,
            kind=kind, source=source, start=start, command=command, started=time.time(),
        )
        self.regions.append(region)
        self._save_region(region)
        return region

    def _finish(self, region: Region, end: Pos, status: str) -> None:
        if region.done:
            return
        region.end = end
        region.status = status
        region.ended = time.time()
        self._write_region_file(region)
        self._save_region(region)
        if any(region.id in ids for ids in self.saves.values()):
            self.write_saves()

    # -- save_to files ---------------------------------------------------------

    def save_regions(self, path: str, regions: list[Region]) -> None:
        """Copy the regions' output to `path` (a call's save_to), now and
        again as they grow and when they end; raises OSError if it cannot be
        written now."""
        ids = [r.id for r in regions]
        self._write_save(path, ids)
        if any(not r.done for r in regions):
            self.saves[path] = ids
        else:
            self.saves.pop(path, None)
        self._save_state()

    def write_saves(self) -> None:
        """Bring every pending save_to file up to date; one whose regions
        have all ended is written a last time and dropped."""
        if not self.saves:
            return
        by_id = {r.id: r for r in self.regions}
        for path, ids in list(self.saves.items()):
            try:
                self._write_save(path, ids)
            except OSError:
                pass  # reported at the call that asked for it; nothing to tell now
            if all(by_id[i].done for i in ids if i in by_id):
                del self.saves[path]
        self._save_state()

    def _write_save(self, path: str, ids: list[int]) -> None:
        by_id = {r.id: r for r in self.regions}
        text = "\n".join(self.region_text(by_id[i]) for i in ids if i in by_id)
        with open(path, "w") as f:  # in place, as `>` would: a `tail -f` on it keeps following
            f.write(text + ("\n" if text else ""))

    def _save_region(self, region: Region) -> None:
        """Record a region's current state: one appended line, so saving
        stays cheap however many regions the session has."""
        with open(self.index_path, "a") as f:
            f.write(json.dumps(asdict(region)) + "\n")

    def trim_region_files(self) -> None:
        """Keep only the REGION_FILES most recently used region files (older
        sessions kept one per region)."""
        while len(self._files) > REGION_FILES:
            old = self._files.pop(0)
            (self.regions_dir / f"{old:04d}.txt").unlink(missing_ok=True)

    def _load_index(self) -> list[Region]:
        """The saved regions (the last line for each id wins), compacting
        the file to one line per region.  Reads the older regions.json
        format too."""
        latest: dict[int, dict] = {}
        old = self.dir / "regions.json"
        if old.exists():
            for d in json.loads(old.read_text()):
                latest[d["id"]] = d
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash
                latest[d["id"]] = d
        regions = [Region.from_json(latest[i]) for i in sorted(latest)]
        tmp = self.dir / "regions.jsonl.tmp"
        tmp.write_text("".join(json.dumps(asdict(r)) + "\n" for r in regions))
        tmp.replace(self.index_path)
        old.unlink(missing_ok=True)
        return regions

    def region(self, rid: int) -> Region:
        for r in self.regions:
            if r.id == rid:
                return r
        have = f"its regions are {self.regions[0].id}-{self.regions[-1].id}" if self.regions else "it has no regions yet"
        raise KeyError(f"no region {rid} in session {self.name}: {have} (`regions` lists them)")

    def region_path(self, region: Region) -> Path:
        return self.regions_dir / f"{region.id:04d}.txt"

    def _write_region_file(self, region: Region) -> None:
        """Write a finished region's text to its file, read-only: agents read
        these files with their own tools, and an accidental `>` or `sed -i`
        should fail loudly rather than corrupt what `read`/`search` show."""
        path = self.region_path(region)
        path.unlink(missing_ok=True)  # an earlier copy is read-only
        with open(path, "w") as f:
            for _, text in self.iter_span(region.start, self.region_end(region)):
                f.write(text + "\n")
        path.chmod(REGION_FILE_MODE)
        self._used_file(region)

    def _used_file(self, region: Region) -> None:
        """Mark a region's file most recently used, dropping the least
        recently used beyond REGION_FILES.  Any region's text can be
        written out again from session.txt, so nothing is lost."""
        if region.id in self._files:
            self._files.remove(region.id)
        self._files.append(region.id)
        while len(self._files) > REGION_FILES:
            old = self._files.pop(0)
            (self.regions_dir / f"{old:04d}.txt").unlink(missing_ok=True)

    def region_file(self, region: Region) -> Path | None:
        """The region's file, written again if it was dropped (a finished
        region only); for when the region is being worked with."""
        if not region.done:
            return None
        path = self.region_path(region)
        if path.exists():
            self._used_file(region)
        else:
            self._write_region_file(region)
        return path

    def region_location(self, region: Region) -> str:
        """Where the region's text is on disk: its file if it has one, else
        its lines of session.txt."""
        if not region.done:
            return "file when done"
        if region.id in self._files:
            return str(self.region_path(region))
        rows = self.region_rows(region)
        if not rows:
            return "empty"
        return f"{self.text_path} lines {rows.start + 1}-{rows.stop}"

    def content_end(self) -> Pos:
        last = len(self.term.lines) - 1
        return (last, len(self.term.line(last)))

    def output_end(self) -> Pos:
        """content_end, less blank lines below the cursor: rows a program
        reserved by printing line feeds and moving back up (IPython's
        prompt_toolkit keeps room for its completion menu so)."""
        last, row = len(self.term.lines) - 1, self.term.row
        if row < last and not any(self.term.line(r).strip() for r in range(row + 1, last + 1)):
            return (row, len(self.term.line(row)))
        return self.content_end()

    def region_end(self, region: Region) -> Pos:
        return region.end if region.end is not None else self.output_end()

    def region_text(self, region: Region) -> str:
        return self.span_text(region.start, self.region_end(region))

    def region_rows(self, region: Region) -> range:
        """Rows (0-based) the region covers, for search and paging."""
        return self.span_rows(region.start, self.region_end(region))

    def region_at(self, row: int) -> Region | None:
        """The innermost region containing `row`."""
        best = None
        for r in self.regions:
            if row in self.region_rows(r) and (best is None or r.start >= best.start):
                best = r
        return best

    # -- state queries ---------------------------------------------------------

    @property
    def integrated(self) -> bool:
        return bool(self.meta.get("shell_integration"))

    @property
    def at_shell_prompt(self) -> bool:
        # Only markers change shell_state, so this also holds for a shell
        # that gained the integration later (shell_integrate).
        return self.shell_state in ("prompt", "input")

    @property
    def prompt_pattern(self) -> re.Pattern:
        """The session's prompt: its own pattern, or the default."""
        return prompt_regex(self.meta.get("prompt_pattern"))

    def prompt_start(self, line: str) -> re.Match | None:
        """The leftmost prompt in `line`: the prompt pattern without its
        end-of-line anchor, so text typed after the prompt does not hide
        it.  (Leftmost, so "(gdb) p x > 3" is the prompt "(gdb) ".)"""
        source = self.prompt_pattern.pattern
        if source.endswith("$"):
            source = source[:-1]
        try:
            return re.search(source, line)
        except re.error:
            return None

    def typed_between(self, start: Pos, end: Pos, submitted: bool = False) -> str:
        """What was typed at the shell prompt between `start` (its input
        marker) and `end`.  A line break there is output -- a background
        job printing at the idle prompt ("LATE") -- since Enter submits the
        line; only what follows the last one was typed.  (`submitted`: the
        line ends with the Enter that submitted it.)  Continued commands
        (PS2) keep all their lines."""
        text = self.term.text_between(start, end)
        if submitted:
            text = text.rstrip("\n")
        if not self.continuation and "\n" in text:
            text = text.rsplit("\n", 1)[1]
        return text

    def pending_input(self) -> str:
        """Text typed at the prompt but not yet submitted.  At the shell,
        exactly (from its markers); at a program's prompt (gdb, python, a
        prompt_pattern session), what follows the prompt on the cursor line."""
        if self.shell_state == "input" and self.input_start is not None:
            return self.typed_between(self.input_start, self.content_end()).strip()
        if self.term.full_screen or self.term.row != len(self.term.lines) - 1:
            return ""
        line = self.term.line(self.term.row)
        col = self.prompt_col(line)
        if col is None:
            return ""
        m = self.prompt_start(line[col:])
        if m is None:
            return ""
        rest = line[col:][m.end():]
        if rest and not rest[0].isspace() and not m.group(0)[-1:].isspace():
            return ""  # "...F...." is output (test progress), not "..." with text typed after it
        return rest.strip()

    # -- per-agent state ------------------------------------------------------

    @property
    def model_typed(self) -> str | None:
        return self._typed.get(ACTOR.get())

    @property
    def last_screen(self) -> list[str] | None:
        return self._last_screen.get(ACTOR.get())

    @last_screen.setter
    def last_screen(self, rows: list[str]) -> None:
        self._last_screen[ACTOR.get()] = rows

    @property
    def open_at_reply(self) -> set[int]:
        return self._open_at_reply.get(ACTOR.get(), set())

    @open_at_reply.setter
    def open_at_reply(self, ids: set[int]) -> None:
        last = self.regions[-1].id if self.regions else 0
        actor = ACTOR.get()
        if self._open_at_reply.get(actor) != ids or self._last_id_at_reply.get(actor) != last:
            self._open_at_reply[actor] = ids
            self._last_id_at_reply[actor] = last
            self._save_state()

    def finished_unseen(self) -> list[Region]:
        """Regions that finished since this agent's last reply: those still
        running at that reply, and those someone else (a human, another
        agent) started and finished in between.  Nothing before an agent's
        first reply counts: joining a session does not replay its past."""
        last_id = self._last_id_at_reply.get(ACTOR.get())
        if last_id is None:
            return []
        open_then = self.open_at_reply
        return [r for r in self.regions if r.done and (r.id in open_then or r.id > last_id)]

    @property
    def shown_upto(self) -> dict[int, int]:
        return self._shown_upto.setdefault(ACTOR.get(), {})

    def foreign_input(self) -> str:
        """Unsubmitted prompt text the calling agent did not leave there."""
        pending = self.pending_input()
        return "" if pending == (self.model_typed or "") else pending

    def typed_by(self, text: str) -> str:
        """Who left `text` at the prompt: another agent, or a human."""
        for actor, typed in self._typed.items():
            if typed and typed == text and actor != ACTOR.get():
                return actor
        return "human"

    def note_model_input(self) -> None:
        """Record the prompt as the model's own after it typed something.

        Saved so a restarted server still knows; the text is already in the
        log through the terminal's echo, so this records nothing new.
        """
        typed = self.pending_input()
        if typed != self.model_typed:
            self._typed[ACTOR.get()] = typed
            self._save_state()

    def forget_model_input(self) -> None:
        """A line was submitted to a program: nothing of the caller's is left
        unsubmitted.  (What follows the program's prompt now is its output
        -- ">>> step 1" -- not text to erase before typing.)"""
        if self.model_typed:
            self._typed[ACTOR.get()] = ""
            self._save_state()

    def _save_state(self) -> None:
        """What a restarted server needs beyond the log: the model's own
        unsubmitted text, and the prompt kinds programs have shown."""
        (self.dir / "state.json").write_text(json.dumps({
            "model_typed": self._typed, "prompt_kinds": self.prompt_kinds,
            "open_at_reply": {a: sorted(ids) for a, ids in self._open_at_reply.items()},
            "last_id_at_reply": self._last_id_at_reply,
            "interrupt_survivors": sorted(self.interrupt_survivors),
            "habits_noted": sorted(self.habits_noted),
            "prompt_texts": self.prompt_texts,
            "sent": self.sent,
            "place_noted": {a: ([p.host, p.user, p.via] if p else None) for a, p in self.place_noted.items()},
            "last_typed": self.last_typed,
            "shown_upto": {a: {str(k): v for k, v in seen.items()} for a, seen in self._shown_upto.items() if seen},
            "saves": self.saves,
            "cwd_noted": sorted(self.cwd_noted),
        }))

    def note_typed(self) -> None:
        """An agent's typing call (run, send, expect, shell_integrate) ended:
        for session_list's presence line."""
        self.last_typed[ACTOR.get()] = time.time()
        self._save_state()

    def note_place(self, place: "Place | None") -> None:
        """Where this agent's reply said the session is (see server.note_place)."""
        self.place_noted[ACTOR.get()] = place
        self._save_state()

    def note_sent(self) -> None:
        """A `send` typed here: input at a program's prompt from now on may
        be this agent's, not a human's (see claim_foreign_input)."""
        self.sent = (ACTOR.get(), self.offset)
        self._save_state()

    def claim_foreign_input(self) -> Region | None:
        """Someone typed at a program's prompt since the last input region
        ended there -- a human at gdb's (gdb) prompt, say, where no shell
        markers tell: make it a region of theirs, so replies and `regions`
        show it.  Input a `send` typed since then is the sending agent's."""
        last = self.regions[-1] if self.regions else None
        if last is None or last.kind != "input" or last.end is None or last.status not in ("ready", "answered"):
            return None
        row, col = last.end
        if self.term.row <= row or self.term.full_screen:
            return None
        line = self.term.line(row)[col:]
        m = self.prompt_start(line)
        typed = line[m.end():].strip() if m is not None and m.start() == 0 else ""
        if not typed:
            return None  # output after the prompt, nothing typed at it
        who = "human"
        if self.sent is not None and last.raw_end is not None and self.sent[1] >= last.raw_end:
            who = self.sent[0]
        region = self._new_region("input", who, (row + 1, 0), line.strip())
        region.raw_start = last.raw_end
        here = self.term.line(self.term.row)
        at_prompt = self.prompt_here(self.prompt_pattern)
        end = (self.term.row, (self.prompt_col(here) or 0) if at_prompt else len(here))
        region.raw_end = self.offset
        self._finish(region, max(end, region.start), "ready" if at_prompt else "settled")
        return region

    def note_interrupt_survivors(self, pids: set[int]) -> None:
        """Processes a ctrl+c was seen not to stop: later hints stop
        suggesting it."""
        if pids != self.interrupt_survivors:
            self.interrupt_survivors = pids
            self._save_state()

    async def human_active(self) -> bool:
        """Did an attached terminal send input (a key, click, scroll or focus
        change: tmux cannot tell them apart) within HUMAN_IDLE_S?"""
        t = await self.tmux.last_client_activity(self.name)
        return t is not None and time.time() - t < HUMAN_IDLE_S

    def _program_key(self) -> int | str:
        """Which program the prompt belongs to: the running shell command
        (shell integration) and the terminal's foreground process, so gdb or
        python started from a bash without integration is told from that
        bash, and a program gdb runs from gdb."""
        command = self.open_command
        base = command.id if command is not None else 0
        # The foreground process within it too: gdb's inferior, at a
        # prompt of its own ("stock> "), is not gdb.
        return f"{base}:{self.pane_command}" if self.pane_command else base

    def shell_prompt_seen(self) -> bool:
        """Has the shell at this level (the session's own, or the one the
        running command is: ssh) shown a shell prompt?  Asked of the shell,
        not of whatever runs in the foreground now (`sleep`, whose key has
        learned nothing; and the log may still trail the shell's return to
        its prompt)."""
        command = self.open_command
        base = command.id if command is not None else 0
        return any(kind == "sh" and (key == base or str(key).startswith(f"{base}:"))
                   for key, kind in self.prompt_kinds.items())

    def note_pane(self, info) -> None:
        """Take in what tmux says about the pane."""
        self.set_width(info.width)
        self.pane_command = info.current_command

    @staticmethod
    def _prompt_kind(m: re.Match) -> str | None:
        for name in PROMPT_FAMILIES:
            if name in m.re.groupindex and m.group(name) is not None:
                return name
        return None

    def short_shell_label(self, label: str | None) -> str | None:
        """A command typed at a shell without integration, labelled as an
        integrated shell's are: the prompt's sign and the command ("$ make",
        "# make"), not the whole prompt.  A prompt's user, host and
        directory may be missing, stale or wrong (a symlinked directory, an
        ssh that leaves the prompt alone); where the session is, the
        process chain says (`· on HOST`)."""
        if not label:
            return label
        m = self.prompt_start(label)
        if m is None or m.start() != 0 or self._prompt_kind(m) != "sh":
            return label
        prompt = m.group(0).rstrip()
        return f"{prompt[-1:] or '$'} {label[m.end():].lstrip()}".rstrip()

    def prompt_signature(self) -> str | None:
        """What stays the same in this shell's prompt, where no markers
        tell: its text up to and including the first ":" ("user@host:" --
        the directory after it changes), or the whole prompt ("bash-5.3$").
        Known once the shell's prompt has been seen."""
        if not self.shell_in_front():
            return None  # a REPL's prompts vary (sqlite's "...>", guile's "[1]>")
        text = self.prompt_texts.get(self._program_key())
        if not text:
            return None
        cut = min([i + 1 for i in (text.find(":"),) if i >= 0] + [i for i in (text.find(" "),) if i > 0]
                  + [len(text)])
        return text[:cut]

    def shell_in_front(self) -> bool:
        """Is the program at the prompt a shell (or ssh, whose far end is
        one)?  Only a shell's prompt is remembered by its text."""
        return os.path.basename(self.pane_command or "") in SHELLS + CARRIER_NAMES

    def unknown_prompt(self) -> str | None:
        """The shell-prompt-like text at the cursor that is not this shell's
        known prompt ("build> " printed mid-command, or a prompt that
        changed: su, a new PS1), if that is what the cursor shows."""
        signature = self.prompt_signature()
        text = self.term.cursor_text()
        m = re.compile(DEFAULT_PROMPT_PATTERN).search(text)
        if not signature or m is None or self._prompt_kind(m) != "sh" or signature in text:
            return None
        return m.group("sh").strip()

    def note_unknown_prompt(self) -> None:
        """A call ended at an unknown shell-like prompt.  The same one
        ending two calls in a row is the shell's prompt now (su, a new PS1):
        learn it."""
        text, key = self.unknown_prompt(), self._program_key()
        if text is None:
            return
        if self._prompt_candidate == (key, text):
            self.prompt_texts[key] = text
            self._prompt_candidate = None
            self._save_state()
        else:
            self._prompt_candidate = (key, text)

    def prompt_col(self, line: str) -> int | None:
        """Where the prompt starts in `line`: the leftmost match of the
        pattern -- but for a shell whose prompt is known, where that
        prompt's text is (output without a final newline comes before it).
        None: no prompt in the line."""
        m = self.prompt_start(line)
        if m is None:
            return None
        signature = self.prompt_signature()
        if signature and self._prompt_kind(m) == "sh":
            i = line.find(signature)
            return i if i >= 0 else None
        return m.start()

    def prompt_then_output(self, pattern: re.Pattern, after_row: int) -> int | None:
        """The row of a program's prompt that output followed on its own
        line -- gdb with `set debug infrun 1` prints "(gdb) [infrun]
        fetch_inferior_event: exit" after its prompt, so no prompt ends a
        line, and each `step` sat out its timeout.  Only with the cursor at the start of a line, nothing
        printed for a while, and the program waiting at the terminal."""
        term = self.term
        if term.col != 0 or term.full_screen or self.foreground not in ("read", "poll"):
            return None
        if time.monotonic() - self.last_output < PROMPT_QUIET_S:
            return None
        source = pattern.pattern[:-1] if pattern.pattern.endswith("$") else pattern.pattern
        try:
            rx = re.compile(source)
        except re.error:
            return None
        # (Only lines the input brought, and a prompt with text after it:
        # an earlier "(gdb) " above the echo is no answer.)  Within the last
        # three rows, any lines after it; further up, only gdb's debug lines
        # ("[infrun] ...", indented or not) -- killing the program under
        # `set debug infrun 1` printed five after "(gdb) ", and the answer sat out its timeout.
        for row in range(term.row - 1, max(after_row - 1, term.row - DEBUG_TAIL_ROWS - 1), -1):
            line = term.line(row)
            m = rx.match(line)
            if not (m is not None and m.end() > 0 and line[m.end():].strip()
                    and self._prompt_kind(m) not in (None, "sh")):
                continue
            if row >= term.row - 3 or all(not term.line(r).strip() or DEBUG_LINE.match(term.line(r))
                                          for r in range(row + 1, term.row)):
                return row
        return None

    def prompt_here(self, pattern: re.Pattern) -> re.Match | None:
        """The prompt at the end of the cursor line, if any -- of the kind
        this program has shown before, when that is known."""
        return self.prompt_in(self.term.cursor_text(), pattern)

    def prompt_in(self, text: str, pattern: re.Pattern) -> re.Match | None:
        m = pattern.search(text)
        if m is None:
            return None
        kind, known = self._prompt_kind(m), self.prompt_kinds.get(self._program_key())
        # Python's ">>>" where a shell's prompt was learned (python over ssh,
        # whose processes are invisible) -- not "...", which output can be.
        python = known == "sh" and kind == "py" and (PY_MAIN.match(m.group("py")) or self.in_python)
        if kind == "py" and PY_MAIN.match(m.group("py")):
            self.in_python = True
        elif kind == "sh":
            self.in_python = False
        if kind and known and kind not in PROMPT_FAMILIES[known] and (known, kind) not in PROMPT_OVERRIDES \
                and not python:
            return None
        # A shell prompt where this shell's own is known must be it: "build> "
        # printed mid-command over ssh is output.
        signature = self.prompt_signature() if kind == "sh" and not self.at_shell_prompt else None
        if signature and signature not in text:
            return None
        return m

    def learn_prompt(self, pattern: re.Pattern) -> None:
        """Note the kind of the prompt the program is at now -- and, for a
        shell's prompt, its text, the latest seen."""
        m = pattern.search(self.term.cursor_text())
        if m is not None and (kind := self._prompt_kind(m)):
            key = self._program_key()
            changed = key not in self.prompt_kinds
            if changed:
                self.prompt_kinds[key] = kind
            if self.prompt_kinds[key] == "sh" and kind == "sh" and m.group("sh") and self.shell_in_front():
                text = m.group("sh").strip()
                # Output without a final newline comes before the prompt.
                signature = self.prompt_signature()
                if signature and signature in text:
                    text = text[text.index(signature):]
                changed |= self.prompt_texts.get(key) != text
                self.prompt_texts[key] = text
            if changed:
                self._save_state()

    def at_clean_prompt(self, pattern: re.Pattern) -> bool:
        """Is the terminal at a prompt with nothing typed after it?  Then a
        human's recent click or focus change cannot collide with typing."""
        if self.term.full_screen:
            return False
        if self.at_shell_prompt:
            return not self.pending_input()
        return bool(self.prompt_here(pattern))

    # -- driving the terminal --------------------------------------------------

    def reap_gone_shells(self) -> list[Region]:
        """Commands whose shell is gone without saying so: an integrated
        shell exited (`exit 3`) into a parent that is not integrated, which
        prints no marker.  Only shells on this host can be asked."""
        gone = []
        for shell, region in list(self.open_commands.items()):
            if region.done or region.host != LOCAL_HOST or not shell.isdigit():
                continue
            try:
                os.kill(int(shell), 0)
            except ProcessLookupError:
                del self.open_commands[shell]
                region.raw_end = self.offset
                self._finish(region, self.content_end(), "done")
                if self.open_command is None:
                    self.shell_state = "unknown"  # no integrated shell left to mark prompts
                gone.append(region)
            except OSError:
                pass
        self.reaped += len(gone)
        return gone

    def close_stale_input(self, interrupted: bool = False) -> Region | None:
        """A program input left "timeout", "settled" or "waiting" whose
        program is back at its prompt (a ctrl+c in a REPL, or it finished
        later): close it there, so `regions` does not show it pending."""
        last = self.regions[-1] if self.regions else None
        if last is None or last.kind != "input" or last.status not in ("timeout", "settled", "waiting"):
            return None
        pattern = re.compile(last.prompt_pattern) if last.prompt_pattern else self.prompt_pattern
        if self.term.full_screen or self.term.row < last.start[0] or not self.prompt_here(pattern):
            return None
        last.status, last.end = "running", None
        last.raw_end = self.offset
        self._finish(last, (self.term.row, self.prompt_col(self.term.line(self.term.row)) or 0),
                     "interrupted" if interrupted else "ready")
        return last

    def pane_exited(self, info) -> None:
        """The session's own program (its shell) exited: commands still
        open end with it, with its exit status (`exit 3` never reaches a
        prompt to report it)."""
        self.sync()
        for shell, region in list(self.open_commands.items()):
            del self.open_commands[shell]
            if not region.done:
                region.exit_status = info.dead_status
                region.raw_end = self.offset
                self._finish(region, self.content_end(), "done")
        # A command already closed because its shell was found gone (see
        # reap_gone_shells) -- when that shell was the pane's own, its exit
        # status is the pane's.
        for region in self.regions[-3:]:
            if (region.kind == "command" and region.done and region.exit_status is None
                    and region.shell_pid == str(info.pid) and info.dead_status is not None):
                region.exit_status = info.dead_status
                self._save_region(region)

    async def wait(
        self, done, timeout_s: float, settle_s: float | None = None,
        prompt: re.Pattern | None = None, confirm=None, poll_s: float = POLL_S,
    ) -> str:
        """Poll until `done()` is true; return why waiting stopped.

        With `prompt`, "settled" is not declared while the screen already
        shows a prompt that the log has not caught up with.  `confirm(info)`,
        if given, may veto a done() (asked only then: it may cost a ps)."""
        deadline = time.monotonic() + timeout_s
        start = time.monotonic()
        while True:
            info = await self.tmux.pane_info(self.name)
            self.note_pane(info)
            self.sync()
            self.reap_gone_shells()
            await self.check_foreground(info)
            if done() and (confirm is None or await confirm(info)):
                return "done"
            now = time.monotonic()
            if settle_s is not None and now - max(self.last_output, start) >= settle_s:
                # Quiet for the settle time.  Not settled while the screen
                # shows a prompt the log has not caught up with, or while
                # the command is visibly still working (then the deadline
                # ends the wait as "timeout").
                ahead = prompt is not None and await self.screen_ahead(info, prompt)
                if not ahead and not await self.still_working(info, prompt):
                    return "settled"
            if info.dead:
                self.pane_exited(info)
                return "exited"
            if now >= deadline:
                return "timeout"
            await asyncio.sleep(poll_s)

    async def still_working(self, info, prompt: re.Pattern | None) -> bool:
        """Quiet for the settle time -- but is the command still running?
        Yes when the foreground process is busy (running, or sleeping on a
        timer: not reading the terminal, not at an input loop), and, in a
        shell without integration whose prompt has been seen, while nothing
        has been printed at all (a program's own prompt would be mid-line;
        ssh hides the remote processes from ps)."""
        now = time.monotonic()
        if now - self._working_checked < 0.25:
            return self._working
        self.handed_off = False
        shell_pid = info.pid if self.integrated else 0
        state = await foreground_state(info.tty, shell_pid)
        quiet_start = (self.term.col == 0 and prompt is not None and not self.at_shell_prompt
                       and (self.shell_prompt_seen() or self.prompt_kinds.get(self._program_key()) is not None))
        if quiet_start and state in ("read", "poll") and self.input_holders:
            # Waiting for input with nothing new printed, in a process the
            # program handed the terminal to: what it runs is at a prompt it
            # printed before (gdb under gdb, resumed with `continue` after a
            # ctrl+c), or reads without one.  Not working -- unless ssh,
            # whose remote side ps cannot see.  (The program itself waiting
            # in its input loop -- node awaiting a timer -- still is.)
            procs = await foreground_processes(info.tty)
            if procs and not {p.pgid for p in procs} & self.input_holders:
                quiet_start = any(is_carrier(p.pid, p.command) for p in procs)
                self.handed_off = not quiet_start
        self._working = state == "busy" or quiet_start
        self._working_checked = now
        return self._working

    def close_fullscreen_region(self, prompt: re.Pattern) -> Region | None:
        """A full-screen program started by `run` in a shell without
        integration has no end marker: once the screen is back to text and
        the prompt shows, close its region there."""
        last = self.regions[-1] if self.regions else None
        if last is None or last.kind != "input" or last.status != "fullscreen" or self.term.full_screen:
            return None
        if not self.prompt_here(prompt):
            return None
        m = self.prompt_start(self.term.line(self.term.row))
        last.status, last.end = "running", None
        last.raw_end = self.offset
        self._finish(last, (self.term.row, m.start() if m else 0), "ready")
        return last

    async def check_foreground(self, info) -> None:
        """Ask what the terminal's foreground processes are doing, when it
        matters: output has paused part-way through a line, or for a while
        at the start of one (`cat`, reading without a prompt)."""
        quiet = time.monotonic() - self.last_output
        if quiet >= (PROMPT_QUIET_S if self.term.col > 0 else PARTIAL_LINE_QUIET_S):
            # The pane's own process is the integrated shell only in a
            # session started without a command; otherwise it is the
            # program itself (gdb), whose prompt does count.
            shell_pid = info.pid if self.integrated else 0
            state = await foreground_state(info.tty, shell_pid)
            # The processes may have moved on since the log was read (the
            # command finished, and the shell is back at its prompt): read
            # it again, so the state and the text belong together.
            if self.sync():
                state = None
        else:
            state = None
        if state != self.foreground:
            self.foreground_since = time.monotonic()
        self.foreground = state

    def remote_command(self) -> Region | None:
        """The innermost command still running in a shell on another host."""
        here = short_host(LOCAL_HOST)
        remote = [r for r in self.open_commands.values() if not r.done and r.host and short_host(r.host) != here]
        return max(remote, key=lambda r: r.id) if remote else None

    def at_plain_shell_prompt(self) -> bool:
        """Does the cursor line show a shell's prompt (not a program's)?"""
        m = DEFAULT_PROMPT_RE.search(self.term.cursor_text())
        return m is not None and self._prompt_kind(m) == "sh"

    async def where(self, info=None, table: dict | None = None) -> Place | None:
        """Where the session's next line would run, if not here as this
        user: from the process chain between the foreground job and the
        session's own shell (an ssh on it: that host, from its command
        line; a shell of another user: sudo -i, su -), and from an
        integrated shell's prompt markers, which also see past the first
        ssh (a second hop, a remote su).  None: this host, this user."""
        if info is None:
            info = await self.tmux.pane_info(self.name)
        if info.dead:
            return None
        place, remote_pid = await place_of_processes(info.tty, info.pid, table)
        marked = self.prompt_place
        if marked is not None and self.shell_state not in ("prompt", "input") and self.at_plain_shell_prompt():
            # The marked shell's last command never ended, and a shell
            # prompt without markers shows: it exited (`exit` from a remote
            # `sudo -i`), into a shell that says nothing.
            marked = self.prompt_place = None
        if remote_pid is not None:
            # A hop made from an integrated shell over there (`ssh B` on A,
            # `sudo -i`): its command line says where the next one is, when
            # nothing newer (B's own markers) does.
            hop = self.remote_command()
            if hop is not None and (marked is None or tuple(hop.start) >= tuple(marked.at)):
                if (goes := command_place(hop.command or "", short_host(hop.host), hop.user)) is not None:
                    return goes
            if self._remote is None or self._remote[0] != remote_pid:
                self._remote = (remote_pid, self.content_end())
            # Markers count from where the ssh started: the local command
            # that runs it (its region, so a restarted server knows too),
            # else where this server first saw it.
            here = short_host(LOCAL_HOST)
            local = [r for r in self.open_commands.values() if not r.done and short_host(r.host or here) == here]
            outer = max(local, key=lambda r: r.id) if local else None
            since = outer.start if outer is not None else self._remote[1]
            if marked is not None and tuple(marked.at) >= tuple(since) and marked.host != short_host(LOCAL_HOST):
                return marked  # an integrated shell over there says where it is
            return place
        self._remote = None
        if place is not None:
            return place
        if marked is not None and marked.host == short_host(LOCAL_HOST) and (marked.user or LOCAL_USER) != LOCAL_USER:
            return marked
        return None

    async def screen_ahead(self, info, prompt: re.Pattern) -> bool:
        """Does tmux's screen show a prompt at the cursor that the log does
        not show yet?  (The log can trail the screen.)"""
        if self.term.full_screen or self.prompt_here(prompt):
            return False
        row = await self.tmux.capture_row(self.name, info.cursor_y)
        return bool(self.prompt_in(row[: info.cursor_x], prompt))

    def awaiting_input(self, pattern: re.Pattern) -> bool:
        """Does the foreground program look like it is waiting at a prompt?

        Output has paused part-way through a line, and the foreground
        processes are waiting for input: blocked reading the terminal (a
        question), or in an input loop with the line ending in a prompt (or
        quiet for a while).  A build pausing mid-line ("checking for x...")
        is not waiting: its processes are running or sleeping on a timer.
        Where the processes cannot be seen, the line and the pause decide.
        At the start of a line, only a process blocked reading the terminal
        counts (`cat`, `read` with no prompt): an input loop there is as
        likely ssh or an idle program as a question.
        """
        term = self.term
        quiet = time.monotonic() - self.last_output
        if term.col == 0:
            # Nothing on the line to go by: the processes alone decide, so
            # their state must hold, not flicker -- a busy `make -j check`
            # has moments that look like an input loop (a process exiting
            # mid-check, a brief poll), and one such check ended the call.
            if time.monotonic() - self.foreground_since < FOREGROUND_STABLE_S:
                return False
            # A command its shell's markers say is running on another host
            # (or in a container): the processes seen here are the ssh that
            # carries it, idle in its input loop whatever the command does
            # -- a silent build over there is not waiting for input.
            if self.remote_command() is not None:
                return False
            # (The session's own shell reading while a command is open is
            # `read -e`: the builtin, in readline.)
            reading = self.foreground == "read" or (self.foreground == "shell" and self.open_command is not None)
            # An input loop with nothing printed for a while: waiting for
            # input (picocom, minicom) -- or an idle server; either way the
            # call has nothing more to wait for.
            idle = self.foreground == "poll" and quiet >= DEFAULT_SETTLE_MS / 1000
            return (reading and quiet >= PARTIAL_LINE_QUIET_S) or idle
        if quiet < PROMPT_QUIET_S or self.foreground in ("busy", "shell"):
            return False
        if self.foreground == "read":
            return True
        if self.prompt_here(pattern):
            return True
        # A command running on another host: the processes seen here are
        # the ssh that carries it, so a pause mid-line is all there is to
        # go by -- and a test run's dots or a build's "checking x..." pause
        # too.  There, only a line shaped like a question counts.
        if self.remote_command() is not None and not self.term.cursor_text().rstrip().endswith(QUESTION_ENDS):
            return False
        return quiet >= PARTIAL_LINE_QUIET_S

    async def run_shell_command(
        self, line: str, pattern: re.Pattern, timeout_s: float, until: re.Pattern | None = None,
    ) -> tuple[list[Region], str, int]:
        """Submit `line` at the shell prompt and wait for it to finish, or for
        the program it started to stop at a prompt of its own.

        Several lines go one at a time: each must bring the shell back to
        its prompt, or to its continuation prompt (inside a heredoc, a
        quote, a block), before the next is typed.  A line that asks a
        question, runs past the timeout or starts a program never gets the
        lines after it as typed-ahead input: they are not sent.  Returns the
        command regions, why waiting stopped, and how many lines were not
        sent."""
        parts = line.removesuffix("\n").split("\n")
        before = len(self.regions)
        deadline = time.monotonic() + timeout_s
        watch = UntilWatch(self, until)
        poll_s = POLL_S / 5 if len(parts) > 1 else POLL_S
        why, sent = "done", 0
        self.model_driving = ACTOR.get()
        try:
            for i, part in enumerate(parts):
                n0, prompts, continuations = len(self.regions), self.prompts, self.continuations
                deadline += await self.type_line(part)  # (typing like a person takes its time)
                await self.tmux.send_keys(self.name, ["Enter"])
                sent += 1
                why = await self._await_line(n0, prompts, continuations, pattern,
                                             max(deadline - time.monotonic(), 0.1), watch, poll_s)
                # Echo off (stty -echo): the label is what was typed.
                for r in self.regions[n0:]:
                    if r.kind == "command" and not r.command and r.source == ACTOR.get():
                        r.command = part.strip() or None
                        self._save_region(r)
                if i + 1 < len(parts) and why not in ("prompt", "continuation"):
                    break
        finally:
            self.model_driving = None
        regions = [r for r in self.regions[before:] if r.kind == "command"]
        if why in ("waiting", "fullscreen"):
            # The program it started is at its prompt: learn it -- unless the
            # processes are out of sight (a command in a remote shell), where
            # prompt-like output ("user@host:~$ ") cannot be told from one;
            # the connection itself (`ssh host`) is at the remote shell's.
            first = (regions[-1].command or "").split() if regions else []
            if why == "waiting" and (self.pane_command not in CARRIER_NAMES
                                     or (first and os.path.basename(first[0]) in CARRIER_NAMES)):
                self.learn_prompt(pattern)
            why = "done"
        elif why == "prompt":
            why = "done"
        if self.at_shell_prompt:
            self.note_model_input()  # (an incomplete command of its own, at PS2)
        else:
            self.forget_model_input()  # a program it started is at its prompt
        return regions, why, len(parts) - sent

    async def _await_line(
        self, n0: int, prompts: int, continuations: int, pattern: re.Pattern,
        timeout_s: float, watch: UntilWatch, poll_s: float,
    ) -> str:
        """Wait for one submitted line: "prompt" (the shell is back at its
        prompt), "continuation" (at its continuation prompt), "waiting" (the
        command waits for input), "fullscreen", "until", or wait's reasons."""
        state: list[str] = []
        reaped = self.reaped

        def settled() -> bool:
            new = [r for r in self.regions[n0:] if r.kind == "command"]
            quiet = time.monotonic() - self.last_output
            if self.shell_state == "input" and self.prompts > prompts and all(r.done for r in new):
                state.append("prompt")
            elif self.reaped > reaped and all(r.done for r in new):
                state.append("prompt")  # the shell exited (into one without integration)
            elif (self.shell_state == "input" and self.continuation and self.continuations > continuations
                  and not new):
                state.append("continuation")
            elif new and not new[-1].done and watch(new[0].start):
                state.append("until")
            elif new and not new[-1].done and self.awaiting_input(pattern):
                state.append("waiting")
            elif new and not new[-1].done and self.term.full_screen and quiet >= PROMPT_QUIET_S:
                state.append("fullscreen")  # a full-screen program has drawn its screen
            return bool(state)

        why = await self.wait(settled, timeout_s, poll_s=poll_s)
        return state[-1] if why == "done" else why

    async def run_program_input(
        self, line: str, pattern: re.Pattern, timeout_s: float, settle_s: float | None,
        until: re.Pattern | None = None,
    ) -> tuple[Region, Region | None]:
        """Send `line` to a foreground program and wait for its next prompt.

        Like a shell command's region, the input region holds only output:
        it starts after the echoed input line(s); the prompt and input become
        its `command`.  Returns the region and the enclosing shell command,
        if any (it may have finished because the program exited).
        """
        self.sync()
        send_row = self.term.row
        # Text before the cursor that is output of a busy program (progress)
        # is not part of the input's label; a question's text is.
        send_col = 0
        if self.term.col > 0:
            info = await self.tmux.pane_info(self.name)
            shell_pid = info.pid if self.integrated else 0
            if await foreground_state(info.tty, shell_pid) in ("busy", "shell"):
                send_col = self.term.pos[1]
        start = (send_row, 0)
        # Who holds the terminal as the line is typed: the program at its
        # prompt.  If that changes (gdb's `continue` hands it to the program
        # it debugs), a quiet wait for input is the other program's (still_working).
        pane = await self.tmux.pane_info(self.name)
        self.input_holders = {p.pgid for p in await foreground_processes(pane.tty)} or None
        place = await self.where(pane)
        asked = self.regions[-1] if self.regions else None
        if (asked is not None and asked.kind == "input" and asked.status in ("timeout", "settled", "until")
                and self.prompt_here(pattern)):
            # It came back to its prompt while nobody waited (a statement left
            # blocked on purpose with a short timeout_s, done once the other
            # session committed): close it there, not leave it "timeout" for
            # good -- session_close reported ending it, minutes later (the bug
            # hunt's pg-rowmark).
            row = self.term.row
            asked.status, asked.end = "running", None
            self._finish(asked, max((row, self.prompt_col(self.term.line(row)) or 0), asked.start), "ready")
        region = self._new_region("input", ACTOR.get(), start, None)
        if place is not None:
            region.host, region.user = place.host, place.user
        if asked is not None and asked.kind == "input" and asked.status == "waiting":
            asked.status, asked.answered_by = "answered", region.id
            self._save_region(asked)
            send_col = 0  # an answer: the question stays in its label
        region.raw_start = self.offset
        if pattern.pattern != self.prompt_pattern.pattern:
            region.prompt_pattern = pattern.pattern
        enclosing = self.open_command
        # A command the input starts at a shell the markers had not shown
        # (one just integrated) is the caller's too.
        self.model_driving = ACTOR.get()
        try:
            return await self._run_program_input(region, line, pattern, timeout_s, settle_s,
                                                 send_row, send_col, start, enclosing, until)
        finally:
            self.model_driving = None

    async def _run_program_input(
        self, region: Region, line: str, pattern: re.Pattern, timeout_s: float, settle_s: float | None,
        send_row: int, send_col: int, start: Pos, enclosing: Region | None,
        until: re.Pattern | None = None,
    ) -> tuple[Region, Region | None]:
        echo_rows = await self.type_program_input(line)
        asked = False
        watch = UntilWatch(self, until)
        matched: list[str] = []  # what ended the wait: "prompt", "until"

        def question() -> bool:
            """Stopped mid-line, waiting for input, and not at the prompt:
            a question (gdb's "(y or n)").  Without a settle time (the call
            named the prompt) nothing else would end the wait before the
            timeout."""
            term = self.term
            quiet = time.monotonic() - self.last_output
            if term.col == 0:  # nothing printed since: only a blocked read tells
                return term.row > start[0] and self.foreground == "read" and quiet >= PARTIAL_LINE_QUIET_S
            if settle_s is not None or term.row <= start[0]:
                return False
            if self.foreground == "read":
                return quiet >= PROMPT_QUIET_S
            return self.foreground == "poll" and quiet >= DEFAULT_SETTLE_MS / 1000

        def prompted() -> bool:
            nonlocal asked
            if enclosing is not None and enclosing.done:
                return True
            term = self.term
            if term.full_screen and time.monotonic() - self.last_output >= PROMPT_QUIET_S:
                return True  # a full-screen program has drawn its screen
            matched.clear()
            # Past the echoed input: a pasted block's own "..." lines are
            # not the prompt that follows it.
            if term.row >= start[0] + echo_rows and self.prompt_here(pattern):
                matched.append("prompt")
                return True
            if term.row > start[0] + echo_rows and self.prompt_then_output(pattern, start[0] + echo_rows) is not None:
                matched.append("prompt-then-output")
                return True
            if watch((send_row + echo_rows, 0)):
                matched.append("until")
                return True
            asked = question()
            return asked

        async def at_prompt_really(info) -> bool:
            """Prompt text printed while the foreground is busy is output
            (an inferior writing "(gdb) "), not the program's prompt."""
            if matched != ["prompt"]:
                return True
            state = await foreground_state(info.tty, info.pid if self.integrated else 0)
            # Output that arrived meanwhile (the program finished; gdb's real
            # prompt follows) makes the text just judged stale: look again.
            return state != "busy" and not self.sync()

        deadline = time.monotonic() + timeout_s
        why = await self.wait(prompted, timeout_s, settle_s, prompt=pattern, confirm=at_prompt_really)
        if (why == "done" and matched == ["prompt"] and "\n" not in line
                and PY_CONT.fullmatch(self.term.cursor_text().strip())
                and PY_MAIN.match(self.echoed_input(send_row, 1, send_col) or "")):
            # A one-line compound statement ("for x in y: print(x)"): Python
            # waits for the empty line that ends the block.
            row = self.term.row
            await self.tmux.send_keys(self.name, ["Enter"])
            echo_rows += 1
            why = await self.wait(
                lambda: (enclosing is not None and enclosing.done)
                or (self.term.row > row and bool(self.prompt_here(pattern))),
                max(deadline - time.monotonic(), 0.1), settle_s, prompt=pattern)
        if why == "done" and matched == ["until"]:
            end = self.content_end()
            why = "until"
        elif why == "done" and matched == ["prompt-then-output"]:
            end = self.content_end()  # (the output after the prompt is this input's too)
            why = "ready"
        elif why == "done" and asked:
            end = self.content_end()
            why = "waiting"
        elif enclosing is not None and enclosing.done and enclosing.end is not None:
            end = max(start, enclosing.end)  # the command it talked to ended
            why = "finished"
        elif why == "done" and self.term.full_screen:
            end = self.content_end()
            why = "fullscreen"
        elif why == "done":
            # The new prompt belongs to the next input; output printed
            # before it on the same line (no final newline) stays here.
            end = (self.term.row, self.prompt_col(self.term.line(self.term.row)) or 0)
            why = "ready"
            self.learn_prompt(pattern)
        else:
            end = self.content_end()
        if why == "settled" and self.term.col > 0 and self.foreground not in ("busy", "shell"):
            why = "waiting"  # stopped mid-line, not busy: most likely a question
            self.note_unknown_prompt()
        joined_echo = False
        if echo_rows == 1 and "\n" not in line:
            rows, echoed = self.echo_extent(send_row, send_col, line)
            joined_echo = rows > 1
            echo_rows = rows  # (a program that breaks its echo with real newlines: busybox ash at 80 columns)
            if echoed and len(echoed) < len(line.rstrip()) and len(echoed) >= min(20, len(line)):
                region.echo_cut = [len(echoed), len(line.rstrip())]
        region.command = self.echoed_input(send_row, echo_rows, send_col)
        if joined_echo and region.command:
            region.command = region.command.replace("\n", "")  # one typed line, broken by the program
        region.command = self.short_shell_label(region.command)
        region.start = min((send_row + echo_rows, 0), end)
        region.raw_end = self.offset
        self._finish(region, end, why)
        self.forget_model_input()
        return region, enclosing

    async def run_expect(
        self, input: str | None, rules: list[Rule], until: re.Pattern | None,
        prompt: re.Pattern, timeout_s: float, key_names: dict[int, list[str]],
    ) -> ExpectResult:
        """See _run_expect; commands started meanwhile are the model's.

        Several lines go one at a time, as with `run`: each must bring the
        program back to its prompt (at the shell: the shell back at its
        prompt, or its continuation prompt) before the next is sent, so a
        question raised by one line is answered by the rules, not by the
        lines after it."""
        self.model_driving = ACTOR.get()
        try:
            self.sync()
            if input is None or "\n" not in input or self.will_paste(input):
                return await self._run_expect(input, rules, until, prompt, timeout_s, key_names)
            at_shell = self.at_shell_prompt
            lines = input.removesuffix("\n").split("\n")
            deadline = time.monotonic() + timeout_s
            results: list[ExpectResult] = []
            for i, line in enumerate(lines):
                left = max(deadline - time.monotonic(), 0.1)
                results.append(await self._run_expect(line, rules, until, prompt, left, key_names))
                back = (results[-1].outcome in ("done", "ready") and self.at_shell_prompt if at_shell
                        else results[-1].outcome == "ready")
                if not back and i + 1 < len(lines):
                    n = len(lines) - i - 1
                    results[-1].notes.append(
                        f"[stopped: {n} more line{'s' if n > 1 else ''} not sent, "
                        "since this one did not come back to the prompt]")
                    break
            return ExpectResult(
                results[-1].outcome, [t for r in results for t in r.transcript], results[-1].unused,
                [g for r in results for g in r.regions], results[0].text_start,
                [n for r in results for n in r.notes])
        finally:
            self.model_driving = None

    async def _run_expect(
        self, input: str | None, rules: list[Rule], until: re.Pattern | None,
        prompt: re.Pattern, timeout_s: float, key_names: dict[int, list[str]],
    ) -> ExpectResult:
        """Optionally type `input`, then answer whatever matches a rule, until
        the command finishes, the program is back at its prompt, `until`
        matches, a stop rule matches, or output stops at a question no rule
        answers -- the case to hand back rather than time out on.

        Rules match the cooked output not yet consumed: each firing consumes
        everything up to the current end, so one question is answered once.
        """
        self.sync()
        send_row = self.term.row
        at_shell = self.at_shell_prompt
        before = len(self.regions)
        enclosing = self.open_command
        region = None
        started_offset = self.offset
        prompts = (self.prompts, self.continuations)  # a new one, and no command: the shell took the line
        echo_rows = 1
        if input is not None:
            if not at_shell:
                place = await self.where()
                region = self._new_region("input", ACTOR.get(), (send_row, 0), None)
                region.raw_start = self.offset
                if place is not None:
                    region.host, region.user = place.host, place.user
                echo_rows = await self.type_program_input(input)
            else:
                await self.tmux.send_text(self.name, input, literal_tabs=True)
                await self.tmux.send_keys(self.name, ["Enter"])
            consumed = (send_row + max(echo_rows, 1 + input.count("\n")), 0)
        else:
            # Nothing typed: a question already waiting (on the cursor line,
            # or in the input region left waiting) is unanswered output too.
            consumed = (self.term.row, 0)
            last = self.regions[-1] if self.regions else None
            if last is not None and last.kind == "input" and last.status in ("waiting", "settled", "timeout"):
                consumed = min(consumed, last.start)
        text_start = consumed
        scan_from = consumed  # rows before this were already searched
        transcript: list[str] = []
        deadline = time.monotonic() + timeout_s
        outcome = None

        while outcome is None:
            info = await self.tmux.pane_info(self.name)
            self.note_pane(info)
            self.sync()
            await self.check_foreground(info)
            term = self.term
            end = self.content_end()
            # Search only rows not searched before (with an overlap, for
            # patterns spanning lines), so a long build is not rescanned
            # on every poll.
            text_from = max(consumed, scan_from)
            text = self.unconsumed_text(text_from)
            quiet = time.monotonic() - self.last_output

            fired = False
            for i, rule in enumerate(rules):
                if rule.fired >= rule.times:
                    continue
                m = rule.pattern.search(text)
                if not m:
                    continue
                rule.fired += 1
                row = text_from[0] + text.count("\n", 0, m.start())
                what = term.line(row).strip()
                consumed = end
                started = [r for r in self.regions[before:] if r.kind == "command"]
                gone = (enclosing is not None and enclosing.done) or (
                    at_shell and started and all(r.done for r in started))
                if not gone and (enclosing is not None or at_shell) and self.integrated:
                    # The markers may not have arrived yet: ask ps whether the
                    # session's own shell is already back at its prompt.
                    gone = await foreground_state(info.tty, info.pid) == "shell"
                if gone and rule.action != "stop":
                    # The program asking has exited: an answer would go to
                    # the shell as a command.
                    transcript.append(f"line {row + 1}: {what!r} -> not sent: the program had exited")
                    outcome = "finished"
                    break
                transcript.append(f"line {row + 1}: {what!r} -> {rule.describe_action(key_names.get(i))}")
                if rule.action == "stop":
                    outcome = "stopped"
                elif rule.action == "keys":
                    await self.tmux.send_keys(self.name, key_names[i])
                else:
                    await self.tmux.send_text(self.name, rule.value, literal_tabs=True)
                    if rule.action == "line":
                        await self.tmux.send_keys(self.name, ["Enter"])
                if rule.then_stop:
                    outcome = "stopped"
                fired = True
                break
            if outcome is not None:
                break
            if fired:
                await asyncio.sleep(POLL_S)
                continue

            scan_from = max(consumed, (max(0, term.row - EXPECT_OVERLAP_ROWS), 0))
            progressed = self.offset > started_offset
            at_prompt = (
                progressed and term.col > 0 and quiet >= PROMPT_QUIET_S
                and self.foreground not in ("busy", "shell") and bool(self.prompt_here(prompt))
            )
            commands = [r for r in self.regions[before:] if r.kind == "command"]
            if until is not None and until.search(text):
                outcome = "until"
            elif at_shell and input is not None and self.shell_state == "input" and (
                    (commands and all(r.done for r in commands))
                    or (not commands and (self.prompts, self.continuations) != prompts)):  # PS2, a comment
                outcome = "done"
            elif enclosing is not None and enclosing.done:
                outcome = "finished"
            elif at_prompt and not (at_shell and commands and not commands[-1].done
                                    and self.shell_state == "input"):
                outcome = "ready"
            elif term.full_screen and progressed and quiet >= PROMPT_QUIET_S:
                outcome = "fullscreen"
            elif progressed and not at_prompt and self.awaiting_input(prompt):
                outcome = "unanswered"
            elif info.dead:
                self.pane_exited(info)
                outcome = "exited"
            elif time.monotonic() >= deadline:
                outcome = "timeout"
            else:
                await asyncio.sleep(POLL_S)

        regions = [r for r in self.regions[before:] if r.kind == "command"]
        if outcome == "ready":
            self.learn_prompt(prompt)
        if region is not None:
            if outcome == "ready":
                end = (self.term.row, 0)
            elif outcome == "finished" and enclosing is not None and enclosing.end is not None:
                end = enclosing.end
            else:
                end = self.content_end()
            status = {"ready": "ready", "until": "ready", "finished": "finished", "exited": "exited",
                      "timeout": "timeout", "unanswered": "waiting", "stopped": "waiting"}.get(outcome, "settled")
            region.command = self.echoed_input(send_row, echo_rows)
            more = input.split("\n")[1:]
            if echo_rows == 1 and more and region.command:
                # Typed together: one region; its label counts every line.
                region.command += "\n" + "\n".join(more)
            region.start = min((send_row + echo_rows, 0), end)
            region.raw_end = self.offset
            self._finish(region, end, status)
            regions = [region]
        if self.at_shell_prompt:
            self.note_model_input()
        else:
            self.forget_model_input()
        unused = [r.source or r.pattern.pattern for r in rules if r.fired == 0]
        return ExpectResult(outcome, transcript, unused, regions, text_start)

    def unconsumed_text(self, start: Pos) -> str:
        """Output from `start` to the cursor, for matching expect rules.

        Unlike the cooked text, the cursor line keeps its trailing blanks,
        so a rule can match a prompt exactly as shown ("Password: $").
        """
        term = self.term
        if term.pos <= start:
            return ""
        if term.row == start[0]:
            return term.cursor_text()[start[1] :]
        head = term.text_between(start, (term.row, 0))
        return head + term.cursor_text()

    async def resume_input(self, region: Region, pattern: re.Pattern, timeout_s: float,
                           until: re.Pattern | None = None, scan_from: Pos | None = None) -> str:
        """Keep waiting on a program input that timed out or settled, until
        the program is back at its prompt (or asks something), or `until`
        appears in output from `scan_from` on, and close the region there.
        Returns the region's new status."""
        watch = UntilWatch(self, until)
        scan_from = scan_from or region.start
        region.status, region.end = "running", None
        # The command the input went to (an ssh, gdb's shell command): the
        # one open now -- or, if it ended before this call looked (an ssh
        # dropped while nobody waited), the one that was open when the
        # input started; else the shell's prompt after it read as "ready".
        enclosing = self.open_command or next(
            (r for r in reversed(self.regions) if r.kind == "command" and r.start <= region.start
             and r.done and r.end is not None and r.end >= region.start), None)

        def back() -> bool:
            if enclosing is not None and enclosing.done:
                return True
            if self.term.full_screen:
                return True
            # (The region starts after the echoed input: a prompt on its
            # first row is new -- a command that printed nothing.)
            if self.term.row >= region.start[0] and self.prompt_here(pattern):
                return True
            if self.term.row > region.start[0] and self.prompt_then_output(pattern, region.start[0]) is not None:
                return True
            if watch(scan_from):
                return True
            return self.foreground == "read" and self.term.col > 0  # a question

        why = await self.wait(back, timeout_s, prompt=pattern)
        if why == "done" and watch.hit and not (enclosing is not None and enclosing.done) \
                and not self.prompt_here(pattern):
            end, why = self.content_end(), "until"
        elif enclosing is not None and enclosing.done and enclosing.end is not None:
            end, why = max(region.start, enclosing.end), "finished"
        elif why == "done" and self.prompt_here(pattern):
            end, why = (self.term.row, self.prompt_col(self.term.line(self.term.row)) or 0), "ready"
        elif why == "done" and self.prompt_then_output(pattern, region.start[0]) is not None:
            end, why = self.content_end(), "ready"
        else:
            end = self.content_end()
            why = {"done": "waiting"}.get(why, why)
        region.raw_end = self.offset
        self._finish(region, end, why)
        return why

    async def type_program_input(self, text: str) -> int:
        """Send a program `text` + Enter; returns how many rows its echo
        takes.

        Several lines for Python's REPL go as one bracketed paste: typed,
        its auto-indentation would add to the text's own.  The paste ends
        with a newline, and Enter on the empty line after it runs the block.
        (Other programs, gdb among them, also ask for bracketed paste but
        take typed lines as separate commands, which is what is wanted.)
        """
        if self.will_paste(text):
            before = self.offset
            await self.tmux.paste(self.name, text + "\n")
            # Enter once the paste is echoed: arriving with it (over ssh)
            # it is taken as part of the paste, and the block does not run.
            await self.wait(lambda: self.offset > before and time.monotonic() - self.last_output >= 0.1, 2)
            await self.tmux.send_keys(self.name, ["Enter"])
            return text.count("\n") + 2  # each line, and the empty one that ran it
        await self.type_line(text)
        await self.tmux.send_keys(self.name, ["Enter"])
        # Typed lines are answered one by one: only the first is sure to be
        # echoed before any output.
        return 1

    async def type_line(self, text: str) -> float:
        """Type `text` (no Enter): at once, or as a person would when the
        call asked for that (self.typing).  Returns the seconds it took."""
        if not self.typing:
            await self.tmux.send_text(self.name, text, literal_tabs=True)
            return 0.0
        return await type_out(self.tmux, self.name, text, **self.typing)

    def will_paste(self, text: str) -> bool:
        """Is `text` several lines for Python's REPL, sent as one paste?"""
        at_python = bool(PY_AT_END.search(self.term.cursor_text().rstrip()))
        lines = text.split("\n")
        # A block (indented lines, a line ending in ":"), not independent
        # lines, each of whose values should show.
        block = any(l[:1].isspace() for l in lines[1:]) or any(l.rstrip().endswith(":") for l in lines)
        return "\n" in text and self.term.bracketed_paste and at_python and block

    def echo_extent(self, send_row: int, send_col: int, typed: str) -> tuple[int, str]:
        """How many log lines the echo of one typed line takes, and the echo
        itself joined: a program may break a long echo with real line
        breaks (busybox ash at the terminal's width), and may cut the input
        short (its line buffer).  (1, "") when the echo is not the input's
        start (no echo, or edited)."""
        typed = typed.rstrip()
        first = self.term.line(send_row) if send_row < len(self.term.lines) else ""
        # Where the echo starts: after the prompt, whatever it looks like.
        at = first.find(typed[:20]) if typed else -1
        piece = first[at:] if at >= 0 else ""
        if not piece or not typed.startswith(piece.rstrip()):
            return 1, ""
        joined, rows = piece, 1
        while len(joined.rstrip()) < len(typed) and send_row + rows < len(self.term.lines) - 1:
            nxt = self.term.line(send_row + rows)
            if not nxt or not typed.startswith(joined + nxt.rstrip()) and not typed.startswith((joined + nxt).rstrip()):
                break
            joined, rows = joined + nxt, rows + 1
        return rows, joined.rstrip()

    def echoed_input(self, send_row: int, rows: int, send_col: int = 0) -> str | None:
        """The prompt and input as echoed: the label of an input region."""
        last = min(send_row + rows, len(self.term.lines))
        lines = [line.rstrip() for _, line in self.iter_lines(range(send_row, last))]
        col = self.prompt_col(lines[0]) if lines else None
        m = self.prompt_start(lines[0]) if lines else None
        if col:
            lines[0] = lines[0][col:]  # output before the prompt is not input
        elif lines and not m and send_col:
            lines[0] = lines[0][send_col:]  # typed after output, with no prompt
        # The empty "..." line that ran a pasted block is not input.
        while len(lines) > 1 and lines[-1].strip() in ("", "...", "...:", ">>>"):
            lines.pop()
        return "\n".join(lines).strip() or None

    def summary(self, region: Region) -> dict:
        rows = self.region_rows(region)
        return {
            "id": region.id,
            "kind": region.kind,
            "source": region.source,
            "command": region.command,
            "status": region.status,
            "exit_status": region.exit_status,
            "lines": f"{rows.start + 1}-{rows.stop}" if rows else None,
            "file": self.region_location(region),
        }
