"""Table-driven regression tests: one scenario per reproduced finding.

A scenario is a session to create and a list of steps.  Each step is one
tool call with the text its reply must contain ("has"), must not contain
("lacks"), whether it must fail ("error"), and how long it may take
("max_s").  "needs" lists what it needs beyond tmux and bash (needs.py);
without it the scenario is skipped.  "{home}" in any string is the test's
directory.  Sessions get a HOME whose login profile sets PS1='$ ', like the
other end-to-end tests.

Findings from sub-agent rounds and test rounds go here when a few calls
reproduce them; anything needing more logic (two agents, a human typing)
goes into test_server.py.
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from needs import needs
from test_server import Term, plain_home

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="needs tmux")

PLAIN_BASH = "bash --norc --noprofile -i"

SCENARIOS = [
    {
        "name": "cat-reading-with-no-prompt-is-waiting",
        "source": "smoke test follow-up",
        "session": {},
        "steps": [
            {"call": "run", "args": {"input": "cat", "timeout_s": 30}, "max_s": 6,
             "has": ["waiting for input (it is reading the terminal, with no prompt printed)"]},
            {"call": "send", "args": {"keys": ["ctrl+d"]}, "has": ["$ cat · exit 0"]},
        ],
    },
    {
        "name": "heredoc-with-an-explicit-prompt-pattern",
        "source": "evaluation, E2",
        "session": {"command": PLAIN_BASH, "cwd": "{home}"},
        "steps": [
            {"call": "run", "args": {"input": "cat > hd <<'EOF'\none\ntwo\nEOF", "prompt_pattern": r"\$ $",
                                     "timeout_s": 30},
             "max_s": 10, "lacks": ["not sent"]},
            {"call": "run", "args": {"input": "wc -l < hd", "prompt_pattern": r"\$ $"}, "has": ["2"]},
        ],
    },
    {
        "name": "gdb-ex-without-batch-pending-breakpoint-via-expect", "needs": ["gdb", "cc"],
        "source": "guide advice: checks the documented call works",
        "session": {"cwd": "{home}"},
        "steps": [
            {"call": "run", "args": {"input": "printf 'int main(void){return 0;}\\n' > m.c && cc -g -o m m.c"},
             "has": ["exit 0"]},
            {"call": "expect", "args": {
                "input": "DEBUGINFOD_URLS= gdb -q -nx -ex 'break nosuch' -ex 'break main' -ex run -ex bt ./m",
                "respond": [{"pattern": "Make breakpoint pending", "line": "y"}], "timeout_s": 30},
             "max_s": 20, "has": ["[expect: the program is at its prompt · 1 response]", "sent line 'y'",
                                  "#0  main"]},
            {"call": "run", "args": {"input": "print 6*7"}, "has": ["· ready ·", "$1 = 42"]},
            {"call": "expect", "args": {"input": "quit",
                                        "respond": [{"pattern": "Quit anyway", "line": "y"}]},
             "has": ["exit 0"]},
        ],
    },
    # -- sub-agent defect hunt (F numbers from its report) --
    {"name": "F8-trailing-semicolons-survive", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "touch zz; find . -maxdepth 1 -name zz -exec echo found {} \\;"},
                "has": ["found ./zz", "exit 0"], "lacks": ["missing argument"]},
               {"call": "run", "args": {"input": "echo last;"}, "has": ["$ echo last;"]},
               {"call": "send", "args": {"text": "echo semi;", "keys": ["enter"]}, "has": ["echo semi;"]}]},
    {"name": "F2-background-output-at-idle-prompt-is-not-typed-input", "session": {},
     "steps": [{"call": "run", "args": {"input": "(sleep 1; echo LATE) &"}},
               {"call": "expect", "args": {"respond": [], "until": "LATE", "timeout_s": 5}},
               {"call": "run", "args": {"input": "echo mine"}, "has": ["$ echo mine", "exit 0"],
                "lacks": ["$ LATE", "discarded"]}]},
    {"name": "F2b-background-output-then-run-directly", "session": {},
     "steps": [{"call": "run", "args": {"input": "(sleep 1; echo LATE) &"}},
               {"call": "run", "args": {"input": "sleep 2"}},
               {"call": "run", "args": {"input": "echo mine"}, "has": ["$ echo mine", "exit 0"]}]},
    {"name": "F14-alt-screen-left-on-by-finished-command", "session": {},
     "steps": [{"call": "run", "args": {"input": "printf '\\e[?1049h'; echo inside-alt", "timeout_s": 10},
                "has": ["exit 0"]},
               {"call": "run", "args": {"input": "echo after", "timeout_s": 10}, "has": ["after", "exit 0"]}]},
    {"name": "F13-fake-osc133-in-output-does-not-end-command", "session": {},
     "steps": [{"call": "run", "args": {"input": "printf '\\e]133;D;42;pid=%s\\a' $$; sleep 1; echo real-output; false",
                                        "timeout_s": 10},
                "has": ["real-output", "exit 1"], "lacks": ["exit 42"]}]},
    {"name": "F5-gdb-inferior-printing-fake-prompt", "needs": ["gdb", "python3", "/proc syscall"],
     "session": {"command": "gdb -q -nx -iex 'set debuginfod enabled off' --args python3 -c \"import time,sys; sys.stdout.write('(gdb) '); sys.stdout.flush(); time.sleep(3); print('really-done')\""},
     "steps": [{"call": "run", "args": {"input": "run", "timeout_s": 20}, "has": ["really-done", "exited normally"]}]},
    {"name": "F6-session-env-does-not-leak-server-venv",
     "session": {"command": "sh -c 'echo VE=[$VIRTUAL_ENV]; exec cat'"},
     "steps": [{"call": "read", "args": {}, "has": ["VE=[]"]}]},
    {"name": "F10-tab-in-run-input-is-literal", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "touch fake fake.c; echo fak\tx; printf '[%s]\\n' \"a\tb\""},
                "lacks": ["fakex", "[ab]"]}]},
    {"name": "F4-python-one-line-compound-statement-runs", "needs": ["python3"], "session": {"command": "python3 -q"},
     "steps": [{"call": "run", "args": {"input": "for x in range(3): print(x)", "timeout_s": 10}, "has": ["0\n1\n2"]},
               {"call": "run", "args": {"input": "print('next')"}, "has": ["next", "· ready ·"]}]},
    {"name": "F12-typeahead-line-that-runs-is-reported", "session": {},
     "steps": [{"call": "run", "args": {"input": "sleep 3; echo slept", "timeout_s": 1}},
               # Typing into a busy command is refused (it would run later, unreported).
               {"call": "run", "args": {"input": "echo typeahead", "timeout_s": 10}, "error": True,
                "has": ["still running and not waiting for input"]},
               {"call": "wait", "args": {"timeout_s": 10}, "has": ["slept"]},
               {"call": "run", "args": {"input": "echo typeahead"}, "has": ["$ echo typeahead · exit 0"]}]},
    {"name": "F21-multiline-run-timeout-keeps-later-lines", "session": {},
     # The later line is not typed ahead into the running command: not sent, and said so.
     "steps": [{"call": "run", "args": {"input": "sleep 2\necho after-sleep", "timeout_s": 0.5},
                "has": ["[stopped: 1 more line not sent"]},
               {"call": "wait", "args": {"timeout_s": 10}, "has": ["$ sleep 2 · exit 0"]},
               {"call": "regions", "args": {}, "lacks": ["started by human", "after-sleep"]}]},
    {"name": "F16-shell-multiline-stops-at-question", "session": {},
     "steps": [{"call": "run", "args": {"input": "read -p 'Q? ' a\necho \"a=[$a]\"", "timeout_s": 10},
                "has": ["not sent"]}]},
    {"name": "F3-session-shell-exit-closes-region", "session": {},
     "steps": [{"call": "run", "args": {"input": "exit 3", "timeout_s": 10}, "has": ["exit 3"], "lacks": ["· running ·"]},
               {"call": "regions", "args": {}, "lacks": ["· running ·"]}]},
    {"name": "F19-read-region-without-trailing-newline", "session": {},
     "steps": [{"call": "run", "args": {"input": "printf 'tail-no-nl'"}},
               {"call": "run", "args": {"input": "echo second"}},
               {"call": "read", "args": {"region": 1}, "has": ["tail-no-nl"], "lacks": ["echo second"]}]},
    {"name": "F11-wide-char-at-wrap-edge-no-space", "needs": ["python3"], "session": {},
     "steps": [{"call": "run", "args": {"input": "python3 -c \"print('a' + '日'*100 + 'END')\""},
                "has": ["a" + "日" * 100 + "END"]}]},
    {"name": "F1-single-huge-line-is-clipped", "session": {},
     "steps": [{"call": "run", "args": {"input": "head -c 30000 /dev/zero | tr '\\0' y; echo", "max_lines": 5},
                "has": ["exit 0"], "lacks": ["y" * 20000]}]},
    {"name": "F7-noncanonical-reader-without-prompt-is-waiting", "session": {},
     "steps": [{"call": "run", "args": {"input": "read -e x", "timeout_s": 30}, "max_s": 8, "has": ["waiting for input"]},
               {"call": "send", "args": {"keys": ["ctrl+d"]}}]},
    {"name": "F17-exec-into-repl-returns-at-prompt", "needs": ["python3"], "session": {},
     "steps": [{"call": "run", "args": {"input": "exec python3 -q", "timeout_s": 20}, "max_s": 8, "has": ["waiting for input at '>>>'"],
                "lacks": ["still running; use `wait`"]}]},
    {"name": "F9-comment-or-blank-line-returns-quickly", "session": {},
     "steps": [{"call": "run", "args": {"input": "# just a comment", "timeout_s": 20}, "max_s": 5, "lacks": ["timeout"]},
               {"call": "run", "args": {"input": "   ", "timeout_s": 20}, "max_s": 5, "lacks": ["timeout"]}]},
    {"name": "F15-small-max-lines-clips-consistently", "session": {},
     "steps": [{"call": "run", "args": {"input": "seq 1 5", "max_lines": 2},
                "has": ["exit 0 · lines 2-6", "\n1\n[... 3 lines omitted (lines 3-5)", "...]\n5"]},
               {"call": "run", "args": {"input": "seq 1 5", "max_lines": 1},
                "has": ["[... 4 lines omitted (lines 8-11)", "...]\n5"], "lacks": ["\n1\n"]}]},
    {"name": "F18-label-from-typed-text-when-echo-off", "session": {},
     "steps": [{"call": "run", "args": {"input": "stty -echo"}},
               {"call": "run", "args": {"input": "echo noecho-test"}, "has": ["$ echo noecho-test"]},
               {"call": "run", "args": {"input": "stty echo"}}]},
    # -- `run until` (the eval: models passed prompt_pattern="license" and the like) --
    {"name": "run-until-returns-at-the-match-at-the-shell", "session": {},
     "steps": [{"call": "run", "args": {"input": "for i in 1 2 3 4 5 6; do echo step$i; sleep 1; done",
                                        "until": "^step2$", "timeout_s": 30},
                "max_s": 5, "has": ["step2", "`until` matched"], "lacks": ["step5"]},
               {"call": "wait", "args": {"timeout_s": 20}, "has": ["step6", "exit 0"]}]},
    {"name": "run-until-inside-a-program", "needs": ["python3"], "session": {"command": "python3 -q"},
     "steps": [{"call": "run", "args": {"input": "import time; [print('n', i, flush=True) or time.sleep(1) for i in range(6)]",
                                        "until": "^n 1$", "timeout_s": 30},
                "max_s": 5, "has": ["n 1", "· until ·", "`until` matched"], "lacks": ["n 4"]},
               {"call": "wait", "args": {"timeout_s": 20}, "has": ["n 5", "· ready ·"]}]},
    # -- smaller-model round: gdb's inferior at its own prompt --
    {"name": "inferior-prompt-under-gdb-started-from-bash", "needs": ["gdb", "python3"], "session": {},
     "steps": [{"call": "run", "args": {"input": "gdb -q -nx -iex 'set debuginfod enabled off' --args python3 -c \"x = input('stock> '); print('got', x)\""},
                "has": ["waiting for input at '(gdb)'"]},
               {"call": "run", "args": {"input": "run", "timeout_s": 20}, "max_s": 2.5, "has": ["· ready ·"]},
               {"call": "run", "args": {"input": "apple", "timeout_s": 20}, "has": ["got apple", "exited normally"]},
               {"call": "run", "args": {"input": "print 1+1"}, "max_s": 2.5, "has": ["$1 = 2", "· ready ·"]}]},
    # -- smaller-model round --
    {"name": "a-programs-prompt-is-not-called-a-question", "needs": ["python3"], "session": {},
     "steps": [{"call": "run", "args": {"input": "python3 -c \"import sys\nwhile True:\n  x = input('stock> ')\n  print('got', x)\n  x == 'quit' and sys.exit()\""},
                "has": ["waiting for input at 'stock>'"]},
               {"call": "run", "args": {"input": "quit", "prompt_pattern": r"\$ $"},
                "lacks": ["as the answer"]}]},
    {"name": "a-prompt-the-callers-pattern-misses-is-named", "needs": ["gdb"], "session": {"command": "gdb -q -nx -iex 'set debuginfod enabled off'"},
     "steps": [{"call": "run", "args": {"input": "print 1", "prompt_pattern": r"\$ $"},
                "has": ["· waiting ·", "looks like a prompt your prompt_pattern does not match"]}]},
    {"name": "gdb-prompt-under-a-shell-pattern-is-not-a-question", "needs": ["gdb"], "source": "evaluation smoke test",
     "session": {},
     "steps": [{"call": "run", "args": {"input": "gdb -q -nx -iex 'set debuginfod enabled off'"}},
               {"call": "run", "args": {"input": "print 1+1", "prompt_pattern": r"\$ $"},
                "has": ["$1 = 2", "looks like a prompt your prompt_pattern does not match"],
                "lacks": ["as the answer"]}]},
    # -- sub-agent remote-shell hunt (R + its F numbers) --
    {"name": "R-F2-ssh-fake-prompt-mid-command-not-ready", "needs": ["ssh localhost"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}, "has": ["$"]},
               # Over ssh the sleep is invisible: "build> " is not taken for the
               # prompt ("ready"); the call waits, and `wait` gets the real one.
               {"call": "run", "args": {"input": "printf 'build> '; sleep 5; echo done-$((6*7))", "timeout_s": 20},
                "lacks": ["· ready ·"]},
               {"call": "wait", "args": {"timeout_s": 20}, "has": ["done-42", "· ready ·"]},
               {"call": "send", "args": {"text": "exit", "keys": ["enter"]}}]},
    {"name": "R-dots-progress-line-over-ssh-is-not-a-prompt", "needs": ["ssh localhost"], "source": "Ubuntu install test",
     "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               # pytest -q's progress line: its leading dots are no Python
               # "..." prompt with output after it ("ready"), nor one a
               # human typed at.
               {"call": "run", "args": {"input": "printf '....F... [ 17%%]\\n'; sleep 5; echo done-$((6*7))",
                                        "timeout_s": 3}, "lacks": ["· ready ·"]},
               {"call": "wait", "args": {"timeout_s": 20}, "has": ["done-42"], "lacks": ["typed at the prompt"]},
               {"call": "send", "args": {"text": "exit", "keys": ["enter"]}}]},
    {"name": "R-F7b-ssh-integrated-fake-prompt-wait-gets-exit", "needs": ["ssh localhost"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               {"call": "shell_integrate", "args": {}},
               {"call": "run", "args": {"input": "printf 'user@host:~$ '; sleep 3; echo after"}},
               {"call": "wait", "args": {"timeout_s": 10}, "has": ["exit 0", "after"], "lacks": ["still running"],
                "max_s": 6},
               {"call": "run", "args": {"input": "exit"}}]},
    {"name": "R-F10-plain-shell-wait-after-silent-timeout", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "sleep 4", "timeout_s": 2}, "has": ["timeout"]},
               {"call": "wait", "args": {"timeout_s": 10}, "has": ["ready"], "lacks": ["still running"], "max_s": 5}]},
    {"name": "R-F14-ssh-signal-does-not-kill-connection", "needs": ["ssh localhost"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               {"call": "run", "args": {"input": "sleep 30", "timeout_s": 1}},
               {"call": "send", "args": {"signal": "INT"}, "error": True, "has": ["ssh"]},
               {"call": "send", "args": {"keys": ["ctrl+c"]}},
               {"call": "run", "args": {"input": "exit"}}]},
    {"name": "R-F4-ssh-ctrl-c-note-no-term-advice", "needs": ["ssh localhost", "python3"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               {"call": "run", "args": {"input": "python3 -q"}},
               {"call": "run", "args": {"input": "import time; time.sleep(30)", "timeout_s": 2}},
               {"call": "send", "args": {"keys": ["ctrl+c"]}, "has": ["KeyboardInterrupt"],
                "lacks": ["signal=\"TERM\"", "still in the foreground after ctrl+c"]},
               {"call": "send", "args": {"keys": ["ctrl+d"]}},
               {"call": "run", "args": {"input": "sleep 30", "timeout_s": 2}, "lacks": ["ctrl+c did not stop", "TERM"]},
               {"call": "send", "args": {"keys": ["ctrl+c"]}},
               {"call": "run", "args": {"input": "exit"}}]},
    {"name": "R-F13-shell-integrate-refuses-non-shell-foreground", "needs": ["python3"], "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "python3 -q"}},
               {"call": "shell_integrate", "args": {}, "error": True, "has": ["python"], "max_s": 3},
               {"call": "send", "args": {"keys": ["ctrl+d"]}},
               {"call": "run", "args": {"input": "read -s -p 'Password: ' p; echo; echo len=${#p}"}},
               {"call": "shell_integrate", "args": {}, "error": True, "max_s": 3},
               {"call": "send", "args": {"keys": ["ctrl+c"]}}]},
    {"name": "R-F3-ssh-python-prompt-recognised", "needs": ["ssh localhost", "python3"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               {"call": "run", "args": {"input": "python3 -q"}, "lacks": ["question"], "max_s": 2},
               {"call": "run", "args": {"input": "print(6*7)"}, "has": ["42"], "lacks": ["question"], "max_s": 2},
               {"call": "run", "args": {"input": "def f(x):\n    return x * 2\n\nprint(f(21))"}, "has": ["42"],
                "lacks": ["not sent"]},
               {"call": "send", "args": {"keys": ["ctrl+d"]}},
               {"call": "run", "args": {"input": "exit"}}]},
    {"name": "R-F9-integrated-inner-shell-exit-closes", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "bash --norc --noprofile -i"}},
               {"call": "shell_integrate", "args": {}},
               {"call": "run", "args": {"input": "exit 3"}, "lacks": ["still running", "appears to be waiting for input"]},
               {"call": "run", "args": {"input": "echo back"}, "has": ["back"], "lacks": ["exit 3"]}]},
    {"name": "R-F1-plain-shell-no-trailing-newline-output", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "printf 'no newline here'"}, "has": ["no newline here"]},
               {"call": "run", "args": {"input": "echo next"}, "has": ["next"], "lacks": ["no newline here"]}]},
    {"name": "R-F12-plain-shell-continuation-prompt-named", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "echo 'abc"}, "has": ["continuation"]},
               {"call": "send", "args": {"keys": ["ctrl+c"]}}]},
    {"name": "R-F11-question-after-partial-output-not-unsubmitted", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "printf 'build> '; read -p 'Name: ' n; echo n=$n"}},
               {"call": "run", "args": {"input": "bob"}, "has": ["n=bob"], "lacks": ["discarded", "unsubmitted"]}]},
    {"name": "R-F15-ssh-wait-nothing-running-returns", "needs": ["ssh localhost"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "ssh localhost"}},
               {"call": "run", "args": {"input": "echo x"}, "has": ["x"]},
               {"call": "wait", "args": {"timeout_s": 6}, "has": ["the shell at the far end of this ssh is at its prompt"],
                "lacks": ["still running", "waiting for input"], "max_s": 2},
               {"call": "run", "args": {"input": "exit"}}]},
    # -- sub-agent TUI hunt (T + its letters) --
    {"name": "T-A-send-shows-rows-rewritten-in-place", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "printf 'one\\ntwo\\nthree\\n'; read -s -n1; printf '\\033[3AONE\\033[K\\nTWO\\033[K\\nTHREE\\033[K\\n'; read -s -n1", "timeout_s": 10}, "has": ["three"]},
               {"call": "send", "args": {"keys": ["x"]}, "has": ["ONE", "TWO", "THREE"], "lacks": ["no new output"]},
               {"call": "send", "args": {"keys": ["x"]}, "has": ["exit 0"]}]},
    {"name": "T-B-send-text-newline-is-enter", "needs": ["nano"], "session": {"cwd": "{home}", "cols": 120, "rows": 40},
     "steps": [{"call": "run", "args": {"input": "nano {home}/n.txt"}, "has": ["GNU nano"]},
               {"call": "send", "args": {"text": "first line\nsecond line\nthird line", "screen": "full"},
                "has": ["second line"], "lacks": ["Justified", "first line second line"]},
               {"call": "send", "args": {"keys": ["ctrl+x", "n"]}, "has": ["exit 0"]}]},
    {"name": "T-C-screen-view-expands-tabs", "session": {"cwd": "{home}", "cols": 120, "rows": 30},
     "steps": [{"call": "run", "args": {"input": "printf '\\033[?1049h\\033[H\\033[2J\\033[3;5H\\tX\\033[5;1H'; read -s -n1; printf '\\033[?1049l'", "timeout_s": 10},
                "has": ["3\t        X"], "lacks": ["\t    \tX"]},
               {"call": "send", "args": {"keys": ["x"]}, "has": ["exit 0"]}]},
    {"name": "T-D-alt-screen-tab-does-not-indent-log", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "printf '\\033[?1049h\\033[H\\033[2J\\033[3;5H\\tX\\033[5;1H'; read -s -n1; printf '\\033[?1049l'", "timeout_s": 10}},
               {"call": "send", "args": {"keys": ["x"]}, "has": ["exit 0"]},
               {"call": "read", "args": {"limit": 3}, "lacks": ["\t        $"]}]},
    {"name": "T-E-one-placeholder-per-fullscreen-run", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "printf '\\033[?1049hX\\033[?1049l\\r\\033[1d\\r\\n'; echo after", "timeout_s": 10},
                "lacks": ["omitted]\n[full-screen program output omitted]"], "has": ["after"]},
               {"call": "read", "args": {"limit": 6}, "has": ["2\t[full-screen program output omitted]"],
                "lacks": ["3\t[full-screen program output omitted]"]}]},
    # -- sub-agent programs hunt (P + its F numbers) --
    {"name": "P-F1-statusline-is-not-fullscreen", "needs": ["python3"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "python3 -c \"import sys; w=sys.stdout.write; w('\\x1b[1;20r\\x1b7\\x1b[24;1Hstatus: ok\\x1b8'); w('prompt> '); sys.stdout.flush(); s=input(); w('\\x1b7\\x1b[24;1Hstatus: busy\\x1b8'); print('got', s); w('prompt> '); sys.stdout.flush(); input()\"", "timeout_s": 10},
                "lacks": ["full-screen"]},
               {"call": "run", "args": {"input": "hello"}, "has": ["got hello"], "lacks": ["full-screen", "status: busy"]},
               {"call": "send", "args": {"keys": ["ctrl+d"]}}]},
    {"name": "P-F2-fish-prompt-survives-cd", "needs": ["fish"], "session": {"command": "fish -N", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "echo one"}, "has": ["one"]},
               {"call": "run", "args": {"input": "cd /tmp"}, "has": ["ready"], "lacks": ["waiting", "not this shell's"], "max_s": 2.5},
               {"call": "run", "args": {"input": "echo two"}, "has": ["two", "ready"], "lacks": ["waiting"], "max_s": 2.5}]},
    {"name": "P-F3-node-continuation-prompt", "needs": ["node"], "session": {"command": "node"},
     "steps": [{"call": "run", "args": {"input": "function f() {\nreturn 7 }\nf()"}, "has": ["7"], "lacks": ["not sent", "waiting"], "max_s": 4}]},
    {"name": "P-F3-sqlite-continuation-prompt", "needs": ["sqlite3"], "session": {"command": "sqlite3 :memory:"},
     "steps": [{"call": "run", "args": {"input": "select 6*7\n;"}, "has": ["42"], "lacks": ["not sent", "not this shell's"], "max_s": 3}]},
    {"name": "P-F3-guile-error-prompt", "needs": ["guile"], "session": {"command": "guile -q"},
     "steps": [{"call": "run", "args": {"input": "(car '())\n,q\n(+ 1 2)"}, "has": ["$1 = 3"], "lacks": ["not sent", "waiting"], "max_s": 5}]},
    {"name": "P-F3-zsh-ps2-prompt", "needs": ["zsh"], "session": {"command": "zsh -f"},
     "steps": [{"call": "run", "args": {"input": "for i in 1 2\ndo echo n$i\ndone"}, "has": ["n1", "n2"], "lacks": ["not sent"], "max_s": 4}]},
    {"name": "P-F8-program-output-after-a-prompt-is-not-discarded", "needs": ["python3"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "python3 -c \"import sys,tty; print('>>> step 1', end='', flush=True); tty.setraw(0); s = sys.stdin.read(3); tty.setcbreak(0); print(); print('got', repr(s))\"", "timeout_s": 10}, "has": ["step 1"]},
               {"call": "run", "args": {"input": "abc", "timeout_s": 10}, "has": ["got 'abc'"], "lacks": ["discarded", "\\x05"]}]},
    {"name": "P-F4-popup-screen-returned", "needs": ["python3"], "session": {"command": "python3 -c \"import sys; w=sys.stdout.write; w('\\x1b[2J\\x1b[Hline one\\r\\n'); sys.stdout.flush(); input(); w('\\x1b[?25l\\x1b[5;10H+--------+\\x1b[6;10H| POPUP  |\\x1b[7;10H+--------+\\x1b[6;16H\\x1b[?25h'); sys.stdout.flush(); input()\""},
     "steps": [{"call": "send", "args": {"keys": ["enter"], "wait_ms": 1500}, "has": ["POPUP"], "lacks": ["no new output"]}]},
    {"name": "P-F5-poll-idle-program-start-returns", "needs": ["python3"], "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "python3 -c 'import select,sys; print(\"connected\"); select.select([sys.stdin],[],[]); print(\"got\", sys.stdin.readline())'", "timeout_s": 30},
                "has": ["connected"], "lacks": ["still running; use `wait`"], "max_s": 8},
               {"call": "run", "args": {"input": "hi"}, "has": ["got hi"]}]},
    {"name": "P-F7-kermit-prompt", "needs": ["kermit"], "session": {"command": "kermit", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "echo hi there"}, "has": ["hi there", "ready"], "lacks": ["waiting"], "max_s": 3}]},
    {"name": "P-F6-node-await-timer", "needs": ["node"], "session": {"command": "node"},
     "steps": [{"call": "run", "args": {"input": "await new Promise(r => setTimeout(() => r('late'), 5000))"}, "has": ["'late'"], "lacks": ["settled"]}]},
    {"name": "P-python-independent-lines-each-shown", "needs": ["python3"], "session": {"command": "python3 -q"},
     "steps": [{"call": "run", "args": {"input": "[n*n for n in range(5)]\n6*7"}, "has": ["[0, 1, 4, 9, 16]", "42"]}]},
    # -- the evaluation's guidance round --
    {"name": "live-gdb-with-redirected-output-is-named", "needs": ["gdb"], "source": "eval guidance round: the one FAIL", "session": {"cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "gdb -q -nx -iex 'set debuginfod enabled off' > {home}/trace.out 2>&1", "timeout_s": 3},
                "has": ["gdb's output goes to {home}/trace.out", "ctrl+d"]},
               {"call": "send", "args": {"keys": ["ctrl+d"]}}]},
    {"name": "a-timeout-beyond-the-client-limit-is-cut", "source": "eval guidance round: wait timeout_s=60000", "session": {},
     "steps": [{"call": "run", "args": {"input": "echo quick", "timeout_s": 60000},
                "has": ["timeout_s 60000 cut to 1500 s", "quick"]}]},
    {"name": "an-unknown-prompt-gets-a-pattern-offered", "needs": ["python3"], "source": "prompt_pattern after observation",
     "session": {"command": "python3 -c \"import sys\nwhile True:\n  s = input('Enter choice= ')\n  print('chose', s)\""},
     "steps": [{"call": "run", "args": {"input": "1"}, "has": ["chose 1", "· waiting ·", "pass prompt_pattern `Enter choice= $`"]},
               {"call": "run", "args": {"input": "2", "prompt_pattern": "Enter choice= $"}, "has": ["chose 2", "· ready ·"], "max_s": 2}]},
    # -- prompts that change every time --
    {"name": "changing-prompt-time-and-counter", "session": {"command": "bash --norc --noprofile -i"},
     "steps": [{"call": "run", "args": {"input": "PS1='[\\t #\\#] ready: '"},
                "has": ["· waiting ·", "pass prompt_pattern `\\[\\d+:\\d+:\\d+ \\#\\d+\\] ready: $`"],
                "lacks": ["as the answer"]},
               {"call": "run", "args": {"input": "echo one"}, "has": ["one"], "lacks": ["as the answer"]},
               {"call": "run", "args": {"input": "echo two", "prompt_pattern": "^\\[12:00:00 #3\\] ready: $"},
                "has": ["two", "did not match", "does your pattern cover the parts that"]},
               {"call": "run", "args": {"input": "echo three", "prompt_pattern": "^\\[\\d\\d:\\d\\d:\\d\\d #\\d+\\] ready: $"},
                "has": ["three", "· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "sleep 2; echo four", "prompt_pattern": "^\\[\\d\\d:\\d\\d:\\d\\d #\\d+\\] ready: $"},
                "has": ["four", "· ready ·"], "max_s": 4},
               {"call": "run", "args": {"input": "echo 'note: '; sleep 2; echo five", "prompt_pattern": ": $"},
                "has": ["five"]}]},
    {"name": "changing-prompt-counter", "needs": ["python3"],  # (IPython's own "In [n]: " is known by default)
     "session": {"command": "python3 -c \"n=1\nwhile True:\n  s=input(f'calc[{n}]: ')\n  print(f'Out[{n}]:', eval(s))\n  n+=1\""},
     "steps": [{"call": "run", "args": {"input": "6*7"}, "has": ["Out[1]: 42", "pass prompt_pattern `calc\\[\\d+\\]: $`"],
                "lacks": ["as the answer"]},
               {"call": "run", "args": {"input": "1+1", "prompt_pattern": "calc\\[\\d+\\]: $"}, "has": ["Out[2]: 2", "· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "2+2", "prompt_pattern": "calc\\[2\\]: $"}, "has": ["Out[3]: 4", "did not match"]}]},
    {"name": "changing-prompt-cwd", "session": {"command": "bash --norc --noprofile -i", "cwd": "{home}"},
     "steps": [{"call": "run", "args": {"input": "PS1='\\w ready: '; mkdir -p a/b"}},
               {"call": "run", "args": {"input": "cd a", "prompt_pattern": "^\\S+ ready: $"}, "has": ["· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "cd b", "prompt_pattern": "^\\S+ ready: $"}, "has": ["· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "pwd", "prompt_pattern": "^\\S+ ready: $"}, "has": ["a/b", "· ready ·"], "max_s": 2}]},
    {"name": "changing-prompt-exit-status-marker", "session": {"command": "bash --norc --noprofile -i"},
     "steps": [{"call": "run", "args": {"input": "PROMPT_COMMAND='st=$?'; PS1='${st#0}ready: '"}},
               {"call": "run", "args": {"input": "true", "prompt_pattern": "^\\d*ready: $"}, "has": ["· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "false", "prompt_pattern": "^\\d*ready: $"}, "has": ["· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "echo after", "prompt_pattern": "^ready: $"}, "has": ["after", "· ready ·"], "max_s": 2},
               {"call": "run", "args": {"input": "false", "prompt_pattern": "^ready: $"}, "has": ["did not match", "'1ready:'"]}]},
    {"name": "changing-prompt-date", "session": {"command": "bash --norc --noprofile -i"},
     "steps": [{"call": "run", "args": {"input": "PS1='\\D{%Y-%m-%d} ready: '"}, "has": ["pass prompt_pattern `\\d+\\-\\d+\\-\\d+ ready: $`"]},
               {"call": "run", "args": {"input": "echo dated", "prompt_pattern": "^\\d{4}-\\d\\d-\\d\\d ready: $"}, "has": ["dated", "· ready ·"], "max_s": 2}]},
    {"name": "F20-regions-last-validated", "session": {},
     "steps": [{"call": "run", "args": {"input": "echo a"}}, {"call": "run", "args": {"input": "echo b"}},
               {"call": "run", "args": {"input": "echo c"}},
               {"call": "regions", "args": {"last": -2}, "error": True}]},
]


def expand(value, home: str):
    if isinstance(value, str):
        return value.replace("{home}", home)
    if isinstance(value, list):
        return [expand(v, home) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, home) for k, v in value.items()}
    return value


@pytest.fixture
def env(tmp_path):
    socket = f"dtmcp-scn-{os.getpid()}-{tmp_path.name}"
    yield tmp_path, socket
    subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
    tmpdir = os.environ.get("TMUX_TMPDIR", "/tmp")
    (Path(tmpdir) / f"tmux-{os.getuid()}" / socket).unlink(missing_ok=True)


@pytest.mark.parametrize("scenario", [pytest.param(s, id=s["name"], marks=needs(*s.get("needs", ()))) for s in SCENARIOS])
async def test_scenario(env, scenario):
    home = str(env[0])
    t = Term(*env)
    create = {"name": "s", "env": {"HOME": plain_home(env[0])}}
    create.update(expand(scenario.get("session", {}), home))
    await t("session_create", **create)
    loop = asyncio.get_running_loop()
    for i, step in enumerate(scenario["steps"]):
        args = {"session": "s", **expand(step.get("args", {}), home)}
        where = f"step {i + 1} ({step['call']})"
        t0 = loop.time()
        if step.get("error"):
            with pytest.raises(AssertionError) as e:
                await t(step["call"], **args)
            out = str(e.value)
        else:
            out = await t(step["call"], **args)
        took = loop.time() - t0
        if "max_s" in step:
            assert took < step["max_s"], f"{where} took {took:.1f} s:\n{out}"
        for text in expand(step.get("has", []), home):
            assert text in out, f"{where}: {text!r} not in reply:\n{out}"
        for text in expand(step.get("lacks", []), home):
            assert text not in out, f"{where}: {text!r} in reply:\n{out}"
