#!/usr/bin/env python3
"""A simulated human at a docent-terminal-mcp session: a real tmux client attached
from a pty (so the server sees an attached human, and keys arrive as a
terminal's would), driven by commands written to a FIFO.

    human.py SESSION FIFO LOG [COLS ROWS]      # runs until `quit`

Commands, one per line, written to FIFO (e.g. echo 'type ls -l' > FIFO):
    type TEXT        type TEXT (no Enter)
    enter            press Enter
    ctrlc | ctrld | esc | tab
    backspace [N]    press Backspace N times (default 1)
    up | down        arrow keys
    copymode         enter tmux copy mode (prefix C-b, then [) -- scrolled back
    pageup           Page Up (in copy mode: scroll back further)
    q                q (leaves copy mode)
    resize COLS ROWS resize this terminal window
    detach           detach this client (the human leaves)
    attach           attach again
    quit             detach and exit
Every action is appended to LOG with a timestamp.
"""

import fcntl
import os
import pty
import signal
import struct
import subprocess
import sys
import termios
import threading
import time

SOCKET = "docent-terminal-mcp"
KEYS = {"enter": b"\r", "ctrlc": b"\x03", "ctrld": b"\x04", "esc": b"\x1b", "tab": b"\t",
        "up": b"\x1b[A", "down": b"\x1b[B", "pageup": b"\x1b[5~", "q": b"q", "copymode": b"\x02["}


class Human:
    def __init__(self, session, log, cols, rows):
        self.session, self.log, self.cols, self.rows = session, log, cols, rows
        self.proc = None
        self.attach()

    def note(self, what):
        with open(self.log, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {time.time():.2f} HUMAN {what}\n")

    def attach(self):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
        env = {k: v for k, v in os.environ.items() if k != "TMUX"}
        env["TERM"] = "xterm-256color"
        self.proc = subprocess.Popen(["tmux", "-L", SOCKET, "attach", "-t", f"={self.session}"],
                                     stdin=slave, stdout=slave, stderr=slave, start_new_session=True, env=env)
        os.close(slave)
        threading.Thread(target=self.drain, args=(self.master,), daemon=True).start()
        self.note(f"attached {self.cols}x{self.rows}")

    def drain(self, fd):
        try:
            while os.read(fd, 65536):
                pass
        except OSError:
            pass

    def detach(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            os.close(self.master)
            self.proc = None
            self.note("detached")

    def write(self, data: bytes, what: str):
        if not self.proc:
            self.note(f"(not attached; ignored: {what})")
            return
        os.write(self.master, data)
        self.note(what)

    def do(self, line):
        cmd, _, arg = line.strip().partition(" ")
        if cmd == "type":
            # Typed a character at a time, as a person would (roughly).
            for ch in arg:
                os.write(self.master, ch.encode())
                time.sleep(0.03)
            self.note(f"typed {arg!r}")
        elif cmd == "backspace":
            n = int(arg or 1)
            self.write(b"\x7f" * n, f"backspace x{n}")
        elif cmd in KEYS:
            self.write(KEYS[cmd], cmd)
        elif cmd == "resize":
            self.cols, self.rows = (int(x) for x in arg.split())
            fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
            # The client has no controlling terminal here, so the kernel sends
            # it no SIGWINCH; a real terminal's would get one.
            if self.proc:
                os.kill(self.proc.pid, signal.SIGWINCH)
            self.note(f"resized to {self.cols}x{self.rows}")
        elif cmd == "detach":
            self.detach()
        elif cmd == "attach":
            if not self.proc:
                self.attach()
        elif cmd == "quit":
            self.detach()
            return False
        elif cmd:
            self.note(f"unknown command {line.strip()!r}")
        return True


def main():
    session, fifo, log = sys.argv[1:4]
    cols, rows = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (120, 40)
    if not os.path.exists(fifo):
        os.mkfifo(fifo)
    human = Human(session, log, cols, rows)
    while True:
        with open(fifo) as f:  # reopened after each writer closes
            for line in f:
                if not human.do(line):
                    return


if __name__ == "__main__":
    main()
