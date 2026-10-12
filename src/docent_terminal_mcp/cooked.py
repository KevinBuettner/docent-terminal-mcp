"""Line-oriented interpretation of a raw pty output stream.

A terminal emulator renders output onto a fixed grid and wraps long lines at
the grid width.  This interpreter instead keeps an unbounded list of lines of
unbounded length, so a line break in the result is always a real line break
in the stream -- never a wrap.  Control sequences that edit the current line
(carriage return, backspace, erase-in-line, cursor left/right, insert/delete
characters) are applied, which is what makes readline redraws and progress
bars come out as a human would read them.

Absolute cursor addressing has no meaning without a grid, so full-screen
programs are not interpreted: output drawn on the alternate screen (less,
vim, gdb's TUI) is replaced by a single placeholder line, and the rendered
screen should be read from tmux instead.  So is output from a program that
*paints* the normal screen: one that homes the cursor while it has hidden
it (top), or addresses a row other than the first (curses programs without
an alternate screen).  A plain `clear`, however often, stays ordinary
output.  Painting ends at the next shell prompt, when the cursor is shown
again, or at a terminal reset.

Columns are screen cells: a wide character (CJK, most emoji) fills its cell
and an empty filler cell after it, and a combining mark joins the cell before
it, so cursor motion lines up with what programs like readline assume.
Positions handed out -- `pos`, marker positions -- are instead (row, index
into the row's text), which stays valid once a row is frozen into a string.

Lines never wrap in the result, but programs such as readline edit a long
line in terms of the screen rows it wraps onto: they move "up" into its
first row or erase to the end of a screen row.  So when `width` (the
terminal's width) is known, a line is treated as a run of screen rows of
that width for cursor motion and erasing, while its text stays whole.

Positions are (row, col) pairs with row an index into `lines`.  Rows more
than `window` above the furthest row reached are committed: cursor-up cannot
reach them, so they never change again and can be flushed to disk.
"""

from __future__ import annotations

import codecs
import unicodedata
from collections.abc import Callable

ALT_SCREEN_MODES = {"47", "1047", "1049"}
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufe0e", "\ufe0f"}
ALT_PLACEHOLDER = "[full-screen program output omitted]"

Pos = tuple[int, int]
MarkerCallback = Callable[[str, Pos], None]

# Parser states.
GROUND, ESC, CSI, OSC, STRING, CHARSET = range(6)


def is_wide(cell: str) -> bool:
    return bool(cell) and unicodedata.east_asian_width(cell[0]) in "WF"


class CookedTerminal:
    def __init__(self, window: int = 500, on_osc: MarkerCallback | None = None, width: int | None = None):
        # Rows at or above `floor` are editable char lists; committed rows
        # below it are frozen into strings, or None once `committed_line`
        # can supply them (the owner has stored them elsewhere).
        self.lines: list[list[str] | str | None] = [[]]
        self.committed_line: Callable[[int], str] | None = None
        # (text, replacement): a finished line containing `text` is cut there
        # and ends with `replacement` -- for machine-typed noise.
        self.elisions: list[tuple[str, str]] = []
        self.row = 0
        self.col = 0
        self.width = width  # terminal columns; None: no screen-row model
        self._pending = False  # wrote the last cell of a screen row (xterm's pending wrap)
        self.window = window
        self.floor = 0  # rows below this index are committed
        self.alt_screen = False
        self._left_alt = False  # just left the alternate screen (see _set_alt)
        self.bracketed_paste = False  # the program asked for pastes to be marked
        self.painting = False
        self.cursor_hidden = False
        self.on_osc = on_osc
        self._saved: tuple[int, int, bool] = (0, 0, False)
        # A cursor move between saving and restoring the cursor draws a
        # status line (lldb 22, minicom): not painting, and not text.
        self._save_open = False
        self._in_status = False
        self.paintings = 0  # times painting started (a popup drawn and gone still counts)
        self._state = GROUND
        self._buf: list[str] = []
        self._string_esc = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    # -- public interface ------------------------------------------------------

    def feed(self, data: bytes) -> None:
        for ch in self._decoder.decode(data):
            self._step(ch)

    @property
    def full_screen(self) -> bool:
        """Output is being drawn on a grid (alternate screen or painting),
        not written as lines."""
        return self.alt_screen or self.painting

    @property
    def pos(self) -> Pos:
        """The cursor as (row, index into the row's text)."""
        return (self.row, len(self.cursor_text()))

    def cursor_text(self) -> str:
        """The cursor row's text up to the cursor."""
        line = self.lines[self.row]
        if not isinstance(line, list):
            line = self.line(self.row)
        text = "".join(line[: self.col])
        # A cursor moved past the end (readline redraws a prompt's trailing
        # space as a cursor-forward) stands after blank cells, as on screen.
        # (Lengths in cells: a wide character is a cell and an empty one.)
        return text + " " * (self.col - len(line)) if self.col > len(line) else text

    def line(self, i: int) -> str:
        line = self.lines[i]
        if line is None:
            return self.committed_line(i)
        return line if isinstance(line, str) else "".join(line)

    def text_between(self, start: Pos, end: Pos) -> str:
        """Text from `start` up to (not including) `end`, lines joined by \\n.

        Trailing blanks are dropped from each line: they are invisible on a
        screen, and programs that erase by overwriting with spaces leave them.
        """
        (r1, c1), (r2, c2) = start, end
        if (r1, c1) >= (r2, c2):
            return ""
        if r1 == r2:
            return self.line(r1)[c1:c2].rstrip()
        parts = [self.line(r1)[c1:]]
        parts.extend(self.line(r) for r in range(r1 + 1, r2))
        if r2 < len(self.lines):
            parts.append(self.line(r2)[:c2])
        return "\n".join(p.rstrip() for p in parts)

    # -- state machine ---------------------------------------------------------

    def _step(self, ch: str) -> None:
        state = self._state
        if state == GROUND:
            if ch >= " " and ch != "\x7f":
                self._put(ch)
            elif ch == "\x1b":
                self._state = ESC
            else:
                self._control(ch)
        elif state == ESC:
            self._escape(ch)
        elif state == CSI:
            if "\x40" <= ch <= "\x7e":
                self._state = GROUND
                self._csi("".join(self._buf), ch)
            elif ch == "\x1b":  # malformed; restart
                self._state = ESC
            else:
                self._buf.append(ch)
        elif state in (OSC, STRING):
            if ch == "\x07" and state == OSC:
                self._end_string()
            elif self._string_esc:
                self._string_esc = False
                if ch == "\\":
                    self._end_string()
                else:
                    self._buf.append("\x1b" + ch)
            elif ch == "\x1b":
                self._string_esc = True
            else:
                self._buf.append(ch)
        elif state == CHARSET:
            self._state = GROUND

    def _control(self, ch: str) -> None:
        if self._in_status:
            return
        if ch == "\n" or ch == "\x0b" or ch == "\x0c":
            # The tty's onlcr has already turned ordinary newlines into CRLF,
            # so a bare LF here is deliberate: down, same column.
            self._line_feed()
        elif ch == "\r":
            self._settle()
            self.col = self._row_start()
        elif ch == "\b":
            if self._pending:
                # After the last cell, tmux takes BS as leaving the pending
                # wrap only: the cursor is on the last cell.
                self._settle()
                return
            start = self._row_start()
            # At the start of a continuation row, tmux moves back to the end
            # of the row above: it is the same (wrapped) line -- while it
            # still continues here (an erased row no longer does).
            line = self.lines[self.row]
            if self.col > start:
                self.col -= 1
            elif start > 0 and len(line) > start:
                self.col = start - 1
        elif ch == "\t":
            if self.full_screen or self._pending:
                return  # a grid's tab stop, not the text's; past the row's end, no further stop
            self._settle()
            start, end = self._row_start(), self._row_end()
            self.col = start + ((self.col - start) // 8 + 1) * 8
            if end is not None:
                self.col = min(self.col, end - 1)
        # BEL, SO, SI, NUL and the rest have no effect on text.

    def _escape(self, ch: str) -> None:
        self._state = GROUND
        self._buf = []
        if ch == "[":
            self._state = CSI
        elif ch == "]":
            self._state = OSC
        elif ch in "PX^_":
            self._state = STRING
        elif ch in "()*+-./":
            self._state = CHARSET
        elif ch == "7":
            self._saved = (self.row, self.col, self._pending)
            self._save_open = True
        elif ch == "8":
            self._restore()
        elif ch == "D":
            self._line_feed()
        elif ch == "E":
            self._line_feed()
            self.col = self._row_start()
        elif ch == "M":
            self._screen_up(1)
        elif ch == "c":  # full reset
            self._stop_painting()
        # ESC =, ESC > and others: nothing to do for text.

    def _end_string(self) -> None:
        text = "".join(self._buf)
        is_osc = self._state == OSC
        self._state = GROUND
        self._buf = []
        self._string_esc = False
        if is_osc and text.startswith(("133;A", "133;C")):
            # A new prompt or command: painting (if any) is over.
            self._stop_painting()
            self.cursor_hidden = False
        if is_osc and self.on_osc is not None:
            self.on_osc(text, self.pos)

    def _csi(self, params: str, final: str) -> None:
        private = params[:1] in ("?", ">", "<", "=")
        if private:
            if final in "hl" and params[0] == "?":
                modes = set(params[1:].split(";"))
                if modes & ALT_SCREEN_MODES:
                    self._set_alt(final == "h")
                if "25" in modes:
                    self.cursor_hidden = final == "l"
                    if final == "h":  # cursor shown again
                        self._stop_painting()
                if "2004" in modes:
                    self.bracketed_paste = final == "h"
                    if final == "h":
                        # A line editor asking for bracketed paste is at its
                        # prompt: a pager or tree that painted the screen and
                        # quit without showing the cursor again is over.
                        self._stop_painting()
            return
        args = [int(p) if p.isdigit() else 0 for p in params.split(";")] if params else []
        n = args[0] if args else 0
        count = max(n, 1)
        line = self.lines[self.row]
        if final == "m":
            return
        if self._in_status:
            if final == "u":
                self._restore()
            return
        if final in "Hfd" and self._save_open and not self.full_screen:
            self._in_status = True  # a status line, drawn and left again
            return
        if final in "Hfd" and not self.full_screen and not self._left_alt:
            row = n if final == "d" else (args[0] if args and args[0] else 1)
            if row > 1 or self.cursor_hidden:
                self._start_painting()
        if self.full_screen:
            return
        # tmux holds a pending wrap as the column past the row's end: a
        # move left counts from there, and an erase leaves it pending.
        was_pending = self._pending
        if final not in "su":
            self._settle()
        start, end = self._row_start(), self._row_end()
        # The screen row's end within the line: where edits of "the rest of
        # the row" stop.  At or past the line's end, they act on the line.
        row_end = end if end is not None and end < len(line) else None
        if final == "K":
            if n == 0 and was_pending:
                pass  # from past the row's end: nothing to erase
            elif n == 0:
                self._blank(line, self.col, row_end)
            elif n == 1:
                self._blank(line, start, min(self.col + 1, len(line)), keep=True)
            elif n == 2:
                self._blank(line, start, row_end)
            if was_pending:
                self.col, self._pending = self.col + 1, True
        elif final == "J":
            if n == 0:
                del line[self.col :]
                del self.lines[self.row + 1 :]
            else:
                # Clearing the screen must not destroy history: start afresh
                # on a new line if anything is on this one.
                if line:
                    self._down(1)
        elif final == "A":
            self._screen_up(count)
        elif final in "Be":
            for _ in range(count):
                self._line_feed()
        elif final == "E":
            for _ in range(count):
                self._line_feed()
            self.col = self._row_start()
        elif final == "F":
            self._screen_up(count)
            self.col = self._row_start()
        elif final in "Ca":
            self.col += count
            if end is not None:
                self.col = min(self.col, end - 1)
        elif final == "D":
            self.col = max(start, self.col + (1 if was_pending else 0) - count)
        elif final in "G`":
            self.col = start + (count - 1 if end is None else min(count - 1, self.width - 1))
        elif final in "Hf":
            # Home, once, as `clear` does: the row means nothing without a
            # grid; honour the column only.
            self.col = start + (args[1] if len(args) > 1 and args[1] else 1) - 1
        elif final == "P":
            if row_end is None:
                del line[self.col : self.col + count]
            else:  # only this screen row's cells shift left
                cells = line[self.col : row_end]
                line[self.col : row_end] = cells[count:] + [" "] * min(count, len(cells))
        elif final == "@":
            if self.col <= len(line):
                if row_end is None:
                    line[self.col : self.col] = [" "] * count
                    if end is not None and len(line) > end:
                        del line[end:]  # pushed past the right margin: lost
                else:
                    cells = line[self.col : row_end]
                    line[self.col : row_end] = ([" "] * count + cells)[: len(cells)]
        elif final == "X":
            for i in range(self.col, min(self.col + count, len(line), end or len(line))):
                line[i] = " "
        elif final == "s" and not params:
            self._saved = (self.row, self.col, self._pending)
            self._save_open = True
        elif final == "u":
            self._restore()
        # Scroll regions, insert/delete line, tab stops, device queries: ignored.

    # -- screen rows within a line ----------------------------------------------

    def _row_start(self) -> int:
        """Column where the cursor's screen row starts within the line."""
        if not self.width:
            return 0
        return (self.col // self.width) * self.width

    def _row_end(self) -> int | None:
        return None if not self.width else self._row_start() + self.width

    def _settle(self) -> None:
        """Resolve a pending wrap before a cursor motion: the cursor is
        really still on the last cell of the row just written."""
        if self._pending:
            self._pending = False
            self.col -= 1

    @staticmethod
    def _blank(line: list[str], start: int, stop: int | None, keep: bool = False) -> None:
        """Erase cells [start, stop): to the end of the line (stop None) by
        truncating, within it by blanking so later cells keep their place."""
        if stop is None and not keep:
            del line[start:]
        else:
            for i in range(start, min(stop if stop is not None else len(line), len(line))):
                line[i] = " "

    @staticmethod
    def _content_end(line: list[str]) -> int:
        """Cells up to the last one that is not a blank: erasing part of a
        screen row keeps the cells after it (the line's later rows), so a
        line's length alone does not say whether it continues."""
        n = len(line)
        while n > 0 and line[n - 1] == " ":
            n -= 1
        return n

    def _line_feed(self) -> None:
        """Down one screen row, same column: onto the line's next screen row
        if the line continues there, otherwise onto the next line."""
        if self.full_screen:
            return
        self._settle()
        end = self._row_end()
        line = self.lines[self.row]
        if end is not None and self._content_end(line) > end:
            self.col += self.width
            return
        column = self.col - self._row_start()
        self._down(1)
        self.col = column

    def _screen_up(self, n: int) -> None:
        """Up `n` screen rows: within the line while it has rows above the
        cursor, then onto the last screen row of the line before."""
        if self.full_screen:
            return
        for _ in range(n):
            start = self._row_start()
            if start > 0:
                self.col -= self.width
                continue
            if self.row <= self.floor:
                return
            self.row -= 1
            if self.width:
                last = max(len(self.line(self.row)) - 1, 0)
                self.col = (last // self.width) * self.width + self.col

    # -- editing primitives ----------------------------------------------------

    def _put(self, ch: str) -> None:
        if self._in_status:
            return
        self._left_alt = False
        if self.full_screen:
            return
        line = self.lines[self.row]
        if unicodedata.combining(ch) or ch in ZERO_WIDTH:
            # Joins the character before it, in the cell before the cursor.
            c = self.col - 1
            while c > 0 and c < len(line) and line[c] == "":
                c -= 1
            if 0 <= c < len(line):
                line[c] += ch
            return
        width = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if self.width and width == 2 and self.col % self.width == self.width - 1:
            # A wide character does not fit in the last cell: the terminal
            # leaves it empty and wraps first.  An empty cell, not a
            # blank: the program printed nothing there.
            if len(line) <= self.col:
                line.extend(" " * (self.col - len(line)))
                line.append("")  # (a cell already written there keeps its text)
            self.col += 1
        end = self.col + width
        if len(line) < end:
            line.extend(" " * (end - len(line)))
        # Overwriting half of a wide character blanks the other half.
        if line[self.col] == "" and self.col > 0 and is_wide(line[self.col - 1]):
            line[self.col - 1] = " "
        if end < len(line) and line[end] == "":
            line[end] = " "
        line[self.col] = ch
        if width == 2:
            line[self.col + 1] = ""
        self.col = end
        self._pending = bool(self.width) and end % self.width == 0

    def _down(self, n: int, keep_col: bool = False) -> None:
        if self.full_screen:
            return
        self._elide(self.row)
        self.row += n
        while len(self.lines) <= self.row:
            self.lines.append([])
        floor = max(self.floor, self.row - self.window)
        for r in range(self.floor, floor):
            self.lines[r] = self.line(r)
        self.floor = floor
        if not keep_col:
            self.col = 0

    def _elide(self, row: int) -> None:
        line = self.lines[row]
        if not self.elisions or not isinstance(line, list):
            return
        text = "".join(line)
        for needle, replacement in self.elisions:
            i = text.find(needle)
            if i >= 0:
                self.lines[row] = list(text[:i] + replacement)
                return

    def _goto(self, row: int, col: int, pending: bool = False) -> None:
        if self.full_screen:
            return
        self.row = max(self.floor, min(row, len(self.lines) - 1))
        self.col = col
        self._pending = pending

    def _restore(self) -> None:
        """Restore the saved cursor (ESC 8, CSI u); after a status line,
        back to the text where it was."""
        if self._in_status:
            self._in_status = False
            self.row, self.col, self._pending = self._saved
        else:
            self._goto(*self._saved)
        self._save_open = False

    def leave_full_screen(self) -> None:
        """The shell is back at its prompt: whatever drew on a grid has
        finished, even if it left the alternate screen on (a command that
        printed ESC[?1049h, a log of a vim session catted)."""
        self._set_alt(False)
        self._stop_painting()

    def _start_painting(self) -> None:
        """Put the placeholder on a line of its own and ignore output until
        painting ends."""
        self.paintings += 1
        if self.lines[self.row]:
            self._down(1)
        self.lines[self.row] = list(ALT_PLACEHOLDER)
        self._down(1)
        self.painting = True

    def _stop_painting(self) -> None:
        if self.painting:
            self.painting = False
            self.col, self._pending = 0, False

    def _set_alt(self, on: bool) -> None:
        if on == self.alt_screen:
            return
        if on:
            if self.painting:  # already placeheld
                self.painting = False
            else:
                if self.lines[self.row]:
                    self._down(1)
                self.lines[self.row] = list(ALT_PLACEHOLDER)
                self._down(1)
            self.alt_screen = True
        else:
            self.alt_screen = False
            # A program leaving the alternate screen often puts the cursor
            # back by row (cfdisk: ESC[40d): not painting.  Until it writes.
            self._left_alt = True
