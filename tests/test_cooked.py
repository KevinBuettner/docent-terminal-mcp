from docent_terminal_mcp.cooked import ALT_PLACEHOLDER, CookedTerminal


def cooked(data: bytes, **kw) -> list[str]:
    t = CookedTerminal(**kw)
    t.feed(data)
    # Trailing blanks are trimmed, as in every view of the cooked text.
    return [t.line(i).rstrip() for i in range(len(t.lines))]


def test_plain_lines():
    assert cooked(b"one\r\ntwo\r\n") == ["one", "two", ""]


def test_long_line_is_never_wrapped():
    assert cooked(b"x" * 1000 + b"\r\n")[0] == "x" * 1000


def test_carriage_return_overwrites():
    assert cooked(b"progress 10%\rprogress 99%\r\n")[0] == "progress 99%"


def test_backspace_and_erase_to_end_of_line():
    # readline deleting the last word: move back, then erase to end of line.
    assert cooked(b"echo hello\b\b\b\b\b\x1b[K") == ["echo"]


def test_erase_whole_line_and_start():
    assert cooked(b"abcdef\x1b[2K\rxy") == ["xy"]
    assert cooked(b"abcdef\x1b[3D\x1b[1K") == ["    ef"]


def test_sgr_and_osc_title_are_stripped():
    assert cooked(b"\x1b[1;31mred\x1b[0m \x1b]0;title\x07plain") == ["red plain"]


def test_cursor_left_right_and_absolute_column():
    assert cooked(b"abc\x1b[2DX\x1b[CY") == ["aXcY"]
    assert cooked(b"abcdef\x1b[3GZ") == ["abZdef"]


def test_insert_and_delete_characters():
    assert cooked(b"abcdef\r\x1b[2P") == ["cdef"]
    assert cooked(b"abcdef\r\x1b[2@") == ["  abcdef"]


def test_tab_moves_to_next_stop():
    assert cooked(b"a\tb") == ["a       b"]


def test_bare_linefeed_keeps_column():
    assert cooked(b"abc\ndef") == ["abc", "   def"]


def test_cursor_up_rewrites_previous_line():
    assert cooked(b"line1\r\nline2\x1b[A\rLINE1") == ["LINE1", "line2"]


def test_cursor_up_cannot_reach_committed_rows():
    t = CookedTerminal(window=2)
    t.feed(b"a\r\nb\r\nc\r\nd\r\n\x1b[10Ax")
    assert t.floor == 2
    assert [t.line(i) for i in range(5)] == ["a", "b", "x", "d", ""]
    assert isinstance(t.lines[0], str)  # committed rows are frozen


def test_clear_screen_keeps_history():
    assert cooked(b"old\r\n$ clear\r\n\x1b[H\x1b[2Jnew") == ["old", "$ clear", "new"]


def test_alternate_screen_is_replaced_by_placeholder():
    data = b"before\r\n\x1b[?1049h\x1b[Hfull screen junk\x1b[5;1Hmore\x1b[?1049lafter\r\n"
    assert cooked(data) == ["before", ALT_PLACEHOLDER, "after", ""]


def test_utf8_split_across_feeds():
    t = CookedTerminal()
    data = "héllo ✓".encode()
    for i in range(len(data)):
        t.feed(data[i : i + 1])
    assert t.line(0) == "héllo ✓"


def test_osc_markers_report_positions():
    seen = []
    t = CookedTerminal(on_osc=lambda text, pos: seen.append((text, pos)))
    t.feed(b"\x1b]133;A\x07$ \x1b]133;B\x07ls\r\n\x1b]133;C\x1b\\out\r\n\x1b]133;D;0\x07")
    assert seen == [("133;A", (0, 0)), ("133;B", (0, 2)), ("133;C", (1, 0)), ("133;D;0", (2, 0))]


def test_text_between_trims_trailing_blanks():
    t = CookedTerminal()
    t.feed(b"$ ls   \r\nfile1   \r\nfile2\r\n$ ")
    assert t.text_between((1, 0), (3, 0)) == "file1\nfile2\n"
    assert t.text_between((0, 2), (1, 0)) == "ls\n"


def test_chunking_does_not_change_the_result():
    data = b"ab\x1b[31mc\x1b]133;A\x07d\re\r\nf\x1b[2Dg"
    whole = cooked(data)
    t = CookedTerminal()
    for b in data:
        t.feed(bytes([b]))
    assert [t.line(i) for i in range(len(t.lines))] == whole


def test_wide_characters_take_two_cells():
    # Two backspaces cross one wide character, as readline assumes; X then
    # overwrites the first half of 語, which blanks its second half.
    assert cooked("日本語\b\bX".encode()) == ["日本X"]
    assert cooked("日本語\b\b\b\bX".encode()) == ["日X 語"]
    # Cursor-right skips both cells of a wide character.
    assert cooked("日本\r\x1b[2CX".encode()) == ["日X"]


def test_readline_insert_among_wide_characters():
    # What bash 5.3 wrote for: type "echo 日本語テスト", backspace twice,
    # left twice, type "X" -- captured from raw.log of a real session.
    data = (
        b"echo \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88"
        b"\x08\x08\x1b[K\x08\x08\x1b[K\x08\x08\x08\x08\x1b[1@X"
    )
    assert cooked(data) == ["echo 日本X語テ"]


def test_combining_marks_join_the_previous_character():
    assert cooked("éx".encode()) == ["éx"]
    t = CookedTerminal()
    t.feed("é".encode())
    assert t.col == 1 and t.pos == (0, 2)


def test_positions_are_text_indices():
    seen = []
    t = CookedTerminal(on_osc=lambda text, pos: seen.append(pos))
    t.feed("日本 \x1b]133;B\x07ls".encode())
    assert seen == [(0, 3)]  # after "日本 " in the text, though 5 cells in
    assert t.text_between(seen[0], t.pos) == "ls"


def test_clear_then_report_stays_readable():
    # One home (clear) and then ordinary lines: not painting.
    data = b"\x1b]133;C\x07\x1b[H\x1b[2Jreport line 1\r\nreport line 2\r\n"
    assert cooked(data) == ["report line 1", "report line 2", ""]


def test_repainting_program_becomes_a_placeholder():
    # As procps top draws: application keys, hidden cursor, then frames
    # from home -- never the alternate screen.
    frame = b"\x1b[Htop - frame\x1b[K\r\nTasks: 1\x1b[K\r\n"
    data = (b"$ top\r\n\x1b]133;C\x07\x1b[?1h\x1b=\x1b[?25l" + frame + frame + frame
            + b"\x1b[?25h\r\n\x1b]133;A\x07$ ")
    t = CookedTerminal()
    t.feed(data)
    lines = [t.line(i).rstrip() for i in range(len(t.lines))]
    assert lines == ["$ top", ALT_PLACEHOLDER, "", "$"]  # "" from the \r\n after painting
    assert not t.full_screen


def test_row_addressing_starts_painting_at_once():
    t = CookedTerminal()
    t.feed(b"\x1b]133;C\x07\x1b[5;10Hxyz")
    assert t.painting and t.full_screen
    t.feed(b"\x1b]133;A\x07$ ")
    assert not t.painting
    assert [t.line(i).rstrip() for i in range(len(t.lines))] == [ALT_PLACEHOLDER, "$"]


def test_ctrl_l_at_the_prompt_is_not_painting():
    # readline's clear-screen redraws PS1, and with it the prompt markers.
    prompt = b"\x1b]133;A\x07$ \x1b]133;B\x07"
    t = CookedTerminal()
    t.feed(prompt + b"\x1b[H\x1b[2J" + prompt + b"\x1b[H\x1b[2J" + prompt + b"ls")
    assert not t.painting
    assert t.line(t.row).endswith("$ ls")


# Captured from raw.log of a real bash 5.3 session in a 40-column pane:
# "echo " + 39 a's typed (wrapping onto a second screen row), six
# backspaces, ctrl+u, then "echo X" + Enter.  Readline moves back up with
# ESC M once the line fits on one row again.
UNWRAP = (
    b"line-above\r\n"
    b"\x1b]133;A\x07$ \x1b]133;B\x07echo " + b"a" * 39
    + b"\x08\x1b[K" * 5 + b"\r\x1b[K\x1bM\x1b[C\x1b[C\x1b[K"
    + b"\r\n\r\x1b[K\x1bM\x1b[C\x1b[Cecho X\r\n"
)


def test_editing_a_wrapped_line_back_onto_one_row():
    t = CookedTerminal(width=40)
    t.feed(UNWRAP)
    assert [t.line(i).rstrip() for i in range(len(t.lines))] == ["line-above", "$ echo X", ""]


def test_without_a_width_the_same_bytes_climb_into_the_line_above():
    # Why the width matters: with no screen-row model, ESC M leaves the line.
    t = CookedTerminal()
    t.feed(UNWRAP)
    assert t.line(0).rstrip() != "line-above"


def test_wrapped_output_stays_one_line():
    t = CookedTerminal(width=10)
    t.feed(b"x" * 25 + b"\r\nnext\r\n")
    assert [t.line(i).rstrip() for i in range(3)] == ["x" * 25, "next", ""]


def test_carriage_return_goes_to_the_start_of_the_screen_row():
    t = CookedTerminal(width=10)
    t.feed(b"0123456789abcdef\rXY")  # CR returns to the second screen row
    assert t.line(0) == "0123456789XYcdef"


def test_pending_wrap_carriage_return_stays_on_the_full_row():
    t = CookedTerminal(width=10)
    t.feed(b"0123456789\rX")  # the row is full; CR goes back to its start
    assert t.line(0) == "X123456789"


def test_erase_to_end_of_screen_row_keeps_later_rows():
    t = CookedTerminal(width=10)
    t.feed(b"0123456789abcdefghij")  # two full screen rows
    t.feed(b"\x1bM\r\x1b[3C\x1b[K")  # up into row one, column 4, erase to its end
    assert t.line(0).rstrip() == "012       abcdefghij"


def test_repeated_clears_keep_their_text():  # S6
    data = b"\x1b]133;C\x07" + b"".join(
        b"\x1b[H\x1b[2Jtick %d\r\nstatus ok\r\n" % i for i in (1, 2, 3)
    ) + b"\x1b]133;A\x07$ "
    t = CookedTerminal()
    t.feed(data)
    lines = [t.line(i).rstrip() for i in range(len(t.lines))]
    assert "tick 3" in lines and ALT_PLACEHOLDER not in lines


def test_first_frame_with_hidden_cursor_is_painting():  # S2
    t = CookedTerminal()
    t.feed(b"\x1b]133;C\x07\x1b[?25l\x1b[Htop - one frame\r\n")
    assert t.painting


def test_line_feed_after_erasing_a_wrapped_line_starts_a_new_line():  # X1
    """readline, redrawing after a resize, erases each screen row to its
    end and moves down; the erased cells of a line that wrapped stay as
    blanks, and must not make every later line feed a move within that
    line (which joined the rest of a session into one padded line)."""
    t = CookedTerminal(width=20)
    t.feed(b"x" * 30)                      # a line over two screen rows
    t.feed(b"\r\x1b[K")                   # start of row 2, erase it
    t.feed(b"\x1bM\x1b[K")                # up to row 1, erase it: all blank now
    t.feed(b"$ cat\r\none\r\ntwo\r\n")  # a line feed from row 1 of a 30-cell line
    lines = [t.line(i).rstrip() for i in range(len(t.lines))]
    assert lines[:3] == ["$ cat", "one", "two"], lines


def test_bracketed_paste_prompt_ends_painting():
    """A pager (systemctl's less, with no deinit) paints rows by addressing
    them, then quits without showing the cursor; the shell's prompt, which
    turns bracketed paste on, ends the painting."""
    t = CookedTerminal(width=80)
    t.feed(b"$ systemctl status\r\n\x1b[?25l\x1b[3;1Htree\x1b[4;1Hmore")
    assert t.full_screen
    t.feed(b"\x1b[?2004h$ ")
    assert not t.full_screen and t.cursor_text() == "$ "


# Found by differential testing against tmux (tests/difftest.py): the
# cooked view follows tmux, whose rendering the humans see.

def test_backspace_at_a_continuation_row_goes_back_a_row():
    # 40 + 2 cells: the second row continues the line; BS from its start
    # moves to the last cell of the first row, as tmux does.
    assert cooked(b"x" * 40 + b"ab\b\b\bZ", width=40) == ["x" * 39 + "Zab"]


def test_backspace_does_not_go_back_onto_an_erased_continuation():
    assert cooked(b"x" * 40 + b"ab\x1b[2K\r\bZ", width=40) == ["x" * 40 + "Z"]


def test_backspace_from_a_pending_wrap_lands_on_the_last_cell():
    # After the last cell the cursor waits to wrap; BS only ends the wait.
    assert cooked(b"x" * 38 + "日".encode() + b"\bZ", width=40) == ["x" * 38 + " Z"]


def test_cursor_left_from_a_pending_wrap_counts_from_past_the_end():
    assert cooked(b"x" * 40 + b"\x1b[2DZ", width=40) == ["x" * 38 + "Zx"]


def test_erase_at_a_pending_wrap():
    # To the end of the row: nothing (the cursor is past it); the wrap stays pending.
    assert cooked(b"x" * 40 + b"\x1b[KZ", width=40) == ["x" * 40 + "Z"]
    # To the start of the row: the whole row, and the next character wraps.
    assert cooked(b"x" * 40 + b"\x1b[1KZ", width=40) == [" " * 40 + "Z"]


def test_tab_at_a_pending_wrap_does_nothing():
    assert cooked(b"x" * 40 + b"\tZ", width=40) == ["x" * 40 + "Z"]


def test_wide_character_not_fitting_keeps_the_last_cell():
    # The wide character skips the last cell; text already there stays.
    assert cooked(b"x" * 40 + b"\r" + b"y" * 39 + "日".encode(), width=40)[0] == "y" * 39 + "x日"


def test_human_typing_plan_types_the_line_and_repeats():
    from docent_terminal_mcp.typist import plan, replay
    line = "echo the quick brown fox | grep -c fox && ls -la /tmp; make check"
    for seed in range(100):
        keys = plan(line, 2.0, seed, 0.4)
        assert replay(keys) == line
    keys = plan(line, 2.0, 7, 0.4)
    assert any(k.backspace for k in keys)  # mistakes, corrected
    assert [(k.text, k.delay) for k in plan(line, 2.0, 7, 0.4)] == [(k.text, k.delay) for k in keys]
    # About wps words per second: 2.0 wps is about 12 keys a second, plus pauses.
    assert 3 < sum(k.delay for k in plan(line, 2.0, 1, 0.0)) < 15
