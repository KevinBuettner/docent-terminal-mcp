# docent-terminal-mcp's ZDOTDIR for integrated zsh sessions: read the user's own
# .zprofile (from their ZDOTDIR, or HOME) as zsh would, with ZDOTDIR set to
# theirs while it runs, so paths built from it (HISTFILE, plugins) are theirs.
() {
    local ours=$ZDOTDIR
    ZDOTDIR=${__DTMCP_USER_ZDOTDIR:-$HOME}
    [[ -r $ZDOTDIR/.zprofile ]] && source $ZDOTDIR/.zprofile
    ZDOTDIR=$ours
}
