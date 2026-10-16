# docent-terminal-mcp's ZDOTDIR for integrated zsh sessions: read the user's own
# .zshrc (from their ZDOTDIR, or HOME) as zsh would, with ZDOTDIR set to
# theirs while it runs, so paths built from it (HISTFILE, plugins) are theirs.
if [[ -n ${__dtmcp_compinit-} ]]; then  # Ubuntu's global compinit (see .zshenv)
    unset __dtmcp_compinit skip_global_compinit
    autoload -U compinit
    compinit -d ${__DTMCP_USER_ZDOTDIR:-$HOME}/.zcompdump
fi
() {
    local ours=$ZDOTDIR
    ZDOTDIR=${__DTMCP_USER_ZDOTDIR:-$HOME}
    [[ -r $ZDOTDIR/.zshrc ]] && source $ZDOTDIR/.zshrc
    ZDOTDIR=$ours
}
# The session's own commands (`docent-terminal-mcp ssh`, ...) come first on PATH.
[[ -n ${__DTMCP_BIN-} && :$PATH: != *:$__DTMCP_BIN:* ]] && PATH=$__DTMCP_BIN:$PATH
# Brought over by `docent-terminal-mcp ssh`: this directory goes when this shell
# ends (a zsh started inside does not inherit the job).
if [[ -n ${__DTMCP_CLEANUP-} ]]; then
    typeset -g __dtmcp_cleanup=$__DTMCP_CLEANUP
    unset __DTMCP_CLEANUP
    zshexit() { rm -rf -- $__dtmcp_cleanup }
fi
source "${ZDOTDIR}/../integration.zsh"
