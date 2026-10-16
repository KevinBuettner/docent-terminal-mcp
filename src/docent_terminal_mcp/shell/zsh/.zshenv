# docent-terminal-mcp's ZDOTDIR for integrated zsh sessions: read the user's own
# .zshenv (from their ZDOTDIR, or HOME) as zsh would, with ZDOTDIR set to
# theirs while it runs, so paths built from it (HISTFILE, plugins) are theirs.
() {
    local ours=$ZDOTDIR
    ZDOTDIR=${__DTMCP_USER_ZDOTDIR:-$HOME}
    [[ -r $ZDOTDIR/.zshenv ]] && source $ZDOTDIR/.zshenv
    ZDOTDIR=$ours
}
# Ubuntu's /etc/zsh/zshrc runs compinit, which writes its dump to
# $ZDOTDIR/.zcompdump -- into this directory, the installed package.  Run
# it in its place instead (.zshrc), with the dump where it would be.
if [[ -z ${skip_global_compinit-} && -r /etc/zsh/zshrc && -r /etc/os-release ]] &&
   (( ${${(@f)"$(</etc/os-release)"}[(I)ID*=*ubuntu]} )) &&
   [[ "$(</etc/zsh/zshrc)" == *skip_global_compinit* ]]; then
    skip_global_compinit=1 __dtmcp_compinit=1
fi
