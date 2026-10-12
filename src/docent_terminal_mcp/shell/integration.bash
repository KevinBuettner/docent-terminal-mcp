# docent-terminal-mcp shell integration for bash: OSC 133 semantic-prompt markers.
#
#   A  prompt start (k=s: continuation prompt)   B  prompt end, input starts
#   C  output start                              D;<status>  command finished
#
# Every marker carries pid=<shell pid>, so the server can tell nested shells
# apart (and the prompt and output-start markers h=<host> and u=<user>, so
# it knows where a shell reached over ssh or su runs), and n=<the session's nonce> (__DTMCP_NONCE, set by the server), so
# markers a program prints -- another session's log catted -- are ignored.
#
# PROMPT_COMMAND becomes the single exported hook __dtmcp_prompt, which runs
# the shell's own prompt commands from __dtmcp_user_pc -- an ordinary,
# unexported array.  So an interactive bash started from this one inherits
# the integration (the hook functions are exported too) but never a prompt
# command naming a function it lacks, such as one a login profile defined.
#
# The markers are only for the session's own terminal.  A shell that a
# program starts on a pty or pipe of its own (expect, pexpect, `script`,
# GDB's testsuite) inherits the hook too, but must print its prompts
# unchanged: __DTMCP_TTY names the session's terminal, and the hook stays
# silent unless the prompt goes there.
#
# This file is sourced by the session's bashrc, and by shells that cannot
# inherit it, such as one reached over ssh (shell_integrate types the line).

__DTMCP_TTY=$(tty 2>/dev/null)
export __DTMCP_TTY

__dtmcp_prompt() {
    local status=$? pipestatus=("${PIPESTATUS[@]}") __dtmcp_c __dtmcp_here=
    # Called from a prompt command of its own: never run twice.
    [ -n "$__dtmcp_inside" ] && return $status
    local __dtmcp_inside=1
    # Prompts go to stderr; mark them only on the session's terminal.
    [ -n "$__DTMCP_TTY" ] && [ /dev/stderr -ef "$__DTMCP_TTY" ] && __dtmcp_here=1
    if [ -n "$__dtmcp_here" ]; then
        if [ ${#pipestatus[@]} -gt 1 ]; then
            printf '\e]133;D;%s;pid=%s;n=%s;pipestatus=%s\a' "$status" "$$" "$__DTMCP_NONCE" "${pipestatus[*]}"
        else
            printf '\e]133;D;%s;pid=%s;n=%s\a' "$status" "$$" "$__DTMCP_NONCE"
        fi
    fi
    for __dtmcp_c in "${__dtmcp_user_pc[@]}"; do
        __dtmcp_status "$status"  # each prompt command sees the command's status
        eval "$__dtmcp_c"
    done
    [ -n "$__dtmcp_here" ] && __dtmcp_post
    return $status
}

__dtmcp_status() {
    return "$1"
}

__dtmcp_post() {
    # Prompt frameworks often rebuild the prompts; re-mark them if so.
    case $PS1 in
        *'133;A'*) ;;
        *) PS1='\[\e]133;A;pid=$$;n=${__DTMCP_NONCE};h=${HOSTNAME};u=${USER}\a\]'"$PS1"'\[\e]133;B;pid=$$;n=${__DTMCP_NONCE}\a\]' ;;
    esac
    case $PS2 in
        *'133;A'*) ;;
        *) PS2='\[\e]133;A;k=s;pid=$$;n=${__DTMCP_NONCE}\a\]'"$PS2"'\[\e]133;B;pid=$$;n=${__DTMCP_NONCE}\a\]' ;;
    esac
    case $PS0 in
        *'133;C'*) ;;
        *) PS0="$PS0"'\e]133;C;pid=$$;n=${__DTMCP_NONCE};h=${HOSTNAME};u=${USER}\a' ;;
    esac
}

# A bash started from a session inherits PROMPT_COMMAND=__dtmcp_prompt, and
# its profile may have added to it (PROMPT_COMMAND+=(...), or "history -a;
# $PROMPT_COMMAND") before this file runs -- in another server's session,
# say.  The hook is left out of the commands it runs.
case "${#PROMPT_COMMAND[@]}:$PROMPT_COMMAND" in
    1:__dtmcp_prompt) ;;  # already integrated
    *)
        __dtmcp_old=("${__dtmcp_user_pc[@]}")  # this shell's, if it was integrated before
        __dtmcp_user_pc=()
        for __dtmcp_c in "${__dtmcp_old[@]}" "${PROMPT_COMMAND[@]}"; do
            __dtmcp_c=${__dtmcp_c//__dtmcp_prompt/:}
            case $__dtmcp_c in '' | *[!\ \;:]*) ;; *) continue ;; esac  # nothing but itself
            [ -n "$__dtmcp_c" ] || continue
            # Once each: a profile sourced twice may have added one twice.
            for __dtmcp_d in "${__dtmcp_user_pc[@]}"; do
                [ "$__dtmcp_d" = "$__dtmcp_c" ] && continue 2
            done
            __dtmcp_user_pc+=("$__dtmcp_c")
        done
        unset __dtmcp_c __dtmcp_d __dtmcp_old PROMPT_COMMAND
        PROMPT_COMMAND=__dtmcp_prompt
        ;;
esac
export PROMPT_COMMAND
export -f __dtmcp_prompt __dtmcp_status __dtmcp_post
