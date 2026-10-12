# docent-terminal-mcp shell integration for zsh: OSC 133 semantic-prompt markers,
# the same as integration.bash's (see there):
#
#   A  prompt start (k=s: continuation prompt)   B  prompt end, input starts
#   C  output start                              D;<status>  command finished
#
# Each carries pid=<shell pid> and n=<the session's nonce>; A and C also
# h=<host> and u=<user>.  Markers go only to the session's own terminal
# (__DTMCP_TTY), never into a pty a program made (expect, script).
#
# Loaded by docent-terminal-mcp's ZDOTDIR .zshrc (a zsh started inside inherits
# ZDOTDIR, so it loads it too), or by the line shell_integrate types.

typeset -gx __DTMCP_TTY=${TTY:-$(tty 2>/dev/null)}
# Output without a final newline: zsh still starts the prompt on a new line,
# but marks the gap with a "%" that would land in the command's output.
PROMPT_EOL_MARK=''

__dtmcp_here() {
    [[ -n $__DTMCP_TTY && /dev/fd/2 -ef $__DTMCP_TTY ]]
}

__dtmcp_precmd() {
    local st=$? ps="${pipestatus[*]}"
    __dtmcp_here || return $st
    if [[ -n ${__dtmcp_ran-} ]]; then
        if [[ $ps == *" "* ]]; then
            print -rn -- $'\e]133;D;'"$st;pid=$$;n=${__DTMCP_NONCE};pipestatus=$ps"$'\a' >&2
        else
            print -rn -- $'\e]133;D;'"$st;pid=$$;n=${__DTMCP_NONCE}"$'\a' >&2
        fi
    fi
    __dtmcp_ran=
    return $st
}

__dtmcp_post() {
    __dtmcp_here || return 0
    # Prompt frameworks rebuild the prompts in their own precmd: wrap them
    # last, every time, and stay the last precmd.
    if [[ ${precmd_functions[-1]} != __dtmcp_post ]]; then
        precmd_functions=(${precmd_functions:#__dtmcp_post} __dtmcp_post)
    fi
    local a=$'\e]133;A;'"pid=$$;n=${__DTMCP_NONCE};h=${HOST};u=${USER}"$'\a'
    local b=$'\e]133;B;'"pid=$$;n=${__DTMCP_NONCE}"$'\a'
    [[ $PS1 == *$'\e]133;A;'* ]] || PS1="%{$a%}$PS1%{$b%}"
    [[ $PS2 == *$'\e]133;A;'* ]] || PS2="%{"$'\e]133;A;k=s;'"pid=$$;n=${__DTMCP_NONCE}"$'\a'"%}$PS2%{$b%}"
}

__dtmcp_preexec() {
    __dtmcp_ran=1
    __dtmcp_here || return 0
    print -rn -- $'\e]133;C;'"pid=$$;n=${__DTMCP_NONCE};h=${HOST};u=${USER}"$'\a' >&2
}

autoload -Uz add-zsh-hook 2>/dev/null
precmd_functions=(__dtmcp_precmd ${precmd_functions:#__dtmcp_precmd})
precmd_functions=(${precmd_functions:#__dtmcp_post} __dtmcp_post)
preexec_functions=(${preexec_functions:#__dtmcp_preexec} __dtmcp_preexec)
