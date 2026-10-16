"""`NAME ssh HOST`: ssh with the shell integration carried along.

A plain `ssh` stays as it is; running this instead is the choice (and the
consent) to bring the integration to the other side for the length of the
connection.  Like kitty's `kitten ssh`, the setup travels as ssh's remote
command -- nothing is typed at a prompt or shown -- and here it is small
enough to travel whole: a POSIX sh bootstrap that unpacks the integration
into a temporary directory, starts the user's shell (zsh if that is their
login shell, else bash) with it, and removes the directory (bash: once
loaded; zsh: when that shell exits).  Nothing stays on the host.

`NAME install HOST` is the other way, for shells that ssh's remote command
cannot reach (a plain ssh already open, `sudo -i`, a second hop): the user
runs it once per host, it copies the two integration files to
`~/.local/share/NAME/vN` there, and shell_integrate then types one
readable line that reads the file.  Nothing is installed by the server
itself: a host without the files gets no integration, and a reply that
says how to install it.
"""

from __future__ import annotations

import base64
import io
import re
import tarfile
from pathlib import Path

from .names import NAME, SHORT
from .tmux import _SSH_VALUE_OPTS

SHELL_DIR = Path(__file__).parent / "shell"
# Published under these names in the temporary directory: the same layout
# as here, so bashrc finds integration.bash beside it, and zsh/.zshrc finds
# ../integration.zsh.
FILES = ("bashrc", "integration.bash", "integration.zsh",
         "zsh/.zshenv", "zsh/.zprofile", "zsh/.zshrc", "zsh/.zlogin")


# A space- or comma-separated list of environment variables that `NAME ssh`
# and `NAME into` set on the far side to their values here: what a client
# that started the session (an editor, say) tells programs in it about
# where they run.  The list goes along too, so a second hop carries them on.
CARRY_ENV = "DOCENT_TERMINAL_MCP_SSH_ENV"


def carried(environ: dict[str, str]) -> dict[str, str]:
    """The variables CARRY_ENV names, as set here (unset ones skipped)."""
    names = [CARRY_ENV, *re.split(r"[\s,]+", environ.get(CARRY_ENV, "").strip())]
    return {n: environ[n] for n in names
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) and n in environ}


def bootstrap(nonce: str | None, readable: bool = False, carry: dict[str, str] | None = None) -> str:
    """The remote side's script: POSIX sh, with no single quotes in it, so
    it can travel as `exec sh -c '...'` through any login shell.
    `readable`: each file's base64 replaced by where to read the file.
    `carry`: variables to export there (see CARRY_ENV)."""
    lines = [
        f'd=$(mktemp -d "${{TMPDIR:-/tmp}}/{SHORT}.XXXXXX") || exit 1',
        'mkdir "$d/zsh"',
        # (GNU base64 and newer BSDs take -d; older macOS only -D)
        'dec() { base64 -d 2>/dev/null || base64 -D; }',
    ]
    for name in FILES:
        data = (f"<{SHELL_DIR / name}, base64-encoded>" if readable
                else base64.b64encode((SHELL_DIR / name).read_bytes()).decode())
        lines.append(f'printf %s {data} | dec > "$d/{name}"')
    if nonce and nonce.isalnum():  # (a session's nonce is hex)
        lines.append(f"export __DTMCP_NONCE={nonce}")
    for name, value in (carry or {}).items():  # (any value: base64 keeps it whole)
        lines.append(f'{name}=$(printf %s {base64.b64encode(value.encode()).decode()} | dec); export {name}')
    lines += [
        'if [ "${SHELL##*/}" = zsh ] && command -v zsh >/dev/null 2>&1; then',
        '  export __DTMCP_USER_ZDOTDIR="${ZDOTDIR:-}" ZDOTDIR="$d/zsh" __DTMCP_CLEANUP="$d"',
        '  exec zsh -l -i',
        'elif command -v bash >/dev/null 2>&1; then',
        '  printf ". \\"%s/bashrc\\"; rm -rf \\"%s\\"\\n" "$d" "$d" > "$d/rc"',
        '  exec bash --rcfile "$d/rc" -i',
        'fi',
        'rm -rf "$d"',
        f'echo "{NAME} ssh: no bash or zsh here -- a plain login shell, without the integration" >&2',
        'exec "${SHELL:-/bin/sh}" -l',
    ]
    script = "\n".join(line for line in lines if line) + "\n"
    assert readable or "'" not in script
    return script


def split_destination(args: list[str]) -> tuple[list[str], str, list[str]] | None:
    """ssh's arguments as (options, destination, remote command)."""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if not a.startswith("-") or a == "-":
            break
        letters = a[1:]
        for j, c in enumerate(letters):
            if c in _SSH_VALUE_OPTS:
                if not letters[j + 1:]:
                    i += 1  # the value is the next argument
                break
        i += 1
    if i >= len(args):
        return None
    return args[:i], args[i], args[i + 1:]


def ssh_argv(args: list[str], nonce: str | None, ssh: str = "ssh",
             carry: dict[str, str] | None = None) -> list[str]:
    """The ssh command line that carries the integration along."""
    parts = split_destination(args)
    if parts is None:
        raise ValueError(f"usage: {NAME} ssh [ssh options] [user@]host")
    options, dest, command = parts
    if command:
        raise ValueError(f"{NAME} ssh opens an interactive shell; for a remote command use plain ssh")
    return [ssh, "-t", *options, dest, "exec sh -c '" + bootstrap(nonce, carry=carry) + "'"]


# The integration files a host keeps after `NAME install HOST`, and where.
# The directory names the version of the marker protocol: bump it when the
# server needs something of the integration that older files lack, so a
# stale copy reads as "not installed" (a test pins the files' hash to it).
INTEGRATION_VERSION = 1
INSTALLED = ("integration.bash", "integration.zsh")
INSTALL_PARENT = f".local/share/{NAME}"  # under the remote user's home
INSTALL_DIR = f"{INSTALL_PARENT}/v{INTEGRATION_VERSION}"


def install_tar() -> bytes:
    """The integration files as a tar archive, for the remote tar's stdin."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in INSTALLED:
            data = (SHELL_DIR / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            info.mtime = int((SHELL_DIR / name).stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def install_script(remove: bool = False) -> str:
    """What the remote side runs: POSIX sh without single quotes (as
    bootstrap).  Older versions' directories go; they are ours."""
    lines = [f'p="$HOME/{INSTALL_PARENT}"', 'rm -rf "$p"/v[0-9]*']
    if remove:
        lines += ['rmdir "$p" 2>/dev/null', f'echo "{NAME}: integration removed from $(hostname):$p"']
    else:
        lines += [f'd="$HOME/{INSTALL_DIR}"', 'mkdir -p "$d" && tar -C "$d" -xf - || exit 1',
                  f'echo "{NAME}: integration installed in $(hostname):$d"']
    script = "\n".join(lines) + "\n"
    assert "'" not in script
    return script


def install_argv(args: list[str], remove: bool = False, ssh: str = "ssh") -> list[str]:
    """The ssh command line that installs (or removes) the integration;
    its stdin is install_tar()."""
    parts = split_destination(args)
    if parts is None or parts[2]:
        raise ValueError(f"usage: {NAME} install [--print] [--remove] [ssh options] [user@]host")
    options, dest, _ = parts
    return [ssh, *options, dest, "exec sh -c '" + install_script(remove) + "'"]


# su's options that run a command instead of an interactive shell.
SU_COMMAND_OPTS = ("-c", "--command", "--session-command", "-C")


def su_argv(args: list[str], nonce: str | None, carry: dict[str, str] | None = None) -> list[str]:
    """`NAME su [su options] [-] [user]`: su with the bootstrap as the
    target user's command.  --session-command, not -c: -c starts a new
    session without the terminal (no job control, no ctrl+c)."""
    if any(a in SU_COMMAND_OPTS or a.startswith(tuple(o + "=" for o in SU_COMMAND_OPTS if o.startswith("--")))
           for a in args):
        raise ValueError(f"{NAME} su opens an interactive shell; for a command use plain su -c")
    return ["su", "--session-command", "exec sh -c '" + bootstrap(nonce, carry=carry) + "'", *args]


def into_argv(command: list[str], nonce: str | None, carry: dict[str, str] | None = None) -> list[str]:
    """`NAME into COMMAND...`: COMMAND is how a program is run somewhere
    else with a terminal -- `openshell sandbox exec -n box --tty --`,
    `podman exec -it box`, `kubectl exec -it pod --` -- and the bootstrap
    goes as that program: the integration for the length of that shell,
    as `NAME ssh` brings it over ssh."""
    if not command:
        raise ValueError(f"usage: {NAME} into COMMAND...   e.g. {NAME} into openshell sandbox exec -n box --tty --")
    return [*command, "sh", "-c", bootstrap(nonce, carry=carry)]
