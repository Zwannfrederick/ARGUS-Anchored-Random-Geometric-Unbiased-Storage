#!/usr/bin/env bash
# Launch an isolated local ARGUS Claude Code session in the current directory
# without modifying ~/.claude/settings.json or interfering with cloud Claude Code.
set -e

SCRIPT_SOURCE="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"

EXTRA_FLAGS=()
PASS_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--with-plugins" ]; then
        EXTRA_FLAGS+=("--with-plugins")
    else
        PASS_ARGS+=("$arg")
    fi
done

if [ ${#PASS_ARGS[@]} -gt 0 ]; then
    exec "$SCRIPT_DIR/argus_ctl.sh" claude --auto-start "${EXTRA_FLAGS[@]}" -- "${PASS_ARGS[@]}"
else
    exec "$SCRIPT_DIR/argus_ctl.sh" claude --auto-start "${EXTRA_FLAGS[@]}"
fi
