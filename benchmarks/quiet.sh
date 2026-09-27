#!/usr/bin/env bash
# Runs a command on a quieted desktop: background re-indexers and desktop apps that burn CPU
# are paused (SIGSTOP) for the duration and always resumed (SIGCONT), even on failure or ^C.
# Pausing is reversible; nothing is closed. The paused process names go to stderr so the
# caller can record them next to the results.
# Usage: quiet.sh <command> [args...]
# ARGUS_QUIET_EXTRA: extra process names to pause as well. Names are matched against the
# kernel's comm, which is cut to 15 characters (codebase-memory-mcp is "codebase-memory").
set -euo pipefail
names=(codebase-memory antigravity-ide language_server pyrefly clangd.main ChatGPT vesktop ${ARGUS_QUIET_EXTRA:-})
pids=()
for name in "${names[@]}"; do
    while read -r pid; do [[ -n "$pid" && "$pid" != "$$" ]] && pids+=("$pid"); done < <(pgrep -x "$name" || true)
done
resume() { ((${#pids[@]})) && kill -CONT "${pids[@]}" 2>/dev/null || true; }
trap resume EXIT INT TERM
if ((${#pids[@]})); then
    kill -STOP "${pids[@]}" 2>/dev/null || true
    echo "quiet.sh: paused ${#pids[@]} processes: $(ps -o comm= -p "$(IFS=,; echo "${pids[*]}")" | sort | uniq -c | awk '{printf "%s%s x%s", sep, $2, $1; sep=", "}')" >&2
fi
sleep 2  # let paused work drain from the run queue
"$@"
