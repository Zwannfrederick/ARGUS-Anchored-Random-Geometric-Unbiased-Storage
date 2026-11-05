#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
USER_UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$USER_UNIT_DIR" "$HOME/.argus_runtime"
cp -f "$SCRIPT_DIR/systemd/argus.target" "$USER_UNIT_DIR/argus.target"
cp -f "$SCRIPT_DIR/systemd/argus-llm.service" "$USER_UNIT_DIR/argus-llm.service"
cp -f "$SCRIPT_DIR/systemd/argus-studio.service" "$USER_UNIT_DIR/argus-studio.service"
# Launcher entry: opens Studio as an app window (and starts the stack if it is down).
mkdir -p "$HOME/.local/share/applications"
cp -f "$SCRIPT_DIR/systemd/argus-studio.desktop" "$HOME/.local/share/applications/argus-studio.desktop"

systemctl --user daemon-reload
systemctl --user disable --now neo-gateway.service neo-model.service 2>/dev/null || true

# Hand over any manual non-systemd processes cleanly so systemd owns :8080 and :8090
"$PROJECT_ROOT/scripts/argus_ctl.sh" stop >/dev/null 2>&1 || true

# Enabled into argus.target only; the desktop session starts that target after
# login, so the 18 GB model load never competes with the login screen.
systemctl --user disable argus-llm.service argus-studio.service argus.target 2>/dev/null || true
systemctl --user enable argus-llm.service argus-studio.service
systemctl --user restart argus-llm.service argus-studio.service

for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && curl -fsS http://127.0.0.1:8090/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

systemctl --user is-active argus-llm.service argus-studio.service
"$PROJECT_ROOT/scripts/argus_ctl.sh" status
