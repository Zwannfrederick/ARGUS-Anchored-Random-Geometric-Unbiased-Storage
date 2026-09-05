#!/usr/bin/env bash
# Waybar Custom Module & Toggle Controller for ARGUS AI

# Resolve real script path even through symlinks
SCRIPT_SOURCE="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CTL_SCRIPT="$PROJECT_ROOT/scripts/argus_ctl.sh"
TELEMETRY_SCRIPT="$PROJECT_ROOT/scripts/argus_telemetry_ui.py"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
WAYBAR_SIGNAL=8

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

action="${1:-status}"

toggle() {
    status_json=$("$CTL_SCRIPT" status --json 2>/dev/null || echo '{"state":"OFF"}')
    current_state=$(echo "$status_json" | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)

    if [ "$current_state" = "READY" ] || [ "$current_state" = "STARTING" ]; then
        pkill -f "argus_waybar.sh.*wait_then_open_claude" 2>/dev/null || true
        "$CTL_SCRIPT" stop >/dev/null 2>&1
        notify-send -u low -i "system-shutdown" "ARGUS AI" "Gateway durduruldu (Offline)" 2>/dev/null || true
    else
        notify-send -u normal -i "system-run" "ARGUS AI" "Başlatılıyor... (Health & Backend Probe)" 2>/dev/null || true
        "$CTL_SCRIPT" start >/dev/null 2>&1 &
    fi
    pkill -RTMIN+$WAYBAR_SIGNAL waybar-lua-bin 2>/dev/null || pkill -RTMIN+$WAYBAR_SIGNAL waybar 2>/dev/null || true
}

open_telemetry() {
    if command -v kitty >/dev/null 2>&1; then
        kitty --class argus-telemetry --title "ARGUS AI Telemetry & Logs" -e "$PYTHON_BIN" "$TELEMETRY_SCRIPT" &
    elif command -v alacritty >/dev/null 2>&1; then
        alacritty --class argus-telemetry --title "ARGUS AI Telemetry & Logs" -e "$PYTHON_BIN" "$TELEMETRY_SCRIPT" &
    elif command -v foot >/dev/null 2>&1; then
        foot --title "ARGUS AI Telemetry & Logs" "$PYTHON_BIN" "$TELEMETRY_SCRIPT" &
    else
        notify-send "ARGUS AI Telemetry" "$("$CTL_SCRIPT" status 2>/dev/null)"
    fi
}

# Open a Claude Code session bound to the gateway. This deliberately does NOT
# write ~/.claude/settings.json: that file is read by every Claude process on
# the machine, so a redirect there breaks unrelated sessions the moment the
# gateway is down. The redirect lives in this terminal's environment only.
open_claude() {
    if command -v kitty >/dev/null 2>&1; then
        kitty --class argus-claude --title "Claude Code (ARGUS local)" -e "$CTL_SCRIPT" claude &
    elif command -v alacritty >/dev/null 2>&1; then
        alacritty --class argus-claude --title "Claude Code (ARGUS local)" -e "$CTL_SCRIPT" claude &
    elif command -v foot >/dev/null 2>&1; then
        foot --title "Claude Code (ARGUS local)" "$CTL_SCRIPT" claude &
    else
        notify-send -u critical "ARGUS AI" "No supported terminal found. Run: argus_ctl.sh claude" 2>/dev/null || true
        return 1
    fi
}

# Poll until the gateway answers, then hand the user a session. Launching the
# terminal immediately would drop them into a "no gateway" error, since the
# backend needs time to load the model.
wait_then_open_claude() {
    for _ in $(seq 1 120); do
        sleep 2
        state=$("$CTL_SCRIPT" status --json 2>/dev/null | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
        if [ "$state" = "READY" ]; then
            notify-send -u normal -i "system-run" "ARGUS AI" "Hazır — Claude Code açılıyor" 2>/dev/null || true
            open_claude
            return 0
        fi
    done
    notify-send -u critical "ARGUS AI" "Gateway hazır olmadı; loglara bakın" 2>/dev/null || true
    return 1
}

get_status_json() {
    status_json=$("$CTL_SCRIPT" status --json 2>/dev/null || echo '{"state":"OFF","status_text":"ARGUS ● OFF","css_class":"offline"}')
    state=$(echo "$status_json" | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
    vram_used=$(echo "$status_json" | grep -o '"used_mib": *[0-9.]*' | cut -d: -f2 | tr -d ' ')
    vram_total=$(echo "$status_json" | grep -o '"total_mib": *[0-9.]*' | cut -d: -f2 | tr -d ' ')
    # Read the model and context actually reported, rather than naming a model
    # in the template: the tooltip previously claimed Qwen3.8-27B whichever
    # model was really loaded.
    model=$(echo "$status_json" | grep -o '"model_name": *"[^"]*"' | cut -d'"' -f4)
    kv_type=$(echo "$status_json" | grep -o '"kv_cache_type": *"[^"]*"' | cut -d'"' -f4)
    ctx=$(echo "$status_json" | grep -o '"context_length": *[0-9]*' | cut -d: -f2 | tr -d ' ')
    if [ -n "$ctx" ]; then
        ctx_text="$((ctx / 1024))K"
    else
        ctx_text="model yüklü değil"
    fi

    gw_url=$(echo "$status_json" | grep -o '"gateway_url": *"[^"]*"' | cut -d'"' -f4)
    gw_port=$(echo "$gw_url" | grep -o '[0-9]*$')

    case "$state" in
        READY)
            echo "{\"text\":\" 󱚣 ARGUS ● READY \",\"alt\":\"ready\",\"tooltip\":\"ARGUS AI: AKTİF (Port ${gw_port:-8008})\nModel: ${model:-bilinmiyor}\nContext: ${ctx_text}  •  KV: ${kv_type:-?} (backend)\nVRAM: ${vram_used:-0} / ${vram_total:-0} MiB\n[Sol Tık: Kapat | Orta Tık: Claude CLI | Sağ Tık: Telemetri]\",\"class\":\"ready\",\"percentage\":100}"
            ;;
        STARTING)
            echo "{\"text\":\" 󱚣 ARGUS ◐ STARTING \",\"alt\":\"starting\",\"tooltip\":\"ARGUS AI: Başlatılıyor...\nBackend & Health Probe bekleniyor\n[Sol Tık: İptal Et | Sağ Tık: Loglar]\",\"class\":\"starting\",\"percentage\":50}"
            ;;
        ERROR)
            echo "{\"text\":\" 󱚢 ARGUS ! ERROR \",\"alt\":\"error\",\"tooltip\":\"ARGUS AI: HATA!\nPort veya Backend bağlantı sorunu\n[Sol Tık: Yeniden Başlat | Sağ Tık: Loglar]\",\"class\":\"error\",\"percentage\":0}"
            ;;
        *)
            echo "{\"text\":\" 󱚢 ARGUS ● OFF \",\"alt\":\"offline\",\"tooltip\":\"ARGUS AI: KAPALI (Offline)\nClaude Code normal provider'larda\n[Sol Tık: Başlat | Orta Tık: Claude CLI | Sağ Tık: Telemetri]\",\"class\":\"offline\",\"percentage\":0}"
            ;;
    esac
}

case "$action" in
    toggle)
        toggle
        ;;
    telemetry)
        open_telemetry
        ;;
    claude)
        open_claude
        ;;
    status)
        get_status_json
        ;;
    *)
        get_status_json
        ;;
esac
