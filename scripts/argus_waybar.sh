#!/usr/bin/env bash
# Waybar Custom Module & Rofi Control Center for ARGUS AI Studio

SCRIPT_SOURCE="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CTL_SCRIPT="$PROJECT_ROOT/scripts/argus_ctl.sh"
TELEMETRY_SCRIPT="$PROJECT_ROOT/scripts/argus_telemetry_ui.py"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
ROFI_THEME="$HOME/.config/rofi/launchers/type-1/style-1.rasi"
WAYBAR_SIGNAL=8

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

action="${1:-status}"

refresh_waybar() {
    pkill -RTMIN+$WAYBAR_SIGNAL waybar-lua-bin 2>/dev/null || pkill -RTMIN+$WAYBAR_SIGNAL waybar 2>/dev/null || true
}

run_floating_term() {
    local win_class="$1"
    local win_title="$2"
    shift 2
    if command -v kitty >/dev/null 2>&1; then
        if command -v setsid >/dev/null 2>&1; then
            setsid -f kitty --class "$win_class" --title "$win_title" -e "$@" </dev/null >/dev/null 2>&1
        else
            nohup kitty --class "$win_class" --title "$win_title" -e "$@" </dev/null >/dev/null 2>&1 &
        fi
    elif command -v alacritty >/dev/null 2>&1; then
        nohup alacritty --class "$win_class" --title "$win_title" -e "$@" </dev/null >/dev/null 2>&1 &
    elif command -v foot >/dev/null 2>&1; then
        nohup foot --app-id="$win_class" --title "$win_title" "$@" </dev/null >/dev/null 2>&1 &
    else
        notify-send -u critical "ARGUS AI" "Desteklenen terminal bulunamadı (kitty/alacritty/foot)." 2>/dev/null || true
        return 1
    fi
}

wait_and_notify_ready() {
    for _ in $(seq 1 150); do
        sleep 2
        st=$("$CTL_SCRIPT" status --json 2>/dev/null | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
        if [ "$st" = "READY" ]; then
            refresh_waybar
            notify-send -u normal -i "system-run" "ARGUS AI" "Hazır (128K) — ARGUS Studio :8090 aktif" 2>/dev/null || true
            return 0
        fi
    done
    refresh_waybar
    return 1
}

start_bg() {
    notify-send -u normal -i "system-run" "ARGUS AI" "Başlatılıyor... (Qwen3.6 128K yükleniyor)" 2>/dev/null || true
    "$CTL_SCRIPT" start >/dev/null 2>&1 &
    refresh_waybar
    (wait_and_notify_ready) >/dev/null 2>&1 &
}

stop_bg() {
    "$CTL_SCRIPT" stop >/dev/null 2>&1
    refresh_waybar
    notify-send -u low -i "system-shutdown" "ARGUS AI" "Durduruldu — VRAM ve RAM serbest bırakıldı" 2>/dev/null || true
}

toggle() {
    status_json=$("$CTL_SCRIPT" status --json 2>/dev/null || echo '{"state":"OFF"}')
    current_state=$(echo "$status_json" | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
    if [ "$current_state" = "READY" ] || [ "$current_state" = "STARTING" ]; then
        stop_bg
    else
        start_bg
    fi
}

open_telemetry() {
    run_floating_term "argus-telemetry" "ARGUS AI Telemetri & Loglar" "$PYTHON_BIN" "$TELEMETRY_SCRIPT"
}

open_chat() {
    run_floating_term "argus-chat" "ARGUS Hızlı Sohbet" "$CTL_SCRIPT" chat
}

open_web() {
    "$CTL_SCRIPT" web >/dev/null 2>&1 &
}

open_menu() {
    if pgrep -x rofi >/dev/null 2>&1; then
        pkill -x rofi
        return 0
    fi

    status_json=$("$CTL_SCRIPT" status --json 2>/dev/null || echo '{"state":"OFF"}')
    current_state=$(echo "$status_json" | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
    ctx=$(echo "$status_json" | grep -o '"context_length": *[0-9]*' | cut -d: -f2 | tr -d ' ')
    ctx_tag="${ctx:+$((ctx / 1024))K}"

    if [ "$current_state" = "READY" ]; then
        prompt="󱚣 ARGUS (AKTİF${ctx_tag:+ · $ctx_tag})"
        options="󰖟  ARGUS Studio Web Arayüzü (:8090)
󰭹  Hızlı Terminal Sohbeti
󰄨  Canlı Telemetri & Loglar
󰜉  Yeniden Başlat
󰐥  Durdur (VRAM & RAM Boşalt)"
    elif [ "$current_state" = "STARTING" ]; then
        prompt="󱚣 ARGUS (BAŞLATILIYOR…)"
        options="󰄨  Canlı Telemetri & Loglar
󰖟  ARGUS Studio Web Arayüzü (:8090)
󰐥  Başlatmayı İptal Et / Durdur"
    else
        prompt="󱚢 ARGUS (KAPALI)"
        options="󰐊  ARGUS Başlat (128K Arka Plan)
󰖟  Başlat + ARGUS Studio (:8090)
󰭹  Başlat + Terminal Sohbet
󰄨  Canlı Telemetri & Loglar"
    fi

    if [ -f "$ROFI_THEME" ]; then
        chosen=$(printf "%s\n" "$options" | rofi -dmenu -i -no-custom -p "$prompt" -theme "$ROFI_THEME")
    else
        chosen=$(printf "%s\n" "$options" | rofi -dmenu -i -no-custom -p "$prompt")
    fi

    case "$chosen" in
        *'ARGUS Başlat'*)
            start_bg
            ;;
        *'ARGUS Studio'*|*'Web Arayüzü'*)
            open_web
            refresh_waybar
            ;;
        *'Terminal Sohbet'*)
            open_chat
            refresh_waybar
            ;;
        *'Telemetri'*)
            open_telemetry
            ;;
        *'Yeniden Başlat'*)
            notify-send -u normal -i "system-reboot" "ARGUS AI" "Yeniden başlatılıyor..." 2>/dev/null || true
            ("$CTL_SCRIPT" restart >/dev/null 2>&1 && refresh_waybar && notify-send -u normal -i "system-run" "ARGUS AI" "Yeniden başlatıldı (Hazır)" 2>/dev/null || true) &
            refresh_waybar
            ;;
        *'Durdur'*|*'İptal Et'*)
            stop_bg
            ;;
    esac
}

get_status_json() {
    status_json=$("$CTL_SCRIPT" status --json 2>/dev/null || echo '{"state":"OFF","status_text":"ARGUS ● OFF","css_class":"offline"}')
    state=$(echo "$status_json" | grep -o '"state": *"[^"]*"' | cut -d'"' -f4)
    vram_used=$(echo "$status_json" | grep -o '"used_mib": *[0-9.]*' | cut -d: -f2 | tr -d ' ')
    vram_total=$(echo "$status_json" | grep -o '"total_mib": *[0-9.]*' | cut -d: -f2 | tr -d ' ')
    model=$(echo "$status_json" | grep -o '"model_name": *"[^"]*"' | cut -d'"' -f4)
    kv_type=$(echo "$status_json" | grep -o '"kv_cache_type": *"[^"]*"' | cut -d'"' -f4)
    ctx=$(echo "$status_json" | grep -o '"context_length": *[0-9]*' | cut -d: -f2 | tr -d ' ')
    if [ -n "$ctx" ]; then
        ctx_text="$((ctx / 1024))K"
    else
        ctx_text="model yüklü değil"
    fi

    gw_url=$(echo "$status_json" | grep -o '"gateway_url": *"[^"]*"' | cut -d'"' -f4)
    kv_mgr=$(echo "$status_json" | grep -o '"kv_managed_by": *"[^"]*"' | cut -d'"' -f4)

    case "$state" in
        READY)
            echo "{\"text\":\"󱚣 ARGUS\",\"alt\":\"ready\",\"tooltip\":\"ARGUS AI: AKTİF\\nStudio WebUI: ${gw_url:-http://127.0.0.1:8090}\\nModel: ${model:-bilinmiyor}\\nBağlam: ${ctx_text}  •  KV: ${kv_type:-?} (${kv_mgr:-backend})\\nVRAM: ${vram_used:-0} / ${vram_total:-0} MiB\\n[Sol Tık: Menü | Orta Tık: Studio WebUI | Sağ Tık: Durdur]\",\"class\":\"ready\",\"percentage\":100}"
            ;;
        STARTING)
            echo "{\"text\":\"󱚣 ARGUS…\",\"alt\":\"starting\",\"tooltip\":\"ARGUS AI: Başlatılıyor...\\nModel ve KV önbelleği hazırlanıyor\\n[Sol Tık: Menü | Sağ Tık: İptal Et]\",\"class\":\"starting\",\"percentage\":50}"
            ;;
        ERROR)
            echo "{\"text\":\"󱚢 HATA\",\"alt\":\"error\",\"tooltip\":\"ARGUS AI: HATA!\\nPort veya Backend bağlantı sorunu\\n[Sol Tık: Menü | Sağ Tık: Durdur]\",\"class\":\"error\",\"percentage\":0}"
            ;;
        *)
            echo "{\"text\":\"󱚢 ARGUS\",\"alt\":\"offline\",\"tooltip\":\"ARGUS AI: KAPALI (VRAM/RAM boşta)\\n[Sol Tık: Menü | Orta Tık: Studio WebUI | Sağ Tık: Başlat]\",\"class\":\"offline\",\"percentage\":0}"
            ;;
    esac
}

case "$action" in
    menu)
        open_menu
        ;;
    toggle)
        toggle
        ;;
    telemetry)
        open_telemetry
        ;;
    chat)
        open_chat
        ;;
    web)
        open_web
        ;;
    status|*)
        get_status_json
        ;;
esac
