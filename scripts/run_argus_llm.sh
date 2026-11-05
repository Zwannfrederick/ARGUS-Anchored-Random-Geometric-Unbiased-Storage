#!/usr/bin/env bash
set -e
DIR="/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix"
cd "$DIR"
exec "$DIR/.venv/bin/python" "$DIR/scripts/argus_systemd_llm.py"
