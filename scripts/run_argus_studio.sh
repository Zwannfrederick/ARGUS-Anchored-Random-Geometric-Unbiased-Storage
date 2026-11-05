#!/usr/bin/env bash
set -e
DIR="/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix"
cd "$DIR"
exec "$DIR/.venv/bin/python" -m argus_cache.webui.server --host 127.0.0.1 --port 8090 --upstream http://127.0.0.1:8080/v1
