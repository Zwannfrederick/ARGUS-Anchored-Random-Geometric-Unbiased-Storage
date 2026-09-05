#!/bin/bash
set -e
DIR="/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix"
cd "$DIR"
exec "$DIR/.venv/bin/python" -m hermes.neo_mobile_gateway
