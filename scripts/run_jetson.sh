#!/usr/bin/env bash
set -euo pipefail

echo "Setting max performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo "Starting HearPoint realtime inference..."
conda run -n hearpoint-realtime python src/realtime/realtime_inference.py "$@"
