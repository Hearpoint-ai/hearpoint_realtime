#!/usr/bin/env bash
set -euo pipefail

echo "Setting max performance mode..."
if ! sudo nvpmodel -m 0; then
    echo "ERROR: nvpmodel failed — check sudo permissions or Jetson power mode support"
    exit 1
fi
if ! sudo jetson_clocks; then
    echo "ERROR: jetson_clocks failed"
    exit 1
fi

echo "Verifying CUDA is visible to Python..."
if ! conda run -n hearpoint-realtime python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "ERROR: CUDA not available in Python environment"
    exit 1
fi

echo "Starting HearPoint realtime inference..."
conda run --no-capture-output -n hearpoint-realtime python -u src/realtime/realtime_inference.py "$@"
