#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="hearpoint-realtime"

echo "=== Step 0: Check System Dependencies ==="
# Ensure the missing math library is actually there
if [ ! -f /usr/lib/aarch64-linux-gnu/libcusparseLt.so.0 ]; then
    echo "Error: libcusparseLt not found. Please run:"
    echo "sudo apt-get update && sudo apt-get install -y libcusparselt0 libcusparselt-dev"
    exit 1
fi

echo "=== Step 1: Create Environment ==="
# We force numpy<2 during the initial creation
mamba env create -f environment.yml --yes || conda env create -f environment.yml --yes

echo "=== Step 2: Install Jetson PyTorch Wheel ==="
URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
WHEEL="/tmp/$(basename $URL)"

wget -q --show-progress -O "$WHEEL" "$URL"
conda run -n "$ENV_NAME" pip install "$WHEEL"
rm -f "$WHEEL"

echo "=== Step 3: Configure Paths ==="
# Permanently link the system CUDA libs to this conda env
conda env config vars set LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64 -n "$ENV_NAME"

echo "=== Step 4: Verification ==="
conda run -n "$ENV_NAME" python -c "
import torch
import numpy
print(f'PyTorch: {torch.__version__}')
print(f'NumPy:   {numpy.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
"