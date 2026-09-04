#!/bin/bash
# install_sglang_patches.sh: Installs custom NVFP4, LFM2, and DFlash2 Triton kernels into SGLang.
set -euo pipefail

SGLANG_DIR="${1:-}"

if [ -z "$SGLANG_DIR" ]; then
  if [ -d "/sgl-workspace/sglang/python/sglang" ]; then
    SGLANG_DIR="/sgl-workspace/sglang/python/sglang"
  else
    SGLANG_DIR="$(python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))' 2>/dev/null || true)"
  fi
fi

if [ -z "$SGLANG_DIR" ] || [ ! -d "$SGLANG_DIR" ]; then
  echo "Error: Could not locate SGLang directory. Please pass path as argument:"
  echo "  ./install_sglang_patches.sh /path/to/sglang"
  exit 1
fi

echo "Installing SGLang NVFP4, LFM2, and DFlash2 patches into: $SGLANG_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. LFM2 Model definition (hybrid conv1d + attention with RadixAttention)
mkdir -p "$SGLANG_DIR/srt/models"
cp -v "$SCRIPT_DIR/patches/lfm2.py" "$SGLANG_DIR/srt/models/lfm2.py"

# 2. Compressed-tensors NVFP4 W4A4 quantization handler for Blackwell
mkdir -p "$SGLANG_DIR/srt/layers/quantization"
cp -v "$SCRIPT_DIR/patches/compressed_tensors_w4a4_nvfp4.py" "$SGLANG_DIR/srt/layers/quantization/"

# 3. Triton-accelerated causal 1D convolution replacing slow PyTorch fallback
mkdir -p "$SGLANG_DIR/srt/layers"
cp -v "$SCRIPT_DIR/patches/causal_conv1d_triton.py" "$SGLANG_DIR/srt/layers/"

# 4. Triton parallel N-gram proposer and DFlash kernels
mkdir -p "$SGLANG_DIR/srt/speculative"
cp -v "$SCRIPT_DIR/patches/ngram_propose_kernel.py" "$SGLANG_DIR/srt/speculative/"
cp -v "$SCRIPT_DIR/patches/dflash_kernels.py" "$SGLANG_DIR/srt/speculative/"

echo "All SGLang patches successfully installed."
