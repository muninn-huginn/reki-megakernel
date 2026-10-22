#!/usr/bin/env bash
# Environment + inputs for the study, from the repo root: `bash scripts/setup.sh` (then `source .venv/bin/activate`).
# Needs Linux x86-64, an NVIDIA driver >= 570 (CUDA 12.8 wheels), git, uv (https://docs.astral.sh/uv/), ~15 GB disk.
set -euo pipefail
cd "$(dirname "$0")/.."

REKI_COMMIT=b2a3393667f3240e9fad4af1bf302a2c08476193        # kyutai-labs/reki, 2026-10-21
WEIGHTS_REV=589bd83a1237d45764fec65830a134ed8a5e3b69         # kyutai/reki-2b-libero
LIBERO_REV=86958911c0f959db2bbbdb107eb3e17c5f9c798e          # HuggingFaceVLA/libero (dataset)

[ -d .venv ] || uv venv -q -p 3.12 .venv
source .venv/bin/activate

if [ ! -d third_party/reki ]; then
  git clone -q https://github.com/kyutai-labs/reki third_party/reki
fi
git -C third_party/reki checkout -q "$REKI_COMMIT"

# Versions of Kyutai's pixi.lock: torch 2.10.0 (+ triton 3.6.0), transformers 5.3.0, flash-linear-attention 0.4.2.
uv pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -e third_party/reki -r requirements.txt

# Weights (CC-BY-NC-SA 4.0, not redistributed here) and the 7 LIBERO parquet files the benchmark reads
# (24 episodes / 19 tasks / 440 frames at 1 frame/s).
hf download kyutai/reki-2b-libero --revision "$WEIGHTS_REV" --local-dir data/reki-2b-libero
hf download HuggingFaceVLA/libero meta/tasks.parquet \
  $(for i in 000 060 120 180 240 300 360; do echo "data/chunk-000/file-$i.parquet"; done) \
  --repo-type dataset --revision "$LIBERO_REV" --local-dir data/libero

python -c "import torch, triton, transformers, fla, reki; print('torch', torch.__version__, 'triton', triton.__version__, \
'transformers', transformers.__version__, 'fla', fla.__version__, 'gpu', torch.cuda.get_device_name())"
