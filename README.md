# REKI megakernel (draft)

## Reproduce

On a machine with one NVIDIA H100 (driver >= 570), git and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/muninn-huginn/reki-megakernel && cd reki-megakernel
make setup                     # .venv, REKI @ b2a3393, pinned torch/triton/transformers/fla, weights + LIBERO frames
source .venv/bin/activate
make all OUT=results/verification   # latency, correctness, kernels, breakdown, check, tune
make figures
```
