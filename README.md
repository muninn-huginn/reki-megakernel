# A megakernel for an open VLA: Kyutai REKI on one H100

Does a hand-written persistent ("mega") kernel speed up a real vision-language-action model that is
already captured in CUDA graphs? This repository measures it on
[Kyutai REKI](https://kyutai.org/reki/) ([code](https://github.com/kyutai-labs/reki), 2.4B parameters,
weights [`kyutai/reki-2b-libero`](https://huggingface.co/kyutai/reki-2b-libero)) on one H100 SXM at
batch 1, end to end, on 440 real LIBERO frames.

**Result.** Replacing Euler steps 2–10 of REKI's 200M action expert with one persistent Triton kernel
takes a full `predict` from 20.9–21.7 ms (Kyutai's compiled path, two runs; they report 21.8 ms) to
18.3–18.4 ms median (1.14–1.18×), p99 from 22.0–25.0 ms to 18.7 ms. The fused loop itself goes from
6.99 to 4.03 ms (1.74×), kernels per predict from 4,837 to 2,791. Actions stay within 5.5e-3 (max
|Δ|, normalized units) of the compiled path, which is itself 6.9e-3 from eager. The VLM prefill
(9.15 ms, 2,366 kernels) is untouched and is now the largest cost.

All numbers below come from the JSON files in [`results/`](results/); [`results/verification/`](results/verification/)
is an independent re-run from a fresh clone of this repository (see [Verification](#verification)).

## Background

A *megakernel* runs a whole model (or a large part of it) in one GPU kernel launch: co-resident thread
blocks walk the layers themselves and synchronize through global memory, instead of the host launching
hundreds of small kernels. CUDA graphs already remove the host cost of those launches, but each kernel
boundary still costs GPU time (drain, launch, ramp-up, and a trip to HBM for weights and
activations). At batch 1 these boundaries dominate small layers; see Hazy Research's
[Llama-1B megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (ThunderKittens) and
[Mirage Persistent Kernel](https://github.com/mirage-project/mirage) (MPK).

REKI: a Qwen3.5-2B VLM prefills two 256×256 camera views plus the instruction; a 200M flow-matching
DiT action expert (10 layers, width 768, 12 heads, MLP 3072) cross-attends to the last VLM layer and
integrates 10 Euler steps to a chunk of 10 actions; 32 register tokens per frame are cached over K = 3
frames (memory). Kyutai's inference path captures the VLM prefill and the whole expert (with
`torch.compile`) as CUDA graphs.

## Setup

| | |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3 (SXM), driver 595.91, cloud container |
| software | Python 3.12, torch 2.10.0+cu128, triton 3.6.0, transformers 5.3.0, flash-linear-attention 0.4.2 (versions of Kyutai's `pixi.lock`), REKI at commit `b2a3393` |
| measured call | `RekiPolicy.predict`: uint8 images + raw state + task text in, unnormalized action chunk on the host out (Kyutai's definition); batch 1 |
| inputs | `HuggingFaceVLA/libero`, parquet files 000/060/120/180/240/300/360 of chunk 000: 24 episodes, 19 tasks, every 10th frame (1 frame/s, the model's history stride) = 440 frames, images in training orientation; register memory carried within and reset at each episode |
| noise | call *i* draws its flow noise from a CUDA generator seeded 1234 + *i*: every path sees the same noise for the same frame |
| timing | one untimed pass over the 440 frames (captures every graph, saves the chunks for correctness), then 880 timed calls (2 passes) per path, host wall clock around each `predict` |
| paths | **eager**: REKI with `compile=False`; **compiled**: REKI's default (prefill graph + `torch.compile`d expert in one graph); **megakernel**: compiled, with Euler steps 2–10 of the expert replaced by `src/mk_expert.py` (still inside REKI's expert CUDA graph) |

`torch` threads are set to 8 (`REKI_THREADS`): the container had a CFS quota of ~22 CPUs while torch saw
over 100 cores, and with the default thread count CPU preprocessing was throttled every other call
(median 25.7 ms, p99 159 ms in a first attempt; not kept).

## Method

**Why Euler steps 2–10.** In the compiled path one predict runs 2,366 kernels in the prefill graph and
2,453 in the expert graph. Steps 2–10 of the expert are ≈2,047 kernels (≈227 per step, ≈23 per layer:
cuBLAS GEMMs at M = 10 rows, flash varlen attention, Inductor Triton fusions) and take 6.99 ms while
moving ~212 MB of weights per step (106 M bf16 parameters) — an HBM floor of 63 µs per step, 0.57 ms for
9 steps, 12× below the measured loop. The loop is identical from step to step, small, and
launch/latency bound: the best megakernel target. Step 1 differs (it also processes the 32 register
rows and produces the register K/V), so it, the per-step modulations and `register_prepare` (the
cross-attention K/V of the VLM source) stay Kyutai's compiled code.

**Kernel** ([`src/mk_expert.py`](src/mk_expert.py), v4, 516 lines of Triton + Python glue). One launch of 132 programs
(one per SM, co-resident) runs steps 2–10. Per step: 10 layers × 8 stages — (1) AdaRMS + QKV GEMM,
(2) self-attention per head over [cached registers ∥ own actions] with q/k RMSNorm + RoPE, (3) out-proj +
gated residual, (4) AdaRMS + cross-attention query, (5) cross-attention per head over the cached VLM K/V
(online softmax), (6) out-proj + gated residual, (7) AdaRMS + gate/up + SiLU·mul, (8) down + gated
residual — then a step tail over 72 programs (final AdaRMS, fp32 out-projection, Euler update, fp32
in-projection of the next step). Stages are separated by a software grid barrier (monotonic atomic
counter): 1 + 9 × 81 = 730 barriers per launch at ≈1.0 µs each (barrier-only kernel), ≈0.73 ms of the
4.03 ms. Activations live in small L2-resident scratch buffers read with `.cg` loads. Numerics follow
Inductor: bf16 GEMM inputs, fp32 accumulation, one bf16 rounding per Linear output and per fused
elementwise op. `install()` swaps REKI's `_expert_actions` for a version that calls their step-1 code
and then the megakernel; REKI's own CUDA-graph capture (and its capture check) wraps both.

Per-stage medians (µs, including the barrier; `%globaltimer` stamps): QKV 5.5, SA-attn 4.7, SA-o 3.2,
CA-q 5.2, CA-attn 5.3, CA-o 3.5, gate/up 7.5, down 7.4 → 42 µs per layer, plus a 21 µs step tail.
Every stage is latency bound (a barrier plus one or two dependent HBM/L2 round trips), not bandwidth
bound.

**Versions tried** (steps 2–10, graph replay on the same expert inputs; Kyutai compiled: 6.99 ms;
[figure f](#figures)):

| version | idea | ms | source |
|---|---|---:|---|
| v1 ([`archive/`](archive/)) | K-looped GEMM tiles (BK 128), one-program step tail | 5.89 | report; the log of this run was not kept — re-measured in verification |
| v1 tuned | BK 256, 8 warps | 4.84 | `results/original/logs/log_tune5.txt` |
| v2 | single-shot tiles + next stage's weights prefetched into registers before each barrier | 63.3 (13.4 with 4 warps) | `log_tune2.txt`; register spills |
| v3 | single-shot tiles, no large prefetch | 48.6; 10.8 with `maxnreg=255` | `log_tune3/4.txt`; 1.2–1.9 KB spills |
| **v4** (final) | v1 tiles + one-round-trip RMS pass + parallel step tail, 4 warps | **4.01** | `log_tune6–8.txt` |
| v4 + L2 prefetch | idle programs `prefetch.global.L2` the next stages' weights | 4.17–4.18 | `log_tune7/8.txt`: GEMM stages gain, barriers wait on the prefetchers |
| v4, 2 programs per SM | more tiles in flight | 4.17–4.45 | `log_tune9.txt` |

**Mirage MPK** (branch `mpk` @ `ca58a02`) was evaluated and not used: its task library targets LLM
decode (embedding, RMSNorm + linear, paged/MLA decode attention, MoE, sampling) and has none of the
blocks the expert needs (AdaRMS modulation, gated residuals, non-causal attention over cached register
K/V and over an external VLM source with MRoPE positions, fp32 action projections), nor a ViT or
Gated-DeltaNet layer for the prefill.

## Results

### Latency (end-to-end `predict`, 880 timed calls per row)

| path | median ms | p99 ms | p10 / p90 ms | speed-up vs compiled (same run) | kernels per predict |
|---|---:|---:|---:|---:|---|
| eager | 156.2 | 187.8 | 147.7 / 167.6 | 0.14× | 13,972 (13,971 host launches) |
| compiled, run A | 21.73 | 24.98 | 21.10 / 22.71 | 1.00× | 4,837 = 2,366 prefill graph + 2,453 expert graph + 18 outside graphs |
| compiled, run B | 20.94 | 21.95 | 20.86 / 21.12 | 1.00× | same |
| **megakernel, run A** | **18.41** | **18.74** | 18.27 / 18.53 | **1.18×** | 2,791 = 2,366 + 407 (incl. 1 megakernel) + 18 |
| **megakernel, run B** | **18.33** | **18.72** | 18.19 / 18.50 | **1.14×** | same |

Source: `results/original/latency_*.json` (run A), `results/original/run_b/` (run B, compiled and megakernel
back to back), `kernel_counts.json`, `breakdown_*.json`. The two compiled runs differ by 0.8 ms in the
median and run A has a heavier tail; the megakernel runs differ by 0.08 ms. The megakernel saves 2.6–3.3 ms
per chunk (12–15 %) and tightens p99.

### The replaced part in isolation (CUDA-graph replay, median of 200)

| expert part | Kyutai compiled | megakernel path |
|---|---:|---:|
| modulations + `register_prepare` + Euler step 1 | 1.32 ms | 1.33 ms (theirs, unchanged) |
| Euler steps 2–10 | 6.99 ms (0.78 ms/step, ≈2,047 kernels) | **4.03 ms** (1 launch, 0.45 ms/step) — **1.74×** |
| whole expert graph | 8.26 ms (2,453 kernels) | 5.36 ms (407 kernels) |

### Where the time goes (GPU timeline, median of warm predicts)

| stage | compiled ms | megakernel ms | notes |
|---|---:|---:|---|
| host preprocessing | 2.16 | 2.00 | CPU: image resize/patchify, tokenization, state normalization, H2D |
| VLM prefill | 10.98 | 10.95 | graph replay 9.15 ms (vision tower 2.87, LM ≈ 6.3) + host glue; 2,366 kernels |
| CA-source gather + state cond | 0.05 | 0.05 | |
| action expert | 8.21 | 5.49 | |
| register append + decode + D2H | 0.23 | 0.23 | |
| **total** | **21.63** | **18.75** | |

Eager (instrumented, 191.8 ms total): 50.8 ms prefill + 136.6 ms expert, host-launch bound.

### Correctness

Same 440 frames, same noise, register memory carried; max |Δ| over the 7 real action dims × 10 actions
of the normalized chunk (values within ±1.03):

| pair (440 chunks) | max | mean | p99 of per-chunk max |
|---|---:|---:|---:|
| megakernel vs compiled | 5.5e-3 | 7.6e-4 | 4.9e-3 |
| compiled vs eager | 6.9e-3 | 8.9e-4 | 5.8e-3 |
| megakernel vs eager | 5.8e-3 | 6.9e-4 | 4.8e-3 |

The megakernel is as close to the compiled path as Kyutai's eager and compiled paths are to each other
(bf16 rounding order; neither Inductor nor the megakernel is bit-exact against eager). The register cache
is produced by step 1, which stays Kyutai's, so differences do not accumulate across frames. Single-input
check (`src/test_mk.py`): 2.3e-3 megakernel vs compiled (`results/original/logs/`); the
compiled-vs-eager value of that check was not saved in the original run (see verification).

## Verification

VERIFICATION_PLACEHOLDER

## Figures

FIGURES_PLACEHOLDER

## Limitations

- One GPU type (H100 SXM), one cloud machine per run, batch 1, one checkpoint (LIBERO finetune).
- No closed-loop LIBERO evaluation: correctness is measured as |Δ| of the action chunks against Kyutai's
  own paths, not as task success.
- The compiled path varies from run to run (median 20.9 vs 21.7 ms in the original runs); compare speed-ups
  within a run.
- The megakernel is specialized to this expert's shapes (≤ 16 action rows, width 768, 12 heads, batch 1)
  and to H100's 132 SMs being co-resident; it relies on all programs being resident (a software grid
  barrier deadlocks otherwise).
- Only Euler steps 2–10 are fused; the VLM prefill, step 1 and the host preprocessing are Kyutai's code.

## What a full-model megakernel would need

- **Faster expert loop** (4.0 → ~1.5–2 ms, estimate from the stage profile): stream each stage's weights
  into shared memory with TMA / `cp.async.bulk` one or two stages ahead (no registers held, which is what
  broke v2/v3 in Triton), dataflow counters instead of grid-wide barriers (MPK / ThunderKittens style),
  split-head attention — i.e. CUDA C++ rather than Triton.
- **Euler step 1 + modulations + `register_prepare`** (1.32 ms): register rows with plain norms, segmented
  varlen attention over the right-aligned register buffer, the register K/V output and the cross-attention
  K/V projections of the VLM source (31 M weights).
- **VLM prefill** (9.15 ms, 2,366 kernels, the largest lever): a ViT (varlen attention over 2 × 256 patches,
  2-D RoPE, merger), 18 Gated-DeltaNet layers (causal conv + chunked delta-rule scan, today FLA Triton
  kernels) and 6 gated full-attention layers with MRoPE at M ≈ 180–260 tokens. Estimated floor ≈ 1–2 ms
  (~3 GB of non-embedding weights at 3.35 TB/s plus ~0.5–0.8 TFLOP) — up to ~7 ms to gain, but no
  megakernel framework has these blocks today.
- **Host preprocessing** (2.0–2.2 ms on the CPU) is outside any GPU kernel: move it to the GPU or overlap it
  with the previous chunk.

## Implications for real-time control

A 60 Hz loop has 16.7 ms per tick. Kyutai's compiled path (20.9–21.7 ms) misses it by 4.3–5.1 ms; the
megakernel path (18.3–18.4 ms) by 1.7 ms. Taking the 2.0 ms host preprocessing off the critical path would
bring the megakernel path to ≈ 16.3–16.4 ms (estimate, not measured) — just inside the budget, with no
room for camera, network or actuation latency. A 30 Hz loop (33.3 ms) fits every compiled path. REKI
itself predicts 1-s chunks (10 actions at 10 Hz on LIBERO) executed open loop between predictions; for a
fresh chunk every tick at 60 Hz, the remaining lever is the VLM prefill, not the expert.

## Reproduce

On a machine with one NVIDIA H100 (driver ≥ 570), git, make and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/muninn-huginn/reki-megakernel && cd reki-megakernel
make setup                     # .venv, REKI @ b2a3393, pinned torch/triton/transformers/fla, weights + LIBERO frames
source .venv/bin/activate
make all OUT=results/verification   # latency, correctness, kernels, breakdown, single-input check, version ladder
make figures                   # figures/*.png|svg and results/csv/*.csv
```

`make all` takes ≈ 40 min on an H100 SXM. Individual targets: `make latency | correctness | kernels |
breakdown | check | tune`. Inputs go to `data/` (override with `REKI_CKPT`, `REKI_DATA`); CPU threads
with `REKI_THREADS` (default 8).

## Layout

```
src/mk_expert.py          the megakernel (v4) + install() into a REKI model
src/harness.py            frames, fixed noise, timing, chunk capture
src/bench.py              latency + chunks of one path (eager | compiled | mk)
src/compare.py            max |Δ| between paths
src/count_kernels.py      kernels inside the prefill / expert graphs
src/profile_breakdown.py  stage split, isolated graph replays, kernels per predict
src/test_mk.py            single-input check of the megakernel against eager / compiled steps 2–10
src/tune_mk.py            barrier cost, per-stage timeline, tile configs, any kernel version
archive/mk_expert_v1-3.py superseded kernel versions (see "Versions tried"), kept for the ladder
results/original/         the study run: latency (per-call times), breakdowns, kernel counts, correctness, tuning logs
results/verification/     re-run from a fresh clone
results/csv/              tidy CSVs behind each figure
figures/                  make_figures.py and the figures (PNG + SVG)
scripts/setup.sh          environment and inputs
```

## Credits

- [Kyutai](https://kyutai.org/reki/) for REKI (code MIT, weights CC-BY-NC-SA 4.0). The megakernel
  re-implements their action expert, and the scripts drive their inference API; see [NOTICE](NOTICE).
- Hazy Research / [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) and their
  megakernel work for the idea and design vocabulary (no code used).
- [Mirage MPK](https://github.com/mirage-project/mirage) was evaluated as a framework and not used.
- LIBERO frames from [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero).

## License

MIT ([LICENSE](LICENSE)); third-party notices in [NOTICE](NOTICE).
