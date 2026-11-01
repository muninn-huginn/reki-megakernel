# Reproduce the study on one NVIDIA H100 (run `make setup` first, then `source .venv/bin/activate`).
# `make all` re-runs everything into $(OUT) (~75 min on an H100 SXM, most of it eager + torch.compile warm-up).
OUT ?= results/verification
PY ?= python
export REKI_THREADS ?= 8

.PHONY: setup all latency correctness kernels breakdown check tune figures gpu

setup:
	bash scripts/setup.sh

all: latency correctness kernels breakdown check tune

# 880 timed calls per path (2 passes over 440 frames) after an untimed capture pass. Paths: eager, Kyutai's
# compiled path, the megakernel; then both with the host fixes (--fast-host: no per-call eval(), images
# preprocessed on the GPU) and with the compiled prefill on top (--compile-prefill). The compiled and
# megakernel paths are run a second time back to back (run_b) because the compiled path varies from run to run.
latency:
	$(PY) src/bench.py --mode eager --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --out $(OUT)/run_a
	$(PY) src/bench.py --mode mk --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --fast-host --out $(OUT)/run_a
	$(PY) src/bench.py --mode mk --fast-host --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --fast-host --compile-prefill --out $(OUT)/run_a
	$(PY) src/bench.py --mode mk --fast-host --compile-prefill --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --out $(OUT)/run_b
	$(PY) src/bench.py --mode mk --out $(OUT)/run_b
	$(PY) src/bench.py --mode compiled --fast-host --compile-prefill --out $(OUT)/run_b
	$(PY) src/bench.py --mode mk --fast-host --compile-prefill --out $(OUT)/run_b

# max |Δ| of the normalized action chunks of run_a (chunks_*.pt written by bench.py)
correctness:
	$(PY) src/compare.py $(OUT)/run_a

kernels:
	$(PY) src/count_kernels.py --out $(OUT)

breakdown:
	$(PY) src/profile_breakdown.py --mode eager --out $(OUT)
	$(PY) src/profile_breakdown.py --mode compiled --out $(OUT)
	$(PY) src/profile_breakdown.py --mode mk --out $(OUT)
	$(PY) src/profile_breakdown.py --mode compiled --fast-host --compile-prefill --out $(OUT)
	$(PY) src/profile_breakdown.py --mode mk --fast-host --compile-prefill --out $(OUT)

# single-input check of the megakernel vs eager / compiled Euler steps 2..10, plus graph-replay times
check:
	$(PY) src/test_mk.py $(OUT)/test_mk.json

# ladder of kernel versions (archive/ v1-v4, src/ v5) on the same expert inputs
tune:
	$(PY) src/tune_mk.py --impl mk_expert_v1 --configs '{}' '{"BK": 256, "num_warps": 8}' --out $(OUT)/tune/v1.json
	$(PY) src/tune_mk.py --impl mk_expert_v2 --configs '{}' '{"num_warps": 4}' --out $(OUT)/tune/v2.json
	$(PY) src/tune_mk.py --impl mk_expert_v3 --configs '{}' '{"maxnreg": 255}' --out $(OUT)/tune/v3.json
	$(PY) src/tune_mk.py --impl mk_expert_v4 --configs '{}' '{"L2PF": true}' '{"GRID": 2}' --out $(OUT)/tune/v4.json
	$(PY) src/tune_mk.py --impl mk_expert --configs '{}' '{"PF": false}' --out $(OUT)/tune/v5.json

# Another GPU: eager (one timed pass, as the |Δ| reference), Kyutai compiled and the megakernel, each alone and
# with the host fixes + compiled prefill; |Δ|, single-input check, breakdown of the fastest path, v5 with and
# without its L2 prefetch. ~25 min. results/gpus/<gpu>/ in the repo.
gpu:
	$(PY) src/bench.py --mode eager --min-calls 1 --out $(OUT)
	$(PY) src/bench.py --mode compiled --out $(OUT)
	$(PY) src/bench.py --mode mk --out $(OUT)
	$(PY) src/bench.py --mode compiled --fast-host --compile-prefill --out $(OUT)
	$(PY) src/bench.py --mode mk --fast-host --compile-prefill --out $(OUT)
	$(PY) src/compare.py $(OUT)
	$(PY) src/test_mk.py $(OUT)/test_mk.json
	$(PY) src/profile_breakdown.py --mode mk --fast-host --compile-prefill --out $(OUT)
	$(PY) src/tune_mk.py --configs '{}' '{"PF": false}' --out $(OUT)/tune_v5.json

figures:
	$(PY) figures/make_figures.py
