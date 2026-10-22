# Reproduce the study on one NVIDIA H100 (run `make setup` first, then `source .venv/bin/activate`).
# `make all` re-runs everything into $(OUT) (~40 min on an H100 SXM, most of it eager + torch.compile warm-up).
OUT ?= results/verification
PY ?= python
export REKI_THREADS ?= 8

.PHONY: setup all latency correctness kernels breakdown check tune figures

setup:
	bash scripts/setup.sh

all: latency correctness kernels breakdown check tune

# 880 timed calls per path (2 passes over 440 frames) after an untimed capture pass; compiled and megakernel
# are run twice back to back (run_a, run_b) because the compiled path varies from run to run.
latency:
	$(PY) src/bench.py --mode eager --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --out $(OUT)/run_a
	$(PY) src/bench.py --mode mk --out $(OUT)/run_a
	$(PY) src/bench.py --mode compiled --out $(OUT)/run_b
	$(PY) src/bench.py --mode mk --out $(OUT)/run_b

# max |Δ| of the normalized action chunks of run_a (chunks_*.pt written by bench.py)
correctness:
	$(PY) src/compare.py $(OUT)/run_a

kernels:
	$(PY) src/count_kernels.py --out $(OUT)

breakdown:
	$(PY) src/profile_breakdown.py --mode eager --out $(OUT)
	$(PY) src/profile_breakdown.py --mode compiled --out $(OUT)
	$(PY) src/profile_breakdown.py --mode mk --out $(OUT)

# single-input check of the megakernel vs eager / compiled Euler steps 2..10, plus graph-replay times
check:
	$(PY) src/test_mk.py $(OUT)/test_mk.json

# ladder of kernel versions (archive/ v1-v3, src/ v4) on the same expert inputs
tune:
	$(PY) src/tune_mk.py --impl mk_expert_v1 --configs '{}' '{"BK": 256, "num_warps": 8}' --out $(OUT)/tune/v1.json
	$(PY) src/tune_mk.py --impl mk_expert_v2 --configs '{}' '{"num_warps": 4}' --out $(OUT)/tune/v2.json
	$(PY) src/tune_mk.py --impl mk_expert_v3 --configs '{}' '{"maxnreg": 255}' --out $(OUT)/tune/v3.json
	$(PY) src/tune_mk.py --impl mk_expert --configs '{}' '{"L2PF": true}' '{"GRID": 2}' --out $(OUT)/tune/v4.json

figures:
	$(PY) figures/make_figures.py
