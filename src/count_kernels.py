"""Kernels inside each captured CUDA graph of one predict (CUPTI via torch.profiler, one replay each):
VLM prefill graph and expert graph, for the compiled and the megakernel path, each with Kyutai's prefill
graph and with the compiled prefill (``fast_prefill.py``).

    python src/count_kernels.py --out results/run  ->  results/run/kernel_counts.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent))
from harness import CKPT, DATA, FixedNoise, load_episodes, load_path  # noqa: E402


def kernels_in(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
             and 'memcpy' not in e.name.lower() and 'memset' not in e.name.lower()]
    return len(names), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results')
    a = p.parse_args()
    frames = load_episodes(DATA)[0][1][:4]
    out = {}
    for mode, cprefill in (('compiled', False), ('mk', False), ('compiled', True), ('mk', True)):
        pol, tag = load_path(CKPT, mode, compile_prefill=cprefill)
        model = pol.policy.model
        FixedNoise(model)
        pol.reset()
        for fr in frames:
            pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
        lay = next(iter(model._prefill_graphs.graphs.values()))
        ent = next(iter(model._expert_graphs.graphs.values()))
        n_pre, _ = kernels_in(lay.graph.replay)
        n_exp, names = kernels_in(ent.graph.replay)
        out[tag] = {'prefill_graph_kernels': n_pre, 'expert_graph_kernels': n_exp,
                    'expert_megakernel_launches': sum('expert_megakernel' in n for n in names)}
        print(tag, out[tag], flush=True)
        del pol, model, lay, ent
        torch.cuda.empty_cache()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'kernel_counts.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
