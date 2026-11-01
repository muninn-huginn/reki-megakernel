"""Where one REKI predict spends its time, and how many kernels it launches.

    python profile_breakdown.py --mode compiled|eager|mk [--fast-host] [--compile-prefill] --out results/

1. Stage split of warm predicts (CUDA events + host clocks at the stage boundaries): host
   preprocessing, VLM prefill (vision + LM), CA-source gather + state cond, action expert,
   register-cache append, decode / device->host copy.
2. compiled/mk: sub-splits from replaying graphs captured here in isolation: vision tower vs whole
   prefill; expert step modulations + register_prepare + Euler step 1 vs one Euler step 2..N.
3. Kernel count per predict and per stage (torch.profiler / CUPTI, kernels inside CUDA graphs included).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import reki.inference.policy as rpol  # noqa: E402
from harness import CKPT, DATA, FixedNoise, load_episodes, load_path  # noqa: E402

STAGES = ['preprocess', 'prefill', 'gather_state', 'expert', 'register_append', 'decode_d2h']


class Marks:
    def __init__(self):
        self.ev, self.host = [], []

    def mark(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev.append(e)
        self.host.append(time.perf_counter())


class Wrap:
    """Callable object wrapper that marks after the call (for objects called via __call__)."""

    def __init__(self, inner, marks):
        self.inner, self.marks = inner, marks

    def __call__(self, *a, **k):
        out = self.inner(*a, **k)
        self.marks.mark()
        return out

    def __getattr__(self, n):
        return getattr(self.inner, n)


def instrument(pol, marks):
    model = pol.policy.model
    orig_bis = rpol.build_inference_sample

    def bis(*a, **k):
        out = orig_bis(*a, **k)
        marks.mark()
        return out
    rpol.build_inference_sample = bis
    vwe = model.vlm_with_expert
    orig_gather = vwe.gather_cross_attn_source
    orig_state = model._state_adaln_cond

    def gather(*a, **k):
        return orig_gather(*a, **k)

    def state(*a, **k):
        out = orig_state(*a, **k)
        marks.mark()
        return out
    vwe.gather_cross_attn_source = gather
    model._state_adaln_cond = state
    if model._prefill_graphs is not None:
        model._prefill_graphs = Wrap(model._prefill_graphs, marks)
    else:
        orig_prefill = model._prefill

        def prefill(*a, **k):
            out = orig_prefill(*a, **k)
            marks.mark()
            return out
        model._prefill = prefill
    if model._expert_graphs is not None:
        model._expert_graphs = Wrap(model._expert_graphs, marks)
    else:
        orig_exp = model._expert_actions

        def exp(*a, **k):
            out = orig_exp(*a, **k)
            marks.mark()
            return out
        model._expert_actions = exp
    rc = pol.policy._register_cache
    orig_app = rc.append_frame

    def app(*a, **k):
        out = orig_app(*a, **k)
        marks.mark()
        return out
    rc.append_frame = app


def graph_time(fn, n=200, capture=True):
    """Median time (ms) of fn replayed as a CUDA graph (capture=False: fn already replays a graph)."""
    if capture:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def kernel_counts(pol, frames, marks_on):
    """Kernels (CUPTI kernel activities, incl. those replayed inside CUDA graphs) per predict."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for fr in frames:
            pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
        torch.cuda.synchronize()
    ev = prof.events()
    kern = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA and 'memcpy' not in e.name.lower()
            and 'memset' not in e.name.lower()]
    graph_launches = sum(1 for e in ev if e.name in ('cudaGraphLaunch',))
    launches = sum(1 for e in ev if e.name in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC',
                                               'cuLaunchKernelEx'))
    names = defaultdict(int)
    for e in kern:
        names[e.name[:80]] += 1
    top = sorted(names.items(), key=lambda kv: -kv[1])[:25]
    n = len(frames)
    return {'kernels_per_predict': len(kern) / n, 'host_kernel_launch_calls_per_predict': launches / n,
            'graph_launches_per_predict': graph_launches / n, 'top_kernels': top}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['eager', 'compiled', 'mk'], required=True)
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--data', default=DATA)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--out', default='results')
    p.add_argument('--fast-host', action='store_true', help='fast_host.install (tag _fasthost)')
    p.add_argument('--compile-prefill', action='store_true', help='fast_prefill.install (tag _cprefill)')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(a.data)[: a.episodes]
    pol, tag = load_path(a.ckpt, a.mode, a.fast_host, a.compile_prefill)
    model = pol.policy.model
    noise = FixedNoise(model)
    frames = [f for _, fr in episodes for f in fr]
    # warm-up (captures graphs, fills the register cache)
    pol.reset()
    for fr in frames[:12]:
        pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
    torch.cuda.synchronize()

    res = {'mode': tag, 'gpu': torch.cuda.get_device_name()}
    res['kernels'] = kernel_counts(pol, frames[:5], False)

    marks = Marks()
    instrument(pol, marks)
    per = defaultdict(list)
    per_host = defaultdict(list)
    tot = []
    for rep in range(3):
        for fr in frames[: len(frames)]:
            torch.cuda.synchronize()
            marks.ev, marks.host = [], []
            marks.mark()
            pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
            marks.mark()
            torch.cuda.synchronize()
            if len(marks.ev) != len(STAGES) + 1:
                raise RuntimeError(f'{len(marks.ev)} marks')
            for i, s in enumerate(STAGES):
                per[s].append(marks.ev[i].elapsed_time(marks.ev[i + 1]))
                per_host[s].append((marks.host[i + 1] - marks.host[i]) * 1e3)
            tot.append(marks.ev[0].elapsed_time(marks.ev[-1]))
    res['stages_gpu_timeline_ms'] = {s: float(np.median(v)) for s, v in per.items()}
    res['stages_host_ms'] = {s: float(np.median(v)) for s, v in per_host.items()}
    res['total_ms'] = float(np.median(tot))

    if a.mode != 'eager':
        pg = model._prefill_graphs.inner
        lay = next(iter(pg.graphs.values()))
        visual = lay.vwe.image_encoder

        def vis():
            h = visual.patch_embed(lay.pixels) + lay.vis_pos
            for block in visual.blocks:
                h = block(h, cu_seqlens=lay.vis_cu, position_embeddings=lay.vis_pe)
            return visual.merger(h)
        sub = {'prefill_graph_ms': graph_time(lay.graph.replay, capture=False), 'vision_tower_ms': graph_time(vis)}
        eg = model._expert_graphs.inner
        ent = next(iter(eg.graphs.values()))
        sub['expert_graph_ms'] = graph_time(ent.graph.replay, capture=False)
        mdl, rc, adim, nsteps = ent.args
        st = {}

        def first():
            mods, dt = model._step_modulations(nsteps, ent.state_cond)
            st['x'], _, _ = model._expert_first_step(ent.noise, ent.past, rc, mods[:1], dt, adim)
            st['mods'], st['dt'] = mods, dt
        first()
        sub['mods_prepare_step1_ms'] = graph_time(first)
        x0 = st['x'].clone()
        sub['one_euler_step_ms'] = graph_time(
            lambda: model._expert_step(x0, ent.past, st['mods'][1:2], st['dt'], adim))

        def loop():
            x = x0
            for s in range(1, nsteps):
                x = model._expert_step(x, ent.past, st['mods'][s:s + 1], st['dt'], adim)
        sub['ref_steps_2_to_N_ms'] = graph_time(loop)
        if a.mode == 'mk':
            sub['mk_steps_2_to_N_ms'] = graph_time(lambda: model._mk.run(x0, st['mods'], st['dt'], ent.past, adim))
        res['subsplit_graph_replay_ms'] = sub
    print(json.dumps(res, indent=1))
    (out / f'breakdown_{tag}.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
