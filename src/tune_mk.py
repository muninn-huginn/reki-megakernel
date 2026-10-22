"""Megakernel tuning: barrier cost, per-stage timeline (%globaltimer at each grid barrier), tile configs.

    python src/tune_mk.py [--impl mk_expert_v1] [--configs '{"num_warps": 8}' ...] [--out tune.json]

--impl takes any module with an ``ExpertMegakernel`` class: ``mk_expert`` (src/) or the superseded
``mk_expert_v1``/``_v2``/``_v3`` (archive/).
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'archive'))
from harness import CKPT, DATA, FixedNoise, load_episodes, load_policy  # noqa: E402
from mk_expert import _grid_sync  # noqa: E402
from profile_breakdown import graph_time  # noqa: E402

STAGES = ['qkv', 'sa_attn', 'sa_o+res', 'ca_q', 'ca_attn', 'ca_o+res', 'gate_up', 'down+res']


@triton.jit
def _barrier_only(bar_ptr, prof_ptr, n, POLL: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for i in range(n):
        _grid_sync(bar_ptr, i + 1, nprog, prof_ptr, pid, POLL, False)


def setup():
    episodes = load_episodes(DATA)[:1]
    pol = load_policy(CKPT, compile=True)
    model = pol.policy.model
    FixedNoise(model)
    pol.reset()
    for fr in episodes[0][1][:4]:
        pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
    ent = next(iter(model._expert_graphs.graphs.values()))
    _, rc, adim, nsteps = ent.args
    with torch.no_grad():
        mods, dt = model._step_modulations(nsteps, ent.state_cond)
        x1, _, _ = model._expert_first_step(ent.noise, ent.past, rc, mods[:1], dt, adim)
        x1 = x1.clone()
        x = x1
        for s in range(1, nsteps):
            x = model._expert_step(x, ent.past, mods[s:s + 1], dt, adim)
    return model, ent, mods, dt, x1, x.clone(), adim, nsteps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='*', default=['{}'])
    p.add_argument('--impl', default='mk_expert', help='module with ExpertMegakernel (mk_expert, mk_expert_v1, ...)')
    p.add_argument('--out', default=None, help='also write the result JSON here')
    a = p.parse_args()
    model, ent, mods, dt, x1, xref, adim, nsteps = setup()
    mk = importlib.import_module(a.impl).ExpertMegakernel(model)
    out = {}
    nb = mk.barriers(nsteps - 1)
    for poll in (0, 1):
        f = lambda: (mk.bar.zero_(), _barrier_only[(mk.nprog,)](mk.bar, mk.prof, nb, POLL=poll, num_warps=4))  # noqa: E731
        out[f'barrier_only_poll{poll}_us_per_barrier'] = graph_time(f) * 1e3 / nb
    base = dict(mk.cfg)
    for c in a.configs:
        cfg = {**base, **json.loads(c)}
        mk.cfg = cfg
        try:
            with torch.no_grad():
                x = mk.run(x1, mods, dt, ent.past, adim)
                torch.cuda.synchronize()
            diff = float((x - xref)[..., :adim].abs().max())
            diff_all = float((x - xref).abs().max())
            ck = getattr(mk, 'compiled', None)
            regs = {'n_regs': getattr(ck, 'n_regs', None), 'n_spills': getattr(ck, 'n_spills', None)}
            ms = graph_time(lambda: mk.run(x1, mods, dt, ent.past, adim))
            mk.cfg = {**cfg, 'PROFILE': True}
            mk.prof.zero_()
            mk.run(x1, mods, dt, ent.past, adim)
            torch.cuda.synchronize()
            ts = mk.prof[: nb + 1].cpu().numpy().astype(np.int64)
            d = np.diff(ts[1:]) / 1e3  # us between consecutive barriers (barrier 1 .. nb)
            per = {k: [] for k in STAGES + ['tail']}
            per_step = 8 * mk.L + 1
            for i, v in enumerate(d):  # d[i] = time from barrier i+1 to i+2
                j = i % per_step
                per['tail' if j == per_step - 1 else STAGES[j % 8]].append(v)
            out[c] = {'max_abs_diff_vs_compiled': diff, 'max_abs_diff_all_dims': diff_all, 'graph_ms': ms, **regs,
                      'stage_us_median': {k: float(np.median(v)) for k, v in per.items()},
                      'stage_us_sum_ms': {k: float(np.sum(v)) / 1e3 for k, v in per.items()}}
        except Exception as e:  # report and continue with the next config
            out[c] = {'error': repr(e)[:400]}
        mk.cfg = base
        print(c, json.dumps(out[c]), flush=True)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'impl': a.impl, **out}, indent=1) + '\n')


if __name__ == '__main__':
    main()
