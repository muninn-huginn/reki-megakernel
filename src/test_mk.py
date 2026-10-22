"""Isolated check of the megakernel against Euler steps 2..N of the reference, on real expert inputs.

Loads the compiled policy, predicts a few LIBERO frames (fills the register cache, captures graphs),
then from the captured expert inputs runs step modulations + step 1 (reference), and integrates
steps 2..N three ways: eager reference (uncompiled ``_expert_step``), compiled reference (their
path) and the megakernel. Prints max |Δ| and graph-replay times.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from harness import CKPT, DATA, FixedNoise, load_episodes, load_policy  # noqa: E402
from mk_expert import ExpertMegakernel  # noqa: E402
from profile_breakdown import graph_time  # noqa: E402


def main():
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
        eager_step = type(model)._expert_step.__get__(model)
        xs = {}
        for name, fn in (('eager', eager_step), ('compiled', model._expert_step)):
            x = x1
            for s in range(1, nsteps):
                x = fn(x, ent.past, mods[s:s + 1], dt, adim)
            xs[name] = x.clone()
        mk = ExpertMegakernel(model)
        xs['mk'] = mk.run(x1, mods, dt, ent.past, adim).clone()
        torch.cuda.synchronize()
        bar = int(mk.bar.item())
    res = {'barrier_count': bar, 'expected': mk.nprog * mk.barriers(nsteps - 1), 'nprog': mk.nprog,
           'n_valid': int(rc.n_valid.item()), 'S': int(ent.past.ca_cu[1].item()), 'S_pad': ent.past.ca_src.shape[0]}
    for a_, b_ in (('compiled', 'eager'), ('mk', 'compiled'), ('mk', 'eager')):
        d = (xs[a_] - xs[b_]).abs()
        res[f'max|{a_}-{b_}| action dims'] = float(d[..., :adim].max())
        res[f'max|{a_}-{b_}| all dims'] = float(d.max())
    res['ref_max_abs_x'] = float(xs['compiled'][..., :adim].abs().max())

    def loop():
        x = x1
        for s in range(1, nsteps):
            x = model._expert_step(x, ent.past, mods[s:s + 1], dt, adim)
    res['ref_steps_2_to_N_graph_ms'] = graph_time(loop)
    res['mk_steps_2_to_N_graph_ms'] = graph_time(lambda: mk.run(x1, mods, dt, ent.past, adim))
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 1:  # python src/test_mk.py results/run/test_mk.json
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
