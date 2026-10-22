"""Latency ladder + correctness dump of one REKI inference path on real LIBERO frames.

    python bench.py --mode eager|compiled|mk --out results/

Pass 1 predicts every frame of the selected episodes in order (register memory carried, reset per
episode) with fixed noise and saves the normalized chunks (also the warm-up: it captures every graph).
Then the same sequence is replayed, timed per call, until --min-calls calls are timed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from harness import CKPT, DATA, ChunkTap, FixedNoise, load_episodes, load_policy, run_episodes, stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['eager', 'compiled', 'mk'], required=True)
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--data', default=DATA)
    p.add_argument('--max-episodes', type=int, default=24)
    p.add_argument('--min-calls', type=int, default=500)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(a.data)[: a.max_episodes]
    n_frames = sum(len(f) for _, f in episodes)
    print(f'{len(episodes)} episodes, {n_frames} frames, {len({f[0]["task"] for _, f in episodes})} tasks', flush=True)

    t0 = time.time()
    pol = load_policy(a.ckpt, compile=a.mode != 'eager')
    if a.mode == 'mk':
        from mk_expert import install
        install(pol.policy.model)
    model = pol.policy.model
    noise = FixedNoise(model)
    tap = ChunkTap(pol.policy)
    print(f'load {time.time() - t0:.1f}s', flush=True)

    t0 = time.time()
    first = run_episodes(pol, episodes, noise)
    print(f'pass 1 (correctness + warm-up/capture) {time.time() - t0:.1f}s', flush=True)
    torch.save({'episodes': [e for e, _ in episodes], 'chunks': torch.stack(tap.chunks)}, out / f'chunks_{a.mode}.pt')
    tap.on = False

    lat = []
    while len(lat) < a.min_calls:
        noise.i = 0
        lat += run_episodes(pol, episodes, noise)
    s = stats(lat)
    s.update(mode=a.mode, gpu=torch.cuda.get_device_name(), torch=torch.__version__, frames=n_frames,
             first_pass=stats(first))
    print(json.dumps(s), flush=True)
    (out / f'latency_{a.mode}.json').write_text(json.dumps({**s, 'lat_ms': lat}) + '\n')


if __name__ == '__main__':
    main()
