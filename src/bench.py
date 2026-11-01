"""Latency ladder + correctness dump of one REKI inference path on real LIBERO frames.

    python bench.py --mode eager|compiled|mk [--fast-host] [--compile-prefill] --out results/

--fast-host applies ``fast_host.install`` (no per-call ``eval()``, image preprocessing on the GPU) and
adds ``_fasthost`` to the output names; it first checks that the GPU image processor matches the CPU
one bitwise on every frame. --compile-prefill applies ``fast_prefill.install`` (vision blocks and LM
layers compiled inside the prefill graph) and adds ``_cprefill``.

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
from harness import CKPT, DATA, ChunkTap, FixedNoise, load_episodes, load_path, run_episodes, stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['eager', 'compiled', 'mk'], required=True)
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--data', default=DATA)
    p.add_argument('--max-episodes', type=int, default=24)
    p.add_argument('--min-calls', type=int, default=500)
    p.add_argument('--out', default='results')
    p.add_argument('--fast-host', action='store_true')
    p.add_argument('--compile-prefill', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(a.data)[: a.max_episodes]
    n_frames = sum(len(f) for _, f in episodes)
    print(f'{len(episodes)} episodes, {n_frames} frames, {len({f[0]["task"] for _, f in episodes})} tasks', flush=True)

    t0 = time.time()
    pol, tag = load_path(a.ckpt, a.mode, a.fast_host, a.compile_prefill)
    extra = {}
    if a.fast_host:
        import fast_host
        chk = fast_host.check_pixels(pol.preprocessor._fast_img, [f for _, fr in episodes for f in fr],
                                     list(pol.spec.camera_keys))
        print('pixel check', json.dumps(chk), flush=True)
        if chk['bitwise_equal'] != chk['frames']:
            raise SystemExit('GPU image processor differs from the CPU one')
        extra['pixel_check'] = chk
    model = pol.policy.model
    noise = FixedNoise(model)
    tap = ChunkTap(pol.policy)
    print(f'load {time.time() - t0:.1f}s', flush=True)

    t0 = time.time()
    first = run_episodes(pol, episodes, noise)
    print(f'pass 1 (correctness + warm-up/capture) {time.time() - t0:.1f}s', flush=True)
    torch.save({'episodes': [e for e, _ in episodes], 'chunks': torch.stack(tap.chunks)}, out / f'chunks_{tag}.pt')
    tap.on = False

    lat = []
    while len(lat) < a.min_calls:
        noise.i = 0
        lat += run_episodes(pol, episodes, noise)
    s = stats(lat)
    s.update(mode=tag, gpu=torch.cuda.get_device_name(), torch=torch.__version__, frames=n_frames,
             first_pass=stats(first), **extra)
    print(json.dumps(s), flush=True)
    (out / f'latency_{tag}.json').write_text(json.dumps({**s, 'lat_ms': lat}) + '\n')


if __name__ == '__main__':
    main()
