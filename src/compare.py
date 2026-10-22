"""Correctness across paths: max/percentile |Δ| of the normalized action chunks saved by bench.py
(same LIBERO frames, same fixed noise, register memory carried per episode).

    python compare.py results/   ->  results/correctness.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

ACTION_DIM = 7  # LIBERO: the real dims of the 32-wide padded chunk


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
    chunks = {p.stem.removeprefix('chunks_'): torch.load(p)['chunks'][..., :ACTION_DIM] for p in sorted(d.glob('chunks_*.pt'))}
    out = {'frames': {k: int(v.shape[0]) for k, v in chunks.items()}}
    for a, b in combinations(sorted(chunks), 2):
        diff = (chunks[a] - chunks[b]).abs()
        per_frame = diff.flatten(1).amax(1)
        out[f'{a}_vs_{b}'] = {
            'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'p99_frame_max_abs': float(per_frame.quantile(0.99)), 'median_frame_max_abs': float(per_frame.median()),
            'ref_abs_max': float(chunks[b].abs().max()),
        }
    print(json.dumps(out, indent=1))
    (d / 'correctness.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
