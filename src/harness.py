"""REKI inference harness: real LIBERO frames, fixed flow noise, per-call latency, chunk capture.

Shared by every script in src/. Uses the public API of Kyutai's ``reki-vla`` package (torch 2.10,
transformers 5.3, fla 0.4.2 as pinned in their pixi.lock). Inputs default to ``data/`` at the repo
root (``scripts/setup.sh`` downloads them); override with REKI_CKPT / REKI_DATA.
"""
from __future__ import annotations

import glob
import io
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CKPT = os.environ.get('REKI_CKPT', str(ROOT / 'data' / 'reki-2b-libero'))
DATA = os.environ.get('REKI_DATA', str(ROOT / 'data' / 'libero'))
CAMS = {'observation.images.image': 'camera_top', 'observation.images.image2': 'camera_wrist_left'}
# The cloud container has a CFS quota of ~22 CPUs while torch sees 104 cores: with the default
# thread count, CPU ops (image preprocessing) get throttled for tens of ms every other call. 8 threads
# keep the host path inside the quota (set REKI_THREADS to change).
torch.set_num_threads(int(os.environ.get('REKI_THREADS', '8')))


def load_episodes(root: str, stride: int = 10) -> list[tuple[str, list[dict]]]:
    """LIBERO episodes of the HF dataset (HuggingFaceVLA/libero) as REKI sees them at evaluation:
    one frame per second (stride 10 at 10 fps, the model's history stride), uint8 HxWx3 RGB images in
    the training orientation, the raw 8-d observation.state and the task text."""
    tasks = pq.read_table(f'{root}/meta/tasks.parquet').to_pandas()
    task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks['task_index'])}
    out = []
    for f in sorted(glob.glob(f'{root}/data/chunk-000/*.parquet')):
        t = pq.read_table(f).to_pandas()
        for ep, g in t.groupby('episode_index'):
            g = g.sort_values('frame_index').iloc[::stride]
            frames = []
            for _, r in g.iterrows():
                frames.append({
                    'images': {cam: np.asarray(Image.open(io.BytesIO(r[col]['bytes'])).convert('RGB'))
                               for col, cam in CAMS.items()},
                    'state': np.asarray(r['observation.state'], dtype=np.float32),
                    'task': task_of[int(r['task_index'])],
                })
            out.append((f'{f.rsplit("/", 1)[-1]}:ep{ep}', frames))
    return out


class FixedNoise:
    """Replaces RekiFlowModel.sample_noise: the noise of call i is N(0,1) from a CUDA generator seeded
    with seed0 + i, so every path sees the same noise for the same frame."""

    def __init__(self, model, seed0: int = 1234):
        self.i = 0
        self.seed0 = seed0
        self.gen = torch.Generator(device='cuda')
        model.sample_noise = self

    def __call__(self, shape, device):
        self.gen.manual_seed(self.seed0 + self.i)
        self.i += 1
        return torch.randn(shape, generator=self.gen, dtype=torch.float32, device=device)


class ChunkTap:
    """Keeps the normalized chunk [chunk, action_dim] (fp32, CPU) of every predict."""

    def __init__(self, rk):
        self.chunks: list[torch.Tensor] = []
        self.on = True
        inner = rk.predict_action_chunk

        def wrapped(batch):
            c = inner(batch)
            if self.on:
                self.chunks.append(c[0].float().cpu())
            return c
        rk.predict_action_chunk = wrapped


def load_policy(ckpt: str, compile: bool):
    from reki import RekiPolicy
    pol = RekiPolicy.from_pretrained(ckpt, device='cuda', compile=compile)
    if compile:  # every LIBERO task keeps its prefill graph (default LRU: 2 layouts)
        pol.policy.model._prefill_graphs.max_entries = 64
    return pol


def run_episodes(pol, episodes, noise: FixedNoise, timed: bool = False):
    """Predict every frame of every episode in order (reset at each episode start). Returns per-call
    wall latencies in ms (host-synchronous: predict ends with the device->host copy of the chunk)."""
    lat = []
    for _, frames in episodes:
        pol.reset()
        for fr in frames:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pol.predict(images=fr['images'], state=fr['state'], task=fr['task'])
            lat.append((time.perf_counter() - t0) * 1e3)
    return lat


def stats(lat):
    a = np.asarray(lat)
    return {'n': int(a.size), 'median_ms': float(np.median(a)), 'p99_ms': float(np.percentile(a, 99)),
            'p10_ms': float(np.percentile(a, 10)), 'p90_ms': float(np.percentile(a, 90)), 'mean_ms': float(a.mean())}
