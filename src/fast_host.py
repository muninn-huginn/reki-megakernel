"""Host-path fixes for ``RekiPolicy.predict``; they apply to the compiled and the megakernel path alike.

Everything the host does before the prefill graph is launched is on the critical path: the GPU is
idle until then. In Kyutai's predict (reki b2a3393, H100, warm predicts) that window holds:

- ``RekiModel.predict_action_chunk`` calls ``self.eval()`` on every predict: a walk over all 931
  modules, ~1.5 ms. The model is already in eval mode; ``install`` keeps it there and turns the
  per-call ``eval()`` into a no-op.
- ``FastQwenImageProcessor`` normalizes and patchifies the camera images on the CPU in fp32, and
  ``build_inference_sample`` then copies 1.5 MB of pageable fp32 patches to the GPU (~0.65 ms + the
  copy). ``GpuImageProcessor`` uploads the uint8 images (2 x 192 KB, pinned) and does the same fp32
  ops on the GPU. It takes only images that are already at their resize target (no resize, as for
  LIBERO's 256x256 frames); anything else goes through Kyutai's CPU processor, whose
  bicubic+antialias resize it does not reproduce.
- ``RekiPolicy.predict`` wraps each camera array in a PIL image (and the processor converts it
  back), ~0.3 ms for two cameras. ``predict`` below is Kyutai's method with the arrays passed on.

The model sees the same inputs: ``check_pixels`` compares the GPU and CPU processors bitwise, and
``bench.py --fast-host`` checks it on every frame before timing.
"""
from __future__ import annotations

import types

import numpy as np
import reki.inference.policy as rpol
import torch
from PIL import Image
from reki.vlm.qwen3_5.image_processor import smart_resize


class GpuImageProcessor:
    """Drop-in for ``Qwen3_5Preprocessor._fast_img``: same outputs (``pixel_values`` on ``device``,
    ``image_grid_thw`` on the CPU), the fp32 normalization and patchify done on the GPU. Images are
    uint8 HxWx3 arrays or PIL RGB images."""

    def __init__(self, cpu, device: str | torch.device):
        self.cpu = cpu
        self.device = torch.device(device)
        self.mean = cpu._mean.to(self.device)
        self.std = cpu._std.to(self.device)
        self.pinned: torch.Tensor | None = None

    def __getattr__(self, name):  # Qwen3_5Preprocessor reads the CPU processor's merge_size
        return getattr(self.cpu, name)

    def __call__(self, images: list) -> dict[str, torch.Tensor]:
        cpu = self.cpu
        p, m, t = cpu.patch_size, cpu.merge_size, cpu.temporal_patch_size
        arrs = [np.asarray(img) if isinstance(img, (Image.Image, np.ndarray)) else None for img in images]
        h, w = (arrs[0].shape[:2]) if arrs[0] is not None else (0, 0)
        if any(a is None or a.shape != (h, w, 3) or a.dtype != np.uint8 for a in arrs) or \
                smart_resize(h, w, p * m, cpu.min_pixels, cpu.max_pixels) != (h, w):
            return cpu([Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images])
        n = len(arrs)
        if self.pinned is None or self.pinned.shape != (n, h, w, 3):
            self.pinned = torch.empty((n, h, w, 3), dtype=torch.uint8, pin_memory=True)
        # The previous predict's upload has completed: predict ends with a device->host copy.
        pinned_np = self.pinned.numpy()
        for i, a in enumerate(arrs):
            pinned_np[i] = a
        x = self.pinned.to(self.device, non_blocking=True)
        # The CPU processor's ops in the same order and dtype: uint8 -> fp32, * (1/255), - mean, / std.
        x = (x.permute(0, 3, 1, 2).to(torch.float32) * (1.0 / 255.0) - self.mean) / self.std
        C, gh, gw = 3, h // p, w // p
        x = x.unsqueeze(1).expand(n, t, C, h, w)
        x = x.reshape(n, t, C, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        pixel_values = x.reshape(n * gh * gw, C * t * p * p)
        image_grid_thw = torch.tensor([[1, gh, gw]] * n, dtype=torch.long)
        return {'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw}


def predict(self, images: dict[str, np.ndarray], state: np.ndarray, task: str,
            stats_key: str | None = None) -> np.ndarray:
    """``RekiPolicy.predict`` of reki b2a3393, except that the camera arrays reach the image processor
    as arrays instead of PIL images."""
    self._check_inputs(images, state)
    state_stats, action_stats = rpol.select_stats(self.stats, self.spec.per_task_stats, stats_key)
    obs = rpol.EnvObs(
        images=[images[k] for k in self.spec.camera_keys],
        state_raw=rpol.model_state(self.spec, state),
        task_text=task,
    )
    batch = rpol.build_inference_sample(
        obs,
        preprocessor=self.preprocessor,
        train_cfg=self.train_cfg,
        state_stats=state_stats,
        embodiment=self.spec.embodiment,
        control_mode=self.spec.control_mode,
        device=self.device,
        camera_indices=self.spec.camera_indices,
        robot_type=self.spec.robot_type,
        fps=self.spec.prompt_fps,
        action_dim=self.spec.action_dim,
        dataset=self.spec.dataset,
    )
    with torch.no_grad():
        chunk = self.policy.predict_action_chunk(batch)
    chunk = chunk[0, : self.spec.n_action_steps, : self.spec.action_dim].to(torch.float32).cpu()
    rot = self.spec.euler_rot_dims
    return rpol.decode_action(
        chunk,
        action_stats,
        dataset=self.spec.dataset,
        euler_rot_dims=(rot[0], rot[1]) if rot is not None else None,
    ).astype(np.float32)


def install(pol) -> GpuImageProcessor:
    """Apply the three fixes to a ``RekiPolicy``."""
    rk = pol.policy
    rk.eval()
    rk.eval = lambda: rk  # instance attribute: predict_action_chunk's self.eval() becomes a no-op
    pre = pol.preprocessor
    gpu = GpuImageProcessor(pre._fast_img, pol.device)
    pre._fast_img = gpu
    pol.predict = types.MethodType(predict, pol)
    return gpu


def check_pixels(gpu: GpuImageProcessor, frames: list[dict], cams: list[str]) -> dict:
    """Bitwise comparison of the GPU processor (arrays, as ``predict`` passes them) with Kyutai's CPU
    processor (PIL images, as their predict passes them) on every frame."""
    n_equal, max_diff = 0, 0.0
    for fr in frames:
        a = gpu([fr['images'][c] for c in cams])['pixel_values']
        b = gpu.cpu([Image.fromarray(fr['images'][c]) for c in cams])['pixel_values'].to(a.device)
        same = a.shape == b.shape and torch.equal(a, b)
        n_equal += int(same)
        if not same:
            max_diff = max(max_diff, float((a.float() - b.float()).abs().max()))
    return {'frames': len(frames), 'bitwise_equal': n_equal, 'max_abs_diff': max_diff}
