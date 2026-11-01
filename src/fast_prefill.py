"""torch.compile the VLM prefill's vision blocks and LM layers, inside a prefill CUDA graph.

Kyutai's ``PrefillGraphs`` captures the eager prefill kernels (2,366 per predict): unfused norms,
rotary, gating and residual ops around the GEMMs, FLA's Gated-DeltaNet kernels and the attention
kernels. ``install`` compiles each vision block and each LM decoder layer with Inductor (no Inductor
CUDA graphs: the modules run inside the prefill graph capture, as Kyutai's compiled expert runs
inside theirs) and swaps in ``CompiledPrefillGraphs``, which differs from ``PrefillGraphs`` in two
places:

- The capture check compares the replay with an uncaptured run of the same ``_forward`` (Kyutai's
  compares with the eager prefill, which calls the layers through HF's text model with other
  arguments and so runs other compiled graphs).
- The first predict of a layout returns the replay too, so every predict runs the same kernels.

Numerics: Inductor keeps fused elementwise chains in fp32 where eager rounds every op to bf16, as for
Kyutai's compiled expert; the prefill output is not bitwise equal to the eager prefill. ``bench.py
--compile-prefill`` + ``compare.py`` report the action-chunk |Δ| against the other paths.
"""
from __future__ import annotations

from typing import Any

import torch
from reki.vlm.qwen3_5.cuda_graph import capture_checked
from reki.vlm.qwen3_5.hf_patches import fla_inference_configs
from reki.vlm.qwen3_5.inference import CrossAttnCache
from reki.vlm.qwen3_5.prefill_graph import PrefillGraphs, _LayoutGraph


class _CompiledLayoutGraph(_LayoutGraph):
    def capture(self, reference: torch.Tensor) -> None:
        del reference  # the eager prefill's output: other kernels, see the module docstring
        with fla_inference_configs():
            # The first call after a (re)compile is not yet the steady state (|Δ| up to 0.3 on the
            # output, measured); the third run is the reference the replay must equal bitwise.
            for _ in range(2):
                self._forward()
            ref = self._forward().clone()
            self.graph, self.out = capture_checked(self._forward, (ref,), 'prefill')


class CompiledPrefillGraphs(PrefillGraphs):
    def __call__(self, model: Any, vision_inputs: list[dict[str, Any]], tokens: torch.Tensor) -> CrossAttnCache:
        assert len(vision_inputs) == 1 and tokens.shape[0] == 1
        vision = vision_inputs[0]
        key = tuple(
            t.cpu().numpy().tobytes()
            for t in (tokens, vision['image_grid_thw'], vision['mm_token_type_ids'])
        ) + (str(tuple(vision['pixel_values'].shape)).encode(),)
        layout = self.get(key)
        if layout is None:
            self.make_room()
            past = model._prefill(vision_inputs, tokens)  # the layout-only inputs (positions, expert start)
            layout = _CompiledLayoutGraph(model, vision, tokens, past)
            layout.capture(past.vlm_out)
            self.put(key, layout)
        return layout.replay(vision['pixel_values'])


def install(model) -> None:
    vwe = model.vlm_with_expert
    for block in vwe.image_encoder.blocks:
        block.compile(dynamic=False)  # 2 images x 256 patches: one shape
    text = vwe.text_backbone
    for layer in text.layers[: text.config.num_hidden_layers]:
        layer.compile(dynamic=True)  # prompt lengths differ between tasks
    old = model._prefill_graphs
    assert isinstance(old, PrefillGraphs), 'compile the policy (RekiPolicy.from_pretrained(compile=True)) first'
    model._prefill_graphs = CompiledPrefillGraphs(old.max_entries)
