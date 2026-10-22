"""A persistent Triton megakernel for REKI's action expert: Euler steps 2..N in ONE launch (v1, superseded
by mk_expert.py v4; 5.89 ms with these defaults, 4.6-4.8 ms with BK=256 -- see README.md).

Replaces the Python loop of ``RekiFlowModel._expert_actions`` over ``_expert_step`` (one compiled
graph per step, about 30 Inductor/cuBLAS/flash kernels per expert layer) with a single kernel of
``n_SM`` co-resident programs that walk the whole loop: per step 10 expert layers x 8 stages
(QKV, SA attention, SA out-proj + gated residual, CA query, CA attention, CA out-proj + gated
residual, gate/up + SiLU, down + gated residual) and the step tail (final AdaRMS, fp32 action
out-projection, Euler update, zero padding, fp32 action in-projection). The stages are separated
by a software grid barrier (an atomic counter in global memory); activations live in small global
scratch buffers (L2-resident) and are read with ``.cg`` loads (L1 bypass, since other SMs wrote them).

Everything step 1 produces is kept from the reference path: modulations of all steps,
``register_prepare`` (CA source K/V, cos/sin, cu_seqlens) and Euler step 1 with the register rows
(which also yields the rotated memory K/V). The megakernel reads them in place.

Numerics follow the compiled (Inductor) reference: bf16 GEMM inputs with fp32 accumulation and a
bf16 rounding of each Linear output; norms, RoPE, SiLU-mul and gated residuals in fp32 with one
bf16 rounding per fused op; fp32 (IEEE, no TF32) action projections and Euler update.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l', [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL: tl.constexpr, PROFILE: tl.constexpr):
    """Barrier ``nb`` (1-based): all programs arrive, then wait until the counter reaches nb * nprog
    (monotonic counter, so no reset race). Scalar atomics are executed once per program.
    POLL=0 spins with acquire atomics, POLL=1 with volatile loads then one acquire.
    PROFILE: program 0 stores %globaltimer after the barrier at prof_ptr[nb]."""
    target = nb * nprog
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem='release', scope='gpu')
    if POLL == 0:
        v = tl.atomic_add(bar_ptr, 0, sem='acquire', scope='gpu')
        while v < target:
            v = tl.atomic_add(bar_ptr, 0, sem='acquire', scope='gpu')
    else:
        v = tl.load(bar_ptr, volatile=True)
        while v < target:
            v = tl.load(bar_ptr, volatile=True)
        tl.atomic_add(bar_ptr, 0, sem='acquire', scope='gpu')
    tl.debug_barrier()
    if PROFILE:
        if pid == 0:
            tl.store(prof_ptr + nb, _globaltimer())


@triton.jit
def _row_rstd(h_ptr, T, eps, TP: tl.constexpr, H: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, TP)
    acc = tl.zeros([TP], tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(h_ptr + rows[:, None] * H + kk[None, :], mask=rows[:, None] < T, other=0.,
                    cache_modifier='.cg').to(tl.float32)
        acc += tl.sum(x * x, axis=1)
    return tl.math.rsqrt(acc / H + eps)


@triton.jit
def _mm_modnorm(h_ptr, rstd, mod_ptr, w_ptr, n0, T, TP: tl.constexpr, H: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """[TP, BN] fp32 = AdaRMS(h; scale, shift at mod_ptr) @ W[n0:n0+BN, :H]^T (W row-major [N, H])."""
    rows = tl.arange(0, TP)
    nn = n0 + tl.arange(0, BN)
    acc = tl.zeros([TP, BN], tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(h_ptr + rows[:, None] * H + kk[None, :], mask=rows[:, None] < T, other=0.,
                    cache_modifier='.cg').to(tl.float32)
        sc = tl.load(mod_ptr + kk).to(tl.float32)
        sh = tl.load(mod_ptr + H + kk).to(tl.float32)
        a = (x * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]).to(tl.bfloat16)
        w = tl.load(w_ptr + nn[:, None] * H + kk[None, :])
        acc = tl.dot(a, tl.trans(w), acc)
    return acc


@triton.jit
def _mm_plain(a_ptr, w_ptr, n0, T, K: tl.constexpr, TP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """[TP, BN] fp32 = A[:, :K] @ W[n0:n0+BN, :K]^T, A bf16 scratch [TP, K]."""
    rows = tl.arange(0, TP)
    nn = n0 + tl.arange(0, BN)
    acc = tl.zeros([TP, BN], tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=rows[:, None] < T, other=0.,
                    cache_modifier='.cg')
        w = tl.load(w_ptr + nn[:, None] * K + kk[None, :])
        acc = tl.dot(a, tl.trans(w), acc)
    return acc


@triton.jit
def _gated_residual_store(acc, res_ptr, gate_ptr, out_ptr, n0, T, TP: tl.constexpr, H: tl.constexpr,
                          BN: tl.constexpr):
    """out = res + bf16(acc) * gate (fp32 math, one bf16 rounding), columns n0..n0+BN."""
    rows = tl.arange(0, TP)
    nn = n0 + tl.arange(0, BN)
    m = rows[:, None] < T
    o = acc.to(tl.bfloat16).to(tl.float32)
    g = tl.load(gate_ptr + nn).to(tl.float32)
    r = tl.load(res_ptr + rows[:, None] * H + nn[None, :], mask=m, other=0., cache_modifier='.cg').to(tl.float32)
    tl.store(out_ptr + rows[:, None] * H + nn[None, :], (r + o * g[None, :]).to(tl.bfloat16), mask=m)


@triton.jit
def _norm_rope(src_ptr, stride, col0, nw_ptr, cos_ptr, sin_ptr, T, eps, TP: tl.constexpr, HD: tl.constexpr):
    """Per-head RMSNorm (weight 1 + w) then RoPE (rotate_half over HD) of rows [T, HD] at
    src_ptr + row * stride + col0. Returns bf16 [TP, HD]."""
    rows = tl.arange(0, TP)
    d = tl.arange(0, HD)
    dp = (d + HD // 2) % HD
    m = rows[:, None] < T
    x = tl.load(src_ptr + rows[:, None] * stride + col0 + d[None, :], mask=m, other=0., cache_modifier='.cg').to(tl.float32)
    xp = tl.load(src_ptr + rows[:, None] * stride + col0 + dp[None, :], mask=m, other=0., cache_modifier='.cg').to(tl.float32)
    w = 1.0 + tl.load(nw_ptr + d).to(tl.float32)
    wp = 1.0 + tl.load(nw_ptr + dp).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, axis=1) / HD + eps)
    xn = (x * r[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
    xpn = (xp * r[:, None] * wp[None, :]).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(d < HD // 2, -1.0, 1.0)
    c = tl.load(cos_ptr + rows[:, None] * HD + d[None, :], mask=m, other=0.).to(tl.float32)
    s = tl.load(sin_ptr + rows[:, None] * HD + d[None, :], mask=m, other=0.).to(tl.float32)
    return (xn * c + sign[None, :] * xpn * s).to(tl.bfloat16)


@triton.jit
def _sa_attention(head, l, qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, memk_ptr, memv_ptr, kv_start,
                  y_ptr, T, eps, sm_scale, TP: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
                  PM: tl.constexpr, PMP: tl.constexpr, H: tl.constexpr):
    """Action queries of one head over [memory registers (rows kv_start..PM) || own actions]."""
    rows = tl.arange(0, TP)
    d = tl.arange(0, HD)
    q = _norm_rope(qkv_ptr, 3 * H, head * HD, qn_ptr + l * HD, cos_ptr, sin_ptr, T, eps, TP, HD)
    k = _norm_rope(qkv_ptr, 3 * H, H + head * HD, kn_ptr + l * HD, cos_ptr, sin_ptr, T, eps, TP, HD)
    v = tl.load(qkv_ptr + rows[:, None] * (3 * H) + 2 * H + head * HD + d[None, :], mask=rows[:, None] < T,
                other=0., cache_modifier='.cg')
    j = tl.arange(0, PMP)
    jm = (j >= kv_start) & (j < PM)
    moff = ((l * PM + j[:, None]) * NH + head) * HD + d[None, :]
    mk = tl.load(memk_ptr + moff, mask=jm[:, None], other=0.)
    mv = tl.load(memv_ptr + moff, mask=jm[:, None], other=0.)
    s1 = tl.dot(q, tl.trans(mk)) * sm_scale
    s1 = tl.where(jm[None, :], s1, float('-inf'))
    s2 = tl.dot(q, tl.trans(k)) * sm_scale
    s2 = tl.where(rows[None, :] < T, s2, float('-inf'))
    mx = tl.maximum(tl.max(s1, axis=1), tl.max(s2, axis=1))
    p1 = tl.exp(s1 - mx[:, None])
    p2 = tl.exp(s2 - mx[:, None])
    den = tl.sum(p1, axis=1) + tl.sum(p2, axis=1)
    o = tl.dot(p1.to(tl.bfloat16), mv) + tl.dot(p2.to(tl.bfloat16), v)
    o = o / den[:, None]
    tl.store(y_ptr + rows[:, None] * H + head * HD + d[None, :], o.to(tl.bfloat16), mask=rows[:, None] < T)


@triton.jit
def _ca_attention(head, l, q_ptr, qn_ptr, cos_ptr, sin_ptr, kca_ptr, vca_ptr, S, y_ptr, T, eps, sm_scale,
                  TP: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr, SP: tl.constexpr, H: tl.constexpr,
                  BS: tl.constexpr):
    """Action queries of one head over the S valid rows of the cached VLM source K/V (online softmax)."""
    rows = tl.arange(0, TP)
    d = tl.arange(0, HD)
    q = _norm_rope(q_ptr, H, head * HD, qn_ptr + l * HD, cos_ptr, sin_ptr, T, eps, TP, HD)
    mx = tl.full([TP], float('-inf'), tl.float32)
    den = tl.zeros([TP], tl.float32)
    acc = tl.zeros([TP, HD], tl.float32)
    for s0 in range(0, S, BS):
        j = s0 + tl.arange(0, BS)
        jm = j < S
        off = ((l * SP + j[:, None]) * NH + head) * HD + d[None, :]
        k = tl.load(kca_ptr + off, mask=jm[:, None], other=0.)
        v = tl.load(vca_ptr + off, mask=jm[:, None], other=0.)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(jm[None, :], s, float('-inf'))
        mn = tl.maximum(mx, tl.max(s, axis=1))
        alpha = tl.exp(mx - mn)
        p = tl.exp(s - mn[:, None])
        den = den * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        mx = mn
    o = acc / den[:, None]
    tl.store(y_ptr + rows[:, None] * H + head * HD + d[None, :], o.to(tl.bfloat16), mask=rows[:, None] < T)


@triton.jit
def _step_tail(first: tl.constexpr, x_ptr, h_ptr, modf_ptr, wout_ptr, bout_ptr, win_ptr, bin_ptr, dt, action_dim,
               T, eps,
               TP: tl.constexpr, H: tl.constexpr, A: tl.constexpr, BK: tl.constexpr):
    """One program. Unless ``first``: final AdaRMS of h, v = out @ Wout^T + b (fp32), x <- x + dt v.
    Then (always): x_pad = x with dims >= action_dim zeroed, h <- bf16(x_pad @ Win^T + b) for the
    next step. ``first`` only zero-pads x and builds h."""
    rows = tl.arange(0, TP)
    a = tl.arange(0, A)
    xm = rows[:, None] < T
    x = tl.load(x_ptr + rows[:, None] * A + a[None, :], mask=xm, other=0., cache_modifier='.cg')
    xpad = tl.where(a[None, :] < action_dim, x, 0.)
    if not first:
        rstd = _row_rstd(h_ptr, T, eps, TP, H, BK)
        vel = tl.zeros([TP, A], tl.float32)
        for k0 in range(0, H, BK):
            kk = k0 + tl.arange(0, BK)
            hx = tl.load(h_ptr + rows[:, None] * H + kk[None, :], mask=xm, other=0., cache_modifier='.cg').to(tl.float32)
            sc = tl.load(modf_ptr + kk).to(tl.float32)
            sh = tl.load(modf_ptr + H + kk).to(tl.float32)
            o = (hx * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]).to(tl.bfloat16).to(tl.float32)
            w = tl.load(wout_ptr + a[None, :] * H + kk[:, None])  # [BK, A] = Wout^T chunk
            vel = tl.dot(o, w, vel, input_precision='ieee')
        vel += tl.load(bout_ptr + a)[None, :]
        # The previous step's x_pad is what the reference integrates from.
        xpad = xpad + dt * vel
        tl.store(x_ptr + rows[:, None] * A + a[None, :], xpad, mask=xm)
        xpad = tl.where(a[None, :] < action_dim, xpad, 0.)
    for n0 in range(0, H, BK):
        nn = n0 + tl.arange(0, BK)
        w = tl.load(win_ptr + nn[None, :] * A + a[:, None])  # [A, BK] = Win^T chunk
        hn = tl.dot(xpad, w, input_precision='ieee') + tl.load(bin_ptr + nn)[None, :]
        tl.store(h_ptr + rows[:, None] * H + nn[None, :], hn.to(tl.bfloat16), mask=xm)


@triton.jit
def expert_megakernel(
        x_ptr, mods_ptr, dt_ptr,
        win_ptr, bin_ptr, wout_ptr, bout_ptr,
        wqkv_ptr, wo_ptr, wqca_ptr, woca_ptr, wg_ptr, wu_ptr, wd_ptr,
        qn_ptr, kn_ptr, qcan_ptr,
        memk_ptr, memv_ptr, cukv_ptr,
        kca_ptr, vca_ptr, cusrc_ptr,
        sacos_ptr, sasin_ptr, cacos_ptr, casin_ptr,
        h_ptr, h2_ptr, h3_ptr, qkv_ptr, y_ptr, qca_ptr, g_ptr,
        bar_ptr, prof_ptr, n_steps, action_dim, T, eps, qk_eps, sm_scale,
        L: tl.constexpr, TP: tl.constexpr, H: tl.constexpr, I: tl.constexpr, A: tl.constexpr,
        NH: tl.constexpr, HD: tl.constexpr, PM: tl.constexpr, PMP: tl.constexpr, SP: tl.constexpr,
        MODW: tl.constexpr, BK: tl.constexpr, BN_QKV: tl.constexpr, BN_O: tl.constexpr,
        BN_GU: tl.constexpr, BN_D: tl.constexpr, BS: tl.constexpr, POLL: tl.constexpr, PROFILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    dt = tl.load(dt_ptr)
    kv_start = tl.load(cukv_ptr)
    S = tl.load(cusrc_ptr + 1)
    nb = 0
    if pid == 0:
        _step_tail(True, x_ptr, h_ptr, mods_ptr, wout_ptr, bout_ptr, win_ptr, bin_ptr, dt, action_dim, T, eps,
                   TP, H, A, BK)
    nb += 1
    _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
    for s in range(n_steps):
        mrow = mods_ptr + (1 + s) * MODW
        for l in range(L):
            m_sa = mrow + (3 * l) * 3 * H
            m_ca = m_sa + 3 * H
            m_mlp = m_ca + 3 * H
            # 1. QKV of AdaRMS(h)
            rstd = _row_rstd(h_ptr, T, eps, TP, H, BK)
            for t in range(pid, 3 * H // BN_QKV, nprog):
                acc = _mm_modnorm(h_ptr, rstd, m_sa, wqkv_ptr + l * 3 * H * H, t * BN_QKV, T, TP, H, BN_QKV, BK)
                nn = t * BN_QKV + tl.arange(0, BN_QKV)
                rows = tl.arange(0, TP)
                tl.store(qkv_ptr + rows[:, None] * (3 * H) + nn[None, :], acc.to(tl.bfloat16), mask=rows[:, None] < T)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 2. self-attention per head
            if pid < NH:
                _sa_attention(pid, l, qkv_ptr, qn_ptr, kn_ptr, sacos_ptr, sasin_ptr, memk_ptr, memv_ptr, kv_start,
                              y_ptr, T, qk_eps, sm_scale, TP, HD, NH, PM, PMP, H)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 3. o_proj + gated residual -> h2
            for t in range(pid, H // BN_O, nprog):
                acc = _mm_plain(y_ptr, wo_ptr + l * H * H, t * BN_O, T, H, TP, BN_O, BK)
                _gated_residual_store(acc, h_ptr, m_sa + 2 * H, h2_ptr, t * BN_O, T, TP, H, BN_O)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 4. CA query of AdaRMS(h2)
            rstd = _row_rstd(h2_ptr, T, eps, TP, H, BK)
            for t in range(pid, H // BN_O, nprog):
                acc = _mm_modnorm(h2_ptr, rstd, m_ca, wqca_ptr + l * H * H, t * BN_O, T, TP, H, BN_O, BK)
                nn = t * BN_O + tl.arange(0, BN_O)
                rows = tl.arange(0, TP)
                tl.store(qca_ptr + rows[:, None] * H + nn[None, :], acc.to(tl.bfloat16), mask=rows[:, None] < T)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 5. cross-attention per head
            if pid < NH:
                _ca_attention(pid, l, qca_ptr, qcan_ptr, cacos_ptr, casin_ptr, kca_ptr, vca_ptr, S, y_ptr, T,
                              qk_eps, sm_scale, TP, HD, NH, SP, H, BS)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 6. CA o_proj + gated residual -> h3
            for t in range(pid, H // BN_O, nprog):
                acc = _mm_plain(y_ptr, woca_ptr + l * H * H, t * BN_O, T, H, TP, BN_O, BK)
                _gated_residual_store(acc, h2_ptr, m_ca + 2 * H, h3_ptr, t * BN_O, T, TP, H, BN_O)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 7. gate/up of AdaRMS(h3), g = silu(gate) * up
            rstd = _row_rstd(h3_ptr, T, eps, TP, H, BK)
            for t in range(pid, I // BN_GU, nprog):
                ga = _mm_modnorm(h3_ptr, rstd, m_mlp, wg_ptr + l * I * H, t * BN_GU, T, TP, H, BN_GU, BK)
                ua = _mm_modnorm(h3_ptr, rstd, m_mlp, wu_ptr + l * I * H, t * BN_GU, T, TP, H, BN_GU, BK)
                gf = ga.to(tl.bfloat16).to(tl.float32)
                uf = ua.to(tl.bfloat16).to(tl.float32)
                gv = gf / (1.0 + tl.exp(-gf)) * uf
                nn = t * BN_GU + tl.arange(0, BN_GU)
                rows = tl.arange(0, TP)
                tl.store(g_ptr + rows[:, None] * I + nn[None, :], gv.to(tl.bfloat16), mask=rows[:, None] < T)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 8. down + gated residual -> h
            for t in range(pid, H // BN_D, nprog):
                acc = _mm_plain(g_ptr, wd_ptr + l * H * I, t * BN_D, T, I, TP, BN_D, BK)
                _gated_residual_store(acc, h3_ptr, m_mlp + 2 * H, h_ptr, t * BN_D, T, TP, H, BN_D)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
        # step tail: final norm, out-proj, Euler update; in-proj of the next step's input
        if pid == 0:
            _step_tail(False, x_ptr, h_ptr, mrow + 3 * L * 3 * H, wout_ptr, bout_ptr, win_ptr, bin_ptr, dt,
                       action_dim, T, eps, TP, H, A, BK)
        nb += 1
        _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class ExpertMegakernel:
    """Packs the expert weights once; ``run`` integrates Euler steps 2..N in one launch."""

    def __init__(self, model):
        vwe = model.vlm_with_expert
        layers = list(vwe.action_expert.layers)
        sa0 = layers[0].self_attn
        self.L, self.NH, self.HD = len(layers), sa0.n_q_heads, sa0.head_dim
        self.H = self.NH * self.HD
        self.I = layers[0].mlp.gate_proj.out_features
        self.A = model.action_in_proj.in_features
        assert sa0.n_kv_heads == self.NH and layers[0].cross_attn.n_kv_heads == self.NH
        st = lambda f: torch.stack([f(lay).detach() for lay in layers]).contiguous()  # noqa: E731
        self.wqkv = st(lambda m: torch.cat([m.self_attn.q_proj.weight, m.self_attn.k_proj.weight,
                                            m.self_attn.v_proj.weight], 0))
        self.wo = st(lambda m: m.self_attn.o_proj.weight)
        self.wqca = st(lambda m: m.cross_attn.q_proj.weight)
        self.woca = st(lambda m: m.cross_attn.o_proj.weight)
        self.wg = st(lambda m: m.mlp.gate_proj.weight)
        self.wu = st(lambda m: m.mlp.up_proj.weight)
        self.wd = st(lambda m: m.mlp.down_proj.weight)
        self.qn = st(lambda m: m.self_attn.q_norm.weight)
        self.kn = st(lambda m: m.self_attn.k_norm.weight)
        self.qcan = st(lambda m: m.cross_attn.q_norm.weight)
        self.win = model.action_in_proj.weight.detach().float().contiguous()
        self.bin = model.action_in_proj.bias.detach().float().contiguous()
        self.wout = model.action_out_proj.weight.detach().float().contiguous()
        self.bout = model.action_out_proj.bias.detach().float().contiguous()
        self.eps = float(layers[0].input_layernorm.eps)
        assert float(vwe.action_expert.norm.eps) == self.eps
        self.qk_eps = float(sa0.q_norm.eps)
        assert getattr(layers[0].mlp.act_fn, '__class__').__name__ in ('SiLU', 'SiLUActivation')
        dev, bf = self.wqkv.device, self.wqkv.dtype
        self.TP = 16
        z = lambda n: torch.zeros(self.TP, n, device=dev, dtype=bf)  # noqa: E731
        self.h, self.h2, self.h3 = z(self.H), z(self.H), z(self.H)
        self.qkv, self.y, self.qca, self.g = z(3 * self.H), z(self.H), z(self.H), z(self.I)
        self.bar = torch.zeros(1, device=dev, dtype=torch.int32)
        self.nprog = torch.cuda.get_device_properties(dev).multi_processor_count
        self.launches = 0
        self.prof = torch.zeros(4096, device=dev, dtype=torch.int64)
        # Tile sizes / launch options; POLL: barrier spin flavour; PROFILE: per-barrier timestamps.
        self.cfg = dict(BK=128, BN_QKV=32, BN_O=16, BN_GU=32, BN_D=16, BS=64, POLL=0, PROFILE=False,
                        num_warps=4, num_stages=3)

    def run(self, x_t: torch.Tensor, mods: torch.Tensor, dt: torch.Tensor, past, action_dim: int) -> torch.Tensor:
        """x_t: [1, T, A] fp32 after Euler step 1; mods: [N, MODW] (row 0 = step 1, used already);
        returns x_t after step N (padded dims as the reference: 0 + dt * v)."""
        step = past.register_step
        T = x_t.shape[1]
        assert x_t.shape[0] == 1 and T <= self.TP
        memk = torch.stack([m[0] for m in step['mem_kv']])
        memv = torch.stack([m[1] for m in step['mem_kv']])
        kca = torch.stack([kv[0] for kv in past.cross_attn_kv])
        vca = torch.stack([kv[1] for kv in past.cross_attn_kv])
        PM, SP = memk.shape[1], kca.shape[1]
        sa_cos, sa_sin = (t.contiguous() for t in step['sa_pe'])
        ca_cos, ca_sin = (t.contiguous() for t in step['ca_pe'])
        assert sa_cos.shape == (T, self.HD) and ca_cos.shape == (T, self.HD), (sa_cos.shape, ca_cos.shape)
        x = x_t.reshape(T, self.A).float().clone()
        mods = mods.contiguous()
        n_steps = mods.shape[0] - 1
        self.bar.zero_()
        self.compiled = expert_megakernel[(self.nprog,)](
            x, mods, dt,
            self.win, self.bin, self.wout, self.bout,
            self.wqkv, self.wo, self.wqca, self.woca, self.wg, self.wu, self.wd,
            self.qn, self.kn, self.qcan,
            memk, memv, step['cu_kv'],
            kca, vca, step['cu_src'],
            sa_cos, sa_sin, ca_cos, ca_sin,
            self.h, self.h2, self.h3, self.qkv, self.y, self.qca, self.g,
            self.bar, self.prof, n_steps, action_dim, T, self.eps, self.qk_eps, self.HD ** -0.5,
            L=self.L, TP=self.TP, H=self.H, I=self.I, A=self.A, NH=self.NH, HD=self.HD, PM=PM, PMP=_pow2(PM),
            SP=SP, MODW=mods.shape[1], **self.cfg)
        self.launches += 1
        return x.reshape(1, T, self.A)

    def barriers(self, n_steps: int) -> int:
        return 1 + n_steps * (8 * self.L + 1)


def install(model) -> ExpertMegakernel:
    """Route ``RekiFlowModel._expert_actions`` (what ``ExpertGraphs`` captures) through the megakernel
    for steps 2..N; modulations, register_prepare and step 1 stay the compiled reference callables."""
    mk = ExpertMegakernel(model)
    model._mk = mk

    def _expert_actions(noise, cross_attn_cache, register_cache, state_cond, action_dim, num_steps):
        assert noise.shape[0] == 1
        mods, dt = model._step_modulations(num_steps, state_cond)
        x_t, cur_k, cur_v = model._expert_first_step(noise, cross_attn_cache, register_cache, mods[:1], dt, action_dim)
        if num_steps > 1:
            x_t = mk.run(x_t, mods, dt, cross_attn_cache, action_dim)
        return x_t, cur_k, cur_v

    model._expert_actions = _expert_actions
    return mk
