"""A persistent Triton megakernel for REKI's action expert: Euler steps 2..N in ONE launch (v2, superseded:
register spills from the cross-barrier weight prefetch, 63 ms; 13.4 ms with 4 warps -- see README.md).

Replaces the Python loop of ``RekiFlowModel._expert_actions`` over ``_expert_step`` (one compiled
graph per step, ~30 Inductor/cuBLAS/flash kernels per expert layer) with one kernel of ``n_SM``
co-resident programs that walk the whole loop: per step 10 expert layers x 8 stages (QKV, SA
attention, SA out-proj + gated residual, CA query, CA attention, CA out-proj + gated residual,
gate/up + SiLU, down + gated residual) and a step tail (final AdaRMS, fp32 action out-projection,
Euler update, zero padding, fp32 action in-projection). Stages are separated by a software grid
barrier (monotonic atomic counter); activations live in small L2-resident global scratch buffers,
read with ``.cg`` loads (L1 bypass: other SMs wrote them).

v2 vs v1 (``mk_expert_v1.py``): every program owns at most one tile per stage, and the weight tile
of the NEXT stage (plus the attention programs' memory K/V) is loaded into registers BEFORE the
barrier that ends the current stage, so the weight stream overlaps the barrier wait and the
previous stage (the weights never depend on activations). GEMMs are single-shot over K (two
power-of-two chunks: 768 = 512 + 256, 3072 = 2048 + 1024), the step tail is spread over the
programs, and only the real action dims (<= AO) go through the fp32 action projections
(the padded dims of the final x_t are computed once at the end).

Everything step 1 produces stays the reference: modulations of all steps, ``register_prepare``
(CA source K/V, cos/sin, cu_seqlens) and Euler step 1 with the register rows (which also yields
the rotated memory K/V). The megakernel reads them in place.

Numerics follow the compiled (Inductor) reference: bf16 GEMM inputs with fp32 accumulation and a
bf16 rounding of each Linear output; norms, RoPE, SiLU-mul and gated residuals in fp32 with one
bf16 rounding per fused op; fp32 action projections (exact fp32 products, summation order differs)
and Euler update.
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
def _rows(ptr, stride, k0, T, has, TP: tl.constexpr, KC: tl.constexpr):
    """Activation rows [TP, KC] (rows >= T and programs without a tile read 0), L1-bypassing."""
    r = tl.arange(0, TP)
    kk = k0 + tl.arange(0, KC)
    return tl.load(ptr + r[:, None] * stride + kk[None, :], mask=(r[:, None] < T) & has, other=0.,
                   cache_modifier='.cg')


@triton.jit
def _wt(w_ptr, n0, stride, k0, has, BN: tl.constexpr, KC: tl.constexpr):
    """Weight tile W[n0:n0+BN, k0:k0+KC] of a row-major [N, stride] matrix (0 without a tile)."""
    nn = n0 + tl.arange(0, BN)
    kk = k0 + tl.arange(0, KC)
    return tl.load(w_ptr + nn[:, None] * stride + kk[None, :], mask=(nn[:, None] >= 0) & has, other=0.)


@triton.jit
def _modnorm2(xa, xb, mod_ptr, eps, H: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr):
    """AdaRMS of rows split in two column chunks: bf16(x * rsqrt(mean x^2 + eps) * (1 + scale) + shift)."""
    xa = xa.to(tl.float32)
    xb = xb.to(tl.float32)
    r = tl.math.rsqrt((tl.sum(xa * xa, axis=1) + tl.sum(xb * xb, axis=1)) / H + eps)
    ka = tl.arange(0, KA)
    kb = KA + tl.arange(0, KB)
    aa = xa * r[:, None] * (1.0 + tl.load(mod_ptr + ka).to(tl.float32))[None, :] \
        + tl.load(mod_ptr + H + ka).to(tl.float32)[None, :]
    ab = xb * r[:, None] * (1.0 + tl.load(mod_ptr + kb).to(tl.float32))[None, :] \
        + tl.load(mod_ptr + H + kb).to(tl.float32)[None, :]
    return aa.to(tl.bfloat16), ab.to(tl.bfloat16)


@triton.jit
def _store_rows(ptr, stride, n0, val, T, has, TP: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, TP)
    nn = n0 + tl.arange(0, BN)
    tl.store(ptr + r[:, None] * stride + nn[None, :], val, mask=(r[:, None] < T) & has)


@triton.jit
def _gated_residual(acc, res_ptr, gate_ptr, n0, T, has, TP: tl.constexpr, H: tl.constexpr, BN: tl.constexpr):
    """bf16(res + bf16(acc) * gate) for columns n0..n0+BN (fp32 math, one rounding)."""
    nn = n0 + tl.arange(0, BN)
    r = _rows(res_ptr, H, n0, T, has, TP, BN).to(tl.float32)
    g = tl.load(gate_ptr + nn).to(tl.float32)
    return (r + acc.to(tl.bfloat16).to(tl.float32) * g[None, :]).to(tl.bfloat16)


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
def _velocity(h_ptr, modf_ptr, wout_ptr, bout_ptr, T, eps, has, a0, TP: tl.constexpr, H: tl.constexpr,
              A: tl.constexpr, AO: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, BKV: tl.constexpr):
    """v[:, a0:a0+AO] = AdaRMS_final(h) @ Wout[a0:a0+AO]^T + b, fp32 [TP, AO]."""
    r = tl.arange(0, TP)
    xa = _rows(h_ptr, H, 0, T, has, TP, KA).to(tl.float32)
    xb = _rows(h_ptr, H, KA, T, has, TP, KB).to(tl.float32)
    rstd = tl.math.rsqrt((tl.sum(xa * xa, axis=1) + tl.sum(xb * xb, axis=1)) / H + eps)
    ao = a0 + tl.arange(0, AO)
    vel = tl.zeros([TP, AO], tl.float32)
    for k0 in range(0, H, BKV):
        kk = k0 + tl.arange(0, BKV)
        hx = _rows(h_ptr, H, k0, T, has, TP, BKV).to(tl.float32)
        o = (hx * rstd[:, None] * (1.0 + tl.load(modf_ptr + kk).to(tl.float32))[None, :]
             + tl.load(modf_ptr + H + kk).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
        w = tl.load(wout_ptr + ao[:, None] * H + kk[None, :], mask=(ao[:, None] < A) & has, other=0.)
        vel += tl.sum(o[:, None, :] * w[None, :, :], axis=2)
    return vel + tl.load(bout_ptr + ao, mask=ao < A, other=0.)[None, :]


@triton.jit
def expert_megakernel(
        xbuf_ptr, mods_ptr, dt_ptr,
        win_ptr, bin_ptr, wout_ptr, bout_ptr,
        wqkv_ptr, wo_ptr, wqca_ptr, woca_ptr, wg_ptr, wu_ptr, wd_ptr,
        qn_ptr, kn_ptr, qcan_ptr,
        memk_ptr, memv_ptr, cukv_ptr,
        kca_ptr, vca_ptr, cusrc_ptr,
        sacos_ptr, sasin_ptr, cacos_ptr, casin_ptr,
        hh_ptr, h2_ptr, h3_ptr, qkv_ptr, y_ptr, qca_ptr, g_ptr,
        bar_ptr, prof_ptr, n_steps, action_dim, T, eps, qk_eps, sm_scale,
        L: tl.constexpr, TP: tl.constexpr, H: tl.constexpr, I: tl.constexpr, A: tl.constexpr, AO: tl.constexpr,
        NH: tl.constexpr, HD: tl.constexpr, PM: tl.constexpr, PMP: tl.constexpr, SP: tl.constexpr,
        MODW: tl.constexpr, BS: tl.constexpr, BKV: tl.constexpr, POLL: tl.constexpr, PROFILE: tl.constexpr):
    KA: tl.constexpr = H * 2 // 3          # 768 = 512 + 256
    KB: tl.constexpr = H - KA
    IA: tl.constexpr = I * 2 // 3          # 3072 = 2048 + 1024
    IB: tl.constexpr = I - IA
    BQ: tl.constexpr = 32                  # QKV tile: 3H / 32 = 72 tiles
    BO: tl.constexpr = 16                  # o / CA-q / CA-o / down / in-proj tile: H / 16 = 48 tiles
    BG: tl.constexpr = 32                  # gate+up tile: I / 32 = 96 tiles
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    dt = tl.load(dt_ptr)
    kv_start = tl.load(cukv_ptr)
    S = tl.load(cusrc_ptr + 1)
    has_q = pid < 3 * H // BQ
    has_o = pid < H // BO
    has_g = pid < I // BG
    is_att = pid < NH
    rows = tl.arange(0, TP)
    d = tl.arange(0, HD)
    h0_ptr = hh_ptr                        # step input (in-proj output); layers >= 1 read/write hh[1]
    hl_ptr = hh_ptr + TP * H
    nb = 0

    # Step-0 head: h0 = in-proj(zero-pad(x1)) (tile per program), prefetch the first QKV tile.
    xo = tl.load(xbuf_ptr + rows[:, None] * A + tl.arange(0, AO)[None, :], mask=rows[:, None] < T, other=0.,
                 cache_modifier='.cg')
    xo = tl.where(tl.arange(0, AO)[None, :] < action_dim, xo, 0.)
    nn = pid * BO + tl.arange(0, BO)
    wi = tl.load(win_ptr + nn[:, None] * A + tl.arange(0, AO)[None, :], mask=(nn[:, None] >= 0) & has_o, other=0.)
    hi = tl.sum(xo[:, None, :] * wi[None, :, :], axis=2) \
        + tl.load(bin_ptr + nn, mask=(nn >= 0) & has_o, other=0.)[None, :]
    _store_rows(h0_ptr, H, pid * BO, hi.to(tl.bfloat16), T, has_o, TP, BO)
    wq_a = _wt(wqkv_ptr, pid * BQ, H, 0, has_q, BQ, KA)
    wq_b = _wt(wqkv_ptr, pid * BQ, H, KA, has_q, BQ, KB)
    nb += 1
    _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
    for s in range(n_steps):
        mrow = mods_ptr + (1 + s) * MODW
        for l in range(L):
            m_sa = mrow + (3 * l) * 3 * H
            m_ca = m_sa + 3 * H
            m_mlp = m_ca + 3 * H
            hin_ptr = hh_ptr + tl.where(l == 0, 0, TP * H)
            # 1. QKV of AdaRMS(h)  [prefetch: mem K/V of this head]
            xa = _rows(hin_ptr, H, 0, T, has_q, TP, KA)
            xb = _rows(hin_ptr, H, KA, T, has_q, TP, KB)
            aa, ab = _modnorm2(xa, xb, m_sa, eps, H, KA, KB)
            acc = tl.dot(aa, tl.trans(wq_a))
            acc = tl.dot(ab, tl.trans(wq_b), acc)
            _store_rows(qkv_ptr, 3 * H, pid * BQ, acc.to(tl.bfloat16), T, has_q, TP, BQ)
            j = tl.arange(0, PMP)
            jm = (j >= kv_start) & (j < PM) & is_att
            moff = ((l * PM + j[:, None]) * NH + pid) * HD + d[None, :]
            mk = tl.load(memk_ptr + moff, mask=jm[:, None], other=0.)
            mv = tl.load(memv_ptr + moff, mask=jm[:, None], other=0.)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 2. self-attention per head over [memory || own actions]  [prefetch: o_proj tile]
            if is_att:
                q = _norm_rope(qkv_ptr, 3 * H, pid * HD, qn_ptr + l * HD, sacos_ptr, sasin_ptr, T, qk_eps, TP, HD)
                k = _norm_rope(qkv_ptr, 3 * H, H + pid * HD, kn_ptr + l * HD, sacos_ptr, sasin_ptr, T, qk_eps,
                               TP, HD)
                v = tl.load(qkv_ptr + rows[:, None] * (3 * H) + 2 * H + pid * HD + d[None, :],
                            mask=rows[:, None] < T, other=0., cache_modifier='.cg')
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
                tl.store(y_ptr + rows[:, None] * H + pid * HD + d[None, :], o.to(tl.bfloat16), mask=rows[:, None] < T)
            wo_a = _wt(wo_ptr + l * H * H, pid * BO, H, 0, has_o, BO, KA)
            wo_b = _wt(wo_ptr + l * H * H, pid * BO, H, KA, has_o, BO, KB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 3. o_proj + gated residual -> h2  [prefetch: CA-q tile]
            ya = _rows(y_ptr, H, 0, T, has_o, TP, KA)
            yb = _rows(y_ptr, H, KA, T, has_o, TP, KB)
            acc = tl.dot(ya, tl.trans(wo_a))
            acc = tl.dot(yb, tl.trans(wo_b), acc)
            _store_rows(h2_ptr, H, pid * BO, _gated_residual(acc, hin_ptr, m_sa + 2 * H, pid * BO, T, has_o, TP, H, BO),
                        T, has_o, TP, BO)
            wc_a = _wt(wqca_ptr + l * H * H, pid * BO, H, 0, has_o, BO, KA)
            wc_b = _wt(wqca_ptr + l * H * H, pid * BO, H, KA, has_o, BO, KB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 4. CA query of AdaRMS(h2)  [prefetch: CA-o tile]
            xa = _rows(h2_ptr, H, 0, T, has_o, TP, KA)
            xb = _rows(h2_ptr, H, KA, T, has_o, TP, KB)
            aa, ab = _modnorm2(xa, xb, m_ca, eps, H, KA, KB)
            acc = tl.dot(aa, tl.trans(wc_a))
            acc = tl.dot(ab, tl.trans(wc_b), acc)
            _store_rows(qca_ptr, H, pid * BO, acc.to(tl.bfloat16), T, has_o, TP, BO)
            wo_a = _wt(woca_ptr + l * H * H, pid * BO, H, 0, has_o, BO, KA)
            wo_b = _wt(woca_ptr + l * H * H, pid * BO, H, KA, has_o, BO, KB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 5. cross-attention per head
            if is_att:
                _ca_attention(pid, l, qca_ptr, qcan_ptr, cacos_ptr, casin_ptr, kca_ptr, vca_ptr, S, y_ptr, T,
                              qk_eps, sm_scale, TP, HD, NH, SP, H, BS)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 6. CA o_proj + gated residual -> h3  [prefetch: gate/up tiles]
            ya = _rows(y_ptr, H, 0, T, has_o, TP, KA)
            yb = _rows(y_ptr, H, KA, T, has_o, TP, KB)
            acc = tl.dot(ya, tl.trans(wo_a))
            acc = tl.dot(yb, tl.trans(wo_b), acc)
            _store_rows(h3_ptr, H, pid * BO, _gated_residual(acc, h2_ptr, m_ca + 2 * H, pid * BO, T, has_o, TP, H, BO),
                        T, has_o, TP, BO)
            wg_a = _wt(wg_ptr + l * I * H, pid * BG, H, 0, has_g, BG, KA)
            wg_b = _wt(wg_ptr + l * I * H, pid * BG, H, KA, has_g, BG, KB)
            wu_a = _wt(wu_ptr + l * I * H, pid * BG, H, 0, has_g, BG, KA)
            wu_b = _wt(wu_ptr + l * I * H, pid * BG, H, KA, has_g, BG, KB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 7. gate/up of AdaRMS(h3), g = silu(gate) * up  [prefetch: down tile]
            xa = _rows(h3_ptr, H, 0, T, has_g, TP, KA)
            xb = _rows(h3_ptr, H, KA, T, has_g, TP, KB)
            aa, ab = _modnorm2(xa, xb, m_mlp, eps, H, KA, KB)
            ga = tl.dot(aa, tl.trans(wg_a))
            ga = tl.dot(ab, tl.trans(wg_b), ga)
            ua = tl.dot(aa, tl.trans(wu_a))
            ua = tl.dot(ab, tl.trans(wu_b), ua)
            gf = ga.to(tl.bfloat16).to(tl.float32)
            uf = ua.to(tl.bfloat16).to(tl.float32)
            _store_rows(g_ptr, I, pid * BG, (gf / (1.0 + tl.exp(-gf)) * uf).to(tl.bfloat16), T, has_g, TP, BG)
            wd_a = _wt(wd_ptr + l * H * I, pid * BO, I, 0, has_o, BO, IA)
            wd_b = _wt(wd_ptr + l * H * I, pid * BO, I, IA, has_o, BO, IB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
            # 8. down + gated residual -> hh[1]  [prefetch: next layer's QKV tile]
            ga_ = _rows(g_ptr, I, 0, T, has_o, TP, IA)
            gb_ = _rows(g_ptr, I, IA, T, has_o, TP, IB)
            acc = tl.dot(ga_, tl.trans(wd_a))
            acc = tl.dot(gb_, tl.trans(wd_b), acc)
            _store_rows(hl_ptr, H, pid * BO, _gated_residual(acc, h3_ptr, m_mlp + 2 * H, pid * BO, T, has_o, TP, H, BO),
                        T, has_o, TP, BO)
            ln = (l + 1) % L
            wq_a = _wt(wqkv_ptr + ln * 3 * H * H, pid * BQ, H, 0, has_q, BQ, KA)
            wq_b = _wt(wqkv_ptr + ln * 3 * H * H, pid * BQ, H, KA, has_q, BQ, KB)
            nb += 1
            _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
        # Step tail (programs with an in-proj tile): v = out-proj(AdaRMS_final(h)) on the AO real dims,
        # x <- x_pad + dt v (x double-buffered: read slot s%2, program 0 writes slot (s+1)%2),
        # h0 <- in-proj(zero-pad(x)) for the next step.
        vel = _velocity(hl_ptr, mrow + 3 * L * 3 * H, wout_ptr, bout_ptr, T, eps, has_o, 0, TP, H, A, AO, KA, KB, BKV)
        xr = xbuf_ptr + (s % 2) * TP * A
        xw = xbuf_ptr + ((s + 1) % 2) * TP * A
        ao = tl.arange(0, AO)
        xo = tl.load(xr + rows[:, None] * A + ao[None, :], mask=(rows[:, None] < T) & has_o, other=0.,
                     cache_modifier='.cg')
        xo = tl.where(ao[None, :] < action_dim, xo, 0.) + dt * vel
        tl.store(xw + rows[:, None] * A + ao[None, :], xo, mask=(rows[:, None] < T) & (pid == 0))
        xo = tl.where(ao[None, :] < action_dim, xo, 0.)
        nn = pid * BO + tl.arange(0, BO)
        wi = tl.load(win_ptr + nn[:, None] * A + ao[None, :], mask=(nn[:, None] >= 0) & has_o, other=0.)
        hi = tl.sum(xo[:, None, :] * wi[None, :, :], axis=2) \
            + tl.load(bin_ptr + nn, mask=(nn >= 0) & has_o, other=0.)[None, :]
        _store_rows(h0_ptr, H, pid * BO, hi.to(tl.bfloat16), T, has_o, TP, BO)
        nb += 1
        _grid_sync(bar_ptr, nb, nprog, prof_ptr, pid, POLL, PROFILE)
    # Padded action dims AO..A of the final x (never fed back): 0 + dt * v, from the last step's h.
    if pid == 0:
        for a0 in range(AO, A, AO):
            vel = _velocity(hl_ptr, mods_ptr + n_steps * MODW + 3 * L * 3 * H, wout_ptr, bout_ptr, T, eps, pid == 0,
                            a0, TP, H, A, AO, KA, KB, BKV)
            ao = a0 + tl.arange(0, AO)
            tl.store(xbuf_ptr + (n_steps % 2) * TP * A + rows[:, None] * A + ao[None, :], dt * vel,
                     mask=(rows[:, None] < T) & (ao[None, :] < A))


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
        assert (self.H, self.I) == (768, 3072), 'K chunking assumes 768 = 512 + 256 and 3072 = 2048 + 1024'
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
        assert type(layers[0].mlp.act_fn).__name__ in ('SiLU', 'SiLUActivation')
        dev, bf = self.wqkv.device, self.wqkv.dtype
        self.TP = 16
        z = lambda n: torch.zeros(self.TP, n, device=dev, dtype=bf)  # noqa: E731
        self.hh = torch.zeros(2, self.TP, self.H, device=dev, dtype=bf)
        self.h2, self.h3 = z(self.H), z(self.H)
        self.qkv, self.y, self.qca, self.g = z(3 * self.H), z(self.H), z(self.H), z(self.I)
        self.xbuf = torch.zeros(2, self.TP, self.A, device=dev, dtype=torch.float32)
        self.bar = torch.zeros(1, device=dev, dtype=torch.int32)
        self.nprog = torch.cuda.get_device_properties(dev).multi_processor_count
        assert self.nprog >= 3 * self.H // 32 and self.nprog >= self.I // 32, 'one tile per program per stage'
        self.prof = torch.zeros(4096, device=dev, dtype=torch.int64)
        # POLL: barrier spin flavour; PROFILE: per-barrier timestamps (tune_mk.py).
        self.cfg = dict(AO=8, BS=64, BKV=64, POLL=0, PROFILE=False, num_warps=8, num_stages=1)

    def run(self, x_t: torch.Tensor, mods: torch.Tensor, dt: torch.Tensor, past, action_dim: int) -> torch.Tensor:
        """x_t: [1, T, A] fp32 after Euler step 1; mods: [N, MODW] (row 0 = step 1, used already);
        returns x_t after step N (padded dims as the reference: 0 + dt * v)."""
        step = past.register_step
        T = x_t.shape[1]
        assert x_t.shape[0] == 1 and T <= self.TP and action_dim <= self.cfg['AO']
        memk = torch.stack([m[0] for m in step['mem_kv']])
        memv = torch.stack([m[1] for m in step['mem_kv']])
        kca = torch.stack([kv[0] for kv in past.cross_attn_kv])
        vca = torch.stack([kv[1] for kv in past.cross_attn_kv])
        PM, SP = memk.shape[1], kca.shape[1]
        sa_cos, sa_sin = (t.contiguous() for t in step['sa_pe'])
        ca_cos, ca_sin = (t.contiguous() for t in step['ca_pe'])
        assert sa_cos.shape == (T, self.HD) and ca_cos.shape == (T, self.HD), (sa_cos.shape, ca_cos.shape)
        self.xbuf[0, :T].copy_(x_t.reshape(T, self.A))
        mods = mods.contiguous()
        n_steps = mods.shape[0] - 1
        self.bar.zero_()
        expert_megakernel[(self.nprog,)](
            self.xbuf, mods, dt,
            self.win, self.bin, self.wout, self.bout,
            self.wqkv, self.wo, self.wqca, self.woca, self.wg, self.wu, self.wd,
            self.qn, self.kn, self.qcan,
            memk, memv, step['cu_kv'],
            kca, vca, step['cu_src'],
            sa_cos, sa_sin, ca_cos, ca_sin,
            self.hh, self.h2, self.h3, self.qkv, self.y, self.qca, self.g,
            self.bar, self.prof, n_steps, action_dim, T, self.eps, self.qk_eps, self.HD ** -0.5,
            L=self.L, TP=self.TP, H=self.H, I=self.I, A=self.A, NH=self.NH, HD=self.HD, PM=PM, PMP=_pow2(PM),
            SP=SP, MODW=mods.shape[1], **self.cfg)
        return self.xbuf[n_steps % 2, :T].clone().reshape(1, T, self.A)

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
