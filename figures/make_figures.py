"""Figures of the study from the committed result files: writes figures/*.png|svg and the tidy CSV each one is
built from (results/csv/). Run from the repo root: `python figures/make_figures.py`.

Sources: results/original/ (the study run), results/verification/ (re-run from a fresh clone) and results/opt/
(second round: host fixes, compiled prefill, megakernel v5; figures g, h and the v4/v5 points of figure f). The |Δ|
distributions need the per-chunk outputs of bench.py (results/verification/run_a/chunks_*.pt, not committed);
when absent, the committed results/csv/correctness_per_chunk.csv is reused.
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, CSV = ROOT / 'results', ROOT / 'figures', ROOT / 'results' / 'csv'
# Okabe-Ito (colour-blind safe)
C = {'eager': '#999999', 'compiled': '#0072B2', 'mk': '#D55E00', 'a': '#009E73', 'b': '#CC79A7', 'c': '#E69F00',
     'd': '#56B4E9', 'e': '#F0E442', 'k': '#000000'}
LABEL = {'eager': 'eager (Kyutai, compile=False)', 'compiled': 'Kyutai compiled (CUDA graphs)',
         'mk': 'megakernel (Euler steps 2–10)'}
SRC = {'original': RES / 'original', 'verification': RES / 'verification'}
RUNS = {'original': {'run_a': '', 'run_b': 'run_b'}, 'verification': {'run_a': 'run_a', 'run_b': 'run_b'}}
OPT = RES / 'opt'  # second round (2026-11-01): host fixes, compiled prefill, megakernel v5 (fig. g, tuning in f)

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})


def jload(p):
    return json.loads(p.read_text()) if p.exists() else None


def latency(src, run, mode):
    return jload(SRC[src] / RUNS[src][run] / f'latency_{mode}.json')


def write_csv(name, rows):
    CSV.mkdir(parents=True, exist_ok=True)
    with (CSV / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def available_sources():
    return [s for s in SRC if latency(s, 'run_a', 'mk')]


# ------------------------------------------------------------------------------------------------ (a) latency
def fig_latency():
    rows, calls = [], []
    for src in available_sources():
        for run in ('run_a', 'run_b'):
            for mode in ('eager', 'compiled', 'mk'):
                d = latency(src, run, mode)
                if d is None:
                    continue
                rows.append({'source': src, 'run': run, 'mode': mode, 'n': d['n'], 'median_ms': d['median_ms'],
                             'p10_ms': d['p10_ms'], 'p90_ms': d['p90_ms'], 'p99_ms': d['p99_ms'],
                             'mean_ms': d['mean_ms'], 'gpu': d['gpu'], 'torch': d['torch']})
                calls += [{'source': src, 'run': run, 'mode': mode, 'call': i, 'ms': v}
                          for i, v in enumerate(d['lat_ms'])]
    write_csv('latency_summary.csv', rows)
    write_csv('latency_calls.csv', calls)

    groups = [(r['source'], r['run']) for r in rows if r['mode'] == 'mk']
    fig, (ax_e, ax) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 3]})
    for axis, modes in ((ax_e, ['eager']), (ax, ['compiled', 'mk'])):
        xs, labels = [], []
        for gi, (src, run) in enumerate(groups):
            for mi, mode in enumerate(modes):
                r = next((r for r in rows if (r['source'], r['run'], r['mode']) == (src, run, mode)), None)
                if r is None:
                    continue
                x = gi * (len(modes) + 1) + mi
                axis.bar(x, r['median_ms'], color=C[mode], label=LABEL[mode] + ' median' if gi == 0 else None)
                axis.bar(x, r['p99_ms'] - r['median_ms'], bottom=r['median_ms'], color=C[mode], alpha=0.35,
                         label='… p99 (light)' if gi == 0 and mode == modes[-1] else None)
                axis.text(x, r['p99_ms'], f"{r['median_ms']:.1f}\n({r['p99_ms']:.1f})", ha='center', va='bottom',
                          fontsize=8)
                xs.append(x)
                labels.append(f'{"orig." if src == "original" else "verif."}\nrun {run[-1].upper()}')
        axis.set_xticks(xs, labels, fontsize=7)
        axis.set_ylabel('ms per predict (median, p99 in light)')
        axis.set_ylim(0, max(r['p99_ms'] for r in rows if r['mode'] in modes) * 1.18)
    ax_e.set_title('eager (separate scale)')
    ax.set_title('Kyutai compiled vs megakernel')
    ax.axhline(1000 / 60, color=C['k'], lw=0.8, ls='--')
    ax.set_xlim(-2.4, ax.get_xlim()[1])  # empty strip left of the bars for the budget label
    ax.text(-2.35, 1000 / 60 + 0.3, '60 Hz budget\n16.7 ms', ha='left', va='bottom', fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('REKI predict latency, H100 SXM, batch 1, 440 LIBERO frames × 2 (880 timed calls per bar)')
    save(fig, 'a_latency')


# ------------------------------------------------------------------------------------------------ (b) CDF
def fig_cdf():
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {('original', 'run_a'): '-', ('original', 'run_b'): '--', ('verification', 'run_a'): '-.',
              ('verification', 'run_b'): ':'}
    hi = 0
    for src in available_sources():
        for run in ('run_a', 'run_b'):
            for mode in ('compiled', 'mk'):
                d = latency(src, run, mode)
                if d is None:
                    continue
                v = np.sort(d['lat_ms'])
                hi = max(hi, np.percentile(v, 99.5))
                ax.plot(v, np.arange(1, v.size + 1) / v.size, styles[(src, run)], color=C[mode], lw=1.4,
                        label=f'{"compiled" if mode == "compiled" else "megakernel"}, {src} {run.replace("_", " ")}')
    ax.set_xlim(17.5, max(hi, 23) + 0.5)
    ax.set_xlabel('ms per predict')
    ax.set_ylabel('fraction of calls ≤ x')
    ax.set_title('Latency CDF (880 calls per curve; x axis cut at the 99.5th percentile)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    save(fig, 'b_latency_cdf')


# ------------------------------------------------------------------------------------------------ (c) breakdown
PARTS = [('CPU preprocessing', 'c'), ('VLM prefill: vision tower', 'd'), ('VLM prefill: LM + host glue', 'compiled'),
         ('expert: modulations + register_prepare + Euler step 1', 'a'),
         ('expert: rest of the graph (Euler steps 2–10)', 'mk'),
         ('other (gather, register append, decode + D2H)', 'eager')]


def fig_breakdown():
    rows = []
    for src in available_sources():
        for mode in ('compiled', 'mk'):
            d = jload(SRC[src] / f'breakdown_{mode}.json')
            if d is None:
                continue
            st, sub = d['stages_gpu_timeline_ms'], d['subsplit_graph_replay_ms']
            vals = [st['preprocess'], sub['vision_tower_ms'], st['prefill'] - sub['vision_tower_ms'],
                    sub['mods_prepare_step1_ms'], st['expert'] - sub['mods_prepare_step1_ms'],
                    st['gather_state'] + st['register_append'] + st['decode_d2h']]
            for (part, _), v in zip(PARTS, vals):
                rows.append({'source': src, 'mode': mode, 'part': part, 'ms': v, 'total_ms': d['total_ms']})
    write_csv('breakdown.csv', rows)
    bars = sorted({(r['source'], r['mode']) for r in rows}, key=lambda k: (k[0], k[1] != 'compiled'))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (src, mode) in enumerate(bars):
        left = 0
        for part, col in PARTS:
            v = next(r['ms'] for r in rows if (r['source'], r['mode'], r['part']) == (src, mode, part))
            ax.barh(i, v, left=left, color=C[col], label=part if i == 0 else None, edgecolor='white', lw=0.5)
            if v > 0.9:
                ax.text(left + v / 2, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white')
            left += v
        ax.text(left + 0.2, i, f'{left:.1f} ms', va='center', fontsize=8)
    ax.set_yticks(range(len(bars)), [f'{"compiled" if m == "compiled" else "megakernel"} ({s})' for s, m in bars])
    ax.invert_yaxis()
    ax.set_xlabel('ms on the GPU timeline (median of warm predicts; sub-splits from isolated graph replay)')
    ax.set_title('Where one predict spends its time')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, 'c_breakdown')


# ------------------------------------------------------------------------------------------------ (h) breakdown, 2nd round
OPT_PATHS = [('compiled', 'Kyutai compiled'), ('mk', 'megakernel v5'),
             ('compiled_fasthost_cprefill', 'compiled + host fixes + compiled prefill'),
             ('mk_fasthost_cprefill', 'megakernel + host fixes + compiled prefill')]


def fig_breakdown_opt():
    rows = []
    for tag, label in OPT_PATHS:
        d = jload(OPT / f'breakdown_{tag}.json')
        if d is None:
            continue
        st, sub = d['stages_gpu_timeline_ms'], d['subsplit_graph_replay_ms']
        vals = [st['preprocess'], sub['vision_tower_ms'], st['prefill'] - sub['vision_tower_ms'],
                sub['mods_prepare_step1_ms'], st['expert'] - sub['mods_prepare_step1_ms'],
                st['gather_state'] + st['register_append'] + st['decode_d2h']]
        rows += [{'path': tag, 'label': label, 'part': part, 'ms': v, 'total_ms': d['total_ms']}
                 for (part, _), v in zip(PARTS, vals)]
    if not rows:
        return
    write_csv('breakdown_opt.csv', rows)
    paths = [(t, lab) for t, lab in OPT_PATHS if any(r['path'] == t for r in rows)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (tag, _) in enumerate(paths):
        left = 0
        for part, col in PARTS:
            v = next(r['ms'] for r in rows if (r['path'], r['part']) == (tag, part))
            ax.barh(i, v, left=left, color=C[col], label=part if i == 0 else None, edgecolor='white', lw=0.5)
            if v > 0.9:
                ax.text(left + v / 2, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white')
            left += v
        ax.text(left + 0.2, i, f'{left:.1f} ms', va='center', fontsize=8)
    ax.set_yticks(range(len(paths)), [lab for _, lab in paths], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1000 / 60, color=C['k'], lw=0.8, ls='--')
    ax.text(1000 / 60 + 0.1, -0.45, '60 Hz', fontsize=8)
    ax.set_xlabel('ms on the GPU timeline (median of warm predicts; sub-splits from isolated graph replay)')
    ax.set_title('Where one predict spends its time, second round')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, 'h_breakdown_opt')


# ------------------------------------------------------------------------------------------------ (d) kernels
def fig_kernels():
    rows = []
    for src in available_sources():
        kc = jload(SRC[src] / 'kernel_counts.json')
        for mode in ('eager', 'compiled', 'mk'):
            b = jload(SRC[src] / f'breakdown_{mode}.json')
            if b is None:
                continue
            total = b['kernels']['kernels_per_predict']
            row = {'source': src, 'mode': mode, 'kernels_per_predict': total,
                   'host_launch_calls': b['kernels']['host_kernel_launch_calls_per_predict'],
                   'graph_launches': b['kernels']['graph_launches_per_predict'],
                   'prefill_graph': '', 'expert_graph': '', 'outside_graphs': ''}
            if mode != 'eager' and kc:
                row.update(prefill_graph=kc[mode]['prefill_graph_kernels'], expert_graph=kc[mode]['expert_graph_kernels'],
                           outside_graphs=total - kc[mode]['prefill_graph_kernels'] - kc[mode]['expert_graph_kernels'])
            rows.append(row)
    write_csv('kernels.csv', rows)
    src = 'verification' if any(r['source'] == 'verification' for r in rows) else 'original'
    sel = [r for r in rows if r['source'] == src]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, r in enumerate(sel):
        if r['mode'] == 'eager':
            ax.bar(i, r['kernels_per_predict'], color=C['eager'], label='eager: all launched from the host')
        else:
            ax.bar(i, r['prefill_graph'], color=C['compiled'], label='VLM prefill graph' if i == 1 else None)
            ax.bar(i, r['expert_graph'], bottom=r['prefill_graph'], color=C['mk'],
                   label='expert graph' if i == 1 else None)
            ax.bar(i, r['outside_graphs'], bottom=r['prefill_graph'] + r['expert_graph'], color=C['k'],
                   label='outside graphs' if i == 1 else None)
        ax.text(i, r['kernels_per_predict'], f"{r['kernels_per_predict']:,.0f}", ha='center', va='bottom')
    ax.set_xticks(range(len(sel)), [LABEL[r['mode']] for r in sel], fontsize=8)
    ax.set_ylabel('GPU kernels per predict')
    ax.set_title(f'GPU kernels per predict (CUPTI, {src} run)')
    ax.legend(fontsize=8)
    save(fig, 'd_kernels')


# ------------------------------------------------------------------------------------------------ (e) correctness
PAIRS = [('compiled', 'mk', 'megakernel vs compiled'), ('eager', 'compiled', 'compiled vs eager'),
         ('eager', 'mk', 'megakernel vs eager')]


def fig_correctness():
    run = SRC['verification'] / 'run_a'
    if (run / 'chunks_mk.pt').exists():
        import torch
        ch = {m: torch.load(run / f'chunks_{m}.pt')['chunks'][..., :7] for m in ('eager', 'compiled', 'mk')}
        rows = []
        for a, b, name in PAIRS:
            d = (ch[a] - ch[b]).abs()
            per = d.flatten(1).amax(1)
            rows += [{'pair': name, 'chunk': i, 'max_abs': float(per[i]), 'mean_abs': float(d[i].mean())}
                     for i in range(per.numel())]
        write_csv('correctness_per_chunk.csv', rows)
    with (CSV / 'correctness_per_chunk.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    for (_, _, name), col in zip(PAIRS, (C['mk'], C['compiled'], C['b'])):
        v = np.sort([float(r['max_abs']) for r in rows if r['pair'] == name])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, lw=1.6, label=f'{name} (max {v[-1]:.1e})')
    ax.set_xlabel('max |Δ| over the 10 × 7 normalized actions of one chunk')
    ax.set_ylabel('fraction of the 440 chunks ≤ x')
    ax.set_title('Correctness: per-chunk max |Δ| (verification run, same frames and noise)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    save(fig, 'e_correctness')


# ------------------------------------------------------------------------------------------------ (f) tuning
ORIG_LOGS = {  # original tuning logs -> kernel version (log 1, the v1 baseline run, was not kept)
    'log_tune2.txt': 'v2', 'log_tune3.txt': 'v3', 'log_tune4.txt': 'v3', 'log_tune5.txt': 'v1', 'log_tune6.txt': 'v4',
    'log_tune7.txt': 'v4', 'log_tune8.txt': 'v4', 'log_tune9.txt': 'v4'}


def fig_tuning():
    rows = []
    for log, ver in ORIG_LOGS.items():
        for line in (RES / 'original' / 'logs' / log).read_text().splitlines():
            m = re.match(r'^(\{.*?\}) (\{"max_abs.*\})$', line)
            if m and 'graph_ms' in m.group(2):
                d = json.loads(m.group(2))
                rows.append({'source': 'original', 'version': ver, 'config': m.group(1), 'graph_ms': d['graph_ms'],
                             'max_abs_diff_vs_compiled': d['max_abs_diff_vs_compiled'],
                             'spill_bytes': d.get('n_spills')})
    for src, tune_dir in (('verification', SRC['verification'] / 'tune'), ('opt', OPT / 'tune')):
        for p in sorted(tune_dir.glob('v*.json')):
            d = jload(p)
            for cfg, r in d.items():
                if cfg.startswith('{') and 'graph_ms' in r:
                    rows.append({'source': src, 'version': p.stem, 'config': cfg, 'graph_ms': r['graph_ms'],
                                 'max_abs_diff_vs_compiled': r['max_abs_diff_vs_compiled'],
                                 'spill_bytes': r.get('n_spills')})
    write_csv('tuning.csv', rows)
    ref = [jload(SRC[s] / 'breakdown_compiled.json')['subsplit_graph_replay_ms']['ref_steps_2_to_N_ms']
           for s in available_sources()]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    vers = ['v1', 'v2', 'v3', 'v4', 'v5']
    for src, mk, off, col in (('original', 'o', -0.18, C['compiled']), ('verification', 's', 0.0, C['mk']),
                              ('opt', '^', 0.18, C['a'])):
        for r in (r for r in rows if r['source'] == src):
            x = vers.index(r['version']) + off
            ax.scatter(x, r['graph_ms'], marker=mk, color=col, s=28,
                       label={'opt': 'second round'}.get(src, src) if r is next(q for q in rows if q['source'] == src)
                       else None)
    for v in ref:
        ax.axhline(v, color=C['k'], lw=0.8, ls='--')
    ax.text(4.5, ref[0], f'Kyutai compiled, steps 2–10: {ref[0]:.2f} ms', ha='right', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_yticks([4, 5, 7, 10, 20, 40, 70], ['4', '5', '7', '10', '20', '40', '70'])
    ax.minorticks_off()
    ax.set_xticks(range(5), ['v1\nK-looped tiles', 'v2\nregister prefetch', 'v3\nsingle-shot tiles',
                             'v4\nv1 tiles + parallel tail', 'v5 (final)\nv4 + own-tile L2 prefetch'])
    ax.set_ylabel('Euler steps 2–10, graph replay ms (log)')
    ax.set_title('Megakernel versions and tile/launch configs (each dot one config)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    save(fig, 'f_tuning')


# ------------------------------------------------------------------------------------------------ (g) ladder
LADDER = [('compiled', 'Kyutai compiled'), ('compiled_fasthost', '+ host fixes'),
          ('compiled_fasthost_cprefill', '+ host fixes + compiled prefill'), ('mk', 'megakernel v5'),
          ('mk_fasthost', 'megakernel + host fixes'), ('mk_fasthost_cprefill', 'megakernel + host fixes + compiled prefill')]


def fig_ladder():
    rows = []
    for run in ('run_a', 'run_b'):
        for tag, label in LADDER:
            d = jload(OPT / run / f'latency_{tag}.json')
            if d is not None:
                rows.append({'run': run, 'path': tag, 'label': label, 'n': d['n'], 'median_ms': d['median_ms'],
                             'p10_ms': d['p10_ms'], 'p90_ms': d['p90_ms'], 'p99_ms': d['p99_ms'], 'mean_ms': d['mean_ms']})
    if not rows:
        return
    write_csv('ladder_latency.csv', rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ys = {tag: len(LADDER) - 1 - i for i, (tag, _) in enumerate(LADDER)}
    for r in rows:
        y = ys[r['path']] + (0.18 if r['run'] == 'run_a' else -0.18)
        col = C['mk'] if r['path'].startswith('mk') else C['compiled']
        ax.barh(y, r['median_ms'], height=0.32, color=col, alpha=1.0 if r['run'] == 'run_a' else 0.6)
        ax.plot([r['median_ms'], r['p99_ms']], [y, y], color=C['k'], lw=0.8)
        ax.text(r['p99_ms'] + 0.15, y, f"{r['median_ms']:.2f} ({r['p99_ms']:.2f})", va='center', fontsize=7)
    ax.set_yticks(list(ys.values()), [label for _, label in LADDER], fontsize=8)
    ax.axvline(1000 / 60, color=C['k'], lw=0.8, ls='--')
    ax.text(1000 / 60 + 0.1, len(LADDER) - 0.45, '60 Hz: 16.7 ms', fontsize=8)
    ax.set_xlim(0, max(r['p99_ms'] for r in rows) * 1.2)
    ax.set_xlabel('ms per predict: median (bar), whisker to p99; run A dark, run B light')
    ax.set_title('Second round: host fixes, compiled prefill and megakernel v5 (880 timed calls per bar)')
    save(fig, 'g_ladder')


# ------------------------------------------------------------------------------------------------ (i) other GPUs
GPU_PATHS = [('compiled', 'Kyutai compiled', C['compiled'], 1.0), ('mk', 'megakernel v5', C['mk'], 1.0),
             ('compiled_fasthost_cprefill', 'compiled + host fixes + compiled prefill', C['compiled'], 0.55),
             ('mk_fasthost_cprefill', 'megakernel + host fixes + compiled prefill', C['mk'], 0.55)]


def gpu_runs():
    """(label, dir) of every per-GPU run: results/gpus/<slug>/ plus the second-round H100 run A."""
    short = {'RTX PRO 6000 Blackwell Server Edition': 'RTX PRO 6000*', 'A100-SXM4-80GB': 'A100 SXM',
             'RTX 5090': 'RTX 5090*'}
    runs = [('H100 SXM', OPT / 'run_a')]
    for d in sorted((RES / 'gpus').glob('*/')):
        name = (d / 'gpu.txt').read_text().split(',')[0].strip() if (d / 'gpu.txt').exists() else d.name
        name = name.replace('NVIDIA ', '').replace('GeForce ', '')
        runs.append((short.get(name, name), d))
    return runs


def fig_gpus():
    rows = []
    for gpu, d in gpu_runs():
        for tag, label, _, _ in GPU_PATHS:
            r = jload(d / f'latency_{tag}.json')
            if r is not None:
                rows.append({'gpu': gpu, 'path': tag, 'median_ms': r['median_ms'], 'p99_ms': r['p99_ms'],
                             'p10_ms': r['p10_ms'], 'p90_ms': r['p90_ms'], 'n': r['n']})
    if len({r['gpu'] for r in rows}) < 2:
        return
    write_csv('gpus_latency.csv', rows)
    gpus = sorted({r['gpu'] for r in rows},
                  key=lambda g: next((r['median_ms'] for r in rows if r['gpu'] == g and r['path'] == 'compiled'), 1e9))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.2
    for j, (tag, label, col, alpha) in enumerate(GPU_PATHS):
        for i, g in enumerate(gpus):
            r = next((r for r in rows if (r['gpu'], r['path']) == (g, tag)), None)
            if r is None:
                continue
            x = i + (j - 1.5) * w
            ax.bar(x, r['median_ms'], w, color=col, alpha=alpha, label=label if i == 0 else None)
            ax.text(x, r['median_ms'], f"{r['median_ms']:.1f}", ha='center', va='bottom', fontsize=6.5)
    ax.axhline(1000 / 60, color=C['k'], lw=0.8, ls='--')
    ax.text(-0.5, 1000 / 60 + 0.3, '60 Hz', ha='left', fontsize=8)
    ax.set_xticks(range(len(gpus)), gpus, fontsize=8)
    ax.set_ylabel('ms per predict (median)')
    ax.set_xlabel('* sm_120: the max |Δ| vs eager comes from single gripper flips on a decision boundary, see README')
    ax.set_title('REKI predict on different GPUs (batch 1, 440 LIBERO frames, 880 timed calls per bar)')
    ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(0.06, 1.0))
    save(fig, 'i_gpus')


# ------------------------------------------------------------------------------------------------ hero (README top, post)
def fig_hero():
    """One 16:9 picture: the H100 waterfall of the three changes (left) and Kyutai's path vs the final path
    on every GPU (right). Medians of 880 timed predicts."""
    steps = [('compiled', 'Kyutai\ncompiled'), ('compiled_fasthost', '+ host\nfixes'),
             ('compiled_fasthost_cprefill', '+ compiled\nprefill'), ('mk_fasthost_cprefill', '+ Triton\nmegakernel')]
    h100 = [jload(OPT / 'run_a' / f'latency_{t}.json') for t, _ in steps]
    gpu_rows = []
    for gpu, d in gpu_runs():
        a, b = jload(d / 'latency_compiled.json'), jload(d / 'latency_mk_fasthost_cprefill.json')
        if a and b:
            gpu_rows.append((gpu.rstrip('*'), a['median_ms'], b['median_ms']))
    if any(x is None for x in h100) or len(gpu_rows) < 2:
        return
    gpu_rows.sort(key=lambda r: r[2])
    with plt.rc_context({'font.size': 12}):
        fig = plt.figure(figsize=(12.8, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32, left=0.07, right=0.97, top=0.80,
                              bottom=0.12)
        ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        fig.suptitle('Kyutai REKI (2.4B VLA), one robot predict: 21.7 → 12.0 ms on H100 (1.8×)',
                     fontsize=17, fontweight='bold', y=0.965)
        fig.text(0.5, 0.875, 'batch 1 · 440 real LIBERO frames · median of 880 timed calls · actions within ~1e-2 '
                 'of eager (host fixes: bitwise identical)', ha='center', fontsize=11, color='#444444')
        # left: waterfall; the hatched part of each bar is what that change removed
        vals = [d['median_ms'] for d in h100]
        for i, v in enumerate(vals):
            col = C['compiled'] if i == 0 else C['mk'] if i == len(vals) - 1 else '#6baed6'
            ax.bar(i, v, 0.62, color=col)
            ax.text(i, v / 2, f'{v:.1f}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            if i:
                saved = vals[i - 1] - v
                ax.bar(i, saved, 0.62, bottom=v, color='none', edgecolor=C['a'], hatch='///', lw=1.2)
                ax.text(i, vals[i - 1] + 0.3, f'−{saved:.1f} ms', ha='center', va='bottom', fontsize=12,
                        fontweight='bold', color=C['a'])
        ax.set_xticks(range(len(steps)), [s for _, s in steps], fontsize=11)
        ax.set_ylabel('ms per predict (H100 SXM)')
        ax.set_ylim(0, vals[0] * 1.15)
        ax.set_title('What each change saves', fontsize=13, loc='left')
        # right: GPUs
        y = np.arange(len(gpu_rows))
        bx.barh(y - 0.19, [r[1] for r in gpu_rows], 0.36, color=C['compiled'], label='Kyutai compiled')
        bx.barh(y + 0.19, [r[2] for r in gpu_rows], 0.36, color=C['mk'], label='this repo (all three changes)')
        for yi, (_, a, b) in zip(y, gpu_rows):
            bx.text(a + 0.3, yi - 0.19, f'{a:.1f}', va='center', fontsize=10)
            bx.text(b + 0.3, yi + 0.19, f'{b:.1f} ms  ({a / b:.2f}×)', va='center', fontsize=10, fontweight='bold')
        bx.set_yticks(y, [r[0] for r in gpu_rows], fontsize=11)
        bx.invert_yaxis()
        bx.set_xlim(0, max(r[1] for r in gpu_rows) * 1.32)
        bx.set_xlabel('ms per predict')
        bx.set_title('Same code on 8 GPUs', fontsize=13, loc='left')
        bx.set_ylim(len(gpu_rows) - 0.5, -1.3)  # inverted; an empty band on top holds the legend
        bx.legend(loc='upper left', ncol=2, fontsize=10.5, frameon=False, handlelength=1.2)
        fig.text(0.97, 0.02, 'github.com/muninn-huginn/reki-megakernel', ha='right', fontsize=10, color='#666666')
        for ext in ('png', 'svg'):
            fig.savefig(FIG / f'hero.{ext}', dpi=150)
        plt.close(fig)
    print('wrote hero')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    fig_latency()
    fig_cdf()
    fig_breakdown()
    fig_kernels()
    fig_correctness()
    fig_tuning()
    fig_ladder()
    fig_breakdown_opt()
    fig_gpus()
    fig_hero()
