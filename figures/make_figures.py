"""Figures of the study from the committed result files: writes figures/*.png|svg and the tidy CSV each one is
built from (results/csv/). Run from the repo root: `python figures/make_figures.py`.

Sources: results/original/ (the study run) and results/verification/ (re-run from a fresh clone). The |Δ|
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
    ax.text(ax.get_xlim()[1], 1000 / 60, '60 Hz budget 16.7 ms', ha='right', va='bottom', fontsize=8)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
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
    for p in sorted((SRC['verification'] / 'tune').glob('v*.json')):
        d = jload(p)
        for cfg, r in d.items():
            if cfg.startswith('{') and 'graph_ms' in r:
                rows.append({'source': 'verification', 'version': p.stem, 'config': cfg, 'graph_ms': r['graph_ms'],
                             'max_abs_diff_vs_compiled': r['max_abs_diff_vs_compiled'], 'spill_bytes': r.get('n_spills')})
    write_csv('tuning.csv', rows)
    ref = [jload(SRC[s] / 'breakdown_compiled.json')['subsplit_graph_replay_ms']['ref_steps_2_to_N_ms']
           for s in available_sources()]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vers = ['v1', 'v2', 'v3', 'v4']
    for src, mk, off in (('original', 'o', -0.12), ('verification', 's', 0.12)):
        for r in (r for r in rows if r['source'] == src):
            x = vers.index(r['version']) + off
            ax.scatter(x, r['graph_ms'], marker=mk, color=C['compiled'] if src == 'original' else C['mk'], s=28,
                       label=src if r is next(q for q in rows if q['source'] == src) else None)
    for v in ref:
        ax.axhline(v, color=C['k'], lw=0.8, ls='--')
    ax.text(3.5, ref[0], f'Kyutai compiled, steps 2–10: {ref[0]:.2f} ms', ha='right', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_yticks([4, 5, 7, 10, 20, 40, 70], ['4', '5', '7', '10', '20', '40', '70'])
    ax.minorticks_off()
    ax.set_xticks(range(4), ['v1\nK-looped tiles', 'v2\nregister prefetch', 'v3\nsingle-shot tiles',
                             'v4 (final)\nv1 tiles + parallel tail'])
    ax.set_ylabel('Euler steps 2–10, graph replay ms (log)')
    ax.set_title('Megakernel versions and tile/launch configs (each dot one config)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    save(fig, 'f_tuning')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    fig_latency()
    fig_cdf()
    fig_breakdown()
    fig_kernels()
    fig_correctness()
    fig_tuning()
