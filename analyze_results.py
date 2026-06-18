"""
Post-Study Statistical Analysis Script
Run after collecting all participant data to generate:
  - Group comparison plots
  - Statistical tests (Mann-Whitney U, t-test)
  - Summary tables
  - Correlation analysis

Usage:
    python analyze_results.py
"""

import os
import json
import glob
import numpy as np
import csv
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib not installed. Skipping plots.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠ scipy not installed. Skipping statistical tests.")


# ─── Load Data ───────────────────────────────────────────────────────────────
def load_all_summaries(data_dir='data'):
    immediate, adaptive = [], []
    for path in glob.glob(os.path.join(data_dir, '*_summary.json')):
        with open(path) as f:
            d = json.load(f)
        if d['condition'] == 'immediate':
            immediate.append(d)
        else:
            adaptive.append(d)
    return immediate, adaptive


def extract_metric(group, key, sub=None):
    """Pull a metric from summary list, optionally from questionnaire sub-dict."""
    vals = []
    for d in group:
        if sub:
            v = d.get(sub, {}).get(key)
        else:
            v = d.get(key)
        if v is not None:
            vals.append(float(v))
    return vals


# ─── Statistical Tests ────────────────────────────────────────────────────────
def compare_groups(imm_vals, adp_vals, label):
    print(f"\n── {label} ──────────────────────────────────")
    print(f"  Immediate  n={len(imm_vals)}  mean={np.mean(imm_vals):.2f}  median={np.median(imm_vals):.2f}  SD={np.std(imm_vals):.2f}")
    print(f"  Adaptive   n={len(adp_vals)}  mean={np.mean(adp_vals):.2f}  median={np.median(adp_vals):.2f}  SD={np.std(adp_vals):.2f}")

    if HAS_SCIPY and len(imm_vals) >= 3 and len(adp_vals) >= 3:
        u_stat, p_mann = stats.mannwhitneyu(imm_vals, adp_vals, alternative='two-sided')
        t_stat, p_t    = stats.ttest_ind(imm_vals, adp_vals)
        print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_mann:.4f} {'*' if p_mann < 0.05 else ''}")
        print(f"  Independent t  = {t_stat:.2f},  p = {p_t:.4f} {'*' if p_t < 0.05 else ''}")

        # Effect size (Cohen's d)
        pooled_sd = np.sqrt((np.std(imm_vals)**2 + np.std(adp_vals)**2) / 2)
        d = (np.mean(adp_vals) - np.mean(imm_vals)) / (pooled_sd + 1e-9)
        mag = 'small' if abs(d) < 0.5 else ('medium' if abs(d) < 0.8 else 'large')
        print(f"  Cohen's d = {d:.2f} ({mag} effect)")
        return p_mann, d
    return None, None


# ─── Plots ────────────────────────────────────────────────────────────────────
COLORS = {'immediate': '#f59e0b', 'adaptive': '#4f8ef7'}

def box_compare(ax, imm, adp, title, ylabel, invert_better=False):
    bp = ax.boxplot([imm, adp],
                    labels=['Immediate', 'Adaptive'],
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#8892a4'),
                    capprops=dict(color='#8892a4'),
                    flierprops=dict(marker='o', color='#8892a4', markersize=4))
    bp['boxes'][0].set_facecolor(COLORS['immediate'])
    bp['boxes'][1].set_facecolor(COLORS['adaptive'])

    # Scatter points
    for i, (group, vals) in enumerate([(imm, imm), (adp, adp)], 1):
        jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter([i + j for j in jitter], vals, color='white', alpha=0.4, s=20, zorder=3)

    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color='#8892a4')
    ax.tick_params(colors='#8892a4')
    ax.set_facecolor('#1e2230')
    for spine in ax.spines.values():
        spine.set_color('#2a2f3e')


def make_plots(immediate, adaptive, out_dir='data'):
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(14, 10), facecolor='#0d0f14')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    metrics = [
        ('total_alerts',          None, 'Alerts Generated',          'Count',    False),
        ('avg_latency_s',         None, 'Avg Correction Latency',    'Seconds',  False),
        ('pct_time_poor_posture', None, '% Time Poor Posture',       '%',        False),
        ('ignored_alerts',        None, 'Alerts Ignored',            'Count',    False),
        ('score_helpfulness',  'questionnaire', 'Perceived Helpfulness','Score (1–7)', True),
        ('score_annoyance',    'questionnaire', 'Perceived Annoyance',  'Score (1–7)', False),
    ]

    for idx, (key, sub, title, ylabel, _) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        imm_vals = extract_metric(immediate, key, sub)
        adp_vals = extract_metric(adaptive,  key, sub)
        if imm_vals or adp_vals:
            box_compare(ax, imm_vals or [0], adp_vals or [0], title, ylabel)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='#8892a4')
            ax.set_title(title, fontsize=11, color='white')
            ax.set_facecolor('#1e2230')

    fig.suptitle('Immediate vs Adaptive Feedback — Group Comparison',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(out_dir, f'group_comparison_{ts}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0f14')
    plt.close()
    print(f"\n✓ Comparison plot saved: {out}")
    return out


def adaptive_delay_trajectory(adaptive_group, out_dir='data'):
    """Plot how the adaptive delay threshold evolved across alerts for each participant."""
    if not HAS_MATPLOTLIB or not adaptive_group:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d0f14')
    ax.set_facecolor('#1e2230')

    for p in adaptive_group:
        pid = p.get('participant_id', '?')
        # Re-load corresponding alert CSV to get per-alert delay thresholds
        pattern = f"data/{pid}_adaptive_*_alerts.csv"
        files = glob.glob(pattern)
        if not files:
            continue
        delays, latencies = [], []
        with open(files[0]) as f:
            for row in csv.DictReader(f):
                if row['delay_threshold']:
                    delays.append(float(row['delay_threshold']))
                    lat = row['latency']
                    latencies.append(float(lat) if lat else None)

        if delays:
            ax.plot(range(1, len(delays)+1), delays,
                    marker='o', markersize=4, linewidth=1.5,
                    alpha=0.7, label=pid)

    ax.axhline(5.0, color='#8892a4', linestyle='--', linewidth=0.8, label='Initial (5s)')
    ax.set_xlabel('Alert Number', color='#8892a4', fontsize=10)
    ax.set_ylabel('Delay Threshold (s)', color='#8892a4', fontsize=10)
    ax.set_title('Adaptive Delay Threshold Evolution per Participant',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#8892a4')
    ax.legend(fontsize=8, facecolor='#2a2f3e', labelcolor='white', framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_color('#2a2f3e')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(out_dir, f'adaptive_trajectory_{ts}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0f14')
    plt.close()
    print(f"✓ Trajectory plot saved: {out}")


# ─── Summary Table ────────────────────────────────────────────────────────────
def print_summary_table(immediate, adaptive):
    keys = [
        ('total_alerts',          None,            'Total Alerts'),
        ('ignored_alerts',        None,            'Ignored Alerts'),
        ('avg_latency_s',         None,            'Avg Latency (s)'),
        ('median_latency_s',      None,            'Median Latency (s)'),
        ('pct_time_poor_posture', None,            '% Poor Posture'),
        ('score_helpfulness',  'questionnaire',   'Helpfulness (1-7)'),
        ('score_annoyance',    'questionnaire',   'Annoyance (1-7)'),
        ('score_usability',    'questionnaire',   'Usability (1-7)'),
        ('score_willingness',  'questionnaire',   'Willingness (1-7)'),
    ]

    print("\n" + "="*65)
    print(f"{'Metric':<30} {'Immediate':>15} {'Adaptive':>15}")
    print("="*65)
    for key, sub, label in keys:
        imm = extract_metric(immediate, key, sub)
        adp = extract_metric(adaptive,  key, sub)
        i_str = f"{np.mean(imm):.2f} ± {np.std(imm):.2f}" if imm else "—"
        a_str = f"{np.mean(adp):.2f} ± {np.std(adp):.2f}" if adp else "—"
        print(f"  {label:<28} {i_str:>15} {a_str:>15}")
    print("="*65)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "━"*60)
    print("  PostureAware — Study Analysis")
    print("━"*60)

    immediate, adaptive = load_all_summaries()
    print(f"\nLoaded: {len(immediate)} Immediate  |  {len(adaptive)} Adaptive participants")

    if not immediate and not adaptive:
        print("\n⚠  No participant data found in data/ directory.")
        print("   Run sessions first, then re-run this script.")
        return

    # Summary table
    print_summary_table(immediate, adaptive)

    # Statistical comparisons
    obj_metrics = [
        ('total_alerts',          None,          'Total Alerts Generated'),
        ('avg_latency_s',         None,          'Avg Correction Latency (s)'),
        ('pct_time_poor_posture', None,          '% Time in Poor Posture'),
        ('ignored_alerts',        None,          'Ignored Alerts'),
    ]
    for key, sub, label in obj_metrics:
        imm = extract_metric(immediate, key, sub)
        adp = extract_metric(adaptive,  key, sub)
        if imm or adp:
            compare_groups(imm or [], adp or [], label)

    if any(d.get('questionnaire') for d in immediate + adaptive):
        print("\n── Subjective Measures ──────────────────────────────────")
        subj_metrics = [
            ('score_helpfulness', 'questionnaire', 'Perceived Helpfulness'),
            ('score_annoyance',   'questionnaire', 'Perceived Annoyance'),
            ('score_usability',   'questionnaire', 'Perceived Usability'),
            ('score_willingness', 'questionnaire', 'Willingness to Use'),
        ]
        for key, sub, label in subj_metrics:
            imm = extract_metric(immediate, key, sub)
            adp = extract_metric(adaptive,  key, sub)
            if imm or adp:
                compare_groups(imm or [], adp or [], label)

    # Plots
    make_plots(immediate, adaptive)
    adaptive_delay_trajectory(adaptive)

    print("\n✓ Analysis complete.\n")


if __name__ == '__main__':
    main()
