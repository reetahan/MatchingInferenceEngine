"""
Fig. 4 style plot: Top-1, Top-5, and Unmatched rates by lottery decile,
under STB and MTB, for real Chilean preference/priority data.

Decile definition:
  - STB: students are bucketed by their single lottery draw (shared across
    every school they applied to).
  - MTB: students are bucketed by their *best* (lowest) per-school lottery
    draw among their top-5 listed preferences -- i.e. their most favorable
    relative lottery position across the schools they actually ranked
    highest.

Both conditions are averaged over --n_runs independent lottery draws.
Reuses the DA/priority machinery from chilean_real_welfare_comparison.py.

Usage:
    python chile_lottery_decile_stb_mtb.py \
        --individual <path_to_indv_df> \
        --capacity   <path_to_capacity_df> \
        --n_runs 10 \
        --output fig4_chile_lottery_decile.png \
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from file_config import DATA_GENERATION_SEED
from chile_priority_attributes import _prepare_school_capacity_table
from chilean_real_welfare_comparison import (
    load_df, build_applications_long, run_matching,
)

N_DECILES = 10


def compute_student_records(student_ids, student_rankings, matches, lottery_df, decile_source):
    """
    Per-student matched/rank_pos, plus the scalar used for decile bucketing.
    decile_source: 'single'    -> STB, one shared lottery value per student
                   'best_top5' -> MTB, min lottery among preference_number<=5
    """
    if decile_source == 'single':
        decile_val = lottery_df.drop_duplicates('student_idx').set_index('student_idx')['lottery']
    else:
        decile_val = (
            lottery_df[lottery_df['preference_number'] <= 5]
            .groupby('student_idx')['lottery'].min()
        )

    records = []
    for i, sid in enumerate(student_ids):
        matched_idx = int(matches[i])
        ranking = student_rankings[i]
        rank_pos = None
        if matched_idx >= 0:
            try:
                rank_pos = ranking.index(matched_idx) + 1
            except ValueError:
                pass
        records.append({
            'matched': matched_idx >= 0,
            'rank_pos': rank_pos,
            'decile_val': decile_val.get(i, np.nan),
        })
    df = pd.DataFrame(records).dropna(subset=['decile_val'])
    df['decile'] = pd.qcut(df['decile_val'], N_DECILES, labels=range(1, N_DECILES + 1))
    return df


def decile_metrics(df):
    rows = []
    for d in range(1, N_DECILES + 1):
        sub = df[df['decile'] == d]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            'decile': d,
            'pct_unmatched': 100.0 * (~sub['matched']).sum() / n,
            'top1_pct': 100.0 * (sub['rank_pos'] == 1).sum() / n,
            'top5_pct': 100.0 * (sub['rank_pos'] <= 5).sum() / n,
        })
    return pd.DataFrame(rows).set_index('decile')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    parser.add_argument('--capacity', required=True)
    parser.add_argument('--n_runs', type=int, default=10,
                         help='Independent lottery draws averaged per condition.')
    parser.add_argument('--output', default='fig4_chile_lottery_decile.png')
    parser.add_argument('--seed', type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

    print("Loading data...")
    indv_df = load_df(args.individual)
    capacity_df = load_df(args.capacity)
    applications_long = build_applications_long(indv_df)
    school_table = _prepare_school_capacity_table(capacity_df)
    all_student_ids = sorted(applications_long['mrun'].unique().tolist())
    print(f"  Students: {len(all_student_ids):,}   Schools: {len(school_table):,}")

    rng = np.random.default_rng(args.seed)

    metrics_by_condition = {'STB': [], 'MTB': []}
    for run in range(args.n_runs):
        print(f"Run {run + 1}/{args.n_runs}...")

        student_lottery = {sid: float(rng.random()) for sid in all_student_ids}
        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=student_lottery, return_lottery_df=True,
        )
        df = compute_student_records(student_ids, rankings, matches, lottery_df, 'single')
        metrics_by_condition['STB'].append(decile_metrics(df))

        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=None, return_lottery_df=True,
        )
        df = compute_student_records(student_ids, rankings, matches, lottery_df, 'best_top5')
        metrics_by_condition['MTB'].append(decile_metrics(df))

    avg = {
        cond: pd.concat(metrics_by_condition[cond]).groupby(level=0).mean()
        for cond in ['STB', 'MTB']
    }

    # ── plot ──────────────────────────────────────────────────────────────
    COLORS = {'pct_unmatched': '#111111', 'top1_pct': '#1565C0', 'top5_pct': '#AD1457'}
    LABELS = {'pct_unmatched': 'Unmatched', 'top1_pct': 'Top-1', 'top5_pct': 'Top-5'}
    STYLES = {'STB': '-', 'MTB': '--'}
    MARKERS = {'pct_unmatched': 'o', 'top1_pct': 's', 'top5_pct': '^'}

    deciles = list(range(1, N_DECILES + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in ['STB', 'MTB']:
        for metric in ['pct_unmatched', 'top1_pct', 'top5_pct']:
            ax.plot(deciles, avg[cond][metric].reindex(deciles), color=COLORS[metric],
                    linestyle=STYLES[cond], marker=MARKERS[metric], markersize=6,
                    linewidth=1.8, label=f'{LABELS[metric]} ({cond})')

    ax.set_xlabel('Lottery Decile', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_xticks(deciles)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    metric_handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='-', label=LABELS[m])
        for m in ['pct_unmatched', 'top1_pct', 'top5_pct']
    ]
    style_handles = [plt.Line2D([], [], color='gray', linestyle=STYLES[c], label=c) for c in ['STB', 'MTB']]
    leg1 = ax.legend(handles=metric_handles, loc='upper left', fontsize=10, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='upper right', fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
