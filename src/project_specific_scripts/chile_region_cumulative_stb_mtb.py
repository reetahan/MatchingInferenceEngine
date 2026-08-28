"""
Fig. 8 style plot: cumulative assignment outcomes (share matched to top choice,
one of top 5, or any listed school) under MTB and STB, broken out by Chilean
Region.

MTB uses a single real-lottery run (matches the observed mechanism, same
convention as chilean_real_welfare_comparison.py). STB is averaged over
--n_stb_runs independent counterfactual single-draw-per-student runs.

Region labels use Chile's official Roman-numeral region codes (I-XVI,
including Nuble as XVI). The mapping is keyed on the exact 'Region' string
values found in the real individual-level data (see
sample-data/data/chilean_data_processed/chile_priority_config.json's
region_overrides keys, which are produced directly from
indv_df.groupby('Region') in real_chile_priority_generator.py).

Usage:
    python chile_region_cumulative_stb_mtb.py \
        --individual <path_to_indv_df> \
        --capacity   <path_to_capacity_df> \
        --n_stb_runs 10 \
        --output fig8_chile_region_cumulative.png \
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

ROMAN_BY_REGION = {
    'Región de Tarapacá':                              'I',
    'Región de Antofagasta':                            'II',
    'Región de Atacama':                                'III',
    'Región de Coquimbo':                               'IV',
    'Región de Valparaíso':                             'V',
    "Región del Libertador Bernardo O'Higgins":         'VI',
    'Región del Maule':                                 'VII',
    'Región del Bío-Bío':                                'VIII',
    'Región de La Araucanía':                            'IX',
    'Región de Los Lagos':                               'X',
    'Región de Aysén del Gral.Ibañez del Campo':         'XI',
    'Región de Magallanes y Antártica Chilena':          'XII',
    'Región Metropolitana de Santiago':                  'XIII',
    'Región de Los Ríos':                                'XIV',
    'Región de Arica y Parinacota':                      'XV',
    'Región de Ñuble':                                   'XVI',
}
ROMAN_ORDER = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
               'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']


def build_student_region(indv_df: pd.DataFrame) -> pd.Series:
    """Per-student Region (Roman numeral), indexed by mrun string."""
    region_by_mrun = (
        indv_df[['mrun', 'Region']]
        .drop_duplicates(subset='mrun')
        .assign(mrun=lambda x: x['mrun'].astype(str))
        .set_index('mrun')['Region']
    )
    unmapped = set(region_by_mrun.unique()) - set(ROMAN_BY_REGION)
    if unmapped:
        raise ValueError(f"Unrecognized Region value(s), update ROMAN_BY_REGION: {unmapped}")
    return region_by_mrun.map(ROMAN_BY_REGION)


def compute_region_buckets(student_ids, student_rankings, matches, student_region):
    """Per-region % top-1 / % top-2-5 / % top-6+ (all matched students, uncond over all)."""
    records = []
    for i, sid in enumerate(student_ids):
        region = student_region.get(sid)
        if region is None:
            continue
        matched_idx = int(matches[i])
        rank_pos = None
        if matched_idx >= 0:
            ranking = student_rankings[i]
            try:
                rank_pos = ranking.index(matched_idx) + 1
            except ValueError:
                pass
        records.append({'region': region, 'rank_pos': rank_pos})

    df = pd.DataFrame(records)
    rows = []
    for region, sub in df.groupby('region'):
        n = len(sub)
        rows.append({
            'region': region,
            'top1':    100.0 * (sub['rank_pos'] == 1).sum() / n,
            'top2_5':  100.0 * sub['rank_pos'].apply(lambda r: r is not None and 2 <= r <= 5).sum() / n,
            'top6p':   100.0 * sub['rank_pos'].apply(lambda r: r is not None and r >= 6).sum() / n,
        })
    return pd.DataFrame(rows).set_index('region')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    parser.add_argument('--capacity', required=True)
    parser.add_argument('--n_stb_runs', type=int, default=10)
    parser.add_argument('--output', default='fig8_chile_region_cumulative.png')
    parser.add_argument('--seed', type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

    print("Loading data...")
    indv_df = load_df(args.individual)
    capacity_df = load_df(args.capacity)
    applications_long = build_applications_long(indv_df)
    student_region = build_student_region(indv_df)
    school_table = _prepare_school_capacity_table(capacity_df)
    all_student_ids = sorted(applications_long['mrun'].unique().tolist())

    rng = np.random.default_rng(args.seed)

    print("Running MTB matching (real priority, per-school lottery)...")
    student_ids, rankings, matches = run_matching(applications_long, school_table, rng, student_lottery=None)
    mtb_buckets = compute_region_buckets(student_ids, rankings, matches, student_region)

    print(f"Running {args.n_stb_runs} STB counterfactual runs...")
    stb_runs = []
    for run in range(args.n_stb_runs):
        print(f"  STB run {run + 1}/{args.n_stb_runs}...")
        student_lottery = {sid: float(rng.random()) for sid in all_student_ids}
        student_ids, rankings, matches = run_matching(
            applications_long, school_table, rng, student_lottery=student_lottery
        )
        stb_runs.append(compute_region_buckets(student_ids, rankings, matches, student_region))
    stb_buckets = pd.concat(stb_runs).groupby(level=0).mean()

    regions = [r for r in ROMAN_ORDER if r in mtb_buckets.index and r in stb_buckets.index]

    # ── plot ──────────────────────────────────────────────────────────────
    STB_COLORS = {'top1': '#1a2f6e', 'top2_5': '#3f6fd6', 'top6p': '#a9c0f0'}
    MTB_COLORS = {'top1': '#7a2a10', 'top2_5': '#c1652f', 'top6p': '#f0c19b'}

    x = np.arange(len(regions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5))

    for offset, buckets, colors, prefix in [
        (-width / 2, stb_buckets, STB_COLORS, 'STB'),
        (width / 2, mtb_buckets, MTB_COLORS, 'MTB'),
    ]:
        bottoms = np.zeros(len(regions))
        for seg, seg_label in [('top1', 'Top 1'), ('top2_5', 'Top 2-5'), ('top6p', 'Top 6+')]:
            vals = np.array([buckets.loc[r, seg] for r in regions])
            ax.bar(x + offset, vals, width, bottom=bottoms, color=colors[seg],
                   label=f'{prefix} - {seg_label}')
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(ncol=6, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
