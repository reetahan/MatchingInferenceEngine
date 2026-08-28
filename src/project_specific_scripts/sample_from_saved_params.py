"""
Standalone preference sampling from a saved mixture-of-Mallows params pickle
(produced by chilean_experiment_driver.py / nyc_experiment_driver.py / driver.py
as `<log>_params.pkl`).

Usage:
    python sample_from_saved_params.py \
        --params_pkl path/to/run_params.pkl \
        --district Santiago \
        --n_students 5000 \
        --k_ranking_length 10 \
        --output preferences.csv \
        --seed 42

    # Sample every district found in the params file:
    python sample_from_saved_params.py --params_pkl path/to/run_params.pkl \
        --n_students 5000 --output preferences.csv
"""

import argparse
import pickle

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mallows import sample_students_global_mixture


def sample_district(params, district, n_students, k_ranking_length, n_jobs, seed):
    rankings_idx = sample_students_global_mixture(
        params,
        district,
        n_students,
        n_jobs=n_jobs,
        random_seed=seed,
        k_ranking_length=k_ranking_length,
    )
    schools = params['districts'][district]['schools']
    rows = []
    for student_id, ranking in enumerate(rankings_idx):
        row = {'student_id': student_id, 'district': district}
        for rank_pos, idx in enumerate(ranking[:k_ranking_length], start=1):
            row[f'choice_{rank_pos}'] = schools[idx]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_pkl', required=True)
    parser.add_argument('--district', default=None,
                         help='District to sample. Omit to sample all districts in the params file.')
    parser.add_argument('--n_students', type=int, required=True,
                         help='Students to sample per district.')
    parser.add_argument('--k_ranking_length', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', default='sampled_preferences.csv')
    args = parser.parse_args()

    with open(args.params_pkl, 'rb') as f:
        params = pickle.load(f)

    districts = [args.district] if args.district else sorted(params['districts'].keys())
    print(f"Sampling {args.n_students} students/district for {len(districts)} district(s)...")

    all_rows = []
    for district in districts:
        all_rows.extend(
            sample_district(
                params, district, args.n_students,
                args.k_ranking_length, args.n_jobs, args.seed,
            )
        )
        print(f"  {district}: done")

    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f"Saved {len(all_rows)} students to {args.output}")


if __name__ == '__main__':
    main()
