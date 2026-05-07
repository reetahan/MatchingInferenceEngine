import pandas as pd
import matplotlib.pyplot as plt
import os

BOROUGH_NAMES  = {'M': 'Manhattan', 'X': 'Bronx', 'K': 'Brooklyn',
                  'Q': 'Queens',    'R': 'Staten Island'}
BOROUGH_COLORS = {'M': "#035388", 'X': "#3fdff8", 'K': '#27ae60',
                  'Q': '#8e44ad', 'R': '#f39c12'}

df = pd.read_csv('sweep_summary.csv')
borough_df = pd.read_csv('sweep_borough.csv')

max_p = borough_df['p'].max()
b_matched = borough_df[borough_df['p'] == max_p].copy()
b_matched['pct_matched'] = b_matched['top_p_pct']
b_stats = borough_df[borough_df['p'] == 1].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Overall lines
axes[0].plot(df['list_length_min'], df['pct_matched'],
             marker='o', color='black', linewidth=1.5, label='Overall', zorder=5)
axes[1].plot(df['list_length_min'], df['avg_rank'],
             marker='o', color='black', linewidth=1.5, label='Overall', zorder=5)
axes[2].plot(df['list_length_min'], df['rank_variance'],
             marker='o', color='black', linewidth=1.5, label='Overall', zorder=5)

# Borough lines
for borough in ['M', 'X', 'K', 'Q', 'R']:
    color = BOROUGH_COLORS[borough]
    label = BOROUGH_NAMES[borough]

    b_m = b_matched[b_matched['borough'] == borough].sort_values('list_length_min')
    b_s = b_stats[b_stats['borough'] == borough].sort_values('list_length_min')

    if not b_m.empty:
        axes[0].plot(b_m['list_length_min'], b_m['pct_matched'],
                     marker='o', color=color, linewidth=1.5, linestyle='--', label=label)
    if not b_s.empty:
        axes[1].plot(b_s['list_length_min'], b_s['avg_rank'],
                     marker='o', color=color, linewidth=1.5, linestyle='--', label=label)
        axes[2].plot(b_s['list_length_min'], b_s['rank_variance'],
                     marker='o', color=color, linewidth=1.5, linestyle='--', label=label)

for ax in axes:
    ax.relim()
    ax.autoscale_view()

axes[0].set_xlabel('Minimum list length')
axes[0].set_ylabel('% Matched')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].set_xlabel('Minimum list length')
axes[1].set_ylabel('Average rank')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)

axes[2].set_xlabel('Minimum list length')
axes[2].set_ylabel('Rank variance')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('min_list_length_sweep.png', dpi=200, bbox_inches='tight')
print("Saved.")