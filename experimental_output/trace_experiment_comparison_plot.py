
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRIC_PATTERN = re.compile(
    r'^\s*(top\d+|unmatched):\s*'
    r'obs=([\d.]+)%\s+sim=([\d.]+)%'
)
EXPERIMENT_PATTERN = re.compile(r'^EXPERIMENT:\s*(.+?)\s*$', re.MULTILINE)
ANALYSIS_MARKER = '-- Match Rate Analysis'
SEPARATOR = '-----------------------------------------------'
OBSERVED_COLOR = '#0a17d1'
PALETTE = ('#19d308', '#e67e22', '#8e44ad', '#16a085', '#d62728', '#9467bd')


@dataclass
class Experiment:
    name: str
    overall: dict
    districts: dict


def _parse_metric_lines(lines):
    metrics = {}
    for line in lines:
        match = METRIC_PATTERN.match(line)
        if match:
            metric, observed, simulated = match.groups()
            metrics[metric] = {
                'obs': float(observed),
                'sim': float(simulated),
            }
    return metrics


def _parse_report(report):
    lines = report.splitlines()
    overall = {}
    districts = {}
    overall_start = next(
        (index for index, line in enumerate(lines) if line.strip() == 'Overall:'),
        None,
    )
    district_start = next(
        (index for index, line in enumerate(lines) if line.strip() == 'Per district:'),
        None,
    )
    if overall_start is None or district_start is None:
        return overall, districts

    overall = _parse_metric_lines(lines[overall_start + 1:district_start])
    current_district = None
    for line in lines[district_start + 1:]:
        district_match = re.match(r'^\s{2}(.+):\s*$', line)
        if district_match:
            current_district = district_match.group(1).strip()
            districts[current_district] = {}
            continue
        metric_match = METRIC_PATTERN.match(line)
        if metric_match and current_district is not None:
            metric, observed, simulated = metric_match.groups()
            districts[current_district][metric] = {
                'obs': float(observed),
                'sim': float(simulated),
            }
    return overall, districts


def parse_log(path):
    """Return the experiment reports contained in ``path``.

    The explicitly delimited Match Rate Analysis report is preferred when a
    block contains one; otherwise the block's ordinary report is used.
    """
    text = Path(path).read_text()
    matches = list(EXPERIMENT_PATTERN.finditer(text))
    experiments = []
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():block_end]
        if ANALYSIS_MARKER in block:
            report = block.split(ANALYSIS_MARKER, 1)[1]
            report = report.split(SEPARATOR, 1)[0]
        else:
            report = block
        overall, districts = _parse_report(report)
        if overall and districts:
            experiments.append(Experiment(match.group(1).strip(), overall, districts))
    return experiments


def _metric_order(experiments):
    metric_sets = [set(experiment.overall) for experiment in experiments]
    common_metrics = set.intersection(*metric_sets)
    top_metrics = sorted(
        (metric for metric in common_metrics if metric.startswith('top')),
        key=lambda metric: int(metric[3:]),
    )
    if 'unmatched' in common_metrics:
        top_metrics.append('matched')
    return top_metrics


def select_districts(experiments, count=10):
    """Choose deterministic districts, prioritizing Santiago_* districts."""
    common = set.intersection(*(set(experiment.districts) for experiment in experiments))
    common.discard('Santiago')
    santiago_districts = sorted(district for district in common if district.startswith('Santiago_'))
    other_districts = sorted(district for district in common if not district.startswith('Santiago_'))

    santiago_count = min(len(santiago_districts), max(3, count // 2))
    selected = santiago_districts[:santiago_count]
    selected.extend(other_districts[:count - len(selected)])
    if len(selected) < count:
        remaining = [district for district in santiago_districts[santiago_count:] if district not in selected]
        selected.extend(remaining[:count - len(selected)])
    return selected[:count]


def _plot_comparison(metric_values, title, output_path, ylabel='Match rate (%)'):
    metrics = list(metric_values)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    observed = [metric_values[metric]['obs'] for metric in metrics]
    ax.plot(x, observed, marker='o', linewidth=2,
            color=OBSERVED_COLOR, label='Observed')
    simulated_by_experiment = {}
    for experiment_index, name in enumerate(metric_values[metrics[0]]['experiments']):
        simulated = [metric_values[metric]['experiments'][name] for metric in metrics]
        simulated_by_experiment[name] = simulated
        ax.plot(x, simulated, marker='o', linewidth=2,
                color=PALETTE[experiment_index % len(PALETTE)], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlabel('Rank cutoff (p)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plotted_values = observed + [value for values in simulated_by_experiment.values() for value in values]
    lower = max(0, np.floor(min(plotted_values) - 5))
    upper = min(100, np.ceil(max(plotted_values) + 5))
    if upper - lower < 10:
        midpoint = (lower + upper) / 2
        lower = max(0, midpoint - 5)
        upper = min(100, midpoint + 5)
    ax.set_ylim(lower, upper)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_difference(metric_values, experiment_one, experiment_two, title, output_path):
    metrics = list(metric_values)
    differences = [
        metric_values[metric]['experiments'][experiment_two]
        - metric_values[metric]['experiments'][experiment_one]
        for metric in metrics
    ]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, differences, marker='o', linewidth=2, color='#d62728')
    ax.axhline(0, color='black', linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlabel('Rank cutoff (p)')
    ax.set_ylabel('Simulated difference (percentage points)')
    ax.set_title(title)
    limit = max(1, np.ceil(max(abs(value) for value in differences) + 2))
    ax.set_ylim(-limit, limit)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _comparison_values(experiments, districts, metrics):
    def value(record, metric, field):
        if metric == 'matched':
            return 100 - record['unmatched'][field]
        return record[metric][field]

    values_by_district = {}
    for district in [None, *districts]:
        values = {}
        for metric in metrics:
            observed = [
                value(experiment.overall if district is None else experiment.districts[district], metric, 'obs')
                for experiment in experiments
            ]
            simulated = {
                experiment.name: value(
                    experiment.overall if district is None else experiment.districts[district],
                    metric,
                    'sim',
                )
                for experiment in experiments
            }
            values[metric] = {'obs': observed[0], 'experiments': simulated}
        values_by_district[district] = values
    return values_by_district


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', nargs='+', required=True, help='One or more experiment log files.')
    parser.add_argument('--out', required=True, help='Directory for generated plots.')
    args = parser.parse_args()

    experiments = [experiment for log in args.log for experiment in parse_log(log)]
    if not experiments:
        raise SystemExit('No complete EXPERIMENT blocks found.')
    if len(experiments) < 2:
        raise SystemExit('At least two experiments are required for a difference plot.')
    metrics = _metric_order(experiments)
    if not metrics:
        raise SystemExit('No common top-p metrics found across experiment blocks.')

    districts = select_districts(experiments)
    if len(districts) < 10:
        raise SystemExit(f'Only {len(districts)} common districts are available; need at least 10.')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = _comparison_values(experiments, districts, metrics)
    _plot_comparison(values[None], 'Overall top-p comparison', out_dir / 'overall_top_p_comparison.png')
    _plot_difference(
        values[None],
        experiments[0].name,
        experiments[1].name,
        f'Overall top-p difference: {experiments[1].name} - {experiments[0].name}',
        out_dir / 'overall_top_p_difference_experiment_2_minus_1.png',
    )
    for district in districts:
        filename = re.sub(r'[^A-Za-z0-9_.-]+', '_', district).strip('_')
        _plot_comparison(values[district], f'{district} top-p comparison', out_dir / f'{filename}_top_p_comparison.png')
    (out_dir / 'selected_districts.txt').write_text('\n'.join(districts) + '\n')
    print(f'Parsed {len(experiments)} experiments: {", ".join(experiment.name for experiment in experiments)}')
    print(f'Plotted {", ".join(metrics)} for overall and districts: {", ".join(districts)}')
    print(f'Saved plots to {out_dir}')


if __name__ == '__main__':
    main()