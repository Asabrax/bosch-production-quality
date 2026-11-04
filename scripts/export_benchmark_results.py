"""Export the completed benchmark's aggregate receipt and saved notebook figure."""
import argparse
import json
from export_portfolio_results import ROOT, checked_notebook, extract_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    receipt = json.loads((run / 'manifest.json').read_text())
    if receipt['scope'] != 'full files; saved date partitions':
        raise ValueError('Require the full benchmark.')
    notebook = checked_notebook('benchmark_analysis.ipynb', receipt)
    receipt['figure'] = extract_figure(
        notebook, "primary = intervals[", ROOT / 'docs/figures/benchmark_uncertainty.png')
    (ROOT / 'docs/benchmark_results.json').write_text(json.dumps(receipt, indent=2) + '\n')
    capacity_rows = receipt['capacity_results']
    lines = ['# Inspection capacity and defect recall', '',
             'The original balanced chronological HGB is held fixed. These are workload scenarios, not model improvements or a newly selected operating policy. Inspect the highest-ranked ceil(n × capacity) parts using the existing stable tie rule.', '',
             'Use validation data and an actual inspection budget to develop a policy. The evaluation table is retrospective and exploratory; do not choose a new capacity from it. The previously reported uncertainty intervals cover 1% only.', '']
    for partition, label in [('validation', 'Validation'), ('exploratory_evaluation', 'Exploratory evaluation')]:
        subset = [row for row in capacity_rows if row['partition'] == partition]
        lines.extend([f'## {label}', '',
                      f"{subset[0]['parts']:,} parts; {subset[0]['defects']:,} defects.", '',
                      '| Capacity | Inspected | Defects found | False alarms | Defects missed | Precision | Recall |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for row in subset:
            lines.append(f"| {row['capacity_requested']:.0%} | {row['inspected']:,} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} | {row['precision']:.2%} | {row['recall']:.2%} |")
        lines.append('')
    lines.extend(['## Interpretation', '',
                  'Increasing capacity expands the inspected set and can recover more defects, at the cost of more inspections and false alarms. It does not improve ranking quality. Precision is not guaranteed to move monotonically. Costs, available staffing, and the consequences of missed defects are unknown, so no capacity is recommended here.', '',
                  'To improve recall at a fixed capacity, develop better features or compare candidates on validation recall at that predeclared capacity; retain training-only feature selection. Confirmation still requires an untouched future cohort.', '',
                  f"Source run: `{receipt['run_id']}`. See the [executed notebook](../benchmark_analysis.ipynb), [aggregate receipt](benchmark_results.json), and [reproduction instructions](reproduction.md).", ''])
    (ROOT / 'docs/capacity.md').write_text('\n'.join(lines))
    provenance = ROOT / 'docs/figures/README.md'
    text = provenance.read_text().split('## Benchmark uncertainty figure')[0].rstrip()
    text += (f"\n\n## Benchmark uncertainty figure\n\n"
             f"`benchmark_uncertainty.png` is extracted unchanged from `benchmark_analysis.ipynb`, zero-based cell {receipt['figure']['source_cell_zero_based']}. "
             f"Run: `{receipt['run_id']}`. PNG SHA256: `{receipt['figure']['sha256']}`.\n\n"
             "Refresh with `python scripts/export_benchmark_results.py --run artifacts/<benchmark-run-id>`.\n")
    provenance.write_text(text)
    print('Exported benchmark receipt, existing figure and capacity tables.')


if __name__ == '__main__':
    main()
