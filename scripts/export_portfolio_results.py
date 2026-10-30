"""Publish aggregate run receipts and existing notebook figures, without raw rows.

Run only after both full notebooks finish. Figures are extracted, not regenerated.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def checked_notebook(filename, manifest):
    path = ROOT / filename
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if [cell.execution_count for cell in code] != list(range(1, len(code) + 1)):
        raise ValueError(f'{filename}: execution counters are not sequential.')
    if any(output.output_type == 'error' for cell in code for output in cell.outputs):
        raise ValueError(f'{filename}: saved error output.')
    streams = '\n'.join(output.text for cell in code for output in cell.outputs
                        if output.output_type == 'stream')
    if manifest['run_id'] not in streams:
        raise ValueError(f'{filename}: outputs do not identify the specified run.')
    source = [{'cell_type': cell.cell_type, 'source': cell.source} for cell in notebook.cells]
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if digest != manifest['notebook_source_sha256']:
        raise ValueError(f'{filename}: source differs from the executed manifest.')
    for filename, expected in manifest['code_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Code changed since execution: {filename}')
    return notebook


def find_cell(notebook, fragment):
    matches = [(i, cell) for i, cell in enumerate(notebook.cells) if fragment in cell.source]
    if len(matches) != 1:
        raise ValueError(f'Expected one cell containing {fragment!r}; found {len(matches)}.')
    return matches[0]


def extract_figure(notebook, fragment, destination):
    index, cell = find_cell(notebook, fragment)
    outputs = [output.data['image/png'] for output in cell.outputs
               if 'image/png' in output.get('data', {})]
    if len(outputs) != 1:
        raise ValueError(f'Expected one PNG in source cell {index}.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(outputs[0]))
    return {'source_cell_zero_based': index,
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numeric-run', required=True, type=Path)
    parser.add_argument('--date-run', required=True, type=Path)
    args = parser.parse_args()
    numeric = json.loads((args.numeric_run / 'manifest.json').read_text())
    dates = json.loads((args.date_run / 'manifest.json').read_text())
    if numeric['requested_nrows'] is not None or dates['requested_nrows'] is not None:
        raise ValueError('Portfolio results require full-file runs, not prefix smoke tests.')
    if numeric['loaded_rows'] != dates['join_audit']['numeric_rows']:
        raise ValueError('Full-run numeric populations differ.')
    if numeric['source']['bytes'] != dates['sources']['numeric']['bytes']:
        raise ValueError('Numeric input file sizes differ between runs.')
    if numeric['source']['mtime_ns'] != dates['sources']['numeric']['mtime_ns']:
        raise ValueError('Numeric input modification times differ between runs.')
    numeric_notebook = checked_notebook('explore.ipynb', numeric)
    date_notebook = checked_notebook('date_analysis.ipynb', dates)
    _, profile = find_cell(numeric_notebook, 'missing_pct = missingness_profile(')
    profile_text = '\n'.join(output.get('data', {}).get('text/plain', '')
                             for output in profile.outputs)
    mean_match = re.search(r'\bmean\s+([\d.]+)', profile_text)
    _, masks = find_cell(numeric_notebook, 'pattern_rows = []')
    mask_text = '\n'.join(output.get('data', {}).get('text/plain', '')
                          for output in masks.outputs)
    mask_match = re.search(r'L0_S0_F2\s+L0_S1_F24\s+(?:True|False)\s+(\d+)', mask_text)
    if not mean_match or not mask_match:
        raise ValueError('Cannot recover the reported missingness statistics from saved tables.')

    figure_dir = ROOT / 'docs/figures'
    figures = {
        'numeric_missingness.png': {
            'notebook': 'explore.ipynb', 'run_id': numeric['run_id'],
            **extract_figure(numeric_notebook, 'missing_pct = missingness_profile(',
                             figure_dir / 'numeric_missingness.png')},
        'temporal_evaluation.png': {
            'notebook': 'date_analysis.ipynb', 'run_id': dates['run_id'],
            **extract_figure(date_notebook, "fraction, gain, lift = gains_curve(parts['Response']",
                             figure_dir / 'temporal_evaluation.png')},
    }
    # Publish aggregate evidence only: no split IDs, raw rows, local paths or models.
    numeric_keys = ['run_id', 'scope', 'loaded_rows', 'seed', 'python', 'versions',
                    'split_summary', 'selected_model', 'selected_feature_set',
                    'training_selected_focus_stations', 'model_columns',
                    'validation_average_precision', 'exploratory_evaluation_average_precision',
                    'exploratory_evaluation_capacity', 'exploratory_evaluation_threshold',
                    'chosen_threshold', 'primary_capacity', 'independent_holdout',
                    'code_sha256', 'notebook_source_sha256']
    date_keys = ['run_id', 'scope', 'seed', 'python', 'versions', 'join_audit', 'date_quality',
                 'temporal_boundaries', 'exclusions', 'split_summary', 'model_parameters',
                 'validation_results', 'exploratory_evaluation_results', 'independent_holdout',
                 'operational_feature_availability_verified', 'time_units',
                 'code_sha256', 'notebook_source_sha256']
    receipt = {
        'status': 'Executed full-file exploratory experiments; not independent validation.',
        'numeric': {key: numeric[key] for key in numeric_keys},
        'date_extension': {key: dates[key] for key in date_keys},
        'numeric_quality': {
            'mean_sensor_missing_percent_as_displayed': float(mean_match.group(1)),
            'S0_F2_vs_S1_F24_missing_mask_disagreements': int(mask_match.group(1)),
        },
        'figures': figures,
    }
    (ROOT / 'docs/results.json').write_text(json.dumps(receipt, indent=2) + '\n')
    lines = ['# README figure provenance', '',
             'These PNGs are extracted unchanged from saved full-run notebook outputs.',
             'Source cells use zero-based indexes. Aggregates and run hashes are in `../results.json`.', '']
    for name, info in figures.items():
        lines.extend([f'## {name}', '', f"- Notebook: `{info['notebook']}`",
                      f"- Source cell: {info['source_cell_zero_based']}",
                      f"- Run: `{info['run_id']}`", f"- PNG SHA-256: `{info['sha256']}`", ''])
    lines.extend(['## Refresh after rerunning both notebooks', '', '```bash',
                  'python scripts/export_portfolio_results.py \\',
                  '  --numeric-run artifacts/<numeric-run-id> \\',
                  '  --date-run artifacts/<date-run-id>', '```', '',
                  'Review and update the README and interpretation whenever newly executed values change.',
                  'The script rejects prefix runs, stale source hashes and mismatched run outputs.', ''])
    (figure_dir / 'README.md').write_text('\n'.join(lines))
    print('Exported two existing figures and docs/results.json from verified full runs.')


if __name__ == '__main__':
    main()
