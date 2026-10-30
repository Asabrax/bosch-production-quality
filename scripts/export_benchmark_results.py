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
    print('Exported aggregate benchmark receipt and existing notebook figure.')


if __name__ == '__main__':
    main()
