# README figure provenance

These PNGs are extracted unchanged from saved full-run notebook outputs.
Source cells use zero-based indexes. Aggregates and run hashes are in `../results.json`.

## numeric_missingness.png

- Notebook: `explore.ipynb`
- Source cell: 7
- Run: `20261030T212935626972Z`
- PNG SHA-256: `5e47d09afbb7f4a11589b9ff98f17cf26a3249bb0c8fec63215c4cf91f55b603`

## temporal_evaluation.png

- Notebook: `date_analysis.ipynb`
- Source cell: 16
- Run: `dates_20261029T140103667151Z`
- PNG SHA-256: `f70486c0b1a1931692b5007fc78c7dc7dc20e05989af0a3d0083ff64317f6fbb`

## Refresh after rerunning both notebooks

```bash
python scripts/export_portfolio_results.py \
  --numeric-run artifacts/<numeric-run-id> \
  --date-run artifacts/<date-run-id>
```

Review and update the README and interpretation whenever newly executed values change.
The script rejects prefix runs, stale source hashes and mismatched run outputs.

## Benchmark uncertainty figure

`benchmark_uncertainty.png` is extracted unchanged from `benchmark_analysis.ipynb`, zero-based cell 11. Run: `benchmark_20261030T212935628830Z`. PNG SHA256: `5ca28e0d6f919877ae67f67b2870e04149db4855722da5fe4006265dc1a47895`.

Refresh with `python scripts/export_benchmark_results.py --run artifacts/<benchmark-run-id>`.
