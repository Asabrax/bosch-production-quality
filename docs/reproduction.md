# Reproduction and inference

Run commands from the repository root. The verified local environment is Linux with Python 3.14.4. Direct dependencies are in `requirements.txt`; `requirements.lock` records transitive versions. The lock is a tested environment snapshot, not a guarantee for every platform.

## Environment

With `uv` installed:

```bash
uv venv --python 3.14.4 .venv
uv pip install --python .venv/bin/python -r requirements.lock
source .venv/bin/activate
python -m pytest -q
```

Alternatively, use a Python installation providing `venv` and `pip`, create/activate `.venv`, and install with `python -m pip install -r requirements.lock`. In VS Code, select `.venv/bin/python` as the notebook kernel. Restart the kernel and run all cells after changing parameters.

## Source files

Obtain `train_numeric.csv` and `train_date.csv` from the [Bosch competition data page](https://www.kaggle.com/competitions/bosch-production-line-performance/data) using your own Kaggle account and following the competition's access terms. Extract them into `data/`. No datasets are redistributed here.

`BOSCH_DATA_PATH` overrides the numeric path. Otherwise the loader checks `/data/train_numeric.csv` before repository-local `data/train_numeric.csv`. `BOSCH_DATE_PATH` overrides the date path; otherwise it is expected beside the numeric file.

```bash
export BOSCH_DATA_PATH=/absolute/path/to/train_numeric.csv
export BOSCH_DATE_PATH=/absolute/path/to/train_date.csv
```

## Notebook execution

For a bounded software check, use a prefix of 20,000 rows. This is not a random sample and its metrics are not representative performance estimates:

```bash
BOSCH_NROWS=20000 python -m jupyter nbconvert \
  --execute --to notebook --inplace explore.ipynb \
  --ExecutePreprocessor.timeout=600
BOSCH_DATE_NROWS=20000 python -m jupyter nbconvert \
  --execute --to notebook --inplace date_analysis.ipynb \
  --ExecutePreprocessor.timeout=600
```

Run each full notebook explicitly:

```bash
BOSCH_NROWS=0 python -m jupyter nbconvert \
  --execute --to notebook --inplace explore.ipynb \
  --ExecutePreprocessor.timeout=-1
BOSCH_DATE_NROWS=0 python -m jupyter nbconvert \
  --execute --to notebook --inplace date_analysis.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Both default to full-file execution when their row-limit variable is unset. The numeric notebook compares 35 declared candidates sequentially; the date notebook fits four fixed candidates. The notebooks print their actual run scope and write separate run directories. The archived original is retained as historical evidence and is not part of the reproducible execution path.

## Memory

The full numeric frame uses approximately 4.28 GiB, before CSV parsing and model-fitting overhead. Sensor values use float32, IDs int64, labels int8 and count features uint16. The numeric notebook avoids full-frame copies and splits only compact model features. Close other large kernels before a full run in a 20 GB WSL allocation.

The date notebook streams both wide inputs in 10,000-row chunks and retains compact summaries. Set `BOSCH_DATE_CHUNKSIZE` to adjust the chunk size. Dates remain float64 so distinct timestamps are not unnecessarily collapsed. The retained joined frame used approximately 0.225 GiB in the full date run. Neither figure is a process peak-memory measurement. Model thread pools are limited to four threads.

## Outputs and provenance

Each execution creates an ignored `artifacts/<run-id>/` directory containing a manifest, split IDs, model-selection tables, fitted model bundles and charts. Date runs also record unmatched/excluded IDs, missingness profiles and time boundaries. Do not accidentally redistribute source IDs or fitted bundles with the project.

Manifests record code hashes, normalized notebook-source hash, source paths/sizes/mtimes, environment versions, exact selected columns, parameters, selection policies and newly calculated metrics. Source file size/mtime and a loaded-ID digest are not a cryptographic checksum of every sensor value. Keep the original source files and environment when reproducing.

A compact publication receipt, `docs/results.json`, preserves aggregate results and run provenance without source rows or model bundles. README figures are extracted unchanged from saved notebook outputs. See `docs/figures/README.md` for their exact source cells and checksums.

## Numeric-model inference

```python
import joblib
from bosch import load_numeric, predict_unseen

bundle = joblib.load('artifacts/<numeric-run-id>/model.joblib')  # trusted files only
future_raw, _ = load_numeric('/path/to/unlabeled_numeric.csv')
scores = predict_unseen(bundle, future_raw)
```

The schema must contain the same numeric sensors. Column order may differ; missing/extra raw sensor columns are rejected rather than silently filled. For bounded inference, read CSV chunks with the same dtype policy and apply `predict_unseen` to each chunk. The saved schema includes selected sensor values/presence indicators and the model's ordered columns. Balanced-model scores are not calibrated defect probabilities.

For date-model inspection or future inference, `summarize_date_chunk` reproduces the date features, and `bosch.build_features` reproduces numeric counts. Join by unique ID, validate both saved headers and select the stored model columns. These models assume all included observations are available at the intended screening point; an earlier prediction requires constructing explicit as-of snapshots first.

## Automated versus scientific verification

GitHub Actions runs the synthetic tests and checks active notebook structure/code syntax on Python 3.14.4. It downloads Python dependencies but does not download competition files or rerun large experiments. A green workflow is evidence about those software checks, not proof of data validity, independent generalization, business value or deployment readiness.

## Class-weight benchmark and uncertainty

After a full date notebook run, point the benchmark to that run directory:

```bash
BOSCH_BENCHMARK_SOURCE=artifacts/<date-run-id> python -m jupyter nbconvert \
  --execute --to notebook --inplace benchmark_analysis.ipynb \
  --ExecutePreprocessor.timeout=-1
python scripts/export_benchmark_results.py --run artifacts/<benchmark-run-id>
```

The benchmark requires the source manifest, saved split IDs, and trusted fitted bundles. It verifies source file size/mtime, helper hashes, schema and split membership, and reproduces saved balanced-model scores. It streams the original full files again and trains two unweighted comparators. `BOSCH_DATA_PATH` and `BOSCH_DATE_PATH` can relocate unchanged source files. It deliberately rejects partial date runs. Run the date notebook first on a fresh checkout; ignored artifacts are not distributed.

Bootstrap settings are declared in the notebook: 1,000 paired draws at time-bin widths 5, 10, and 20; primary width 10; seed 42. Fixed-model intervals cover chronological evaluation only. Exact draws, part predictions, and model bundles stay in ignored `artifacts/`; only aggregates, the figure, and capacity tables are exported. Regenerate the written interpretation if rerunning changes the results.
