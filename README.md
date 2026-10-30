# Bosch Production Quality: Data Cleaning and Defect Prediction

An industrial data-science portfolio project about **what missing sensor measurements reveal**, how to preserve that information, and how to evaluate defect screening under severe class imbalance.

The project refactors an existing Bosch investigation rather than replacing it. Start with [explore.ipynb](explore.ipynb); explore the separate [date-data extension](date_analysis.ipynb) for join quality, timestamp diagnostics and chronological validation; the unmodified source notebook and its historical outputs are preserved in [archive/explore_original.ipynb](archive/explore_original.ipynb). The [audit](docs/audit.md) explains which decisions were retained and which were corrected.

## Business question

Can numeric sensor measurements and their availability help prioritize manufactured parts for inspection? `Response=1` denotes a defective part. An illustrative operating policy inspects the highest-risk 1% of parts. Costs and a confirmed factory inspection budget are unavailable, so this project reports detection performance rather than invented profit.

## What the investigation found

The original saved full-data outputs describe 1,183,747 parts, 968 numeric sensors, 50 stations, and 6,879 defective parts (about 0.58%). Mean sensor missingness was 81.08%.

- **Missingness has structure.** Selected sensors within a station have identical missing-value masks. Two S0/S1 sensors have very similar missing rates but disagree on 1,642 parts. Similar percentages do not imply identical missingness.
- **Station coverage differs from individual-sensor coverage.** S24 has at least one measurement for about 15.52% of parts, while many of its sensors are much sparser. Availability is a production-route proxy, not proof of a station visit or a causal explanation of defects.
- **Sparse features can carry information.** Do not automatically discard sensors with more than 90%, 95% or 99% missingness. No blanket `dropna()` or global mean imputation is applied.
- **Extremes are investigated, not automatically deleted.** The original S32 sensor investigation found 105 IQR flags among 24,543 recorded values. Without process knowledge, those values could be genuine manufacturing behavior.
- **Feature and model selection need separation from evaluation.** Historical station choices used full-data target comparisons; the old test set was examined repeatedly. A new random split cannot make those observations independent again.

These findings come from archived saved outputs. The refactor has not independently rerun the complete 1.18-million-row analysis.

## Method and retained implementations

The notebook follows eight sections: Business Understanding, Data Understanding, Data Cleaning and Preparation, Exploratory Data Analysis, Feature Engineering, Modeling, Model Evaluation, Conclusions and Limitations. It draws on the local Data Science lectures 4–12; see the audit for references.

The original float32 sensor / int64 ID / int8 label policy is retained. Feature generation builds a compact separate table containing total measurement count, all station counts, selected sensor values, sensor-presence indicators, and the original `S32_only` rule:

```python
S32_only = (L3_S32_count > 0) & (L3_S33_count == 0)
```

`S32_only` is generated for analysis and excluded from every default model candidate, as in the archived modeling code. An explicitly named optional ablation can include it. It does **not** mean that no other station was visited.

The experiment retains the original stratified split sequence (80/20, followed by 80/20 within development: approximately **64/16/20** overall), seed 42, Decision Tree baseline, and balanced HistGradientBoosting models. The HGB grid preserves leaf limits 7/15/31 and learning rates 0.05/0.1. Training-only feature choices compare station counts, total-count addition, high-coverage sensors, focus-station sensor values, and presence indicators.

The important corrections are:

1. Rank focus stations using training-only target relationships; select sensor coverage on training rows. The historical fixed S32/S33/S34/S24 list is retained as provenance, not silently reused as an unbiased selection.
2. Fit the tree's median imputer inside its training pipeline. HGB retains `NaN` values. Preserve missing indicators rather than erase measurement availability.
3. Use one frozen feature schema for training, validation, evaluation and unlabeled inference. Reject missing/extra raw sensor columns.
4. Select the candidate by validation AP and the illustrative score threshold by validation F1. Keep the selected fitted model rather than changing it through a final refit after threshold selection. HGB automatic early stopping is disabled for an explicit 100-iteration experiment.
5. Evaluate the selected configuration on a partition explicitly labeled **exploratory evaluation**, with no claim of an untouched holdout. Top-capacity rounding and ties are explicit: `ceil(n × capacity)`, stable input order.

These are deliberate methodological changes, so new results need not match the historical scores.

## Historical exploratory results

The following values are preserved from the archived notebook, **not recalculated or presented as new final results**. Validation AP appears in original zero-based cell 49; test results appear in cell 55.

| Metric | Historical value (rounded) |
|---|---:|
| Validation Average Precision | 5.42% |
| Previously inspected test Average Precision | 4.35% |
| Test precision at highest-risk 1% | 8.79% |
| Test recall at highest-risk 1% | 15.12% |

That experiment used 50 station counts plus 15 focus-station sensors; `S32_only` was excluded. The final HGB was refit on the outer 80% training split. Historical test inspection selected 2,367 parts and found 208 of 1,376 defects. Full-data exploratory station selection and prior test inspection limit what these scores establish.

The refactored notebook includes AP, prevalence baseline, precision, recall, both threshold/capacity confusion matrices, validation precision–recall, and evaluation gains/lift curves. Scores from a bounded smoke run are software-verification output only, not representative portfolio performance estimates.

## Project structure

```text
Bosch/
├── explore.ipynb                 # Refactored, documented analysis
├── bosch.py                      # Loading, features, splitting, selection, metrics, inference
├── date_analysis.ipynb           # Separate date-quality and chronological experiment
├── bosch_dates.py                # Chunked summaries, join audit and temporal splits
├── tests/test_bosch.py           # Lightweight checks using synthetic data
├── docs/audit.md                 # Leakage audit, preservation map, verification
├── docs/date_analysis.md         # Date-extension design and reproduction
├── archive/explore_original.ipynb # Original notebook, unchanged
├── requirements.txt             # Direct dependencies, pinned to verified versions
├── requirements.lock            # Installed transitive versions (Python 3.14 environment)
├── data/                        # Local competition files; ignored
└── artifacts/                   # Per-run manifests/models/split IDs; ignored
```

## Reproduce

Commands below run from the `Bosch` directory. Verification used Python 3.14.4 on Linux. With `uv` installed:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock
source .venv/bin/activate
python -m pytest -q
```

Alternatively, create a virtual environment with a Python installation that provides `venv`/`pip`, then run `python -m pip install -r requirements.lock`. The lock records tested versions; it is not a cross-platform compatibility guarantee.

Obtain `train_numeric.csv` from the [Bosch Production Line Performance competition data page](https://www.kaggle.com/competitions/bosch-production-line-performance/data) using your own Kaggle account and complying with its access terms. Extract it locally to `data/train_numeric.csv`, place it at `/data/train_numeric.csv`, or set:

```bash
export BOSCH_DATA_PATH=/absolute/path/to/train_numeric.csv
```

An explicit path wins; otherwise `/data/train_numeric.csv` is checked before project-local `data/train_numeric.csv`. This workspace uses the project-local path. `train_date.csv` is not consumed by the original numeric notebook; the separate `date_analysis.ipynb` now uses it. No dataset is included for redistribution.

Run a bounded end-to-end check (first 20,000 rows; **not random sampling**):

```bash
BOSCH_NROWS=20000 python -m jupyter nbconvert \
  --execute --to notebook --inplace explore.ipynb \
  --ExecutePreprocessor.timeout=600
```

Run the full experiment explicitly:

```bash
BOSCH_NROWS=0 python -m jupyter nbconvert \
  --execute --to notebook --inplace explore.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Open the notebook in VS Code and select `.venv/bin/python` to work interactively. Restart the kernel and run all cells after changing parameters. `BOSCH_NROWS` defaults to `0` (full data). Each run prints its actual scope; the saved notebook delivered with this refactor contains a clearly labeled 20,000-row smoke execution.

**Memory:** the original loaded frame used about 4.28 GiB; that is not peak process memory. CSV parsing, selected-column slices and model fitting need additional RAM. The code avoids full-frame copies and full sensor-sized missingness matrices, and limits model thread pools to four threads. Only compact feature matrices are split. The full grid contains 35 candidates by default, run sequentially. Close other large kernels before a full run in a 20 GB WSL allocation. Full-data peak memory/runtime have not been measured in this refactor.

Runs write unique subdirectories under `artifacts/` with a manifest, split IDs, validation comparisons, training station summaries, gains/lift figure and a fitted inference bundle. The manifest records code hashes, input path/size/mtime, loaded ID digest, versions, selected columns, model parameters, thresholds and new metrics. It does not claim a cryptographic checksum of all sensor values. Preserve the source CSV and environment for reproduction.

For future unlabeled numeric data (same 968-sensor header):

```python
import joblib
from bosch import load_numeric, predict_unseen

bundle = joblib.load('artifacts/<run-id>/model.joblib')  # load only trusted files
future_raw, _ = load_numeric('/path/to/unlabeled_numeric.csv')
scores = predict_unseen(bundle, future_raw)
```

The same helper can score successive CSV chunks with the same header to limit inference memory. Do not silently replace absent sensor columns with missing values. Scores are not calibrated defect probabilities, particularly for balanced class weights.

## Limitations and verification

Full-data exploration, prior test reuse, unknown process timing, potential batch/time dependence, and few positive examples constrain generalization claims. The focus-station selector is a simple exploratory univariate rule that can favor rare groups; missingness differences alone do not identify causal mechanisms. No cost-benefit claim, uncertainty interval, deployment readiness claim, or genuinely independent performance estimate is made.

The refactor passed lightweight tests and a top-to-bottom 20,000-row notebook run. Full-dataset execution remains unverified. Review [docs/audit.md](docs/audit.md), confirm the intended screening point with manufacturing knowledge, and freeze the entire workflow before evaluating genuinely new untouched data. Review ignored files before any GitHub publication; `.gitignore` does not untrack data already committed elsewhere.

## Date-data extension

[date_analysis.ipynb](date_analysis.ipynb) adds chunked timestamp-quality checks, a validated one-to-one ID join, recorded time ranges, signed station gaps, and a fixed random-versus-chronological HGB experiment. Chronological splitting keeps timestamp ties together and purges earlier records that overlap the next partition boundary. Unknown-time rows remain in the quality analysis and are explicitly excluded from chronology.

This is an exploratory end-of-record screening experiment, not an early-warning claim. Full-route timestamps and counts require a verified prediction point; supplied timestamp units and label-availability timing remain unconfirmed. The numeric baseline and historical results remain separate. See [the date-extension guide](docs/date_analysis.md) for execution commands, limitations and actual verification status.

The date extension has now completed a full-file run on **1,183,747 parts**: all IDs matched, 582 parts lacked recorded times, and 25,647 parts crossed the chosen temporal boundaries. The selected date-augmented model achieved exploratory AP of **6.60% on the random split versus 1.66% on the chronological split**. These are different evaluation populations, not a controlled estimate of improvement or leakage. The result strengthens the project's validation story: a favorable random split does not establish performance on later parts. Exact populations, selection rules and newly executed results are recorded in the date-extension guide.
