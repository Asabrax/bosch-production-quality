# Date-data extension

This is a separate second phase of the Bosch project. Open [date_analysis.ipynb](../date_analysis.ipynb). The original `explore.ipynb`, archived source notebook and numeric helpers remain unchanged.

## Questions and scope

1. Do numeric/date files identify the same parts, with a safe one-to-one join?
2. Which timestamps are missing, zero, negative or repeated within a station?
3. What do earliest/latest recorded timestamps, recorded spans and selected signed interstation gaps show?
4. How does a fixed end-of-record screening experiment behave with a random split versus a purged chronological split?

Timestamps are treated as finite float64 values in the supplied dataset coordinate system. A physical time unit or calendar origin is not established. A recorded span is not a certified processing duration, and numeric station order is not a verified production sequence.

## Memory and data integrity

`bosch_dates.py` streams each CSV in 10,000-row chunks. Numeric measurements use float32; dates use float64 to preserve ordering/ties more accurately. After each chunk, only compact per-part summaries and aggregate profiles survive. The full 1,156-column date matrix is never retained alongside the full numeric matrix. No source CSV is modified.

The date summaries preserve earliest/latest recorded time, total recorded timestamp count, number of stations with date records, and first/last times for four historically discussed stations. Numeric counts reuse the existing implementation. The four station-specific date summaries are for EDA only; they are not used as predictors.

Duplicate headers, duplicate IDs (including across chunks), missing IDs, malformed labels and infinite values fail explicitly. The one-to-one left join preserves all numeric rows. The audit separately reports numeric-only IDs, date-only IDs and matched rows with no recorded timestamps. Unmatched ID lists are saved under ignored artifacts. Prefix execution audits only those loaded prefixes, not the complete CSVs.

Zero is a recorded value. Negative timestamps are counted for review but not automatically deleted. All-missing timestamp rows keep unknown first/last/span values and zero counts. The numeric definition of `S32_only` is preserved: S32 numeric count > 0 and S33 numeric count = 0. Date presence never silently replaces numeric presence.

## Temporal validation design

Cut points come from earliest recorded timestamps at approximate 64% and 80% positions, without using target labels. Tied timestamps stay in the same interval, so actual proportions can differ. Unknown dates are quarantined from the chronology experiment, not dropped from the quality table.

Training parts must have their latest recorded timestamp strictly before validation starts. Validation parts must finish strictly before evaluation starts. Boundary-overlap exclusions are saved by ID and reason. This purge prevents recorded observation ranges from crossing the next boundary. It does not establish defect-label availability or remove unknown batch/product dependence.

The random comparator uses exactly the retained chronological cohort. Its split is stratified using the existing seed-42 64/16/20 procedure. Random and chronological evaluation contain different parts and can have different defect prevalence. Their AP/precision/recall differences are descriptive; they do not isolate the causal effect of chronology, leakage or adding dates.

## Fixed model comparison

For each strategy, compare two declared feature sets on validation AP:

- 50 numeric station counts plus total numeric measurement count.
- Those same counts plus earliest recorded time, latest recorded time, recorded span, date-record count and date-station count.

Use HGB with 15 leaves, learning rate 0.1, 100 iterations, balanced weights, no automatic early stopping and seed 42. There is no new hyperparameter search. Per-part feature construction is deterministic; HGB handles NaNs. IDs, targets, merge indicators, S32_only and the four station-specific diagnostic timestamps are excluded from all candidates.

Select the feature set using validation AP (numeric-only wins ties), then evaluate only that selected fitted model once for each strategy. Report AP, evaluation prevalence, precision, recall, confusion counts and lift at a prespecified highest-risk 1% capacity, with ceiling rounding and stable input-order ties. Include gains/lift charts. Do not select a new model from evaluation outcomes.

These are **new exploratory experiments**. The historical scores from the original notebook are neither reused nor claimed as comparable benchmarks. A date candidate winning on validation is not proof of a stable performance gain.

## Prediction-time limitation

This is retrospective **end-of-record screening**. Earliest/latest times, the final span and full-route counts use all observed records. They are not valid inputs for an earlier prediction unless computed from an explicit as-of snapshot. Even at end of record, process review must establish that inputs exist before the defect outcome is known and do not reflect outcome-triggered inspection or rework.

Chronological splitting does not undo prior full-data exploration. Truly independent evidence requires new untouched parts, frozen feature/selection rules, a declared prediction event, label-maturity information and relevant grouping constraints.

## Reproduction

Use the existing environment and dependencies; no additional packages are required. Run from the Bosch directory:

```bash
source .venv/bin/activate
python -m pytest -q

# Software verification only: prefixes of both files, not a random sample.
BOSCH_DATE_NROWS=20000 python -m jupyter nbconvert \
  --execute --to notebook --inplace date_analysis.ipynb \
  --ExecutePreprocessor.timeout=600

# Full files: chunked source scans and four fixed model fits.
BOSCH_DATE_NROWS=0 python -m jupyter nbconvert \
  --execute --to notebook --inplace date_analysis.ipynb \
  --ExecutePreprocessor.timeout=-1
```

`BOSCH_DATA_PATH` selects the numeric file using the existing path resolver. `BOSCH_DATE_PATH` selects the date file; by default it sits beside the resolved numeric CSV. `BOSCH_DATE_CHUNKSIZE` defaults to 10000. Full-file execution is the notebook default (`BOSCH_DATE_NROWS=0`).

Every run writes an ignored `artifacts/dates_<timestamp>/` directory with source metadata, normalized notebook-source and helper hashes, schemas, quality profiles, unmatched IDs, split and exclusion IDs, model comparisons, selected model bundles and a gains/lift figure. Source size/mtime is not a full CSV content checksum. Keep the exact CSVs and pinned environment for reproduction; bundles/IDs are not intended for public redistribution.

For new data, `summarize_date_chunk` applies the same date transformation with a frozen header, and `bosch.build_features` reproduces numeric counts. Join by unique ID, validate the schema, then use the saved model column order. Artifacts remain exploratory rather than deployment-ready.

## Validation status

Verification findings and run scope are recorded below after execution. Successful prefix checks alone do not establish full-data results. The numeric baseline was subsequently executed in full on 2026-10-30; see `audit.md` for that separate run.

### Completed verification — 2026-10-29

- **19 tests passed** across the project (11 numeric and 8 date tests). New checks cover zero/missing times, chunk-size invariance, cross-chunk duplicate IDs, duplicate headers/infinities, both-sided join accounting, preserved row order, temporal ties, overlap purging, unknown-time quarantine, label-independent time boundaries, equal comparator cohorts, numeric target validation and model-selection isolation from evaluation labels.
- A 20,000-row prefix execution passed first. The date notebook then executed **top-to-bottom on both full files**, including all four model fits. Nine code cells completed in sequence, without error outputs. At this stage the separate numeric notebook had only a prefix check; its full run was later completed on 2026-10-30 and is recorded in `audit.md`.
- Full run: `artifacts/dates_20261029T140103667151Z/`. Notebook format, Python syntax, source hashes and normalized notebook-source hash were checked against the manifest. Stored split/exclusion IDs reconcile to all 1,183,747 parts with no overlap; the random and chronological strategies use exactly the same retained cohort. Evaluation confusion counts and precision/recall arithmetic were independently reconciled against the manifest counts.
- Three rendered notebook figures were visually inspected for labels, units, ranges, legends and clipping. The numeric notebook, numeric helper and archived original retained their pre-extension SHA-256 hashes. Full-page browser rendering was not performed.
- The retained joined frame reported approximately **0.225 GiB** of DataFrame memory. This is not peak process memory; chunk parsing, transformations and model fitting need additional memory. Process peak memory and formal runtime benchmarking were not measured.

### Full-file quality findings

| Check | Observed result |
|---|---:|
| Parts in each file / matched IDs | 1,183,747 |
| Numeric-only / date-only IDs | 0 / 0 |
| Date columns / stations represented in date names | 1,156 / 52 |
| Mean column timestamp missingness | 82.2436% |
| Parts with all timestamps missing | 582 (including 2 defective parts) |
| Parts with a zero recorded span | 120 |
| Recorded zero timestamp cells / negative timestamp cells | 12 / 0 |
| Observed timestamp range | 0.00–1718.48 dataset time units |
| Training records crossing validation boundary | 23,179 |
| Validation records crossing evaluation boundary | 2,468 |
| Retained common modeling cohort | 1,157,518 |

Date column names include L3_S42 and L3_S46, which are absent from the numeric sensor header. Therefore the global number of date-observed stations need not equal the number of numeric-observed stations. Neither measure proves a station visit without process documentation.

The chronological boundaries were 1046.36 and 1356.56 in supplied time units. All 582 unknown-time parts and 25,647 boundary-overlap parts remain in the quality analysis, while their IDs/reasons are saved outside the modeling cohort. There were no negative S33-first-minus-S32-last gaps among 17,976 parts with both stations' recorded dates; this observation does not establish a universal route order. The numeric-defined S32_only count was 6,567 in this scan; historical model metrics were not reused.

The chronological partitions also differ in defect prevalence: **0.7196% training, 0.3072% validation and 0.4074% evaluation**. That shift is relevant when interpreting performance; it is not by itself proof of sensor or concept drift.

### Newly executed exploratory model results

Validation AP selected the date-augmented candidate for both strategies:

| Validation strategy | Numeric counts AP | Numeric counts + dates AP |
|---|---:|---:|
| Random, common cohort | 4.5179% | 4.9885% |
| Chronological, overlap-purged | 5.1950% | 5.2426% |

The chronological validation difference is small and has no uncertainty interval; do not claim a proven date-feature benefit. Only the selected model for each strategy was evaluated afterward:

| Exploratory evaluation | Random, common cohort | Chronological, overlap-purged |
|---|---:|---:|
| Parts / defective parts | 231,504 / 1,364 | 236,634 / 964 |
| Prevalence (constant-score AP baseline) | 0.5892% | 0.4074% |
| Average Precision | 6.6006% | 1.6582% |
| Parts inspected at approximately 1% | 2,316 | 2,367 |
| Defects detected | 244 | 113 |
| Precision at inspection capacity | 10.5354% | 4.7740% |
| Recall at inspection capacity | 17.8886% | 11.7220% |

The later-time evaluation was substantially weaker in this run, despite a similar validation AP. This supports taking temporal evaluation seriously and avoiding a generalization claim based only on the random-split score. The two evaluation populations and their prevalence differ, and earlier historical exploration remains a limitation: **the gap is not an isolated estimate of leakage, nor an independently validated performance forecast**. No tuning was performed in response to these evaluation outcomes.

## Later class-weight benchmark

The [benchmark extension](benchmark.md) holds this experiment's selected features and saved partitions fixed, compares unweighted HGB, and adds paired time-bin uncertainty for chronological evaluation. It does not test the causal value of date features or replace the original validation selection.
