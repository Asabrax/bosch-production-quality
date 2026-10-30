# Preservation, leakage audit and verification

Audit date: 2026-10-29. Cell references below are **zero-based** in `archive/explore_original.ipynb`.

## Source inspected

The first on-disk inspection found a 33-cell notebook ending at measurement-count bins. Before modification, the file had updated to a 56-cell version; that complete version was copied to the archive and then inspected. It contains the original model code and saved outputs, including the reported 5.42%/4.35% AP reference values. One active Bosch project notebook was found; additional versions exist in VS Code history (see recovery search below). The archived original is never executed or edited by the refactor. Saved historical outputs are not independently verified or reproduced metrics.

### Follow-up recovery search

At the user's request, searched project notebooks (including hidden checkpoint names) and available VS Code local-history locations. The applicable history index is `/home/alex/.vscode-server/data/User/History/-16fab7a4/entries.json`; its resource identifies `/home/alex/projects/Bosch/explore.ipynb` in WSL Ubuntu.

| VS Code history file | Cells | Saved modeling code |
|---|---:|---|
| `Ze8i.ipynb` | 1 | No |
| `OD8U.ipynb` | 33 | No |
| `fN9V.ipynb` | 33 | No |
| `2SeO.ipynb` | 33 | No |
| `TYaK.ipynb` | 56 | Decision Tree, HGB, sensor features, tuning, precision–recall, inspection capacity |

`TYaK.ipynb` is **byte-for-byte identical** to `archive/explore_original.ipynb`, with SHA-256 `071c3d5f9d449789bf3b795798ddb4e16bf7dad854a0d9076d1d3ce524a4eb24`. Thus the archive preserves an independently located saved copy of the modeling code; a replacement modeling notebook was unnecessary. No additional Bosch modeling notebook was found in the project search. The checked VS Code backup directories were absent; this search cannot recover unsaved in-memory kernel state or establish that no copy exists elsewhere.

The user's confirmed formula exactly matches original cells 36–37: `S32_only` is a uint8 indicator for S32 having a measurement and S33 having none. The archived crosstab lists 5,553 nondefective and 1,014 defective parts in this group, totaling 6,567. These are historical reference counts corroborated by saved output, **not a new full-data calculation**. The current feature helper and tests retain this definition. Supervised station selection remains restricted to training rows; the historical feature idea and previously inspected test data still prevent an independent-holdout claim.

The CSV exists at `data/train_numeric.csv`; `/data/train_numeric.csv` was absent in this environment. The companion `train_date.csv` was left unused. No original datasets were moved, rewritten or added to version control.

## Preservation map

| Original cells | Retained analysis or implementation | Refactor treatment |
|---|---|---|
| 0, 2 | float32 sensors; int64 Id; int8 Response; dimensions/memory | Reusable loader, explicit path resolution and schema checks |
| 3–13, 19 | Missingness masks/crosstabs within and across stations | Consolidated comparisons, no duplicated exploratory print cells |
| 14–22 | Station discovery, coverage, S24 sparsity, missingness thresholds | Reused station-name parsing and per-column counts; no wholesale sparse-feature deletion |
| 23–25 | Target-conditioned availability differences | Same diagnostic calculation, restricted to training rows |
| 26–30 | S32 distribution, IQR flags, boxplots and target table | Preserved diagnostics and plots on training; no clipping/removal |
| 31–32 | Total measurement count and failure-rate bins | Separate uint16 feature table; final bin extends to infinity so no larger count is omitted |
| 33–36 | 50 station counts, station failure rates, S32/S33 combination | Same count/combination logic; new target comparisons train-only |
| 37 | `S32_only = s32 & ~s33` | Exact definition retained; not “S32 and no other station” |
| 38–43 | Stratified splits, tree baseline, thresholds, HGB baseline | Retained model families, seed, outer/inner split proportions; consolidated imports/selection |
| 44 | Precision–recall curve | Retained for selected validation model, with prevalence baseline |
| 45–47 | 20 training-coverage-selected sensors | Preserved selection on training rows, consistently transformed elsewhere |
| 48–51 | Focus-station sensor values/presence; train-only station ranking | Focus stations now chosen by train-only ranking instead of hardcoded full-data-informed list |
| 52 | HGB leaves 7/15/31 × learning rate .05/.1 grid | Preserved grid, balanced weights, max_iter=100; explicit early_stopping=False |
| 53–55 | Capacity comparisons, final evaluation | Validation capacities retained; one selected model evaluated once; explicit rounding/ties; no independent-holdout claim |

The original baseline uses 50 station counts. Total measurement count is retained as a separately named ablation. Sensor-value and presence comparisons use the same selected focus columns. The original saved plots and outputs remain inspectable in the archive; the main notebook's plots are newly executed and labeled by scope.

## Leakage findings

### Confirmed full-data feature-selection exposure

Cells 24–25 rank sensor availability by full-data target differences. Cell 35 ranks station failure-rate differences on the full dataset; cells 36–37 construct the S32/S33 hypothesis after looking at all labels. The focus station list in cell 48 (`L3_S32`, `L3_S33`, `L3_S34`, `L1_S24`) follows that investigation. Selecting the highest-coverage sensors within those stations on `X_fit` does not erase the earlier full-data station choice.

**Correction:** split before all new target-based analysis. Rank candidate focus stations by absolute present-minus-absent failure rates on training rows only, then choose up to five highest-coverage sensors within each of the top four stations on the same training rows. Keep the top-20 coverage alternative. This changes the station rule and can change selected features/results; it is not presented as an exact reproduction of historical scores. The selector has no minimum subgroup support threshold: inspect training denominators and treat rare-station rankings cautiously.

### S32_only audit: consistent exclusion, not a discovered inconsistency

Cell 37 creates `S32_only`. Cell 39 explicitly drops it from `features`. Cell 40 creates `X_fit`/`X_val` from that reduced feature list. Later HGB additions build on `X_fit`/`X_val`. Final cell 55 uses `X_train[features]`/`X_test[features]`. Therefore, assuming sequential execution of the saved code, **all original fitted model experiments exclude S32_only**.

There is no evidence of the suspected inclusion/exclusion bug in the saved code. The original notebook does have incomplete execution counters, so arbitrary historical interactive state cannot be proven from the file alone. The refactor declares feature sets explicitly, records an inclusion flag for every candidate and stores the winning columns. The original definition is generated for EDA but excluded from all default candidates; an optional named ablation is visibly different.

### Repeated test inspection

The outer 20% test partition appears in the initial tree evaluation (39), thresholded tree evaluation (41), and final HGB evaluation (55). Decisions continued after earlier test inspection. The project brief also confirms prior test exposure.

**Correction:** model and threshold decisions use training/validation only; the selected configuration receives one subsequent exploratory evaluation. Historical exposure cannot be undone by a new seed, a new split, or code cleanup. Independent evidence requires new untouched observations after freezing the whole procedure.

### Direct target and preprocessing leakage

No direct use of `Response` or `Id` as a model predictor was found in the archived model matrices. Per-row counts are deterministic and may be computed before splitting without target leakage; the risky step was choosing which routes/sensors to use based on all labels. No fitted global imputation/scaling was found in the saved original. HGB handled raw sensor NaNs.

The refactor excludes labels/IDs from features, preserves sensor NaNs, fits median imputation only within the tree pipeline, and checks inference schema. Station/measurement counts are deterministic per row and do not learn from evaluation rows. The new imputer is needed for sensor-valued tree comparisons; it does not impute the entire raw dataset.

### Operational leakage remains unresolved

A sensor/route flag might arise after the desired screening point or reflect rework/inspection triggered by defects. The numeric file alone does not establish when measurements were available. No claim is made that station counts can be deployed before all stations are complete. Production-timing and process review are required.

## Intentional differences from historical modeling

- The original stratified split sequence and seed are retained, with sorted positional arrays used consistently afterward. Exact split IDs are saved per new run.
- The focus-station selection is now training-only and may select S31 or other stations rather than the old list.
- The original tree parameters and HGB search ranges are retained. The refactor compares the declared configurations on shared validation rows rather than preserving repeated scratch experiments.
- HGB early stopping is explicitly disabled. The original `early_stopping='auto'` could reserve an internal validation fraction; the new fixed 100-iteration procedure may yield different scores.
- No train+validation refit is performed after choosing the validation threshold. This keeps the threshold attached to the same fitted estimator; the historical final HGB was refit on all outer training rows.
- AP selects the model; validation F1 across a declared grid selects an illustrative threshold. F1 is not a factory cost function. Capacity 1% is declared before evaluation.
- The original top-k code used floor rounding and default `argsort` tie behavior; the new code uses ceiling (at most one additional part) and stable input order. Both the requested and actual inspection fractions are recorded.
- Counts use uint16 and selected sensors float32. Raw sensor values, all-missing parts and sparse features are preserved. Entirely missing training sensors stay in the imputer schema.

## Lecture alignment

Materials were read locally under `../data_aai_sose26/`. Relevant PDF sections and examples were inspected; these course files are not copied into the portfolio repository.

- Lecture 4, `lecture04_missing_values/l04-missing_values.pdf` and the Pima missing-values solution: investigate missingness and justify treatment from context.
- Lecture 5, `lecture05_outliers/l05-outliers.pdf` and `l05e2-solution-Iris_Dataset_Outliers_Merging.ipynb`: separate outlier detection from treatment; retain diagnostic boxplots.
- Lecture 6, `lecture06_visual_EDA/l06-visual_EDA.pdf` and visualization demo: univariate and target-conditioned visual exploration.
- Lecture 7, `lecture07_feature_engineering/l07-feature_engineering.pdf` and Ames feature-engineering solution: reusable transformations; train-only learned feature steps.
- Lecture 8, `lecture08_modelling_part_1/l08-modelling_part_1.pdf` and train/test fitting demo: complexity control and train/validation/evaluation separation.
- Lecture 9, `lecture09_modelling_part_2/l09-modelling_part_2.pdf` and modeling-pipeline solution: leakage and fitted preprocessing inside the training procedure. Nested CV is discussed as future work, not claimed to have run.
- Lectures 10–11, evaluation PDFs and threshold/profit demos: distinguish score thresholds from inspection decisions; do not estimate profit without costs or treat tuning scores as untainted evaluation.
- Lecture 12, `lecture12_evaluation_part_3/l12-evaluation_part_3.pdf` and gains/lift demo: plot positive-class recall against the inspected population fraction, and lift against random ranking.

The eight requested notebook sections adapt CRISP-DM to this project; they are not a claim that deployment was performed.

## Verification record

See the final verification details appended below. Checks use bounded source rows and synthetic fixtures; archived historical metrics are preserved as evidence, not rerun. Full-data execution and full-data peak memory/runtime remain unverified. Running all notebook cells with `BOSCH_NROWS=0` is the next reproducibility check, after inspecting the new selection specification.

Final checks performed:

- `python -m pytest -q`: **11 passed**. Tests cover exact counts, zero-measurement rows, S32/S33 semantics, dtype preservation, chunk/reordered/unlabeled schema equivalence, missing/extra-column rejection, training-only sensor and station selection, split disjointness/coverage/stratification, exact capacity/confusion/gains calculations, model-selection isolation, and saved-bundle inference including presence features.
- `python -m compileall -q bosch.py tests`; all notebook code cells also parsed successfully. `nbformat.validate` passed.
- The notebook executed top-to-bottom with `BOSCH_NROWS=20000`: 13 code cells, execution counters 1–13, no error outputs, all 35 candidate fits. The 20,000-row prefix contained 112 positive examples; the exploratory evaluation partition contained only 22 positives. These are smoke-test scope facts, not full-data performance estimates.
- Final run artifacts: `artifacts/20261029T125911739576Z/`. Source-file and normalized notebook-source hashes were checked against the saved manifest. Evaluation confusion-matrix totals reconcile to 4,000 rows and the 1% policy flags exactly 40 rows. Every default candidate excludes S32_only.
- Seven rendered chart images were extracted from the executed notebook and visually inspected for labels, units, ranges, legends and clipping, including a full-size inspection of gains/lift. The station-coverage plot includes all 50 station labels. An HTML preview was exported to ignored `artifacts/notebook_preview.html`; the full HTML page was not browser-inspected. The exporter reports missing image alt text; chart explanations and labeled tables remain in the notebook, but accessible image descriptions are a possible presentation improvement.
- Archived original SHA-256: `071c3d5f9d449789bf3b795798ddb4e16bf7dad854a0d9076d1d3ce524a4eb24` (56 cells). The original outputs are historical evidence only; they were not re-executed.
- At the initial refactor verification, no Git repository had been initialized or commit made. `.gitignore` excludes local CSVs, environments, model bundles, split IDs and run artifacts from subsequent publication.

Manual review priorities: the training-only focus-station rule versus the historical list; the operational timing/meaning of S32/S33 and other station measurements; whether 1% inspection and F1 threshold selection match the actual business objective; and a full-data run before publishing any new performance result. A genuinely independent validation cohort is still required for generalization claims.

## Separate date extension

A later authorized extension added `date_analysis.ipynb`, `bosch_dates.py` and `tests/test_dates.py`. It completed a full-file join/timestamp-quality scan and fixed chronological-versus-random experiment; see `docs/date_analysis.md` for definitions, actual findings and limitations. The original numeric notebook, its helpers and the archived notebook were unchanged (hash-checked). The project's combined test suite now has 19 passing tests. Full execution of the date extension does not retroactively verify the original numeric notebook's complete experiment or its historical metrics.
