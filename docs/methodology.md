# Detailed methodology and original evaluation

See the [project summary](../README.md) for the current benchmark and uncertainty findings.

## Cleaning decisions

| Issue | Evidence | Treatment | Why / validation |
|---|---|---|---|
| Structured missingness | Shared within-station masks; differing route coverage | Retain NaNs and create per-part/station counts | Avoid deleting route information; do not equate missingness with zero |
| Extreme sensor values | IQR diagnostics flag unusual recorded values | Investigate and retain; no automatic clipping/removal | A statistical outlier is not proof of a measurement error |
| Identity and schema integrity | Unique-ID/schema checks; all full-file IDs match | Reject duplicates or unexpected schema; validate a one-to-one join | Prevent duplicated parts, silent schema changes and join inflation |
| Unmatched IDs | Zero in the full files; synthetic tests cover mismatches | Preserve numeric rows and report both unmatched directions | An inner join must not silently discard parts |
| Missing timestamps | 582 parts have no recorded dates | Keep unknown ranges; exclude only from chronology | Unknown time must not be replaced by time zero |
| Temporal overlap | 23,179 train and 2,468 validation parts cross the next boundary | Save IDs/reasons and exclude from temporal fitting/validation | Keep recorded observation ranges before the next partition |
| Learned preprocessing and selection | Historical focus stations used all labels | Select focus stations on train only; fit tree imputation inside a pipeline | Keep validation/evaluation labels out of fitting and supervised selection |

Raw measurements, legitimate zeros, sparse sensors and all-missing parts are preserved. HGB uses NaNs directly; tree comparisons involving sensor values use training-fitted median imputation and missing indicators. [Detailed audit](../docs/audit.md).


## Model evaluation

The operational example is inspecting the highest-risk **1% of parts**. This is an illustrative capacity, not a verified factory requirement. AP evaluates the ranking; precision and recall at capacity describe the inspection workload and the defects detected. Scores are not calibrated probabilities, and costs are unavailable.

These are **newly executed full-file experiments**, with different feature sets, selection procedures and evaluation populations:

| Exploratory evaluation | Parts / defects | Prevalence | AP | Precision at ~1% | Recall at ~1% |
|---|---:|---:|---:|---:|---:|
| Numeric notebook — selected HGB | 236,750 / 1,376 | 0.5812% | **4.8740%** | 8.7838% | 15.1163% |
| Date extension — random, common cohort | 231,504 / 1,364 | 0.5892% | **6.6006%** | 10.5354% | 17.8886% |
| Date extension — chronological, overlap-purged | 236,634 / 964 | 0.4074% | **1.6582%** | 4.7740% | 11.7220% |

The numeric run selected 50 station counts plus 14 sensor values from training-selected stations, with HGB using 7 leaves and learning rate 0.1. Validation AP was 5.6112%; selection preceded evaluation. At 1% capacity, it inspected 2,368 parts and found 208 defects.

The date extension compares a fixed HGB on numeric counts versus counts plus five global date summaries, selecting the feature set on validation AP separately for each split strategy. Both selected the date-augmented candidate. The chronological validation difference was small and has no uncertainty interval; this is not proof of a stable date-feature benefit.

![Full-run gains and lift curves for the date experiment; random-split gains exceed chronological gains, but the evaluation populations differ.](../docs/figures/temporal_evaluation.png)

*Different evaluation populations: random prevalence 0.5892%, chronological prevalence 0.4074%. At approximately 1% capacity, chronological evaluation detects 113 of 964 defects among 2,367 inspected parts. Prior exploration, unknown label timing and different populations limit interpretation. [Exact aggregate results and run hashes](../docs/results.json).*

**Historical reference only:** the archived notebook reported 5.42% validation AP and 4.35% test AP with a different configuration. Those historical experiments were not rerun and are not substituted for the results above. The [unchanged original notebook](../archive/explore_original.ipynb) and [audit](../docs/audit.md) preserve their provenance.


## Method

The numeric notebook follows eight CRISP-DM sections, from business understanding through limitations. It retains the Decision Tree and HGB model families, the original seed-42 stratified 64/16/20 split sequence, and the small HGB parameter grid. Five declared feature sets produce 35 candidates; validation AP chooses the model. The full run selected focus stations S32/S33/S34/S31 using training labels only, replacing the historical handpicked S32/S33/S34/S24 list.

Total measurement count, all station counts, selected values and sensor-presence indicators are generated consistently. The original station-combination definition is preserved:

```python
S32_only = ((L3_S32_count > 0) & (L3_S33_count == 0)).astype('uint8')
```

It means numeric S32 recorded and numeric S33 absent, not absence of every other station. It is generated for analysis and excluded from all default model candidates, consistently with the original saved models.

The date extension streams wide inputs in chunks, joins on unique IDs, keeps timestamp ties together, and purges earlier parts whose recorded ranges cross the next temporal boundary. Its random comparator uses the same retained cohort. Dates remain in their supplied numeric units; earliest/latest time and their difference are **recorded ranges**, not certified factory processing durations. [Date design and results](../docs/date_analysis.md).
