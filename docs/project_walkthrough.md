# Project walkthrough

## Problem and scope

The Bosch Production Line Performance dataset contains 1,183,747 manufactured parts, with about 0.58% labeled defective. The project evaluates whether recorded measurements can prioritize inspections. Inspecting the highest-risk 1% is an illustrative capacity assumption.

Predicting every part as good gives roughly 99.42% accuracy while finding no defects. Average precision (AP) measures ranking; precision and recall at a fixed inspection capacity describe workload and defect coverage.

## Data quality and features

About 81% of numeric sensor values are missing on average. Some sensors within a station share identical missing-value masks, but similar missing percentages do not imply identical coverage: two S0/S1 sensors disagree on 1,642 parts. Availability is therefore retained as a possible production-route signal.

Features include total measurement counts and station-level counts. Numeric values use float32; counts use small integer types. Date files are processed in chunks, with schema and ID checks. See the [cleaning decisions](methodology.md#cleaning-decisions).

`S32_only` means numeric measurements at S32 and none at S33; other stations may still have measurements. It is excluded from default model candidates because the hypothesis arose during full-data exploration. HGB retains NaNs; Decision Tree sensor-value comparisons use training-fitted median imputation and missingness indicators.

## Evaluation design

Early exploration used full-data defect relationships and inspected test results. Later supervised feature selection uses training rows, and model selection uses validation AP. The later evaluation remains exploratory because the dataset had already informed hypotheses.

Numeric and date files join one-to-one on all IDs. The 582 parts without timestamps remain in quality analysis but are excluded from chronological modeling. The date experiment also excludes 25,647 earlier records whose recorded time ranges cross the next split boundary. Equal first timestamps stay together.

Timestamp units, calendar origin and physical processing durations are not established. Features based on complete histories require a prediction point at which those measurements are available.

## Results

The selected date-augmented HGB achieves **6.60% AP with random splitting** and **1.66% AP on later-time evaluation**. At approximately 1% inspection capacity, the chronological model catches **113 of 964 defects among 2,367 inspections**: **4.77% precision and 11.72% recall**.

The populations have different defect prevalence: about 0.589% and 0.407%. The AP difference describes sensitivity to the evaluation setup; it does not isolate leakage or concept drift. Most defects remain undetected at 1% capacity.

The matched unweighted HGB scores **1.19% chronological AP** and catches **98 defects**, versus 1.66% and 113 for the balanced model. Paired time-bin bootstrap intervals include zero at all three tested widths. See [benchmark results](benchmark.md).

The [capacity analysis](capacity.md) compares inspection rates of 1%, 2%, 5% and 10%, reporting validation and exploratory evaluation separately. Higher inspection capacity can increase recall without changing the model.

## Verification and next steps

Automated checks cover synthetic features, temporal splits, metrics and notebook structure. They do not rerun the full dataset. The [reproduction instructions](../README.md) distinguish executed results from historical references.

Further evaluation requires a defined prediction point, measurement and label availability, an independent future cohort, and inspection/missed-defect costs. Detailed assumptions and evidence are in the [methodology](methodology.md) and [audit](audit.md).
