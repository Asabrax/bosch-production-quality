# Five-minute project walkthrough

Use this as a speaking outline, not a script to memorize. Open the README's uncertainty figure and the [detailed methodology](methodology.md) figures alongside it. The full numeric results and date experiment are separate runs; do not substitute historical scores for the executed results.

## 0:00–0:40 — Problem and scope

> I investigated manufacturing data from the Bosch Production Line Performance competition. Each row is a manufactured part, and the target identifies defects. There are about 1.18 million parts, but only about 0.58% are defective. The practical question is whether recorded measurements can help prioritize a limited inspection budget. I used a highest-risk 1% policy as an illustrative operating point, not a verified factory requirement.

Explain why accuracy is inadequate: predicting every part as good would achieve roughly 99.42% accuracy while finding no defects. AP assesses ranking across recall levels; precision and recall at capacity explain the inspection workload and coverage.

## 0:40–1:40 — The cleaning decision worth defending

> My first finding was that missing measurements were structured. About 81% of numeric sensor values were missing on average, and some sensors within a station shared identical missing-value masks. But similar missing percentages did not always mean the same parts were missing: two S0/S1 sensors disagreed on 1,642 parts. I treated measurement availability as a possible production-route signal, rather than automatically deleting incomplete rows or filling the whole dataset with means.

Show the missingness evidence and the [cleaning decision table](methodology.md#cleaning-decisions). Explain that zero is a recorded measurement, not a missing value. An IQR outlier flag identifies an unusual value; it does not prove an error. Without engineering evidence, retain extreme values and report the diagnostic.

## 1:40–2:30 — Features, memory and reproducibility

> I converted availability into total measurement count and station-level counts. The reusable transformation produces the same columns for training, validation and future data. Numeric sensors use float32, counts use small integer types, and the date extension reads wide files in chunks. I added explicit schema and ID checks so unexpected inputs fail visibly instead of silently changing the analysis.

Be ready to explain `S32_only`: at least one **numeric** measurement at S32, with no numeric measurements at S33. It does not mean that no other station was visited. The feature is preserved for analysis but excluded from default model candidates because its hypothesis arose during full-data exploration.

Discuss fitted preprocessing: HGB keeps NaNs; the Decision Tree sensor-value comparisons use a median imputer fitted only on training data, with missingness indicators. Counts are deterministic per row and do not use labels.

## 2:30–3:25 — Leakage audit and date checks

> Some original focus stations had been chosen after looking at full-data defect relationships, and the test set had already been inspected. I changed new supervised station selection to use training rows only and used validation AP for model decisions. I still call the later evaluation exploratory, because changing the code cannot erase earlier human exposure to the data.

Both source files join one-to-one on all 1,183,747 IDs, but 582 parts have no recorded timestamps. Preserve those parts in the quality analysis; exclude them only where chronology is required. The date experiment also excludes 25,647 earlier records whose recorded time ranges cross the next split boundary. Equal first timestamps stay together.

Use the term **recorded time span**, not factory processing duration. Timestamp units, calendar origin and the true sequence of stations are not established by the numeric file names.

## 3:25–4:25 — The most informative modeling result

> In the fixed date experiment, the selected date-augmented HGB achieved 6.60% AP under random splitting and 1.66% AP on the later-time evaluation. At approximately 1% inspection capacity, the chronological model found 113 of 964 defects among 2,367 inspected parts: about 4.77% precision and 11.72% recall. The chronological result was much weaker, so I would not rely on the random-split score as a forecast for future production.

Show the gains/lift figure. Clarify that the evaluation populations differ: their defect prevalence is about 0.589% versus 0.407%. The AP gap is descriptive, not a controlled measurement of leakage or proof of concept drift. Both scores are exploratory. The chronological AP is above its own prevalence baseline, but that alone does not establish commercial usefulness.

## 4:25–5:00 — Limits and next decision

> Before claiming deployment readiness, I would define the prediction point, confirm which measurements exist then, establish when defect labels become available, and evaluate the frozen workflow on genuinely new parts. Earliest/latest timestamps and full-route counts can contain future information for an early-warning use case. Inspection costs and missed-defect costs are also needed before deciding whether the model is useful.

Close with what the project demonstrates: justified data-quality decisions, consistent features, explicit evaluation populations, and reproducible software checks. The README links to all three executed notebooks, the audit and the synthetic-test workflow.

## Likely follow-up questions

| Question | Answer to practice |
|---|---|
| Why not drop features with 95% missingness? | Sparsity can identify a particular route or subprocess. Coverage alone is not evidence that a feature is useless. Retain raw data and learn any supervised selection on training rows. |
| Why not just impute everything? | Global filling can obscure availability patterns and leak fitted statistics across splits. Use model-appropriate, training-fitted preprocessing. |
| Does missingness prove MNAR? | No. The patterns motivate investigation, but these observations alone do not identify the formal missing-data mechanism. |
| Does the date experiment prove the random model leaked? | No. Time, prevalence, training populations and evaluation populations differ. The result shows sensitivity to temporal evaluation, not a causal decomposition. |
| Why does a model with only 1.66% AP matter? | The evaluation prevalence is about 0.407%, so ranking is better than a constant-score baseline. But operational usefulness requires capacity-specific outcomes and costs; most defects are still missed at 1% capacity. |
| Did adding dates definitely improve the model? | No. Dates won validation selection, but the chronological AP difference was small and no uncertainty interval was computed. Only the selected model was evaluated afterward. |
| Why exclude rows at temporal boundaries? | A training part whose last record occurs in the validation period uses observations from beyond that boundary. The purge is explicit, and those rows remain in the quality analysis. |
| Is the evaluation truly independent? | No. The original dataset informed exploratory hypotheses. A genuinely untouched future cohort is still required. |
| What does CI verify? | Synthetic feature/split/metric tests and notebook schema/syntax. It does not execute the large dataset or certify scientific conclusions. |

## Benchmark follow-up

The unweighted HGB changes only class weighting, keeping features and splits fixed. Chronological AP is 1.19% versus 1.66% for the balanced model; at 1% capacity it finds 98 versus 113 defects. Paired time-bin bootstrap intervals for the differences include zero at all three tested widths. Explain why a higher point estimate does not establish an advantage, and why fixed-model evaluation intervals omit retraining and future drift. See the [benchmark analysis](benchmark.md) for exact intervals and assumptions.

## Inspection-capacity follow-up

Use the [capacity table](capacity.md) to explain how inspecting 1%, 2%, 5%, or 10% changes defect coverage and false alarms. Validation and exploratory evaluation are shown separately. More inspections can improve recall without improving the model itself. An operating policy needs a real budget and should be chosen using validation data before independent evaluation.

## Before the interview

Run the tests yourself, trace one synthetic part through `build_features`, and explain the temporal split on a handful of timestamps. Know which results are newly executed, which are historical references, and which assumptions remain unverified. Describe your exploratory work and the refactoring assistance accurately; claim understanding only for code and decisions you can explain.

A concise CV description:

> Investigated structured sensor missingness and timestamp quality across 1.18 million manufactured parts; built reproducible defect-screening workflows with training-only feature selection, temporal overlap checks and automated synthetic tests.
