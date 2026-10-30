# Inspection capacity and defect recall

The original balanced chronological HGB is held fixed. These are workload scenarios, not model improvements or a newly selected operating policy. Inspect the highest-ranked ceil(n × capacity) parts using the existing stable tie rule.

Use validation data and an actual inspection budget to develop a policy. The evaluation table is retrospective and exploratory; do not choose a new capacity from it. The previously reported uncertainty intervals cover 1% only.

## Validation

186,840 parts; 574 defects.

| Capacity | Inspected | Defects found | False alarms | Defects missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 1,869 | 112 | 1,757 | 462 | 5.99% | 19.51% |
| 2% | 3,737 | 116 | 3,621 | 458 | 3.10% | 20.21% |
| 5% | 9,342 | 127 | 9,215 | 447 | 1.36% | 22.13% |
| 10% | 18,684 | 157 | 18,527 | 417 | 0.84% | 27.35% |

## Exploratory evaluation

236,634 parts; 964 defects.

| Capacity | Inspected | Defects found | False alarms | Defects missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 2,367 | 113 | 2,254 | 851 | 4.77% | 11.72% |
| 2% | 4,733 | 122 | 4,611 | 842 | 2.58% | 12.66% |
| 5% | 11,832 | 157 | 11,675 | 807 | 1.33% | 16.29% |
| 10% | 23,664 | 211 | 23,453 | 753 | 0.89% | 21.89% |

## Interpretation

Increasing capacity expands the inspected set and can recover more defects, at the cost of more inspections and false alarms. It does not improve ranking quality. Precision is not guaranteed to move monotonically. Costs, available staffing, and the consequences of missed defects are unknown, so no capacity is recommended here.

To improve recall at a fixed capacity, develop better features or compare candidates on validation recall at that predeclared capacity; retain training-only feature selection. Confirmation still requires an untouched future cohort.

Source run: `benchmark_20261030T212935628830Z`. See the [executed notebook](../benchmark_analysis.ipynb), [aggregate receipt](benchmark_results.json), and [reproduction instructions](reproduction.md).
