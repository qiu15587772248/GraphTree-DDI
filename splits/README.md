# Experiment splits

These files reproduce the **final paper protocol** used in the pair-mode and
drug-level (S1/S2) experiments. They contain **no labels** and **no plaintext
pair IDs** (pair file is hashed).

## How they were obtained

The official run does **not** persist fold indices. Splits are generated
deterministically by `graphtree_ddi.models.run_pipeline_final` /
`graphtree_ddi.models.baselines.common`:

- Sealed test: `sklearn.model_selection.train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- 5-fold CV on the remaining 80%: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Training seeds `42 / 123 / 2024` only re-initialise model RNG. They **do not**
  redraw the pair folds. Columns `fold_seed42`, `fold_seed123`, and
  `fold_seed2024` are therefore identical (fold id of the validation slice; `0`
  = sealed test).
- Drug partitions: `numpy.random.default_rng(seed).permutation` over
  `sorted(unique drug IDs)`, hold out `round(0.20 * n_drugs)` drugs.
  `partition_{0,1,2}` correspond to seeds `42, 123, 2024`.

This archive was regenerated from the local v2 `pairs.csv` + `y.npy` (not
shipped) with scikit-learn 1.6.1 and checked
against `cloud_results` counts.

## Pair file: `pair_splits.csv.gz`

| field | meaning |
|---|---|
| `pair_hash` | `sha256("{min(id_a,id_b)}|{max(id_a,id_b)}")` hex; IDs are DrugBank strings |
| `sealed_test` | `1` if in the 20% sealed test, else `0` |
| `fold_seed42` / `fold_seed123` / `fold_seed2024` | validation-fold id `1..5`, or `0` if sealed test |

- Rows: **1,332,174**
- Sealed test: **266,435** (expected 266,435)
- Train+val: **1,065,739**
- Unique drugs in pairs: **14,615**

Per-fold validation counts (same for every training seed):

| fold | n_val |
|---:|---:|
| 1 | 213,148 |
| 2 | 213,148 |
| 3 | 213,148 |
| 4 | 213,148 |
| 5 | 213,147 |

## Drug partitions

Held-out DrugBank IDs (plaintext identifiers only):

| file | seed | \|U\| | S1 pairs | S2 pairs | matches cloud_results |
|---|---:|---:|---:|---:|---|
| `drug_partitions/partition_0_heldout_drugs.txt` | 42 | 2,923 | 434,394 | 56,212 | True |
| `drug_partitions/partition_1_heldout_drugs.txt` | 123 | 2,923 | 435,857 | 56,186 | True |
| `drug_partitions/partition_2_heldout_drugs.txt` | 2024 | 2,923 | 419,049 | 50,888 | True |

Cloud S1/S2 targets: seed 42 → 434394 / 56212; seed 123 → 435857 / 56186; seed 2024 → 419049 / 50888.

## Verify locally

After you rebuild `pairs.csv` under an academic DrugBank licence:

```bash
python scripts/verify_splits.py --pairs data/processed/v2/pairs.csv
```
