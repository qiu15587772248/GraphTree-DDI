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
- Folds are drawn once with `SPLIT_RANDOM_STATE=42` and shared by training
  seeds 42/123/2024. Those seeds only re-initialise model RNG; they do **not**
  redraw the pair folds. Sealed-test rows have `fold = -1`.
- Drug partitions: `numpy.random.default_rng(seed).permutation` over
  `sorted(unique drug IDs)`, hold out `round(0.20 * n_drugs)` drugs.
  `partition_{0,1,2}` correspond to seeds `42, 123, 2024`.

The manifest was regenerated with the released code under scikit-learn 1.6.1;
per-split counts match the logged runs (sealed test n = 266,435; S1/S2 pair
counts per partition listed below). scikit-learn 1.6.x also reproduces these
counts; the install pins use 1.5.2 (`requirements.txt` / `scripts/setup_env.sh`).

## Pair file: `pair_splits.csv.gz`

| field | meaning |
|---|---|
| `pair_hash` | first **32 hex characters** (128 bits) of `sha256("{min(id_a,id_b)}\|{max(id_a,id_b)}")`; IDs are DrugBank strings |
| `sealed_test` | `1` if in the 20% sealed test, else `0` |
| `fold` | validation-fold id `1..5`, or **`-1` if sealed test** |

Collision chance at n ≈ 1.3×10^6 with a 128-bit prefix is on the order of
n² / 2^{129} ≈ 10^{-27} and can be ignored. `scripts/verify_splits.py` uses
the same truncation.

- Rows: **1,332,174**
- Sealed test (`fold = -1`): **266,435**
- Train+val: **1,065,739**
- Unique drugs in pairs: **14,615**

Per-fold validation counts (same for every training seed):

| fold | n |
|---:|---:|
| -1 (sealed test) | 266,435 |
| 1 | 213,148 |
| 2 | 213,148 |
| 3 | 213,148 |
| 4 | 213,148 |
| 5 | 213,147 |

## Drug partitions

Held-out DrugBank IDs (plaintext identifiers only):

| file | seed | held-out drugs \|U\| | S1 pairs | S2 pairs | matches logged runs |
|---|---:|---:|---:|---:|---|
| `drug_partitions/partition_0_heldout_drugs.txt` | 42 | 2,923 | 434,394 | 56,212 | yes |
| `drug_partitions/partition_1_heldout_drugs.txt` | 123 | 2,923 | 435,857 | 56,186 | yes |
| `drug_partitions/partition_2_heldout_drugs.txt` | 2024 | 2,923 | 419,049 | 50,888 | yes |

## Verify locally

After you rebuild `pairs.csv` under an academic DrugBank licence:

```bash
python scripts/verify_splits.py --pairs data/processed/v2/pairs.csv
```
