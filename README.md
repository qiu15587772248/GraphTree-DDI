# GraphTree-DDI

Graph–tree fusion for description-derived five-tier drug–drug interaction (DDI) severity grading.

This repository releases the preprocessing, model, evaluation, and external-comparison
source code that accompanies the manuscript, together with the 184-phrase → tier
table, hashed train/validation/test membership lists, and DDInter mapping scripts.
**No DrugBank records, descriptions, labels, fingerprints, or trained weights are
redistributed.**

## Repository layout

```
graphtree_ddi/
  data/           # XML/SDF parse, pair features, v2 rebuild, phrase map
  models/         # R-GCN / R-GAT, Exp-1–5 pipeline, pair inference, baselines
  analysis/       # metrics, paired bootstrap, tables, figures, SHAP helpers
  external/ddinter/   # DDInter download / clean / map / crosstab / inference
scripts/          # environment, GPU runners, split verification
resources/        # phrase_map_184.csv, PHRASE_MAP_RULES.md, drug_index.csv
splits/           # hashed pair folds + drug-level held-out ID lists
data/raw/         # you place DrugBank 5.1.15 XML/SDF here (not shipped)
data/processed/   # you generate features here (not shipped)
```

## Environment

- Python 3.10 (conda-forge 3.10.20 was used for split regeneration)
- See `requirements.txt` for the library set inferred from the training code
  (`torch`, `torch_geometric`, `xgboost`, `rdkit`, `scikit-learn`, `shap`, …)
- On a Linux CUDA host, `bash scripts/setup_env.sh` installs a pinned stack
  (PyTorch 2.4.1 + PyG 2.6.1 + XGBoost 2.1.3). Override the channel with
  `CUDA_CHANNEL=cu118` or `cu124` if needed.
- Pair-split regeneration used scikit-learn 1.6.1; `setup_env.sh` pins 1.5.2.
  Fold counts should still match `splits/README.md`; if they differ, install
  1.6.1 before comparing hashes.

```bash
conda create -n graphtree_ddi python=3.10 -y
conda activate graphtree_ddi
# GPU: follow scripts/setup_env.sh
pip install -r requirements.txt
```

## Data preparation (DrugBank academic licence)

DrugBank 5.1.15 is **not** included. Academic users must apply for a licence and
download the **full database XML** and **structures SDF** themselves. Place them at:

```
data/raw/full database.xml
data/raw/structures.sdf
```

Then, from the repository root:

```bash
export PYTHONIOENCODING=utf-8
python -m graphtree_ddi.data.download_data
python -m graphtree_ddi.data.preprocess
python -m graphtree_ddi.data.build_v2 \
  --input-dir data/processed \
  --output-dir data/processed/v2 \
  --ddi-csv data/raw/drugbank_ddi.csv \
  --drugs-csv data/raw/drugbank_drugs.csv
```

`download_data` writes `drugbank_drugs.csv` and `drugbank_ddi.csv` under `data/raw/`.
`preprocess` builds pair features (4120-d) and labels via the cascade in
`resources/PHRASE_MAP_RULES.md`. `build_v2` re-applies the 184-phrase map, drops
unordered-pair duplicates, and canonicalises `drug1_id < drug2_id`.

Expected v2 scale (for the paper snapshot): **1,332,174** pairs, **14,615** drugs,
sealed test **266,435** pairs. After you rebuild `pairs.csv`, run:

```bash
python scripts/verify_splits.py --pairs data/processed/v2/pairs.csv
```

Node order for fingerprints (DrugBank ID + integer index only) is
`resources/drug_index.csv`.

## Reproduce experiments

All commands assume v2 tables are already in `data/processed/v2/`
(`X_full.dat`, `y.npy`, `pairs.csv`, `meta.json`) and `data/raw/drugbank_drugs.csv`
exists. Writes go to `results/` (gitignored). Inspect flags with `--help` if a
host-specific option is missing.

### Exp-1–5, pair mode (20% sealed test, 5-fold × 3 seeds)

```bash
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_pair \
  --split_mode pair \
  --n_folds 5 \
  --seeds 42 123 2024
```

To skip Exp-1 on a memory-tight GPU (as in `scripts/run_gpu_a.sh`):

```bash
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_pair \
  --split_mode pair --n_folds 5 --seeds 42 123 2024 \
  --skip_methods Exp-1
```

Exp-1 only (large DMatrix; `scripts/run_gpu_b.sh` step 01):

```bash
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_pair_exp1 \
  --split_mode pair --n_folds 5 --seeds 42 123 2024 \
  --methods Exp-1
```

Merge the two pair directories:

```bash
python scripts/collect_results.py \
  --pair_dir results/final_v2_pair \
  --exp1_dir results/final_v2_pair_exp1 \
  --out_dir results/final_v2_pair_merged --copy
```

### Drug-level S1 / S2 (three partitions)

```bash
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_drug \
  --split_mode drug \
  --drug_split_seed 42 123 2024
```

Default methods are Exp-1, Exp-3, Exp-5. Held-out DrugBank IDs:
`splits/drug_partitions/partition_{0,1,2}_heldout_drugs.txt`.

### Full development-set bundle (`--final_full`)

```bash
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_full \
  --final_full
```

### Baselines

```bash
python -u -m graphtree_ddi.models.baselines.run_baselines \
  --data_dir data/processed/v2 \
  --out_dir results/baselines_v2_pair \
  --split_mode pair \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --folds 1 2 3 4 5 --seeds 42 123 2024 \
  --epochs 40 --patience 8 --batch_size 2048 --preload cpu --device cuda

python -u -m graphtree_ddi.models.baselines.run_baselines \
  --data_dir data/processed/v2 \
  --out_dir results/baselines_v2_drug \
  --split_mode drug \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --drug_split_seed 42 123 2024 \
  --epochs 40 --patience 8 --batch_size 2048 --preload cpu --device cuda
```

See `graphtree_ddi/models/baselines/README.md` for `--help` details (`--preload`,
`--kge_entity`, smoke mode).

### Paired bootstrap, tables, figures

```bash
python -u -m graphtree_ddi.analysis.make_all \
  --pred_dir results/final_v2_pair_merged results/baselines_v2_pair \
             results/final_v2_drug results/baselines_v2_drug \
  --out_dir results/analysis

# or call modules directly — check --help on each host
python -m graphtree_ddi.analysis.paired_bootstrap --help
python -m graphtree_ddi.analysis.plot_figures --help
python -m graphtree_ddi.analysis.summarize_runs --help
```

`make_all.py` / `paired_bootstrap.py` expect prediction `*.npz` written by the
pipeline. If a flag name differs on your checkout, use `--help` rather than
guessing.

### Smoke (not for manuscript numbers)

```bash
bash scripts/smoke.sh
# or:
python -u -m graphtree_ddi.models.run_pipeline_final --smoke \
  --data_dir data/processed/v2 --out_dir results/smoke_pair \
  --split_mode pair --n_folds 1 --seeds 42 --methods Exp-1 Exp-3 Exp-5 --force_reset
```

Two-GPU dispatcher (prints usage only): `bash scripts/run_all.sh`.

## DDInter label comparison

DDInter ATC-batch CSVs are **not** shipped (CC BY-NC-SA 4.0). Download them
yourself from the DDInter 2022 / DDInter 2.0 site, then:

```bash
python graphtree_ddi/external/ddinter/01_download.py
python graphtree_ddi/external/ddinter/02_clean.py
python graphtree_ddi/external/ddinter/03_map_drugs.py
python graphtree_ddi/external/ddinter/04_map_pairs.py
```

`03_map_drugs.py` looks for name tables under `data/processed/`
(`drugbank_name_aliases.csv`, `drugbank_name_i18n.csv`) that you derive from
**your** DrugBank XML. Those tables are not redistributed.

After a `--final_full` bundle exists:

```bash
python graphtree_ddi/external/ddinter/05_run_inference.py \
  --bundle results/final_v2_full/final_full \
  --data_dir data/processed/v2 \
  --out_dir results/ddinter --subset all --model both
python graphtree_ddi/external/ddinter/06_summarize.py \
  --pred results/ddinter/ddinter_pred_all_exp5.csv \
  --out_dir results/ddinter --subset all --model exp5
```

Confirm flags with `--help` on `05_run_inference.py` / `06_summarize.py`.
Crosstab logic lives in `04_map_pairs.py` and `06_summarize.py`.

Please cite DDInter and respect CC BY-NC-SA 4.0 (attribution, non-commercial,
share-alike).

## Licence

- **Code** in this repository: MIT (see `LICENSE`).
- **DrugBank** XML/SDF and every table derived from them (descriptions, labels,
  fingerprints, pair lists, feature matrices) are **not redistributed**. Obtain
  an academic licence from DrugBank and keep those files local.
- **DDInter** files that you download remain under CC BY-NC-SA 4.0; this
  repository only ships mapping/crosstab **scripts**.

## Citation

Citation: manuscript under review; will be added upon publication.

This repository is anonymized for peer review.
