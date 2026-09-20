#!/usr/bin/env bash
# GPU-A: main pipeline pair suite, skip Exp-1 (XGBoost full DMatrix lives on GPU-B).
set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ}"
mkdir -p "${PROJ}/logs" "${PROJ}/results"

ts() { date -Iseconds; }

LOG="${PROJ}/logs/run_gpu_a_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " run_gpu_a  (pair, skip Exp-1)"
echo " time=$(ts)"
echo " host=$(hostname)"
echo " pwd=${PROJ}"
echo " log=${LOG}"
echo " python=$(command -v python || true)"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] 缺少 $1" >&2
    exit 1
  fi
}
require_file "${PROJ}/data/processed/v2/X_full.dat"
require_file "${PROJ}/data/processed/v2/y.npy"
require_file "${PROJ}/data/processed/v2/pairs.csv"
require_file "${PROJ}/data/raw/drugbank_drugs.csv"

echo
echo "======== START pipeline_pair_skip_exp1 $(ts) ========"
python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_pair \
  --split_mode pair \
  --n_folds 5 \
  --seeds 42 123 2024 \
  --skip_methods Exp-1
echo "======== END pipeline_pair_skip_exp1 $(ts) ========"
echo
echo "[run_gpu_a] 完成  log=${LOG}  time=$(ts)"
echo "[run_gpu_a] 与 GPU-B 的 Exp-1 合并: python scripts/collect_results.py"
