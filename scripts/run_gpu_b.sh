#!/usr/bin/env bash
# GPU-B: Exp-1 pair + drug cold-start + final_full + baselines + DDInter.
# Each step is internally resume-safe. Completed steps may be skipped via marker files.
set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ}"
mkdir -p "${PROJ}/logs" "${PROJ}/results" "${PROJ}/logs/gpu_b_done"

ts() { date -Iseconds; }

LOG="${PROJ}/logs/run_gpu_b_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " run_gpu_b"
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

DONE_DIR="${PROJ}/logs/gpu_b_done"

run_or_skip() {
  local tag="$1"
  shift
  local marker="${DONE_DIR}/${tag}.done"
  echo
  if [ -f "${marker}" ]; then
    echo "[skip] ${tag}  已有完成标记 ${marker}  time=$(ts)"
    return 0
  fi
  echo "======== START ${tag} $(ts) ========"
  "$@"
  touch "${marker}"
  echo "======== END ${tag} $(ts)  marker=${marker} ========"
}

# (i) main pipeline pair, Exp-1 only
run_or_skip "01_pair_exp1" python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_pair_exp1 \
  --split_mode pair \
  --n_folds 5 \
  --seeds 42 123 2024 \
  --methods Exp-1

# (ii) drug cold-start (default methods Exp-1 / Exp-3 / Exp-5)
run_or_skip "02_drug" python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_drug \
  --split_mode drug \
  --drug_split_seed 42 123 2024

# (iii) deployment bundle on the 80% development set
run_or_skip "03_final_full" python -u -m graphtree_ddi.models.run_pipeline_final \
  --data_dir data/processed/v2 \
  --out_dir results/final_v2_full \
  --final_full

# (iv) baselines pair
run_or_skip "04_baselines_pair" python -u -m graphtree_ddi.models.baselines.run_baselines \
  --data_dir data/processed/v2 \
  --out_dir results/baselines_v2_pair \
  --split_mode pair \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --folds 1 2 3 4 5 \
  --seeds 42 123 2024 \
  --epochs 40 \
  --patience 8 \
  --batch_size 2048 \
  --preload cpu \
  --device cuda

# (v) baselines drug
run_or_skip "05_baselines_drug" python -u -m graphtree_ddi.models.baselines.run_baselines \
  --data_dir data/processed/v2 \
  --out_dir results/baselines_v2_drug \
  --split_mode drug \
  --methods MLP DDIMDL DeepDDI_SSP DistMult ComplEx \
  --drug_split_seed 42 123 2024 \
  --epochs 40 \
  --patience 8 \
  --batch_size 2048 \
  --preload cpu \
  --device cuda

# (vi) DDInter inference + summarize
# verify CLI before run  (05/06 由另一执行者维护；若 CLI 变更请先核对这些参数)
ddinter_infer_and_summarize() {
  python -u graphtree_ddi/external/ddinter/05_run_inference.py \
    --bundle results/final_v2_full/final_full \
    --data_dir data/processed/v2 \
    --out_dir results/ddinter \
    --subset all \
    --model both
  # verify CLI before run
  python -u graphtree_ddi/external/ddinter/06_summarize.py \
    --pred results/ddinter/ddinter_pred_all_exp5.csv \
    --out_dir results/ddinter \
    --subset all \
    --model exp5
  # verify CLI before run
  python -u graphtree_ddi/external/ddinter/06_summarize.py \
    --pred results/ddinter/ddinter_pred_all_exp1.csv \
    --out_dir results/ddinter \
    --subset all \
    --model exp1
}

run_or_skip "06_ddinter" ddinter_infer_and_summarize

echo
echo "[run_gpu_b] 完成  log=${LOG}  time=$(ts)"
echo "[run_gpu_b] 与 GPU-A 的 pair 非 Exp-1 合并: python scripts/collect_results.py"
echo "[run_gpu_b] 重跑某步: 删除 logs/gpu_b_done/<tag>.done 后再执行本脚本"
