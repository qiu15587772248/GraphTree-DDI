#!/usr/bin/env bash
# Cloud smoke on v2 data. Metrics are NOT paper-ready.
set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ}"
mkdir -p "${PROJ}/logs" "${PROJ}/results"

ts() { date -Iseconds; }

LOG="${PROJ}/logs/smoke_cloud_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " smoke_cloud"
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

run_step() {
  local name="$1"
  shift
  echo
  echo "======== START ${name} $(ts) ========"
  "$@"
  echo "======== END ${name} $(ts) ========"
}

run_step "pipeline_pair_smoke" python -u -m graphtree_ddi.models.run_pipeline_final \
  --smoke \
  --data_dir data/processed/v2 \
  --out_dir results/smoke_v2_pair \
  --split_mode pair \
  --n_folds 1 \
  --seeds 42 \
  --methods Exp-1 Exp-3 Exp-5 \
  --force_reset

run_step "pipeline_drug_smoke" python -u -m graphtree_ddi.models.run_pipeline_final \
  --smoke \
  --data_dir data/processed/v2 \
  --out_dir results/smoke_v2_drug \
  --split_mode drug \
  --drug_split_seed 42 \
  --methods Exp-1 Exp-3 Exp-5 \
  --force_reset

run_step "pipeline_final_full_smoke" python -u -m graphtree_ddi.models.run_pipeline_final \
  --smoke \
  --data_dir data/processed/v2 \
  --out_dir results/smoke_v2_full \
  --final_full \
  --force_reset

run_step "baselines_pair_drug_smoke" python -u -m graphtree_ddi.models.baselines.run_baselines \
  --smoke \
  --split_mode pair drug \
  --preload cpu \
  --data_dir data/processed/v2 \
  --out_dir results/smoke_v2_baselines \
  --device cuda

echo
echo "======== START smoke_npz_state_summary $(ts) ========"
python -u - <<'PY'
from pathlib import Path
import json

dirs = [
    "results/smoke_v2_pair",
    "results/smoke_v2_drug",
    "results/smoke_v2_full",
    "results/smoke_v2_baselines",
]
print(f"{'dir':<42} {'npz':>6} {'state_entries':>14} {'state_files'}")
for d in dirs:
    p = Path(d)
    n_npz = len(list(p.rglob("*.npz"))) if p.exists() else 0
    state_files = sorted(p.glob("*state.json")) if p.exists() else []
    n_runs = 0
    for sf in state_files:
        try:
            st = json.loads(sf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs = st.get("runs") or {}
        n_runs += len(runs)
    names = ",".join(sf.name for sf in state_files) or "-"
    print(f"{d:<42} {n_npz:6d} {n_runs:14d} {names}")
PY
echo "======== END smoke_npz_state_summary $(ts) ========"
echo
echo "[smoke_cloud] 完成  log=${LOG}  time=$(ts)"
echo "[smoke_cloud] 冒烟指标不能写入论文。"
