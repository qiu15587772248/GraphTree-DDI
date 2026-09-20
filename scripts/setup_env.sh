#!/usr/bin/env bash
# 在 Ubuntu GPU 云主机上创建训练环境。
# 用法:
#   bash scripts/setup_env.sh
#   CUDA_CHANNEL=cu118 bash scripts/setup_env.sh    # 驱动只支持 CUDA 11.8 时
#   USE_VENV=1 bash scripts/setup_env.sh            # 强制 venv，不用 conda
set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-ddi_v2}"
# cu121 = PyTorch CUDA 12.1 轮子（默认）；cu118 = CUDA 11.8 备选
CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"
PY_VER="${PY_VER:-3.10}"

echo "============================================================"
echo " DDI cloud env setup"
echo " PROJ=${PROJ}"
echo " ENV_NAME=${ENV_NAME}"
echo " CUDA_CHANNEL=${CUDA_CHANNEL}"
echo "============================================================"

if [[ "${CUDA_CHANNEL}" != "cu121" && "${CUDA_CHANNEL}" != "cu118" && "${CUDA_CHANNEL}" != "cu124" ]]; then
  echo "CUDA_CHANNEL 必须是 cu121 / cu118 / cu124，当前: ${CUDA_CHANNEL}" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[warn] 未找到 nvidia-smi，仍继续安装 CUDA 轮子。"
fi

# ---------------------------------------------------------------------------
# Python: 优先 conda，否则 python3 venv
# ---------------------------------------------------------------------------
have_conda=0
if [[ "${USE_VENV:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  have_conda=1
elif [[ "${USE_VENV:-0}" != "1" ]] && [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  have_conda=1
elif [[ "${USE_VENV:-0}" != "1" ]] && [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  have_conda=1
fi

if [[ "${have_conda}" -eq 1 ]]; then
  if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[env] conda env ${ENV_NAME} 已存在，直接激活"
  else
    echo "[env] conda create -n ${ENV_NAME} python=${PY_VER}"
    conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
  fi
  conda activate "${ENV_NAME}"
  PYTHON="$(command -v python)"
else
  echo "[env] 未检测到 conda，使用 venv: ${PROJ}/.venv"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "需要 python3。可: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" >&2
    exit 1
  fi
  if [[ ! -d "${PROJ}/.venv" ]]; then
    python3 -m venv "${PROJ}/.venv"
  fi
  # shellcheck source=/dev/null
  source "${PROJ}/.venv/bin/activate"
  PYTHON="$(command -v python)"
fi

echo "[env] PYTHON=${PYTHON}  $($PYTHON -V)"
$PYTHON -m pip install -U pip wheel setuptools

# ---------------------------------------------------------------------------
# PyTorch
#   默认: torch==2.4.1 + CUDA 12.1
#   备选 CUDA 11.8（取消下一行注释，或 CUDA_CHANNEL=cu118）:
#     pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \\
#       --index-url https://download.pytorch.org/whl/cu118
#   备选 CUDA 12.4:
#     pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \\
#       --index-url https://download.pytorch.org/whl/cu124
# ---------------------------------------------------------------------------
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_CHANNEL}"
echo "[pip] torch==2.4.1 from ${TORCH_INDEX}"
$PYTHON -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# torch-geometric==2.6.1 + CUDA 扩展轮子
# PyG 扩展轮子与 torch 2.4.0+cuXXX 对齐（2.4.1 共用该目录）
# ---------------------------------------------------------------------------
echo "[pip] torch-geometric==2.6.1"
$PYTHON -m pip install torch-geometric==2.6.1

PYG_WHL="https://data.pyg.org/whl/torch-2.4.0+${CUDA_CHANNEL}.html"
echo "[pip] pyg optional deps from ${PYG_WHL}"
# 个别轮子可能缺 pyg_lib；失败不阻断，后面自检会标明
set +e
$PYTHON -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "${PYG_WHL}"
pyg_ext_rc=$?
set -e
if [[ "${pyg_ext_rc}" -ne 0 ]]; then
  echo "[warn] 部分 PyG 扩展安装失败。可手动:"
  echo "  pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib -f ${PYG_WHL}"
fi

# ---------------------------------------------------------------------------
# 其余依赖（XGBoost 官方 wheel 含 GPU；流水线使用 device=cuda, tree_method=hist）
# ---------------------------------------------------------------------------
echo "[pip] scientific stack"
$PYTHON -m pip install \
  "numpy>=1.26,<2.1" \
  pandas \
  "scikit-learn==1.5.2" \
  "xgboost==2.1.3" \
  scipy \
  pyyaml \
  tqdm \
  matplotlib \
  seaborn \
  zstandard \
  shap \
  rdkit

# 中文图（论文双语图）；无 root 时跳过
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y || true
    sudo apt-get install -y fonts-wqy-microhei fonts-noto-cjk zstd || true
    rm -rf "${HOME}/.cache/matplotlib" || true
  fi
fi

# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
echo "[check] Python self-test"
$PYTHON - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))

import numpy
print("numpy", numpy.__version__)
import pandas
print("pandas", pandas.__version__)
import sklearn
print("sklearn", sklearn.__version__)

import torch
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]", torch.cuda.get_device_name(i))
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}].total_memory_GB", round(p.total_memory / 1e9, 2))
    x = torch.zeros(1, device="cuda")
    print("cuda_tensor_ok", bool(x.is_cuda))
else:
    print("gpu_name", None)

import torch_geometric
print("torch_geometric", torch_geometric.__version__)

for name in ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "pyg_lib"):
    try:
        m = __import__(name)
        print(name, getattr(m, "__version__", "imported"))
    except Exception as e:
        print(name, "MISSING", type(e).__name__, str(e).split("\n")[0][:200])

import xgboost
print("xgboost", xgboost.__version__)
try:
    from xgboost.core import build_info
    info = build_info() if callable(build_info) else {}
    print("xgboost_use_cuda", info.get("USE_CUDA", info.get("use_cuda", "n/a")))
except Exception as e:
    print("xgboost_build_info", type(e).__name__, e)

import zstandard, tqdm, matplotlib, seaborn, shap
print("zstandard", zstandard.__version__)
print("tqdm", tqdm.__version__)
print("matplotlib", matplotlib.__version__)
print("seaborn", seaborn.__version__)
print("shap", shap.__version__)
try:
    import rdkit
    from rdkit import Chem
    print("rdkit", getattr(rdkit, "__version__", "ok"), "Chem.MolFromSmiles", Chem.MolFromSmiles("CCO") is not None)
except Exception as e:
    print("rdkit FAIL", e)
    sys.exit(1)

if not torch.cuda.is_available():
    print("[warn] CUDA 不可用：请核对该机器驱动与 CUDA_CHANNEL（cu121/cu118/cu124）是否匹配。")
    sys.exit(2)
print("[check] OK")
PY

echo "============================================================"
echo " setup_env 完成。之后每次:"
if [[ "${have_conda}" -eq 1 ]]; then
  echo "   conda activate ${ENV_NAME}"
else
  echo "   source ${PROJ}/.venv/bin/activate"
fi
echo "============================================================"
