#!/usr/bin/env bash
# Dual-GPU dispatcher: print usage only. Does not launch training.
set -euo pipefail
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ}"
mkdir -p "${PROJ}/logs"

ts() { date -Iseconds; }

LOG="${PROJ}/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " run_all  (usage only)"
echo " time=$(ts)"
echo " host=$(hostname)"
echo " pwd=${PROJ}"
echo " log=${LOG}"
echo "============================================================"
echo
echo "本脚本不启动训练。请把两张卡（或两台单卡机器）拆开跑："
echo
echo "  GPU-A  (≥24 GB VRAM, Exp-2/3/4/5 pair):"
echo "    CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_a.sh"
echo
echo "  GPU-B  (≥40 GB VRAM, Exp-1 + drug + final_full + baselines + DDInter):"
echo "    CUDA_VISIBLE_DEVICES=1 bash scripts/run_gpu_b.sh"
echo
echo "两台单卡机器等价于一台双卡：各跑上面一条即可，不必设 CUDA_VISIBLE_DEVICES。"
echo
echo "建议先冒烟："
echo "    bash scripts/smoke_cloud.sh"
echo
echo "tmux（SSH 断开不杀任务）："
echo "    tmux new -s ddi_a"
echo "    conda activate ddi_v2    # 或 source .venv/bin/activate"
echo "    export PYTHONIOENCODING=utf-8"
echo "    cd ${PROJ}"
echo "    CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_a.sh"
echo "    # 断线后: tmux attach -t ddi_a"
echo
echo "    tmux new -s ddi_b"
echo "    conda activate ddi_v2"
echo "    export PYTHONIOENCODING=utf-8"
echo "    cd ${PROJ}"
echo "    CUDA_VISIBLE_DEVICES=1 bash scripts/run_gpu_b.sh"
echo "    # 断线后: tmux attach -t ddi_b"
echo
echo "无 tmux 时："
echo "    mkdir -p logs"
echo "    CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_gpu_a.sh > logs/gpu_a.nohup.out 2>&1 &"
echo "    echo \$! > logs/gpu_a.pid"
echo "    CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_gpu_b.sh > logs/gpu_b.nohup.out 2>&1 &"
echo "    echo \$! > logs/gpu_b.pid"
echo
echo "两边 pair 都跑完后合并 Exp-1 与其余方法："
echo "    python scripts/collect_results.py"
echo
echo "断点续跑：同一命令再执行即可（已完成 run 会跳过）。不要加 --force_reset。"
echo "GPU-B 整步跳过：若 logs/gpu_b_done/<tag>.done 存在则跳过该步；删掉标记可重跑该步。"
echo "日志: logs/<name>_YYYYMMDD_HHMMSS.log"
echo
echo "[run_all] 用法已打印。log=${LOG}"
exit 0
