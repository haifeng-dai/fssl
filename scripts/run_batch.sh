#!/usr/bin/env bash
# =============================================================================
# 批量顺序运行多个算法实验
# =============================================================================

set -u

# 捕获用户中断 (Ctrl+C)：允许用户随时彻底终止整个批处理任务
trap 'echo -e "\n[!] 收到 Ctrl+C 中断信号，已退出批处理。"; exit 130' INT

# 定位项目根目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# 1. 公共默认参数 (可按需修改)
COMMON_ARGS="--client_gpus 0,1,2,3 --gpu_processes 0:2,1:2,2:2,3:2"

# 2. 待运行任务清单 (每行一个算法与独立参数，按需增删)
TASKS=(
    # "test.py"
    # "sage.py"
    "proxyfl.py"
    # "fedavg_a.py"
    # "fedavg_l.py"
    # "fixmatch_gpl.py"
    # "fixmatch_lpl.py"
    # "fixmatch_glpl.py"
    # "fixmatch_glpl_g.py"
    # "fixmatch_gplpl_p.py"
)

for cmd in "${TASKS[@]}"; do
    echo "================================================================="
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始运行: python ${cmd} ${COMMON_ARGS}"
    echo "================================================================="

    python ${cmd} ${COMMON_ARGS}
    exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ 运行完成: ${cmd}"
    elif [ ${exit_code} -eq 130 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [!] 任务被手动中断，终止批处理。"
        exit 130
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ 运行报错 (退出码: ${exit_code}): ${cmd}，跳过并继续执行下一个任务..."
    fi
done

echo "================================================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有任务执行完毕！"
echo "================================================================="
