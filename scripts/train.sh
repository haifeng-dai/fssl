#!/bin/bash

# 默认参数值
PYTHON_SCRIPT="proxyfl.py"
DATASET="CIFAR10"
ALPHA=0.1
GPU_ID=2
CLIENT_GPUS=""
SERVER_GPU=""
MAX_PARALLEL_CLIENTS=""
GPU_PROCESSES=""
DATALOADER_WORKERS=0
PIN_MEMORY=0
METHOD="proxyfl"
LR_SERVER=0.005
BS_SERVER=4
SERVER_EPOCHS=100
TOTAL_SERVER_EPOCHS=30000
LOCAL_EPOCHS=5
NUM_ONLINE_CLIENTS=20
NUM_CLIENTS=20
TOPK=0.2
THRES=0.95

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s| --script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        -d| --dataset)
            DATASET="$2"
            shift 2
            ;;
        -a| --alpha)
            ALPHA="$2"
            shift 2
            ;;
        -g| --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --client_gpus)
            CLIENT_GPUS="$2"
            shift 2
            ;;
        --server_gpu)
            SERVER_GPU="$2"
            shift 2
            ;;
        --max_parallel_clients)
            MAX_PARALLEL_CLIENTS="$2"
            shift 2
            ;;
        --gpu_processes)
            GPU_PROCESSES="$2"
            shift 2
            ;;
        --dataloader_workers)
            DATALOADER_WORKERS="$2"
            shift 2
            ;;
        --pin_memory)
            PIN_MEMORY=1
            shift
            ;;
        -lrs| --lr_server)
            LR_SERVER="$2"
            shift 2
            ;;
        -bss| --bs_server)
            BS_SERVER="$2"
            shift 2
            ;;
        -ep_s| --server_epochs)
            SERVER_EPOCHS="$2"
            shift 2
            ;;
        -ep_all| --total_server_epochs)
            TOTAL_SERVER_EPOCHS="$2"
            shift 2
            ;;
        -tao| --threshold)
            THRES="$2"
            shift 2
            ;;
        -ep_loc| --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        -n_on| --num_online_clients)
            NUM_ONLINE_CLIENTS="$2"
            shift 2
            ;;
        -n| --num_clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -m| --method)
            METHOD="$2"
            shift 2
            ;;
        -t| --topk)
            TOPK="$2"
            shift 2
            ;;
        -h| --help)
            show_help
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done


# 执行 Python 命令
CMD=(python "$PYTHON_SCRIPT" \
    --dataset "$DATASET" \
    --alpha "$ALPHA" \
    --gpu_id "$GPU_ID" \
    --method "$METHOD" \
    --lr_server "$LR_SERVER" \
    --bs_server "$BS_SERVER" \
    --server_epochs "$SERVER_EPOCHS" \
    --total_server_epochs "$TOTAL_SERVER_EPOCHS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --num_online_clients "$NUM_ONLINE_CLIENTS" \
    --num_clients "$NUM_CLIENTS" \
    --topk "$TOPK" \
    --threshold "$THRES" \
    --dataloader_workers "$DATALOADER_WORKERS")

if [[ -n "$CLIENT_GPUS" ]]; then
    CMD+=(--client_gpus "$CLIENT_GPUS")
fi
if [[ -n "$SERVER_GPU" ]]; then
    CMD+=(--server_gpu "$SERVER_GPU")
fi
if [[ -n "$MAX_PARALLEL_CLIENTS" ]]; then
    CMD+=(--max_parallel_clients "$MAX_PARALLEL_CLIENTS")
fi
if [[ -n "$GPU_PROCESSES" ]]; then
    CMD+=(--gpu_processes "$GPU_PROCESSES")
fi
if [[ "$PIN_MEMORY" -eq 1 ]]; then
    CMD+=(--pin_memory)
fi

"${CMD[@]}"


# bash scripts/train.sh --dataset CIFAR100 --alpha 0.1 --gpu 0
# bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu 1
# bash scripts/train.sh --dataset SVHN --alpha 0.1 --gpu 2
# bash scripts/train.sh --dataset CINIC10 --alpha 0.1 --gpu 3
