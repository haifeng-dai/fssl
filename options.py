# -*-coding:utf-8-*-
import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser(
        description="联邦半监督学习 (ProxyFL) 命令行参数解析"
    )

    # =========================================================================
    # 1. 基础实验与联邦拓扑参数 (Federated Learning & Basic Config)
    # =========================================================================
    parser.add_argument(
        "--method",
        type=str,
        default="ProxyFL",
        help="训练方法/算法标识名称 (保存路径与日志命名前缀)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="基准数据集: 支持 CIFAR10 / CIFAR100 / SVHN / CINIC10",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=20,
        help="参与联邦学习系统的客户端总数量 (K)",
    )
    parser.add_argument(
        "--num_online_clients",
        type=int,
        default=20,
        help="每一轮通信中被随机选中参与训练的活跃客户端数量 (C*K)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="狄利克雷分布浓度参数 alpha (为 0 时表示 IID 独立同分布划分，大于 0 时表示 Non-IID 异构程度)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="全局随机数种子，用于数据划分与网络初始化复现",
    )

    # =========================================================================
    # 2. 多卡/多进程与硬件加速参数 (Multi-GPU & Worker Pool Config)
    # =========================================================================
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="单 GPU 兼容参数；当未指定 --client_gpus 时默认使用的物理 GPU 卡号",
    )
    parser.add_argument(
        "--client_gpus",
        type=str,
        default=None,
        help="客户端并行 Worker 进程可调度的 GPU 列表 (逗号分隔，如 '0,1,2,3')",
    )
    parser.add_argument(
        "--server_gpu",
        type=int,
        default=None,
        help="服务端聚合、代理优化与全局评估使用的 GPU 卡号 (默认使用 client_gpus 中的第一张卡)",
    )
    parser.add_argument(
        "--max_parallel_clients",
        type=int,
        default=None,
        help="最大并发客户端 Worker 进程总数上限 (默认等于参与调度的 GPU 数量)",
    )
    parser.add_argument(
        "--gpu_processes",
        type=str,
        default=None,
        help="各 GPU 上分配的进程数配置 (格式为 GPU:进程数，如 '0:2,1:1' 表示 0号卡跑2进程，1号卡跑1进程)",
    )

    # =========================================================================
    # 3. 客户端本地半监督训练参数 (Client-side FixMatch & ICPL Config)
    # =========================================================================
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="每个客户端在每一轮通信中本地迭代训练的 epoch 轮数 (E)",
    )
    parser.add_argument(
        "--lr_local_training",
        type=float,
        default=0.1,
        help="客户端本地 SGD 优化器的初始学习率 (eta_l)",
    )
    parser.add_argument(
        "--batch_size_local_labeled_fixmatch",
        type=int,
        default=128,
        help="FixMatch 训练时有标签数据的批次大小 (B)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=512,
        help="服务端评估全局模型测试集时的批次大小",
    )
    parser.add_argument(
        "--mu",
        type=int,
        default=2,
        help="无标签样本与有标签样本的批大小比例倍数 (即无标签 batch size = B * mu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="FixMatch 伪标签置信度截断阈值 (tau)，超过该阈值的无标签样本才计算分类交叉熵损失",
    )
    parser.add_argument(
        "--lambda_u",
        type=float,
        default=1.0,
        help="客户端损失函数中无标签损失 (Lu) 与代理对比损失 (Lc) 的加权系数",
    )
    parser.add_argument(
        "--lambda_proto",
        type=float,
        default=1.0,
        help="客户端原型对比损失权重",
    )
    parser.add_argument(
        "--proto_temperature",
        type=float,
        default=0.1,
        help="样本-原型 InfoNCE 对比损失的温度系数",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="计算软伪标签时的温度系数 (Temperature)",
    )

    # =========================================================================
    # 4. 服务端代理调优参数 (Server-side ProxyFL / GPT Config)
    # =========================================================================
    parser.add_argument(
        "--lr_server",
        type=float,
        default=0.01,
        help="服务端代理网络 (GPT 分类头) 的优化学习率 (eta_s)",
    )
    parser.add_argument(
        "--total_server_epochs",
        type=int,
        default=30000,
        help="整个联邦训练全流程中服务端代理网络累计优化的总 epoch 轮数 (用于计算每轮轮数与实验命名)",
    )
    parser.add_argument(
        "--server_epochs",
        type=int,
        default=100,
        help="服务端每轮通信中代理网络优化的目标轮数 (若代码自动计算则作为基准参考)",
    )
    parser.add_argument(
        "--bs_server",
        type=int,
        default=10,
        help="服务端代理训练时 DataLoader 加载虚拟代理样本 (Proxies) 的批大小",
    )
    parser.add_argument(
        "--gpt_threshold",
        type=float,
        default=100.0,
        help="服务端代理训练中类别距离安全间隔 (Margin) 截断上限，防止训练发散",
    )

    # =========================================================================
    # 5. 脚本调用与日志输出兼容参数 (Logging & Shell Compatibility)
    # =========================================================================
    parser.add_argument(
        "--batch_label",
        type=int,
        default=128,
        help="日志与旧脚本兼容参数：有标签批次大小",
    )
    parser.add_argument(
        "--batch_unlabel",
        type=int,
        default=128,
        help="日志与旧脚本兼容参数：无标签批次大小",
    )
    parser.add_argument(
        "--topk",
        type=float,
        default=0.2,
        help="训练脚本 (train.sh) 传递的兼容参数",
    )

    # =========================================================================
    # 6. 本地数据集文件路径配置 (Dataset Path Config)
    # =========================================================================
    dataset_dir = os.path.expanduser("~/datasets")
    parser.add_argument(
        "--path",
        type=str,
        default=dataset_dir,
        help="所有数据集统一使用的根目录，默认使用 ~/datasets",
    )

    parser.add_argument(
        "--anchor_lr",
        type=float,
        default=0.01,
        help="服务端全局锚点学习率",
    )
    parser.add_argument(
        "--anchor_margin",
        type=float,
        default=1.0,
        help="锚点 L2 对比学习的负样本距离间隔",
    )
    parser.add_argument(
        "--anchor_steps",
        type=int,
        default=20,
        help="每轮服务端全局锚点优化步数",
    )

    args = parser.parse_args()
    return args
