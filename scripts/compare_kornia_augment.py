"""比较当前 PIL 增强与近似 Kornia GPU 增强的数值分布和吞吐。

Usage:
    uv run python scripts/compare_kornia_augment.py --data-root /home/dhf/datasets

该脚本不报告分类精度：两套随机增强不是逐像素等价实现，最终精度需用
相同随机种子、超参数和训练轮数的 A/B 实验比较。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.dataset import Indices2Dataset_unlabeled_fixmatch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)

try:
    import kornia.augmentation as K
except ImportError as error:
    raise SystemExit(
        "Kornia 未安装。请先执行 `uv add kornia`，再运行本脚本。"
    ) from error


def normalize(images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return (images - mean) / std


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summary(name, images, elapsed=None):
    channel_mean = images.mean(dim=(0, 2, 3)).cpu().tolist()
    channel_std = images.std(dim=(0, 2, 3)).cpu().tolist()
    message = (
        f"{name}: mean={np.round(channel_mean, 4).tolist()} "
        f"std={np.round(channel_std, 4).tolist()}"
    )
    if elapsed is not None:
        message += f" time={elapsed * 1000:.1f} ms"
    print(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/dhf/datasets")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.data_root, train=True, download=False)
    sample_count = min(args.num_samples, len(dataset))
    indices = list(range(sample_count))

    # 当前实现：PIL + CPU RandAugmentMC。
    cpu_dataset = Indices2Dataset_unlabeled_fixmatch(dataset)
    cpu_dataset.load(indices)
    start = time.perf_counter()
    cpu_pairs = [cpu_dataset[index] for index in range(sample_count)]
    cpu_elapsed = time.perf_counter() - start
    cpu_weak = torch.stack([pair[0] for pair in cpu_pairs])
    cpu_strong = torch.stack([pair[1] for pair in cpu_pairs])
    summary("CPU/PIL weak", cpu_weak, cpu_elapsed)
    summary("CPU/PIL strong", cpu_strong)

    raw = torch.from_numpy(
        np.stack([np.asarray(dataset[index][0], dtype=np.uint8) for index in indices])
    ).permute(0, 3, 1, 2)

    # 仅测 uint8 -> float -> normalize 的数值误差；这部分应接近 float32 舍入误差。
    cpu_plain = normalize(raw.float().div(255))
    gpu_plain = normalize(raw.to(device, non_blocking=True).float().div(255))
    plain_error = (cpu_plain - gpu_plain.cpu()).abs()
    print(
        "转换误差: "
        f"mean_abs={plain_error.mean().item():.8f} "
        f"max_abs={plain_error.max().item():.8f}"
    )

    weak_augment = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomCrop((32, 32), padding=4, padding_mode="reflect", p=1.0),
    ).to(device)
    # 这是与当前 RandAugmentMC 语义相近、但非逐操作复刻的 GPU 强增强。
    strong_augment = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomCrop((32, 32), padding=4, padding_mode="reflect", p=1.0),
        K.RandomAffine(degrees=30, translate=(0.3, 0.3), shear=(-0.3, 0.3), p=0.5),
        K.ColorJitter(0.9, 0.9, 0.9, 0.0, p=0.5),
        K.RandomSolarize(thresholds=0.5, p=0.5),
        K.RandomPosterize(bits=4, p=0.5),
        K.RandomErasing(scale=(0.0, 0.25), ratio=(1.0, 1.0), value=127 / 255, p=0.5),
    ).to(device)

    gpu_weak_batches, gpu_strong_batches = [], []
    synchronize(device)
    start = time.perf_counter()
    for batch_start in range(0, sample_count, args.batch_size):
        images = raw[batch_start : batch_start + args.batch_size].to(
            device, non_blocking=True
        ).float().div(255)
        gpu_weak_batches.append(normalize(weak_augment(images)).cpu())
        gpu_strong_batches.append(normalize(strong_augment(images)).cpu())
    synchronize(device)
    gpu_elapsed = time.perf_counter() - start
    summary("GPU/Kornia weak", torch.cat(gpu_weak_batches), gpu_elapsed)
    summary("GPU/Kornia strong", torch.cat(gpu_strong_batches))
    print(f"增强吞吐倍率（CPU/GPU）：{cpu_elapsed / gpu_elapsed:.2f}x")


if __name__ == "__main__":
    main()
