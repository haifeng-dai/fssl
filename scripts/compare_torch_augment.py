"""对比当前 PIL RandAugmentMC 与 Torch/GPU 版增强。

Usage:
    uv run python scripts/compare_torch_augment.py \
        --data-root /home/dhf/datasets \
        --device cuda:0 \
        --warmup 3 --repeats 10

Torch 版本保留当前策略：
- weak: 每样本独立 Flip + ReflectPad + RandomCrop
- strong: weak + 随机抽两个操作；每个操作 50% 概率执行；
          magnitude 在 [1, 10] 随机；最后固定 Cutout。
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.dataset import Indices2Dataset_unlabeled_fixmatch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)

IMAGE_SIZE = 32
PADDING = 4
RAND_AUG_N = 2
RAND_AUG_M = 10
CUTOUT_SIZE = 16
CUTOUT_VALUE = 127 / 255


def normalize(images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return (images - mean) / std


def denormalize(images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return images * std + mean


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(name, fn, device, warmup, repeats):
    """预热后重复计时，返回最后一次结果和每轮耗时（秒）。

    CUDA 操作是异步的，因此每个 warmup/测量边界都同步，避免把尚未完成的
    kernel 计到下一段或漏记在本段之外。
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    durations = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        durations.append(time.perf_counter() - start)

    mean_seconds = float(np.mean(durations))
    print(
        f"{name}: {mean_seconds * 1000:.2f} ms/run "
        f"(min {min(durations) * 1000:.2f} ms, n={repeats})"
    )
    return result, mean_seconds


def to_float_batch(raw_uint8, device):
    return raw_uint8.to(device, non_blocking=True).float().div(255)


def random_crop_reflect(images, generator):
    """每个样本独立进行 reflect padding 和 32x32 随机裁剪。"""
    batch_size = images.shape[0]
    padded = F.pad(images, (PADDING, PADDING, PADDING, PADDING), mode="reflect")

    top = torch.randint(
        0,
        PADDING * 2 + 1,
        (batch_size,),
        device=images.device,
        generator=generator,
    )
    left = torch.randint(
        0,
        PADDING * 2 + 1,
        (batch_size,),
        device=images.device,
        generator=generator,
    )

    # [B, C, 9, 9, 32, 32]，再按每张图自己的 top/left 取窗口。
    windows = padded.unfold(2, IMAGE_SIZE, 1).unfold(3, IMAGE_SIZE, 1)
    batch_index = torch.arange(batch_size, device=images.device)
    return windows[batch_index, :, top, left]


def torch_weak_augment(images, generator):
    """当前 weak 增强的 GPU/Tensor 版，输入为 [B, 3, 32, 32] float [0, 1]。"""
    batch_size = images.shape[0]

    flip_mask = torch.rand(batch_size, device=images.device, generator=generator) < 0.5
    output = images.clone()
    output[flip_mask] = torch.flip(output[flip_mask], dims=(-1,))
    return random_crop_reflect(output, generator)


def gray(images):
    """RGB 转灰度，供 Color/Contrast 使用。"""
    weights = images.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def adjust_brightness(images, factor):
    return (images * factor.view(-1, 1, 1, 1)).clamp(0, 1)


def adjust_contrast(images, factor):
    base = gray(images).mean(dim=(2, 3), keepdim=True)
    return (base + factor.view(-1, 1, 1, 1) * (images - base)).clamp(0, 1)


def adjust_color(images, factor):
    base = gray(images)
    return (base + factor.view(-1, 1, 1, 1) * (images - base)).clamp(0, 1)


def adjust_sharpness(images, factor):
    """近似 PIL ImageEnhance.Sharpness。"""
    channels = images.shape[1]
    kernel = (
        images.new_tensor(
            [
                [1, 1, 1],
                [1, 5, 1],
                [1, 1, 1],
            ]
        ).view(1, 1, 3, 3)
        / 13
    )

    blurred = F.conv2d(
        images,
        kernel.repeat(channels, 1, 1, 1),
        padding=1,
        groups=channels,
    )
    return (blurred + factor.view(-1, 1, 1, 1) * (images - blurred)).clamp(0, 1)


def autocontrast(images):
    """逐样本、逐通道 min-max 拉伸。"""
    minimum = images.amin(dim=(2, 3), keepdim=True)
    maximum = images.amax(dim=(2, 3), keepdim=True)
    scale = (maximum - minimum).clamp_min(1 / 255)
    return ((images - minimum) / scale).clamp(0, 1)


def equalize_one_channel(channel):
    """单通道直方图均衡化；在 GPU 上运行，但每通道单独处理。"""
    pixels = (channel.clamp(0, 1) * 255).round().to(torch.int64)
    histogram = torch.bincount(pixels.flatten(), minlength=256)
    cdf = histogram.cumsum(0)

    nonzero = cdf[cdf > 0]
    if nonzero.numel() == 0:
        return channel

    cdf_min = nonzero[0]
    denominator = pixels.numel() - cdf_min
    if denominator <= 0:
        return channel

    lut = ((cdf - cdf_min) * 255 / denominator).round().clamp(0, 255)
    return lut[pixels].to(channel.dtype).div(255)


def equalize(images):
    """逐样本、逐通道均衡化。"""
    output = images.clone()
    for sample_index in range(images.shape[0]):
        for channel_index in range(images.shape[1]):
            output[sample_index, channel_index] = equalize_one_channel(
                images[sample_index, channel_index]
            )
    return output


def posterize(images, bits):
    """bits 为每张样本保留的 bit 数，范围 [1, 8]。"""
    output = images.clone()
    uint8_images = (images.clamp(0, 1) * 255).round().to(torch.uint8)

    for bit in range(1, 9):
        indices = torch.where(bits == bit)[0]
        if indices.numel() == 0:
            continue
        mask = (0xFF << (8 - bit)) & 0xFF
        output[indices] = (uint8_images[indices] & mask).float().div(255)
    return output


def solarize(images, threshold):
    """threshold 是每样本阈值，范围 [0, 1]。"""
    threshold = threshold.view(-1, 1, 1, 1)
    return torch.where(images >= threshold, 1 - images, images)


def affine(images, angle_deg=None, tx=None, ty=None, shear_x=None, shear_y=None):
    """以 batch 方式应用每样本独立仿射变换，填充值近似 PIL 的灰色背景。"""
    batch_size = images.shape[0]
    device = images.device
    dtype = images.dtype

    angle_deg = (
        torch.zeros(batch_size, device=device, dtype=dtype)
        if angle_deg is None
        else angle_deg
    )
    tx = torch.zeros(batch_size, device=device, dtype=dtype) if tx is None else tx
    ty = torch.zeros(batch_size, device=device, dtype=dtype) if ty is None else ty
    shear_x = (
        torch.zeros(batch_size, device=device, dtype=dtype)
        if shear_x is None
        else shear_x
    )
    shear_y = (
        torch.zeros(batch_size, device=device, dtype=dtype)
        if shear_y is None
        else shear_y
    )

    radians = torch.deg2rad(angle_deg)
    cos_angle = torch.cos(radians)
    sin_angle = torch.sin(radians)

    # PIL Shear 的 factor 语义近似，不是严格逐像素复刻。
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_angle + shear_y * sin_angle
    theta[:, 0, 1] = shear_x * cos_angle - sin_angle
    theta[:, 1, 0] = sin_angle - shear_y * cos_angle
    theta[:, 1, 1] = shear_x * sin_angle + cos_angle

    # affine_grid 使用归一化坐标。
    theta[:, 0, 2] = tx * 2 / IMAGE_SIZE
    theta[:, 1, 2] = ty * 2 / IMAGE_SIZE

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def cutout(images, generator, size=CUTOUT_SIZE):
    """每个样本固定执行一次、位置独立的方形 Cutout。"""
    batch_size, _, height, width = images.shape
    center_y = torch.randint(
        0, height, (batch_size,), device=images.device, generator=generator
    )
    center_x = torch.randint(
        0, width, (batch_size,), device=images.device, generator=generator
    )

    y = torch.arange(height, device=images.device).view(1, height, 1)
    x = torch.arange(width, device=images.device).view(1, 1, width)

    top = (center_y - size // 2).view(batch_size, 1, 1)
    left = (center_x - size // 2).view(batch_size, 1, 1)

    mask = (y >= top) & (y < top + size) & (x >= left) & (x < left + size)
    return images.masked_fill(mask.unsqueeze(1), CUTOUT_VALUE)


OP_NAMES = (
    "identity",
    "autocontrast",
    "brightness",
    "color",
    "contrast",
    "equalize",
    "posterize",
    "rotate",
    "sharpness",
    "shear_x",
    "shear_y",
    "solarize",
    "translate_x",
    "translate_y",
)


def signed_value(magnitude, max_value, generator):
    sign = torch.where(
        torch.rand(magnitude.shape, device=magnitude.device, generator=generator) < 0.5,
        -1.0,
        1.0,
    )
    return sign * magnitude.float() / RAND_AUG_M * max_value


def apply_randaugment_op(images, op_ids, magnitudes, generator):
    """对 batch 内不同样本按 op 分组执行；每张样本参数独立。"""
    output = images.clone()

    for op_id, op_name in enumerate(OP_NAMES):
        indices = torch.where(op_ids == op_id)[0]
        if indices.numel() == 0:
            continue

        subset = output[indices]
        m = magnitudes[indices]

        if op_name == "identity":
            continue

        if op_name == "autocontrast":
            output[indices] = autocontrast(subset)

        elif op_name == "brightness":
            # 当前 PIL Brightness 的近似范围：0.1 ~ 1.9。
            factor = 1.0 + signed_value(m, 0.9, generator)
            output[indices] = adjust_brightness(subset, factor)

        elif op_name == "color":
            factor = 1.0 + signed_value(m, 0.9, generator)
            output[indices] = adjust_color(subset, factor)

        elif op_name == "contrast":
            factor = 1.0 + signed_value(m, 0.9, generator)
            output[indices] = adjust_contrast(subset, factor)

        elif op_name == "equalize":
            output[indices] = equalize(subset)

        elif op_name == "posterize":
            # magnitude 越大，保留 bit 越少，扰动越强。
            bits = (8 - (m.float() / RAND_AUG_M * 4).round()).clamp(1, 8)
            output[indices] = posterize(subset, bits.to(torch.int64))

        elif op_name == "rotate":
            angle = signed_value(m, 30.0, generator)
            output[indices] = affine(subset, angle_deg=angle)

        elif op_name == "sharpness":
            factor = 1.0 + signed_value(m, 0.9, generator)
            output[indices] = adjust_sharpness(subset, factor)

        elif op_name == "shear_x":
            shear_x = signed_value(m, 0.3, generator)
            output[indices] = affine(subset, shear_x=shear_x)

        elif op_name == "shear_y":
            shear_y = signed_value(m, 0.3, generator)
            output[indices] = affine(subset, shear_y=shear_y)

        elif op_name == "solarize":
            threshold = 1.0 - m.float() / RAND_AUG_M
            output[indices] = solarize(subset, threshold)

        elif op_name == "translate_x":
            tx = signed_value(m, 0.3 * IMAGE_SIZE, generator)
            output[indices] = affine(subset, tx=tx)

        elif op_name == "translate_y":
            ty = signed_value(m, 0.3 * IMAGE_SIZE, generator)
            output[indices] = affine(subset, ty=ty)

    return output


def torch_strong_augment(images, generator):
    """当前 RandAugmentMC(n=2, m=10) 的 Torch/GPU 版近似实现。"""
    output = torch_weak_augment(images, generator)
    batch_size = output.shape[0]

    for _ in range(RAND_AUG_N):
        op_ids = torch.randint(
            0,
            len(OP_NAMES),
            (batch_size,),
            device=output.device,
            generator=generator,
        )
        magnitudes = torch.randint(
            1,
            RAND_AUG_M + 1,
            (batch_size,),
            device=output.device,
            generator=generator,
        )

        # 与当前 RandAugmentMC 一样：每个抽中的算子有 50% 概率执行。
        apply_mask = (
            torch.rand(batch_size, device=output.device, generator=generator) < 0.5
        )

        if apply_mask.any():
            chosen = torch.where(apply_mask)[0]
            output[chosen] = apply_randaugment_op(
                output[chosen],
                op_ids[chosen],
                magnitudes[chosen],
                generator,
            )

    return cutout(output, generator)


def stats(name, images):
    values = images.detach().float()
    means = values.mean(dim=(0, 2, 3)).cpu().numpy()
    stds = values.std(dim=(0, 2, 3)).cpu().numpy()

    text = (
        f"{name}\n"
        f"  mean: {np.round(means, 4).tolist()}\n"
        f"  std : {np.round(stds, 4).tolist()}"
    )
    print(text)


def diff_stats(name, left, right):
    diff = (left.float() - right.float()).abs()
    print(
        f"{name}\n"
        f"  mean absolute difference (仅作分布诊断，非逐像素一致性): "
        f"{diff.mean().item():.6f}\n"
        f"  max abs: {diff.max().item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/dhf/datasets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup 必须非负，--repeats 必须至少为 1")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = datasets.CIFAR10(args.data_root, train=True, download=False)
    count = min(args.num_samples, len(dataset))
    indices = list(range(count))

    # 当前 PIL 路线。
    pil_dataset = Indices2Dataset_unlabeled_fixmatch(dataset)
    pil_dataset.load(indices)

    # 将读取原图排除在计时外，和下方预先构造 raw 的 Torch 路线保持一致。
    pil_images = [dataset[index][0] for index in indices]

    def pil_weak_route():
        return torch.stack(
            [pil_dataset.normalize(pil_dataset.weak(image)) for image in pil_images]
        )

    def pil_strong_route():
        return torch.stack(
            [pil_dataset.normalize(pil_dataset.strong(image)) for image in pil_images]
        )

    # 原始 uint8 batch；实际训练中应由 DataLoader 返回这个形式。
    raw = torch.from_numpy(
        np.stack([np.asarray(dataset[index][0], dtype=np.uint8) for index in indices])
    ).permute(0, 3, 1, 2)

    def torch_route(augment):
        # 每条路线各有 RNG，避免 weak/strong 相互影响，也避免 warmup 复用输出。
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

        def run():
            batches = []
            for begin in range(0, count, args.batch_size):
                images = to_float_batch(raw[begin : begin + args.batch_size], device)
                batches.append(normalize(augment(images, generator)).cpu())
            return torch.cat(batches)

        return run

    print("\n=== Benchmark（均值） ===")
    pil_weak, pil_weak_seconds = benchmark(
        "CPU/PIL weak", pil_weak_route, device, args.warmup, args.repeats
    )
    pil_strong, pil_strong_seconds = benchmark(
        "CPU/PIL strong", pil_strong_route, device, args.warmup, args.repeats
    )
    torch_weak, torch_weak_seconds = benchmark(
        "Torch weak", torch_route(torch_weak_augment), device, args.warmup, args.repeats
    )
    torch_strong, torch_strong_seconds = benchmark(
        "Torch strong", torch_route(torch_strong_augment), device, args.warmup, args.repeats
    )

    print("\n=== 输出分布 ===")
    stats("CPU/PIL weak", pil_weak)
    stats("CPU/PIL strong", pil_strong)
    stats("Torch weak", torch_weak)
    stats("Torch strong", torch_strong)

    print("\n=== 同一原图上的数值差异（非逐像素一致性测试） ===")
    print("注：随机流和算子实现均不同；以下绝对差仅用于辅助观察分布，不能衡量实现正确性。")
    diff_stats("weak: PIL vs Torch", pil_weak, torch_weak)
    diff_stats("strong: PIL vs Torch", pil_strong, torch_strong)

    print("\n=== 吞吐 ===")
    for name, seconds in (
        ("CPU/PIL weak", pil_weak_seconds),
        ("CPU/PIL strong", pil_strong_seconds),
        ("Torch weak", torch_weak_seconds),
        ("Torch strong", torch_strong_seconds),
    ):
        print(f"{name:14s}: {count / seconds:,.1f} samples/s ({seconds:.3f} s/run)")


if __name__ == "__main__":
    main()
