import atexit
import copy
import dataclasses
import json
import logging
import os
import queue
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from Dataset.CINIC10 import CINIC10
from Dataset.dataset import (
    Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch,
    SharedImageDataset,
    classify_label,
    partition_train,
    show_clients_data_distribution,
)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from Model.resnet import ResNet_PC
from options import args_parser

logger = logging.getLogger(__name__)


class Global:
    """服务端状态：维护全局模型并执行标准的 FedAvg 参数加权聚合与评估。"""

    def __init__(self, args):
        self.args = args
        self.gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = ResNet_PC(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
        self.model.to(self.device)
        self.num_classes = args.num_classes

    def aggregate(
        self,
        list_dicts_local_params: list[dict[str, torch.Tensor]],
        list_nums_local_data: list[int],
    ):
        """标准的 FedAvg 加权聚合算法。"""
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        total_samples = sum(list_nums_local_data)

        for name_param in list_dicts_local_params[0]:
            first_value = list_dicts_local_params[0][name_param]
            # BatchNorm 的整数统计量 (如 num_batches_tracked) 不能参与小数加权平均，保留第一个客户端的值
            if not torch.is_floating_point(first_value):
                fedavg_global_params[name_param] = first_value.clone()
                continue

            list_values_param = [
                dict_local_params[name_param] * num_local_data
                for dict_local_params, num_local_data in zip(
                    list_dicts_local_params, list_nums_local_data
                )
            ]
            value_global_param = (
                torch.stack(list_values_param).sum(dim=0) / total_samples
            )
            fedavg_global_params[name_param] = value_global_param

        # 解除对 fedavg_eval 的隐式依赖，聚合后直接更新全局模型
        self.model.load_state_dict(fedavg_global_params)
        return fedavg_global_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        """在全局测试集上评估模型精度。"""
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(
                data_test,
                batch_size_test,
            )
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, outputs = self.model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.eq(predicts.cpu(), labels.cpu()).sum().item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        """将全局模型参数提取至 CPU，供客户端分发任务。"""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def aggregate_prototypes(self, prototypes, counts, previous=None):
        """按类别样本数聚合本地原型；本轮缺失类别沿用旧原型。"""
        device = self.device
        total = torch.zeros(self.num_classes, device=device)
        weighted = torch.zeros_like(prototypes[0], device=device)
        for proto, count in zip(prototypes, counts):
            proto, count = proto.to(device), count.to(device)
            weighted += proto * count.unsqueeze(1)
            total += count
        result = weighted / total.clamp_min(1).unsqueeze(1)
        if previous is not None:
            valid = total > 0
            result[~valid] = previous.to(device)[~valid]
        return result.cpu()

    def learn_anchors(self, prototypes, counts, initial_prototypes):
        """以聚合原型初始化可学习锚点，并用全部局部原型进行 L2 对比学习。"""
        anchor = torch.nn.Parameter(initial_prototypes.to(self.device).clone())
        optimizer = SGD([anchor], lr=self.args.anchor_lr)
        local_prototypes = []
        local_labels = []
        for client_proto, client_count in zip(prototypes, counts):
            valid = client_count > 0
            local_prototypes.append(client_proto[valid].to(self.device))
            local_labels.append(torch.arange(self.num_classes)[valid].to(self.device))
        if not local_prototypes:
            return initial_prototypes.detach().cpu().clone()
        local_prototypes = torch.cat(local_prototypes)
        local_labels = torch.cat(local_labels)
        for _ in range(self.args.anchor_steps):
            distances = torch.cdist(local_prototypes, anchor, p=2)
            positive = distances[
                torch.arange(distances.size(0), device=self.device), local_labels
            ]
            negative_mask = F.one_hot(local_labels, self.num_classes).bool()
            negative = distances.masked_fill(negative_mask, float("inf"))
            loss = (
                positive.square().mean()
                + F.relu(self.args.anchor_margin - negative)
                .square()
                .masked_fill(negative_mask, 0)
                .mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return anchor.detach().cpu().clone()


class Local:
    """客户端本地半监督训练器 (标准 FixMatch)。"""

    def __init__(self, args, device=None):
        self.device = device or torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        self.local_model = ResNet_PC(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
        self.local_model.to(self.device)

        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.num_classes = args.num_classes

    def train(
        self,
        args,
        data_client_labeled,
        data_client_unlabeled,
        global_params,
        global_anchors=None,
    ):
        """标准的 FixMatch 客户端半监督训练：监督交叉熵损失 + 弱强一致性伪标签损失。"""
        train_start = time.perf_counter()
        labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
        )

        unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
        )

        # 载入本轮最新的全局参数
        self.local_model.load_state_dict(global_params)
        # 彻底清空 SGD 动量，保证各客户端状态隔离
        self.optimizer.state.clear()
        self.local_model.train()

        # 初始化伪标签指标统计变量
        num_pseudo_corrects = 0
        num_pseudo_total = 0
        num_u_valid = 0
        pseudo_client_acc = 0.0
        u_client_valid = 0.0
        # 每个本地 epoch 按无标签 DataLoader 的实际批次数训练。
        local_iter = len(unlabeled_trainloader)

        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(labeled_trainloader)
            unlabeled_iter = iter(unlabeled_trainloader)

            for _ in range(local_iter):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = next(labeled_iter)

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)

                inputs_x = inputs_x.to(self.device)
                inputs_u_w = inputs_u_w.to(self.device)
                inputs_u_s = inputs_u_s.to(self.device)
                targets_x = targets_x.to(self.device)
                targets_u_groundtruth = targets_u_groundtruth.to(self.device)

                batch_size = inputs_x.shape[0]
                # 交错输入以保持 BatchNorm 统计量平稳
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                )

                features, logits = self.local_model(inputs)
                features = self.de_interleave(features, 2 * args.mu + 1)
                logits = self.de_interleave(logits, 2 * args.mu + 1)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                features_x = features[:batch_size]

                # 1. 有标签监督交叉熵损失
                Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

                # 2. 无标签弱增强生成伪标签 (Stop Gradient)
                with torch.no_grad():
                    pseudo_label = torch.softmax(logits_u_w / args.T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(args.threshold).float()

                # 3. 无标签强增强的一致性预测损失 (标准 CrossEntropy + 置信度阈值 Mask)
                Lu = (
                    F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
                ).mean()

                L_proto = torch.zeros((), device=self.device)
                if global_anchors is not None:
                    target_x = global_anchors.to(self.device)[targets_x]
                    L_proto = F.mse_loss(features_x, target_x)
                    features_u_w, _ = features[batch_size:].chunk(2)
                    valid = mask.bool()
                    if valid.any():
                        target_u = global_anchors.to(self.device)[targets_u[valid]]
                        L_proto = L_proto + F.mse_loss(features_u_w[valid], target_u)

                # 总损失 (纯净的 FixMatch 目标函数)
                loss = Lx + args.lambda_u * Lu + args.lambda_proto * L_proto

                # 统计最后一个 epoch 的伪标签质量
                if local_epoch + 1 == args.local_epochs:
                    num_pseudo_corrects += (
                        torch.eq(targets_u.cpu(), targets_u_groundtruth.cpu())
                        .sum()
                        .item()
                    )
                    num_pseudo_total += len(targets_u)
                    num_u_valid += int(mask.sum().item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if local_epoch + 1 == args.local_epochs:
                pseudo_client_acc = (
                    num_pseudo_corrects / num_pseudo_total if num_pseudo_total else 0.0
                )
                u_client_valid = (
                    num_u_valid / num_pseudo_total if num_pseudo_total else 0.0
                )

        pseudo_status = [
            num_pseudo_total,
            num_pseudo_corrects,
            num_u_valid,
            pseudo_client_acc,
            u_client_valid,
        ]
        prototypes, prototype_counts = self.compute_prototypes(
            args, data_client_labeled, data_client_unlabeled
        )
        return (
            {
                name: value.detach().cpu().clone()
                for name, value in self.local_model.state_dict().items()
            },
            pseudo_status,
            prototypes,
            prototype_counts,
            time.perf_counter() - train_start,
        )

    @torch.no_grad()
    def compute_prototypes(self, args, labeled_dataset, unlabeled_dataset):
        """用本地训练完成后的最终模型计算本地类别原型。"""
        self.local_model.eval()
        sums = torch.zeros(args.num_classes, self.local_model.dim, device=self.device)
        counts = torch.zeros(args.num_classes, device=self.device)
        loader = DataLoader(
            # 训练视图包含 repeat=2000；原型只需对每个原始有标签样本计算一次。
            Subset(labeled_dataset, range(len(labeled_dataset.indices))),
            args.batch_size_local_labeled_fixmatch,
        )
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            features, _ = self.local_model(images)
            for c in range(args.num_classes):
                valid = labels == c
                if valid.any():
                    sums[c] += features[valid].sum(0)
                    counts[c] += valid.sum()
        loader = DataLoader(
            unlabeled_dataset,
            args.batch_size_local_labeled_fixmatch,
            shuffle=False,
        )
        for images, _, _ in loader:
            images = images.to(self.device)
            features, logits = self.local_model(images)
            confidence, labels = torch.softmax(logits / args.T, -1).max(1)
            for c in range(args.num_classes):
                valid = (labels == c) & (confidence >= args.threshold)
                if valid.any():
                    sums[c] += features[valid].sum(0)
                    counts[c] += valid.sum()
        return (sums / counts.clamp_min(1).unsqueeze(1)).cpu(), counts.cpu()

    def interleave(self, x, size):
        """将有标签和无标签样本交错排列，以配合 BatchNorm 训练。"""
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        """恢复交错前的样本顺序。"""
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


@dataclasses.dataclass
class ClientTask:
    """主进程下发给客户端 Worker 的单次训练任务。"""

    round: int
    client_id: int
    labeled_indices: list[int]
    unlabeled_indices: list[int]
    global_params: dict[str, torch.Tensor]
    global_anchors: torch.Tensor | None
    args: object


@dataclasses.dataclass
class ClientResult:
    """客户端 Worker 训练结束返回的结果对象。"""

    ok: bool
    client_id: int
    gpu_id: int
    params: dict[str, torch.Tensor] | None = None
    num_samples: int = 0
    pseudo_status: list[float] | None = None
    prototypes: torch.Tensor | None = None
    prototype_counts: torch.Tensor | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


def preload_shared_dataset(dataset):
    """一次性读取原始图片，并将图片和标签放入 CPU 共享内存。"""
    first_image, _ = dataset[0]
    first_image = np.asarray(first_image, dtype=np.uint8)
    images = np.empty((len(dataset), *first_image.shape), dtype=np.uint8)
    labels = np.empty(len(dataset), dtype=np.int64)

    images[0] = first_image
    labels[0] = dataset[0][1]
    for index in range(1, len(dataset)):
        image, label = dataset[index]
        images[index] = np.asarray(image, dtype=np.uint8)
        labels[index] = int(label)

    shared_images = torch.from_numpy(images).share_memory_()
    shared_labels = torch.from_numpy(labels).share_memory_()
    logger.info(
        "已将 %d 张原始图片加载到 CPU 共享内存，形状为 %s",
        len(dataset),
        tuple(shared_images.shape),
    )
    return SharedImageDataset(shared_images, shared_labels)


def _client_worker(
    gpu_id, args, shared_dataset, task_queue, result_queue, log_file=None
):
    """Worker 进程：在固定 GPU 上循环领取任务并执行客户端训练。"""
    try:
        # spawn 创建的子进程需要配置 logger，保证客户端训练日志正常输出
        if log_file:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                filename=log_file,
                force=True,
            )

        mp.set_sharing_strategy("file_system")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        local = Local(args, device=device)

        while True:
            task = task_queue.get()
            if task is None:
                return

            try:
                seed = task.args.seed + task.round * 100_000 + task.client_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                labeled_view = Indices2Dataset_labeled(shared_dataset)
                labeled_view.load(task.labeled_indices)
                unlabeled_view = Indices2Dataset_unlabeled_fixmatch(shared_dataset)
                unlabeled_view.load(task.unlabeled_indices)

                (
                    params,
                    pseudo_status,
                    prototypes,
                    prototype_counts,
                    elapsed_seconds,
                ) = local.train(
                    task.args,
                    labeled_view,
                    unlabeled_view,
                    task.global_params,
                    task.global_anchors,
                )
                result_queue.put(
                    ClientResult(
                        ok=True,
                        client_id=task.client_id,
                        gpu_id=gpu_id,
                        params=params,
                        num_samples=len(task.labeled_indices) * labeled_view.repeat
                        + len(task.unlabeled_indices),
                        pseudo_status=pseudo_status,
                        prototypes=prototypes,
                        prototype_counts=prototype_counts,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
            except (
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                AttributeError,
            ) as exc:
                result_queue.put(
                    ClientResult(
                        ok=False,
                        client_id=task.client_id,
                        gpu_id=gpu_id,
                        error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                )
    except (
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OSError,
    ) as exc:
        result_queue.put(
            ClientResult(
                ok=False,
                client_id=-1,
                gpu_id=gpu_id,
                error=f"Worker initialization failed ({type(exc).__name__}: {exc}):\n{traceback.format_exc()}",
            )
        )


class ClientWorkerPool:
    """管理客户端训练进程池，支持多卡/单卡多进程调度。"""

    def __init__(self, gpu_ids, args, shared_dataset, log_file=None):
        mp.set_sharing_strategy("file_system")
        self.gpu_ids = list(gpu_ids)
        self.args = copy.deepcopy(args)
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.closed = False
        atexit.register(self.terminate)

        for gpu_id in self.gpu_ids:
            process = self.ctx.Process(
                target=_client_worker,
                args=(
                    gpu_id,
                    copy.deepcopy(args),
                    shared_dataset,
                    self.task_queue,
                    self.result_queue,
                    log_file,
                ),
                name=f"fssl-client-gpu-{gpu_id}",
            )
            process.start()
            self.processes.append(process)

    def run_round(self, tasks):
        """提交一轮客户端任务并收集按 client_id 排序后的结果。"""
        tasks = list(tasks)
        for task in tasks:
            self.task_queue.put(task)

        results = []
        try:
            for _ in tasks:
                while True:
                    try:
                        result = self.result_queue.get(timeout=5)
                        break
                    except queue.Empty:
                        dead = [
                            process.name
                            for process in self.processes
                            if not process.is_alive()
                        ]
                        if dead:
                            raise RuntimeError(
                                "client worker exited without returning a result: "
                                + ", ".join(dead)
                            )
                results.append(result)
                if not result.ok:
                    raise RuntimeError(
                        f"client {result.client_id} failed on GPU {result.gpu_id}:\n"
                        f"{result.error}"
                    )
        except Exception as exc:
            round_id = tasks[0].round if tasks else -1
            failure_dir = os.path.join(
                "results", self.args.dataset, "parallel_failures"
            )
            os.makedirs(failure_dir, exist_ok=True)
            failure_file = os.path.join(
                failure_dir,
                f"failure_round_{round_id}_{time.strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(failure_file, "w", encoding="utf8") as file:
                json.dump(
                    {
                        "round": round_id,
                        "error": str(exc),
                        "tasks": [task.client_id for task in tasks],
                        "args": vars(self.args),
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            self.terminate()
            raise

        return sorted(results, key=lambda result: result.client_id)

    def close(self):
        """发送退出信号并回收所有 Worker。"""
        if self.closed:
            return
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()
        self.closed = True

    def terminate(self):
        """强制终止 Worker 进程。"""
        if self.closed:
            return
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        self.closed = True


def _parse_worker_gpus(args):
    """解析 GPU 和进程数配置。"""
    if args.client_gpus:
        try:
            gpu_ids = [int(value.strip()) for value in args.client_gpus.split(",")]
        except ValueError as exc:
            raise ValueError("--client_gpus 必须是逗号分隔的 GPU 整数列表") from exc
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--client_gpus 至少需要包含一个不重复的 GPU 编号")
    else:
        gpu_ids = [args.gpu_id]

    if not torch.cuda.is_available():
        raise RuntimeError("多 GPU 训练需要可用的 CUDA 环境")

    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"GPU 编号 {invalid} 不可用；当前可见 GPU 数量为 {device_count}"
        )

    args.server_gpu = args.server_gpu if args.server_gpu is not None else gpu_ids[0]
    if args.server_gpu < 0 or args.server_gpu >= device_count:
        raise ValueError(f"--server_gpu {args.server_gpu} 超出可用 GPU 范围")

    if args.gpu_processes:
        process_counts = {}
        try:
            for item in args.gpu_processes.split(","):
                gpu_text, count_text = item.split(":", 1)
                gpu_id, count = int(gpu_text), int(count_text)
                if count < 1 or gpu_id in process_counts:
                    raise ValueError
                process_counts[gpu_id] = count
        except ValueError as exc:
            raise ValueError(
                "--gpu_processes 格式必须为 GPU:进程数，例如 0:2,1:1"
            ) from exc
        if set(process_counts) != set(gpu_ids):
            raise ValueError(
                "--gpu_processes 必须为每个 --client_gpus 中的 GPU 指定进程数"
            )
    else:
        process_counts = {gpu_id: 1 for gpu_id in gpu_ids}
        if args.max_parallel_clients is not None:
            if args.max_parallel_clients < 1:
                raise ValueError("--max_parallel_clients 必须大于 0")
            gpu_ids = gpu_ids[: args.max_parallel_clients]
            process_counts = {gpu_id: 1 for gpu_id in gpu_ids}

    return [gpu_id for gpu_id in gpu_ids for _ in range(process_counts[gpu_id])]


def fedavg_fixmatch(alpha, args=None):
    """执行纯净的联邦半监督学习 (FedAvg + FixMatch)。"""
    if args is None:
        args = args_parser()
    args.method = "FedAvg_FixMatch"

    log_dir = f"./results/{args.dataset}/logs"
    os.makedirs(log_dir, exist_ok=True)
    cr_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_file = os.path.join(log_dir, f"{args.method}_α={alpha}_{cr_time}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        force=True,
    )

    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        data_local_training = datasets.CIFAR10(
            args.path, train=True, download=True, transform=None
        )
        data_global_test = datasets.CIFAR10(
            args.path, train=False, transform=transform_test
        )

    elif args.dataset == "CIFAR100":
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        data_local_training = datasets.CIFAR100(
            args.path, train=True, download=True, transform=None
        )
        data_global_test = datasets.CIFAR100(
            args.path, train=False, transform=transform_test
        )

    elif args.dataset == "SVHN":
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                ),
            ]
        )
        data_local_training = datasets.SVHN(
            args.path, split="train", download=True, transform=None
        )
        data_global_test = datasets.SVHN(
            args.path, split="test", transform=transform_test, download=True
        )

    elif args.dataset == "CINIC10":
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)
                ),
            ]
        )
        data_local_training = CINIC10(root=args.path, split="train", transform=None)
        data_global_test = CINIC10(
            root=args.path, split="test", transform=transform_test
        )

    else:
        print(f"Error: Unsupported dataset {args.dataset}.")
        sys.exit(1)

    logger.info(
        f"dataset:{args.dataset}\n"
        f"num_classes:{args.num_classes}\n"
        f"num_labeled:{args.num_labeled}\n"
        f"non_iid:{args.alpha}\n"
        f"mu:{args.mu}\n"
        f"num_rounds:{args.num_rounds}\n"
        f"batch_labeled:{args.batch_size_local_labeled_fixmatch}\n"
        f"batch_unlabeled:{args.batch_size_local_labeled_fixmatch * args.mu}"
    )

    random_state = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(
        list_label2indices, args.num_labeled
    )

    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(
            list_label2indices=list_label2indices_labeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
        )
        list_client2indices_unlabeled = clients_indices_homo(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
        )
    else:
        list_client2indices_labeled = clients_indices(
            list_label2indices=list_label2indices_labeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=0,
        )
        list_client2indices_unlabeled = clients_indices(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=0,
        )

    show_clients_data_distribution(
        data_local_training,
        list_client2indices_labeled,
        list_client2indices_unlabeled,
        args.num_classes,
    )

    for client in range(args.num_clients):
        list_client2indices_unlabeled[client] = np.concatenate(
            [
                np.asarray(list_client2indices_unlabeled[client]),
                np.asarray(list_client2indices_labeled[client]),
            ]
        )

    client_gpus = _parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    global_model = Global(args)

    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(data_local_training)
    worker_pool = ClientWorkerPool(client_gpus, args, shared_dataset, log_file=log_file)

    total_clients = list(range(args.num_clients))

    fedavg_acc: list[float] = []
    fedavg_pseudo_acc: list[float] = []
    fedavg_num_valid: list[int] = []
    fedavg_valid_ratio: list[float] = []
    # 第一轮尚未有服务器端原型，聚合时作为 previous 传入 None。
    global_prototypes = None
    global_anchors = None

    progress = tqdm(range(1, args.num_rounds + 1), desc="Test")
    for r in progress:
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(
            total_clients, args.num_online_clients, replace=False
        )

        num_clients_u_total = 0
        num_clients_u_corrects = 0
        num_clients_u_valid = 0

        tasks = [
            ClientTask(
                round=r,
                client_id=int(client),
                labeled_indices=(
                    list_client2indices_labeled[client].tolist()
                    if hasattr(list_client2indices_labeled[client], "tolist")
                    else list(list_client2indices_labeled[client])
                ),
                unlabeled_indices=(
                    list_client2indices_unlabeled[client].tolist()
                    if hasattr(list_client2indices_unlabeled[client], "tolist")
                    else list(list_client2indices_unlabeled[client])
                ),
                global_params=dict_global_params,
                global_anchors=global_anchors,
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        list_dicts_local_params = [result.params for result in results]
        list_nums_local_data = [result.num_samples for result in results]
        global_prototypes = global_model.aggregate_prototypes(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            global_prototypes,
        )
        global_anchors = global_model.learn_anchors(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            global_prototypes,
        )

        for result in results:
            pseudo_status = result.pseudo_status
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]

        pseudo_acc = (
            num_clients_u_corrects / num_clients_u_total if num_clients_u_total else 0.0
        )
        pseudo_valid_ratio = (
            num_clients_u_valid / num_clients_u_total if num_clients_u_total else 0.0
        )
        fedavg_pseudo_acc.append(pseudo_acc)
        fedavg_valid_ratio.append(pseudo_valid_ratio)
        fedavg_num_valid.append(num_clients_u_valid)

        # 执行纯净的标准 FedAvg 聚合
        fedavg_params = global_model.aggregate(
            list_dicts_local_params, list_nums_local_data
        )

        # 评估全局模型
        global_acc = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test
        )
        fedavg_acc.append(global_acc)

        progress.set_postfix(acc=f"{global_acc:.2%}")

        result_dir = f"./results/{args.dataset}"
        os.makedirs(result_dir, exist_ok=True)
        result_dir_spec = f"{result_dir}/{args.method}_α={alpha}_{cr_time}"
        os.makedirs(result_dir_spec, exist_ok=True)

        if (
            r == 1
            or r == args.num_rounds
            or (r % 50 == 0 and r > 0.8 * args.num_rounds)
        ):
            torch.save(fedavg_params, f"{result_dir_spec}/fedavg_params_round_{r}.pth")
            print(f"Saved model for round {r}")

        result_file = f"{result_dir}/{args.method}_α={alpha}_{cr_time}.csv"
        acc_df = pd.DataFrame(
            {"acc": fedavg_acc}, index=list(range(1, len(fedavg_acc) + 1))
        )
        acc_df.to_csv(result_file, encoding="utf8")

        result_pseudo_file = (
            f"{result_dir}/{args.method}_α={alpha}_pseudo_{cr_time}.csv"
        )
        min_length = min(
            len(fedavg_pseudo_acc),
            len(fedavg_valid_ratio),
            len(fedavg_num_valid),
            len(fedavg_acc),
        )
        metrics_df = pd.DataFrame(
            {
                "round": list(range(1, min_length + 1)),
                "acc": fedavg_acc[:min_length],
                "pseudo_acc": fedavg_pseudo_acc[:min_length],
                "valid_ratio": fedavg_valid_ratio[:min_length],
                "num_valid": fedavg_num_valid[:min_length],
            }
        )
        metrics_df.set_index("round", inplace=True)
        metrics_df.to_csv(result_pseudo_file, encoding="utf8")

    worker_pool.close()


if __name__ == "__main__":
    args = args_parser()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    fedavg_fixmatch(args.alpha, args)
