import copy
import logging
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from Dataset.CINIC10 import CINIC10
from Dataset.dataset import (
    classify_label,
    partition_train,
    show_clients_data_distribution,
)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from Model.factory import build_model
from options import args_parser
from utils.client_pool import (
    ClientTask,
    ClientWorkerPool,
    parse_worker_gpus,
    preload_shared_dataset,
    run_main,
)
from utils.logging_setup import log_args, setup_logging
from utils.run_registry import create_run

logger = logging.getLogger(__name__)

DATASET_STATS = {
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "CINIC10": ((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
}


def evaluation_transform(dataset):
    mean, std = DATASET_STATS[dataset]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class IndexedEvalDataset(Dataset):
    """共享原始数据的确定性按索引评估视图。"""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label

    def __len__(self):
        return len(self.indices)


def dist_contrastive_loss(
    features,
    prototypes,
    targets,
    margin=0.0,
    temperature=1.0,
):
    """距离对比损失，支持单标签 ``[B]`` 或候选标签集合 ``[B, C]``。"""
    single_label = targets.ndim == 1
    candidate_mask = (
        F.one_hot(targets, prototypes.size(0)).bool()
        if single_label
        else targets.to(device=features.device, dtype=torch.bool)
    )
    distances = torch.cdist(features, prototypes.to(features.device), p=2.0)
    if margin:
        distances = distances + candidate_mask.to(distances.dtype) * margin
    logits = -distances / temperature
    if single_label:
        return F.cross_entropy(logits, targets)

    valid = candidate_mask.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0

    valid_logits = logits[valid]
    valid_candidates = candidate_mask[valid]
    log_all = torch.logsumexp(valid_logits, dim=1)
    log_positive = torch.logsumexp(
        valid_logits.masked_fill(~valid_candidates, float("-inf")),
        dim=1,
    )
    return (log_all - log_positive).sum() / features.size(0)


class Global:
    def __init__(self, args):
        self.args = args
        gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = build_model(args).to(self.device)
        self.num_classes = args.num_classes

    def aggregate(self, local_params, sample_counts):
        result = copy.deepcopy(local_params[0])
        total = sum(sample_counts)
        for name, first in local_params[0].items():
            if not torch.is_floating_point(first):
                result[name] = first.clone()
                continue
            result[name] = (
                sum(
                    params[name] * count
                    for params, count in zip(local_params, sample_counts)
                )
                / total
            )
        self.model.load_state_dict(result)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        params,
        dataset,
        batch_size,
        prototypes=None,
        prototype_mask=None,
    ):
        """在测试集上同时评估分类头和原始聚合原型。"""
        self.model.load_state_dict(params)
        self.model.eval()
        model_correct = 0
        prototype_correct = 0
        raw_rescued = 0
        low_conf_total = 0
        low_conf_model_wrong = 0
        proto_rescued_low_conf = 0
        model_topk_low_conf = [0, 0, 0]
        proto_topk_low_conf = [0, 0, 0]
        union_topk_low_conf = [0, 0, 0]
        complement_low_conf = [0, 0, 0, 0]  # MM, MP, PM, PP
        prototypes = prototypes.to(self.device) if prototypes is not None else None
        valid = (
            prototype_mask.to(self.device).bool()
            if prototype_mask is not None
            else None
        )
        has_valid_prototypes = (
            prototypes is not None and valid is not None and bool(valid.any())
        )

        def nearest(features, centers, center_mask=None):
            distances = torch.cdist(features, centers).square() / features.size(1)
            if center_mask is not None:
                distances = distances.masked_fill(
                    ~center_mask.unsqueeze(0), float("inf")
                )
            return distances.argmin(dim=1)

        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(self.device), labels.to(self.device)
            features, logits = self.model(images)
            probs = torch.softmax(logits / self.args.T, dim=-1)
            confidence, prediction = probs.max(dim=1)
            model_is_correct = prediction == labels
            model_correct += model_is_correct.sum().item()
            if has_valid_prototypes:
                distances = torch.cdist(features, prototypes).square() / features.size(
                    1
                )
                distances = distances.masked_fill(~valid.unsqueeze(0), float("inf"))
                proto_order = distances.argsort(dim=1)
                proto_prediction = proto_order[:, 0]
                prototype_is_correct = proto_prediction == labels
                prototype_correct += prototype_is_correct.sum().item()
                raw_rescued += (prototype_is_correct & ~model_is_correct).sum().item()
            if has_valid_prototypes:
                low_conf = confidence < self.args.threshold
                low_wrong = low_conf & ~model_is_correct
                low_conf_total += low_conf.sum().item()
                low_conf_model_wrong += low_wrong.sum().item()
                proto_rescued_low_conf += (
                    (low_wrong & prototype_is_correct).sum().item()
                )
                model_order = logits.argsort(dim=1, descending=True)
                for index, k in enumerate((1, 2, 3)):
                    model_hits = (model_order[:, :k] == labels.unsqueeze(1)).any(dim=1)
                    proto_hits = (proto_order[:, :k] == labels.unsqueeze(1)).any(dim=1)
                    union_hits = model_hits | proto_hits
                    model_topk_low_conf[index] += (model_hits & low_conf).sum().item()
                    proto_topk_low_conf[index] += (proto_hits & low_conf).sum().item()
                    union_topk_low_conf[index] += (union_hits & low_conf).sum().item()
                mm = low_conf & model_is_correct & prototype_is_correct
                mp = low_conf & model_is_correct & ~prototype_is_correct
                pm = low_conf & ~model_is_correct & prototype_is_correct
                pp = low_conf & ~model_is_correct & ~prototype_is_correct
                complement_low_conf[0] += mm.sum().item()
                complement_low_conf[1] += mp.sum().item()
                complement_low_conf[2] += pm.sum().item()
                complement_low_conf[3] += pp.sum().item()

        total = len(dataset)
        model_errors = total - model_correct
        result = {
            "global_test_acc": model_correct / total,
            "global_test_raw_prototype_acc": (
                prototype_correct / total if has_valid_prototypes else np.nan
            ),
            "global_test_raw_prototype_correct_given_model_wrong": (
                raw_rescued / model_errors
                if has_valid_prototypes and model_errors
                else np.nan
            ),
        }
        if has_valid_prototypes and low_conf_total:
            result.update(
                {
                    "global_test_low_conf_total": low_conf_total,
                    "global_test_low_conf_model_wrong": low_conf_model_wrong,
                    "global_test_proto_rescue_low_conf": (
                        proto_rescued_low_conf / low_conf_model_wrong
                        if low_conf_model_wrong
                        else np.nan
                    ),
                    "global_test_proto_rescued_low_conf": proto_rescued_low_conf,
                    "global_test_model_top1_low_conf": model_topk_low_conf[0]
                    / low_conf_total,
                    "global_test_model_top2_low_conf": model_topk_low_conf[1]
                    / low_conf_total,
                    "global_test_model_top3_low_conf": model_topk_low_conf[2]
                    / low_conf_total,
                    "global_test_proto_top1_low_conf": proto_topk_low_conf[0]
                    / low_conf_total,
                    "global_test_proto_top2_low_conf": proto_topk_low_conf[1]
                    / low_conf_total,
                    "global_test_proto_top3_low_conf": proto_topk_low_conf[2]
                    / low_conf_total,
                    "global_test_union_top2_low_conf": union_topk_low_conf[1]
                    / low_conf_total,
                    "global_test_union_top3_low_conf": union_topk_low_conf[2]
                    / low_conf_total,
                    "global_test_low_conf_MM": complement_low_conf[0],
                    "global_test_low_conf_MP": complement_low_conf[1],
                    "global_test_low_conf_PM": complement_low_conf[2],
                    "global_test_low_conf_PP": complement_low_conf[3],
                }
            )
        else:
            for name in (
                "low_conf_total",
                "low_conf_model_wrong",
                "proto_rescued_low_conf",
                "model_top1_low_conf",
                "model_top2_low_conf",
                "model_top3_low_conf",
                "proto_top1_low_conf",
                "proto_top2_low_conf",
                "proto_top3_low_conf",
                "union_top2_low_conf",
                "union_top3_low_conf",
                "low_conf_MM",
                "low_conf_MP",
                "low_conf_PM",
                "low_conf_PP",
            ):
                result[f"global_test_{name}"] = np.nan
        return result

    def download_params(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def aggregate_prototypes(
        self, prototypes, counts, previous=None, previous_mask=None
    ):
        """按类别样本数聚合；本轮缺失的类别沿用历史原型。"""
        total = torch.zeros(self.num_classes, device=self.device)
        weighted = torch.zeros_like(prototypes[0], device=self.device)
        for local_prototypes, local_counts in zip(prototypes, counts):
            local_prototypes = local_prototypes.to(self.device)
            local_counts = local_counts.to(self.device)
            weighted += local_prototypes * local_counts.unsqueeze(1)
            total += local_counts
        result = weighted / total.clamp_min(1).unsqueeze(1)
        valid = total > 0
        if previous is not None and previous_mask is not None:
            carry = ~valid & previous_mask.to(self.device)
            result[carry] = previous.to(self.device)[carry]
            valid |= previous_mask.to(self.device)
        return result.cpu(), valid.cpu()


class Local:
    """使用监督 CE、高置信伪标签 CE 和平均原型对比损失训练。"""

    def __init__(self, args, device=None):
        self.device = device or torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.local_model = build_model(args).to(self.device)
        self.global_model = build_model(args).to(self.device)
        self.optimizer = SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def train(
        self,
        args,
        labeled_dataset,
        u_pool_dataset,
        eval_labeled_dataset,
        global_params,
        global_prototypes=None,
        global_prototype_mask=None,
        global_prototype_max_radius=None,
    ):
        start = time.perf_counter()
        labeled_loader = DataLoader(
            labeled_dataset,
            sampler=RandomSampler(labeled_dataset),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
        )
        u_pool_loader = DataLoader(
            u_pool_dataset,
            sampler=RandomSampler(u_pool_dataset),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
        )
        self.local_model.load_state_dict(global_params)
        self.global_model.load_state_dict(global_params)
        self.global_model.eval()
        self.optimizer.state.clear()
        self.local_model.train()
        has_prototypes = (
            global_prototypes is not None and global_prototype_mask is not None
        )
        training_prototypes = (
            global_prototypes.to(self.device) if has_prototypes else None
        )
        prototype_mask = (
            global_prototype_mask.to(self.device).bool() if has_prototypes else None
        )
        prototype_max_radius = (
            torch.as_tensor(
                global_prototype_max_radius,
                device=self.device,
                dtype=torch.float32,
            )
            if has_prototypes and global_prototype_max_radius is not None
            else None
        )
        set_size_hist = torch.zeros(args.num_classes + 1, device=self.device)
        set_size_hits = torch.zeros(args.num_classes + 1, device=self.device)
        low_total = 0
        true_label_in_set = 0
        local_steps = int(len(u_pool_dataset) / args.batch_size_local_labeled_fixmatch)

        for local_epoch in range(args.local_epochs):
            labeled_iter, u_pool_iter = iter(labeled_loader), iter(u_pool_loader)
            for _ in range(local_steps):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)
                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(u_pool_iter)
                except StopIteration:
                    u_pool_iter = iter(u_pool_loader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(u_pool_iter)

                inputs_x = inputs_x.to(self.device)
                targets_x = targets_x.to(self.device)
                inputs_u_w = inputs_u_w.to(self.device)
                inputs_u_s = inputs_u_s.to(self.device)
                targets_u_groundtruth = targets_u_groundtruth.to(self.device)
                batch_size = inputs_x.size(0)
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                    2 * args.mu + 1,
                )
                features, logits = self.local_model(inputs)
                features = self.de_interleave(features, 2 * args.mu + 1)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                features_x = features[:batch_size]
                features_u_w, features_u_s = features[batch_size:].chunk(2)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                supervised_loss = F.cross_entropy(logits_x, targets_x)

                with torch.no_grad():
                    local_confidence = torch.softmax(logits_u_w / args.T, dim=-1).amax(
                        dim=-1
                    )
                    _, global_logits = self.global_model(inputs_u_w)
                    global_probs = torch.softmax(global_logits / args.T, dim=-1)
                    global_confidence, pseudo_targets = global_probs.max(dim=-1)
                    mask = (
                        local_confidence.ge(args.threshold)
                        | global_confidence.ge(args.threshold)
                    ).float()

                unsupervised_loss = (
                    F.cross_entropy(
                        logits_u_s,
                        pseudo_targets,
                        reduction="none",
                    )
                    * mask
                ).mean()

                labeled_prototype_loss = torch.zeros((), device=self.device)
                high_u_prototype_loss = torch.zeros((), device=self.device)
                if has_prototypes:
                    valid_classes = prototype_mask.nonzero(as_tuple=True)[0]
                    class_remap = torch.full(
                        (args.num_classes,),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    class_remap[valid_classes] = torch.arange(
                        valid_classes.numel(), device=self.device
                    )

                    valid_labeled = prototype_mask[targets_x]
                    if valid_labeled.any():
                        labeled_prototype_loss = dist_contrastive_loss(
                            features_x[valid_labeled],
                            training_prototypes[valid_classes],
                            class_remap[targets_x[valid_labeled]],
                            args.prototype_contrastive_margin,
                        )

                    high_valid = mask.bool() & prototype_mask[pseudo_targets]
                    if high_valid.any():
                        accepted_loss = dist_contrastive_loss(
                            features_u_s[high_valid],
                            training_prototypes[valid_classes],
                            class_remap[pseudo_targets[high_valid]],
                            args.prototype_contrastive_margin,
                        )
                        high_u_prototype_loss = accepted_loss * (
                            high_valid.sum() / high_valid.numel()
                        )

                if (
                    local_epoch + 1 == args.local_epochs
                    and prototype_max_radius is not None
                ):
                    low_mask = ~mask.bool()
                    if low_mask.any():
                        distances = torch.cdist(
                            features_u_w[low_mask], training_prototypes, p=2.0
                        )
                        candidate_mask = (
                            distances <= prototype_max_radius.unsqueeze(0)
                        ) & prototype_mask.unsqueeze(0)
                        set_sizes = candidate_mask.sum(dim=1)
                        # 单元素集合缺少候选比较信息；诊断时补入第二近的有效原型。
                        if prototype_mask.sum() >= 2:
                            singleton = set_sizes == 1
                            if singleton.any():
                                ordered_classes = distances.masked_fill(
                                    ~prototype_mask.unsqueeze(0), float("inf")
                                ).argsort(dim=1)
                                candidate_mask[
                                    singleton, ordered_classes[singleton, 1]
                                ] = True
                                set_sizes = candidate_mask.sum(dim=1)
                        hits = candidate_mask.gather(
                            1,
                            targets_u_groundtruth[low_mask].unsqueeze(1),
                        ).squeeze(1)
                        set_size_hist += torch.bincount(
                            set_sizes, minlength=args.num_classes + 1
                        )
                        set_size_hits += torch.bincount(
                            set_sizes,
                            weights=hits.float(),
                            minlength=args.num_classes + 1,
                        )
                        low_total += int(low_mask.sum().item())
                        true_label_in_set += int(hits.sum().item())

                prototype_loss = (
                    labeled_prototype_loss
                    + args.lambda_proto_high * high_u_prototype_loss
                )

                loss = (
                    supervised_loss
                    + args.lambda_u * unsupervised_loss
                    + args.lambda_proto * prototype_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        prototypes, prototype_counts = self.compute_prototypes(
            args, eval_labeled_dataset
        )
        params = {
            name: value.detach().cpu().clone()
            for name, value in self.local_model.state_dict().items()
        }
        return (
            params,
            prototypes,
            prototype_counts,
            time.perf_counter() - start,
            {
                "low_total": low_total,
                "set_size_hist": set_size_hist.cpu().tolist(),
                "set_size_hits": set_size_hits.cpu().tolist(),
                "true_label_in_set": true_label_in_set,
            }
            if prototype_max_radius is not None
            else None,
        )

    @torch.no_grad()
    def compute_prototypes(self, args, dataset):
        self.local_model.eval()
        sums = torch.zeros(args.num_classes, self.local_model.dim, device=self.device)
        counts = torch.zeros(args.num_classes, device=self.device)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size_local_labeled_fixmatch,
            shuffle=False,
        )
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            features, _ = self.local_model(images)
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        return (
            (sums / counts.clamp_min(1).unsqueeze(1)).cpu(),
            counts.cpu(),
        )

    @staticmethod
    def interleave(x, size):
        shape = list(x.shape)
        return (
            x.reshape([-1, size] + shape[1:]).transpose(0, 1).reshape([-1] + shape[1:])
        )

    @staticmethod
    def de_interleave(x, size):
        shape = list(x.shape)
        return (
            x.reshape([size, -1] + shape[1:]).transpose(0, 1).reshape([-1] + shape[1:])
        )


class ClientTrainer:
    def __init__(self, args, device):
        self.local = Local(args, device=device)

    def train(self, task, labeled_view, unlabeled_view):
        transform = evaluation_transform(task.args.dataset)
        eval_labeled = IndexedEvalDataset(
            labeled_view.dataset, task.labeled_indices, transform
        )
        params, prototypes, counts, elapsed, candidate_stats = self.local.train(
            task.args,
            labeled_view,
            unlabeled_view,
            eval_labeled,
            task.global_params,
            task.global_prototypes,
            task.global_prototype_mask,
            task.global_prototype_max_radius,
        )
        return {
            "params": params,
            "prototypes": prototypes,
            "prototype_counts": counts,
            "elapsed_seconds": elapsed,
            "candidate_stats": candidate_stats,
        }


def load_datasets(args):
    if args.dataset == "CIFAR10":
        args.num_classes, args.num_labeled = 10, 500
        train = datasets.CIFAR10(args.path, train=True, download=True, transform=None)
        test = datasets.CIFAR10(
            args.path, train=False, transform=evaluation_transform(args.dataset)
        )
    elif args.dataset == "CIFAR100":
        args.num_classes, args.num_labeled = 100, 50
        train = datasets.CIFAR100(args.path, train=True, download=True, transform=None)
        test = datasets.CIFAR100(
            args.path, train=False, transform=evaluation_transform(args.dataset)
        )
    elif args.dataset == "SVHN":
        args.num_classes, args.num_labeled = 10, 460
        train = datasets.SVHN(args.path, split="train", download=True, transform=None)
        test = datasets.SVHN(
            args.path,
            split="test",
            download=True,
            transform=evaluation_transform(args.dataset),
        )
    elif args.dataset == "CINIC10":
        args.num_classes, args.num_labeled = 10, 900
        train = CINIC10(root=args.path, split="train", transform=None)
        test = CINIC10(
            root=args.path,
            split="test",
            transform=evaluation_transform(args.dataset),
        )
    else:
        raise ValueError(f"不支持的数据集：{args.dataset}")
    return train, test


def rename_dataset_metrics(metrics, prefix):
    """把通用评估结果改成指定数据域的列名。"""
    renamed = {
        f"{prefix}_global_model_acc": metrics["global_test_acc"],
        f"{prefix}_raw_prototype_acc": metrics["global_test_raw_prototype_acc"],
        f"{prefix}_raw_prototype_correct_given_model_wrong": metrics[
            "global_test_raw_prototype_correct_given_model_wrong"
        ],
    }
    diagnostic_prefix = "global_test_"
    for name, value in metrics.items():
        if name.startswith(diagnostic_prefix):
            short_name = name[len(diagnostic_prefix) :]
            if short_name not in {
                "acc",
                "raw_prototype_acc",
                "raw_prototype_correct_given_model_wrong",
            }:
                renamed[f"{prefix}_{short_name}"] = value
    return renamed


def prototype_norm_metrics(prototypes, valid_mask):
    """返回每个类别原始聚合原型的 L2 范数。"""
    metrics = {}
    for class_id, valid in enumerate(valid_mask.tolist()):
        metrics[f"raw_prototype_l2_norm_class_{class_id}"] = (
            prototypes[class_id].norm().item() if valid else np.nan
        )
    return metrics


def prototype_geometry_log(prototypes, valid_mask):
    """格式化原型间距离矩阵及其几何统计，供训练日志输出。"""
    valid_indices = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if len(valid_indices) < 1:
        return "无有效原型"

    centers = prototypes[valid_indices]
    distances = torch.cdist(centers, centers, p=2).cpu().numpy()
    radii = centers.norm(dim=1).cpu().numpy()
    lines = [
        f"有效类别：{valid_indices.tolist()}",
        "原型 L2 范数：[" + ", ".join(f"{value:.4f}" for value in radii) + "]",
        "原型 L2 距离矩阵：\n"
        + "\n".join(
            "  [" + ", ".join(f"{value:.4f}" for value in row) + "]"
            for row in distances
        ),
    ]
    if len(valid_indices) > 1:
        off_diagonal = distances[~np.eye(len(distances), dtype=bool)]
        nearest = distances.copy()
        np.fill_diagonal(nearest, np.inf)
        lines.append(
            "类间距离摘要：最小=%.4f，平均=%.4f，最大=%.4f，"
            "最近邻平均=%.4f"
            % (
                off_diagonal.min(),
                off_diagonal.mean(),
                off_diagonal.max(),
                nearest.min(axis=1).mean(),
            )
        )
    return "\n".join(lines)


@torch.no_grad()
def prototype_class_radius_metrics(
    model,
    prototypes,
    prototype_mask,
    labeled_dataset,
    unlabeled_dataset,
    batch_size,
    threshold,
    temperature,
):
    """计算三类样本相对聚合原型的类内平均半径。"""
    device = next(model.parameters()).device
    prototypes = prototypes.to(device)
    valid = prototype_mask.to(device).bool()
    distances_by_group = [[[] for _ in range(prototypes.size(0))] for _ in range(3)]

    def collect(dataset, mode):
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(device), labels.to(device)
            features, logits = model(images)
            if mode == "labeled":
                sample_labels = labels
                correct = logits.argmax(dim=1) == labels
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                confidence, sample_labels = probs.max(dim=1)
                keep = confidence.ge(threshold)
                if not keep.any():
                    continue
                features, sample_labels = features[keep], sample_labels[keep]
                correct = None

            keep_valid = valid[sample_labels]
            if not keep_valid.any():
                continue
            features = features[keep_valid]
            sample_labels = sample_labels[keep_valid]
            distances = (features - prototypes[sample_labels]).norm(dim=1)
            group = 0 if mode == "labeled" else 1
            for class_id in sample_labels.unique().tolist():
                distances_by_group[group][class_id].extend(
                    distances[sample_labels == class_id].cpu().tolist()
                )
            if mode == "labeled":
                correct = correct[keep_valid]
                correct_labels = sample_labels[correct]
                correct_distances = distances[correct]
                for class_id in correct_labels.unique().tolist():
                    distances_by_group[2][class_id].extend(
                        correct_distances[correct_labels == class_id].cpu().tolist()
                    )

    collect(labeled_dataset, "labeled")
    collect(unlabeled_dataset, "unlabeled")
    # 第二组包含第一组有标签样本和高置信无标签样本。
    for class_id in range(prototypes.size(0)):
        distances_by_group[1][class_id] = (
            distances_by_group[0][class_id] + distances_by_group[1][class_id]
        )

    stats = []
    for group in distances_by_group:
        group_stats = []
        for values in group:
            if not values:
                group_stats.append(
                    {
                        key: np.nan
                        for key in (
                            "mean",
                            "median",
                            "max",
                            "max90",
                            "max30",
                            "max50",
                            "max70",
                        )
                    }
                )
                continue
            values = np.sort(np.asarray(values))

            def prefix_max(fraction):
                count = max(1, int(np.ceil(fraction * len(values))))
                return values[count - 1]

            group_stats.append(
                {
                    "mean": values.mean(),
                    "median": np.median(values),
                    "max": values[-1],
                    "max90": prefix_max(0.9),
                    "max30": prefix_max(0.3),
                    "max50": prefix_max(0.5),
                    "max70": prefix_max(0.7),
                }
            )
        stats.append(group_stats)
    return stats


def fedavg_fixmatch(alpha, args=None):
    if args is None:
        args = args_parser()
    args.method = "test"
    train_dataset, test_dataset = load_datasets(args)
    run = create_run(args)
    setup_logging(run.log_file, level=args.log_level)
    logger.info("运行 ID：%s，结果目录：%s", run.run_id, run.dir)
    log_args(args)

    random_state = np.random.RandomState(args.seed)
    label_indices = classify_label(train_dataset, args.num_classes)
    labeled_indices, unlabeled_indices = partition_train(
        label_indices, args.num_labeled
    )
    partition = clients_indices_homo if alpha == 0 else clients_indices
    common = {"num_classes": args.num_classes, "num_clients": args.num_clients}
    if alpha == 0:
        client_labeled = partition(list_label2indices=labeled_indices, **common)
        client_unlabeled = partition(list_label2indices=unlabeled_indices, **common)
    else:
        client_labeled = partition(
            list_label2indices=labeled_indices,
            non_iid_alpha=alpha,
            seed=args.seed,
            **common,
        )
        client_unlabeled = partition(
            list_label2indices=unlabeled_indices,
            non_iid_alpha=alpha,
            seed=args.seed,
            **common,
        )
    show_clients_data_distribution(
        train_dataset,
        client_labeled,
        client_unlabeled,
        args.num_classes,
    )
    client_unlabeled_eval = copy.deepcopy(client_unlabeled)
    for client_id in range(args.num_clients):
        client_unlabeled[client_id] = np.concatenate(
            (client_unlabeled[client_id], client_labeled[client_id])
        )

    worker_gpus = parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    server = Global(args)
    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(train_dataset)
    worker_pool = ClientWorkerPool(
        worker_gpus,
        args,
        shared_dataset,
        trainer_cls=ClientTrainer,
        log_file=str(run.log_file),
    )

    metrics = []
    global_prototypes = None
    global_prototype_mask = None
    global_prototype_max_radius = None
    all_clients = list(range(args.num_clients))
    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
    for round_id in progress:
        params = server.download_params()
        online_clients = random_state.choice(
            all_clients, args.num_online_clients, replace=False
        )
        tasks = [
            ClientTask(
                round=round_id,
                client_id=int(client),
                labeled_indices=list(np.asarray(client_labeled[client]).tolist()),
                unlabeled_indices=list(np.asarray(client_unlabeled[client]).tolist()),
                global_params=params,
                global_prototypes=global_prototypes,
                global_prototype_mask=global_prototype_mask,
                global_prototype_max_radius=global_prototype_max_radius,
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        sample_counts = [result.num_samples for result in results]
        global_prototypes, global_prototype_mask = server.aggregate_prototypes(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            global_prototypes,
            global_prototype_mask,
        )
        norm_metrics = prototype_norm_metrics(global_prototypes, global_prototype_mask)
        aggregated_params = server.aggregate(
            [result.params for result in results],
            sample_counts,
        )

        selected_labeled_indices = np.concatenate(
            [np.asarray(client_labeled[client]) for client in online_clients]
        )
        selected_u_pool_indices = np.concatenate(
            [np.asarray(client_unlabeled_eval[client]) for client in online_clients]
        )
        transform = evaluation_transform(args.dataset)
        labeled_eval_dataset = IndexedEvalDataset(
            train_dataset, selected_labeled_indices, transform
        )
        u_pool_eval_dataset = IndexedEvalDataset(
            train_dataset, selected_u_pool_indices, transform
        )
        labeled_metrics = server.evaluate(
            aggregated_params,
            labeled_eval_dataset,
            args.batch_size_test,
            global_prototypes,
            global_prototype_mask,
        )
        u_pool_metrics = server.evaluate(
            aggregated_params,
            u_pool_eval_dataset,
            args.batch_size_test,
            global_prototypes,
            global_prototype_mask,
        )
        client_metrics = {
            **rename_dataset_metrics(labeled_metrics, "labeled"),
            **rename_dataset_metrics(u_pool_metrics, "u_pool"),
        }
        test_metrics = server.evaluate(
            aggregated_params,
            test_dataset,
            args.batch_size_test,
            global_prototypes,
            global_prototype_mask,
        )
        radius_metrics = None
        if global_prototypes is not None:
            radius_metrics = prototype_class_radius_metrics(
                server.model,
                global_prototypes,
                global_prototype_mask,
                labeled_eval_dataset,
                u_pool_eval_dataset,
                args.batch_size_test,
                args.threshold,
                args.T,
            )
            global_prototype_max_radius = [
                class_stats["max"] for class_stats in radius_metrics[1]
            ]
        row = {
            "round": round_id,
            **test_metrics,
            **client_metrics,
            **norm_metrics,
        }
        metrics.append(row)
        pd.DataFrame(metrics).set_index("round").to_csv(
            run.dir / "metrics.csv", encoding="utf8"
        )

        def display(value):
            return "—" if np.isnan(value) else f"{value:.2%}"

        logger.info(
            "第 %d 轮准确率：\n"
            "  global_test  模型/平均原型：%s / %s\n"
            "  labeled     模型/平均原型：%s / %s\n"
            "  u_pool      模型/平均原型：%s / %s",
            round_id,
            display(test_metrics["global_test_acc"]),
            display(test_metrics["global_test_raw_prototype_acc"]),
            display(client_metrics["labeled_global_model_acc"]),
            display(client_metrics["labeled_raw_prototype_acc"]),
            display(client_metrics["u_pool_global_model_acc"]),
            display(client_metrics["u_pool_raw_prototype_acc"]),
        )
        logger.info(
            "第 %d 轮低置信原型互补诊断（u_pool）：低置信=%s，模型错误=%s，"
            "原型救回=%s，救回率=%s；模型 Top1/2/3=%s/%s/%s；"
            "原型 Top1/2/3=%s/%s/%s；并集 Top2/3=%s/%s；"
            "互补 MM/MP/PM/PP=%s/%s/%s/%s",
            round_id,
            client_metrics["u_pool_low_conf_total"],
            client_metrics["u_pool_low_conf_model_wrong"],
            client_metrics["u_pool_proto_rescued_low_conf"],
            display(client_metrics["u_pool_proto_rescue_low_conf"]),
            display(client_metrics["u_pool_model_top1_low_conf"]),
            display(client_metrics["u_pool_model_top2_low_conf"]),
            display(client_metrics["u_pool_model_top3_low_conf"]),
            display(client_metrics["u_pool_proto_top1_low_conf"]),
            display(client_metrics["u_pool_proto_top2_low_conf"]),
            display(client_metrics["u_pool_proto_top3_low_conf"]),
            display(client_metrics["u_pool_union_top2_low_conf"]),
            display(client_metrics["u_pool_union_top3_low_conf"]),
            client_metrics["u_pool_low_conf_MM"],
            client_metrics["u_pool_low_conf_MP"],
            client_metrics["u_pool_low_conf_PM"],
            client_metrics["u_pool_low_conf_PP"],
        )
        logger.info(
            "第 %d 轮各类平均原型 L2 范数：\n  %s",
            round_id,
            np.round(
                [
                    norm_metrics[f"raw_prototype_l2_norm_class_{class_id}"]
                    for class_id in range(args.num_classes)
                ],
                4,
            ).tolist(),
        )
        logger.info(
            "第 %d 轮原型几何诊断：\n%s",
            round_id,
            prototype_geometry_log(global_prototypes, global_prototype_mask),
        )
        if radius_metrics is not None:
            stat_names = (
                ("有标签", 0),
                ("有标签+高置信无标签", 1),
                ("预测正确有标签", 2),
            )
            metric_names = (
                ("mean", "平均"),
                ("median", "中位数"),
                ("max", "最大值"),
                ("max90", "前90%样本最大距离"),
                ("max70", "前70%样本最大距离"),
                ("max50", "前50%样本最大距离"),
                ("max30", "前30%样本最大距离"),
            )
            radius_lines = []
            for group_name, group_index in stat_names:
                values = radius_metrics[group_index]
                order = sorted(
                    range(len(values)),
                    key=lambda class_id: values[class_id]["max"],
                    reverse=True,
                )
                sorted_values = [values[class_id] for class_id in order]
                radius_lines.append(f"  {group_name}：")
                radius_lines.append(f"    类别顺序（按最大值降序）：{order}")
                for metric_name, metric_label in metric_names:
                    radius_lines.append(
                        f"    {metric_label}：["
                        + ", ".join(
                            f"{stats[metric_name]:.4f}" for stats in sorted_values
                        )
                        + "]"
                    )
            logger.info(
                "第 %d 轮类内原型半径：\n%s",
                round_id,
                "\n".join(radius_lines),
            )
        candidate_results = [
            result for result in results if result.candidate_stats is not None
        ]
        if candidate_results:
            client_lines = []
            ratios = []
            coverages = []
            per_size_coverages = []
            total_hist = np.zeros(args.num_classes + 1)
            total_size_hits = np.zeros(args.num_classes + 1)
            total_low = total_hits = 0
            for result in candidate_results:
                stats = result.candidate_stats
                low_total = stats["low_total"]
                if not low_total:
                    client_lines.append(f"客户端 {result.client_id}：低置信=0")
                    continue
                hist = np.asarray(stats["set_size_hist"])
                ratio = hist / low_total
                coverage = stats["true_label_in_set"] / low_total
                size_hits = np.asarray(stats["set_size_hits"])
                per_size_coverage = np.divide(
                    size_hits,
                    hist,
                    out=np.full(args.num_classes + 1, np.nan),
                    where=hist > 0,
                )
                client_lines.append(
                    f"客户端 {result.client_id}：低置信={low_total}，"
                    f"整体真实标签∈集合={coverage:.2%}\n"
                    f"  集合大小 0..{args.num_classes} 比例："
                    f"{np.round(ratio, 4).tolist()}\n"
                    f"  集合大小 0..{args.num_classes} 覆盖率："
                    f"{np.round(per_size_coverage, 4).tolist()}"
                )
                ratios.append(ratio)
                coverages.append(coverage)
                per_size_coverages.append(per_size_coverage)
                total_hist += hist
                total_size_hits += size_hits
                total_low += low_total
                total_hits += stats["true_label_in_set"]
            if ratios:
                size_coverage_array = np.asarray(per_size_coverages)
                size_coverage_count = np.isfinite(size_coverage_array).sum(axis=0)
                macro_size_coverage = np.divide(
                    np.nansum(size_coverage_array, axis=0),
                    size_coverage_count,
                    out=np.full(args.num_classes + 1, np.nan),
                    where=size_coverage_count > 0,
                )
                logger.info(
                    "第 %d 轮最大半径候选集合诊断（单元素集合补入第二近类别）：\n%s\n"
                    "客户端宏平均：\n"
                    "  集合大小 0..%d 比例：%s\n"
                    "  集合大小 0..%d 覆盖率：%s\n"
                    "  整体真实标签∈集合：%.2f%%\n"
                    "样本微平均（总低置信=%d）：\n"
                    "  集合大小 0..%d 比例：%s\n"
                    "  集合大小 0..%d 覆盖率：%s\n"
                    "  整体真实标签∈集合：%.2f%%",
                    round_id,
                    "\n".join(client_lines),
                    args.num_classes,
                    np.round(np.mean(ratios, axis=0), 4).tolist(),
                    args.num_classes,
                    np.round(macro_size_coverage, 4).tolist(),
                    np.mean(coverages) * 100,
                    total_low,
                    args.num_classes,
                    np.round(total_hist / total_low, 4).tolist(),
                    args.num_classes,
                    np.round(
                        np.divide(
                            total_size_hits,
                            total_hist,
                            out=np.full(args.num_classes + 1, np.nan),
                            where=total_hist > 0,
                        ),
                        4,
                    ).tolist(),
                    total_hits / total_low * 100,
                )
        progress.set_postfix(acc=f"{test_metrics['global_test_acc']:.2%}")
        if (
            round_id == 1
            or round_id == args.num_rounds
            or (round_id % 50 == 0 and round_id > 0.8 * args.num_rounds)
        ):
            torch.save(
                aggregated_params,
                run.checkpoint_dir / f"fedavg_params_round_{round_id}.pth",
            )

    worker_pool.close()
    run.finish(
        best_acc=max(row["global_test_acc"] for row in metrics) if metrics else None,
        num_rounds=args.num_rounds,
    )


if __name__ == "__main__":
    args = args_parser()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    run_main(lambda: fedavg_fixmatch(args.alpha, args))
