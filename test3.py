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

        def nearest(features, centers):
            distances = torch.cdist(features, centers).square() / features.size(1)
            return distances.masked_fill(~valid.unsqueeze(0), float("inf")).argmin(
                dim=1
            )

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
    """仅用监督 CE 和高置信 KL 训练；原型不参与反向传播。"""

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
        prototypes = global_prototypes.to(self.device) if has_prototypes else None
        prototype_mask = (
            global_prototype_mask.to(self.device).bool() if has_prototypes else None
        )
        local_steps = int(len(u_pool_dataset) / args.batch_size_local_labeled_fixmatch)

        for _ in range(args.local_epochs):
            labeled_iter, u_pool_iter = iter(labeled_loader), iter(u_pool_loader)
            for _ in range(local_steps):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)
                try:
                    inputs_u_w, inputs_u_s, _ = next(u_pool_iter)
                except StopIteration:
                    u_pool_iter = iter(u_pool_loader)
                    inputs_u_w, inputs_u_s, _ = next(u_pool_iter)

                inputs_x = inputs_x.to(self.device)
                targets_x = targets_x.to(self.device)
                inputs_u_w = inputs_u_w.to(self.device)
                inputs_u_s = inputs_u_s.to(self.device)
                batch_size = inputs_x.size(0)
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                    2 * args.mu + 1,
                )
                features, logits = self.local_model(inputs)
                features = self.de_interleave(features, 2 * args.mu + 1)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                features_x = features[:batch_size]
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

                student_probs = torch.softmax(logits_u_s, dim=-1).clamp_min(1e-10)
                hard_targets = F.one_hot(
                    pseudo_targets, num_classes=args.num_classes
                ).float()
                unsupervised_loss = (
                    F.kl_div(student_probs.log(), hard_targets, reduction="none").sum(
                        dim=-1
                    )
                    * mask
                ).mean()

                prototype_loss = torch.zeros((), device=self.device)
                if has_prototypes:
                    valid_labeled = prototype_mask[targets_x]
                    if valid_labeled.any():
                        prototype_loss = F.mse_loss(
                            features_x[valid_labeled],
                            prototypes[targets_x[valid_labeled]],
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
        params, prototypes, counts, elapsed = self.local.train(
            task.args,
            labeled_view,
            unlabeled_view,
            eval_labeled,
            task.global_params,
            task.global_prototypes,
            task.global_prototype_mask,
        )
        return {
            "params": params,
            "prototypes": prototypes,
            "prototype_counts": counts,
            "elapsed_seconds": elapsed,
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


def fedavg_fixmatch(alpha, args=None):
    if args is None:
        args = args_parser()
    args.method = "test3"
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
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)

        global_prototypes, global_prototype_mask = server.aggregate_prototypes(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            global_prototypes,
            global_prototype_mask,
        )
        norm_metrics = prototype_norm_metrics(global_prototypes, global_prototype_mask)
        aggregated_params = server.aggregate(
            [result.params for result in results],
            [result.num_samples for result in results],
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
            "  global_test  模型/原型：%s / %s\n"
            "  labeled     模型/原型：%s / %s\n"
            "  u_pool      模型/原型：%s / %s",
            round_id,
            display(test_metrics["global_test_acc"]),
            display(test_metrics["global_test_raw_prototype_acc"]),
            display(client_metrics["labeled_global_model_acc"]),
            display(client_metrics["labeled_raw_prototype_acc"]),
            display(client_metrics["u_pool_global_model_acc"]),
            display(client_metrics["u_pool_raw_prototype_acc"]),
        )
        logger.info(
            "第 %d 轮模型错误但原型正确的比例：\n"
            "  global_test：%s\n"
            "  labeled：%s\n"
            "  u_pool：%s",
            round_id,
            display(
                test_metrics["global_test_raw_prototype_correct_given_model_wrong"]
            ),
            display(client_metrics["labeled_raw_prototype_correct_given_model_wrong"]),
            display(client_metrics["u_pool_raw_prototype_correct_given_model_wrong"]),
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
            "第 %d 轮各类原型 L2 范数：\n  %s",
            round_id,
            np.round(
                [
                    norm_metrics[f"raw_prototype_l2_norm_class_{class_id}"]
                    for class_id in range(args.num_classes)
                ],
                4,
            ).tolist(),
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
