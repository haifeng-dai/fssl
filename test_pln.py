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
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
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

# PLN 与服务端特定超参数
SERVER_EPOCHS = 10
PLN_WIDTH = 512
PLN_DEPTH = 1

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
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """服务端基于距离的原型对比损失，使各个类别的锚点相互分离。"""
    device = features.device
    num_classes = prototypes.shape[0]
    dist = torch.cdist(features, prototypes.to(device), p=2.0)
    if margin > 0:
        one_hot = F.one_hot(labels, num_classes).to(device)
        dist = dist + one_hot * margin
    return F.cross_entropy(-dist, labels)


class PLN(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 512,
        feature_dim: int = 512,
        depth: int = 1,
        fixed: int = 0,
        init_emb: int = 0,
    ):
        super().__init__()
        self.embedings = torch.nn.Embedding(num_classes, width)
        self._init_embedings(init_emb)
        if fixed:
            self.embedings.weight.requires_grad = False
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.middle = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU())
                for _ in range(depth)
            ]
        )
        self.fc = torch.nn.Linear(width, feature_dim)

    def _init_embedings(self, init_emb: int):
        initializers = {
            1: lambda: torch.nn.init.uniform_(self.embedings.weight, -0.1, 0.1),
            2: lambda: torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.1),
            3: lambda: torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.01),
            4: lambda: torch.nn.init.xavier_uniform_(self.embedings.weight),
            5: lambda: torch.nn.init.xavier_normal_(self.embedings.weight),
            6: lambda: torch.nn.init.kaiming_uniform_(
                self.embedings.weight, nonlinearity="linear"
            ),
            7: lambda: torch.nn.init.orthogonal_(self.embedings.weight),
        }
        if init_emb not in initializers and init_emb != 0:
            raise ValueError("Unknown init_emb value")
        if init_emb in initializers:
            initializers[init_emb]()

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.middle(self.embedings(class_ids)))


class Global:
    def __init__(
        self,
        args,
        width_pln: int = PLN_WIDTH,
        depth_pln: int = PLN_DEPTH,
        server_epochs: int = SERVER_EPOCHS,
    ):
        self.args = args
        gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = build_model(args).to(self.device)
        self.num_classes = args.num_classes
        self.all_classes = torch.arange(args.num_classes, device=self.device)
        self.server_epochs = server_epochs
        self.pln = PLN(
            num_classes=args.num_classes,
            width=width_pln,
            feature_dim=self.model.dim,
            depth=depth_pln,
            fixed=0,
            init_emb=0,
        ).to(self.device)
        self.pln_optimizer = SGD(
            self.pln.parameters(),
            lr=getattr(args, "lr_server", 0.01),
            momentum=0.9,
            weight_decay=1e-4,
        )

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

    def aggregate_prototypes(self, prototypes, counts, previous=None):
        """按类别样本数加权聚合客户端原始特征原型。"""
        total = torch.zeros(self.num_classes, device=self.device)
        weighted = torch.zeros_like(prototypes[0], device=self.device)
        for local_prototypes, local_counts in zip(prototypes, counts):
            local_prototypes = local_prototypes.to(self.device)
            local_counts = local_counts.to(self.device)
            weighted += local_prototypes * local_counts.unsqueeze(1)
            total += local_counts
        result = weighted / total.clamp_min(1).unsqueeze(1)
        if previous is not None:
            carry = total == 0
            if carry.any():
                result[carry] = previous.to(self.device)[carry]
        return result.cpu()

    def train_pln(self, uploaded_features: torch.Tensor, uploaded_labels: torch.Tensor):
        """利用各客户端上传的原型及对应标签作为数据集，训练 PLN 网络分离类别锚点。"""
        if uploaded_features.numel() == 0:
            return 0.0, 0
        self.pln.train()
        dataset = TensorDataset(uploaded_features, uploaded_labels)
        batch_size = min(len(dataset), max(1, getattr(self.args, "bs_server", 10)))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        total_loss = 0.0
        batches = 0
        for _ in range(self.server_epochs):
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                current_prototypes = self.pln(self.all_classes)
                loss = dist_contrastive_loss(
                    features,
                    current_prototypes,
                    labels,
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("PLN 训练中 loss 为 NaN/Inf，跳过此 step")
                    continue
                self.pln_optimizer.zero_grad()
                loss.backward()
                self.pln_optimizer.step()
                total_loss += loss.item()
                batches += 1
        avg_loss = total_loss / max(1, batches)
        logger.info(
            "服务端 PLN 训练完成：epoch=%d, 批次数=%d, 平均 loss=%.4f",
            self.server_epochs,
            batches,
            avg_loss,
        )
        return total_loss, batches

    @torch.no_grad()
    def get_pln_prototypes(self):
        """生成 PLN 类别原型，并记录是否发生类别塌缩或 ReLU 死亡。"""
        self.pln.eval()
        embeddings = self.pln.embedings(self.all_classes)
        middle_output = self.pln.middle(embeddings)
        prototypes = self.pln.fc(middle_output)

        # `middle` 的末层为 ReLU；非零比例为 0 时，所有类别都会只剩 fc.bias。
        active_fraction = (middle_output != 0).float().mean().item()
        pairwise_distances = torch.pdist(prototypes)
        classwise_std = prototypes.std(dim=0, unbiased=False).mean().item()
        logger.info(
            "PLN 健康度：middle 非零比例=%.2f%%，类别原型逐维 std=%.6f，"
            "两两距离 min/mean/max=%.6f/%.6f/%.6f",
            active_fraction * 100,
            classwise_std,
            pairwise_distances.min().item(),
            pairwise_distances.mean().item(),
            pairwise_distances.max().item(),
        )
        return prototypes.detach().cpu()

    @torch.no_grad()
    def evaluate(
        self,
        params,
        dataset,
        batch_size,
        raw_prototypes=None,
        pln_prototypes=None,
    ):
        """在测试集上评估分类头、原始聚合原型与 PLN 原型。"""
        self.model.load_state_dict(params)
        self.model.eval()
        model_correct = 0
        raw_correct = 0
        pln_correct = 0

        raw_protos = (
            raw_prototypes.to(self.device) if raw_prototypes is not None else None
        )
        pln_protos = (
            pln_prototypes.to(self.device) if pln_prototypes is not None else None
        )

        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(self.device), labels.to(self.device)
            features, logits = self.model(images)
            probs = torch.softmax(logits / self.args.T, dim=-1)
            confidence, prediction = probs.max(dim=1)
            model_correct += (prediction == labels).sum().item()

            if raw_protos is not None:
                dist_raw = torch.cdist(features, raw_protos)
                raw_pred = dist_raw.argmin(dim=1)
                raw_correct += (raw_pred == labels).sum().item()

            if pln_protos is not None:
                dist_pln = torch.cdist(features, pln_protos)
                pln_pred = dist_pln.argmin(dim=1)
                pln_correct += (pln_pred == labels).sum().item()

        total = len(dataset)
        result = {
            "global_test_acc": model_correct / total,
            "global_test_raw_prototype_acc": (
                raw_correct / total if raw_protos is not None else np.nan
            ),
            "global_test_pln_prototype_acc": (
                pln_correct / total if pln_protos is not None else np.nan
            ),
        }
        return result

    def download_params(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }


class Local:
    """使用监督 CE、高置信伪标签 CE 和全局原型 MSE 对齐训练。"""

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

        has_prototypes = global_prototypes is not None
        training_prototypes = (
            global_prototypes.to(self.device) if has_prototypes else None
        )

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

                loss_proto_x = torch.zeros((), device=self.device)
                loss_proto_u = torch.zeros((), device=self.device)

                if has_prototypes:
                    # 1. 有标签样本特征与对应真实类别的原型进行 MSE 对齐
                    proto_targets_x = training_prototypes[targets_x]
                    loss_proto_x = F.mse_loss(features_x, proto_targets_x)

                    # 2. 高置信度无标签样本特征与伪标签对应的原型进行 MSE 对齐
                    high_mask = mask.bool()
                    if high_mask.any():
                        proto_targets_u = training_prototypes[pseudo_targets[high_mask]]
                        loss_proto_u = F.mse_loss(
                            features_u_s[high_mask], proto_targets_u
                        )

                proto_loss = (
                    loss_proto_x
                    + getattr(args, "lambda_proto_high", 1.0) * loss_proto_u
                )
                loss = (
                    supervised_loss
                    + args.lambda_u * unsupervised_loss
                    + args.lambda_proto * proto_loss
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
    return {
        f"{prefix}_global_model_acc": metrics["global_test_acc"],
        f"{prefix}_raw_prototype_acc": metrics["global_test_raw_prototype_acc"],
        f"{prefix}_pln_prototype_acc": metrics["global_test_pln_prototype_acc"],
    }


def prototype_norm_metrics(prototypes, prefix="raw"):
    """返回每个类别原型的 L2 范数。"""
    metrics = {}
    if prototypes is None:
        return metrics
    for class_id in range(prototypes.size(0)):
        metrics[f"{prefix}_prototype_l2_norm_class_{class_id}"] = (
            prototypes[class_id].norm().item()
        )
    return metrics


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
    common = {"num_classes": args.num_classes, "num_clients": args.num_clients}
    if alpha == 0:
        client_labeled = clients_indices_homo(
            list_label2indices=labeled_indices, **common
        )
        client_unlabeled = clients_indices_homo(
            list_label2indices=unlabeled_indices, **common
        )
    else:
        client_labeled = clients_indices(
            list_label2indices=labeled_indices,
            non_iid_alpha=alpha,
            seed=args.seed,
            **common,
        )
        client_unlabeled = clients_indices(
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
    server = Global(
        args,
        width_pln=PLN_WIDTH,
        depth_pln=PLN_DEPTH,
        server_epochs=SERVER_EPOCHS,
    )
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
    raw_prototypes = None
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
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        sample_counts = [result.num_samples for result in results]

        # 1. 聚合全局模型权重
        aggregated_params = server.aggregate(
            [result.params for result in results],
            sample_counts,
        )

        # 2. 聚合客户端原始原型 (供记录与对比)
        raw_prototypes = server.aggregate_prototypes(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            previous=raw_prototypes,
        )

        # 3. 收集所有客户端上传的带标签原型，在服务端训练 PLN 使得各个类别锚点相互分离
        uploaded_features = []
        uploaded_labels = []
        uploaded_client_counts = torch.zeros(args.num_classes, dtype=torch.long)
        uploaded_sample_counts = torch.zeros(args.num_classes, dtype=torch.long)
        for result in results:
            client_protos = result.prototypes
            client_counts = result.prototype_counts
            for c in range(args.num_classes):
                if client_counts[c] > 0:
                    uploaded_features.append(client_protos[c])
                    uploaded_labels.append(c)
                    uploaded_client_counts[c] += 1
                    uploaded_sample_counts[c] += int(client_counts[c])

        missing_classes = torch.where(uploaded_client_counts == 0)[0].tolist()
        logger.info(
            "第 %d 轮 PLN 原型输入：每类客户端原型数=%s，"
            "每类有标签样本数=%s，缺失类=%s",
            round_id,
            uploaded_client_counts.tolist(),
            uploaded_sample_counts.tolist(),
            missing_classes if missing_classes else "无",
        )

        if uploaded_features:
            uploaded_features = torch.stack(uploaded_features)
            uploaded_labels = torch.tensor(uploaded_labels, dtype=torch.long)
            server.train_pln(uploaded_features, uploaded_labels)

        # 4. 获取最新的全局 PLN 原型，供评估并在下一轮下发给各客户端
        global_prototypes = server.get_pln_prototypes()

        # 5. 评估
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
            raw_prototypes=raw_prototypes,
            pln_prototypes=global_prototypes,
        )
        u_pool_metrics = server.evaluate(
            aggregated_params,
            u_pool_eval_dataset,
            args.batch_size_test,
            raw_prototypes=raw_prototypes,
            pln_prototypes=global_prototypes,
        )
        client_metrics = {
            **rename_dataset_metrics(labeled_metrics, "labeled"),
            **rename_dataset_metrics(u_pool_metrics, "u_pool"),
        }
        test_metrics = server.evaluate(
            aggregated_params,
            test_dataset,
            args.batch_size_test,
            raw_prototypes=raw_prototypes,
            pln_prototypes=global_prototypes,
        )
        norm_metrics = {
            **prototype_norm_metrics(raw_prototypes, prefix="raw"),
            **prototype_norm_metrics(global_prototypes, prefix="pln"),
        }
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
            "  global_test  模型/均值原型/PLN：%s / %s / %s\n"
            "  labeled     模型/均值原型/PLN：%s / %s / %s\n"
            "  u_pool      模型/均值原型/PLN：%s / %s / %s",
            round_id,
            display(test_metrics["global_test_acc"]),
            display(test_metrics["global_test_raw_prototype_acc"]),
            display(test_metrics["global_test_pln_prototype_acc"]),
            display(client_metrics["labeled_global_model_acc"]),
            display(client_metrics["labeled_raw_prototype_acc"]),
            display(client_metrics["labeled_pln_prototype_acc"]),
            display(client_metrics["u_pool_global_model_acc"]),
            display(client_metrics["u_pool_raw_prototype_acc"]),
            display(client_metrics["u_pool_pln_prototype_acc"]),
        )
        logger.info(
            "第 %d 轮各类原型 L2 范数：\n  均值：%s\n  PLN：%s",
            round_id,
            np.round(
                [
                    norm_metrics.get(f"raw_prototype_l2_norm_class_{c}", np.nan)
                    for c in range(args.num_classes)
                ],
                4,
            ).tolist(),
            np.round(
                [
                    norm_metrics.get(f"pln_prototype_l2_norm_class_{c}", np.nan)
                    for c in range(args.num_classes)
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
