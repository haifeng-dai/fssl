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


def dist_contrastive_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """以类别锚点为候选的距离交叉熵。"""
    distances = torch.cdist(features, prototypes, p=2.0)
    return F.cross_entropy(-distances, labels)


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

    def learn_anchors(
        self,
        initial_prototypes,
        prototype_mask,
        client_prototypes,
        client_counts,
        objective,
    ):
        """将聚合原型转为可训练锚点，并按指定目标在服务端优化。"""
        valid_classes = prototype_mask.to(self.device, dtype=torch.bool).nonzero(
            as_tuple=True
        )[0]
        if valid_classes.numel() == 0:
            return initial_prototypes, {"contrastive_loss": np.nan, "classifier_loss": np.nan}

        anchor = torch.nn.Parameter(initial_prototypes.to(self.device).clone())
        optimizer = SGD([anchor], lr=self.args.anchor_lr)
        class_remap = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=self.device
        )
        class_remap[valid_classes] = torch.arange(
            valid_classes.numel(), device=self.device
        )
        uploaded_features, uploaded_labels = [], []
        for prototypes, counts in zip(client_prototypes, client_counts):
            present = counts.to(self.device, dtype=torch.bool) & prototype_mask.to(
                self.device, dtype=torch.bool
            )
            if present.any():
                class_ids = present.nonzero(as_tuple=True)[0]
                uploaded_features.append(prototypes.to(self.device)[class_ids])
                uploaded_labels.append(class_remap[class_ids])

        contrastive_loss = np.nan
        if objective in {"contrastive", "hybrid"} and uploaded_features:
            features = torch.cat(uploaded_features)
            labels = torch.cat(uploaded_labels)
            for _ in range(self.args.anchor_steps):
                loss = dist_contrastive_loss(features, anchor[valid_classes], labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            contrastive_loss = loss.item()

        classifier_loss = np.nan
        if objective in {"classifier", "hybrid"}:
            # FedAvg 分类器只提供固定的判别约束，梯度仅流向 anchor。
            classifier = self.model.classifier
            previous_requires_grad = [param.requires_grad for param in classifier.parameters()]
            try:
                classifier.eval()
                for param in classifier.parameters():
                    param.requires_grad_(False)
                labels = valid_classes
                for _ in range(self.args.anchor_steps):
                    loss = F.cross_entropy(classifier(anchor[valid_classes]), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                classifier_loss = loss.item()
            finally:
                for param, requires_grad in zip(classifier.parameters(), previous_requires_grad):
                    param.requires_grad_(requires_grad)

        return anchor.detach().cpu(), {
            "contrastive_loss": contrastive_loss,
            "classifier_loss": classifier_loss,
        }

    @torch.no_grad()
    def evaluate(
        self,
        params,
        dataset,
        batch_size,
        anchors=None,
        anchor_mask=None,
    ):
        """在测试集上评估分类头和服务端训练后的类别锚点。"""
        self.model.load_state_dict(params)
        self.model.eval()
        model_correct = 0
        prototype_correct = 0
        prototypes = (
            anchors.to(self.device) if anchors is not None else None
        )
        prototype_mask = (
            anchor_mask.to(self.device, dtype=torch.bool)
            if anchor_mask is not None
            else None
        )

        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(self.device), labels.to(self.device)
            features, logits = self.model(images)
            probs = torch.softmax(logits / self.args.T, dim=-1)
            confidence, prediction = probs.max(dim=1)
            model_correct += (prediction == labels).sum().item()

            if prototypes is not None and prototype_mask is not None and prototype_mask.any():
                prototype_distances = torch.cdist(features, prototypes).masked_fill(
                    ~prototype_mask.unsqueeze(0), float("inf")
                )
                prototype_pred = prototype_distances.argmin(dim=1)
                prototype_correct += (prototype_pred == labels).sum().item()

        total = len(dataset)
        result = {
            "global_test_acc": model_correct / total,
            "global_test_anchor_acc": (
                prototype_correct / total
                if prototypes is not None and prototype_mask is not None and prototype_mask.any()
                else np.nan
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

        training_prototypes = (
            global_prototypes.to(self.device)
            if global_prototypes is not None
            else None
        )
        prototype_valid_mask = (
            global_prototype_mask.to(self.device, dtype=torch.bool)
            if global_prototype_mask is not None
            else None
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

                if training_prototypes is not None and prototype_valid_mask is not None:
                    # 1. 有标签样本特征与对应真实类别的原型进行 MSE 对齐
                    valid_x = prototype_valid_mask[targets_x]
                    if valid_x.any():
                        proto_targets_x = training_prototypes[targets_x[valid_x]]
                        loss_proto_x = F.mse_loss(features_x[valid_x], proto_targets_x)

                    # 2. 高置信度无标签样本特征与伪标签对应的原型进行 MSE 对齐
                    high_mask = mask.bool() & prototype_valid_mask[pseudo_targets]
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
    return {
        f"{prefix}_global_model_acc": metrics["global_test_acc"],
        f"{prefix}_anchor_acc": metrics["global_test_anchor_acc"],
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


def fedavg_fixmatch(alpha, args=None, anchor_objective="contrastive"):
    if args is None:
        args = args_parser()
    args.method = f"test_anchor_{anchor_objective}"
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
    anchors = None
    anchor_mask = torch.zeros(args.num_classes, dtype=torch.bool)
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
                global_prototypes=anchors,
                global_prototype_mask=anchor_mask,
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

        # 2. 聚合原型后在服务端转化为可训练 anchor，并在下一轮下发。
        round_anchor_mask = torch.stack(
            [result.prototype_counts.to(torch.bool) for result in results]
        ).any(dim=0)
        mean_prototypes = server.aggregate_prototypes(
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            previous=anchors,
        )
        anchor_mask |= round_anchor_mask
        anchors, anchor_diagnostics = server.learn_anchors(
            mean_prototypes,
            anchor_mask,
            [result.prototypes for result in results],
            [result.prototype_counts for result in results],
            anchor_objective,
        )

        # 3. 评估
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
            anchors=anchors,
            anchor_mask=anchor_mask,
        )
        u_pool_metrics = server.evaluate(
            aggregated_params,
            u_pool_eval_dataset,
            args.batch_size_test,
            anchors=anchors,
            anchor_mask=anchor_mask,
        )
        client_metrics = {
            **rename_dataset_metrics(labeled_metrics, "labeled"),
            **rename_dataset_metrics(u_pool_metrics, "u_pool"),
        }
        test_metrics = server.evaluate(
            aggregated_params,
            test_dataset,
            args.batch_size_test,
            anchors=anchors,
            anchor_mask=anchor_mask,
        )
        norm_metrics = {
            **prototype_norm_metrics(anchors, prefix="anchor"),
        }
        row = {
            "round": round_id,
            **test_metrics,
            **client_metrics,
            **norm_metrics,
            **anchor_diagnostics,
        }
        metrics.append(row)
        pd.DataFrame(metrics).set_index("round").to_csv(
            run.dir / "metrics.csv", encoding="utf8"
        )

        def display(value):
            return "—" if np.isnan(value) else f"{value:.2%}"

        logger.info(
            "第 %d 轮准确率：\n"
            "  global_test  模型/锚点：%s / %s\n"
            "  labeled     模型/锚点：%s / %s\n"
            "  u_pool      模型/锚点：%s / %s",
            round_id,
            display(test_metrics["global_test_acc"]),
            display(test_metrics["global_test_anchor_acc"]),
            display(client_metrics["labeled_global_model_acc"]),
            display(client_metrics["labeled_anchor_acc"]),
            display(client_metrics["u_pool_global_model_acc"]),
            display(client_metrics["u_pool_anchor_acc"]),
        )
        logger.info(
            "第 %d 轮 anchor 训练：contrastive loss=%s，classifier loss=%s；各类锚点 L2 范数：%s",
            round_id,
            "—" if np.isnan(anchor_diagnostics["contrastive_loss"]) else f"{anchor_diagnostics['contrastive_loss']:.6f}",
            "—" if np.isnan(anchor_diagnostics["classifier_loss"]) else f"{anchor_diagnostics['classifier_loss']:.6f}",
            np.round(
                [
                    norm_metrics.get(f"anchor_prototype_l2_norm_class_{c}", np.nan)
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


def run_experiment(anchor_objective="contrastive"):
    args = args_parser()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    run_main(lambda: fedavg_fixmatch(args.alpha, args, anchor_objective))


if __name__ == "__main__":
    run_experiment()
