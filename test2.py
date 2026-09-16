import copy
import logging
import random
import sys
import time

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


def l2_prototype_contrastive_loss(features, labels, anchors):
    """L2 距离版 InfoNCE：标签类 anchor 为正对，其余 anchor 均为负对。"""
    negative_squared_distances = -torch.cdist(features, anchors, p=2).square()
    logits = negative_squared_distances
    return F.cross_entropy(logits, labels)


def prototype_probabilities(features, anchors):
    """根据样本到全局 anchor 的负平方 L2 距离计算类别概率。"""
    logits = -torch.cdist(features, anchors, p=2).square()
    return torch.softmax(logits, dim=-1)


def prototype_set_contrastive_loss(
    features, positive_anchors, label_sets, unlabeled_start
):
    """标签集合对比：正对为 anchor/anchor 加权和，负对仅来自集合不相交样本。"""
    positive_logits = -(features - positive_anchors).square().sum(dim=1)
    pairwise_logits = -torch.cdist(features, features, p=2).square()

    overlap = (label_sets.unsqueeze(1) & label_sets.unsqueeze(0)).any(dim=2)
    negative_mask = ~overlap
    negative_mask.fill_diagonal_(False)
    negative_logits = pairwise_logits.masked_fill(~negative_mask, float("-inf"))

    logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
    targets = torch.zeros(
        logits.size(0) - unlabeled_start, dtype=torch.long, device=features.device
    )
    return F.cross_entropy(logits[unlabeled_start:], targets)


class Global:
    """服务端状态：维护全局模型并执行标准的 FedAvg 参数加权聚合与评估。"""

    def __init__(self, args):
        self.args = args
        self.gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = build_model(args)
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
        """以聚合原型初始化 anchor，并以客户端类别原型执行 L2 InfoNCE。"""
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
            loss = l2_prototype_contrastive_loss(
                local_prototypes,
                local_labels,
                anchor,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return anchor.detach().cpu().clone()


class Local:
    """客户端训练器：本地/全局置信度联合筛选，使用全局模型伪标签。"""

    def __init__(self, args, device=None):
        self.device = device or torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        self.local_model = build_model(args)
        self.local_model.to(self.device)

        # 与局部模型分离为两份实例，避免本地更新改变本轮全局模型的预测。
        self.global_model = build_model(args)
        self.global_model.to(self.device)

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
        global_proto_conf=None,
    ):
        """本地/全局置信度并集筛选，固定全局模型提供无标签目标。"""
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
        self.global_model.load_state_dict(global_params)
        self.global_model.eval()
        # 彻底清空 SGD 动量，保证各客户端状态隔离
        self.optimizer.state.clear()
        self.local_model.train()

        # 初始化伪标签指标统计变量
        num_pseudo_corrects = 0
        num_pseudo_total = 0
        num_u_valid = 0
        pseudo_client_acc = 0.0
        u_client_valid = 0.0
        # 低置信样本统计：标签集大小直方图与真实标签命中率
        # 直方图按大小直接索引（0 ~ num_classes），大小 0 表示无候选类；
        # hit_hist 记录各大小桶内真实标签∈标签集的命中数
        num_low_total = 0
        num_gt_in_set_total = 0
        set_size_hist = [0] * (args.num_classes + 1)
        set_size_hit_hist = [0] * (args.num_classes + 1)
        num_high_corrects = 0
        # 原型置信度阈值统计：逐类累积 (有标签 + 高置信样本) 的原型概率。
        conf_sum = torch.zeros(args.num_classes, device=self.device)
        conf_cnt = torch.zeros(args.num_classes, device=self.device)
        had_anchors = global_anchors is not None
        # 与 ProxyFL 对齐：按有标签 batch 大小确定本地更新次数。
        # 无标签 batch 为 B * mu，DataLoader 耗尽时会在下方循环重建。
        local_iter = int(
            len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch
        )

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

                # 2. 本地或全局弱增强预测达到阈值即采纳，但始终使用全局目标类别。
                with torch.no_grad():
                    probs_u_w_local = torch.softmax(logits_u_w / args.T, dim=-1)
                    max_probs_local, _ = torch.max(probs_u_w_local, dim=-1)
                    mask_local = max_probs_local.ge(args.threshold)

                    _, logits_u_w_global = self.global_model(inputs_u_w)
                    probs_u_w_global = torch.softmax(logits_u_w_global / args.T, dim=-1)
                    max_probs_global, targets_u = torch.max(probs_u_w_global, dim=-1)
                    mask_global = max_probs_global.ge(args.threshold)

                    mask = torch.logical_or(mask_local, mask_global).float()

                # 3. 无标签强增强的一致性 KL 损失，以全局模型硬伪标签为目标。
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1).clamp_min(1e-10)
                targets_u_one_hot = F.one_hot(
                    targets_u, num_classes=args.num_classes
                ).float()
                Lu = (
                    F.kl_div(
                        logits_u_s_probs.log(),
                        targets_u_one_hot,
                        reduction="none",
                    ).sum(dim=-1)
                    * mask
                ).mean()

                # 原型标签集合对比：低置信无标签样本使用弱增强原型概率生成候选集合。
                # 集合阈值由服务器用上一轮各客户端（有标签 + 高置信样本）的平均
                # 原型置信度下发；阈值未就绪时集合不生成、L_proto 为零，
                # 但置信度统计照常进行，为服务器积累阈值估计。
                L_proto = torch.zeros((), device=self.device)
                low_sets = None
                contrast_valid = None
                if global_anchors is not None:
                    anchors = global_anchors.to(self.device)
                    features_u_w, _ = features[batch_size:].chunk(2)
                    _, features_u_s = features[batch_size:].chunk(2)
                    with torch.no_grad():
                        proto_probs_u = prototype_probabilities(
                            features_u_w.detach(), anchors
                        )
                        proto_probs_x = prototype_probabilities(
                            features_x.detach(), anchors
                        )
                        if global_proto_conf is not None:
                            prior = torch.as_tensor(
                                global_proto_conf,
                                device=self.device,
                                dtype=features.dtype,
                            ).unsqueeze(0)
                            low_sets = proto_probs_u > prior
                            candidate_valid = low_sets.any(dim=1)

                            high_sets = F.one_hot(
                                targets_u, num_classes=args.num_classes
                            ).bool()
                            unlabeled_sets = torch.where(
                                mask.bool().unsqueeze(1), high_sets, low_sets
                            )
                            low_positive = (proto_probs_u * low_sets).matmul(anchors)
                            high_positive = anchors[targets_u]
                            unlabeled_positive = torch.where(
                                mask.bool().unsqueeze(1), high_positive, low_positive
                            )
                            contrast_valid = mask.bool() | candidate_valid

                            labeled_sets = F.one_hot(
                                targets_x, num_classes=args.num_classes
                            ).bool()
                            labeled_positive = anchors[targets_x]

                    # 阈值统计：仅有标签样本与高置信无标签样本参与。
                    # conf_sum[c] = 属于类 c 的可信样本对类 c 的原型概率之和。
                    if local_epoch + 1 == args.local_epochs:
                        one_hot_x = F.one_hot(targets_x, args.num_classes).float()
                        conf_sum += (one_hot_x * proto_probs_x).sum(dim=0)
                        conf_cnt += one_hot_x.sum(dim=0)
                        valid = mask.bool()
                        if valid.any():
                            one_hot_u = F.one_hot(
                                targets_u[valid], args.num_classes
                            ).float()
                            conf_sum += (one_hot_u * proto_probs_u[valid]).sum(dim=0)
                            conf_cnt += one_hot_u.sum(dim=0)

                    if contrast_valid is not None and contrast_valid.any():
                        contrast_features = torch.cat(
                            [
                                features_x,
                                features_u_w[contrast_valid],
                                features_u_s[contrast_valid],
                            ],
                            dim=0,
                        )
                        contrast_positive = torch.cat(
                            [
                                labeled_positive,
                                unlabeled_positive[contrast_valid],
                                unlabeled_positive[contrast_valid],
                            ],
                            dim=0,
                        )
                        contrast_sets = torch.cat(
                            [
                                labeled_sets,
                                unlabeled_sets[contrast_valid],
                                unlabeled_sets[contrast_valid],
                            ],
                            dim=0,
                        )
                        # 原型校准损失（MSE 版本）：特征直接回归到目标 anchor。
                        # 与集合对比 CE 不同，MSE 对有标签段也生效（真类 anchor）。
                        L_proto = F.mse_loss(contrast_features, contrast_positive)
                        # L_proto = prototype_set_contrastive_loss(
                        #     contrast_features,
                        #     contrast_positive,
                        #     contrast_sets,
                        #     unlabeled_start=batch_size,
                        # )

                # 总损失：FixMatch 分类目标加上原型标签集合对比目标。
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
                    valid = mask.bool()
                    if valid.any():
                        num_high_corrects += (
                            torch.eq(
                                targets_u[valid].cpu(),
                                targets_u_groundtruth[valid].cpu(),
                            )
                            .sum()
                            .item()
                        )

                    # 低置信样本：候选标签集 = 弱增强原型概率 > 类别先验（需全局 anchor）
                    low_mask = ~mask.bool()
                    if low_mask.any():
                        num_low_total += int(low_mask.sum().item())
                        if low_sets is not None:
                            low_sizes = low_sets.sum(dim=1)[low_mask]
                            gt_hits = (
                                low_sets[low_mask]
                                .gather(1, targets_u_groundtruth[low_mask].unsqueeze(1))
                                .squeeze(1)
                            )
                            for size, hit in zip(low_sizes.tolist(), gt_hits.tolist()):
                                set_size_hist[int(size)] += 1
                                set_size_hit_hist[int(size)] += int(hit)
                            num_gt_in_set_total += int(gt_hits.sum().item())

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

        prototypes, prototype_counts = self.compute_prototypes(
            args, data_client_labeled, data_client_unlabeled
        )
        proto_model_counts = self.prototype_model_eval(
            args,
            prototypes,
            prototype_counts,
            global_anchors,
            data_client_labeled,
            data_client_unlabeled,
        )

        pseudo_status = [
            num_pseudo_total,
            num_pseudo_corrects,
            num_u_valid,
            pseudo_client_acc,
            u_client_valid,
            num_low_total,
            num_gt_in_set_total,
            set_size_hist,
            set_size_hit_hist,
            conf_sum.tolist() if had_anchors else None,
            conf_cnt.tolist() if had_anchors else None,
            num_high_corrects,
            proto_model_counts,
        ]
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
        """用本地训练完成后的最终模型、仅基于有标签样本计算本地类别原型。"""
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
        return (sums / counts.clamp_min(1).unsqueeze(1)).cpu(), counts.cpu()

    @torch.no_grad()
    def prototype_model_eval(
        self,
        args,
        prototypes,
        prototype_counts,
        global_anchors,
        labeled_dataset,
        unlabeled_dataset,
    ):
        """本地/全局的原型最近邻与模型分类器，在本客户端数据上的准确率计数。

        返回 20 元素列表（5 组 × [有标签命中, 有标签总数, 无标签命中, 无标签总数]）：
        [0:4] 本地原型、[4:8] 本地模型、[8:12] 全局模型、[12:16] 全局原型(anchors)、
        [16:20] 初始锚点分类（anchors × 训练前的全局模型特征）。
        全局原型/初始锚点在第 1 轮缺失时总数为 0，显示层作 "—"。
        本地无样本的类别（零向量原型）不参与本地原型最近邻。
        """
        self.local_model.eval()
        local_proto = prototypes.to(self.device)
        valid_proto = prototype_counts.to(self.device) > 0
        counts = torch.zeros(20, device=self.device)
        has_anchors = global_anchors is not None
        anchors = global_anchors.to(self.device) if has_anchors else None

        def nearest(features, centers, valid_mask=None):
            distances = torch.cdist(features, centers)
            if valid_mask is not None:
                distances = distances.masked_fill(
                    ~valid_mask.unsqueeze(0), float("inf")
                )
            return distances.argmin(dim=1)

        loader = DataLoader(
            # 训练视图包含 repeat=2000；评估对每个原始有标签样本算一次。
            Subset(labeled_dataset, range(len(labeled_dataset.indices))),
            args.batch_size_local_labeled_fixmatch,
        )
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            features, logits = self.local_model(images)
            counts[0] += (nearest(features, local_proto, valid_proto) == labels).sum()
            counts[1] += labels.numel()
            counts[4] += (logits.argmax(dim=1) == labels).sum()
            counts[5] += labels.numel()
            global_feats, global_logits = self.global_model(images)
            counts[8] += (global_logits.argmax(dim=1) == labels).sum()
            counts[9] += labels.numel()
            if has_anchors:
                counts[12] += (nearest(features, anchors) == labels).sum()
                counts[13] += labels.numel()
                counts[16] += (nearest(global_feats, anchors) == labels).sum()
                counts[17] += labels.numel()

        loader = DataLoader(
            unlabeled_dataset,
            args.batch_size_local_labeled_fixmatch,
            shuffle=False,
        )
        for images, _, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            features, logits = self.local_model(images)
            counts[2] += (nearest(features, local_proto, valid_proto) == labels).sum()
            counts[3] += labels.numel()
            counts[6] += (logits.argmax(dim=1) == labels).sum()
            counts[7] += labels.numel()
            global_feats, global_logits = self.global_model(images)
            counts[10] += (global_logits.argmax(dim=1) == labels).sum()
            counts[11] += labels.numel()
            if has_anchors:
                counts[14] += (nearest(features, anchors) == labels).sum()
                counts[15] += labels.numel()
                counts[18] += (nearest(global_feats, anchors) == labels).sum()
                counts[19] += labels.numel()
        return counts.tolist()

    def interleave(self, x, size):
        """将有标签和无标签样本交错排列，以配合 BatchNorm 训练。"""
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        """恢复交错前的样本顺序。"""
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class ClientTrainer:
    """Worker 进程内的客户端训练适配器，供公共进程池调用。"""

    def __init__(self, args, device):
        self.local = Local(args, device=device)

    def train(self, task, labeled_view, unlabeled_view):
        (
            params,
            pseudo_status,
            prototypes,
            prototype_counts,
            elapsed_seconds,
        ) = self.local.train(
            task.args,
            labeled_view,
            unlabeled_view,
            task.global_params,
            task.global_anchors,
            global_proto_conf=getattr(task.args, "global_proto_conf", None),
        )
        return {
            "params": params,
            "pseudo_status": pseudo_status,
            "prototypes": prototypes,
            "prototype_counts": prototype_counts,
            "elapsed_seconds": elapsed_seconds,
        }


def fedavg_fixmatch(alpha, args=None):
    """执行纯净的联邦半监督学习 (FedAvg + FixMatch)。"""
    if args is None:
        args = args_parser()
    args.method = "test"

    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.num_labeled = 500
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
        logger.error("不支持的数据集：%s", args.dataset)
        sys.exit(1)

    # ==================== 注册实验运行（唯一目录 + SQLite 索引） ====================
    run = create_run(args)
    setup_logging(run.log_file, level=args.log_level)
    logger.info("运行 ID：%s，结果目录：%s", run.run_id, run.dir)

    log_args(args)

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
            seed=args.seed,
        )
        list_client2indices_unlabeled = clients_indices(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=args.seed,
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

    client_gpus = parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    global_model = Global(args)

    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(data_local_training)
    worker_pool = ClientWorkerPool(
        client_gpus,
        args,
        shared_dataset,
        trainer_cls=ClientTrainer,
        log_file=str(run.log_file),
    )

    total_clients = list(range(args.num_clients))

    fedavg_acc: list[float] = []
    fedavg_pseudo_acc: list[float] = []
    fedavg_num_valid: list[int] = []
    fedavg_valid_ratio: list[float] = []
    # 第一轮尚未有服务器端原型，聚合时作为 previous 传入 None。
    global_prototypes = None
    global_anchors = None
    # 全局原型置信度阈值：由服务器跨客户端平均（有标签 + 高置信样本）得到，
    # 首轮未就绪时为 None，客户端不生成候选集合。
    global_proto_conf = None
    args.global_proto_conf = None

    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
    for r in progress:
        logger.info("========== 第 %d 轮 ==========", r)
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
        sum_proto_counts = torch.stack(
            [result.prototype_counts for result in results]
        ).sum(dim=0)
        logger.info(
            "第 %d 轮：全局原型与 anchor 已更新，本轮 %d/%d 个类别有原型覆盖",
            r,
            int((sum_proto_counts > 0).sum()),
            len(sum_proto_counts),
        )

        for result in results:
            pseudo_status = result.pseudo_status
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]

        # ==================== 全局原型置信度阈值（跨客户端加权平均） ====================
        conf_sum_total = None
        conf_cnt_total = None
        for result in results:
            client_conf_sum, client_conf_cnt = (
                result.pseudo_status[9],
                result.pseudo_status[10],
            )
            if client_conf_sum is None:
                continue
            client_sum = torch.tensor(client_conf_sum)
            client_cnt = torch.tensor(client_conf_cnt)
            conf_sum_total = (
                client_sum if conf_sum_total is None else conf_sum_total + client_sum
            )
            conf_cnt_total = (
                client_cnt if conf_cnt_total is None else conf_cnt_total + client_cnt
            )
        if conf_sum_total is not None:
            # count=0 的类相除得到 NaN，proto_probs > NaN 恒为 False，
            # 该类在取得证据前自然不进入候选集。
            global_proto_conf = (conf_sum_total / conf_cnt_total).tolist()
            args.global_proto_conf = global_proto_conf
            logger.info(
                "全局原型置信度阈值：%s",
                np.round(np.asarray(global_proto_conf), 4),
            )

        # ==================== 伪标签与标签集统计（拼接为一条 DEBUG） ====================
        sum_total = sum_valid = sum_low = sum_gt_in_set = 0
        sum_high_corrects = 0
        sum_hist = None
        sum_hit_hist = None
        proto_model_total = [0] * 20
        lines = []

        def format_counts(hist, low_total):
            """标签集大小段的每桶计数与占低置信样本比例。"""
            parts = []
            for size, count in enumerate(hist):
                if not count:
                    continue
                if size == 0:
                    # 空标签集：一个候选类都没有
                    parts.append(f"0(空集): {count}(占{count / low_total:.1%})")
                else:
                    parts.append(f"{size}: {count}(占{count / low_total:.1%})")
            return "{" + ", ".join(parts) + "}" if parts else "空"

        def format_eval_pair(counts, offset):
            """原型/模型分类段的半边：有标签与无标签两个准确率。"""
            parts = []
            for offset_base, name in ((offset, "有标签"), (offset + 2, "无标签")):
                hit, total = int(counts[offset_base]), int(counts[offset_base + 1])
                parts.append(
                    f"{name} {hit / total:.1%}({hit}/{total})" if total else f"{name} —"
                )
            return f"[{', '.join(parts)}]"

        def format_acc(hist, hit_hist):
            """准确率段的逐桶命中率；零命中的桶（如空集合）不显示。"""
            parts = [
                f"{size}: {hit_hist[size] / count:.1%}({hit_hist[size]}/{count})"
                for size, count in enumerate(hist)
                if count and hit_hist[size]
            ]
            return "{" + ", ".join(parts) + "}" if parts else "空"

        for result in results:
            status = result.pseudo_status
            num_total, num_valid = status[0], status[2]
            num_low, num_gt_in_set, hist, hit_hist = (
                status[5],
                status[6],
                status[7],
                status[8],
            )
            num_high_corrects = status[11]
            proto_model_counts = status[12]
            for idx, value in enumerate(proto_model_counts):
                proto_model_total[idx] += value
            sum_total += num_total
            sum_valid += num_valid
            sum_low += num_low
            sum_gt_in_set += num_gt_in_set
            sum_high_corrects += num_high_corrects
            if sum_hist is None:
                sum_hist = list(hist)
                sum_hit_hist = list(hit_hist)
            else:
                sum_hist = [a + b for a, b in zip(sum_hist, hist)]
                sum_hit_hist = [a + b for a, b in zip(sum_hit_hist, hit_hist)]

            valid_part = (
                f"{num_valid / num_total:.1%}({num_valid}/{num_total})"
                if num_total
                else "0.0%(0/0)"
            )
            high_acc_part = (
                f"{num_high_corrects / num_valid:.1%}({num_high_corrects}/{num_valid})"
                if num_valid
                else "—"
            )
            if num_low:
                hit_part = f"{num_gt_in_set / num_low:.1%}({num_gt_in_set}/{num_low})"
                if any(hist):
                    acc_part = format_acc(hist, hit_hist)
                    set_part = format_counts(hist, num_low)
                else:
                    # anchor 或阈值未就绪，本轮未生成候选集合
                    acc_part = "无集合"
                    set_part = "无集合"
            else:
                set_part = "无低置信样本"
                acc_part = "—"
                hit_part = "—"
            lines.append(
                f"客户端 {result.client_id}：高置信 {valid_part}｜"
                f"高置信伪标签准确率 {high_acc_part}｜"
                f"低置信标签集大小 {set_part}｜真实标签∈标签集 {acc_part}｜整体 {hit_part}"
            )
            lines.append(
                f"客户端 {result.client_id}：本地原型分类 {format_eval_pair(proto_model_counts, 0)}｜"
                f"本地模型分类 {format_eval_pair(proto_model_counts, 4)}｜"
                f"全局模型分类 {format_eval_pair(proto_model_counts, 8)}｜"
                f"全局原型分类 {format_eval_pair(proto_model_counts, 12)}｜"
                f"初始锚点分类 {format_eval_pair(proto_model_counts, 16)}"
            )
        total_valid_part = (
            f"{sum_valid / sum_total:.1%}({sum_valid}/{sum_total})"
            if sum_total
            else "0.0%(0/0)"
        )
        total_high_acc_part = (
            f"{sum_high_corrects / sum_valid:.1%}({sum_high_corrects}/{sum_valid})"
            if sum_valid
            else "—"
        )
        total_hit_part = (
            f"{sum_gt_in_set / sum_low:.1%}({sum_gt_in_set}/{sum_low})"
            if sum_low
            else "—"
        )
        total_set_part = (
            format_counts(sum_hist, sum_low)
            if sum_low and any(sum_hist)
            else ("无低置信样本" if not sum_low else "无集合")
        )
        total_acc_part = (
            format_acc(sum_hist, sum_hit_hist)
            if sum_low and any(sum_hist)
            else ("无低置信样本" if not sum_low else "无集合")
        )
        lines.append(
            f"第 {r} 轮合计：高置信 {total_valid_part}｜"
            f"高置信伪标签准确率 {total_high_acc_part}｜"
            f"低置信标签集大小 {total_set_part}｜真实标签∈标签集 {total_acc_part}｜整体 {total_hit_part}"
        )
        lines.append(
            f"第 {r} 轮合计：本地原型分类 {format_eval_pair(proto_model_total, 0)}｜"
            f"本地模型分类 {format_eval_pair(proto_model_total, 4)}｜"
            f"全局模型分类 {format_eval_pair(proto_model_total, 8)}｜"
            f"全局原型分类 {format_eval_pair(proto_model_total, 12)}｜"
            f"初始锚点分类 {format_eval_pair(proto_model_total, 16)}"
        )
        logger.debug("第 %d 轮伪标签统计：\n%s", r, "\n".join(lines))

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
        logger.info("第 %d 轮全局模型精度：%.2f%%", r, global_acc * 100)

        progress.set_postfix(acc=f"{global_acc:.2%}")

        result_dir_spec = run.checkpoint_dir

        if (
            r == 1
            or r == args.num_rounds
            or (r % 50 == 0 and r > 0.8 * args.num_rounds)
        ):
            torch.save(fedavg_params, f"{result_dir_spec}/fedavg_params_round_{r}.pth")

        result_file = run.dir / "metrics.csv"
        acc_df = pd.DataFrame(
            {"acc": fedavg_acc}, index=list(range(1, len(fedavg_acc) + 1))
        )
        acc_df.to_csv(result_file, encoding="utf8")

        result_pseudo_file = run.dir / "pseudo_metrics.csv"
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
    run.finish(
        best_acc=max(fedavg_acc) if fedavg_acc else None, num_rounds=args.num_rounds
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
