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
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
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
    """服务端状态：维护全局模型、代理分类器并完成聚合评估。"""

    def __init__(self, args):
        """在指定的服务端 GPU 上创建全局模型和 GPT 代理分类器。"""
        self.args = args
        self.gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
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

        self.GPT = nn.Linear(
            in_features=self.model.dim,
            out_features=self.num_classes,
        )
        self.GPT.to(self.device)
        self.GPT_opt = SGD(self.GPT.parameters(), lr=args.lr_server)

        self.server_epochs = args.total_server_epochs // args.num_rounds
        self.ema_beta = 0.99

    def initialize_for_model_fusion(
        self, list_dicts_local_params: list, list_nums_local_data: list
    ):
        """按客户端样本量执行 FedAvg，并使用聚合分类器更新 GPT。"""
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        fedavg_mlp_params = {}
        for name_param in list_dicts_local_params[0]:
            first_value = list_dicts_local_params[0][name_param]
            if not torch.is_floating_point(first_value):
                # BatchNorm 的整数缓冲区不能参与带小数的 FedAvg，保留第一个客户端的值。
                fedavg_global_params[name_param] = first_value.clone()
                continue

            list_values_param = []
            for dict_local_params, num_local_data in zip(
                list_dicts_local_params, list_nums_local_data
            ):
                list_values_param.append(dict_local_params[name_param] * num_local_data)

            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param

            if "classifier" in name_param:
                new_key = name_param[len("classifier.") :]
                fedavg_mlp_params[new_key] = value_global_param

        self.update_GPT(list_dicts_local_params, fedavg_mlp_params)
        for name, param in self.GPT.named_parameters():
            full_param_name = f"classifier.{name}"
            if full_param_name in fedavg_global_params:
                fedavg_global_params[full_param_name] = param.detach().cpu().clone()
            else:
                logger.warning(
                    f"Parameter {full_param_name} not found in global params"
                )

        all_classifier_weights = []
        # 遍历每个客户端的参数
        for client_params in list_dicts_local_params:
            # 提取分类器权重（假设键名为 classifier.weight）
            if "classifier.weight" in client_params:
                classifier_weights = client_params["classifier.weight"]
                all_classifier_weights.append(classifier_weights)

        return fedavg_global_params, all_classifier_weights

    def update_GPT(self, list_dicts_local_params: list, fedavg_mlp_params: dict):
        """使用客户端分类器权重训练服务端代理分类器。"""
        self.GPT.load_state_dict(
            {name: value.to(self.device) for name, value in fedavg_mlp_params.items()}
        )
        self.GPT.train()
        uploaded_proxies = self.upload_proxies(list_dicts_local_params)
        max_dist = self.update_max_dist(fedavg_mlp_params)

        # 服务端代理训练
        proxy_loader = DataLoader(
            uploaded_proxies,
            self.args.bs_server,
            drop_last=False,
            shuffle=True,
        )
        for _ in range(self.server_epochs):
            for proxy, y in proxy_loader:
                proxy = proxy.to(self.device)
                y = y.to(self.device)
                proxy_g = self.GPT.weight
                dist = torch.cdist(proxy, proxy_g)

                one_hot = F.one_hot(y, self.num_classes)
                dist = dist + one_hot * min(max_dist.item(), self.args.gpt_threshold)
                loss = F.cross_entropy(-dist, y)

                self.GPT_opt.zero_grad()
                loss.backward()
                self.GPT_opt.step()

        self.GPT.eval()

    def upload_proxies(self, list_dicts_local_params: list):
        """将每个客户端的分类器权重整理为代理样本和类别标签。"""
        # 1. 从客户端参数中提取分类器权重并构建训练数据集
        all_classifier_weights = []
        all_class_labels = []

        for client_params in list_dicts_local_params:
            if "classifier.weight" in client_params:
                classifier_weights = client_params["classifier.weight"]

                if classifier_weights.dim() == 2:
                    num_classes, _ = classifier_weights.shape

                    for class_id in range(num_classes):
                        weight_vector = classifier_weights[class_id]
                        all_classifier_weights.append(weight_vector)
                        all_class_labels.append(class_id)

        classifier_tensors = torch.stack(all_classifier_weights)
        label_tensors = torch.tensor(all_class_labels, dtype=torch.long)

        return TensorDataset(classifier_tensors, label_tensors)

    def update_max_dist(self, fedavg_mlp_params):
        """计算不同类别代理之间最近距离中的最大值。"""
        avg_proxies = fedavg_mlp_params["weight"]
        dist_mat = torch.cdist(avg_proxies, avg_proxies, p=2)  # (C, C)
        dist_mat.fill_diagonal_(float("inf"))
        min_dist_cls = torch.min(dist_mat, dim=-1)[0].to(self.device)
        max_dist_all_cls = torch.max(min_dist_cls)

        return max_dist_all_cls

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        """在测试集上评估聚合后的全局模型。"""
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
        """复制全局模型参数到 CPU，作为客户端任务的输入。"""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def update_global_distribution(
        self,
        current_global_dist: np.ndarray | None = None,
        list_class_counts: list[np.ndarray] | None = None,
    ):
        """根据客户端类别统计量更新 EMA 全局类别分布。"""
        """
        输入: 所有客户端上传的 class counts 列表 (list of numpy arrays)
        输出: 经过 EMA 平滑后的全局分布 (numpy array)
        """
        if current_global_dist is None:
            logger.info("Initialized Global EMA Distribution.")
            return np.ones(self.num_classes) / self.num_classes

        # 1. 计算本轮的实时类别分布
        if not list_class_counts:
            return current_global_dist
        total_counts = np.sum(np.stack(list_class_counts), axis=0)
        total_sum = np.sum(total_counts)

        if total_sum == 0:
            return current_global_dist

        current_round_dist = total_counts / total_sum
        current_global_dist = (
            self.ema_beta * current_global_dist
            + (1 - self.ema_beta) * current_round_dist
        )

        logger.info(f"Current Round Dist: {current_round_dist}")
        logger.info(f"Updated EMA Global Dist: {current_global_dist}")

        return current_global_dist


class Local:
    """客户端训练器，封装本地模型、教师模型、优化器和 FixMatch 损失。"""

    def __init__(self, args, device=None):
        """在指定设备上创建客户端训练所需的两个模型。"""
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

        self.local_G = ResNet_PC(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        )

        self.local_model.to(self.device)
        self.local_G.to(self.device)

        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.num_classes = args.num_classes

    def fixmatch_train(
        self,
        args,
        data_client_labeled,
        data_client_unlabeled,
        global_params,
        r,
        client_idx,
        global_class_dist=None,
    ):
        """使用当前全局参数执行一次客户端本地训练并返回结果。"""
        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
        )

        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
        )

        self.local_model.load_state_dict(global_params)
        # worker 会复用本地模型，但不同客户端之间不能共享 SGD 动量。
        self.optimizer.state.clear()
        self.local_model.train()

        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        # 初始化本地类别计数器
        local_class_counts = torch.zeros(args.num_classes, device=self.device)

        # 初始化伪标签指标统计变量
        num_pseudo_corrects = 0
        num_pseudo_total = 0
        num_u_valid = 0
        pseudo_client_acc = 0.0
        u_client_valid = 0.0

        # 默认本地训练轮数为 5
        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            local_iter = int(
                len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch
            )

            for _ in range(local_iter):
                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                except StopIteration:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = (
                        unlabeled_iter.__next__()
                    )
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = (
                        unlabeled_iter.__next__()
                    )

                inputs_x = inputs_x.to(self.device)
                inputs_u_w = inputs_u_w.to(self.device)
                inputs_u_s = inputs_u_s.to(self.device)
                targets_x = targets_x.to(self.device)
                targets_u_groundtruth = targets_u_groundtruth.to(
                    self.device
                )

                batch_size = inputs_x.shape[0]
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                )

                feats, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                feats = self.de_interleave(feats, 2 * args.mu + 1)

                with torch.no_grad():
                    _, logits_glob = self.local_G(inputs)
                    logits_glob = self.de_interleave(logits_glob, 2 * args.mu + 1)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                logits_x_glob = logits_glob[:batch_size]
                logits_u_w_glob, _ = logits_glob[batch_size:].chunk(2)

                feats_x = feats[:batch_size]
                # ICPL 同时使用无标签弱增强和强增强的特征。
                feats_u = feats[batch_size:]

                # 1. 有标签数据的交叉熵损失
                Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

                # 2. 无标签数据伪标签与损失
                with torch.no_grad():
                    # 本地弱增强预测
                    probs_u_w_local = torch.softmax(logits_u_w / args.T, dim=-1)
                    max_probs_local, _ = torch.max(probs_u_w_local, dim=-1)
                    mask_local = max_probs_local.ge(args.threshold).float()

                    # 全局弱增强预测
                    probs_u_w_glob = torch.softmax(logits_u_w_glob / args.T, dim=-1)
                    max_probs_glob, targets_u_global = torch.max(probs_u_w_glob, dim=-1)
                    mask_global = max_probs_glob.ge(args.threshold).float()

                    # 全局强增强预测
                    pseudo_label_global = torch.softmax(
                        logits_u_w_glob / args.T, dim=-1
                    )
                    targets_u_global_one_hot = F.one_hot(
                        targets_u_global, args.num_classes
                    )

                    probs_x_glob = torch.softmax(logits_x_glob / args.T, dim=-1)
                    max_probs_x_glob, _ = torch.max(probs_x_glob, dim=-1)

                mask_valid = torch.max(mask_local, mask_global)
                mask_x = max_probs_x_glob.ge(args.threshold).float()

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1)
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = (
                    targets_u_global_one_hot + 1e-10
                )  # final_targets_u + 1e-10

                Lu = (
                    F.kl_div(
                        logits_u_s_probs.log(), final_targets_u, reduction="none"
                    ).sum(-1)
                    * mask_valid
                ).mean()

                projs_u = self.local_model.feat_proj(feats_u)
                projs_x = self.local_model.feat_proj(feats_x)
                projs = torch.cat([projs_x, projs_u], dim=0)

                projs_prob = torch.cat(
                    [probs_x_glob, pseudo_label_global, pseudo_label_global], dim=0
                ).detach()
                conf_mask = torch.cat([mask_x, mask_valid, mask_valid], dim=0).bool()
                Lc = self.ICPL(
                    projs,
                    projs_prob,
                    conf_mask,
                    conf_x_num=projs_x.shape[0],
                    prior_distribution=global_class_dist,
                )

                loss = Lx + args.lambda_u * Lu + args.lambda_u * Lc

                # 统计最后一个 epoch 的类别和伪标签指标
                if local_epoch + 1 == args.local_epochs:
                    ##############

                    # 1. 统计有标签数据 targets_x
                    labels_one_hot = (
                        F.one_hot(targets_x, args.num_classes).float().sum(dim=0)
                    )
                    local_class_counts += labels_one_hot

                    # 2. 统计由 mask_valid 选中的高置信度无标签数据
                    # 这里统计实际参与训练（被 mask 选中）的样本的伪标签类别
                    if mask_valid.sum() > 0:
                        valid_indices = mask_valid.bool()
                        pseudo_labels_selected = targets_u_global[valid_indices]
                        pseudo_one_hot = (
                            F.one_hot(pseudo_labels_selected, args.num_classes)
                            .float()
                            .sum(dim=0)
                        )
                        local_class_counts += pseudo_one_hot

                    ##############
                    num_pseudo_corrects += (
                        torch.eq(targets_u_global.cpu(), targets_u_groundtruth.cpu())
                        .sum()
                        .item()
                    )
                    num_pseudo_total += len(targets_u_global)
                    num_u_valid += int(mask_valid.sum().item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 输出最后一个本地 epoch 的伪标签指标
            if local_epoch + 1 == args.local_epochs:
                pseudo_client_acc = (
                    num_pseudo_corrects / num_pseudo_total if num_pseudo_total else 0.0
                )
                u_client_valid = (
                    num_u_valid / num_pseudo_total if num_pseudo_total else 0.0
                )

                # 记录日志
                logger.info(
                    f"Round {r}, Local Epoch {local_epoch}, Client {client_idx}, pseudo_acc = {pseudo_client_acc: .4f}, pseudo_num_valid = {num_u_valid}, valid_ratio = {u_client_valid}"
                )

        pseudo_status = [
            num_pseudo_total,
            num_pseudo_corrects,
            num_u_valid,
            pseudo_client_acc,
            u_client_valid,
        ]
        return (
            {
                name: value.detach().cpu().clone()
                for name, value in self.local_model.state_dict().items()
            },
            pseudo_status,
            local_class_counts.cpu().numpy(),
        )

    def ICPL(self, feature, projs_prob, conf_mask, prior_distribution, conf_x_num=None):
        """计算基于代理相似度、类别先验和置信度掩码的 ICPL 损失。"""
        feature = F.normalize(feature, p=2, dim=1)
        proxy = self.local_model.classifier.weight
        proxy = F.normalize(self.local_model.proxy_proj(proxy), p=2.0, dim=1)

        if prior_distribution is None:
            prior_tensor = torch.full(
                (1, projs_prob.size(1)),
                1.0 / projs_prob.size(1),
                device=projs_prob.device,
                dtype=projs_prob.dtype,
            )
        elif not isinstance(prior_distribution, torch.Tensor):
            prior_tensor = torch.tensor(
                prior_distribution, device=projs_prob.device, dtype=projs_prob.dtype
            )
        else:
            prior_tensor = prior_distribution.to(projs_prob.device)
        # 将维度对齐为 (1, C)
        if prior_tensor.dim() == 1:
            prior_tensor = prior_tensor.unsqueeze(0)
        sets_class = projs_prob > prior_tensor

        top1_class = F.one_hot(
            projs_prob.argmax(dim=1), num_classes=projs_prob.size(1)
        ).bool()
        pred_class = sets_class * ~conf_mask.unsqueeze(
            1
        ) + top1_class * conf_mask.unsqueeze(1)

        # 计算候选类别加权的代理相似度
        candidate_weight = projs_prob * sets_class
        candidate_proxy = (candidate_weight.unsqueeze(2) * proxy.unsqueeze(0)).sum(
            dim=1
        )
        candidate_sim = (feature * candidate_proxy).sum(dim=1)

        # 计算最高概率类别的代理相似度
        pred = feature @ proxy.T
        target = projs_prob.argmax(dim=1)
        proxy_sim = pred[torch.arange(feature.size(0), device=feature.device), target]

        pos_pair = torch.where(conf_mask, proxy_sim, candidate_sim)

        # 负样本掩码：预测类别集合没有交集的样本视为负样本。
        overlap = (pred_class.unsqueeze(1) & pred_class.unsqueeze(0)).any(dim=2)
        neg_matrix = ~overlap

        pairwise_sim = feature @ feature.T
        neg_pair = pairwise_sim.masked_fill(
            ~neg_matrix | (pairwise_sim < 1e-6), float("-inf")
        )

        logits = torch.cat([pos_pair.unsqueeze(1), neg_pair], dim=1)

        if conf_x_num is not None:
            logits = logits[conf_x_num:]

        label = torch.zeros(logits.size(0), dtype=torch.long, device=feature.device)
        loss = F.cross_entropy(logits, label)

        return loss

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
    """主进程发送给客户端 Worker 的一轮训练任务。"""

    round: int
    client_id: int
    labeled_indices: list
    unlabeled_indices: list
    global_params: dict
    global_class_dist: np.ndarray | None
    args: object


@dataclasses.dataclass
class ClientResult:
    """客户端 Worker 返回的训练结果或异常信息。"""

    ok: bool
    client_id: int
    gpu_id: int
    params: dict | None = None
    num_samples: int = 0
    pseudo_status: list | None = None
    class_counts: np.ndarray | None = None
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


def _client_worker(gpu_id, args, shared_dataset, task_queue, result_queue):
    """Worker 进程：在固定 GPU 上循环领取任务并执行客户端训练。"""
    try:
        # spawn 创建的子进程需要重新设置 Tensor 共享策略。
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

                params, pseudo_status, class_counts = local.fixmatch_train(
                    task.args,
                    labeled_view,
                    unlabeled_view,
                    task.global_params,
                    task.round,
                    task.client_id,
                    global_class_dist=task.global_class_dist,
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
                        class_counts=class_counts,
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
        # 初始化失败时没有具体客户端编号，使用 -1 让主进程记录设备信息并终止本轮。
        result_queue.put(
            ClientResult(
                ok=False,
                client_id=-1,
                gpu_id=gpu_id,
                error=f"Worker initialization failed ({type(exc).__name__}: {exc}):\n{traceback.format_exc()}",
            )
        )


class ClientWorkerPool:
    """管理客户端训练进程，并通过队列分发任务、收集结果。"""

    def __init__(self, gpu_ids, args, shared_dataset):
        """按 GPU 槽位启动 Worker；同一 GPU 可对应多个进程。"""
        # 使用共享内存文件传递 Tensor，避免默认文件描述符策略耗尽 FD。
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
                ),
                name=f"proxyfl-client-gpu-{gpu_id}",
            )
            process.start()
            self.processes.append(process)

    def run_round(self, tasks):
        """提交一轮客户端任务，等待全部结果并按客户端编号排序。"""
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
        """发送退出信号并正常回收所有 Worker。"""
        if self.closed:
            return
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()
        self.closed = True

    def terminate(self):
        """强制终止并回收异常或未完成的 Worker。"""
        if self.closed:
            return
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        self.closed = True


def _parse_worker_gpus(args):
    """解析 GPU 和进程数配置，返回按 Worker 展开的 GPU 编号列表。"""
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
        raise RuntimeError("多 GPU 客户端训练需要可用的 CUDA 环境")

    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"GPU 编号 {invalid} 不可用；当前可见 GPU 数量为 {device_count}"
        )

    args.server_gpu = args.server_gpu if args.server_gpu is not None else gpu_ids[0]
    if args.server_gpu not in gpu_ids:
        raise ValueError("--server_gpu 必须包含在 --client_gpus 中")

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


def fixmatch(alpha, args=None):
    """执行完整的 ProxyFL：数据划分、并行本地训练、聚合、评估和保存。"""

    # ==================== 初始化参数和日志 ====================
    if args is None:
        args = args_parser()
    args.method = f"ProxyFL_{args.total_server_epochs // 1000}k"

    log_dir = f"./results/{args.dataset}/logs"
    os.makedirs(log_dir, exist_ok=True)
    cr_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_file = os.path.join(log_dir, f"{args.method}_α={alpha}_{cr_time}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
    )

    # ==================== 创建训练集和测试集 ====================
    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        args.total_server_epochs = 30000
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

    elif (
        args.dataset == "CIFAR100"
    ):  # training:50k; testing:10k; for training, each class includes 500 images
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        args.total_server_epochs = 5000
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
        args.total_server_epochs = 15000
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
        args.total_server_epochs = 40000
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)
                ),
            ]
        )
        data_local_training = CINIC10(
            root=args.path, split="train", transform=None
        )
        data_global_test = CINIC10(
            root=args.path, split="test", transform=transform_test
        )

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN."
        )
        sys.exit(1)

    logger.info(
        f"dataset:{args.dataset}\n"
        f"num_classes:{args.num_classes}\n"
        f"num_labeled:{args.num_labeled}\n"
        f"non_iid:{args.alpha}\n"
        f"mu:{args.mu}\n"
        f"num_rounds:{args.num_rounds}\n"
        f"batch_label:{args.batch_size_local_labeled_fixmatch}, "
        f"batch_unlabel:{args.batch_size_local_labeled_fixmatch * args.mu}"
    )

    # ==================== 按类别划分数据索引 ====================
    random_state = np.random.RandomState(args.seed)
    # 按类别收集数据集样本下标
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # 从每个类别中随机抽取 ipc 个样本，其余样本作为无标签数据
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(
        list_label2indices, args.num_labeled
    )

    # 独立同分布划分
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
    # 非独立同分布划分
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

    # ==================== 划分客户端数据并显示分布 ====================
    show_clients_data_distribution(
        data_local_training,
        list_client2indices_labeled,
        list_client2indices_unlabeled,
        args.num_classes,
    )

    # 将有标签样本的索引加入无标签数据索引，训练时不使用其标签。
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client] = np.concatenate(
            [
                np.asarray(list_client2indices_unlabeled[client]),
                np.asarray(list_client2indices_labeled[client]),
            ]
        )

    # ==================== 创建服务端和客户端 Worker ====================
    client_gpus = _parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    global_model = Global(args)
    # 在创建共享数据集之前设置共享策略，避免预加载阶段产生大量文件描述符。
    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(data_local_training)
    worker_pool = ClientWorkerPool(client_gpus, args, shared_dataset)

    total_clients = list(range(args.num_clients))

    fedavg_acc = []
    fedavg_pseudo_acc = []
    fedavg_num_valid = []
    fedavg_valid_ratio = []

    current_global_dist = global_model.update_global_distribution()

    # ==================== 联邦学习主循环 ====================
    for r in tqdm(range(1, args.num_rounds + 1), desc="Server"):
        list_local_class_counts = []

        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(
            total_clients, args.num_online_clients, replace=False
        )
        # 生成本轮任务，并行执行客户端本地训练。
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
                global_class_dist=current_global_dist,
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        list_dicts_local_params = [result.params for result in results]
        list_nums_local_data = [result.num_samples for result in results]

        for result in results:
            pseudo_status = result.pseudo_status
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]
            list_local_class_counts.append(result.class_counts)

        pseudo_acc = (
            num_clients_u_corrects / num_clients_u_total if num_clients_u_total else 0.0
        )
        pseudo_valid_ratio = (
            num_clients_u_valid / num_clients_u_total if num_clients_u_total else 0.0
        )
        fedavg_pseudo_acc.append(pseudo_acc)
        fedavg_valid_ratio.append(pseudo_valid_ratio)
        fedavg_num_valid.append(num_clients_u_valid)

        # 根据客户端统计更新全局类别分布，并聚合模型参数。
        current_global_dist = global_model.update_global_distribution(
            current_global_dist, list_local_class_counts
        )

        fedavg_params, all_classifier_weights = (
            global_model.initialize_for_model_fusion(
                list_dicts_local_params, list_nums_local_data
            )
        )

        # 评估聚合后的全局模型。
        global_acc = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test
        )
        fedavg_acc.append(global_acc)

        print(
            f"round {r}, accuracy:{global_acc}, pseudo_acc:{fedavg_pseudo_acc[-1]}, num_valid:{fedavg_num_valid[-1]}, valid_ratio:{fedavg_valid_ratio[-1]}"
        )

        # ==================== 保存模型和训练指标 ====================
        result_dir = f"./results/{args.dataset}"
        os.makedirs(result_dir, exist_ok=True)

        # 创建当前实验的结果目录
        result_dir_spec = f"{result_dir}/{args.method}_α={alpha}_{cr_time}"
        os.makedirs(result_dir_spec, exist_ok=True)

        if (
            r == 1
            or r == args.num_rounds
            or (r % 50 == 0 and r > 0.8 * args.num_rounds)
        ):
            # 保存 fedavg_params
            torch.save(fedavg_params, f"{result_dir_spec}/fedavg_params_round_{r}.pth")
            # 保存 all_classifier_weights
            torch.save(
                all_classifier_weights,
                f"{result_dir_spec}/all_classifier_weights_round_{r}.pt",
            )
            # 同时保存为 numpy 格式以便后续分析
            classifier_weights_numpy = [
                weight.cpu().numpy() for weight in all_classifier_weights
            ]
            np.save(
                f"{result_dir_spec}/all_classifier_weights_round_{r}.npy",
                classifier_weights_numpy,
            )
            print(f"Saved model and proxy for round {r}")

        result_file = f"{result_dir}/{args.method}_α={alpha}_{cr_time}.csv"
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc) + 1))
        acc_num_pseudo_label_csv_df = pd.DataFrame(
            {"acc": fedavg_acc}, index=acc_num_pseudo_label_csv_index
        )
        # 保存文件
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding="utf8")

        result_pseudo_file = (
            f"{result_dir}/{args.method}_α={alpha}_pseudo_{cr_time}.csv"
        )
        # 取各项指标长度的最小值，确保 CSV 行数一致
        min_length = min(
            len(fedavg_pseudo_acc),
            len(fedavg_valid_ratio),
            len(fedavg_num_valid),
            len(fedavg_acc),
        )
        # 其他低置信度指标暂未写入结果文件

        # 创建 DataFrame，包含所有指标
        metrics_df = pd.DataFrame(
            {
                "round": list(range(1, min_length + 1)),
                "acc": fedavg_acc[:min_length],
                "pseudo_acc": fedavg_pseudo_acc[:min_length],
                "valid_ratio": fedavg_valid_ratio[:min_length],
                "num_valid": fedavg_num_valid[:min_length],
            }
        )
        # 设置轮次为索引
        metrics_df.set_index("round", inplace=True)

        # 保存文件
        metrics_df.to_csv(result_pseudo_file, encoding="utf8")
        print(f"Metrics saved to {result_pseudo_file}")

    # 所有轮次完成后，向 Worker 发送退出信号并回收进程。
    worker_pool.close()


if __name__ == "__main__":
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    fixmatch(args.alpha, args)
