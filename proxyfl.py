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
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
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


class Global:
    """服务端状态：维护全局模型、代理分类器并完成聚合评估。"""

    def __init__(self, args):
        """在指定的服务端 GPU 上创建全局模型和 GPT 代理分类器。"""
        self.args = args
        self.gpu_id = args.server_gpu if args.server_gpu is not None else args.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model = build_model(args)

        self.model.to(self.device)
        self.num_classes = args.num_classes

        self.GPT = nn.Linear(
            in_features=self.model.dim,
            out_features=self.num_classes,
        )
        self.GPT.to(self.device)
        self.GPT_opt = SGD(self.GPT.parameters(), lr=args.lr_server)

        self.server_epochs = args.server_epochs
        self.ema_beta = 0.99

    def initialize_for_model_fusion(
        self, list_dicts_local_params: list, list_nums_local_data: list
    ):
        """按客户端样本量执行 FedAvg，并使用聚合分类器更新 GPT。"""
        t_fusion_0 = time.perf_counter()
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

        t_fedavg = time.perf_counter() - t_fusion_0

        t_gpt_0 = time.perf_counter()
        self.update_GPT(list_dicts_local_params, fedavg_mlp_params)
        t_update_gpt = time.perf_counter() - t_gpt_0
        for name, param in self.GPT.named_parameters():
            full_param_name = f"classifier.{name}"
            if full_param_name in fedavg_global_params:
                fedavg_global_params[full_param_name] = param.detach().cpu().clone()
            else:
                logger.warning("参数 %s 不在全局参数中", full_param_name)

        all_classifier_weights = []
        # 遍历每个客户端的参数
        for client_params in list_dicts_local_params:
            # 提取分类器权重（假设键名为 classifier.weight）
            if "classifier.weight" in client_params:
                classifier_weights = client_params["classifier.weight"]
                all_classifier_weights.append(classifier_weights)

        return fedavg_global_params, all_classifier_weights, t_fedavg, t_update_gpt

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
            logger.info("已初始化全局 EMA 类别分布。")
            return np.ones(self.num_classes) / self.num_classes

        # 1. 计算本轮的实时类别分布
        if not list_class_counts:
            return current_global_dist
        total_counts = np.sum(np.stack(list_class_counts), axis=0)
        total_sum = np.sum(total_counts)

        if total_sum == 0:
            return current_global_dist

        current_round_dist = total_counts / total_sum
        updated_dist = (
            self.ema_beta * current_global_dist
            + (1 - self.ema_beta) * current_round_dist
        )

        logger.debug(
            "类别分布：\n当前轮：%s\n更新后的 EMA 全局：%s",
            np.round(current_round_dist, 4),
            np.round(updated_dist, 4),
        )

        return updated_dist


class Local:
    """客户端训练器，封装全局模型、局部模型、优化器和 FixMatch 损失。"""

    def __init__(self, args, device=None):
        """在指定设备上创建客户端训练所需的两个模型。"""
        self.device = device or torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        self.local_model = build_model(args)

        self.local_G = build_model(args)

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
        # 高置信子集的伪标签正确数（仅统计 mask_valid 选中的样本）
        num_high_corrects = 0
        # 低置信样本统计：标签集大小直方图与真实标签命中率
        # 直方图按大小直接索引（0 ~ num_classes），大小 0 表示无候选类；
        # hit_hist 记录各大小桶内真实标签∈标签集的命中数
        num_low_total = 0
        num_gt_in_set_total = 0
        set_size_hist = [0] * (args.num_classes + 1)
        set_size_hit_hist = [0] * (args.num_classes + 1)
        pseudo_client_acc = 0.0
        u_client_valid = 0.0
        model_counts = torch.zeros(8, device=self.device)

        train_start = time.perf_counter()
        t_data = 0.0
        t_fwd_local = 0.0
        t_fwd_glob = 0.0
        t_loss_fixmatch = 0.0
        t_loss_icpl = 0.0
        t_stats = 0.0
        t_bwd_opt = 0.0

        local_iter = int(
            len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch
        )

        # 默认本地训练轮数为 5
        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            for _ in range(local_iter):
                t_i0 = time.perf_counter()
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
                targets_u_groundtruth = targets_u_groundtruth.to(self.device)
                t_i1 = time.perf_counter()
                t_data += t_i1 - t_i0

                batch_size = inputs_x.shape[0]
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                )

                feats, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                feats = self.de_interleave(feats, 2 * args.mu + 1)
                t_i2 = time.perf_counter()
                t_fwd_local += t_i2 - t_i1

                with torch.no_grad():
                    _, logits_glob = self.local_G(inputs)
                    logits_glob = self.de_interleave(logits_glob, 2 * args.mu + 1)
                t_i3 = time.perf_counter()
                t_fwd_glob += t_i3 - t_i2

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
                t_i4 = time.perf_counter()
                t_loss_fixmatch += t_i4 - t_i3

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
                t_i5 = time.perf_counter()
                t_loss_icpl += t_i5 - t_i4

                t_stat_start = time.perf_counter()
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
                    if mask_valid.sum() > 0:
                        valid_idx = mask_valid.bool()
                        num_high_corrects += (
                            torch.eq(
                                targets_u_global[valid_idx].cpu(),
                                targets_u_groundtruth[valid_idx].cpu(),
                            )
                            .sum()
                            .item()
                        )

                    # 低置信样本：候选标签集 = 全局概率 > 类别先验（与 ICPL 的 sets_class 同构）
                    low_mask = ~mask_valid.bool()
                    if low_mask.any():
                        if global_class_dist is None:
                            prior_u = torch.full(
                                (1, args.num_classes),
                                1.0 / args.num_classes,
                                device=probs_u_w_glob.device,
                                dtype=probs_u_w_glob.dtype,
                            )
                        else:
                            prior_u = torch.as_tensor(
                                global_class_dist,
                                device=probs_u_w_glob.device,
                                dtype=probs_u_w_glob.dtype,
                            ).unsqueeze(0)
                        sets_u = pseudo_label_global > prior_u
                        low_sizes = sets_u.sum(dim=1)[low_mask]
                        gt_hits = (
                            sets_u[low_mask]
                            .gather(1, targets_u_groundtruth[low_mask].unsqueeze(1))
                            .squeeze(1)
                        )
                        for size, hit in zip(low_sizes.tolist(), gt_hits.tolist()):
                            set_size_hist[int(size)] += 1
                            set_size_hit_hist[int(size)] += int(hit)
                        num_gt_in_set_total += int(gt_hits.sum().item())
                        num_low_total += int(low_mask.sum().item())

                    # 统计本地模型与全局模型在有标签/无标签数据上的分类命中数（复用当步前向结果）
                    model_counts[0] += (logits_x.argmax(dim=1) == targets_x).sum()
                    model_counts[1] += targets_x.numel()
                    model_counts[2] += (
                        logits_u_w.argmax(dim=1) == targets_u_groundtruth
                    ).sum()
                    model_counts[3] += targets_u_groundtruth.numel()
                    model_counts[4] += (logits_x_glob.argmax(dim=1) == targets_x).sum()
                    model_counts[5] += targets_x.numel()
                    model_counts[6] += (
                        logits_u_w_glob.argmax(dim=1) == targets_u_groundtruth
                    ).sum()
                    model_counts[7] += targets_u_groundtruth.numel()

                t_stat_end = time.perf_counter()
                t_stats += t_stat_end - t_stat_start

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                t_i6 = time.perf_counter()
                t_bwd_opt += t_i6 - t_stat_end

            # 输出最后一个本地 epoch 的伪标签指标
            if local_epoch + 1 == args.local_epochs:
                pseudo_client_acc = (
                    num_pseudo_corrects / num_pseudo_total if num_pseudo_total else 0.0
                )
                u_client_valid = (
                    num_u_valid / num_pseudo_total if num_pseudo_total else 0.0
                )

        model_counts = model_counts.tolist()

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
            num_high_corrects,
            model_counts,
        ]
        total_time = time.perf_counter() - train_start
        timing_stats = {
            "t_data": t_data,
            "t_fwd_local": t_fwd_local,
            "t_fwd_glob": t_fwd_glob,
            "t_loss_fixmatch": t_loss_fixmatch,
            "t_loss_icpl": t_loss_icpl,
            "t_stats": t_stats,
            "t_bwd_opt": t_bwd_opt,
            "total_time": total_time,
        }
        return (
            {
                name: value.detach().cpu().clone()
                for name, value in self.local_model.state_dict().items()
            },
            pseudo_status,
            local_class_counts.cpu().numpy(),
            timing_stats,
        )

    @torch.no_grad()
    def model_eval(self, args, labeled_dataset, unlabeled_dataset):
        """本地模型与冻结全局模型(local_G)在本地数据上的分类准确率计数。

        训练全部结束后仅调用一次；batch 只是本次评估遍历的分批机制。
        返回 8 元素列表（4 组 × [命中, 总数] × 有标签/无标签）：
        [0:2] 本地模型有标签、[2:4] 本地模型无标签、
        [4:6] 全局模型有标签、[6:8] 全局模型无标签。
        """
        self.local_model.eval()
        counts = torch.zeros(8, device=self.device)
        loader = DataLoader(labeled_dataset, args.batch_size_local_labeled_fixmatch)
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            _, logits = self.local_model(images)
            _, logits_glob = self.local_G(images)
            counts[0] += (logits.argmax(dim=1) == labels).sum()
            counts[1] += labels.numel()
            counts[4] += (logits_glob.argmax(dim=1) == labels).sum()
            counts[5] += labels.numel()
        loader = DataLoader(unlabeled_dataset, args.batch_size_local_labeled_fixmatch)
        for images_u_w, _, labels in loader:
            # 无标签视图无独立基视图，评估用弱增强图（与掩码预测同源）。
            images_u_w = images_u_w.to(self.device)
            labels = labels.to(self.device)
            _, logits = self.local_model(images_u_w)
            _, logits_glob = self.local_G(images_u_w)
            counts[2] += (logits.argmax(dim=1) == labels).sum()
            counts[3] += labels.numel()
            counts[6] += (logits_glob.argmax(dim=1) == labels).sum()
            counts[7] += labels.numel()
        return counts.tolist()

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


class ClientTrainer:
    """Worker 进程内的客户端训练适配器，供公共进程池调用。"""

    def __init__(self, args, device):
        self.local = Local(args, device=device)

    def train(self, task, labeled_view, unlabeled_view):
        params, pseudo_status, class_counts, timing_stats = self.local.fixmatch_train(
            task.args,
            labeled_view,
            unlabeled_view,
            task.global_params,
            global_class_dist=task.global_class_dist,
        )
        return {
            "params": params,
            "pseudo_status": pseudo_status,
            "class_counts": class_counts,
            "elapsed_seconds": timing_stats["total_time"],
            "candidate_stats": timing_stats,
        }


def fixmatch(alpha, args=None, global_cls=Global, method="proxyfl"):
    """执行完整的 ProxyFL：数据划分、并行本地训练、聚合、评估和保存。"""

    # ==================== 初始化参数和日志 ====================
    if args is None:
        args = args_parser()
    # ==================== 创建训练集和测试集 ====================
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

    elif (
        args.dataset == "CIFAR100"
    ):  # training:50k; testing:10k; for training, each class includes 500 images
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
        logger.error(
            "不支持的数据集 %s，请从以下选项中选择：CIFAR10、CIFAR100、CINIC10 或 SVHN。",
            args.dataset,
        )
        sys.exit(1)

    args.method = method

    # ==================== 注册实验运行（唯一目录 + SQLite 索引） ====================
    run = create_run(args)
    setup_logging(run.log_file, level=args.log_level)
    logger.info("运行 ID：%s，结果目录：%s", run.run_id, run.dir)

    log_args(args)

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
            seed=args.seed,
        )
        list_client2indices_unlabeled = clients_indices(
            list_label2indices=list_label2indices_unlabeled,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            non_iid_alpha=alpha,
            seed=args.seed,
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
    client_gpus = parse_worker_gpus(args, require_server_gpu_in_clients=True)
    args.gpu_id = args.server_gpu
    global_model = global_cls(args)
    # 在创建共享数据集之前设置共享策略，避免预加载阶段产生大量文件描述符。
    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(data_local_training)
    worker_pool = ClientWorkerPool(
        client_gpus, args, shared_dataset, trainer_cls=ClientTrainer
    )

    total_clients = list(range(args.num_clients))

    fedavg_acc = []
    fedavg_pseudo_acc = []
    fedavg_num_valid = []
    fedavg_valid_ratio = []

    current_global_dist = global_model.update_global_distribution()

    # ==================== 联邦学习主循环 ====================
    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
    for r in progress:
        logger.info("========== 第 %d 轮 ==========", r)
        t_round_start = time.perf_counter()
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
        t_tasks_start = time.perf_counter()
        results = worker_pool.run_round(tasks)
        t_clients_done = time.perf_counter()
        list_dicts_local_params = [result.params for result in results]
        list_nums_local_data = [result.num_samples for result in results]

        for result in results:
            pseudo_status = result.pseudo_status
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]
            list_local_class_counts.append(result.class_counts)

        # ==================== 伪标签与标签集统计（拼接为一条 DEBUG） ====================
        sum_total = sum_valid = sum_low = sum_gt_in_set = 0
        sum_high_corrects = 0
        sum_hist = None
        sum_hit_hist = None
        model_counts_total = [0] * 8
        lines = []

        def format_eval_pair(counts, offset):
            """模型分类段的半边：有标签与无标签两个准确率。"""
            parts = []
            for offset_base, name in ((offset, "有标签"), (offset + 2, "无标签")):
                hit, total = int(counts[offset_base]), int(counts[offset_base + 1])
                parts.append(
                    f"{name} {hit / total:.1%}({hit}/{total})" if total else f"{name} —"
                )
            return f"[{', '.join(parts)}]"

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

        def format_acc(hist, hit_hist):
            """准确率段的逐桶命中率；空集与零命中的桶不显示。"""
            parts = [
                f"{size}: {hit_hist[size] / count:.1%}({hit_hist[size]}/{count})"
                for size, count in enumerate(hist)
                if count and size and hit_hist[size]
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
            num_high_corrects = status[9]
            model_counts = status[10]
            for idx, value in enumerate(model_counts):
                model_counts_total[idx] += value
            sum_total += num_total
            sum_valid += num_valid
            sum_low += num_low
            sum_gt_in_set += num_gt_in_set
            sum_high_corrects += num_high_corrects
            if sum_hist is None or sum_hit_hist is None:
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
                set_part = format_counts(hist, num_low)
                acc_part = format_acc(hist, hit_hist)
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
                f"客户端 {result.client_id}：模型分类 {format_eval_pair(model_counts, 0)}｜"
                f"全局模型分类 {format_eval_pair(model_counts, 4)}"
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
        total_set_part = (
            format_counts(sum_hist, sum_low)
            if sum_hist is not None and sum_low
            else ("无低置信样本" if not sum_low else "空")
        )
        total_acc_part = (
            format_acc(sum_hist, sum_hit_hist)
            if sum_hist is not None and sum_low
            else "—"
        )
        total_hit_part = (
            f"{sum_gt_in_set / sum_low:.1%}({sum_gt_in_set}/{sum_low})"
            if sum_low
            else "—"
        )
        lines.append(
            f"第 {r} 轮合计：高置信 {total_valid_part}｜"
            f"高置信伪标签准确率 {total_high_acc_part}｜"
            f"低置信标签集大小 {total_set_part}｜真实标签∈标签集 {total_acc_part}｜整体 {total_hit_part}"
        )
        lines.append(
            f"第 {r} 轮合计：模型分类 {format_eval_pair(model_counts_total, 0)}｜"
            f"全局模型分类 {format_eval_pair(model_counts_total, 4)}"
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

        # 根据客户端统计更新全局类别分布，并聚合模型参数。
        current_global_dist = global_model.update_global_distribution(
            current_global_dist, list_local_class_counts
        )
        t_dist_done = time.perf_counter()

        fedavg_params, all_classifier_weights, t_fedavg, t_update_gpt = (
            global_model.initialize_for_model_fusion(
                list_dicts_local_params, list_nums_local_data
            )
        )
        t_fusion_done = time.perf_counter()

        # 评估聚合后的全局模型。
        global_acc = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test
        )
        fedavg_acc.append(global_acc)
        t_eval_done = time.perf_counter()

        t_round_total = t_eval_done - t_round_start
        t_client_wall = t_clients_done - t_tasks_start
        client_times = [
            result.elapsed_seconds
            for result in results
            if hasattr(result, "elapsed_seconds")
        ]
        avg_client_time = float(np.mean(client_times)) if client_times else 0.0
        max_client_time = float(np.max(client_times)) if client_times else 0.0

        client_breakdowns = [
            result.candidate_stats
            for result in results
            if getattr(result, "candidate_stats", None) is not None
        ]
        if client_breakdowns:
            avg_data = float(np.mean([b["t_data"] for b in client_breakdowns]))
            avg_fwd_loc = float(np.mean([b["t_fwd_local"] for b in client_breakdowns]))
            avg_fwd_glob = float(np.mean([b["t_fwd_glob"] for b in client_breakdowns]))
            avg_loss_fm = float(np.mean([b["t_loss_fixmatch"] for b in client_breakdowns]))
            avg_loss_icpl = float(np.mean([b["t_loss_icpl"] for b in client_breakdowns]))
            avg_bwd_opt = float(np.mean([b["t_bwd_opt"] for b in client_breakdowns]))
            avg_stats = float(np.mean([b["t_stats"] for b in client_breakdowns]))
        else:
            avg_data = avg_fwd_loc = avg_fwd_glob = avg_loss_fm = avg_loss_icpl = avg_bwd_opt = avg_stats = 0.0

        logger.info(
            "第 %d 轮全局模型精度：%.2f%% (本轮总耗时: %.2fs)",
            r,
            global_acc * 100,
            t_round_total,
        )
        logger.info(
            "\n" + "=" * 65 + "\n"
            "【第 %d 轮耗时细粒度定位】总耗时: %.2fs\n"
            "  1. 客户端并行训练 (WorkerPool等待): %.2fs\n"
            "     ├─ 单客户端耗时: 平均 %.2fs，最慢 %.2fs\n"
            "     └─ 单客户端各阶段平均耗时:\n"
            "        ├─ 数据读取与GPU传输 (DataLoader): %.2fs (占比 %.1f%%)\n"
            "        ├─ 本地模型前向 (local_model):    %.2fs (占比 %.1f%%)\n"
            "        ├─ 全局模型前向 (local_G):        %.2fs (占比 %.1f%%)\n"
            "        ├─ FixMatch 损失 (Lx+Lu):         %.2fs (占比 %.1f%%)\n"
            "        ├─ ICPL 投影与对比损失 (Lc):      %.2fs (占比 %.1f%%)\n"
            "        ├─ 反向传播与优化器 (backward+step): %.2fs (占比 %.1f%%)\n"
            "        └─ 统计指标累加计算:               %.2fs (占比 %.1f%%)\n"
            "  2. 服务端处理与融合: %.2fs\n"
            "     ├─ 全局类别分布更新: %.2fs\n"
            "     ├─ FedAvg 参数加权平均: %.2fs\n"
            "     └─ update_GPT (100轮代理优化): %.2fs\n"
            "  3. 全局测试集评估 (fedavg_eval): %.2fs\n"
            + "=" * 65,
            r,
            t_round_total,
            t_client_wall,
            avg_client_time,
            max_client_time,
            avg_data,
            (avg_data / avg_client_time * 100) if avg_client_time else 0.0,
            avg_fwd_loc,
            (avg_fwd_loc / avg_client_time * 100) if avg_client_time else 0.0,
            avg_fwd_glob,
            (avg_fwd_glob / avg_client_time * 100) if avg_client_time else 0.0,
            avg_loss_fm,
            (avg_loss_fm / avg_client_time * 100) if avg_client_time else 0.0,
            avg_loss_icpl,
            (avg_loss_icpl / avg_client_time * 100) if avg_client_time else 0.0,
            avg_bwd_opt,
            (avg_bwd_opt / avg_client_time * 100) if avg_client_time else 0.0,
            avg_stats,
            (avg_stats / avg_client_time * 100) if avg_client_time else 0.0,
            (t_fusion_done - t_clients_done),
            (t_dist_done - t_clients_done),
            t_fedavg,
            t_update_gpt,
            (t_eval_done - t_fusion_done),
        )

        progress.set_postfix(acc=f"{global_acc:.2%}")

        # ==================== 保存模型和训练指标 ====================
        result_dir_spec = run.checkpoint_dir

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
            logger.info("第 %d 轮模型与代理已保存", r)

        result_file = run.dir / "metrics.csv"
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc) + 1))
        acc_num_pseudo_label_csv_df = pd.DataFrame(
            {"acc": fedavg_acc}, index=acc_num_pseudo_label_csv_index
        )
        # 保存文件
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding="utf8")

        result_pseudo_file = run.dir / "pseudo_metrics.csv"
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

    # 所有轮次完成后，向 Worker 发送退出信号并回收进程。
    worker_pool.close()
    run.finish(
        best_acc=max(fedavg_acc) if fedavg_acc else None, num_rounds=args.num_rounds
    )


if __name__ == "__main__":
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    run_main(lambda: fixmatch(args.alpha, args))
