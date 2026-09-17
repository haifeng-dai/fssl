import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import fixmatch_gpl as gpl
from options import args_parser
from utils.client_pool import run_main

# SAGE CDSC 敏感度固定全局参数：满足 delta_c = 0.05 时 lambda_dynamic = 0.5
KAPPA = math.log(2.0) / 0.05


class Local(gpl.Local):
    """SAGE 客户端训练器：基于本地模型与全局模型的置信度差异进行动态软修正 (CDSC)。"""

    def train(
        self,
        args,
        data_client_labeled,
        data_client_unlabeled,
        global_params,
    ):
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
        num_high_corrects = 0
        pseudo_client_acc = 0.0
        u_client_valid = 0.0

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
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1,
                )

                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                # 1. 有标签监督交叉熵损失
                Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

                # 2. 全局与本地双视角伪标签预测
                with torch.no_grad():
                    _, logits_u_w_global = self.global_model(inputs_u_w)
                    pseudo_label_global = torch.softmax(
                        logits_u_w_global / args.T, dim=-1
                    )
                    max_probs_global, targets_u_global = torch.max(
                        pseudo_label_global, dim=-1
                    )

                pseudo_label_local = torch.softmax(
                    logits_u_w.detach() / args.T, dim=-1
                )
                max_probs_local, targets_u_local = torch.max(
                    pseudo_label_local, dim=-1
                )

                targets_u_local_one_hot = F.one_hot(
                    targets_u_local, args.num_classes
                ).float()
                targets_u_global_one_hot = F.one_hot(
                    targets_u_global, args.num_classes
                ).float()

                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                # 3. 置信度差异与动态权重 (Confidence-Driven Soft Correction)
                delta_c = torch.abs(max_probs_local - max_probs_global) + 1e-6
                delta_c = torch.clamp(delta_c, min=1e-6, max=1.0)

                lambda_dynamic = torch.exp(-KAPPA * delta_c)
                lambda_dynamic = torch.clamp(lambda_dynamic, min=1e-6, max=1.0)

                # 4. 软伪标签动态修正
                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot
                    + (1.0 - lambda_dynamic).unsqueeze(1)
                    * targets_u_global_one_hot,
                    targets_u_global_one_hot,
                )

                mask_valid = torch.max(mask_local, mask_global)

                # 5. KL 散度无标签一致性损失
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u_prob = final_targets_u + 1e-10
                Lu = (
                    F.kl_div(
                        logits_u_s_probs.log(),
                        final_targets_u_prob,
                        reduction="none",
                    ).sum(-1)
                    * mask_valid
                ).mean()

                loss = Lx + args.lambda_u * Lu

                # 统计最后一个 epoch 的伪标签质量
                if local_epoch + 1 == args.local_epochs:
                    targets_u = final_targets_u.argmax(dim=-1)
                    num_pseudo_corrects += (
                        torch.eq(targets_u.cpu(), targets_u_groundtruth.cpu())
                        .sum()
                        .item()
                    )
                    num_pseudo_total += len(targets_u)
                    num_u_valid += int(mask_valid.sum().item())
                    if mask_valid.sum() > 0:
                        valid_idx = mask_valid.bool()
                        num_high_corrects += (
                            torch.eq(
                                targets_u[valid_idx].cpu(),
                                targets_u_groundtruth[valid_idx].cpu(),
                            )
                            .sum()
                            .item()
                        )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if local_epoch + 1 == args.local_epochs:
                pseudo_client_acc = (
                    num_pseudo_corrects / num_pseudo_total
                    if num_pseudo_total
                    else 0.0
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
            num_high_corrects,
        ]
        return (
            {
                name: value.detach().cpu().clone()
                for name, value in self.local_model.state_dict().items()
            },
            pseudo_status,
            time.perf_counter() - train_start,
        )


class ClientTrainer(gpl.ClientTrainer):
    """Worker 进程内的客户端训练适配器，供公共进程池调用。"""

    def __init__(self, args, device):
        self.local = Local(args, device=device)

    def train(self, task, labeled_view, unlabeled_view):
        params, pseudo_status, elapsed_seconds = self.local.train(
            task.args,
            labeled_view,
            unlabeled_view,
            task.global_params,
        )
        return {
            "params": params,
            "pseudo_status": pseudo_status,
            "elapsed_seconds": elapsed_seconds,
        }


def fedavg_sage(alpha, args=None):
    """执行 SAGE 联邦半监督学习。"""
    return gpl.fedavg_fixmatch(
        alpha,
        args,
        trainer_cls=ClientTrainer,
        method="sage",
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

    run_main(lambda: fedavg_sage(args.alpha, args))
