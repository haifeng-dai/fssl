import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler

import test_proto as base
from Model.factory import build_model
from options import args_parser
from utils.client_pool import run_main


class FineRegulator(nn.Module):
    """论文 F-reg：128 宽的 MLP，把强增强预测映射为 [0, 1] 样本权重。"""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, probabilities):
        return self.net(probabilities).squeeze(1)


class FedDureLocal:
    def __init__(self, args, device=None):
        self.device = device or torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.local_model = build_model(args).to(self.device)
        # C-reg 与局部模型同构，但仅作为客户端内的粗粒度调节代理。
        self.c_reg = build_model(args).to(self.device)
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
        del eval_labeled_dataset, global_prototypes, global_prototype_mask
        start = time.perf_counter()
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.optimizer.state.clear()
        self.c_reg.load_state_dict(global_params)
        self.c_reg.train()
        f_reg = FineRegulator(args.num_classes).to(self.device)
        f_reg_optimizer = Adam(f_reg.parameters(), lr=args.feddure_meta_lr)
        c_reg_optimizer = SGD(
            self.c_reg.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )
        labeled_loader = DataLoader(
            labeled_dataset,
            sampler=RandomSampler(labeled_dataset),
            batch_size=min(
                args.batch_size_local_labeled_fixmatch, len(labeled_dataset)
            ),
            drop_last=False,
        )
        unlabeled_loader = DataLoader(
            u_pool_dataset,
            sampler=RandomSampler(u_pool_dataset),
            batch_size=min(
                args.batch_size_local_labeled_fixmatch * args.mu,
                len(u_pool_dataset),
            ),
            drop_last=False,
        )
        # 一个本地 epoch 恰遍历一遍 u-pool；有标签 loader 不足时循环取样。
        steps = len(unlabeled_loader)
        if len(labeled_loader) == 0 or steps == 0:
            return (
                self._params(),
                torch.zeros(args.num_classes, self.local_model.dim),
                torch.zeros(args.num_classes),
                time.perf_counter() - start,
            )
        x_iter, u_iter = iter(labeled_loader), iter(unlabeled_loader)
        for step in range(args.local_epochs * steps):
            try:
                images_x, targets_x = next(x_iter)
            except StopIteration:
                x_iter = iter(labeled_loader)
                images_x, targets_x = next(x_iter)
            try:
                images_u_w, images_u_s, _ = next(u_iter)
            except StopIteration:
                u_iter = iter(unlabeled_loader)
                images_u_w, images_u_s, _ = next(u_iter)
            images_x, targets_x = images_x.to(self.device), targets_x.to(self.device)
            images_u_w, images_u_s = (
                images_u_w.to(self.device),
                images_u_s.to(self.device),
            )
            with torch.no_grad():
                _, weak_logits = self.local_model(images_u_w)
                pseudo_targets = weak_logits.argmax(dim=-1)

            # (1) 双层更新只虚拟更新 C-reg 的 phi；local_model 在此阶段固定。
            # F-reg 的输入是 C-reg 对强增强样本的预测，而非 weak-view 概率。
            self.c_reg.eval()
            _, c_logits_u = self.c_reg(images_u_s)
            meta_weights = f_reg(torch.softmax(c_logits_u.detach(), dim=-1))
            c_virtual_loss = (
                meta_weights
                * F.cross_entropy(c_logits_u, pseudo_targets, reduction="none")
            ).mean()
            named_parameters = tuple(self.c_reg.named_parameters())
            grads = torch.autograd.grad(
                c_virtual_loss,
                tuple(parameter for _, parameter in named_parameters),
                create_graph=True,
                allow_unused=True,
            )
            fast_params = {
                name: parameter - args.lr_local_training * grad
                if grad is not None
                else parameter
                for (name, parameter), grad in zip(named_parameters, grads)
            }
            _, meta_logits = functional_call(self.c_reg, fast_params, (images_x,))
            meta_loss = F.cross_entropy(meta_logits, targets_x)
            f_reg_optimizer.zero_grad()
            meta_loss.backward()
            f_reg_optimizer.step()
            # meta backward 也会在 C-reg 上留下梯度；它不是这一步的更新对象。
            c_reg_optimizer.zero_grad()

            # (2) 用更新后的 F-reg 正式训练 C-reg，并以真实标签改善量得到 d。
            self.c_reg.eval()
            with torch.no_grad():
                _, c_logits_x_before = self.c_reg(images_x)
                loss_before = F.cross_entropy(c_logits_x_before, targets_x)
            self.c_reg.train()
            _, c_logits_u = self.c_reg(images_u_s)
            with torch.no_grad():
                c_weights = f_reg(torch.softmax(c_logits_u.detach(), dim=-1))
            c_loss = (
                c_weights
                * F.cross_entropy(c_logits_u, pseudo_targets, reduction="none")
            ).mean()
            c_reg_optimizer.zero_grad()
            c_loss.backward()
            c_reg_optimizer.step()
            self.c_reg.eval()
            with torch.no_grad():
                _, c_logits_x_after = self.c_reg(images_x)
                d = (
                    loss_before - F.cross_entropy(c_logits_x_after, targets_x)
                ).detach()

            # (3) 真实局部模型：细粒度和粗粒度是两条相加的无监督梯度。
            self.local_model.train()
            _, logits_x = self.local_model(images_x)
            _, logits_u_s = self.local_model(images_u_s)
            per_sample = F.cross_entropy(logits_u_s, pseudo_targets, reduction="none")
            warmup = min(1.0, (step + 1) / max(1, args.feddure_warmup_steps))
            with torch.no_grad():
                sample_weights = f_reg(torch.softmax(logits_u_s.detach(), dim=-1))
            fine_loss = (sample_weights * per_sample).mean()
            coarse_loss = d * per_sample.mean()
            loss = F.cross_entropy(logits_x, targets_x) + args.lambda_u * warmup * (
                fine_loss + coarse_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return (
            self._params(),
            torch.zeros(args.num_classes, self.local_model.dim),
            torch.zeros(args.num_classes),
            time.perf_counter() - start,
        )

    def _params(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.local_model.state_dict().items()
        }


class FedDureTrainer:
    def __init__(self, args, device):
        self.local = FedDureLocal(args, device)

    def train(self, task, labeled_view, unlabeled_view):
        params, prototypes, counts, elapsed = self.local.train(
            task.args,
            labeled_view,
            unlabeled_view,
            None,
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


def run_experiment():
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Reuse the project's dataset split, worker pool and FedAvg implementation; this
    # trainer ignores all prototype pathways, so no PLN/prototype loss is involved.
    base.ClientTrainer = FedDureTrainer
    original = base.fedavg_fixmatch

    def run(alpha, supplied_args):
        supplied_args.lambda_proto = 0.0
        return original(alpha, supplied_args, method="test_feddure")

    run_main(lambda: run(args.alpha, args))


if __name__ == "__main__":
    run_experiment()
