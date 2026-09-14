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
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler
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
from Model.resnet import ResNet_PC
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
        # 高置信子集的伪标签正确数（仅统计 mask 选中的样本）
        num_high_corrects = 0
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

                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
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

                # 纯 FixMatch 目标：不包含原型、锚点或其他特征约束。
                loss = Lx + args.lambda_u * Lu

                # 统计最后一个 epoch 的伪标签质量
                if local_epoch + 1 == args.local_epochs:
                    num_pseudo_corrects += (
                        torch.eq(targets_u.cpu(), targets_u_groundtruth.cpu())
                        .sum()
                        .item()
                    )
                    num_pseudo_total += len(targets_u)
                    num_u_valid += int(mask.sum().item())
                    if mask.sum() > 0:
                        valid_idx = mask.bool()
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


def fedavg_fixmatch(alpha, args=None):
    """执行纯净的联邦半监督学习 (FedAvg + FixMatch)。"""
    if args is None:
        args = args_parser()
    args.method = "fixmatch_lpl"

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
    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
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
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        list_dicts_local_params = [result.params for result in results]
        list_nums_local_data = [result.num_samples for result in results]

        # ==================== 高置信统计（逐客户端 + 合计，一条 DEBUG） ====================
        sum_total = sum_valid = sum_high_corrects = 0
        high_lines = []
        for result in results:
            pseudo_status = result.pseudo_status
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]
            num_total, num_valid, num_high_corrects = (
                pseudo_status[0],
                pseudo_status[2],
                pseudo_status[5],
            )
            sum_total += num_total
            sum_valid += num_valid
            sum_high_corrects += num_high_corrects
            ratio_part = (
                f"{num_valid / num_total:.1%}({num_valid}/{num_total})"
                if num_total
                else "0.0%(0/0)"
            )
            acc_part = (
                f"{num_high_corrects / num_valid:.1%}({num_high_corrects}/{num_valid})"
                if num_valid
                else "—"
            )
            high_lines.append(
                f"客户端 {result.client_id}：比例 {ratio_part}｜准确率 {acc_part}"
            )
        total_ratio_part = (
            f"{sum_valid / sum_total:.1%}({sum_valid}/{sum_total})"
            if sum_total
            else "0.0%(0/0)"
        )
        total_acc_part = (
            f"{sum_high_corrects / sum_valid:.1%}({sum_high_corrects}/{sum_valid})"
            if sum_valid
            else "—"
        )
        high_lines.append(
            f"第 {r} 轮合计：比例 {total_ratio_part}｜准确率 {total_acc_part}"
        )
        logger.debug("第 %d 轮高置信统计：\n%s", r, "\n".join(high_lines))

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
