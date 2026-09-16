"""纯监督 FedAvg：客户端本地训练只使用有标签样本。"""

import copy
import logging
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from Dataset.dataset import classify_label, partition_train, show_clients_data_distribution
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from options import args_parser
from test import IndexedEvalDataset, build_model, evaluation_transform, load_datasets
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


class LabeledOnlyTrainer:
    """单客户端纯监督训练器，不创建或读取无标签训练 batch。"""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = build_model(args).to(device)
        self.optimizer = SGD(
            self.model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def train(self, task, labeled_view, _unlabeled_view):
        self.model.load_state_dict(task.global_params)
        self.model.train()
        self.optimizer.state.clear()
        labeled_loader = DataLoader(
            labeled_view,
            sampler=RandomSampler(labeled_view),
            batch_size=self.args.batch_size_local_labeled_fixmatch,
            drop_last=False,
        )
        schedule_size = len(_unlabeled_view) or len(labeled_view)
        local_steps = int(
            schedule_size / self.args.batch_size_local_labeled_fixmatch
        )
        labeled_iter = iter(labeled_loader)
        for _ in range(self.args.local_epochs):
            for _ in range(local_steps):
                try:
                    images, labels = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    images, labels = next(labeled_iter)
                images, labels = images.to(self.device), labels.to(self.device)
                _, logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return {
            "params": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            }
        }


def fedavg_l(alpha, args=None):
    if args is None:
        args = args_parser()
    args.method = "fedavg_l"
    train_dataset, test_dataset = load_datasets(args)
    run = create_run(args)
    setup_logging(run.log_file, level=args.log_level)
    logger.info("运行 ID：%s，结果目录：%s", run.run_id, run.dir)
    log_args(args)

    random_state = np.random.RandomState(args.seed)
    label_indices = classify_label(train_dataset, args.num_classes)
    labeled_indices, unlabeled_indices = partition_train(label_indices, args.num_labeled)
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
        [[] for _ in range(args.num_clients)],
        args.num_classes,
    )

    worker_gpus = parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    server_model = build_model(args).to(
        torch.device(f"cuda:{args.server_gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    server_device = next(server_model.parameters()).device
    shared_dataset = preload_shared_dataset(train_dataset)
    worker_pool = ClientWorkerPool(
        worker_gpus,
        args,
        shared_dataset,
        trainer_cls=LabeledOnlyTrainer,
        log_file=str(run.log_file),
    )

    global_params = {
        name: value.detach().cpu().clone()
        for name, value in server_model.state_dict().items()
    }
    metrics = []
    all_clients = list(range(args.num_clients))
    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
    transform = evaluation_transform(args.dataset)
    for round_id in progress:
        online_clients = random_state.choice(
            all_clients, args.num_online_clients, replace=False
        )
        tasks = [
            ClientTask(
                round=round_id,
                client_id=int(client),
                labeled_indices=list(np.asarray(client_labeled[client]).tolist()),
                unlabeled_indices=list(
                    np.asarray(client_unlabeled[client]).tolist()
                ),
                global_params=global_params,
                args=copy.deepcopy(args),
            )
            for client in online_clients
        ]
        results = worker_pool.run_round(tasks)
        labeled_counts = {
            int(client): len(client_labeled[int(client)]) for client in online_clients
        }
        total = sum(labeled_counts[result.client_id] for result in results)
        global_params = {
            name: sum(
                result.params[name] * labeled_counts[result.client_id]
                for result in results
            )
            / total
            for name in results[0].params
            if torch.is_floating_point(results[0].params[name])
        }
        for name, value in results[0].params.items():
            if not torch.is_floating_point(value):
                global_params[name] = value.clone()
        server_model.load_state_dict(global_params)

        selected_indices = np.concatenate(
            [np.asarray(client_labeled[client]) for client in online_clients]
        )
        labeled_eval = IndexedEvalDataset(train_dataset, selected_indices, transform)
        server_model.eval()
        def accuracy(dataset):
            correct = total_count = 0
            with torch.no_grad():
                for images, labels in DataLoader(dataset, args.batch_size_test):
                    _, logits = server_model(images.to(server_device))
                    correct += (logits.argmax(1).cpu() == labels).sum().item()
                    total_count += labels.numel()
            return correct / total_count if total_count else 0.0

        row = {
            "round": round_id,
            "global_test_acc": accuracy(test_dataset),
            "labeled_global_model_acc": accuracy(labeled_eval),
        }
        metrics.append(row)
        import pandas as pd
        pd.DataFrame(metrics).set_index("round").to_csv(
            run.dir / "metrics.csv", encoding="utf8"
        )
        logger.info(
            "第 %d 轮准确率：test=%.2f%%，labeled=%.2f%%",
            round_id,
            row["global_test_acc"] * 100,
            row["labeled_global_model_acc"] * 100,
        )

    worker_pool.close()
    run.finish(
        best_acc=max(row["global_test_acc"] for row in metrics),
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
    run_main(lambda: fedavg_l(args.alpha, args))
