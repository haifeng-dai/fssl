"""FedMatch (labels-at-client) 的 PyTorch 复现。

实现论文的三个关键组成：参数分解 ``theta = sigma + psi``、监督/无监督
损失的 disjoint learning，以及由服务端基于模型表征选择的 helper-client
一致性和投票伪标签。它不使用项目中的 PLN 或原型损失。
"""

import copy
import logging
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.func import functional_call
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import test_proto as base
from options import args_parser
from utils.client_pool import ClientTask, ClientWorkerPool, parse_worker_gpus, preload_shared_dataset, run_main
from utils.logging_setup import log_args, setup_logging
from utils.run_registry import create_run

logger = logging.getLogger(__name__)


def _clone_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _combine(sigma, psi):
    """只相加可训练参数；BN 计数器等 buffer 保持 σ 的值。"""
    return {
        name: value + psi[name] if name in psi else value
        for name, value in sigma.items()
    }


class FedMatchLocal:
    def __init__(self, args, device):
        self.device = device
        self.model = base.build_model(args).to(device)
        self.parameter_names = set(dict(self.model.named_parameters()))

    def _state(self, sigma, psi, *, sigma_grad, psi_grad):
        state = {}
        for name, value in sigma.items():
            value = value.to(self.device)
            if name in self.parameter_names:
                sig = value.detach().clone().requires_grad_(sigma_grad)
                local = psi[name].to(self.device).detach().clone().requires_grad_(psi_grad)
                state[name] = sig + local
            else:
                state[name] = value.detach().clone()
        return state

    def _helper_logits(self, inputs, sigma, helpers):
        predictions = []
        with torch.no_grad():
            for helper_psi in helpers or []:
                helper_state = _combine(sigma, helper_psi)
                helper_state = {k: v.to(self.device) for k, v in helper_state.items()}
                _, logits = functional_call(self.model, helper_state, (inputs,))
                predictions.append(logits)
        return predictions

    def train(self, task, labeled_dataset, unlabeled_dataset):
        args = task.args
        start = time.perf_counter()
        labeled_loader = DataLoader(
            labeled_dataset, sampler=RandomSampler(labeled_dataset),
            batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True,
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True,
        )
        if len(labeled_loader) == 0 or len(unlabeled_loader) == 0:
            raise ValueError("FedMatch 客户端需要至少一个完整的有标签和无标签批次")

        sigma = {name: value.to(self.device) for name, value in task.global_params.items()}
        psi = {name: value.to(self.device) for name, value in task.fedmatch_psi.items()}
        # σ 仅由 Ls 更新，ψ 仅由 Lu+正则更新；这是论文所称 disjoint learning。
        sigma_vars = {name: value.detach().clone().requires_grad_() for name, value in sigma.items() if name in self.parameter_names}
        psi_vars = {name: value.detach().clone().requires_grad_() for name, value in psi.items()}
        sigma_opt = SGD(sigma_vars.values(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)
        psi_opt = SGD(psi_vars.values(), lr=args.lr_local_training, momentum=0.9)
        helpers = task.fedmatch_helpers or []
        local_steps = max(1, len(unlabeled_dataset) // args.batch_size_local_labeled_fixmatch)
        accepted = 0

        for _ in range(args.local_epochs):
            x_iter, u_iter = iter(labeled_loader), iter(unlabeled_loader)
            for _ in range(local_steps):
                try:
                    inputs_x, targets_x = next(x_iter)
                except StopIteration:
                    x_iter = iter(labeled_loader); inputs_x, targets_x = next(x_iter)
                try:
                    inputs_u_w, inputs_u_s, _ = next(u_iter)
                except StopIteration:
                    u_iter = iter(unlabeled_loader); inputs_u_w, inputs_u_s, _ = next(u_iter)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
                inputs_u_w, inputs_u_s = inputs_u_w.to(self.device), inputs_u_s.to(self.device)

                # Ls: ψ 视为常数，因而只更新共享 σ。
                supervised_state = {
                    name: (sigma_vars[name] + psi_vars[name].detach()) if name in self.parameter_names else sigma[name]
                    for name in sigma
                }
                _, logits_x = functional_call(self.model, supervised_state, (inputs_x,))
                sigma_opt.zero_grad()
                F.cross_entropy(logits_x, targets_x).backward()
                sigma_opt.step()

                # Lu: σ 视为常数，因而只更新个体 ψ。
                unsup_state = {
                    name: (sigma_vars[name].detach() + psi_vars[name]) if name in self.parameter_names else sigma[name]
                    for name in sigma
                }
                _, logits_w = functional_call(self.model, unsup_state, (inputs_u_w,))
                _, logits_s = functional_call(self.model, unsup_state, (inputs_u_s,))
                probs_w = torch.softmax(logits_w.detach() / args.T, dim=-1)
                confidence, local_labels = probs_w.max(dim=-1)
                keep = confidence.ge(args.threshold)
                accepted += int(keep.sum())
                helper_logits = self._helper_logits(inputs_u_w, sigma, helpers)

                agreement = torch.zeros((), device=self.device)
                if helper_logits and task.round > 1 and keep.any():
                    local_log_probs = F.log_softmax(logits_w[keep], dim=-1)
                    for logits_h in helper_logits:
                        agreement = agreement + F.kl_div(
                            local_log_probs, torch.softmax(logits_h[keep] / args.T, dim=-1),
                            reduction="batchmean",
                        ) / len(helper_logits)

                votes = F.one_hot(local_labels, args.num_classes).float()
                for logits_h in helper_logits:
                    votes += F.one_hot(logits_h.argmax(dim=-1), args.num_classes).float()
                voted_labels = votes.argmax(dim=-1)
                pseudo_loss = (
                    F.cross_entropy(logits_s, voted_labels, reduction="none") * keep.float()
                ).mean()
                l1 = sum(value.abs().sum() for value in psi_vars.values())
                l2 = sum((sigma_vars[name].detach() - value).square().sum() for name, value in psi_vars.items())
                unsup_loss = (
                    args.fedmatch_lambda_i * agreement
                    + args.fedmatch_lambda_a * pseudo_loss
                    + args.fedmatch_lambda_l1 * l1
                    + args.fedmatch_lambda_l2 * l2
                )
                psi_opt.zero_grad()
                unsup_loss.backward()
                psi_opt.step()

        sigma_out = _clone_state({**sigma, **sigma_vars})
        psi_out = _clone_state(psi_vars)
        # 对齐原始 FedMatch 的 hard-threshold sparsify：低幅值 ψ 不参与上传、
        # helper 恢复或下一轮的有效模型。
        for value in psi_out.values():
            value.masked_fill_(value.abs() < args.fedmatch_l1_threshold, 0)
        psi_nonzero = sum(value.count_nonzero().item() for value in psi_out.values())
        psi_total = sum(value.numel() for value in psi_out.values())
        return {
            "params": _combine(sigma_out, psi_out),
            "fedmatch_psi": psi_out,
            "elapsed_seconds": time.perf_counter() - start,
            "candidate_stats": {
                "fedmatch_accepted": accepted,
                "fedmatch_helpers": len(helpers),
                "fedmatch_psi_nonzero_ratio": psi_nonzero / max(psi_total, 1),
            },
        }


class ClientTrainer:
    def __init__(self, args, device):
        self.local = FedMatchLocal(args, device)

    def train(self, task, labeled_view, unlabeled_view):
        return self.local.train(task, labeled_view, unlabeled_view)


def uniform_average(states):
    result = _clone_state(states[0])
    for name, value in result.items():
        if torch.is_floating_point(value):
            result[name] = sum(state[name] for state in states) / len(states)
    return result


@torch.no_grad()
def model_signature(server, full_params, probe):
    server.model.load_state_dict(full_params)
    server.model.eval()
    _, logits = server.model(probe.to(server.device))
    return logits.flatten().cpu()


def select_helpers(client_id, signatures, client_psis, num_helpers):
    if client_id not in signatures or len(signatures) < 2:
        return []
    target = signatures[client_id]
    peers = [(torch.dist(target, signature).item(), other) for other, signature in signatures.items() if other != client_id]
    return [copy.deepcopy(client_psis[other]) for _, other in sorted(peers)[:num_helpers]]


def fedmatch(args):
    args.method = "test_fedmatch"
    train_dataset, test_dataset = base.load_datasets(args)
    run = create_run(args)
    setup_logging(run.log_file, level=args.log_level)
    logger.info("运行 ID：%s，结果目录：%s", run.run_id, run.dir)
    log_args(args)
    rng = np.random.RandomState(args.seed)
    label_indices = base.classify_label(train_dataset, args.num_classes)
    labeled_indices, unlabeled_indices = base.partition_train(label_indices, args.num_labeled)
    common = {"num_classes": args.num_classes, "num_clients": args.num_clients}
    partition = base.clients_indices_homo if args.alpha == 0 else base.clients_indices
    partition_args = common if args.alpha == 0 else {**common, "non_iid_alpha": args.alpha, "seed": args.seed}
    client_labeled = partition(list_label2indices=labeled_indices, **partition_args)
    client_unlabeled = partition(list_label2indices=unlabeled_indices, **partition_args)
    base.show_clients_data_distribution(train_dataset, client_labeled, client_unlabeled, args.num_classes)
    client_unlabeled_eval = copy.deepcopy(client_unlabeled)
    for cid in range(args.num_clients):
        client_unlabeled[cid] = np.concatenate((client_unlabeled[cid], client_labeled[cid]))

    worker_gpus = parse_worker_gpus(args)
    args.gpu_id = args.server_gpu
    server = base.Global(args)
    sigma = server.download_params()
    # FedMatch 的 ψ 初始为 σ 的比例，而不是“先置零再缩放”。
    # 否则 ψ 恒为 0，初始有效模型会错误地退化成仅有 σ。
    psi = {
        name: value.detach().cpu().clone() * args.fedmatch_psi_factor
        for name, value in sigma.items()
        if name in dict(server.model.named_parameters())
    }
    probe = torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(args.seed))
    signatures, client_psis = {}, {}
    mp.set_sharing_strategy("file_system")
    shared_dataset = preload_shared_dataset(train_dataset)
    pool = ClientWorkerPool(worker_gpus, args, shared_dataset, ClientTrainer, log_file=str(run.log_file))
    rows, all_clients = [], list(range(args.num_clients))
    progress = tqdm(range(1, args.num_rounds + 1), desc=args.method)
    try:
        for round_id in progress:
            online = rng.choice(all_clients, args.num_online_clients, replace=False)
            # 原实现按 h_interval 通信轮才请求一次相似 helper。
            helpers_enabled = round_id > 1 and round_id % args.fedmatch_helper_interval == 0
            tasks = [ClientTask(
                round=round_id, client_id=int(cid),
                labeled_indices=list(np.asarray(client_labeled[cid]).tolist()),
                unlabeled_indices=list(np.asarray(client_unlabeled[cid]).tolist()),
                global_params=sigma, fedmatch_psi=psi,
                fedmatch_helpers=(select_helpers(int(cid), signatures, client_psis, args.fedmatch_num_helpers) if helpers_enabled else []),
                args=copy.deepcopy(args),
            ) for cid in online]
            results = pool.run_round(tasks)
            sigma = uniform_average([{name: result.params[name] - result.fedmatch_psi[name] if name in result.fedmatch_psi else result.params[name] for name in result.params} for result in results])
            psi = uniform_average([result.fedmatch_psi for result in results])
            full_params = _combine(sigma, psi)
            for result in results:
                client_psis[result.client_id] = result.fedmatch_psi
                signatures[result.client_id] = model_signature(server, result.params, probe)

            transform = base.evaluation_transform(args.dataset)
            selected_l = np.concatenate([np.asarray(client_labeled[c]) for c in online])
            selected_u = np.concatenate([np.asarray(client_unlabeled_eval[c]) for c in online])
            test_metrics = server.evaluate(full_params, test_dataset, args.batch_size_test)
            labeled_metrics = server.evaluate(full_params, base.IndexedEvalDataset(train_dataset, selected_l, transform), args.batch_size_test)
            u_metrics = server.evaluate(full_params, base.IndexedEvalDataset(train_dataset, selected_u, transform), args.batch_size_test)
            accepted = sum(r.candidate_stats["fedmatch_accepted"] for r in results)
            helpers = np.mean([r.candidate_stats["fedmatch_helpers"] for r in results])
            psi_ratio = np.mean([r.candidate_stats["fedmatch_psi_nonzero_ratio"] for r in results])
            row = {"round": round_id, **test_metrics, **base.rename_dataset_metrics(labeled_metrics, "labeled"), **base.rename_dataset_metrics(u_metrics, "u_pool"), "fedmatch_accepted": accepted, "fedmatch_mean_helpers": helpers, "fedmatch_psi_nonzero_ratio": psi_ratio}
            rows.append(row)
            pd.DataFrame(rows).set_index("round").to_csv(run.dir / "metrics.csv", encoding="utf8")
            logger.info("第 %d 轮 FedMatch：global_test=%.2f%%，labeled=%.2f%%，u_pool=%.2f%%，高置信样本=%d，helper=%.1f，ψ 非零率=%.2f%%", round_id, 100 * test_metrics["global_test_acc"], 100 * labeled_metrics["global_test_acc"], 100 * u_metrics["global_test_acc"], accepted, helpers, 100 * psi_ratio)
            progress.set_postfix(acc=f"{test_metrics['global_test_acc']:.2%}")
            if round_id in (1, args.num_rounds) or (round_id % 50 == 0 and round_id > .8 * args.num_rounds):
                torch.save(full_params, run.checkpoint_dir / f"fedmatch_params_round_{round_id}.pth")
    finally:
        pool.close()
    run.finish(best_acc=max(row["global_test_acc"] for row in rows) if rows else None, num_rounds=args.num_rounds)


if __name__ == "__main__":
    args = args_parser()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    run_main(lambda: fedmatch(args))
