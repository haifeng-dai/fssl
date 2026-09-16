"""GLPL-FixMatch：全局与本地模型联合生成伪标签。"""

import random

import numpy as np
import torch

import fixmatch_gpl as gpl
from options import args_parser
from utils.client_pool import run_main


class Local(gpl.Local):
    """任一模型高置信即接收，伪标签取两者中置信度更高者。"""

    pseudo_label_mode = "combined"


class ClientTrainer:
    """为公共客户端进程池构造 GLPL 本地训练器。"""

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
    """执行 GLPL-FixMatch。"""
    return gpl.fedavg_fixmatch(
        alpha,
        args,
        trainer_cls=ClientTrainer,
        method="fixmatch_glpl",
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
