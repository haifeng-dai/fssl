"""ProxyFL 均匀先验消融：始终使用 1/C，不更新动态类别先验。"""

import random

import numpy as np
import torch

from options import args_parser
from proxyfl import Global, fixmatch
from utils.client_pool import run_main


class UniformPriorGlobal(Global):
    """保持 ProxyFL 其他服务端逻辑不变，仅固定类别先验为均匀分布。"""

    def update_global_distribution(
        self,
        current_global_dist=None,
        list_class_counts=None,
    ):
        del current_global_dist, list_class_counts
        return np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float64)


def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    fixmatch(
        args.alpha,
        args,
        global_cls=UniformPriorGlobal,
        method="proxyfl_uniform",
    )


if __name__ == "__main__":
    run_main(main)
