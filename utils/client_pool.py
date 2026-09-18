"""多 GPU 客户端训练进程池的公共实现，供各联邦算法复用。

各算法文件只需提供一个可序列化的 Trainer 适配类（在 Worker 进程内构造、
持有本地训练器），进程池的调度、结果收集与退出清理统一在这里处理。
"""

import copy
import dataclasses
import gc
import logging
import os
import queue
import random
import signal
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

from Dataset.dataset import (
    Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch,
    SharedImageDataset,
)
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

_ACTIVE_POOL = None


def terminate_active_pool():
    """立即终止主进程当前的客户端进程池。"""
    if _ACTIVE_POOL is not None:
        _ACTIVE_POOL.terminate()


@dataclasses.dataclass
class ClientTask:
    """主进程下发给客户端 Worker 的单次训练任务。"""

    round: int
    client_id: int
    labeled_indices: list
    unlabeled_indices: list
    global_params: dict
    global_prototypes: object | None = None
    global_prototype_mask: object | None = None
    global_pln_params: dict | None = None
    global_prototype_max_radius: object | None = None
    global_prototype_max_cosine_distance: object | None = None
    global_anchors: object | None = None
    global_class_dist: object | None = None
    # FedMatch: globally broadcast parameter components and the sparse
    # unsupervised components of the selected helper clients.
    fedmatch_psi: dict | None = None
    fedmatch_helpers: list | None = None
    args: object = None


@dataclasses.dataclass
class ClientResult:
    """客户端 Worker 训练结束返回的结果对象。"""

    ok: bool
    client_id: int
    gpu_id: int
    params: dict | None = None
    num_samples: int = 0
    pseudo_status: list | None = None
    prototypes: object | None = None
    prototype_counts: object | None = None
    pln_params: dict | None = None
    pln_num_samples: int = 0
    fedmatch_psi: dict | None = None
    class_counts: object | None = None
    eval_counts: object | None = None
    candidate_stats: object | None = None
    elapsed_seconds: float = 0.0
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


def client_worker(
    gpu_id, args, shared_dataset, task_queue, result_queue, trainer_cls, log_file=None
):
    """Worker 进程：在固定 GPU 上循环领取任务并执行客户端训练。"""
    try:
        # spawn 创建的子进程需要配置 logger，保证客户端训练日志正常输出
        if log_file:
            setup_logging(log_file, level=args.log_level)

        # 子进程不响应 Ctrl-C，退出统一由主进程控制。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # spawn 创建的子进程需要重新设置 Tensor 共享策略。
        mp.set_sharing_strategy("file_system")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        trainer = trainer_cls(args, device=device)

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

                fields = trainer.train(task, labeled_view, unlabeled_view)
                result_queue.put(
                    ClientResult(
                        ok=True,
                        client_id=task.client_id,
                        gpu_id=gpu_id,
                        num_samples=len(task.labeled_indices) * labeled_view.repeat
                        + len(task.unlabeled_indices),
                        **fields,
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                result_queue.put(
                    ClientResult(
                        ok=False,
                        client_id=task.client_id,
                        gpu_id=gpu_id,
                        error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                )
                # 任务失败后直接退出 Worker，避免继续训练占用显存或放大故障。
                return
            finally:
                # 每个任务结束（含失败）后清空本任务残留的显存缓存。
                gc.collect()
                torch.cuda.empty_cache()
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            ClientResult(
                ok=False,
                client_id=-1,
                gpu_id=gpu_id,
                error=f"Worker initialization failed ({type(exc).__name__}: {exc}):\n{traceback.format_exc()}",
            )
        )


class ClientWorkerPool:
    """管理客户端训练进程池，支持多卡/单卡多进程调度。"""

    def __init__(self, gpu_ids, args, shared_dataset, trainer_cls, log_file=None):
        global _ACTIVE_POOL
        if _ACTIVE_POOL is not None and not _ACTIVE_POOL.closed:
            raise RuntimeError("一个主进程只能创建一个 ClientWorkerPool")

        mp.set_sharing_strategy("file_system")
        self.gpu_ids = list(gpu_ids)
        self.args = copy.deepcopy(args)
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.closed = False
        _ACTIVE_POOL = self

        for gpu_id in self.gpu_ids:
            process = self.ctx.Process(
                target=client_worker,
                args=(
                    gpu_id,
                    copy.deepcopy(args),
                    shared_dataset,
                    self.task_queue,
                    self.result_queue,
                    trainer_cls,
                    log_file,
                ),
                name=f"fssl-client-gpu-{gpu_id}",
            )
            process.start()
            self.processes.append(process)

    def run_round(self, tasks):
        """提交一轮客户端任务并收集按 client_id 排序后的结果。"""
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
        except BaseException:
            round_id = tasks[0].round if tasks else -1
            # 失败详情（含 traceback）写入运行日志文件，便于事后排查；
            # 随后保持原有抛出行为。
            logger.exception("第 %d 轮收集客户端结果时失败", round_id)
            self.terminate()
            raise

        return sorted(results, key=lambda result: result.client_id)

    def close(self):
        """发送退出信号并回收所有 Worker。"""
        global _ACTIVE_POOL
        if self.closed:
            return
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join()
        self.task_queue.close()
        self.result_queue.close()
        self.closed = True
        if _ACTIVE_POOL is self:
            _ACTIVE_POOL = None

    def terminate(self):
        """立即杀死并回收所有 Worker。"""
        global _ACTIVE_POOL
        if self.closed:
            return
        self.closed = True
        for process in self.processes:
            if process.is_alive():
                process.kill()
        for process in self.processes:
            process.join(timeout=1)
        for work_queue in (self.task_queue, self.result_queue):
            work_queue.cancel_join_thread()
            work_queue.close()
        if _ACTIVE_POOL is self:
            _ACTIVE_POOL = None


def run_main(func):
    """主进程入口：异常或 Ctrl-C 时先回收 Worker，再硬退出。"""

    try:
        func()
    except SystemExit:
        terminate_active_pool()
        raise
    except KeyboardInterrupt:
        terminate_active_pool()
        os._exit(130)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        terminate_active_pool()
        os._exit(1)


def parse_worker_gpus(args, require_server_gpu_in_clients=False):
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
        raise RuntimeError("多 GPU 训练需要可用的 CUDA 环境")

    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"GPU 编号 {invalid} 不可用；当前可见 GPU 数量为 {device_count}"
        )

    args.server_gpu = args.server_gpu if args.server_gpu is not None else gpu_ids[0]
    if require_server_gpu_in_clients:
        if args.server_gpu not in gpu_ids:
            raise ValueError("--server_gpu 必须包含在 --client_gpus 中")
    elif args.server_gpu < 0 or args.server_gpu >= device_count:
        raise ValueError(f"--server_gpu {args.server_gpu} 超出可用 GPU 范围")

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
