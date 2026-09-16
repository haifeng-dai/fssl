"""按实验组绘制测试精度曲线（均值±标准差），支持跨重复聚合。

Usage:
    python plot/plot_acc.py --list                        # 按实验组查看运行
    python plot/plot_acc.py --method proxyfl --alpha 0.1  # 最新组的均值±方差带
    python plot/plot_acc.py --method proxyfl --latest 3   # 最新 3 个组
    python plot/plot_acc.py --method proxyfl --individual # 每条运行一条线（旧行为）

概念：
  组 (group)  = 算法 + 参数组合，由 args 快照自动计算指纹（seed/repeat 等不参与，
                因此换 seed 的重复自动归入同组；历史运行无需补标）。
  次 (repeat) = 组内第几次重复，训练时 --repeat N 指定（默认 1）；同一组同一次的
                多次运行（如崩溃重跑）只取 status=done 且 created_at 最新的一条。
  默认对最新的组跨 repeat 聚合：均值粗线 + 标准差阴影 + 各重复细线。

所有筛选参数 (--method, --dataset, --alpha, --num_clients, --lr, --seed,
--status) 直接映射 runs 表列。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_registry import query

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / "figs"

# CLI 参数与 runs 表列的对应关系
FILTER_ARGS = {
    "method": str,
    "dataset": str,
    "alpha": float,
    "num_clients": int,
    "lr": float,
    "seed": int,
    "status": str,
}

# 计算组指纹时排除的参数：run 相关信息与随重复变化的量
FINGERPRINT_EXCLUDE = {"run_id", "hostname", "git_commit", "seed", "repeat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy curves of experiment groups (mean ± std across repeats)."
    )
    for name, arg_type in FILTER_ARGS.items():
        parser.add_argument(f"--{name}", type=arg_type, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="按实验组列出匹配的运行，而不画图。",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=1,
        metavar="N",
        help="画最新 N 个实验组（默认 1）。",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="旧行为：每条 done 运行单独一条线，不聚合。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the PNG output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    if not args.list and not any(getattr(args, n) is not None for n in FILTER_ARGS):
        parser.error("Provide at least one filter (e.g. --method) or use --list.")
    return args


def collect_filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: getattr(args, name)
        for name in FILTER_ARGS
        if getattr(args, name) is not None
    }


def config_fingerprint(params: dict) -> str:
    """对参数快照计算组指纹；同指纹 = 同一实验组（seed/repeat 不同的重复）。"""
    payload = {k: v for k, v in params.items() if k not in FINGERPRINT_EXCLUDE}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(text.encode("utf8")).hexdigest()[:8]


def group_runs(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    """按指纹与 repeat 编号分组：{指纹: {repeat: [该次的所有运行，乱序]}}。"""
    groups: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        params = row["params"]
        fingerprint = config_fingerprint(params)
        repeat = int(params.get("repeat", 1))
        groups.setdefault(fingerprint, {}).setdefault(repeat, []).append(row)
    return groups


def latest_run(runs: list[dict]) -> dict | None:
    """同组同次的多个运行里 created_at 最新的一条（不限状态）。"""
    return max(runs, key=lambda run: run["created_at"]) if runs else None


def latest_done(runs: list[dict]) -> dict | None:
    """同组同次的多个运行里，取 status=done 且 created_at 最新的一条。"""
    done = [run for run in runs if run["status"] == "done"]
    if not done:
        return None
    return max(done, key=lambda run: run["created_at"])


def group_latest_time(group: dict[int, list[dict]]) -> str:
    """组的排序键：组内所有运行（含 running/failed）的最新创建时间。"""
    all_runs = [run for runs in group.values() for run in runs]
    return max(run["created_at"] for run in all_runs)


def selected_runs(group: dict[int, list[dict]]) -> list[dict]:
    """参与聚合的运行：每个 repeat 取最新 done；无 done 的 repeat 跳过。"""
    runs = []
    for repeat in sorted(group):
        run = latest_done(group[repeat])
        if run is not None:
            runs.append(run)
    return runs


def group_label(runs: list[dict]) -> str:
    """聚合曲线的图例：配置摘要 + 重复数。"""
    params = runs[0]["params"]
    parts = [str(params.get("method") or "?")]
    if params.get("alpha") is not None:
        parts.append(f"α={params['alpha']:g}")
    if params.get("num_clients") is not None:
        parts.append(f"K={params['num_clients']}")
    if params.get("lr_local_training") is not None:
        parts.append(f"lr={params['lr_local_training']:g}")
    parts.append(f"(n={len(runs)})")
    return " ".join(parts)


def run_label(row: dict[str, object]) -> str:
    """个体线的图例（--individual 模式）。"""
    parts = [str(row.get("method") or "?")]
    if row.get("alpha") is not None:
        parts.append(f"α={row['alpha']:g}")
    if row.get("num_clients") is not None:
        parts.append(f"K={row['num_clients']}")
    if row.get("lr") is not None:
        parts.append(f"lr={row['lr']:g}")
    if row.get("seed") is not None:
        parts.append(f"seed={row['seed']}")
    return " ".join(parts)


def load_run_accuracy(run_path: str) -> tuple[list[float], list[float]]:
    """Read the round/accuracy columns from a run's metrics.csv."""
    csv_path = Path(run_path) / "metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found in run dir: {csv_path}")

    with open(csv_path, newline="", encoding="utf8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        accuracy_column = (
            "global_test_acc"
            if header is not None and "global_test_acc" in header
            else "acc"
        )
        if header is None or accuracy_column not in header:
            raise ValueError(
                f"CSV must contain a 'global_test_acc' or 'acc' column: {csv_path}"
            )
        round_idx = header.index("round") if "round" in header else 0
        acc_idx = header.index(accuracy_column)

        rounds: list[float] = []
        accs: list[float] = []
        for row in reader:
            if not row:
                continue
            rounds.append(float(row[round_idx]))
            accs.append(float(row[acc_idx]) * 100)
    if not rounds:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    return rounds, accs


def aggregate_repeats(runs: list[dict]):
    """跨重复聚合：轮号取交集，返回 (rounds, mean, std, 各重复曲线)。"""
    curves = []
    for run in runs:
        rounds, accs = load_run_accuracy(run["path"])
        curves.append(dict(zip(rounds, accs)))
    common_rounds = set(curves[0])
    for curve in curves[1:]:
        common_rounds &= set(curve)
    common_rounds = sorted(common_rounds)
    if len(common_rounds) < max(len(curve) for curve in curves):
        print(f"注意：各重复轮数不一致，仅对 {len(common_rounds)} 个公共轮取均值/方差")
    matrix = np.asarray([[curve[rd] for rd in common_rounds] for curve in curves])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return common_rounds, mean, std, curves


def list_runs(filters: dict[str, object]) -> None:
    rows = query(**filters) if filters else query()
    groups = group_runs(rows)
    if not groups:
        print("No matching runs.")
        return
    ordered = sorted(
        groups.items(), key=lambda item: group_latest_time(item[1]), reverse=True
    )
    print(
        f"{'group':<10} {'method':<18} {'α':<6} {'K':<4} {'lr':<6} "
        f"{'rounds':<8} {'repeats':<22} {'latest':<20} best_acc(mean±std)"
    )
    for fingerprint, group in ordered:
        all_runs = [run for runs in group.values() for run in runs]
        params = all_runs[0]["params"]
        repeat_statuses = []
        for repeat in sorted(group):
            latest = latest_run(group[repeat])
            status = latest["status"] if latest is not None else "none"
            repeat_statuses.append(f"{repeat}:{status}")
        repeat_desc = " ".join(repeat_statuses)
        runs = selected_runs(group)
        best_accs = [run["best_acc"] for run in runs if run["best_acc"] is not None]
        if best_accs:
            best_desc = (
                f"{np.mean(best_accs):.4f}±{np.std(best_accs, ddof=1):.4f}"
                if len(best_accs) > 1
                else f"{best_accs[0]:.4f}"
            )
        else:
            best_desc = "-"
        latest_time = group_latest_time(group)
        print(
            f"{fingerprint:<10} {(params.get('method') or '-')!s:<18} "
            f"{params.get('alpha')!s:<6} {params.get('num_clients')!s:<4} "
            f"{params.get('lr_local_training')!s:<6} {params.get('num_rounds')!s:<8} "
            f"{repeat_desc:<22} {latest_time:<20} {best_desc}"
        )


def plot_runs(
    filters: dict[str, object],
    output_dir: Path,
    latest: int = 1,
    individual: bool = False,
) -> Path:
    rows = query(**filters)
    if not rows:
        raise ValueError(
            "No runs match the given filters. "
            "Run `python plot/plot_acc.py --list` to see what is available."
        )

    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    plotted = 0
    selected = []

    if individual:
        # 旧行为：每条 done 运行单独一条线。
        done_rows = [row for row in rows if row["status"] == "done"]
        suffix = len(done_rows) > 1
        for row in done_rows:
            try:
                rounds, accs = load_run_accuracy(row["path"])
            except (FileNotFoundError, ValueError) as error:
                print(f"Skipping {row['run_id']}: {error}")
                continue
            label = f"{run_label(row)} (max acc={np.nanmax(accs):.2f}%)"
            if suffix:
                label += f" [{row['run_id'][:15]}]"
            axis.plot(
                rounds, accs, marker="o", markersize=3, linewidth=1.8, label=label
            )
            plotted += 1
    else:
        # 组模式：取最新 N 个有 done 运行的组，每组画均值±标准差。
        groups = group_runs(rows)
        plottable = [
            (fingerprint, group)
            for fingerprint, group in groups.items()
            if selected_runs(group)
        ]
        plottable.sort(key=lambda item: group_latest_time(item[1]), reverse=True)
        selected = plottable[: max(latest, 1)]
        if not selected:
            raise ValueError("No matching run has a usable metrics.csv.")
        for group_index, (fingerprint, group) in enumerate(selected):
            runs = selected_runs(group)
            try:
                rounds, mean, std, curves = aggregate_repeats(runs)
            except (FileNotFoundError, ValueError) as error:
                print(f"Skipping group {fingerprint}: {error}")
                continue
            color = f"C{group_index % 10}"
            label = f"{group_label(runs)} (max acc={np.nanmax(mean):.2f}%)"
            if len(runs) > 1:
                axis.fill_between(
                    rounds, mean - std, mean + std, color=color, alpha=0.18
                )
            for curve in curves:
                per_rounds = sorted(curve)
                axis.plot(
                    per_rounds,
                    [curve[rd] for rd in per_rounds],
                    color=color,
                    alpha=0.25,
                    linewidth=0.9,
                )
            axis.plot(
                rounds,
                mean,
                color=color,
                marker="o",
                markersize=3,
                linewidth=2.0,
                label=label,
            )
            plotted += 1

    if plotted == 0:
        raise ValueError("No matching run has a usable metrics.csv.")

    axis.set_xlabel("Communication Round")
    axis.set_ylabel("Test Accuracy (%)")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.set_ylim(bottom=0)
    if plotted > 0:
        axis.legend(loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    name_parts = [str(filters.get("method") or "runs")]
    for key, value in filters.items():
        name_parts.append(f"{key[0]}{value}" if key != "alpha" else f"a{value}")
    if not individual:
        name_parts.append(f"grp{selected[0][0][:6]}")
        if len(selected) > 1:
            name_parts.append(f"{len(selected)}groups")
    output_path = output_dir / ("_".join(name_parts) + "_acc.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    filters = collect_filters(args)
    try:
        if args.list:
            list_runs(filters)
            return
        output_path = plot_runs(
            filters, args.output_dir, latest=args.latest, individual=args.individual
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved accuracy curve to {output_path}")


if __name__ == "__main__":
    main()
