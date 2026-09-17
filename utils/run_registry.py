"""实验运行注册表：唯一 run 目录 + SQLite 索引 + args 快照。

每次训练调用 create_run(args) 得到一个 Run 对象：
  - 目录 results/runs/{日期}/{时分秒-短随机ID}/，exist_ok=False 保证绝不互相覆盖；
  - 目录内写入 args.json（完整参数快照，含 git commit），数据库损坏时可重建索引；
  - results/runs.db 记录全量参数（JSON 列）与常用筛选列，支持并行写入（WAL 模式）。

训练正常结束调用 run.finish(best_acc=...)，异常时调用 run.fail(error=...)。
"""

import json
import logging
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("./results")
DB_PATH = RESULTS_ROOT / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    method       TEXT,
    dataset      TEXT,
    alpha        REAL,
    num_clients  INTEGER,
    lr           REAL,
    seed         INTEGER,
    num_rounds   INTEGER,
    status       TEXT NOT NULL DEFAULT 'running',
    best_acc     REAL,
    hostname     TEXT,
    git_commit   TEXT,
    git_diff     TEXT,
    code         TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    params       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_method_alpha ON runs (method, dataset, alpha);
"""


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # 非 git 仓库或 git 不可用时跳过，不影响实验运行
        return None


def _git_diff():
    """获取工作区未提交的代码 diff，非 git 仓库或无修改时返回 None。"""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return diff if diff else None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _get_entry_code(args) -> tuple[str | None, str | None]:
    """获取当前正在运行的算法入口源码文本以及文件名。"""
    # 1. 优先读取当前直接执行的主脚本（如 python sage.py ... 时的 sage.py）
    if sys.argv:
        try:
            entry_path = Path(sys.argv[0]).resolve()
            if entry_path.is_file() and entry_path.suffix == ".py":
                return entry_path.read_text(encoding="utf8"), entry_path.name
        except Exception:
            pass

    # 2. 备用尝试：根据 args.method 查找对应的脚本文件（如 proxyfl.py）
    method = getattr(args, "method", None)
    if method:
        try:
            method_path = Path(f"{method}.py").resolve()
            if method_path.is_file():
                return method_path.read_text(encoding="utf8"), method_path.name
        except Exception:
            pass

    return None, None


def _migrate_schema(conn: sqlite3.Connection):
    """自动向现有 runs 表增量补充 code 和 git_diff 字段（兼容历史数据库）。"""
    cursor = conn.execute("PRAGMA table_info(runs)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "code" not in existing_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN code TEXT")
    if "git_diff" not in existing_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN git_diff TEXT")


def _connect():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    return conn


class Run:
    """一次实验运行的句柄。"""

    def __init__(self, run_id: str, dir_path: Path):
        self.run_id = run_id
        self.dir = dir_path
        self.log_file = dir_path / "train.log"
        self.checkpoint_dir = dir_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        # 记录句柄创建时刻，用于 finish 时统计总耗时
        self._start = time.monotonic()

    def _update(self, **fields):
        fields["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        sets = ", ".join(f"{k} = ?" for k in fields)
        with _connect() as conn:
            conn.execute(
                f"UPDATE runs SET {sets} WHERE run_id = ?",
                (*fields.values(), self.run_id),
            )

    def finish(self, best_acc, num_rounds):
        self._update(status="done", best_acc=best_acc)
        total_seconds = time.monotonic() - self._start
        logger.info(
            "运行 %s 结束，最佳精度=%s，总耗时=%.1f 秒，平均每轮=%.1f 秒",
            self.run_id,
            f"{best_acc:.2%}" if best_acc is not None else "—",
            total_seconds,
            total_seconds / num_rounds,
        )

    def fail(self, error="", exc_info=False):
        self._update(status="failed", best_acc=None)
        # exc_info=True 时在日志中附带当前异常的完整 traceback
        logger.error("运行 %s 失败：%s", self.run_id, error, exc_info=exc_info)


def create_run(args) -> Run:
    """创建唯一运行目录并登记进数据库。返回 Run 句柄。

    请在数据集分支设置完 args（num_rounds 等）之后调用，保证快照是最终参数。
    """
    # 日期分层便于长期管理；时分秒前缀保证目录按时间排序，随机后缀保证同秒并行启动不碰撞
    timestamp = time.strftime("%H%M%S")
    date_dir = RESULTS_ROOT / "runs" / time.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = f"{timestamp}-{secrets.token_hex(3)}"
        run_dir = date_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue

    run = Run(run_id, run_dir)

    # 全量参数快照：目录自描述，数据库损坏时可据此重建索引
    snapshot = dict(vars(args))
    snapshot["git_commit"] = _git_commit()
    snapshot["hostname"] = socket.gethostname()
    snapshot["run_id"] = run_id
    with open(run_dir / "args.json", "w", encoding="utf8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

    # 获取运行时的完整算法代码及当前未提交的 git diff 快照
    code_text, code_filename = _get_entry_code(args)
    git_diff_text = _git_diff()

    # 在运行目录备份运行时的算法源码文件
    if code_text is not None:
        save_name = code_filename or "code.py"
        with open(run_dir / save_name, "w", encoding="utf8") as f:
            f.write(code_text)

    # 在运行目录备份未提交的代码差异
    if git_diff_text:
        with open(run_dir / "git.patch", "w", encoding="utf8") as f:
            f.write(git_diff_text)

    row = {
        "run_id": run_id,
        "path": str(run_dir),
        "method": getattr(args, "method", None),
        "dataset": getattr(args, "dataset", None),
        "alpha": getattr(args, "alpha", None),
        "num_clients": getattr(args, "num_clients", None),
        "lr": getattr(args, "lr_local_training", None),
        "seed": getattr(args, "seed", None),
        "num_rounds": getattr(args, "num_rounds", None),
        "status": "running",
        "hostname": snapshot["hostname"],
        "git_commit": snapshot["git_commit"],
        "git_diff": git_diff_text,
        "code": code_text,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": json.dumps(snapshot, ensure_ascii=False, default=str),
    }
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with _connect() as conn:
        conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", tuple(row.values()))

    # 此时尚未 setup_logging，注册信息由入口文件在配置完成后记录的
    # "Run ID: ..., results dir: ..." 日志承担。
    return run


def query(**filters):
    """按列等值查询运行记录，返回 dict 列表（params 已反序列化）。"""
    if filters:
        conds = " AND ".join(f"{k} = ?" for k in filters)
        sql = f"SELECT * FROM runs WHERE {conds} ORDER BY created_at"
        params = tuple(filters.values())
    else:
        sql = "SELECT * FROM runs ORDER BY created_at"
        params = ()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params)]
    for r in rows:
        r["params"] = json.loads(r["params"])
    return rows


def get_run_code(run_id_or_prefix: str) -> tuple[str | None, str | None]:
    """根据 run_id 或其前缀查询该次运行时的算法代码及 git diff。

    返回: (code_text, git_diff_text)
    """
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT code, git_diff FROM runs WHERE run_id LIKE ? ORDER BY created_at DESC LIMIT 1",
            (f"{run_id_or_prefix}%",),
        )
        row = cursor.fetchone()
        if row:
            return row[0], row[1]
    return None, None


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="查看实验运行或导出历史代码快照")
    parser.add_argument("--code", type=str, metavar="RUN_ID", help="打印指定 run_id 运行时的算法源码")
    parser.add_argument("--diff", type=str, metavar="RUN_ID", help="打印指定 run_id 运行时的未提交 git diff")
    cli_args = parser.parse_args()

    if cli_args.code:
        code, _ = get_run_code(cli_args.code)
        if code:
            print(code)
        else:
            print(f"未找到 run_id 前缀为 '{cli_args.code}' 的代码快照。")
    elif cli_args.diff:
        _, diff = get_run_code(cli_args.diff)
        if diff:
            print(diff)
        else:
            print(f"run_id '{cli_args.diff}' 无未提交 git diff 或未找到记录。")
    else:
        pprint.pprint(query())
