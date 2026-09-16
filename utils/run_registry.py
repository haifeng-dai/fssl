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
    except subprocess.CalledProcessError, FileNotFoundError, OSError:
        # 非 git 仓库或 git 不可用时跳过，不影响实验运行
        return None


def _connect():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
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


if __name__ == "__main__":
    import pprint

    pprint.pprint(query())
