import logging


def setup_logging(log_file, level="INFO", force=True):
    """配置根 logger：本项目日志按指定级别写入运行日志文件。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s\n%(message)s",
        filename=str(log_file),
        force=force,
    )


def log_args(args, logger=None):
    """将算法名与全部运行参数记录为一条日志（各算法参数不同，逐行通用输出）。"""
    items = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
    (logger or logging.getLogger(__name__)).info(
        "算法：%s\n%s", getattr(args, "method", "未知"), items
    )

