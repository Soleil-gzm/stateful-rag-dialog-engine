"""
统一日志配置。

用法（仅在应用入口调用一次）：
    from .utils.logger import setup_logging
    setup_logging(level="INFO")

其他模块照常：
    import logging
    logger = logging.getLogger(__name__)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    also_console: bool = True,
) -> Path:
    """
    初始化 root logger：
      - 写入 <log_dir>/run_YYYYMMDD_HHMMSS.log
      - 可选同时输出到 stderr（默认开，避免污染 stdout 的 JSON 协议）
      - 幂等：重复调用会清掉旧 handler，不会重复输出

    :param log_dir: 日志目录，默认 <项目根>/Log
    :param level: 日志级别，DEBUG/INFO/WARNING/ERROR
    :param also_console: 是否同时打到 stderr
    :return: 本次运行的日志文件路径
    """
    # 默认放到项目根的 Log 目录
    # __file__ = <项目根>/stateful_rag_dialog_engine/utils/logger.py
    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "Log"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"

    root = logging.getLogger()
    # 幂等：先清掉旧 handler（比如热重载、测试重复调用）
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level.upper())

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if also_console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    return log_file