"""
pipeline.py —— 编排 corpus 构建全流程。

对应原代码:
    xlsx2txt-condition.py + txt_merge.py + save_db.py 的手工三连跑

流程:
    1. read_excel    -> ExcelData
    2. write_all     -> 逐模块写 txt (output_*.txt)
    3. merge_all     -> expand + append
    4. build_all     -> 为每个 txt 建 FAISS

用法:
    python -m stateful_rag_dialog_engine.corpus.pipeline --config config/corpus.yaml
    python -m stateful_rag_dialog_engine.corpus.pipeline --config config/corpus.yaml --skip-faiss
    python -m stateful_rag_dialog_engine.corpus.pipeline --config config/corpus.yaml --clean
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import CorpusConfig, load_config
from .excel_reader import read_excel
from .txt_writer import write_all, WriteResult
from .merger import merge_all, MergeReport
from .faiss_builder import build_all as build_faiss_all, FaissReport

logger = logging.getLogger(__name__)


# ==================================================================
# 总报告
# ==================================================================

@dataclass
class PipelineReport:
    config_path: Path
    excel_path: Path

    excel_ok: bool = False
    write_results: list[WriteResult] = field(default_factory=list)
    merge_report: MergeReport | None = None
    faiss_report: FaissReport | None = None

    total_seconds: float = 0.0
    stopped_at: str = ""              # 出错时记录阶段名

    # ── 汇总统计 ──

    @property
    def txt_written(self) -> int:
        return sum(len(r.files_written) for r in self.write_results)

    @property
    def txt_rows(self) -> int:
        return sum(r.rows_total for r in self.write_results)

    @property
    def faiss_built(self) -> int:
        return len(self.faiss_report.built) if self.faiss_report else 0

    @property
    def has_warnings(self) -> bool:
        if any(r.warnings for r in self.write_results):
            return True
        if self.merge_report and self.merge_report.warnings:
            return True
        if self.faiss_report and self.faiss_report.failed:
            return True
        return False


# ==================================================================
# 各步骤封装
# ==================================================================

def _step_clean(config: CorpusConfig) -> None:
    """清理上一次构建产物（保留 manifest 和 cache 可选）。"""
    for d in (config.output.txt_dir, config.output.faiss_dir):
        if d.exists():
            logger.info(f"[clean] 删除 {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _step_read(config: CorpusConfig):
    logger.info("=" * 70)
    logger.info("[1/4] 读取 Excel")
    logger.info("=" * 70)
    return read_excel(config.input.excel, config)


def _step_write(excel_data, config: CorpusConfig) -> list[WriteResult]:
    logger.info("=" * 70)
    logger.info("[2/4] 写入 txt")
    logger.info("=" * 70)
    return write_all(excel_data, config)


def _step_merge(config: CorpusConfig) -> MergeReport:
    logger.info("=" * 70)
    logger.info("[3/4] 合并（expand + append）")
    logger.info("=" * 70)
    return merge_all(config)


def _step_faiss(config: CorpusConfig) -> FaissReport:
    logger.info("=" * 70)
    logger.info("[4/4] 构建 FAISS")
    logger.info("=" * 70)
    return build_faiss_all(config)


# ==================================================================
# 主流程
# ==================================================================

def run_pipeline(
    config: CorpusConfig,
    *,
    skip_faiss: bool = False,
    clean: bool = False,
) -> PipelineReport:
    report = PipelineReport(
        config_path=config._config_path,
        excel_path=config.input.excel,
    )
    t_start = time.time()

    # 可选：先清理
    if clean:
        _step_clean(config)

    # ── Step 1: read ──
    try:
        excel_data = _step_read(config)
        report.excel_ok = True
    except Exception as e:
        logger.exception(f"读取 Excel 失败: {e}")
        report.stopped_at = "read_excel"
        report.total_seconds = time.time() - t_start
        return report

    # ── Step 2: write ──
    try:
        report.write_results = _step_write(excel_data, config)
    except Exception as e:
        logger.exception(f"写入 txt 失败: {e}")
        report.stopped_at = "write_all"
        report.total_seconds = time.time() - t_start
        return report

    # ── Step 3: merge ──
    try:
        report.merge_report = _step_merge(config)
    except Exception as e:
        logger.exception(f"合并失败: {e}")
        report.stopped_at = "merge_all"
        report.total_seconds = time.time() - t_start
        return report

    # ── Step 4: faiss ──
    if skip_faiss:
        logger.info("=" * 70)
        logger.info("[4/4] 跳过 FAISS 构建 (--skip-faiss)")
        logger.info("=" * 70)
    else:
        try:
            report.faiss_report = _step_faiss(config)
        except Exception as e:
            logger.exception(f"构建 FAISS 失败: {e}")
            report.stopped_at = "build_all"
            report.total_seconds = time.time() - t_start
            return report

    report.total_seconds = time.time() - t_start
    return report


# ==================================================================
# 报告打印
# ==================================================================

def print_report(report: PipelineReport, config: CorpusConfig) -> None:
    print()
    print("=" * 78)
    print("构建报告")
    print("=" * 78)

    print(f"  配置文件 : {report.config_path}")
    print(f"  Excel    : {report.excel_path}")

    if not report.excel_ok:
        print()
        print("  [失败] 读取 Excel 阶段中止")
        print(f"  耗时   : {report.total_seconds:.1f}s")
        return

    # ── txt 写入 ──
    print()
    print("  [1/4] 写入 txt")
    if report.write_results:
        for r in report.write_results:
            flag = ""
            if r.warnings:
                flag = f"  [warn x{len(r.warnings)}]"
            print(
                f"    {r.module_id:<12s} | files={len(r.files_written):>3d} | "
                f"rows={r.rows_total:>4d}{flag}"
            )
        print(f"    合计: {report.txt_written} 文件, {report.txt_rows} 行")
    else:
        print("    (无)")

    # ── merge ──
    print()
    print("  [2/4] 合并")
    mr = report.merge_report
    if mr:
        print(f"    expanded : {len(mr.expanded)}")
        print(f"    appended : {len(mr.appended)}")
        print(f"    skipped  : {len(mr.skipped)}")
        if mr.warnings:
            print(f"    warnings : {len(mr.warnings)}")
            for w in mr.warnings:
                print(f"      [warn] {w}")
    else:
        print("    (未执行或失败)")

    # ── faiss ──
    print()
    print("  [3/4] FAISS")
    fr = report.faiss_report
    if fr:
        print(f"    built    : {len(fr.built)}")
        print(f"    failed   : {len(fr.failed)}")
        print(f"    questions: {fr.total_questions}")
        if fr.failed:
            for p, err in fr.failed:
                print(f"      [fail] {p.name}: {err}")
    else:
        print("    (跳过)")

    # ── 路径 ──
    print()
    print("  产出目录:")
    print(f"    txt    : {config.output.txt_dir}")
    print(f"    faiss  : {config.output.faiss_dir}")

    # ── 状态 ──
    print()
    if report.stopped_at:
        print(f"  [中止] 阶段: {report.stopped_at}")
    elif report.has_warnings:
        print(f"  [完成] 有警告，请查看上面 [warn] 行")
    else:
        print(f"  [完成] 无警告")
    print(f"  总耗时 : {report.total_seconds:.1f}s")
    print("=" * 78)


# ==================================================================
# CLI
# ==================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Corpus 构建流水线：Excel -> txt -> merge -> FAISS"
    )
    p.add_argument(
        "--config",
        required=True,
        help="corpus.yaml 路径",
    )
    p.add_argument(
        "--skip-faiss",
        action="store_true",
        help="只构建 txt 和 merge，跳过 FAISS（用于快速迭代话术）",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="构建前清空 txt_dir 和 faiss_dir",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s | %(message)s",
    )

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return 2

    logger.info(f"business = {config.business}")
    logger.info(f"txt_dir  = {config.output.txt_dir}")
    logger.info(f"faiss_dir= {config.output.faiss_dir}")

    report = run_pipeline(
        config,
        skip_faiss=args.skip_faiss,
        clean=args.clean,
    )
    print_report(report, config)

    # 退出码：有失败返回 1
    if report.stopped_at:
        return 1
    if report.faiss_report and report.faiss_report.failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())