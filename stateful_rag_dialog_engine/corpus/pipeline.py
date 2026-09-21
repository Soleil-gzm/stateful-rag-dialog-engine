"""
pipeline.py —— 编排 corpus 构建全流程。

对应原代码:
    xlsx2txt-condition.py + txt_merge.py + save_db.py 的手工三连跑

流程:
    1. read    -> 读 Excel 到内存
    2. write   -> 逐模块写 txt (output_*.txt)
    3. merge   -> expand + append
    4. faiss   -> 为每个 txt 建 FAISS

用法:
    # 全跑
    python -m stateful_rag_dialog_engine.corpus.pipeline --config config/corpus.yaml

    # 只跑前 3 步（改话术快速迭代）
    python -m stateful_rag_dialog_engine.corpus.pipeline \
        --config config/corpus.yaml --only=read,write,merge

    # 只重建 FAISS（txt 没变，只换了 embedding 模型）
    python -m stateful_rag_dialog_engine.corpus.pipeline \
        --config config/corpus.yaml --only=faiss

    # 只跑 merge（手工调过 txt，只想重合并）
    python -m stateful_rag_dialog_engine.corpus.pipeline \
        --config config/corpus.yaml --only=merge

    # 清空 txt_dir + faiss_dir 再重建
    python -m stateful_rag_dialog_engine.corpus.pipeline \
        --config config/corpus.yaml --clean
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
from .excel_reader import ExcelData, read_excel
from .txt_writer import write_all, WriteResult
from .merger import merge_all, MergeReport
from .faiss_builder import build_all as build_faiss_all, FaissReport

logger = logging.getLogger(__name__)


# ==================================================================
# 阶段定义
# ==================================================================

VALID_STAGES: tuple[str, ...] = ("read", "write", "merge", "faiss")


def parse_stages(only_str: str | None, skip_faiss: bool) -> list[str]:
    """
    解析 --only / --skip-faiss，返回按全局顺序排列的阶段列表。

    规则:
        - --only 未提供: 默认全部阶段
        - --only 提供: 按用户给的集合，但按 VALID_STAGES 顺序重排
        - write 存在时自动补 read（内存依赖）
        - --skip-faiss 等价于 --only=read,write,merge（若 --only 未提供）

    异常:
        --only 里出现未知阶段名 -> ValueError
    """
    if only_str is not None and only_str.strip() != "":
        raw = [s.strip() for s in only_str.split(",") if s.strip()]
        invalid = [s for s in raw if s not in VALID_STAGES]
        if invalid:
            raise ValueError(
                f"--only 中有未知阶段: {invalid}; "
                f"可选: {list(VALID_STAGES)}"
            )
        requested = set(raw)
    else:
        requested = set(VALID_STAGES)
        if skip_faiss:
            requested.discard("faiss")

    # write 需要 read 在内存里；即使 user 只写 --only=write 也补上 read
    if "write" in requested:
        requested.add("read")

    # 按全局顺序重排
    return [s for s in VALID_STAGES if s in requested]


# ==================================================================
# 总报告
# ==================================================================

@dataclass
class PipelineReport:
    config_path: Path
    excel_path: Path
    stages: list[str] = field(default_factory=list)

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

    def ran(self, stage: str) -> bool:
        return stage in self.stages


# ==================================================================
# 各步骤封装
# ==================================================================

def _step_clean(config: CorpusConfig, stages: list[str]) -> None:
    """
    按阶段清理产物。

    规则:
        - 有 write 或 merge -> 清空 txt_dir
          （merge 原地更新 txt，通常也需要干净起点）
        - 有 faiss -> 清空 faiss_dir
        - 只跑 read（无 write/merge/faiss）-> 不删任何东西
    """
    if "write" in stages or "merge" in stages:
        d = config.output.txt_dir
        if d.exists():
            logger.info(f"[clean] 删除 {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    if "faiss" in stages:
        d = config.output.faiss_dir
        if d.exists():
            logger.info(f"[clean] 删除 {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _step_read(config: CorpusConfig) -> ExcelData:
    logger.info("=" * 70)
    logger.info("[read] 读取 Excel")
    logger.info("=" * 70)
    return read_excel(config.input.excel, config)


def _step_write(excel_data: ExcelData, config: CorpusConfig) -> list[WriteResult]:
    logger.info("=" * 70)
    logger.info("[write] 写入 txt")
    logger.info("=" * 70)
    return write_all(excel_data, config)


def _step_merge(config: CorpusConfig) -> MergeReport:
    logger.info("=" * 70)
    logger.info("[merge] 合并（expand + append）")
    logger.info("=" * 70)
    return merge_all(config)


def _step_faiss(config: CorpusConfig) -> FaissReport:
    logger.info("=" * 70)
    logger.info("[faiss] 构建 FAISS")
    logger.info("=" * 70)
    return build_faiss_all(config)


# ==================================================================
# 主流程
# ==================================================================

def run_pipeline(
    config: CorpusConfig,
    *,
    stages: list[str],
    clean: bool = False,
) -> PipelineReport:
    """
    按 stages 顺序执行各阶段。

    stages 由 parse_stages 生成，保证顺序为 read -> write -> merge -> faiss 的子集。
    """
    report = PipelineReport(
        config_path=config._config_path,
        excel_path=config.input.excel,
        stages=stages,
    )
    t_start = time.time()

    if clean:
        _step_clean(config, stages)

    # ── read ──
    excel_data: ExcelData | None = None
    if "read" in stages:
        try:
            excel_data = _step_read(config)
            report.excel_ok = True
        except Exception as e:
            logger.exception(f"读取 Excel 失败: {e}")
            report.stopped_at = "read"
            report.total_seconds = time.time() - t_start
            return report

    # ── write ──
    if "write" in stages:
        # parse_stages 保证 write 时必有 read，所以这里 excel_data 不会是 None
        assert excel_data is not None, "write 阶段缺少 read 阶段的数据"
        try:
            report.write_results = _step_write(excel_data, config)
        except Exception as e:
            logger.exception(f"写入 txt 失败: {e}")
            report.stopped_at = "write"
            report.total_seconds = time.time() - t_start
            return report

    # ── merge ──
    if "merge" in stages:
        try:
            report.merge_report = _step_merge(config)
        except Exception as e:
            logger.exception(f"合并失败: {e}")
            report.stopped_at = "merge"
            report.total_seconds = time.time() - t_start
            return report

    # ── faiss ──
    if "faiss" in stages:
        try:
            report.faiss_report = _step_faiss(config)
        except Exception as e:
            logger.exception(f"构建 FAISS 失败: {e}")
            report.stopped_at = "faiss"
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
    print(f"  执行阶段 : {', '.join(report.stages) or '(无)'}")

    # read 失败直接终止展示
    if report.ran("read") and not report.excel_ok:
        print()
        print("  [失败] 读取 Excel 阶段中止")
        print(f"  耗时   : {report.total_seconds:.1f}s")
        print("=" * 78)
        return

    # ── read ──
    print()
    if report.ran("read"):
        print("  [read ] 读取 Excel         : OK")
    else:
        print("  [read ] 读取 Excel         : (跳过)")

    # ── write ──
    print()
    if report.ran("write"):
        print("  [write] 写入 txt")
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
    else:
        print("  [write] 写入 txt           : (跳过)")

    # ── merge ──
    print()
    if report.ran("merge"):
        print("  [merge] 合并")
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
            print("    (未产出)")
    else:
        print("  [merge] 合并               : (跳过)")

    # ── faiss ──
    print()
    if report.ran("faiss"):
        print("  [faiss] FAISS")
        fr = report.faiss_report
        if fr:
            print(f"    built    : {len(fr.built)}")
            print(f"    failed   : {len(fr.failed)}")
            print(f"    questions: {fr.total_questions}")
            if fr.failed:
                for p, err in fr.failed:
                    print(f"      [fail] {p.name}: {err}")
        else:
            print("    (未产出)")
    else:
        print("  [faiss] FAISS              : (跳过)")

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
        description="Corpus 构建流水线：Excel -> txt -> merge -> FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  全跑:
    %(prog)s --config config/corpus.yaml

  改话术时跳过 FAISS（快）:
    %(prog)s --config config/corpus.yaml --only=read,write,merge

  只重建 FAISS（换了 embedding 模型时）:
    %(prog)s --config config/corpus.yaml --only=faiss

  只重跑 merge（手工调过 txt）:
    %(prog)s --config config/corpus.yaml --only=merge

  清空重建:
    %(prog)s --config config/corpus.yaml --clean
""",
    )
    p.add_argument(
        "--config",
        required=True,
        help="corpus.yaml 路径",
    )
    p.add_argument(
        "--only",
        default=None,
        help="只执行指定阶段，逗号分隔。可选: read,write,merge,faiss。"
             "write 会自动补 read。",
    )
    p.add_argument(
        "--skip-faiss",
        action="store_true",
        help="[等价于 --only=read,write,merge] 跳过 FAISS 构建",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="构建前清空 txt_dir（有 write/merge 时）和 faiss_dir（有 faiss 时）",
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

    # 解析 stages
    try:
        stages = parse_stages(args.only, args.skip_faiss)
    except ValueError as e:
        logger.error(str(e))
        return 2

    if not stages:
        logger.error("没有可执行的阶段")
        return 2

    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return 2

    logger.info(f"business = {config.business}")
    logger.info(f"stages   = {stages}")
    logger.info(f"txt_dir  = {config.output.txt_dir}")
    logger.info(f"faiss_dir= {config.output.faiss_dir}")

    report = run_pipeline(config, stages=stages, clean=args.clean)
    print_report(report, config)

    # 退出码
    if report.stopped_at:
        return 1
    if report.faiss_report and report.faiss_report.failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())