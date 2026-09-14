"""
merger.py —— 语料合并。

对应原代码:
    txt_merge.py
    - 把无条件模块复制到每个 combo 名下（expand）
    - 把 append 源合并进目标模块（append）

与原代码的关键差异:
    1. 幂等：重复跑不会重复 append
    2. 数据驱动：规则从 corpus.yaml 的 merges 段读，不硬编码模块名
    3. 路径统一：文件名用 build_output_path，不手写字符串
    4. 计数回退：源模块的 count 不足时，自动回退到其 max_count

使用前提:
    merger 假定 txt_dir 里的文件处于 "txt_writer 刚跑完" 的状态。
    pipeline.py 保证按 txt_writer → merger 的顺序执行。
    如果单独跑 merger 两次而中间不重跑 txt_writer，
    已有的 append 结果可能不会被覆盖（但末尾检测会跳过重复追加）。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import CorpusConfig, ModuleSpec
from .conditions import generate_combos
from .txt_writer import build_output_path, _format_count

logger = logging.getLogger(__name__)


# ==================================================================
# 报告
# ==================================================================

@dataclass
class MergeReport:
    expanded: list[Path] = field(default_factory=list)
    appended: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"expanded={len(self.expanded)}, "
            f"appended={len(self.appended)}, "
            f"skipped={len(self.skipped)}, "
            f"warnings={len(self.warnings)}"
        )


# ==================================================================
# 工具
# ==================================================================

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source_count_for_target(
    target_count,
    source_module: ModuleSpec,
) -> int:
    """
    决定用源的哪个 count 文件。

    规则:
        - 源只有 1 个 count -> 总是用 count=1
        - 否则 -> min(目标 count, 源 max_count)

    对应原 txt_merge.py 里 source 固定为 count1 的写法，
    推广到源有多个 count 的场景。
    """
    if source_module.max_count == 1:
        return 1
    try:
        tc = int(target_count)
    except (TypeError, ValueError):
        return 1
    return min(tc, source_module.max_count)


# ==================================================================
# expand
# ==================================================================

def _apply_expand(rule, config: CorpusConfig, combos, report: MergeReport) -> None:
    """
    把无条件模块的 {stem}_{sheet}_{count}.txt
    复制到 {stem}_{sheet}_count{N}_combo{I}_{combo_str}.txt
    对应原 txt_merge.py 段 1。
    """
    module = config.get_module(rule.module)

    for count in rule.counts:
        src = build_output_path(module, count, combo=None, config=config)
        if not src.exists():
            msg = f"expand: 源文件不存在 {src}"
            logger.error(msg)
            report.warnings.append(msg)
            continue

        content = _read_text(src)
        written = 0
        for combo in combos:
            dst = build_output_path(module, count, combo=combo, config=config)
            _write_text(dst, content)
            report.expanded.append(dst)
            written += 1

        logger.info(
            f"[expand] {module.id} count={_format_count(count)} "
            f"-> {written} 个 combo 文件"
        )


# ==================================================================
# append
# ==================================================================

def _apply_append(rule, config: CorpusConfig, combos, report: MergeReport) -> None:
    """
    把源的 combo 文件追加到目标的 combo 文件。
    对应原 txt_merge.py 段 2。
    """
    source_module = config.get_module(rule.source)

    for target in rule.targets:
        target_module = config.get_module(target.module)

        for target_count in target.counts:
            source_count = _source_count_for_target(target_count, source_module)

            n_appended = 0
            n_skipped = 0
            n_missing = 0

            for combo in combos:
                src = build_output_path(
                    source_module, source_count, combo=combo, config=config
                )
                if not src.exists():
                    msg = f"append: 源文件不存在 {src}"
                    logger.warning(msg)
                    report.warnings.append(msg)
                    n_missing += 1
                    continue

                dst = build_output_path(
                    target_module, target_count, combo=combo, config=config
                )

                src_content = _read_text(src)
                existing = _read_text(dst) if dst.exists() else ""

                # 幂等检查：如果目标已经以源内容结尾，说明追加过了
                if existing and existing.endswith(src_content):
                    logger.debug(f"[append] 跳过（已合并） {dst.name}")
                    report.skipped.append(dst)
                    n_skipped += 1
                    continue

                _write_text(dst, existing + src_content)
                report.appended.append(dst)
                n_appended += 1

            logger.info(
                f"[append] {source_module.id}(count{_format_count(source_count)}) "
                f"-> {target_module.id}(count{_format_count(target_count)}): "
                f"appended={n_appended}, skipped={n_skipped}, missing_src={n_missing}"
            )


# ==================================================================
# 主入口
# ==================================================================

def merge_all(config: CorpusConfig) -> MergeReport:
    """按 corpus.yaml 的 merges 段执行所有合并规则。"""
    combos = generate_combos(config, validate_count=True)
    report = MergeReport()

    logger.info(f"生成 {len(combos)} 个条件组合")

    # Phase 1: expand
    if config.merges.expand:
        logger.info(f"=== expand: {len(config.merges.expand)} 条规则 ===")
        for rule in config.merges.expand:
            _apply_expand(rule, config, combos, report)

    # Phase 2: append
    if config.merges.append:
        logger.info(f"=== append: {len(config.merges.append)} 条规则 ===")
        for rule in config.merges.append:
            _apply_append(rule, config, combos, report)

    logger.info(f"合并完成: {report.summary}")
    return report


# ==================================================================
# 自检
# ==================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.merger <corpus.yaml>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )

    from .config import load_config

    cfg = load_config(sys.argv[1])
    report = merge_all(cfg)

    print()
    print("=" * 78)
    print("合并报告")
    print("=" * 78)
    print(f"  expanded : {len(report.expanded)} 个文件")
    print(f"  appended : {len(report.appended)} 个文件")
    print(f"  skipped  : {len(report.skipped)} 个文件（已合并过）")
    if report.warnings:
        print(f"  warnings : {len(report.warnings)} 条")
        for w in report.warnings:
            print(f"    [warn] {w}")
    else:
        print(f"  warnings : 无")
    print()
    print(f"  输出目录: {cfg.output.txt_dir}")