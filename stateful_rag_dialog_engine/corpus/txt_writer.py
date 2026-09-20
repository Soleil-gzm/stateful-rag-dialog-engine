"""
txt_writer.py —— DataFrame 写入 Question/Answer/Label 格式的 txt。

对应原代码:
    xlsx2txt-condition.py 的 write_qa_txt + process_sheet_to_txt + process_condition_sheet

关键约束:
    输出文件名和内容必须与原脚本字节级一致，
    否则运行时的 core 会找不到文件或 label 不匹配。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import CorpusConfig, ModuleSpec
from .conditions import ComboSpec, condition_matches, generate_combos
from .excel_reader import ExcelData, SheetData

logger = logging.getLogger(__name__)


# ==================================================================
# count 格式化
# ==================================================================

def _format_count(count) -> str:
    """
    把 count 规范成文件名中的字符串。

    原代码在 condition 分支里做了同样的处理:
        count_str = str(int(count)) if float(count).is_integer() else str(count)

    例:
        1    -> "1"
        1.0  -> "1"
        1.5  -> "1.5"
    """
    try:
        if float(count).is_integer():
            return str(int(count))
    except (TypeError, ValueError):
        pass
    return str(count)


# ==================================================================
# 单文件写入
# ==================================================================

def write_qa_txt(path: Path, df: pd.DataFrame, label: str) -> int:
    """
    逐行写txt Question / Answer / Label。
    返回写入的行数（QA 对数）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            line = "Question: {}\nAnswer: {}\nLabel: {}\n".format(
                row["user"], row["agent"], label
            )
            f.write(line)
            n += 1
    return n


# ==================================================================
# 路径构建
# ==================================================================

def build_output_path(
    module: ModuleSpec,
    count,
    combo: ComboSpec | None,
    config: CorpusConfig,
) -> Path:
    """
    按 naming: legacy 构建输出路径。

    无条件模块（combo=None）:
        {txt_dir}/{file_stem}_{sheet}_{count}.txt

    有条件模块（combo 提供）:
        {txt_dir}/{file_stem}_{sheet}_count{count}_combo{combo_id}_{combo_str}.txt
    """
    txt_dir = config.output.txt_dir
    stem = config.output.file_stem
    count_str = _format_count(count)

    if config.output.naming == "legacy":
        if combo is None:
            filename = f"{stem}_{module.sheet}_{count_str}.txt"
        else:
            filename = (
                f"{stem}_{module.sheet}_count{count_str}_"
                f"combo{combo.combo_id}_{combo.combo_str}.txt"
            )
        return txt_dir / filename

    raise NotImplementedError(f"naming={config.output.naming} 尚未实现")


# ==================================================================
# 结果
# ==================================================================

@dataclass
class WriteResult:
    module_id: str
    files_written: list[Path] = field(default_factory=list)
    rows_total: int = 0
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"WriteResult(module={self.module_id!r}, "
            f"files={len(self.files_written)}, rows={self.rows_total})"
        )


# ==================================================================
# 单模块
# ==================================================================

def write_module(
    sheet: SheetData,
    module: ModuleSpec,
    config: CorpusConfig,
) -> WriteResult:
    """
    处理单个模块的所有 count，写入 txt 文件。

    分流:
        - sheet.has_condition=True  -> 按 count + combo 拆分，写多份
        - sheet.has_condition=False -> 按 count 分组，直接写
    """
    result = WriteResult(module_id=module.id)

    if sheet.skipped or sheet.df is None:
        result.warnings.append(f"sheet {sheet.sheet_name!r} 被跳过")
        return result

    df = sheet.df
    if "count" not in df.columns:
        result.warnings.append(f"sheet {sheet.sheet_name!r} 没有 count 列")
        return result

    # 数据为准：如果和 config 声明不一致，给 warning，但按数据实际处理
    if sheet.has_condition != module.excel_has_condition:
        msg = (
            f"module {module.id!r}: config.excel_has_condition="
            f"{module.excel_has_condition}, 但 Excel 检测到 "
            f"has_condition={sheet.has_condition}; 以数据为准"
        )
        logger.warning(msg)
        result.warnings.append(msg)

    if sheet.has_condition:
        _write_conditional(sheet, module, config, result)
    else:
        _write_unconditional(sheet, module, config, result)

    return result


def _write_unconditional(
    sheet: SheetData,
    module: ModuleSpec,
    config: CorpusConfig,
    result: WriteResult,
) -> None:
    """
    对应原 process_sheet_to_txt 的无 condition 分支:
        max_count = df['count'].max()
        for count in range(1, int(max_count) + 1):
            df_count = df[df['count'] == count]
            if not df_count.empty:
                write_qa_txt(output_file, df_count, sheet_name)
    """
    df = sheet.df
    max_count = df["count"].max()
    if pd.isna(max_count):
        result.warnings.append(f"sheet {sheet.sheet_name!r} 的 count 列全为空")
        return

    for count in range(1, int(max_count) + 1):
        df_count = df[df["count"] == count]
        if df_count.empty:
            logger.info(f"sheet {sheet.sheet_name!r} count={count} 无数据，跳过")
            continue

        out_path = build_output_path(module, count, combo=None, config=config)
        n = write_qa_txt(out_path, df_count, module.sheet)
        result.files_written.append(out_path)
        result.rows_total += n
        logger.info(f"[写入] {module.id} count={count} -> {out_path.name} ({n} 行)")


def _write_conditional(
    sheet: SheetData,
    module: ModuleSpec,
    config: CorpusConfig,
    result: WriteResult,
) -> None:
    """
    对应原 process_condition_sheet:
        count_values = sorted(df['count'].dropna().unique())
        for count in count_values:
            df_count = df[df['count'] == count]
            for combo in itertools.product(...):
                mask = df_count['condition'].apply(condition_matches)
                combo_df = df_count[mask]
                if combo_df.empty: continue
                write_qa_txt(output_file, combo_df, sheet_name)
    """
    df = sheet.df
    combos = generate_combos(config, validate_count=True)

    count_values = sorted(df["count"].dropna().unique())
    if len(count_values) == 0:
        result.warnings.append(f"sheet {sheet.sheet_name!r} 的 count 列全为空")
        return

    for count in count_values:
        df_count = df[df["count"] == count]
        if df_count.empty:
            continue

        for combo in combos:
            mask = df_count["condition"].apply(
                lambda x: condition_matches(x, combo.conditions)
            )
            combo_df = df_count[mask]
            if combo_df.empty:
                logger.info(
                    f"sheet {sheet.sheet_name!r} count={_format_count(count)} "
                    f"combo{combo.combo_id} 无数据，跳过"
                )
                continue

            out_path = build_output_path(module, count, combo=combo, config=config)
            n = write_qa_txt(out_path, combo_df, module.sheet)
            result.files_written.append(out_path)
            result.rows_total += n
            logger.info(
                f"[写入] {module.id} count={_format_count(count)} "
                f"combo{combo.combo_id} -> {out_path.name} ({n} 行)"
            )


# ==================================================================
# 批量
# ==================================================================

def write_all(
    excel_data: ExcelData,
    config: CorpusConfig,
) -> list[WriteResult]:
    """
    按 config.modules 逐个写入。

    只处理 config.modules 里列出的 sheet，其他 sheet 忽略。
    """
    sheet_map = {s.sheet_name: s for s in excel_data.sheets}

    results: list[WriteResult] = []
    for module in config.modules:
        sheet = sheet_map.get(module.sheet)
        if sheet is None:
            logger.warning(
                f"yaml 定义 module {module.id!r} (sheet={module.sheet!r})，"
                f"但 Excel 里没有该 sheet，跳过"
            )
            r = WriteResult(module_id=module.id)
            r.warnings.append(f"sheet {module.sheet!r} 在 Excel 中不存在")
            results.append(r)
            continue

        r = write_module(sheet, module, config)
        results.append(r)

    return results


# ==================================================================
# 自检
# ==================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.txt_writer <corpus.yaml>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )

    from .config import load_config
    from .excel_reader import read_excel

    cfg = load_config(sys.argv[1])
    data = read_excel(cfg.input.excel, cfg)
    results = write_all(data, cfg)

    print()
    print("=" * 78)
    print("写入汇总")
    print("=" * 78)
    for r in results:
        print(
            f"  {r.module_id:<12s} | files={len(r.files_written):>3d} | "
            f"rows={r.rows_total:>4d}"
        )
        for w in r.warnings:
            print(f"    [warn] {w}")

    total_files = sum(len(r.files_written) for r in results)
    total_rows = sum(r.rows_total for r in results)
    print()
    print(f"  合计: {total_files} 个文件, {total_rows} 行")
    print(f"  输出目录: {cfg.output.txt_dir}")