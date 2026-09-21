"""
excel_reader.py —— 读 Excel，规范化 sheet 数据。

对应原代码:
    xlsx2txt-condition.py 的 process_sheet_to_txt 读取部分
    - pd.ExcelFile / pd.read_excel
    - replace_newline
    - 按列数判断 (4 列 / 5 列)
    - 跳过 skip_sheets

职责边界:
    本模块只做"读 + 规范化"，不生成 txt，不做条件匹配。
    输出的 DataFrame 列名固定为:
        无条件 sheet: [id, user, agent, count]
        有条件 sheet: [id, user, agent, count, condition]
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import CorpusConfig

logger = logging.getLogger(__name__)


# 标准列名 —— 与写入 txt 时的 df['user'] / df['agent'] 对应
STD_COLS_NO_COND: list[str] = ["id", "user", "agent", "count"]
STD_COLS_WITH_COND: list[str] = ["id", "user", "agent", "count", "condition"]


# ==================================================================
# 换行符处理
# ==================================================================

def replace_newline(text):
    """与原代码一致：只处理 str，其他类型原样返回。"""
    if isinstance(text, str):
        return text.replace("\n", " ")
    return text


def _replace_newline_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    对整个 DataFrame 逐单元格做换行替换。

    兼容 pandas 版本:
        pandas >= 2.1 : df.map
        pandas <  2.1 : df.applymap
    """
    if hasattr(df, "map"):
        return df.map(replace_newline)
    return df.applymap(replace_newline)


# ==================================================================
# 数据模型
# ==================================================================

@dataclass
class SheetData:
    """一个 sheet 的读取结果。"""
    sheet_name: str
    df: pd.DataFrame | None            # 跳过的 sheet 为 None
    has_condition: bool
    skipped: bool = False
    skip_reason: str = ""

    def __repr__(self) -> str:
        if self.skipped:
            return f"SheetData({self.sheet_name!r}, SKIPPED: {self.skip_reason})"
        n = len(self.df) if self.df is not None else 0
        return (
            f"SheetData({self.sheet_name!r}, rows={n}, "
            f"cond={self.has_condition})"
        )


@dataclass
class ExcelData:
    """整个 Excel 的读取结果。"""
    path: Path
    sheets: list[SheetData] = field(default_factory=list)

    @property
    def valid_sheets(self) -> list[SheetData]:
        return [s for s in self.sheets if not s.skipped]

    @property
    def skipped_sheets(self) -> list[SheetData]:
        return [s for s in self.sheets if s.skipped]

    def get_sheet(self, name: str) -> SheetData:
        for s in self.sheets:
            if s.sheet_name == name:
                return s
        raise KeyError(f"未找到 sheet: {name}")


# ==================================================================
# 主入口
# ==================================================================

def read_excel(
    path: str | Path,
    config: CorpusConfig | None = None,
) -> ExcelData:
    """
    读取 Excel 所有 sheet，规范化列名与换行符。

    规则 (对应原 xlsx2txt-condition.py):
        - skip_sheets 里的 sheet 直接跳过
        - 4 列 -> 无 condition (列名 [id, user, agent, count])
        - 5 列 -> 有 condition (列名 [id, user, agent, count, condition])
        - 其他列数 -> 跳过并记录原因 (原代码同样跳过)

    参数:
        path:   Excel 文件路径
        config: 可选。提供时使用 config.input.skip_sheets。
                不提供则不跳过任何 sheet（除空表和列数不对的）。

    返回:
        ExcelData，包含所有 sheet（含跳过的）。
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {path}")

    skip_sheets: set[str] = set()
    if config is not None:
        skip_sheets = set(config.input.skip_sheets)

    logger.info(f"读取 Excel: {path}")
    xls = pd.ExcelFile(path)
    result = ExcelData(path=path)

    for sheet_name in xls.sheet_names:
        sheet = _process_one_sheet(xls, sheet_name, skip_sheets)
        result.sheets.append(sheet)

    logger.info(
        f"共 {len(result.sheets)} 个 sheet, "
        f"有效 {len(result.valid_sheets)}, 跳过 {len(result.skipped_sheets)}"
    )
    return result


# ==================================================================
# 单个 sheet 处理
# ==================================================================

def _process_one_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    skip_sheets: set[str],
) -> SheetData:
    # 1. 用户显式跳过
    if sheet_name in skip_sheets:
        logger.info(f"[跳过] sheet {sheet_name!r} (在 skip_sheets 中)")
        return SheetData(
            sheet_name=sheet_name,
            df=None,
            has_condition=False,
            skipped=True,
            skip_reason="in skip_sheets",
        )

    # 2. 读
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        logger.warning(f"[跳过] sheet {sheet_name!r} 读取失败: {e}")
        return SheetData(
            sheet_name=sheet_name,
            df=None,
            has_condition=False,
            skipped=True,
            skip_reason=f"read_error: {e}",
        )

    # 3. 空表
    if df.empty:
        logger.warning(f"[跳过] sheet {sheet_name!r} 是空表")
        return SheetData(
            sheet_name=sheet_name,
            df=None,
            has_condition=False,
            skipped=True,
            skip_reason="empty sheet",
        )

    # 4. 换行替换
    df = _replace_newline_in_df(df)

    # 5. 按列数分流
    n_cols = len(df.columns)

    if n_cols == 4:
        df.columns = STD_COLS_NO_COND
        logger.info(f"[OK]   sheet {sheet_name!r}: {len(df)} 行, 无 condition")
        return SheetData(
            sheet_name=sheet_name,
            df=df,
            has_condition=False,
        )

    if n_cols == 5:
        df.columns = STD_COLS_WITH_COND
        # condition 列规范化: str + strip（与原代码一致，NaN 会变成 'nan' 字符串）
        df["condition"] = df["condition"].astype(str).str.strip()
        logger.info(f"[OK]   sheet {sheet_name!r}: {len(df)} 行, 有 condition")
        return SheetData(
            sheet_name=sheet_name,
            df=df,
            has_condition=True,
        )

    logger.warning(
        f"[跳过] sheet {sheet_name!r} 列数为 {n_cols}, 既不是 4 也不是 5"
    )
    return SheetData(
        sheet_name=sheet_name,
        df=None,
        has_condition=False,
        skipped=True,
        skip_reason=f"invalid column count: {n_cols}",
    )


# ==================================================================
# 自检
# ==================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.excel_reader <corpus.yaml>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )

    from .config import load_config

    cfg = load_config(sys.argv[1])
    data = read_excel(cfg.input.excel, cfg)

    print()
    print("=" * 78)
    print(f"Sheet 概览 ({data.path.name})")
    print("=" * 78)
    for s in data.sheets:
        if s.skipped:
            print(f"  [跳过] {s.sheet_name:<20s} | {s.skip_reason}")
        else:
            n = len(s.df)
            cols = ",".join(s.df.columns)
            tag = "有cond" if s.has_condition else "无cond"
            print(f"  [OK]   {s.sheet_name:<20s} | rows={n:>3d} | {tag} | cols=[{cols}]")

    print()
    print("=" * 78)
    print("对照 corpus.yaml 的 modules")
    print("=" * 78)
    defined_sheets = {m.sheet for m in cfg.modules}
    excel_sheets = {s.sheet_name for s in data.valid_sheets}

    missing = defined_sheets - excel_sheets
    extra = excel_sheets - defined_sheets

    if missing:
        print(f"  [警告] yaml 定义了但 Excel 里没找到: {sorted(missing)}")
    if extra:
        print(f"  [警告] Excel 里有但 yaml 没定义: {sorted(extra)}")
    if not missing and not extra:
        print("  完全匹配 ✓")

    print()
    print("=" * 78)
    print("每个有效 sheet 前 3 行")
    print("=" * 78)
    for s in data.valid_sheets:
        print(f"\n--- {s.sheet_name} ---")
        print(s.df.head(3).to_string(max_colwidth=40))