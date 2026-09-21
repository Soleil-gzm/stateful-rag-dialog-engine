"""
conditions.py —— 条件组合生成 + 行匹配判定。

对应原代码:
    xlsx2txt-condition.py 的 condition_matches + itertools.product
    txt_merge.py 的 combos 生成
    structure_frontend_v2.py 的 get_case_modules 里 combo_str 拼接

关键约束: 组合顺序必须与原代码 itertools.product(pair1, pair2, pair3) 完全一致,
否则 combo_id 与既有文件名对不上。
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from typing import Iterable

from .config import ConditionsConfig, CorpusConfig


# ==================================================================
# 数据模型
# ==================================================================

@dataclass(frozen=True)
class ComboSpec:
    """一个条件组合。

    例: combo_id=1, combo_str="营销优惠卖点非空_额度非空_预计借款利率区间非空"
    """
    combo_id: int
    combo_str: str
    conditions: frozenset[str]

    def filename_segment(self) -> str:
        """原代码里的文件名片段: combo1_营销优惠卖点非空_额度非空_预计借款利率区间非空"""
        return f"combo{self.combo_id}_{self.combo_str}"

    def __str__(self) -> str:
        return f"Combo#{self.combo_id}({self.combo_str})"


# ==================================================================
# 组合生成
# ==================================================================

def generate_combos(
    config_or_conditions: CorpusConfig | ConditionsConfig,
    *,
    validate_count: bool = False,
) -> list[ComboSpec]:
    """
    生成所有条件组合。

    顺序与 itertools.product(*pairs) 一致:
        - 第一个维度变化最慢
        - 最后一个维度变化最快
        - combo_id 从 1 开始

    参数:
        config_or_conditions: CorpusConfig 或 ConditionsConfig
        validate_count: True 时校验组合数是否超过 config.conditions.max_combos

    返回: list[ComboSpec]，长度 = 2 ** len(dimensions)
    """
    if isinstance(config_or_conditions, CorpusConfig):
        cond_cfg = config_or_conditions.conditions
        max_combos = getattr(cond_cfg, "max_combos", None)
    else:
        cond_cfg = config_or_conditions
        max_combos = getattr(cond_cfg, "max_combos", None)

    # 每个维度是 (nonnull_label, null_label) 的二元组
    pairs: list[tuple[str, str]] = [
        (d.nonnull_label, d.null_label) for d in cond_cfg.dimensions
    ]

    combos: list[ComboSpec] = []
    for combo_id, labels in enumerate(itertools.product(*pairs), start=1):
        combos.append(
            ComboSpec(
                combo_id=combo_id,
                combo_str="_".join(labels),
                conditions=frozenset(labels),
            )
        )

    if validate_count and max_combos is not None:
        if len(combos) > max_combos:
            raise ValueError(
                f"条件组合数 {len(combos)} 超过 max_combos={max_combos}; "
                f"请检查 conditions.dimensions 是否过多"
            )

    return combos


# ==================================================================
# 行匹配：Excel 某行的 Condition 值是否属于当前 combo
# ==================================================================

def condition_matches(
    condition_str: str,
    combo_conditions: Iterable[str],
) -> bool:
    """
    判断某行的 Condition 值是否属于当前 combo。

    规则 (与原代码完全一致):
        - "通用"    -> 匹配所有 combo
        - "A"       -> 匹配 conditions 里包含 A 的 combo
        - "A&B"     -> 匹配同时包含 A 和 B 的 combo
        - "" / 全&  -> 不匹配任何 combo
        - 空白字符  -> strip 后处理

    例:
        combo_conditions = {"营销优惠卖点非空", "额度非空", "预计借款利率区间非空"}

        condition_matches("通用", combo_conditions)                       -> True
        condition_matches("额度非空", combo_conditions)                    -> True
        condition_matches("额度非空&预计借款利率区间非空", combo_conditions) -> True
        condition_matches("额度为空", combo_conditions)                    -> False
        condition_matches("营销优惠卖点非空&额度为空", combo_conditions)     -> False
        condition_matches("", combo_conditions)                           -> False
    """
    condition_str = (condition_str or "").strip()

    if condition_str == "通用":
        return True

    conditions = [c.strip() for c in condition_str.split("&") if c.strip()]
    if not conditions:
        return False

    combo_set = set(combo_conditions)
    return all(c in combo_set for c in conditions)


# ==================================================================
# 运行时：从 case_info 生成 combo_str
# ==================================================================

def build_combo_str_from_case(
    case_info: dict,
    conditions_config: ConditionsConfig,
) -> str:
    """
    运行时用: 根据 case_info 生成 combo_str。

    对应 structure_frontend_v2.py 的:
        marketing_not_null = False if case_info.get('sellingpoint') == "" else True
        ...
        combo_str = f"{marketing_cond}_{quota_cond}_{rate_cond}"

    规则:
        value 为 "" / None / 缺失  -> null_label
        其他                       -> nonnull_label

    与 generate_combos 用同一份 dimensions，保证顺序一致。
    """
    labels: list[str] = []
    for dim in conditions_config.dimensions:
        value = case_info.get(dim.field, "")
        if value is None or value == "":
            labels.append(dim.null_label)
        else:
            labels.append(dim.nonnull_label)
    return "_".join(labels)


# ==================================================================
# 自检
# ==================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.conditions <corpus.yaml>")
        sys.exit(1)

    from .config import load_config

    cfg = load_config(sys.argv[1])

    print("=" * 70)
    print("1. 生成条件组合")
    print("=" * 70)
    combos = generate_combos(cfg, validate_count=True)
    print(f"共 {len(combos)} 个组合:")
    for c in combos:
        print(f"  combo{c.combo_id}: {c.combo_str}")
        print(f"     filename_segment: {c.filename_segment()}")

    print()
    print("=" * 70)
    print("2. 测试 condition_matches")
    print("=" * 70)

    combo1 = combos[0]
    combo2 = combos[1]
    print(f"combo1.conditions = {sorted(combo1.conditions)}")
    print(f"combo2.conditions = {sorted(combo2.conditions)}")
    print()

    test_cases = [
        ("通用", combo1, True),
        ("通用", combo2, True),
        ("额度非空", combo1, True),
        ("额度为空", combo1, False),
        ("预计借款利率区间非空", combo1, True),
        ("预计借款利率区间非空", combo2, False),
        ("额度非空&预计借款利率区间非空", combo1, True),
        ("额度非空&预计借款利率区间非空", combo2, False),
        ("", combo1, False),
        ("&", combo1, False),
        (" 额度非空 ", combo1, True),
    ]

    all_ok = True
    for cond, combo, expected in test_cases:
        got = condition_matches(cond, combo.conditions)
        mark = "OK " if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"  [{mark}] condition_matches({cond!r}, combo{combo.combo_id}) = {got}  (期望 {expected})")

    print()
    print("=" * 70)
    print("3. 测试 build_combo_str_from_case")
    print("=" * 70)
    cases = [
        ({"sellingpoint": "7.9折", "quota": "10000", "interest": "3%-5%"}, "全非空"),
        ({"sellingpoint": "", "quota": "", "interest": ""}, "全为空"),
        ({"sellingpoint": "7.9折", "quota": "", "interest": "3%-5%"}, "卖点非空/额度为空/利率非空"),
        ({}, "字段缺失"),
    ]
    for case_info, desc in cases:
        combo_str = build_combo_str_from_case(case_info, cfg.conditions)
        print(f"  {desc:20s} -> {combo_str}")

    print()
    print("=" * 70)
    print("自检结果:", "全部通过" if all_ok else "存在失败")
    print("=" * 70)