"""
按 case_info 的条件，筛选出"当前这通电话该用哪一套模块"。

原逻辑在 Pathway.get_case_modules 里，这里抽出来做纯函数，方便单测。
"""
import logging
import re

from .specs import ModuleSpec

logger = logging.getLogger(__name__)


# 需要按条件筛选的类别
_CONDITIONAL_CATEGORIES = {'确认', '答疑'}


def _extract_count(file_prefix: str) -> int:
    """从文件名前缀中提取 count 数字，用于排序"""
    match = re.search(r'count(\d+)', file_prefix)
    if match:
        return int(match.group(1))
    return 0


def select_specs(
    specs_all: dict[str, list[ModuleSpec]],
    case_info: dict,
) -> dict[str, list[ModuleSpec]]:
    """
    根据客户 case 的条件（营销优惠卖点 / 额度 / 利率是否为空），
    筛选出 确认 / 答疑 类别的具体模块。其他类别原样返回。

    返回: {category: [ModuleSpec, ...]}（顺序与 count 一致）
    """
    marketing_not_null = False if case_info.get('sellingpoint') == "" else True
    quota_not_null = False if case_info.get('quota') == "" else True
    rate_not_null = False if case_info.get('interest') == "" else True

    marketing_cond = '营销优惠卖点非空' if marketing_not_null else '营销优惠卖点为空'
    quota_cond = '额度非空' if quota_not_null else '额度为空'
    rate_cond = '预计借款利率区间非空' if rate_not_null else '预计借款利率区间为空'
    combo_str = f"{marketing_cond}_{quota_cond}_{rate_cond}"

    selected: dict[str, list[ModuleSpec]] = {}

    for category, specs in specs_all.items():
        if category in _CONDITIONAL_CATEGORIES:
            # 从当前类别中筛选出文件前缀包含 combo_str 的模块
            matched = [s for s in specs if combo_str in s.file_prefix]
            # 按文件名中的 count 数字排序（确保 repeat 索引对应正确的 count 顺序）
            matched.sort(key=lambda s: _extract_count(s.file_prefix))
            selected[category] = matched
            # 可选：打印警告
            if len(matched) == 0:
                logger.warning("未找到 %s 类别且满足条件 %s 的模块", category, combo_str)
        else:
            selected[category] = specs

    return selected     # 筛选符合客户的话术模块吗？