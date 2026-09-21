"""
模块清单定义。这里是"哪些模块存在"的唯一事实来源。

注意：本模块只描述配置（ModuleSpec），不构造任何重量级对象。
真正的 Module（含 FAISS）由 ModuleRegistry 在运行时按需创建。
"""
from dataclasses import dataclass
import itertools


@dataclass(frozen=True)
class ModuleSpec:
    """一个话术模块的轻量描述（只有 id，没有内容）。"""
    category: str
    file_prefix: str


# (类别, 基础名称, 最大count, 是否有条件)
MODULE_CONFIGS = [
    ('核实', '模块1-确认身份', 1, False),
    ('产介', '模块2-产介', 1, False),
    ('三方', '模块3-三方', 1, False),
    ('确认', '模块4-意愿确认', 2, True),   # 有条件，count从1到2
    ('答疑', '模块5-异议处理', 2, True),   # 有条件，count从1到2
    ('投诉', '投诉倾向', 1, False),
    ('留言', '语音留言', 1, False),
    # ('信息', '信息问题', 1, True)
]

# 三对二值条件
CONDITION_PAIRS = [
    ('营销优惠卖点非空', '营销优惠卖点为空'),
    ('额度非空', '额度为空'),
    ('预计借款利率区间非空', '预计借款利率区间为空'),
]


def build_all_specs() -> dict[str, list[ModuleSpec]]:
    """
    生成所有模块的 spec。

    返回: {category: [ModuleSpec, ...]}
    无条件的模块只有 1 个 spec；有条件的按 count × 8 种条件组合生成。
    """
    specs_all: dict[str, list[ModuleSpec]] = {}

    for category, base_name, max_count, has_condition in MODULE_CONFIGS:
        specs: list[ModuleSpec] = []
        if not has_condition:
            # 无条件的模块：仅一个（count固定为1）
            file_prefix = f"output_{base_name}_1"
            specs.append(ModuleSpec(category, file_prefix))
        else:
            # 有条件的模块：按count和8种组合生成
            for count in range(1, max_count + 1):
                combo_id = 1
                for combo in itertools.product(*CONDITION_PAIRS):
                    combo_str = '_'.join(combo)
                    file_prefix = f"output_{base_name}_count{count}_combo{combo_id}_{combo_str}"
                    specs.append(ModuleSpec(category, file_prefix))
                    combo_id += 1
        specs_all[category] = specs

    return specs_all