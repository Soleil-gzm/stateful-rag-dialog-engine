# combo 生成 + condition_matches

"""
corpus.yaml 的加载与校验。

设计原则：
1. schema 对应 config/corpus.yaml，字段一一对应
2. 用 Pydantic v2，错误信息友好
3. 提供便利方法（如 path 拼接、module 查找），避免调用方到处拼字符串
4. 所有相对路径统一以 yaml 文件所在目录为基准解析为绝对路径
   —— 这样无论从哪个 cwd 跑 pipeline 都能正确工作
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ==================================================================
# 子模型
# ==================================================================

class ColumnMapping(BaseModel):
    """Excel 列名别名映射。取第一个匹配的列作为标准列。"""
    id: list[str] = Field(default_factory=lambda: ["id"])
    user: list[str] = Field(default_factory=lambda: ["user"])
    agent: list[str] = Field(default_factory=lambda: ["agent"])
    count: list[str] = Field(default_factory=lambda: ["count"])
    condition: list[str] = Field(default_factory=lambda: ["condition"])


class InputConfig(BaseModel):
    excel: Path
    columns: ColumnMapping = Field(default_factory=ColumnMapping)
    skip_sheets: list[str] = Field(default_factory=list)


class ConditionDim(BaseModel):
    """一个二值条件维度，如 sellingpoint 的 非空/为空。"""
    field: str
    nonnull_label: str
    null_label: str

    @model_validator(mode="after")
    def _check_labels_differ(self) -> "ConditionDim":
        if self.nonnull_label == self.null_label:
            raise ValueError(
                f"condition field '{self.field}': "
                f"nonnull_label 与 null_label 不能相同"
            )
        return self


class ConditionsConfig(BaseModel):
    dimensions: list[ConditionDim]

    @field_validator("dimensions")
    @classmethod
    def _check_unique_fields(cls, v: list[ConditionDim]) -> list[ConditionDim]:
        fields = [d.field for d in v]
        dup = [f for f in fields if fields.count(f) > 1]
        if dup:
            raise ValueError(f"conditions.dimensions 存在重复 field: {set(dup)}")
        if not v:
            raise ValueError("conditions.dimensions 不能为空")
        return v

    def labels_of(self, field: str) -> tuple[str, str]:
        """返回 (nonnull_label, null_label)。"""
        for d in self.dimensions:
            if d.field == field:
                return d.nonnull_label, d.null_label
        raise KeyError(f"未定义的条件维度: {field}")


class ModuleSpec(BaseModel):
    """一个业务模块（对应一个 Excel sheet）。"""
    sheet: str                       # Excel 里的中文 sheet 名，写进 txt 的 Label
    id: str                          # 逻辑 id，runtime.yaml 引用它
    display_name: str = ""
    max_count: int = 1
    conditional: bool = False        # 运行时是否按 combo 筛
    excel_has_condition: bool = False  # Excel 里是否已有 Condition 列
    is_append_source: bool = False   # 是否作为 append 源合并到其他模块

    @field_validator("max_count")
    @classmethod
    def _check_max_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_count 必须 >= 1")
        return v

    @model_validator(mode="after")
    def _check_flags(self) -> "ModuleSpec":
        # 有 Condition 列 ⇒ 必须按 combo 展开
        if self.excel_has_condition and not self.conditional:
            raise ValueError(
                f"module '{self.id}': excel_has_condition=True 时 "
                f"conditional 也应为 True"
            )
        return self

    def counts(self) -> list[int]:
        """返回该模块的 count 列表，如 [1, 2]。"""
        return list(range(1, self.max_count + 1))


class ExpandRule(BaseModel):
    """把无条件模块复制到 N 个 combo 名下。"""
    module: str
    counts: list[int]


class AppendTarget(BaseModel):
    module: str
    counts: list[int]


class AppendRule(BaseModel):
    source: str
    targets: list[AppendTarget]


class MergesConfig(BaseModel):
    expand: list[ExpandRule] = Field(default_factory=list)
    append: list[AppendRule] = Field(default_factory=list)


class OutputConfig(BaseModel):
    txt_dir: Path
    faiss_dir: Path
    naming: Literal["legacy", "new"] = "legacy"
    file_stem: str = "output"          # ← 新增，文件名前缀
    manifest: Path
    cache: Path


class EmbeddingConfig(BaseModel):
    model_path: Path
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = 32
    normalize: bool = True


class RenderConfig(BaseModel):
    """话术模板变量替换的字段白名单，供未来 renderer 用。"""
    fields: list[str] = Field(default_factory=list)


# ==================================================================
# 顶层配置
# ==================================================================

class CorpusConfig(BaseModel):
    business: str
    schema_version: int = 1

    input: InputConfig
    conditions: ConditionsConfig
    modules: list[ModuleSpec]
    merges: MergesConfig = Field(default_factory=MergesConfig)
    output: OutputConfig
    embedding: EmbeddingConfig
    render: RenderConfig = Field(default_factory=RenderConfig)

    # 由 load_config 注入，非 yaml 字段
    _config_path: Optional[Path] = None
    _base_dir: Optional[Path] = None

    # ──────────────── 校验 ────────────────

    @field_validator("modules")
    @classmethod
    def _check_unique_ids(cls, v: list[ModuleSpec]) -> list[ModuleSpec]:
        ids = [m.id for m in v]
        dup = [i for i in ids if ids.count(i) > 1]
        if dup:
            raise ValueError(f"modules 存在重复 id: {set(dup)}")

        sheets = [m.sheet for m in v]
        dup_sheet = [s for s in sheets if sheets.count(s) > 1]
        if dup_sheet:
            raise ValueError(f"modules 存在重复 sheet: {set(dup_sheet)}")
        return v

    @model_validator(mode="after")
    def _check_refs(self) -> "CorpusConfig":
        valid_ids = {m.id for m in self.modules}

        for rule in self.merges.expand:
            if rule.module not in valid_ids:
                raise ValueError(f"merges.expand 引用了不存在的 module: {rule.module}")

        for rule in self.merges.append:
            if rule.source not in valid_ids:
                raise ValueError(f"merges.append.source 不存在: {rule.source}")
            for t in rule.targets:
                if t.module not in valid_ids:
                    raise ValueError(
                        f"merges.append.targets 引用了不存在的 module: {t.module}"
                    )
        return self

    @model_validator(mode="after")
    def _check_append_source_flag(self) -> "CorpusConfig":
        """被 append 引用的 source 必须标 is_append_source=True。"""
        append_sources = {r.source for r in self.merges.append}
        for m in self.modules:
            if m.id in append_sources and not m.is_append_source:
                raise ValueError(
                    f"module '{m.id}' 被 merges.append 引用，"
                    f"但 is_append_source=False"
                )
        return self

    # ──────────────── 便利方法 ────────────────

    def get_module(self, module_id: str) -> ModuleSpec:
        for m in self.modules:
            if m.id == module_id:
                return m
        raise KeyError(f"未找到 module id: {module_id}")

    def all_module_ids(self) -> list[str]:
        return [m.id for m in self.modules]

    @property
    def base_dir(self) -> Path:
        """yaml 文件所在目录，用于解析相对路径。"""
        if self._base_dir is None:
            raise RuntimeError("base_dir 未初始化，请通过 load_config() 加载")
        return self._base_dir

    def resolve_path(self, p: Path) -> Path:
        """把 yaml 里的相对路径解析成绝对路径。"""
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()


# ==================================================================
# 加载器
# ==================================================================

def load_config(path: str | Path) -> CorpusConfig:
    """
    从 yaml 加载 CorpusConfig。

    - 相对路径以 yaml 文件所在目录为基准解析为绝对路径
    - 校验失败时抛 pydantic.ValidationError，信息包含字段路径
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"yaml 顶层必须是 mapping，实际是 {type(raw).__name__}")

    cfg = CorpusConfig.model_validate(raw)

    # 注入 base_dir，并把所有相对路径解析为绝对
    cfg._config_path = path
    cfg._base_dir = path.parent

    cfg.input.excel = cfg.resolve_path(cfg.input.excel)
    cfg.output.txt_dir = cfg.resolve_path(cfg.output.txt_dir)
    cfg.output.faiss_dir = cfg.resolve_path(cfg.output.faiss_dir)
    cfg.output.manifest = cfg.resolve_path(cfg.output.manifest)
    cfg.output.cache = cfg.resolve_path(cfg.output.cache)
    cfg.embedding.model_path = cfg.resolve_path(cfg.embedding.model_path)

    return cfg


# ==================================================================
# 自检：直接运行会打印配置摘要
# ==================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m stateful_rag_dialog_engine.corpus.config <corpus.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    print(f"business      : {cfg.business}")
    print(f"base_dir      : {cfg.base_dir}")
    print(f"excel         : {cfg.input.excel}")
    print(f"dimensions    : {[d.field for d in cfg.conditions.dimensions]}")
    print(f"modules ({len(cfg.modules)}):")
    for m in cfg.modules:
        flags = []
        if m.conditional:
            flags.append("conditional")
        if m.excel_has_condition:
            flags.append("excel_cond")
        if m.is_append_source:
            flags.append("append_src")
        print(f"  - {m.id:<12} sheet={m.sheet}  max_count={m.max_count}  [{','.join(flags)}]")
    print(f"merges.expand : {[(r.module, r.counts) for r in cfg.merges.expand]}")
    print(f"merges.append : {[(r.source, [t.module for t in r.targets]) for r in cfg.merges.append]}")
    print(f"txt_dir       : {cfg.output.txt_dir}")
    print(f"faiss_dir     : {cfg.output.faiss_dir}")
    print(f"embedding     : {cfg.embedding.model_path} ({cfg.embedding.device})")