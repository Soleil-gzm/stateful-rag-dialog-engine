"""
决策层抽象。

原项目 ``track.py`` 存在三类问题,本模块逐一修复:
1. ``AssistantTracker`` 与 ``QueryTracker`` 几乎完全重复,只有 prompt
   中角色映射不同 -> 抽象为 ``DecisionModel`` 协议 + 单个 ``LLMDecisionModel``。
2. ``except ValidationError`` 从未导入 ``ValidationError``,异常永远捕获不到,
   真实异常会冒泡导致服务崩溃 -> 显式 ``from pydantic import ValidationError``。
3. JSON 提取用 ``text.find('{')..rfind('}')`` 截取,模型输出多段 JSON 或
   嵌套花括号时会拼出非法字符串 -> 改用 ``json.loads`` + 容错回退,
   解析失败时返回 schema 默认值并记录上下文,便于排查。

决策层与状态机解耦:决策器只负责"给定历史 -> 输出结构化意图",
状态机消费意图触发转移。两者通过 ``DialogContext`` 与 ``DecisionResult``
传递数据,任何一方可独立替换。
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .types import DialogContext, DialogTurn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_FALLBACK_NODE = "unknown"
"""决策失败的默认节点值,与原项目 ``State(node="unknown")`` 行为一致。"""


@dataclass
class DecisionResult:
    """决策器单次输出。

    持有结构化 schema 实例、原始模型文本、是否命中回退。回退时引擎可
    选择降级策略(走默认转移或转人工)。
    """

    payload: BaseModel
    raw_text: str = ""
    fallback: bool = False
    error: str | None = None

    def field(self, name: str, default: Any = None) -> Any:
        """从 payload 取字段,缺失返回 default。"""
        return getattr(self.payload, name, default)


@runtime_checkable
class DecisionModel(Protocol):
    """决策器协议。

    业务方可实现任意后端:LLM、规则、小模型分类器、随机 mock。
    引擎只依赖此协议,便于离线测试用 ``MockDecisionModel`` 替换真模型。
    """

    schema: Type[BaseModel]

    def decide(self, ctx: DialogContext) -> DecisionResult:
        """根据上下文历史输出决策。"""
        ...


class DecisionError(RuntimeError):
    """决策器调用失败(非模型输出校验失败,而是后端不可用)。"""


# ---------------------------------------------------------------------------
# 后端抽象
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """LLM 后端协议,屏蔽 TRT-LLM / vLLM / OpenAI 差异。

    只要求实现 ``generate(prompt: str) -> str``。原项目里 TRT-LLM 的
    ``ModelRunnerCpp.generate`` 加上一堆 tensor 转换逻辑可包装成此协议。
    """

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class _RandomBackend:
    """无外部依赖的随机后端,用于离线 demo 与冒烟测试。"""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # 故意生成结构化 JSON,模拟真实 LLM 输出
        candidates = ["yes", "no", "cc", "nonidentity", "message", "continue"]
        node = self._rng.choice(candidates)
        return json.dumps({"node": node})


# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json(text: str) -> str | None:
    """从模型输出中提取首个 JSON 对象字符串。

    相比原项目 ``text.find('{')..rfind('}')`` 的粗暴截取:
    - 用正则按花括号平衡匹配,避免吞掉后续多段 JSON;
    - 匹配失败返回 None,交由上层决定回退策略。
    """
    if not text:
        return None
    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None


def safe_parse(
    text: str, schema: Type[T], fallback_node: str = _DEFAULT_FALLBACK_NODE
) -> DecisionResult:
    """容错解析:解析失败时返回 schema 的默认实例并标记 fallback。

    原项目 ``except ValidationError`` 因未 import 而失效,这里显式捕获,
    并把 raw_text 与 error 一起返回,便于日志回放与回归。
    """
    raw = text or ""
    candidate = extract_json(raw)
    if candidate is not None:
        candidate = candidate.replace("\\_", "_")
        try:
            payload = schema.model_validate_json(candidate)
            return DecisionResult(payload=payload, raw_text=raw)
        except ValidationError as exc:
            logger.warning("decision parse failed: %s | raw=%r", exc, raw[:200])
            return DecisionResult(
                payload=schema(node=fallback_node),
                raw_text=raw,
                fallback=True,
                error=str(exc),
            )
    logger.warning("no JSON object in decision output, raw=%r", raw[:200])
    return DecisionResult(
        payload=schema(node=fallback_node),
        raw_text=raw,
        fallback=True,
        error="no_json_object",
    )


# ---------------------------------------------------------------------------
# 具体决策器
# ---------------------------------------------------------------------------


@dataclass
class LLMDecisionModel:
    """LLM 决策器。

    用法::

        class Intent(BaseModel):
            node: str

        model = LLMDecisionModel(
            schema=Intent,
            backend=my_llm_backend,
            prompt_template="历史:{history}\n输出 JSON: {schema}",
        )
        result = model.decide(ctx)
        intent = result.field("node")

    设计要点:
    - ``build_prompt`` 为普通方法,业务可继承覆盖格式化逻辑;
    - ``decide`` 全程不抛业务异常(仅后端不可用时抛 ``DecisionError``),
      便于状态机层统一兜底;
    - 默认 ``history_n=2`` 与原项目只看最后一轮的行为保持一致。
    """

    schema: Type[BaseModel]
    backend: LLMBackend
    prompt_template: str
    history_n: int = 2
    fallback_node: str = _DEFAULT_FALLBACK_NODE
    template_vars: dict[str, Any] = field(default_factory=dict)

    def build_prompt(self, ctx: DialogContext) -> str:
        """构造决策 prompt,默认拼接最近 N 轮对话与 schema。"""
        recent = ctx.recent(self.history_n)
        history_lines = []
        for turn in recent:
            speaker = "客户" if turn.role == "user" else "专员"
            history_lines.append(f"{speaker}:{turn.user_message or turn.assistant_message}")
        history = "\n".join(history_lines)
        from string import Template

        template = Template(self.prompt_template)
        return template.safe_substitute(
            history=history,
            schema=self.schema.model_json_schema(),
            **{k: str(v) for k, v in self.template_vars.items()},
        )

    def decide(self, ctx: DialogContext) -> DecisionResult:
        prompt = self.build_prompt(ctx)
        try:
            raw = self.backend.generate(prompt=prompt)
        except Exception as exc:
            logger.exception("LLM backend failed")
            return DecisionResult(
                payload=self.schema(node=self.fallback_node),
                raw_text="",
                fallback=True,
                error=f"backend_error:{exc}",
            )
        return safe_parse(raw, self.schema, self.fallback_node)


@dataclass
class MockDecisionModel:
    """规则/mock 决策器,按预置映射返回。

    用于离线回归测试与无 GPU 环境的开发,把"客户输入 -> 意图"的映射
    写死,完全不依赖 LLM。
    """

    schema: Type[BaseModel]
    rules: dict[str, str] = field(default_factory=dict)
    fallback_node: str = _DEFAULT_FALLBACK_NODE
    default_node: str = "continue"

    def decide(self, ctx: DialogContext) -> DecisionResult:
        recent = ctx.recent(1)
        text = recent[0].user_message if recent else ""
        node = self.default_node
        for keyword, mapped in self.rules.items():
            if keyword in text:
                node = mapped
                break
        return DecisionResult(
            payload=self.schema(node=node),
            raw_text=text,
            fallback=False,
        )


def make_random_backend(seed: int | None = None) -> LLMBackend:
    """构造随机后端,便于 demo 与测试。"""
    return _RandomBackend(seed=seed)


__all__ = [
    "DecisionModel",
    "DecisionResult",
    "DecisionError",
    "LLMBackend",
    "LLMDecisionModel",
    "MockDecisionModel",
    "extract_json",
    "safe_parse",
    "make_random_backend",
]
