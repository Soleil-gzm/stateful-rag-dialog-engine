"""
stateful_rag_dialog_engine
==========================

面向任务型对话的轻量框架:状态机 + 决策层 + (后续)RAG 话术层。

核心抽象:
- ``StateMachine``: 转移表驱动的对话状态机,替代硬编码 if/elif。
- ``DecisionModel``: 决策器协议,可插拔 LLM / 规则 / 小模型后端。
- ``DialogContext``: 贯穿一次会话的运行时上下文。

详见 ``stateful_rag_dialog_engine.core``。
"""

from .core.types import (
    DialogContext,
    DialogTurn,
    State,
    Transition,
    TriggerKind,
)
from .core.state_machine import (
    EXCEED_REPEAT_EVENT,
    StateMachine,
    StateMachineError,
    build_from_mappings,
)
from .core.decision_model import (
    DecisionError,
    DecisionModel,
    DecisionResult,
    LLMBackend,
    LLMDecisionModel,
    MockDecisionModel,
    extract_json,
    make_random_backend,
    safe_parse,
)

__version__ = "0.1.0"

__all__ = [
    # types
    "DialogContext",
    "DialogTurn",
    "State",
    "Transition",
    "TriggerKind",
    # state machine
    "StateMachine",
    "StateMachineError",
    "build_from_mappings",
    "EXCEED_REPEAT_EVENT",
    # decision
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
