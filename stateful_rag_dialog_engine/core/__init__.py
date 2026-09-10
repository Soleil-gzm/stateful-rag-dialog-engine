"""核心子包:状态机、决策层、共享类型。"""

from .types import (
    DialogContext,
    DialogTurn,
    State,
    Transition,
    TriggerKind,
    evaluate_guard,
    merge_slots,
)
from .state_machine import (
    EXCEED_REPEAT_EVENT,
    StateMachine,
    StateMachineError,
    build_from_mappings,
)
from .decision_model import (
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

__all__ = [
    "DialogContext",
    "DialogTurn",
    "State",
    "Transition",
    "TriggerKind",
    "evaluate_guard",
    "merge_slots",
    "StateMachine",
    "StateMachineError",
    "build_from_mappings",
    "EXCEED_REPEAT_EVENT",
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
