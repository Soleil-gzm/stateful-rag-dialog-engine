"""
核心数据结构。

本模块定义状态机与决策层共用的基础类型,刻意保持零业务耦合:
- State:状态节点描述
- Transition:状态转移规则
- DialogContext:贯穿一次对话的运行时上下文
- DialogTurn:单轮对话的输入/输出快照

业务方通过组装这些数据结构描述自身流程,无需修改引擎代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


GuardFn = Callable[["DialogContext"], bool]
"""守卫函数:返回 True 时允许转移执行。用于表达"仅当 repeat 达到上限才跳走"这类条件。"""

ActionFn = Callable[["DialogContext"], Any]
"""动作函数:转移触发时执行的副作用,如更新 repeat、写日志、调用外部分类器。"""


class TriggerKind(Enum):
    """触发转移的事件类型。"""

    EVENT = "event"
    """显式事件,由业务代码或决策器抛出。"""

    CONDITION = "condition"
    """条件触发,每次 tick 评估所有 CONDITION 转移。"""

    TERMINAL = "terminal"
    """终态标记,进入后不再转移。"""


@dataclass(frozen=True)
class State:
    """状态节点。

    一个状态对应业务流程中的一个阶段(如核实身份、产品介绍、异议处理)。
    引擎本身不关心状态语义,只负责按转移表驱动。
    """

    name: str
    terminal: bool = False
    on_enter: ActionFn | None = field(default=None, repr=False)
    on_exit: ActionFn | None = field(default=None, repr=False)
    max_repeat: int | None = None
    """该状态内允许的最大轮数;超过后引擎会触发 ``exceed_repeat`` 事件。"""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("State name must be non-empty")


@dataclass(frozen=True)
class Transition:
    """状态转移规则。

    一条转移 = ``source --[event/guard]--> target``,附带可选 action。
    设计为不可变以便在多线程下共享、便于单测断言。
    """

    source: str
    target: str
    event: str | None = None
    guard: GuardFn | None = field(default=None, repr=False)
    action: ActionFn | None = field(default=None, repr=False)
    kind: TriggerKind = TriggerKind.EVENT

    def __post_init__(self) -> None:
        if self.source == self.target and self.kind != TriggerKind.CONDITION:
            # 自环只在 CONDITION 场景下有意义(如"继续留在当前状态")
            raise ValueError(
                f"Self-transition on {self.source} requires kind=CONDITION"
            )
        if self.kind == TriggerKind.EVENT and not self.event:
            raise ValueError("EVENT transition requires an event name")


@dataclass
class DialogTurn:
    """单轮对话的输入/输出快照,用于状态机内部传递与日志回放。"""

    user_message: str = ""
    assistant_message: str = ""
    role: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogContext:
    """贯穿一次对话会话的运行时上下文。

    持有当前状态、轮数计数器、历史记录与业务自定义字段。引擎只读写
    其公共字段,业务可通过 ``slots`` 注入任意数据供决策器或 prompt 使用。
    """

    state: str
    repeat: int = 0
    history: list[DialogTurn] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    """业务自定义槽位,如客户信息、产品字段、条件组合键等。"""

    # 以下字段由引擎维护,业务只读
    turn_count: int = 0
    terminal_reached: bool = False
    error: Exception | None = None

    def push_turn(self, turn: DialogTurn) -> None:
        self.history.append(turn)
        self.turn_count += 1

    def recent(self, n: int = 2) -> list[DialogTurn]:
        """返回最近 n 轮,供决策器构造 prompt 时截取上下文。"""
        return self.history[-n:] if n > 0 else []

    def as_dict(self) -> dict[str, Any]:
        """扁平化用于日志/传输。"""
        return {
            "state": self.state,
            "repeat": self.repeat,
            "turn_count": self.turn_count,
            "terminal": self.terminal_reached,
            "slots": dict(self.slots),
        }


def evaluate_guard(guard: GuardFn | None, ctx: DialogContext) -> bool:
    """安全执行守卫函数;守卫抛异常时视为不通过并记录到 ctx.error。"""
    if guard is None:
        return True
    try:
        return bool(guard(ctx))
    except Exception as exc:
        ctx.error = exc
        return False


def merge_slots(ctx: DialogContext, extra: Mapping[str, Any] | None) -> None:
    """合并外部槽位到上下文,用于运行时注入业务字段。"""
    if extra:
        ctx.slots.update(extra)
