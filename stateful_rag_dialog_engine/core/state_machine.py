"""
状态机引擎。

核心设计:
- 转移表驱动:业务方声明 ``list[Transition]``,引擎按规则匹配,避免
  原项目中 ``runLLM`` 内近 190 行 ``if/elif state==...`` 的圈复杂度灾难。
- 钩子:``on_enter``/``on_exit`` 在状态切换前后执行,业务副作用与
  流控逻辑解耦。
- 轮数上限:每个状态可配置 ``max_repeat``,超出后触发 ``exceed_repeat``
  事件,便于实现"连续 N 轮未推进则收尾"的兜底。
- 纯函数式核心:``tick`` 不做 IO,所有副作用通过钩子注入,易于单测。

引擎不感知 LLM、RAG、stdin/stdout,只关心状态与事件,因此可独立测试
与复用于非对话场景(如订单流转、工单状态机)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .types import (
    ActionFn,
    DialogContext,
    State,
    Transition,
    TriggerKind,
    evaluate_guard,
)

logger = logging.getLogger(__name__)

EXCEED_REPEAT_EVENT = "exceed_repeat"
"""轮数超限事件名,引擎在状态 ``max_repeat`` 超出时自动抛出。"""


class StateMachineError(RuntimeError):
    """状态机层级的可预期错误,如初始状态未注册、转移目标不存在。"""


@dataclass
class StateMachine:
    """状态机引擎。

    用法::

        states = [State(name="greet"), State(name="end", terminal=True)]
        transitions = [
            Transition(source="greet", target="end", event="bye"),
        ]
        sm = StateMachine(states=states, transitions=transitions, initial="greet")
        ctx = sm.create_context()
        sm.tick(ctx, event="bye")
        assert ctx.state == "end" and ctx.terminal_reached
    """

    states: list[State]
    transitions: list[Transition]
    initial: str

    _by_name: dict[str, State] = field(default_factory=dict, init=False, repr=False)
    _outgoing: dict[str, list[Transition]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.states:
            raise StateMachineError("states must be non-empty")
        for s in self.states:
            if s.name in self._by_name:
                raise StateMachineError(f"duplicate state name: {s.name!r}")
            self._by_name[s.name] = s
            self._outgoing.setdefault(s.name, [])
        if self.initial not in self._by_name:
            raise StateMachineError(
                f"initial state {self.initial!r} not registered"
            )
        for t in self.transitions:
            self._validate_transition(t)
            self._outgoing[t.source].append(t)

    def _validate_transition(self, t: Transition) -> None:
        if t.source not in self._by_name:
            raise StateMachineError(
                f"transition source {t.source!r} not registered"
            )
        if t.target not in self._by_name:
            raise StateMachineError(
                f"transition target {t.target!r} not registered"
            )

    # ---- 工厂 -------------------------------------------------------------

    def create_context(self, slots: dict | None = None) -> DialogContext:
        """构造初始上下文,执行 initial 状态的 on_enter。"""
        ctx = DialogContext(state=self.initial, slots=dict(slots or {}))
        self._run_action(self._by_name[self.initial].on_enter, ctx)
        return ctx

    # ---- 查询 -------------------------------------------------------------

    @property
    def terminal_states(self) -> frozenset[str]:
        return frozenset(s.name for s in self.states if s.terminal)

    def is_terminal(self, ctx: DialogContext) -> bool:
        return self._by_name[ctx.state].terminal

    # ---- 驱动 -------------------------------------------------------------

    def tick(
        self,
        ctx: DialogContext,
        event: str | None = None,
    ) -> bool:
        """推进一次状态转移。

        触发顺序:
        1. 若当前状态轮数超限,优先触发 ``exceed_repeat`` 兜底转移;
        2. 若显式传入 event,匹配 EVENT 转移;
        3. 否则评估所有 CONDITION 转移,首个 guard 通过者生效;
        4. 未命中任何转移则原地保留,``repeat`` 自增。

        返回是否发生了状态切换。
        """
        if ctx.terminal_reached:
            return False

        matched = self._match(ctx, event)
        if matched is None:
            self._bump_repeat(ctx)
            return False
        self._apply(ctx, matched)
        return True

    def _match(self, ctx: DialogContext, event: str | None) -> Transition | None:
        # 1. 轮数超限兜底
        cur = self._by_name[ctx.state]
        if cur.max_repeat is not None and ctx.repeat >= cur.max_repeat:
            exc = self._find_event(ctx, EXCEED_REPEAT_EVENT)
            if exc is not None:
                logger.debug(
                    "state=%s repeat=%d exceeded, firing %s",
                    ctx.state, ctx.repeat, EXCEED_REPEAT_EVENT,
                )
                return exc

        # 2. 显式事件
        if event is not None:
            hit = self._find_event(ctx, event)
            if hit is not None:
                return hit

        # 3. 条件转移
        for t in self._outgoing.get(ctx.state, ()):
            if t.kind is not TriggerKind.CONDITION:
                continue
            if evaluate_guard(t.guard, ctx):
                return t
        return None

    def _find_event(self, ctx: DialogContext, event: str) -> Transition | None:
        for t in self._outgoing.get(ctx.state, ()):
            if t.kind is TriggerKind.EVENT and t.event == event:
                if evaluate_guard(t.guard, ctx):
                    return t
        return None

    def _apply(self, ctx: DialogContext, t: Transition) -> None:
        src = self._by_name[ctx.state]
        tgt = self._by_name[t.target]
        self._run_action(src.on_exit, ctx)
        self._run_action(t.action, ctx)
        ctx.state = t.target
        ctx.repeat = 0
        self._run_action(tgt.on_enter, ctx)
        if tgt.terminal:
            ctx.terminal_reached = True
        logger.debug("transition %s -> %s (event=%s)", t.source, t.target, t.event)

    def _bump_repeat(self, ctx: DialogContext) -> None:
        ctx.repeat += 1

    @staticmethod
    def _run_action(action: ActionFn | None, ctx: DialogContext) -> None:
        if action is None:
            return
        try:
            action(ctx)
        except Exception as exc:
            ctx.error = exc
            logger.exception("action raised, recorded in ctx.error")

    # ---- 迭代 -------------------------------------------------------------

    def iter_transitions(self, ctx: DialogContext) -> Iterator[Transition]:
        """遍历从当前状态出发的合法转移,供 UI/调试展示。"""
        yield from self._outgoing.get(ctx.state, ())

    def iter_all_transitions(self) -> Iterator[Transition]:
        yield from self.transitions

    # ---- 校验 -------------------------------------------------------------

    def validate(self) -> list[str]:
        """静态校验:检查孤立状态、无可达终态等,返回告警列表。"""
        warnings: list[str] = []
        reachable = self._reachable_states()
        for s in self.states:
            if s.name != self.initial and s.name not in reachable:
                warnings.append(f"state {s.name!r} unreachable from initial")
        if not self.terminal_states:
            warnings.append("no terminal state declared")
        return warnings

    def _reachable_states(self) -> set[str]:
        seen: set[str] = set()
        stack = [self.initial]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for t in self._outgoing.get(cur, ()):
                if t.target not in seen:
                    stack.append(t.target)
        return seen


def build_from_mappings(
    states: Iterable[tuple[str, ...] | dict],
    transitions: Iterable[tuple | dict],
    initial: str,
) -> StateMachine:
    """便捷构造器:从纯数据(元组/字典)构建引擎,便于 YAML 加载。

    ``states`` 每项可为 ``(name,)``、``(name, terminal)``、
    ``{"name": ..., "terminal": ..., "max_repeat": ...}``。
    ``transitions`` 每项可为 ``(source, target, event)`` 或字典。
    """
    state_objs: list[State] = []
    for item in states:
        if isinstance(item, dict):
            state_objs.append(State(**item))
        else:
            name = item[0]
            terminal = item[1] if len(item) > 1 else False
            state_objs.append(State(name=name, terminal=terminal))
    trans_objs: list[Transition] = []
    for item in transitions:
        if isinstance(item, dict):
            trans_objs.append(Transition(**item))
        else:
            trans_objs.append(
                Transition(source=item[0], target=item[1], event=item[2])
            )
    return StateMachine(
        states=state_objs, transitions=trans_objs, initial=initial
    )
