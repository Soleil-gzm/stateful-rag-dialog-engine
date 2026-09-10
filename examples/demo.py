"""
可运行 demo:状态机 + 随机决策后端,零外部依赖即可跑通闭环。

运行方式(在该目录下)::

    cd stateful-rag-dialog-engine
    PYTHONPATH=. python examples/demo.py

输出会演示一次完整会话:从 greet 推进到 end,期间通过决策器触发若干转移。
"""

from __future__ import annotations

import json
import random
import sys

from pydantic import BaseModel

from stateful_rag_dialog_engine import (
    DialogContext,
    DialogTurn,
    LLMBackend,
    LLMDecisionModel,
    MockDecisionModel,
    State,
    StateMachine,
    Transition,
    TriggerKind,
)


class Intent(BaseModel):
    """决策器输出 schema,与原项目 ``State`` 等价但语义中性。"""

    node: str


def build_demo_machine() -> StateMachine:
    """构造 demo 状态机:七个状态 + 兜底转移,演示 max_repeat 与条件转移。"""
    states = [
        State(name="greet"),
        State(name="verify", max_repeat=2),
        State(name="introduce", max_repeat=2),
        State(name="confirm", max_repeat=3),
        State(name="objection", max_repeat=3),
        State(name="faq"),
        State(name="handoff", terminal=True),
        State(name="end", terminal=True),
    ]
    transitions = [
        # greet -> verify
        Transition("greet", "verify", event="proceed"),
        # verify 分流
        Transition("verify", "handoff", event="complaint"),
        Transition("verify", "end", event="voicemail"),
        Transition("verify", "introduce", event="proceed"),
        # verify 兜底:超 repeat 上限自动收尾
        Transition("verify", "end", event="exceed_repeat"),
        # introduce
        Transition("introduce", "confirm", event="willing"),
        Transition("introduce", "objection", event="unwilling"),
        Transition("introduce", "handoff", event="complaint"),
        # confirm
        Transition("confirm", "objection", event="no"),
        Transition("confirm", "faq", event="question"),
        Transition("confirm", "end", event="yes"),
        Transition("confirm", "end", event="exceed_repeat"),
        # objection
        Transition("objection", "confirm", event="yes"),
        Transition("objection", "faq", event="question"),
        Transition("objection", "end", event="exceed_repeat"),
        # faq 回流
        Transition("faq", "confirm", event="back"),
    ]
    sm = StateMachine(states=states, transitions=transitions, initial="greet")
    for warning in sm.validate():
        print(f"[warn] {warning}")
    return sm


class _DemoRandomBackend:
    """候选事件对齐 demo 转移表的随机后端。

    故意把 ``proceed`` 权重调高,让会话能往前推进而不是长期卡在首状态;
    其余事件按等概率出现,演示决策器输出不命中时引擎原地自增 repeat,
    最终触发 ``exceed_repeat`` 兜底。
    """

    EVENTS = [
        "proceed", "proceed", "proceed",   # 加权推进
        "willing", "unwilling", "yes", "no",
        "question", "complaint", "voicemail", "back",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs: object) -> str:
        node = self._rng.choice(self.EVENTS)
        return json.dumps({"node": node})


def run_with_random_backend() -> None:
    """用随机后端跑通完整会话,验证状态机与决策器协作。"""
    sm = build_demo_machine()
    ctx = sm.create_context()

    model = LLMDecisionModel(
        schema=Intent,
        backend=_DemoRandomBackend(seed=7),
        prompt_template="history:{history}\nschema:{schema}\nnode?",
    )

    print(f"initial state: {ctx.state}")
    max_turns = 30
    for _ in range(max_turns):
        if ctx.terminal_reached:
            break
        ctx.push_turn(DialogTurn(user_message="(用户发言)"))
        result = model.decide(ctx)
        moved = sm.tick(ctx, event=result.field("node", "proceed"))
        print(
            f"turn={ctx.turn_count} state={ctx.state} repeat={ctx.repeat} "
            f"node={result.field('node')} moved={moved} fallback={result.fallback}"
        )

    print(f"final: state={ctx.state} terminal={ctx.terminal_reached}")


def run_with_mock_backend() -> None:
    """规则决策器演示:不依赖 LLM,确定性输出,便于回归测试。"""
    sm = build_demo_machine()
    ctx = sm.create_context()

    mock = MockDecisionModel(
        schema=Intent,
        rules={"问题": "question", "不": "no", "好": "yes", "投诉": "complaint"},
        default_node="proceed",
    )

    print("\n[mock] initial:", ctx.state)
    scripted_inputs = ["你好", "是我", "介绍一下", "费用多少", "不要了", "投诉"]
    for utterance in scripted_inputs:
        if ctx.terminal_reached:
            break
        ctx.push_turn(DialogTurn(user_message=utterance))
        result = mock.decide(ctx)
        sm.tick(ctx, event=result.field("node", "proceed"))
        print(f"  in={utterance!r} -> node={result.field('node')} state={ctx.state}")
    print("[mock] final:", ctx.state)


if __name__ == "__main__":
    run_with_random_backend()
    run_with_mock_backend()
    sys.exit(0)
