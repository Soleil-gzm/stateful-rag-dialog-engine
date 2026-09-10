# stateful-rag-dialog-engine

面向任务型对话的轻量框架:状态机 + 决策层 + (规划中)RAG 话术层。

## 动机

重构自一个生产对话后端,原项目存在以下问题,本框架逐项修复:

- **状态机硬编码**:近 190 行 `if/elif state==...` 嵌套,扩展一个状态要改多处。
- **决策器重复**:`AssistantTracker` 与 `QueryTracker` 几乎完全相同,只有 prompt 角色映射不同。
- **异常处理失效**:`except ValidationError` 从未 `import ValidationError`,真实异常会冒泡导致服务崩溃。
- **JSON 解析脆弱**:`text.find('{')..rfind('}')` 截取,模型输出多段 JSON 或嵌套花括号时拼出非法字符串。
- **路径硬编码**:绝对路径写死他人家目录,无法迁移。

## 架构

```
stateful_rag_dialog_engine/
├── core/
│   ├── types.py           # State / Transition / DialogContext / DialogTurn
│   ├── state_machine.py   # 转移表驱动的状态机引擎
│   └── decision_model.py  # DecisionModel 协议 + LLM/Mock 实现
└── examples/
    ├── business_config.yaml  # 脱敏业务描述示例
    └── demo.py               # 可运行 demo(零外部依赖)
```

### 核心抽象

- `StateMachine`:转移表驱动,业务方声明 `list[Transition]` 即可,避免硬编码分支。
- `DecisionModel`:决策器协议,可插拔 LLM / 规则 / 小模型后端,引擎只依赖协议。
- `DialogContext`:贯穿一次会话的运行时上下文,持有状态、轮数、历史、槽位。

### 设计原则

1. **纯函数式核心**:`tick` 不做 IO,所有副作用通过 `on_enter`/`on_exit`/`action` 钩子注入,易于单测。
2. **数据驱动**:状态与转移用数据结构描述,业务方无需改引擎代码即可增删状态。
3. **后端无关**:LLM 后端抽象为 `LLMBackend` 协议,TRT-LLM / vLLM / OpenAI 均可包装接入。
4. **容错优先**:决策器解析失败返回 schema 默认值并标记 `fallback`,不抛业务异常。

## 快速开始

```bash
cd stateful-rag-dialog-engine
PYTHONPATH=. python examples/demo.py
```

输出会演示一次完整会话:从 `greet` 推进到 `end`,期间通过决策器触发若干转移,并展示 `max_repeat` 超限兜底。

## 用法示例

```python
from pydantic import BaseModel
from stateful_rag_dialog_engine import (
    State, StateMachine, Transition, LLMDecisionModel, DialogContext, DialogTurn
)

class Intent(BaseModel):
    node: str

states = [
    State(name="greet"),
    State(name="verify", max_repeat=2),
    State(name="end", terminal=True),
]
transitions = [
    Transition("greet", "verify", event="proceed"),
    Transition("verify", "end", event="exceed_repeat"),
    Transition("verify", "end", event="yes"),
]
sm = StateMachine(states=states, transitions=transitions, initial="greet")

ctx = sm.create_context()
ctx.push_turn(DialogTurn(user_message="你好"))
# result = model.decide(ctx)  # 由决策器输出 Intent
# sm.tick(ctx, event=result.field("node", "proceed"))
```

## 后续规划

- [ ] RAG 话术层:Corpus 抽象 + Hybrid Retriever(BM25 + FAISS + RRF 融合)
- [ ] 模板填充器:schema 驱动,替代硬编码字段名
- [ ] 运行时层:stdin/HTTP/gRPC transport 抽象 + 组件生命周期管理
- [ ] 评估闭环:决策准确率、状态转移正确率、端到端完成率
- [ ] YAML 业务描述加载器

## 技术栈

- Python 3.11+
- pydantic v2(schema 校验与序列化)
- 标准库 only(状态机与决策器核心,LLM 后端按需引入)
