# stateful-rag-dialog-engine 架构说明

> 一句话：面向电话外呼场景的**决策式对话引擎**，状态机控制流程，RAG 从预设话术库中选最合适的一句。
> 与普通 LLM 聊天不同：话术不自由生成，而是从人工审核过的 QA 中检索 + 变量填充。

## 1. 系统组件

### 1.1 全局视图

```
┌──────────┐  HTTP/SSE   ┌────────────┐  stdin/stdout   ┌──────────────┐
│  前端    │ ◄─────────► │ Node.js    │ ◄─────────────► │ Python 引擎  │
│ (浏览器) │             │ index.js   │  一行 JSON 进    │ (状态机+RAG) │
└──────────┘             └────────────┘  两行 JSON 出   └──────────────┘
                                │
                                └─ 多 worker（每 GPU 一个），轮询分发
```

职责：

- **Node**：对外 HTTP/SSE、会话管理、worker 轮询、进程生命周期

- **Python**：无状态引擎，每次请求带完整 state，处理完写回新 state

### 1.2 Node.js 服务（index.js）

- 监听端口、HTTPS

- 两个 API：`POST /api/llm/generate`（提交）、`GET /api/llm/:chatId/:questionId/stream`（取流）

- `activeListeners` Map 的作用（key 是 `chatId_questionId`）

- worker 轮询（`getNextWorker`）

- 信号处理（SIGINT/SIGTERM 杀子进程）

**证据**：`index.js`，看注释掉的端口、`gpuIndices` 配置。### 1.3 Python 引擎
TODO: 单进程、stdin/stdout、状态机 + RAG、无状态（状态由前端传回）。

### 1.3 Python 引擎

- 入口 `__main__.py`，做三件事：加载参数/组件、构建 specs、跑 `Pathway.traverse`

- **无状态**：`state` / `repeat` / `check_count` 都从前端每次请求带进来，Python 不保存会话

- 状态更新通过 `<END_OF_STREAMING_SIGNAL>` 行回传给 Node

## 2. 通信协议（Node ↔ Python）

### 2.1 Node → Python（请求）

```json
{
  "task_id": "13827",
  "chatHistory": {
    "history": [
      {"role": "user", "message": "喂"},
      {"role": "assistant", "message": "您好"}
    ],
    "state": 1,
    "node": "continue",
    "repeat": 0,
    "check_count": -1,
    "query_node": "continue"
  },
  "prompt": { "jobnumber": "...", "info_name": "...", "quota": "...", ... }
}
```

- `task_id`：本次请求唯一 ID，Node 用它匹配 listener

- `chatHistory.history`：完整对话历史（最后一条是空 assistant 占位）

- `chatHistory.state`：当前状态（1~7）

- `chatHistory.repeat`：当前 state 内已重复次数

- `chatHistory.check_count`：信息问题计数

- `chatHistory.node`：`continue` / `stop`

- `prompt`：客户画像，用于话术变量填充

### 2.2 Python → Node（响应）

```json
{"task_id": "13827", "response": "那您可以直接在微信找到苏宁任性花小程序，首页点击借钱申请..."}
{"task_id": "13827", "response": "<END_OF_STREAMING_SIGNAL>state|node|repeat|check_count|query_node"}
```

**关键约束**：

- 每行是**独立、完整**的 JSON，以 `\n` 结尾并 `flush`

- 顺序固定：先话术、后信号

- 信号行**必须**存在（前端状态机靠它推进）

### 2.3 Node → 前端（SSE）

**写什么**：SSE 事件格式。

**证据**：`index.js` 里 `res.write(...)` 那两处。

- 话术：`data: <话术>\n\n`

- 结束：`event: end\ndata: state@ X | node@ Y | repeat@ Z | check_count@ N | query_node@ M\n\n`

**注意**：字段名带 `@` 和空格是前端约定的，别改。

### 2.4 协议约束

 stdout 里不能有 print，日志只能走 stderr。

- 外部系统（Node）**逐行**解析 Python 的 stdout

- 任何非 JSON 行（调试 print、警告）都会破坏协议

- 因此：**日志走 stderr**，`print` 一律禁用（`corpus/` 除外，它是独立 CLI 工具）

- 对应代码：`utils/logger.py` 的 `setup_logging` 把 handler 绑到 `sys.stderr`

## 3. 对话状态机

### 3.1 状态清单

|state|模块|话术来源|max_repeat|
|---|---|---|---|
|1|核实|确认身份|1|
|2|产介|产品介绍|1|
|3|三方|非本人接听|0|
|4|确认|意愿确认|1|
|5|答疑|异议处理|1|
|6|投诉|投诉倾向|0|
|7|留言|语音留言|0|

### 3.2 状态转移图

```
state=1 repeat=0  ──生成核实话术──► repeat=1
state=1 repeat=1  ──调 tracker_check──►
     ├─ nonidentity ──► state=3
     ├─ message     ──► state=7
     └─ checked     ──调 tracker_willing──►
            ├─ cc       ──► state=6
            └─ 其他     ──► state=2

state=2 repeat=0  ──生成产介话术──► repeat=1
state=2 repeat=1  ──调 tracker_check──►
     ├─ nonidentity ──► state=3
     ├─ message     ──► state=7
     └─ checked     ──调 tracker_willing──►
            ├─ cc    ──► state=6
            ├─ yes   ──► state=4
            └─ 其他  ──► state=5

state=4  ──调 tracker_willing──►
     ├─ cc   ──► state=6
     ├─ no   ──► state=5
     └─ 其他 ──► repeat += 1

state=5  ──调 tracker_willing──►
     ├─ cc   ──► state=6
     └─ 其他 ──► repeat += 1
```

### 3.3 repeat 与终止条件

- 每个 state 有 `max_repeat`

- 当 `repeat > max_repeat` 时，写固定兜底话术，不再生成

- `check_count > 5` 时走另一条客服兜底

- `node == 'stop'` 且 `query == "。"` 时也走兜底

**证据**：`runLLM` 顶部的三个 `if / elif`。

## 4. 话术模块与条件组合

### 4.1 模块文件命名规则

- 无条件：`output_模块1-确认身份_1.txt`

- 有条件：`output_模块4-意愿确认_count1_combo1_营销优惠卖点非空_额度非空_预计借款利率区间非空.txt`

对应 FAISS 目录：`build/db_saves-condition/output_.../`

### 4.2 条件维度与 combo

|combo|营销优惠卖点|额度|利率区间|
|---|---|---|---|
|1|非空|非空|非空|
|2|非空|非空|空|
|3|非空|空|非空|
|4|非空|空|空|
|5|空|非空|非空|
|6|空|非空|空|
|7|空|空|非空|
|8|空|空|空|

**证据**：`modules/specs.py` 的 `build_all_specs`。

### 4.3 selector 逻辑

1. 看 `sellingpoint` / `quota` / `interest` 是否为空字符串

2. 拼出 `combo_str`

3. 对"确认"和"答疑"两类，筛出前缀含 `combo_str` 的 spec

4. 按文件名里的 `count` 排序（决定 `repeat` 索引对应哪个文件）

**证据**：`modules/selector.py`。

## 5. 检索层（RAG）

### 5.1 两路检索

BM25 和 FAISS 各自的输入、输出。

**证据**：`retrieval/rag.py` + `modules/module.py` 的 `generate_rag`。

**注意坑**：

- BM25 的 `texts` 来自 `RecursiveCharacterTextSplitter` 切分

- FAISS 是离线构建好的，直接 `load_local`

- 两边**文档粒度可能不一致**——这是已知风险（RRF 用文本内容做 key）

### 5.2 RRF 融合

为什么用 RRF，m_vector / m_text 的作用。

**证据**：`retrieval/rag.py` 的 `rrf` 函数。

**参考**：

- RRF 用于把**两个异构排序**融合成一个

- 公式：`score = Σ 1 / (rank + m)`

- `m_vector=20` / `m_text=100` 是两路的平滑参数，值越大越"公平"

- 取 top-5 后**只取第 1 条**（原代码 `question = rrf_res[0]`）

### 5.3 答案选取与变量填充

从 question 到最终 response。

**参考**：

1. `self.questions.index(question)` 找到下标

2. 从 `self.answers[i]` 取 `Answer: xxx`

3. 从 `self.labels[i]` 取 `Label: xxx`（如"信息问题"）

4. `load_and_format_prompt(answer, case_info)` 把 `{姓名}` `{额度}` 等占位符替换

5. 返回 `(response, label)`

**注意坑**：`self.questions.index(question)` 如果 RRF 返回的 question 与列表里的不完全一致会 `ValueError`。已知风险，写进第 10 节。

## 6. 追踪模型（Tracker）

### 6.1 现状：_MockTracker

```python
class _MockTracker:
    def main(self, *a, **kw):
        return {"node": "unknown"}
```

**影响**：state=1/2/4/5 的 tracker 调用永远返回 unknown，**永远不跳转到 state=3/6/7**。功能层完全失效。

### 6.2 目标架构

- `QueryTracker` / `AssistantTracker` 各自的 prompt

- `query_check` 返回 `checked` / `nonidentity` / `message`

- `willing` 返回 `yes` / `no` / `cc`

- 生成流程：build_prompt → model.generate → 解析 JSON → 返回 node

**证据**：`tracking/trackers.py` + 你贴的 `prompt_check.txt` / `prompt_willing.txt`。

### 6.3 依赖清单

TODO: 需要哪些资源（TRT-LLM 引擎 / HF 权重 / prompt 文件）。

## 7. 数据与配置

### 7.1 目录结构

```json
stateful-rag-dialog-engine/
├── config/args/         运行参数（qwen.json 等）
├── datas/               原始数据
├── models/              模型权重（bge-large-zh-v1.5）
├── build/               构建产物（txt-condition / db_saves-condition）
├── Log/                 运行日志（自动生成，gitignore）
├── test/                golden test
├── stateful_rag_dialog_engine/    Python 包
│   ├── __main__.py      入口
│   ├── config/          参数 + 模板
│   ├── dialogue/        状态机
│   ├── modules/         话术模块 + spec + selector + registry
│   ├── generation/      prompt 构建 + 变量填充
│   ├── retrieval/       BM25 + FAISS + RRF
│   ├── tracking/        QueryTracker / AssistantTracker
│   ├── corpus/          语料构建工具链
│   └── utils/           日志 + 金额转中文
└── index.js             Node 服务
```

### 7.2 关键配置

 config/args/qwen.json
 config/corpus.yaml    数据预处理

### 7.3 数据流：从 Excel 到 FAISS

```
Excel → corpus/excel_reader → corpus/txt_writer → build/txt-condition/*.txt
                                                    ↓
                                         corpus/faiss_builder
                                                    ↓
                                     build/db_saves-condition/*/index.faiss
```

命令：`python -m stateful_rag_dialog_engine.corpus.pipeline config/corpus.yaml`

## 8. 部署与运行

### 8.1 前置依赖

- Python 环境依赖（`requirements.txt` 或 `pyproject.toml`）

- Node 依赖（`npm install`）

- 启动顺序：先生成 FAISS，再启 Node，Node 自己 spawn Python

**证据**：`index.js` 顶部的 `spawn("python3", ["run_module.py", ...])` 那段。

### 8.2 生成语料与索引

### 8.3 启动服务

### 8.4 环境变量

## 9. 测试

### 9.1 golden test 机制

- 输入：`test/test_input.jsonl`（每行一条请求）

- 输出：`test/golden.txt`（每条 2 行，话术 + 信号）

- 运行：`python test/run_tests.py`

- 更新：`python test/run_tests.py --update`

### 9.2 覆盖矩阵

### 9.3 如何新增用例

1. 编辑 `test/test_input.jsonl`，加一行 JSON

2. `python test/run_tests.py --update`

3. `git add test/`

## 10. 已知问题与 TODO

### 10.1 功能缺口

- `_MockTracker` 未接真模型

- 模型/引擎依赖未到位

### 10.2 已知 bug

- **`index.js` 行缓冲**：`data` 事件不保证按行切分，长话术可能丢首行（**如果你还没修，就写"待修"**）

- `self.questions.index(question)` 可能 `ValueError`

- `Pathway.traverse` 读到 EOF 时的处理（**已修则删，未修则写**）

### 10.3 未来优化方向

- 延迟加载 embeddings 模型（省 1 GB 内存）

- `corpus/` 里的 print 改为 logging（独立任务）

- 合并 `QueryTracker` / `AssistantTracker`（DRY，但优先级低）

## 附录 A：常见改动指引

|需求|改哪里|
|---|---|
|加一个 state|`dialogue/pathway.py` 的 `STATE_CONFIG` + `modules/specs.py` 的 `MODULE_CONFIGS` + `modules/selector.py` 的转移规则|
|加一个条件维度|`modules/specs.py` 的 `CONDITION_PAIRS` + `modules/selector.py`|
|换 RAG 策略|只改 `modules/module.py` 的 `generate_rag`|
|改话术|改 `build/txt-condition/*.txt`，重建 FAISS|
|调日志级别|环境变量 `LOG_LEVEL=DEBUG`|

## 附录 B：术语表

|术语|含义|
|---|---|
|state|对话阶段（1~7）|
|repeat|同一 state 内重复次数|
|node|控制节点（continue / stop），外部可干预|
|check_count|触发"信息问题"的次数|
|combo|3 对布尔条件组成的 8 种组合之一|
|spec|模块的轻量描述（只有 id，无内容）|
|registry|按需构造 + LRU 缓存的 Module 管理器|
|RRF|Reciprocal Rank Fusion，两路检索结果融合算法|

---
