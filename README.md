# ApexLogic Deep Research Multi-Agent

基于 `LangGraph` 的深度研究多智能体系统，采用 `Researcher -> Writer -> Reviewer` 循环流程，支持 BGE 语义检索筛选、结构化评审与多格式报告导出。

## 1. 项目目标

- 输入一个研究主题，自动完成检索、写作、评审与迭代修订。
- 输出两类报告：`user`（面向读者）与 `debug`（可观测执行过程）。
- 在 `both` 模式下额外导出 BGE 检索明细 JSON，便于离线分析召回与重排质量。

## 2. 当前架构（目录与职责）

- `core/state.py`：定义全局状态 `ResearchState` 与初始化函数。
- `core/graph.py`：构建 `StateGraph`，配置节点与条件路由。
- `agents/researchers.py`：检索代理，负责查询词、广搜、去重、BGE 两阶段筛选。
- `agents/writer.py`：写作代理，基于上下文与评审反馈生成/修订草稿。
- `agents/reviewer.py`：评审代理，优先 LLM 结构化评审，失败回退规则评审。
- `bge/retriever.py`：向量召回（粗筛，默认 top20）。
- `bge/reranker.py`：重排序（精排，默认 top10）。
- `tools/search_tool.py`：DDG + Tavily 搜索封装。
- `tools/arxiv_tool.py`：ArXiv 检索封装（含多轮回退查询策略）。
- `prompts/system_prompts.py`：三类 Agent 的系统提示词与用户提示词构造。
- `main.py`：命令行运行入口（打印核心状态与 trace）。
- `export_report.py`：报告导出入口（`user/debug/both/user_only`）。
- `tools/arxiv_synonyms.sample.json`：ArXiv 同义词词表示例。

## 3. 全流程链路说明

一次完整请求按以下步骤执行：

1. **初始化状态**
   - 使用 `create_initial_state(topic, output_mode)` 创建初始状态。
   - 关键字段包含：`topic`、`search_queries`、`retrieved_context`、`draft`、`review_result`、`execution_trace`、`errors` 等。

2. **Researcher 节点（检索与筛选）**
   - 先生成基础查询词；如有 `critique_feedback` / `revision_directives`，会补充定向检索词。
   - 若配置了 DeepSeek，优先做查询词重写（失败自动回退规则查询词）。
   - 按每轮总配额广搜（默认）：
     - `DDG_TOTAL_RESULTS=35`
     - `ARXIV_TOTAL_RESULTS=20`
     - `TAVILY_TOTAL_RESULTS=5`
   - 三路结果统一去重、标准化后，进入 BGE 两阶段筛选：
     - Retriever：top20
     - Reranker：top10
   - 最终上下文写入 `retrieved_context`，并连续编号 `citation_id`（`S1...`）。

3. **Writer 节点（生成/修订草稿）**
   - 消费 `retrieved_context` 与 `critique_feedback`。
   - 生成结构化研究草稿（摘要、背景、关键发现、风险局限、结论建议）。
   - 迭代时基于 `revision_directives.must_fix` 做定向修订。
   - 输出 `feedback_paragraph_mapping`（问题到段落映射）用于 debug 可追踪。

4. **Reviewer 节点（结构化评审）**
   - 评审维度：事实、逻辑、信息缺口、争议点与证据裁决。
   - 期望输出结构化 JSON：`is_satisfactory`、`needs_more_research`、`fact_issues`、`logic_issues`、`info_gaps` 等。
   - 如果模型输出不可解析，自动回退到规则评审，保证流程不中断。

5. **条件路由与循环终止**
   - `is_satisfactory=True` -> `END`
   - 或 `revision_step >= MAX_REVISIONS` -> `END`
   - 否则按 `next_route` 回到 `researcher` 或 `writer`。

6. **导出阶段**
   - `user`：正文 + 参考文献。
   - `debug`：元信息 + 正文 + 评审 + 历史 + 执行轨迹 + 错误。
   - `both`：一次运行输出 `user/debug` 两份 Markdown，并额外输出 BGE 明细 JSON。

## 4. 安装与环境配置

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 配置 `.env`

复制 `.env.example` 为 `.env`，至少配置：

- `DEEPSEEK_API_KEY`
- `BGE_EMBED_API_KEY`
- `BGE_RERANK_API_KEY`

可选：

- `TAVILY_API_KEY`（建议配置，提升网页质量）
- `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com/v1`）
- `DEEPSEEK_MODEL`（默认 `deepseek-chat`）
- `MAX_REVISIONS`（默认 `3`）
- `SEARCH_QUERY_BUDGET`（默认 `3`）
- `DDG_TOTAL_RESULTS` / `ARXIV_TOTAL_RESULTS` / `TAVILY_TOTAL_RESULTS`（默认 `35/20/5`）
- `BGE_RETRIEVER_TOP_K` / `BGE_RERANKER_TOP_K`（默认 `20/10`）
- `ARXIV_SYNONYM_FILE`（可选，同义词词典文件路径）

兼容变量：

- 若未设置 `*_TOTAL_RESULTS`，会自动读取旧变量
  `DDG_RESULTS_PER_QUERY` / `ARXIV_RESULTS_PER_QUERY` / `TAVILY_RESULTS_PER_QUERY`。

## 5. 常用运行命令

### 5.1 终端运行（查看状态与 trace）

```bash
python main.py --topic "多智能体系统在科研自动化中的应用" --output-mode debug
```

### 5.2 仅导出用户版

```bash
python export_report.py --topic "RISC-C和RISC-V架构的异同点" --output-mode user
```

### 5.3 仅导出调试版

```bash
python export_report.py --topic "RISC-C和RISC-V架构的异同点" --output-mode debug
```

### 5.4 一次运行导出 user + debug + BGE JSON（推荐）

```bash
python export_report.py --topic "OPPO FIND X8 ULTRA和iPhone17 pro max的性能对比" --output-mode both --output reports/oppo-vs-iphone.md
```

将生成：

- `reports/oppo-vs-iphone-user.md`
- `reports/oppo-vs-iphone-debug.md`
- `reports/oppo-vs-iphone-bge-details.json`

### 5.5 仅对外导出 user（内部仍跑完整链路）

```bash
python export_report.py --topic "OPPO FIND X8 ULTRA和iPhone17 pro max的性能对比" --output-mode user_only --output reports/oppo-vs-iphone-user.md
```

## 6. 输出文件说明

### 6.1 `*-user.md`

- 面向最终读者。
- 包含研究正文与参考文献。

### 6.2 `*-debug.md`

- 面向调试与评估。
- 包含运行元信息、评审反馈、迭代历史、执行轨迹、错误记录。
- 不包含 BGE 全量逐条明细（这些放在 JSON）。

### 6.3 `*-bge-details.json`（仅 `both` 模式）

- 包含本轮 BGE 过程的结构化明细：
  - 广搜配额统计（targets/attempted/fetched）
  - Retriever 记录（selected/dropped）
  - Reranker 记录（selected/dropped）
  - 每条记录含 `query/url/title/score/reason/timestamp` 等字段

示例片段：

```json
{
  "rank": 1,
  "title": "中美GDP差距再次缩小！25年中国GDP达20万亿美元，占美国 ... - 网易",
  "source": "tavily",
  "url": "https://www.163.com/dy/article/KJUUQ2L5055651K3.html",
  "score": 0.780851,
  "reason": "selected_top_k",
  "timestamp": "2026-03-19T15:28:21",
  "query": "2025年中国和美国GDP细分领域对比\n年中国与美国GDP构成预测：消费、投资、净出口占比对比\n中美产业结构对比 2025：制造业、服务业、数字经济增加值\n年中美GDP细分领域增长驱动力分析：科技创新与投资"
}
```

## 7. ArXiv 检索机制（通用增强版）

`tools/arxiv_tool.py` 采用通用多轮回退查询，避免中文或混合 query 直接 0 命中：

1. 原始 query
2. token 精简 query
3. 中英通用意图词同义词扩展 query
4. 英文 token-only query

可选外部词库：

- 设置 `ARXIV_SYNONYM_FILE=/path/to/your_synonyms.json`
- 可参考 `tools/arxiv_synonyms.sample.json`

## 8. 容错与降级策略

- 任一外部 API 调用失败，错误会记录到 `errors`，流程尽量继续。
- 未配置 LLM Key 时，Writer/Reviewer 启用本地回退逻辑，保证图可运行。
- Reviewer 输出非 JSON 时，会做解析修复与规则化兜底，避免链路中断。
- 广搜三路配额若都被配置成 0，会自动回退默认配额。

## 9. 调试建议

- 先看 `debug` 报告中的：
  - `BGE Provider 统计`
  - `BGE Retriever/Reranker` 输入输出条数
  - `错误与降级记录`
- 再看 `*-bge-details.json`：
  - 检查 `selected_records` 是否主题相关
  - 对比 `dropped_records` 与 `selected_records` 的分数分布

## 10. 快速自检

```bash
python main.py --topic "多智能体系统中的反思机制与自我优化" --output-mode debug
python export_report.py --topic "评测agent性能的几种常见benchmark概述与比较" --output-mode both --output reports/smoke.md
```

## 11. 安全提示

- 不要把真实密钥提交到仓库。
- `.env.example` 只放占位值。
- 建议将报告与日志输出目录纳入版本管理策略（如按需 `.gitignore`）。

