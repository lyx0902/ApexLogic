# ApexLogic Deep Research Multi-Agent

基于 `LangGraph` 的深度研究多智能体系统，采用 `Researcher -> Writer -> Reviewer` 循环流程，支持 MAB 自适应检索预算、图扩展查询、BGE 语义精排、四维量化评审与多格式报告导出。

## 1. 项目目标

- 输入一个研究主题，自动完成检索、写作、评审与迭代修订。
- 输出两类报告：`user`（面向读者）与 `debug`（可观测执行过程）。
- 在 `both` 模式下额外导出 BGE 检索明细 JSON，便于离线分析召回与重排质量。

## 2. 当前架构（目录与职责）

- `core/state.py`：定义全局状态 `ResearchState` 与初始化函数。
- `core/graph.py`：构建 `StateGraph`，配置节点与条件路由。
- `agents/researchers.py`：检索代理，负责查询词生成、MAB 预算分配、广搜、去重、图扩展、BGE 两阶段筛选。
- `agents/writer.py`：写作代理，基于上下文与评审反馈生成/修订草稿。
- `agents/reviewer.py`：评审代理，四维量化打分（S1-S4），优先 LLM 评审，失败回退规则评审。
- `bge/retriever.py`：向量召回（粗筛，默认 top20）。
- `bge/reranker.py`：重排序（精排，默认 top10）。
- `optim/mab_search.py`：Thompson Sampling MAB，自适应分配三路检索预算。
- `optim/graph_expand.py`：图扩展查询，从已检索文档构建共现图，补搜核心概念方向。
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
   - 关键字段包含：`topic`、`search_queries`、`retrieved_context`、`draft`、`review_result`、`execution_trace`、`errors`、`mab_state` 等。

2. **Researcher 节点（检索与筛选）**
   - 先生成基础查询词；如有 `critique_feedback` / `revision_directives`，会补充定向检索词。
   - 若配置了 DeepSeek，优先做查询词重写（失败自动回退规则查询词）。
   - **MAB 自适应预算**：根据各来源历史表现（Thompson Sampling），动态调整本轮三路检索配额（默认基础配额 DDG=35 / ArXiv=20 / Tavily=5）。
   - 三路广搜结果统一去重、标准化。
   - **图扩展查询**：从去重后的候选文档提取核心共现概念，生成补充查询，用 DDG 补搜后合并回候选池（通过环境变量 `GRAPH_EXPAND_QUERIES` 控制扩展条数，默认 2）。
   - 进入 BGE 两阶段筛选：
     - Retriever：top20
     - Reranker：top10
   - MAB 依据最终精排结果更新各来源的奖励参数，供下轮参考。
   - 最终上下文写入 `retrieved_context`，并连续编号 `citation_id`（`S1...`）。

3. **Writer 节点（生成/修订草稿）**
   - 消费 `retrieved_context` 与 `critique_feedback`。
   - 生成结构化研究草稿（摘要、背景、关键发现、风险局限、结论建议）。
   - 迭代时基于 `revision_directives.must_fix` 做定向修订。
   - 输出 `feedback_paragraph_mapping`（问题到段落映射）用于 debug 可追踪。

4. **Reviewer 节点（四维量化评审）**
   - 将 top-10 原始来源文本传给 LLM，逐条核查引用是否有真实依据。
   - 四个评审维度：
     - S1 事实准确性（权重 35%）
     - S2 逻辑完整性（权重 25%）
     - S3 信息覆盖广度（权重 25%）
     - S4 结论可执行性（权重 15%）
   - 加权总分 ≥ 8.0 视为通过（可通过 `REVIEWER_PASS_THRESHOLD` 调整）。
   - 输出结构化 JSON：`scores`、`weighted_score`、`is_satisfactory`、`needs_more_research`、`fact_issues`、`logic_issues`、`info_gaps`、`citation_checks`、`supporter`、`skeptic`、`evidence_verdicts` 等。
   - 如果模型输出不可解析，自动回退到规则评审，保证流程不中断。

5. **条件路由与循环终止**
   - `is_satisfactory=True` -> `END`
   - 或 `revision_step >= MAX_REVISIONS` -> `END`
   - 否则按 `next_route` 回到 `researcher` 或 `writer`。

6. **导出阶段**
   - `user`：正文 + 参考文献。
   - `debug`：元信息 + 评审（四维评分表、引用核查、支持/质疑观点）+ 迭代历史 + 执行轨迹 + 错误记录 + MAB 预算分配记录 + 图扩展查询摘要。
   - `both`：一次运行输出 `user/debug` 两份 Markdown，并额外输出 BGE 明细 JSON（含图扩展数据）。

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
- `GRAPH_EXPAND_QUERIES`（默认 `2`，设为 `0` 可关闭图扩展）
- `REVIEWER_PASS_THRESHOLD`（默认 `8.0`）
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

- 面向调试与评估，包含以下各节：
  1. 运行元信息（迭代轮次、评审模式、BGE 统计）
  2. 评审结果（四维评分表、综合反馈、事实/逻辑问题、引用核查、支持与质疑观点）
  3. 每轮草稿与评审历史
  4. 参考上下文摘录
  5. 执行轨迹
  6. 错误与降级记录
  7. MAB 自适应检索预算（各来源 α/β 参数、逐轮预算对比）
  8. 图扩展查询（扩展查询列表、新增文档数、合并后总数）

### 6.3 `*-bge-details.json`（仅 `both` 模式）

- 包含本轮检索过程的结构化明细：
  - 广搜配额统计（targets/attempted/fetched）
  - MAB 本轮预算分配与历史参数
  - 图扩展查询及对应检索到的原始文档（`graph_expand.extra_contexts`）
  - Retriever 记录（selected/dropped）
  - Reranker 记录（selected/dropped）
  - 每条记录含 `query/url/title/score/reason/timestamp` 等字段

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
- `networkx` 未安装或候选文档为空时，图扩展模块静默跳过，不影响主流程。

## 9. 调试建议

- 先看 `debug` 报告中的：
  - `BGE Provider 统计`
  - `BGE Retriever/Reranker` 输入输出条数
  - `MAB 自适应检索预算` 各来源期望奖励趋势
  - `图扩展查询` 是否生成了有效补充方向
  - `错误与降级记录`
- 再看 `*-bge-details.json`：
  - 检查 `selected_records` 是否主题相关
  - 对比 `dropped_records` 与 `selected_records` 的分数分布
  - 查看 `graph_expand.extra_contexts` 评估图扩展文档质量

## 10. 快速自检

```bash
python main.py --topic "多智能体系统中的反思机制与自我优化" --output-mode debug
python export_report.py --topic "评测agent性能的几种常见benchmark概述与比较" --output-mode both --output reports/smoke.md
```

## 11. 安全提示

- 不要把真实密钥提交到仓库。
- `.env.example` 只放占位值。
- 建议将报告与日志输出目录纳入版本管理策略（如按需 `.gitignore`）。
