# ApexLogic Deep Research Multi-Agent

基于 LangGraph 的深度研究多智能体系统（Researcher / Writer / Reviewer）。

## 1) 安装

```bash
pip install -r requirements.txt
```

## 2) 配置环境变量

复制 `.env.example` 为 `.env`，填写 API Key：

- `DEEPSEEK_API_KEY`
- `TAVILY_API_KEY`（可选，仅在启用 Tavily 回退时需要）

可选参数：

- `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com/v1`）
- `DEEPSEEK_MODEL`（默认 `deepseek-chat`）
- `MAX_REVISIONS`（默认 `3`）
- `SEARCH_QUERY_BUDGET`（默认 `3`，每轮最多使用多少条查询词）
- `DDG_RESULTS_PER_QUERY`（默认 `10`）
- `ARXIV_RESULTS_PER_QUERY`（默认 `5`）
- `TAVILY_RESULTS_PER_QUERY`（默认 `3`）
- `MIN_SOURCE_QUALITY` / `MIN_SOURCE_SIGNAL` / `MIN_SOURCE_RELEVANCE` / `MIN_SOURCE_COMPOSITE`
- `FILTER_WEIGHT_QUALITY` / `FILTER_WEIGHT_SIGNAL` / `FILTER_WEIGHT_RELEVANCE`
- `MIN_CONTEXT_KEEP`（过滤后最少保留上下文数量）

> 安全建议：不要把真实密钥提交到 `.env.example`。真实密钥仅放在本地 `.env`。

## 3) 运行

```bash
python main.py --topic "多智能体系统在科研自动化中的应用" --report-length medium --output-mode debug
```

可选参数：

- `--report-length {short,medium,long}`：控制正文篇幅和细节粒度。
- `--output-mode {user,debug}`：控制导出呈现方式。

## 4) 当前实现说明

- `core/state.py`：全局状态定义（`ResearchState`）
- `core/graph.py`：LangGraph 工作流与条件路由
- `agents/researchers.py`：检索代理（阶梯式调用 `DuckDuckGo + ArXiv + Tavily`）
- `agents/writer.py`：主笔代理（DeepSeek 生成/修订草稿，消费带引用的上下文）
- `agents/reviewer.py`：评审代理（优先 DeepSeek 结构化 JSON 评审，失败回退规则评审）
- `tools/search_tool.py`：检索工具封装（DuckDuckGo 与 Tavily）
- `tools/arxiv_tool.py`：ArXiv 检索封装
- `prompts/system_prompts.py`：统一系统提示词与提示构造函数

## 5) 降级行为

在缺少依赖、无 API Key、或外部调用失败时，系统会自动降级到本地占位逻辑并记录到 `errors`，保证图流程可继续执行。

## 6) 执行流程（实际运行）

系统按如下顺序执行：

1. `researcher` 生成查询词（可选 LLM 重写）并按阶梯配额检索：
   - DuckDuckGo：每条 query 拉取约 10 条网页结果
   - ArXiv：每条 query 拉取约 5 条学术结果
   - Tavily：每条 query 拉取约 3 条高质量网页补充
2. Researcher 对上下文去重并分配 `citation_id`（如 `S1`, `S2`）。
3. `writer` 基于上下文与反馈生成草稿。
4. `reviewer` 输出结构化评审结果（`is_satisfactory`, `needs_more_research`, `critique_feedback`）。
5. 依据 `next_route` 路由到 `END` / `researcher` / `writer`，直到满意或达到最大迭代。

运行结束后，状态中可查看：

- `review_result`：结构化评审结果
- `execution_trace`：每个节点的执行轨迹摘要
- `errors`：降级与异常信息

其中 `review_result` 额外包含：

- `fact_issues`：事实性问题列表
- `logic_issues`：逻辑性问题列表
- `info_gaps`：信息缺口列表

解释：

- `fact_issues`：结论与证据不一致、引用不支持结论、事实可能错误。
- `logic_issues`：论证链条断裂、因果跳跃、结论无法由前文推出。
- `info_gaps`：当前检索上下文缺失关键材料，导致无法充分评估。

## 7) 导出完整报告（不改变 `main.py` 输出）

```bash
python export_report.py --topic "gemini 3.1pro和gpt5.3 codex的benchmark比较" --report-length long --output-mode debug
```

可选参数：

- `--max-revisions 1`
- `--output reports/custom-report.md`
- `--output-mode user`（仅输出最终报告正文）
- `--output-mode debug`（输出每轮草稿、评审、轨迹与错误）

## 8) 统一搜索策略（阶梯式配额）

当前默认三路搜索 API：

1. **DuckDuckGo Search**：无需 Key，免费大批量，负责广域网页检索。
2. **ArXiv API**：免费学术预印本检索。
3. **Tavily API**：高质量网页补充检索（建议控制配额）。

工程实现上，`agents/researchers.py` 会对每条 query 依次调用三类 provider，并在后处理阶段统一去重和筛选。

### 配额调参

- `SEARCH_QUERY_BUDGET`：控制每轮查询词数量。
- `DDG_RESULTS_PER_QUERY` / `ARXIV_RESULTS_PER_QUERY` / `TAVILY_RESULTS_PER_QUERY`：控制每条 query 的三路检索配额。

## 9) ArXiv 工具检索逻辑

`tools/arxiv_tool.py` 的执行链路如下：

1. 组装 ArXiv API 请求：`https://export.arxiv.org/api/query?search_query=all:<query>&max_results=<n>`。
2. 使用 `urllib` 发起请求，并带上 `User-Agent` 避免被服务端按匿名脚本拒绝。
3. 使用 `certifi` 提供的 CA 证书创建 SSL 上下文，减少 Windows 证书链导致的 `CERTIFICATE_VERIFY_FAILED`。
4. 解析 Atom XML，提取 `title / summary / link`，转换为统一上下文结构：`title/url/source/content`。
5. 返回给 `researcher` 节点，与 Tavily 结果统一去重并打上 `citation_id`。

## 10) 新增优化能力

- **反馈修订映射**：`writer` 会把 `must_fix` 项映射到本轮草稿章节，`debug` 报告中可查看 `issue -> section`。
- **篇幅硬约束**：`short/medium/long` 对应净字数区间 `800-1200` / `1800-2600` / `3200-4500`，超限会自动裁剪并记录。
- **来源质量评分**：`researcher` 会按来源类型与域名生成 `quality_tier`（A/B/C）和 `quality_score`，并输出 `source_quality_summary`。
- **来源过滤阈值**：`researcher` 会按 `quality/signal/relevance/composite` 四维评分过滤低质量来源，并在 `debug` 报告中输出阈值、权重、均值与剔除原因。

评分依据说明：

- `quality`：来源可信度先验（source 类型 + domain 规则分）。
- `signal`：文本信号比（中文/英文/数字占比），用于过滤导航噪声页。
- `relevance`：主题关键词在标题/摘要/正文的加权命中率。
- `composite`：`quality*Wq + signal*Ws + relevance*Wr`（权重由环境变量控制）。

