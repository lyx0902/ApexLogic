# ApexLogic Deep Research Multi-Agent

基于 LangGraph 的深度研究多智能体系统（Researcher / Writer / Reviewer）。

## 1) 安装

```bash
pip install -r requirements.txt
```

## 2) 配置环境变量

复制 `.env.example` 为 `.env`，填写 API Key：

- `DEEPSEEK_API_KEY`
- `TAVILY_API_KEY`

可选参数：

- `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com/v1`）
- `DEEPSEEK_MODEL`（默认 `deepseek-chat`）
- `MAX_REVISIONS`（默认 `3`）

> 安全建议：不要把真实密钥提交到 `.env.example`。真实密钥仅放在本地 `.env`。

## 3) 运行

```bash
python main.py --topic "多智能体系统在科研自动化中的应用"
```

## 4) 当前实现说明

- `core/state.py`：全局状态定义（`ResearchState`）
- `core/graph.py`：LangGraph 工作流与条件路由
- `agents/researchers.py`：检索代理（Tavily + ArXiv，支持查询重写、去重、`citation_id` 编号）
- `agents/writer.py`：主笔代理（DeepSeek 生成/修订草稿，消费带引用的上下文）
- `agents/reviewer.py`：评审代理（优先 DeepSeek 结构化 JSON 评审，失败回退规则评审）
- `tools/search_tool.py`：Tavily 检索封装
- `tools/arxiv_tool.py`：ArXiv 检索封装
- `prompts/system_prompts.py`：统一系统提示词与提示构造函数

## 5) 降级行为

在缺少依赖、无 API Key、或外部调用失败时，系统会自动降级到本地占位逻辑并记录到 `errors`，保证图流程可继续执行。

## 6) 执行流程（实际运行）

系统按如下顺序执行：

1. `researcher` 生成查询词（可选 LLM 重写）并检索 Tavily + ArXiv。
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
python export_report.py --topic "gemini 3.1pro和gpt5.3 codex的benchmark比较"
```

可选参数：

- `--max-revisions 1`
- `--output reports/custom-report.md`

