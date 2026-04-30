# ApexLogic Deep Research Multi-Agent

基于 **LangGraph** 的深度研究多智能体系统。输入一个研究主题，系统自动调度三个 Agent 协作——**Researcher** 执行广搜与精筛、**Writer** 生成结构化草稿、**Reviewer** 四维量化评审——形成"检索→起草→评审→修订"闭环，直至质量达标或达到轮次上限。

相比标准 RAG 流水线，ApexLogic 在检索端叠加了四层优化（MAB 自适应预算 / 概念共现图扩展 / 自适应查询分解 / 交错链式推理补搜），写作端支持基于引用核查的定向修订，评审端输出可追踪的四维评分（事实准确性、逻辑完整性、信息覆盖广度、结论可执行性）。

## 工作流

```
START → Researcher → Writer → Reviewer ─┬─ 评审通过 ──→ END
                      ↑                  ├─ 信息不足   ──→ Researcher（补充检索）
                      └──────────────────┴─ 需修订     ──→ Writer（重写草稿）
```

每轮迭代，Reviewer 给出加权总分（满分 10，默认 7.5 通过）及具体反馈：事实错误、逻辑漏洞、信息缺口。Writer 据此定向修订对应段落，或 Researcher 针对缺口补充检索。超过 `MAX_REVISIONS`（默认 4 轮）仍未通过则强制终止，输出当前最佳版本。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，至少填写 DEEPSEEK_API_KEY、BGE_EMBED_API_KEY、BGE_RERANK_API_KEY

# 3. 启动 Web UI（推荐）
streamlit run app.py

# 或命令行运行
python main.py --topic "你的研究主题" --output-mode debug
```

## 核心特性

**四层检索优化**，逐层作用于同一候选文档池，最后经 BGE 两阶段精筛选出 Top-10 高质量上下文：

- **Thompson Sampling MAB**：根据 DDG / ArXiv / Tavily 三路搜索源的历史表现，动态分配每轮检索预算，优质来源获得更多配额
- **Graph Expand**：从已检索文档中提取核心概念，构建共现图（PageRank），生成扩展查询方向补搜
- **AQD（自适应查询分解）**：LLM 将主题拆解为多个子问题，拓扑排序后逐子问题检索，覆盖不同切入角度
- **IRCoT（交错链式推理补搜）**：LLM 多跳推理链识别信息缺口，针对缺口定向补搜，推理链跨迭代累积

**BGE 两阶段精筛**：Retriever（向量相似度粗筛 top-20）→ Reranker（交叉编码器精排 top-10）

**四维量化评审**：Reviewer 从事实准确性（35%）、逻辑完整性（25%）、信息覆盖广度（25%）、结论可执行性（15%）四个维度打分。S1 < 5 或 S3 < 4 时强制回到 Researcher 补充检索，不依赖总分判定。

## 配置要点

`.env` 中必填的三个 Key：

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | LLM（查询重写/起草/评审/推理） |
| `BGE_EMBED_API_KEY` | BGE 向量嵌入（粗筛） |
| `BGE_RERANK_API_KEY` | BGE 重排序（精排） |

常用可选配置：`MAX_REVISIONS`（最大迭代轮数，默认 4）、`REVIEWER_PASS_THRESHOLD`（通过阈值，默认 7.5/10）、`TAVILY_API_KEY`（启用 Tavily 搜索源）。所有优化层（Graph Expand / AQD / IRCoT）均可通过环境变量独立开关或调整参数，详见 `.env.example`。

## 输出模式

`export_report.py` 支持四种输出模式，一条命令切换：

```bash
python export_report.py --topic "你的研究主题" --output-mode user/debug/both/user_only
```

| 模式 | 产物 |
|---|---|
| `user` | 干净的研究报告 + 参考文献 |
| `debug` | 完整报告 + 四维评分详情 + 迭代历史 + 执行轨迹 + MAB/图扩展/AQD/IRCoT 各层摘要 |
| `both` | 同时输出 user + debug 两份 Markdown + BGE 检索明细 JSON |
| `user_only` | 内部运行完整流水线，仅导出用户侧 Markdown |

输出文件落地 `reports/` 目录，文件名含时间戳。

## Streamlit Web UI

`streamlit run app.py` 启动可视化界面，整体信息架构如下：

**侧边栏（参数配置）**
- 研究主题输入、最大反思轮数滑块、通过阈值滑块
- "开始深度研究"按钮触发执行
- 历史记录列表：过往运行结果以 JSON 形式保存在 `appstats/` 目录，可随时回看

**主区域（实时流式执行）**
- 系统通过 LangGraph 的 `stream()` 模式逐节点推送状态，前端实时渲染而非等待全流程结束
- 每个节点完成后展示对应面板，三个面板按执行顺序依次展开：

1. **Researcher 面板**：展示本轮检索词、MAB 三路预算分配、去重统计、AQD 子问题分解详情（含每个子问题补搜到的文档链接）、IRCoT 逐跳推理链与 gap 查询。若跨轮次推理，会标注哪些推理链是本轮新增、哪些继承自历史轮次
2. **Writer 面板**：分离展示 DeepSeek 的 `<think>` 内部思维链与正文草稿，草稿预览前 600 字
3. **Reviewer 面板**：四维评分仪表盘（每维得分 + 加权总分与阈值的差值）、评审意见文字反馈、结构化修订指令、下一跳路由决策（输出最终报告 / 补充检索 / 修订草稿）

**最终输出区**
- 完整研究报告（正文中的 `[S1]` `[R1]` 等引用自动转为可点击超链接）
- 一键下载 Markdown 报告
- BGE Reranker 精选的 Top-10 参考资料列表
- IRCoT 推理链专属参考文献（独立于 BGE pipeline，不经过 BGE 筛选，确保推理发现的关键信息不丢失）

**历史记录浏览**
- 过往运行结果完整回放：最终报告、迭代快照、每轮思维链、四维评分，展示逻辑与直播执行完全一致

## 常用命令

### 研究报告导出

```bash
# 一条命令，--output-mode 切换 user / debug / both / user_only
python export_report.py --topic "多智能体系统中的反思机制" --output-mode both
```

### 批量评测（HotpotQA / Bamboogle）

评测 ApexLogic 自身流水线在多跳问答数据集上的表现，支持并发与 LLM 语义判定：

```bash
# HotpotQA hard 难度，限制 50 题，LLM 语义判定，4 线程并发
python eval_runner.py --dataset hotpotqa --difficulty hard --limit 50 --scorer llm --concurrency 4

# Bamboogle 全量，Exact Match 模式
python eval_runner.py --dataset bamboogle --scorer em
```

结果 JSON 默认输出到 `tests/` 目录，文件命名含数据集、难度、时间戳。评测脚本与 `main.py`/`export_report.py` 完全独立，不依赖 Streamlit。

### 商业模型基线评测

绕过 ApexLogic 流水线，直接调商业模型 API 作答，用于横向对比：

```bash
# Qwen / Doubao 基线，HotpotQA hard 难度
python eval_baselines.py --provider qwen --dataset hotpotqa --level hard --limit 50 --scorer llm
python eval_baselines.py --provider doubao --dataset hotpotqa --level hard --limit 50 --scorer llm
```

需要预先在 `.env` 中配置 `QWEN_API_KEY` 或 `DOUBAO_API_KEY` + `DOUBAO_ENDPOINT_ID`。

### 评测耗时统计

```bash
python -u evals/elapsed_stats.py --input tests/results_hotpotqa_20260411_172430.json
```

输出 `total_rows`、`valid_rows`、`unique_ids`、`overall_avg` 等统计。

## 容错策略

系统遵循"降级不中断"原则——LLM 不可用时回退规则生成，BGE 服务异常时跳过不截断候选集，Reviewer 输出非法 JSON 时规则评审器接管，任意优化层异常只记录错误日志不中断主流程。

## 安全提示

- 不要把真实 API Key 提交到仓库，`.env` 已在 `.gitignore` 中
- `.env.example` 只放占位值
