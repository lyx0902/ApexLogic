# 研究操作识别、追问与报告版本

[返回索引](index.md)

## 职责与范围

`core/intent.py` 将输入归入新研究、查看、恢复、取消、报告追问、更新、改写、核验或澄清。显式控制指令由规则识别；其他输入最多调用一次模型返回结构化分类。程序再次验证目标任务、状态和可执行路径。模型不能直接调用调度器。页面也提供明确的追问、更新、改写、核验选项，避免分类失败时无法操作。

报告操作仅支持已完成的 PostgreSQL 任务。SQLite 旧任务继续按原方式查看和恢复，不进入后台操作队列。当前项目没有用户认证或多租户授权；任务 ID 和页面选中项只是定位任务，不能当作访问控制。

## 操作到模块的映射

| 操作 | 入口与执行模块 | 读取与产出 |
| --- | --- | --- |
| 新研究、查看、恢复、取消 | `core/intent.py` 分类，`app.py` 调用既有调度或页面导航 | 研究任务与原 checkpoint；不写 `conversation_turns` |
| 追问 | `core/conversation.py` 排队，Worker 调用 `core/followup.py` | 从原 checkpoint 读取报告、S/R 摘录和推理链；保存带引用的回答，不检索新资料 |
| 改写 | Worker 调用 `core/report_operations.py` | 读取当前报告版本和已有来源，生成独立 Markdown 版本；超长报告需要指定章节标题 |
| 更新 | Worker 调用 `core/report_operations.py` | 新检索后生成带 U 引用的增量章节，附到当前版本并保存为下一版 |
| 核验 | Worker 调用 `core/report_operations.py` | 对照旧报告和新检索摘录，保存核验回答与来源；不生成报告版本 |

自动识别只决定操作类型与约束，不直接执行工具；页面显式选择追问、更新、改写或核验时，以用户选中的操作类型提交当前任务。控制类指令继续走原任务调度路径。

## 数据流与证据

1. `app.py` 在完成任务的详情和历史页面收取操作文本。控制类操作仍调用既有 `BackgroundScheduler`；追问、更新、改写、核验提交到 `ConversationRepository`。
2. 提交时检查原任务为已完成且有报告，使用幂等键和待处理容量约束，在 PostgreSQL `conversation_turns` 中留下 `queued` 记录。
3. Worker 周期扫描 PostgreSQL。数据库是权威投递位置；这些操作不依赖新增 Redis Stream。多个 Worker 领取不同任务的操作，同一研究任务按顺序处理。
4. `core/followup.py` 从已完成任务的 checkpoint 读取原报告、普通 S 来源、IRCoT R 来源和已保存推理链；历史展示 JSON 只有部分截断资料，不用于生成答案。根据问题选择有界摘录，并优先保留上一轮问答引用的原始来源；此前问答正文只用于指代消解。
5. 回答使用原任务实际存在的 `[S#]` / `[R#]` 或推理链编号。程序拒绝不存在的引用，页面沿用报告引用链接展示。缺少来源、证据不足或要求新时效事实时，不声称已联网核验。
6. `core/report_operations.py` 的更新和核验用“任务主题＋本次要求＋时间范围”组成一次查询，DuckDuckGo 与 Tavily 各请求最多 3 条，去重并过滤缺少 URL/摘录的结果后最多保留 6 条。新来源使用 `[U#]`，与原始 S/R 编号分开；后续版本从已有最大 U 编号继续。模型只能引用本次提供的来源编号。
7. 更新要求增量章节至少引用一条本次 U 来源；没有可引用的新资料时，不生成新版。核验保存结果与新来源，不改写报告；即便新检索有结果，也不能把搜索摘录宣称为已阅读全文核实。

改写报告超过单次模型输入预算时，需在要求中写出 Markdown 章节标题；系统只改写匹配章节并保留其他章节。若指定章节本身仍过长，会提示缩小范围。

这些操作不修改原报告、ResearchState、checkpoint、研究配置快照或来源记忆。`report_versions` 保存独立 Markdown 版本、父版本和来源清单；`conversation_turns.result_json` 保存新检索摘录与操作结果。Worker 按同一任务的提交顺序执行，下一次更新或改写以最新已保存版本为基础。版本创建与操作完成在同一事务，并受租约 fencing 约束。引用编号存在性检查只约束新操作结果，不能成为原报告的额外通过门槛。`request_json` 保存本次识别的时间范围、约束和输出格式，不保存密钥。

## 生命周期与恢复

`conversation_turns` 使用 `queued/running/completed/failed`。Worker 领取后持有租约并发送心跳；租约过期可由其他 Worker 接管。提交、领取和完成都在 PostgreSQL 中完成，完成更新受 `lease_owner` 和有效租约约束；旧 Worker 即使返回结果也不能覆盖新持有者。模型调用崩溃边界允许重复，不能承诺外部调用恰好一次。失败最多尝试三次，最终状态和异常类型保存在表中；不保存可能包含凭据的异常原文。

`APEXLOGIC_FOLLOWUP_MAX_PENDING` 限制排队加执行中的操作数，默认 100；`APEXLOGIC_GLOBAL_FOLLOWUP_MAX_INFLIGHT` 使用 PostgreSQL advisory lock 限制跨 Worker 并发，默认 2。操作由已有 `worker.py` 进程处理。Worker 不运行时页面仍可显示排队记录，启动 Worker 后继续处理。模型请求设置 90 秒超时，分类请求设置 30 秒超时。任一搜索服务失败时只使用实际取得的来源，页面显示 DuckDuckGo/Tavily 各自保存条数；只有一路结果时提示跨服务对照不足。两路请求强制旁路可丢弃的搜索缓存，但继续使用搜索工具已有的来源并发与分钟配额控制。不同服务收录同一网页不构成独立交叉验证，检索摘录也不是网页全文。`TAVILY_API_KEY` 必须在 Worker 环境可用；`ENABLE_TAVILY_FALLBACK` 影响其他调用统一搜索入口的路径，报告更新和核验固定采用上述双源配额。

## 使用与验收

先运行 `python -m scripts.postgres_admin init` 应用版本 6 迁移，再启动 Worker 和 Streamlit。在 PostgreSQL 已完成任务详情选择操作类型并输入要求。历史页重新打开时从数据库加载问答、核验来源和报告版本；新版报告可展开查看和下载。增量更新是带检索时间的附录，未重新运行完整 Researcher/Writer/Reviewer 流程，也不自动发布为来源记忆。

`tests/test_intent.py` 覆盖显式控制、目标任务和结构化分类；`tests/test_conversation.py` 覆盖 S/R 引用、证据不足，并可在 `APEXLOGIC_TEST_CONVERSATION_PG=1` 时创建独立临时数据库验证幂等、顺序领取、Worker 处理、失联接管和版本写入 fencing。`tests/test_report_operations.py` 覆盖三类操作、双源各取 3 条、单路失败、无新来源和非法引用。测试数据库会在结束后删除；这些测试不对真实供应商进行调用。
