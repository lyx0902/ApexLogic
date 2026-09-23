# 测试与评测

[返回索引](index.md)

## 职责

定义模块行为如何验收、真实基础设施如何验证、研究质量与复用收益如何比较。本文不将已有测试文件的存在当作当前全部测试通过的证据。

## 三类验证

| 类型 | 验证什么 | 不能证明什么 |
| --- | --- | --- |
| 模拟单元 / 回归 | 分支、状态、参数、审计和错误处理 | 真实供应商质量与稳定延迟 |
| PostgreSQL / Redis 集成 | 锁、事务、恢复、缓存与真实服务协议 | 搜索和 LLM 答案质量 |
| 真实研究任务 | 检索效果、报告质量、端到端耗时 | 单次结果不能代表稳定收益 |

普通测试入口：

```bash
python -m pytest -q
```

tests/conftest.py 默认隔离生产存储和缓存配置。真实服务测试通过显式变量启用，不应把 skipped 报告成已验证。

## 模块验收映射

| 模块 | 主要测试文件 |
| --- | --- |
| 执行恢复 | test_recovery.py |
| 存储契约 | test_storage_backends.py |
| 来源记忆 / 发布 | test_semantic_memory.py |
| 免搜与必要证据保留 | test_memory_first.py |
| 查询复用 / 逐题资料 | test_retrieval_cleanup.py |
| 评审与移除硬门槛 | test_reviewer_logic.py、test_review_without_gate.py、test_premise_review.py |
| 缓存 | test_cache.py |
| 真实数据库 | test_postgres_integration.py |
| 操作识别与报告追问 | test_intent.py、test_conversation.py |
| 真实 Redis | test_redis_integration.py |

文件均位于 tests 目录。这里不固定测试条数，避免新增测试后文档宣称过时的通过数量。

`test_conversation.py` 默认运行离线追问证据测试。设置 `APEXLOGIC_TEST_CONVERSATION_PG=1` 后，它会用 `.env` 的 PostgreSQL 账号创建独立临时数据库，测试追问迁移、幂等提交、跨 Worker 领取、租约接管和完成隔离，结束后删除临时库。此测试不调用真实模型或搜索 API。

广搜并发测试在 `test_retrieval_cleanup.py` 使用模拟延迟，核对来源并发上限、冻结配置传递、缓存计数、异常审计以及串行/并发结果顺序一致。它不证明真实 provider 有稳定提速；真实验证须在相同配额、主题、缓存状态和服务条件下分别运行总线程数 1 与默认并发，并记录实际外部调用数、错误和报告质量。

## 基础设施集成

PostgreSQL 测试需设置 APEXLOGIC_TEST_POSTGRES_DSN，账号须有 CREATEDB 权限。fixture 会建立随机 apexlogic_test_<uuid> 数据库，结束后删除测试库，覆盖初始化重放、进程崩溃恢复、锁、记忆导入与发布等。

Redis 测试需设置 APEXLOGIC_TEST_REDIS=1，并读取 .env 的 APEXLOGIC_REDIS_* 配置；测试创建随机前缀键，结束时只清理该前缀。

```bash
python -m pytest -q tests/test_postgres_integration.py
python -m pytest -q tests/test_redis_integration.py
```

后台 Worker 集成测试会向真实 PostgreSQL 队列提交模拟任务。先停止普通 Worker 和 Streamlit，再单独设置 `APEXLOGIC_TEST_BACKGROUND_EXCLUSIVE=1` 运行 `tests/test_background.py`；活动 Worker 会抢先领取测试任务，因此不能在共享队列上并行测试。进程强杀用例会启动两个临时 Worker 子进程，在第二个模拟节点执行中终止第一个进程，等租约过期后验证同一任务从 checkpoint 接管、只重跑未提交节点，并生成历史 JSON。测试还覆盖 Redis 在 Worker 启动后失联时的 PostgreSQL 扫描补偿、心跳重试及租约转移后的节点边界停止。进程用例发现其他排队或运行中任务时会跳过，以免扫描执行真实研究任务；测试创建的任务、checkpoint 和消息会清理。模拟节点不调用搜索或 LLM API。

```powershell
$env:APEXLOGIC_TEST_BACKGROUND_EXCLUSIVE = '1'
.\.venv\Scripts\python.exe -m pytest -q tests/test_background.py
```

历史快照的离线投影和排序测试为 `tests/test_history.py`。`tests/test_parallel_governance.py` 用模拟延迟验证有界并发、稳定合并、AQD 依赖、IRCoT 跳间屏障和全局名额耗尽。独占的 `test_background.py` 另核对排队容量、跨 Worker 入场、跨进程 provider 名额与分钟配额。模拟测试不等于真实供应商端到端稳定性验证；真实 PostgreSQL 和多 Worker 的压测需在独占测试环境执行。

执行前安装对应可选依赖。PostgreSQL 初始迁移 SQL 必须随源码可用。

## 端到端恢复验收

1. 新建任务，记录 run_id、后端与配置。
2. 等至少一个节点提交后终止 Streamlit 进程。
3. 重启并打开原任务，核对 checkpoint 与下一待执行节点。
4. 继续研究，检查已提交节点未从头运行，未提交节点允许重跑。
5. 检查 attempts 新增执行记录，最终报告与来源存在。
6. 再打开已完成任务，确认没有新的研究节点执行。

强杀前未保存的时长不可精确恢复。若发现任务身份改变，那是新任务而非恢复。

## 记忆发布验收

正常任务：区分 completed、skipped 和 failed，并对照回执 item_ids 与实际向量。

部分失败：优先使用模拟向量服务，在第 N 条失败；确认此前成功向量保留、失败尝试记录保留。再次打开完成任务后，只补剩余向量，并产生新的发布尝试。不要通过删除真实业务数据制造测试条件。

强杀：running 尝试只是缺少结束记录，不能自动当作 failed。数据库故障下不保证诊断日志仍落盘。

## 研究问答评测

```bash
python eval_runner.py --dataset hotpotqa --difficulty hard --limit 50 --scorer llm --concurrency 4
python eval_runner.py --dataset bamboogle --limit 10 --scorer em
python eval_baselines.py --provider qwen --dataset hotpotqa --level hard --limit 50 --scorer llm
```

eval_runner 调用研究图进行批量问答，不能代替 ResearchRunner 的持久化恢复验收。concurrency 是多题并发，不代表 Researcher 内部并行。

Exact Match 检查答案匹配，LLM 评分具有模型偏差。比较时固定题目集合、模型、评分器、预算、并发和重试口径，并保留错误样本。基线调用需要独立服务凭证。

## 复用收益实验

至少设置以下对照，使用新的 run_id：

| 对照 | 记忆策略 | Redis |
| --- | --- | --- |
| 基础组 | off | 关闭 |
| 缓存组 | off | 开启 |
| 覆盖观察组 | observe | 与复用组保持相同 |
| 记忆复用组 | reuse | 与观察组保持相同 |

准备稳定主题和冷启动主题；分开冷缓存 / 热缓存，固定命名空间与库内容，随机化执行顺序并多次重复。

记录实际工具调用、文本向量调用、缓存命中、免搜子问题数、召回 / 入选 / 引用、报告质量和包含发布的总等待时间。研究 attempt 耗时不包含完成后的全部发布时间，应独立测量端到端时长。

不要将 searches_skipped 当作节省 API 数，也不要将单次耗时降低作为统计结论。MAB 采样、查询生成和服务延迟都会带来波动。
