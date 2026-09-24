# 执行与恢复

[返回索引](index.md)

## 职责

管理任务身份、执行尝试、节点 checkpoint、并发互斥与恢复校验。研究策略由图节点实现；数据表和连接由[持久化后端](storage.md)负责。

核心入口为 [ResearchRunner](../../core/runner.py)，图定义为 [compile_graph](../../core/graph.py)。

## 对外操作

| 方法 | 行为 | 可能产生写入 |
| --- | --- | --- |
| create | 建立任务及配置快照，不启动研究 | 任务记录 |
| stream | 创建执行尝试，逐节点产生 RunEvent | 任务、尝试、checkpoint；完成后可能发布记忆 |
| run | 完整消费 stream，返回最终状态 | 同 stream |
| inspect | 获取状态并在可获锁时校正过期运行记录 | 可能更新任务 / 尝试；不触发记忆发布 |
| peek | 只读获取当前任务快照与尝试；首节点未保存前返回空状态 | 无 |
| history | 读取 checkpoint 历史，按 execution_trace 长度去重 | 不执行研究节点 |
| completed_state | 经 inspect 检查后返回已完成状态 | 可能校正元数据；不重新研究或发布 |

因此 --status 和从已完成任务导出不调用研究模型，但不能视为严格的零写入数据库查询。严格只读检查应使用数据库只读事务。

## 生命周期

新任务通常经历 created → running → completed；异常或中断后进入 interrupted，再恢复为 running。强制终止时，数据库可能暂时残留 running，下一次可获得执行锁的 inspect / stream 会结合 checkpoint 校正。

每次真正执行图都会产生一条 attempt。重新打开已经完成的任务不增加研究执行尝试，但如果记忆尚未发布完，可能产生独立的发布尝试。

结束原因分为：

- passed：报告达到接受条件且不是有限回答。
- limited：接受了明确披露不足的有限结论。
- max_revisions：达到上限退出，并不代表质量通过。

详细接受判定见[写作与评审](writing-review.md)。

## 恢复协议

1. 读取任务，核对存储后端、schema_version、workflow_version。
2. 验证配置内容与 config_hash。
3. 查找 thread_id 对应 checkpoint，核对其中的任务身份与配置。
4. 获得任务锁后校正旧执行尝试。
5. 无 checkpoint 且从未执行过：使用初始状态启动。
6. 有待执行节点：使用 graph.stream(None, ...) 恢复。
7. 已没有待执行节点：直接返回完成结果，按需补齐记忆发布。

若已有执行记录而 checkpoint 缺失，Runner 报错并保留数据，不会静默从头重跑。配置或版本不兼容同样拒绝恢复。

## 持久化和锁

图执行采用 durability="sync"。恢复边界是已提交节点，Researcher 内的某次 HTTP 请求或 Writer 内的 Token 输出不是独立恢复点。

SQLite 使用进程间文件锁；PostgreSQL 使用会话 advisory lock，执行时 Checkpointer 共用该锁连接。锁连接丢失也使该执行路径无法继续写 checkpoint。

这能避免同一任务被两个正常执行器同时推进；它不构成跨网络服务的恰好一次事务。未提交节点中的外部调用仍可能重复。

## 时间与配置

运行耗时使用 monotonic 计时并在尝试结束时持久化。强杀未保存的耗时不会补估为精确值。记忆发布在研究尝试结束之后进行，因此研究耗时不等于包含发布在内的页面总等待时间。

研究配置和 research_as_of 在创建时冻结；密钥、数据库连接及部分基础设施配置仍从当前环境读取。详细分类见[使用与配置](operations.md)。

## PostgreSQL 后台多 Worker

Streamlit 的 PostgreSQL 路径现在只提交和查询任务。`BackgroundScheduler.submit` 在同一事务写入 `runs`、`research_jobs` 和 `research_outbox`；Worker 将 outbox 通知投到独立 Redis Stream，消费者组收到后领取 PostgreSQL 租约。Redis 消息仅用于唤醒，周期扫描 PostgreSQL 会补偿丢失的通知和过期租约。重复通知必须再次竞争领取，研究图仍由原有 PostgreSQL advisory lock 排他执行。

Worker 通过 `ResearchRunner.stream` 执行，保留原配置快照与 checkpoint 校验。每完成一个节点更新 `last_node`；心跳续租。短暂的心跳数据库错误会记录异常类型并重试；若租约已被其他执行者领取，原 Worker 在节点返回边界停止继续推进。取消请求写入调度记录，Worker 在节点返回后检查并停止，不会中断正在执行的同步模型或搜索调用。外部调用在未提交节点崩溃后仍可能重复。

研究完成后，Worker 从最终状态和 checkpoint 历史生成与普通 Streamlit 任务相同的 appstats 展示快照。快照生成失败只记录 `HistoryExport:*` 诊断，研究仍保持完成；再次打开该任务时可从 checkpoint 补建，无需重新调用研究节点。

调度状态 `queued/running/completed/cancelled` 与研究状态 `created/running/interrupted/completed` 分开。Worker 失联后租约到期可接管；实际恢复位置以 checkpoint 为准。多个 Worker 可并行领取不同任务；领取事务按创建时间选择最早符合条件的任务。`APEXLOGIC_QUEUE_MAX_PENDING` 在提交事务中限制等待任务数量，满额提交不创建孤立 run。PostgreSQL 会话 advisory lock 将同时运行的任务限制在 `APEXLOGIC_GLOBAL_RESEARCH_MAX_INFLIGHT` 内，并分别限制 DDG、arXiv、Tavily 的在途请求；连接在进程崩溃时释放。版本 4 表保存逐来源滚动一分钟请求票据，崩溃后的票据不会丢失。所有 Worker 必须使用相同的全局上限配置。等待 provider 名额或分钟配额超过 `APEXLOGIC_PROVIDER_WAIT_SECONDS` 会作为该次搜索失败进入审计。CLI `main.py` 仍支持旧任务直接执行；不要对同一个后台任务同时使用 CLI 恢复。

启动及验证步骤见[使用与配置](operations.md)。

Worker 还会扫描 PostgreSQL 中的追问、更新、改写和核验记录。这些操作有独立租约与并发名额，读取已完成任务的 checkpoint，但不改变原研究的调度状态或节点进度。更新与改写的新报告写入 `report_versions`，原报告仍在 checkpoint 中；核验只保存操作结果。其排队、接管与失败规则见[意图识别与报告操作](intent-and-followup.md)。

版本 7 的 `worker_instances` 保存 Worker 空闲与执行时的心跳，`worker_targets` 保存本机 Streamlit 管理的目标数量。页面每 5 秒查询并校准，启动时先在 PostgreSQL 占位再创建隐藏进程，避免多个会话同时重复启动。超过 20 秒没有心跳的实例不计入在线数量；界面只对本机管理的 Worker 请求协作式退出。若 Worker 在 UI 关闭后结束，需再次打开页面才会补足目标数量；任务本身仍由 PostgreSQL 保留和恢复。

## 验证

```bash
python main.py --list-runs
python main.py --status RUN_ID
python main.py --resume RUN_ID --storage-backend sqlite
```

使用实际 run_id 和原后端。强杀测试应检查同一个任务的节点轨迹、checkpoint、attempt 数与最终结果；只看到页面恢复并不能证明没有从头重跑。具体验收见[测试与评测](evaluation.md)。
