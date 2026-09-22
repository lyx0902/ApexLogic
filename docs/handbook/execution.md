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

## 验证

```bash
python main.py --list-runs
python main.py --status RUN_ID
python main.py --resume RUN_ID --storage-backend sqlite
```

使用实际 run_id 和原后端。强杀测试应检查同一个任务的节点轨迹、checkpoint、attempt 数与最终结果；只看到页面恢复并不能证明没有从头重跑。具体验收见[测试与评测](evaluation.md)。
