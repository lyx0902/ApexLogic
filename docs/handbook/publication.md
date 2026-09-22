# 记忆发布与诊断

[返回索引](index.md)

## 职责与时机

将被接受报告中的合格原文证据发布到长期记忆，补齐向量并记录每次尝试。入口为 [MemoryService.publish](../../memory/service.py)，诊断结构见 [publication_log.py](../../memory/publication_log.py)。

研究图已经结束、研究执行尝试已经记为完成后，Runner 才调用发布。发布失败不触发 Reviewer 重审，也不要求重新执行研究节点。

## 资格筛选

_eligible 仅考虑接受的报告，包括 quality_accepted=true 的有限回答。候选来自 Reviewer 的 evidence_verdicts，当前最多检查前 30 项，并要求：

- 未明确标记 critical=false。
- status 为支持 / 验证类值。
- claim 非空且不属于显式时效断言。
- quote 规范化后长度在 8–400 之间，并匹配检索原文。
- 引用 ID 出现在报告中，且能映射到来源。
- 来源 URL 是合法 HTTP(S) 地址，无嵌入用户名密码。
- 来源不是已经召回的旧记忆，避免自我强化和续期。

来源范围与 Writer 对齐：前 10 条 S 资料，以及启用 IRCoT 时前 15 条 R 资料。证据合格只说明符合发布规则，不保证语义上已由独立验证器证明。

observed_at 使用完成时间；若资料来自缓存，取完成时间与缓存抓取时间的较早值，避免把旧缓存伪装成刚观察到的证据。

## 发布过程

1. 查询 run_id + checkpoint_id + namespace 对应回执。
2. 回执已 completed / skipped：直接返回，不新增虚假的发布尝试。
3. 建立本次 attempt，持久化初始诊断。
4. 没有回执时先保存筛选出的证据与 pending 回执。
5. 检查已有向量，形成待补齐列表。
6. 每次读取一条证据，生成 claim + content 的向量并写入。
7. 全部完成后更新回执为 completed，并结束尝试。

文本和已经成功写入的向量不会因后续网络失败被全部撤销。重开完成任务时只补缺失向量，不自动重新运行整个研究。

## 回执与尝试历史

| 存储 | 粒度 | 用途 |
| --- | --- | --- |
| memory_publications | 一个任务最终 checkpoint 与命名空间 | 最近发布状态、关联 item_ids、skip_reason |
| memory_publication_attempts | 每次实际调用的 attempt_id | 保留各次阶段、成功数量与错误诊断 |

同一次尝试的行会随阶段更新；不是每个阶段新插入一行。不同尝试分别保留，后来的成功不会覆盖此前失败。

阶段包括 prepare_evidence、check_existing_vectors、read_evidence、embedding_request、write_vector、finish_publication。触发类型包括 research_completed、reopen_completed、direct。

completed_before 是尝试前已有向量数，completed_this_attempt 是本次新增数，total 是本回执关联证据数。它们不是整个记忆库总量。

## 状态解释

- pending：证据回执已准备，向量尚待完成。
- completed：本次回执所需向量已经齐全。
- skipped：没有可发布证据，不属于网络发布失败。
- failed：发布调用捕获到失败；查看阶段与异常链。
- running：尝试未写入结束，可能仍在执行或已被强杀。

常见 skip_reason 为 report_not_accepted、no_reviewed_claims、only_time_sensitive_claims、no_eligible_original_evidence。报告成功但记忆 skipped 可能完全符合规则。

## 诊断边界

记录最多 8 层白名单异常链，包括异常类型、数值 errno / winerror 和 HTTP 状态码；不保存原始异常消息、密钥、请求正文或代理地址。

阶段说明程序走到了哪里，不足以单独证明根因。例如 embedding_request 失败需结合异常类型区分连接、超时或响应问题。缺失历史日志无法事后补造。

数据库不可用时不能保证尝试日志持久化，返回结果会尽量携带日志错误和当前诊断。当前没有后台发布队列或新增自动重试调度器。

## 查看与恢复

UI 的完成页、状态页和新历史快照展示最近 50 条尝试；数据库保留全部。打开任务状态只查询回执，重新打开完成结果的执行路径才可能尝试补齐。

数据库查询方式见[持久化后端](storage.md)，相关流程验收见[测试与评测](evaluation.md)。
