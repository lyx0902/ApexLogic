# 记忆优先调度

[返回索引](index.md)

## 职责

决定“哪些子问题已有足够原文，可以省去预计划搜索”。它不改变 Reviewer 接受阈值，不把用户问题的预设当成事实，也不负责记忆最终写入。

入口：[prepare_memory_first / retain_required_memory](../../memory/first.py)。调用方的两条检索路径见[检索编排](retrieval.md)。

## 模式

| 模式 | 召回与覆盖检查 | 减少预计划搜索 |
| --- | --- | --- |
| off | 不走此策略 | 否 |
| observe | 条件满足时执行，保存观察结果 | 否 |
| reuse | 条件满足时执行 | 仅对可复用子问题 |

新任务默认 reuse；历史任务的快照缺少 memory_first 时按 off。MEMORY_ENABLED=0 会关闭记忆模块，MEMORY_FIRST_MODE=off 只关闭免搜调度。

## 前置条件

策略要求有 run_id、记忆已启用、AQD 已启用，并且不是修订轮。以下情况直接回到在线流程：

- 主题包含时效敏感关键词。
- APEXLOGIC_CACHE_FORCE_REFRESH=1。
- 没有属于其他任务的有效记忆候选。
- 规划失败、子问题结构非法或数量超过校验上限。

空库检查在前置 AQD 之前进行，避免无意义分解。模型覆盖检查失败时也不能减少搜索，但已经生成的有效计划仍可保留。

## 覆盖检查步骤

1. AQD 输出子问题的 id、question、search_query 和依赖关系。
2. 按子问题召回原始证据，合并去重并执行全局字符预算。
3. 模型只阅读提供的原文，返回 covered / partial / missing，以及 answer、memory_id 和逐字摘录。
4. 本地检查 ID 属于该子问题的召回集合，摘录长度和原文匹配合法，answer 非空。
5. 存在 depends_on 的子问题继续搜索，避免前置事实未落定时使用旧关系。
6. 再次读取原证据，检查 active、有效期、版本和内容未变化。
7. 检查最终上下文是否有足够保留名额；超预算时恢复搜索。
8. reuse 模式下才将通过的子问题标为 search_skipped。

覆盖模型可以依据原文纠正错误前提，但不能靠模型自身知识补齐缺口。本地摘录匹配忽略空白与大小写，不是完整的语义蕴含证明。

## 保留预算

可保留的不同记忆证据数上限为：

```text
min(
    MEMORY_TOP_K,
    BGE_RETRIEVER_TOP_K,
    floor(min(10, BGE_RERANKER_TOP_K) / 2)
)
```

默认值对应最多 5 条。多个子问题可共享同一条证据，按不同 memory_id 计数。支持免搜的资料标记 memory_required，在最终 S 上下文保留；至少一部分名额留给新证据。

## 结果契约

返回 summary、hits、stats、questions、remaining。

| 观察字段 | 意义 |
| --- | --- |
| status | disabled、empty、ok、failed 或具体旁路原因 |
| applied | 是否实际进入免搜路径 |
| covered | 通过原文检查且处于预算内的子问题数 |
| searches_skipped | reuse 下省去预计划搜索的子问题数 |
| subquestions[].reason | 覆盖不足、依赖、时效、版本变化、预算等原因 |
| subquestions[].evidence_ids | 支持该题免搜的记忆证据 |
| subquestions[].search_skipped | 该题是否实际省去搜索 |

searches_skipped 不是节省的 API 次数。observe 可以 covered>0 而 applied=false；这属于预期行为。

## 验证

用稳定主题积累一次已完成发布的记忆，再以新 run_id 对照 observe / reuse，检查证据 ID、逐题 reason、在线调用记录及最终保留情况。单凭相同主题或数据库可连接不能证明免搜生效。

测试：[test_memory_first.py](../../tests/test_memory_first.py)。实际收益衡量见[测试与评测](evaluation.md)。
