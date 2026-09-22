# 写作与评审

[返回索引](index.md)

## 职责与接口

Writer 根据证据生成或定向修订草稿；Reviewer 评估草稿并给出下一跳建议。本文拥有报告接受规则，不负责定义记忆免搜或发布筛选条件。

代码入口：[writer.py](../../agents/writer.py)、[reviewer.py](../../agents/reviewer.py)、[system_prompts.py](../../prompts/system_prompts.py)、[evidence.py](../../core/evidence.py)。

## Writer

输入包括 topic、retrieved_context、reasoning_contexts、已有草稿和 revision_directives。

- 初稿围绕主题整合资料并使用 S / R 引用。
- 修订时结合 must_fix 等结构化反馈，将问题映射到对应段落或章节。
- 保存草稿与反馈段落映射，供 Reviewer 与调试界面使用。
- 输出模式影响呈现：用户报告、调试输出、评测答题备忘录不应混为同一种格式。

research_time_hint 注入任务创建时的研究截至时间。它不是实时刷新网页的机制，时效证据仍要由检索模块提供。

## Reviewer 的统一收口

LLM 评审、文本解析兜底和规则兜底都经 _finalize_review 计算本地加权分，不信任模型自报 weighted_score。

| 维度 | 权重 |
| --- | --- |
| S1 事实准确性 | 0.35 |
| S2 逻辑完整性 | 0.25 |
| S3 信息覆盖广度 | 0.25 |
| S4 结论可执行性 | 0.15 |

默认阈值 7.5。达到阈值且没有被降级放行策略阻止时，先判定质量接受。降级模式默认不因分数足够而放行，除非 REVIEWER_ALLOW_DEGRADED_PASS=1。

未通过时，S1 < 5、S3 < 4 或评审建议需要补充研究，都会使 needs_more_research 为真；其他写作问题可以回到 Writer。

## 回答标签与图路由

_label_answer 将接受的回答区分为 complete、corrected、limited：

- complete：完整回答。
- corrected：纠正问题前提后的回答。
- limited：披露未解决部分的有限回答，quality_accepted=true，但 is_satisfactory=false。
- not_passed：未被接受。

图先判断 is_satisfactory 或 limited 是否应结束，再判断迭代上限，最后使用 researcher / writer 路由。limited 可以正常结束，但不应显示为完整回答已全部满足。

达到上限输出当前报告，没有实现“自动回滚到历史最高分报告”。

## 证据记录与已删除功能

当前不存在“所有关键证据必须完全支持，否则强制否决整份报告”的额外硬门槛。问题预设不成立可以通过 corrected 回答说明，不能因纠正了前提就自动判失败。

evidence_verdicts 仍用于记录断言与证据，并供后续记忆发布筛选。某条证据不适合长期记忆，不等价于整份报告不能通过。

记忆覆盖检查只决定是否免搜；记忆发布资格只决定哪些原文保存。二者不能重新叠加为 Reviewer 的通过门槛。

## 可观测与限制

review_result 保存分数、模式、降级状态、回答标签和反馈；review_stats、review_degraded_rounds 用于统计兜底行为。iteration_history 保存轮次快照。

支持 / 质疑视角是评审设计，不代表存在独立训练的 Red / Blue 模型或自动事实认证。模型可能高估质量，引用存在也不保证摘录完全支持断言。

验收入口：[test_reviewer_logic.py](../../tests/test_reviewer_logic.py)、[test_review_without_gate.py](../../tests/test_review_without_gate.py)、[test_premise_review.py](../../tests/test_premise_review.py)。
