# 检索编排

[返回索引](index.md)

## 职责与接口

[researcher_node](../../agents/researchers.py) 接收主题、历史状态和评审反馈，输出候选证据、搜索记录、推理补搜结果与最终写作上下文。

本模块决定“查什么、什么时候查”；排序细节归[排序与证据上下文](ranking.md)，历史证据是否可免搜归[记忆优先调度](memory-policy.md)。

## 分支入口

Researcher 先尝试记忆优先准备，再决定是否生成普通重写查询。

- 模式关闭、记忆关闭、时效任务、修订轮等条件直接走普通路径。
- 会先检查有效候选是否存在；空记忆库不会为了记忆优先额外调用 AQD 分解。
- 有候选才进行前置规划、逐题召回和覆盖判断。
- 至少一个子问题真正可以免搜，且模式为 reuse，才进入复用路径。
- 前置计划有效而召回或覆盖失败时，保留计划，供普通路径的 AQD 补搜复用。

## 路径 A：记忆复用生效

1. 取未覆盖子问题的 search_query，形成搜索计划。
2. SEARCH_QUERY_BUDGET 范围内的查询调用多源广搜。
3. 超出该范围的子问题继续由 DDG 补搜，不能把广搜预算当作丢弃子问题的理由。
4. 不再调用普通查询重写、图扩展和后置 AQD。
5. IRCoT 继续研究，发现缺口后仍可搜索。
6. 合并在线结果与记忆，进入 BGE 筛选；保留支持免搜的证据。

所有子问题已覆盖也不意味着绝不联网，IRCoT 仍可能补搜。

## 路径 B：普通检索

1. 依据主题与反馈生成普通查询，构成初始广搜。
2. MAB 分配各搜索源的结果配额。
3. 从初始资料建立概念共现图，选择图扩展查询。
4. AQD 对子问题补搜：有有效前置计划就复用，否则重新分解。
5. 已经搜索过的同一查询可以复用广搜记录，避免独立重复搜索。
6. IRCoT 交替推理与补搜，随后进入证据筛选。

复用前置计划不等于读完网页后再次规划；后续新缺口主要由 IRCoT 处理。

## 搜索优化各自解决什么

| 组件 | 输入 | 输出与边界 |
| --- | --- | --- |
| ThompsonSamplingMAB | 历史搜索源奖励、候选源 | 返回结果配额；不是 Token 或费用额度 |
| 概念图扩展 | 检索资料中的概念关系 | PageRank 扩展词及 DDG 搜索 |
| AdaptiveQueryPlanner | 主题，可选已有资料 / 计划 | 子问题、search_query、depends_on、执行顺序 |
| IterativeRetrievalOptimizer | 当前资料与主题 | reasoning_chains、gap 查询、reasoning_contexts |

AQD_RESULTS_PER_SUBQ 控制每题返回条数。拓扑顺序处理子问题不代表异步并行 DAG。IRCoT 的可选末尾总结不是额外一跳搜索；不能直接把推理记录长度当作请求数。

MAB 的奖励来自在线资料质量反馈，记忆来源不作为搜索工具本轮表现参与奖励。

## 逐题审计

[search_audit.py](../../optim/search_audit.py) 在工具入口记录 provider、query、请求条数上限、状态、异常类型与资料。

- planned_search_queries：规划查询，可能包括后来省去的查询。
- search_queries：实际尝试查询词的去重列表；缓存返回或失败也属于尝试。
- query_plan.sub_results：每题搜索记录、匹配的已有搜索记录、返回资料与记忆。
- selected / citation_id：是否进入最终普通写作上下文及对应 S 编号。

同一 query 可对应多个子问题，所以查询字符串不能作为唯一归属键。arXiv 内部查询变体没有逐个展开为独立的子问题审计记录。

## 故障边界与验证

可选阶段失败会记录错误并按调用方处理，不应承诺所有模型、网络或配置错误都不中断。检查真实路径时同时读取 memory_first.applied、query_plan、检索摘要和 errors。

测试入口：[test_retrieval_cleanup.py](../../tests/test_retrieval_cleanup.py)、[test_memory_first.py](../../tests/test_memory_first.py)。
