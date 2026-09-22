# 来源记忆

[返回索引](index.md)

## 职责

管理跨任务证据内容、有效性、去重、版本、关系与语义召回。它不是聊天消息历史，也不是 LangGraph checkpoint。

入口：[MemoryService](../../memory/service.py)、[SQLite Repository](../../memory/repository.py)、[PostgresMemoryRepository](../../memory/postgres_repository.py)。是否免搜由[记忆优先调度](memory-policy.md)决定，完成后的写入流程归[记忆发布](publication.md)。

## 记忆内容模型

| 字段 | 含义 |
| --- | --- |
| id、namespace | 证据身份与逻辑作用域 |
| url、title | 原始来源定位 |
| content | 匹配检索原文的摘录 |
| claim | Reviewer 关联的断言，用于检索表达；不是独立验证事实 |
| topic、source_run_id | 来源研究主题和任务 |
| content_hash | 规范化摘录摘要 |
| observed_at、valid_until | 观察时间与有效期 |
| status、version | 当前状态和版本 |

向量单独存在 memory_embeddings，以 item_id + model 为键，保存维度及向量。模型身份是模型名加接口地址摘要；不同接口的同名模型不会直接共用记忆向量空间。

content 不是全文网页归档，claim 也不是要无条件相信的结论。长期记忆应始终连同 URL、摘录、时间和状态使用。

## 召回流程

1. 判断查询是否显式时效敏感；必要时返回 fresh_search_required。
2. 按命名空间、active 状态、有效期、向量模型身份读取候选。
3. 排除本任务来源及不适合稳定复用的时效断言。
4. 对查询生成向量，SQLite 使用 NumPy 余弦比较，PostgreSQL 使用 pgvector 排序。
5. 应用 min_score、top_k、字符预算，组装带 memory_id / 版本 / 观察时间的来源资料。
6. 保存访问记录，后续更新实际入选与被引用的 ID。

仓库候选读取有数量上限，当前默认最多 10000 个候选；这不是全量无限检索。PostgreSQL 当前是候选上的精确相似度排序，不能描述为已经建立 HNSW / IVFFlat 近似索引。

无候选时无需请求查询向量。调用失败由上层返回错误状态并继续既有检索路径，不能据此标记为“记忆已覆盖”。

## 去重、更新与冲突

基础身份结合 namespace、规范化 URL 与摘录摘要。重复来源可能复用已有证据 ID，因此一次任务关联 6 条，不一定向 memory_items 新增 6 行。

有效证据重复发布不自动续期。只有过期后取得新的在线观察且满足条件时才建立新版本和 supersedes 关系；旧记忆再次被召回不构成新的在线观察。

冲突检测是有限启发式：当句子结构相同而数字或否定发生变化时，可能标记 possible_conflict，并使双方成为 conflicted。它不是通用自然语言逻辑冲突检测器。

关系登记支持 supports、contradicts、duplicates、supersedes 等。当前 memory/__main__.py 直接构造 SQLite 仓库，不能把 python -m memory 当作 PostgreSQL 通用管理入口。

## 参数

| 配置 | 默认 |
| --- | --- |
| MEMORY_ENABLED | 1，新任务启用 |
| MEMORY_NAMESPACE | workspace/default |
| MEMORY_TOP_K | 5 |
| MEMORY_MIN_SCORE | 0.65 |
| MEMORY_TTL_DAYS | 30 |
| MEMORY_CHAR_BUDGET | 3000 |

参数在任务配置中冻结。旧任务未配置记忆不会因为改了环境就自动导入历史报告；修改向量模型身份也不会自动迁移已有向量。

## 观察与验证

memory_stats.recalled 表示召回候选，selected / selected_ids 表示进入上下文，memory_used_ids 表示报告使用的记忆。应逐层解释损耗，不直接以低引用率判断召回故障。

测试入口：[test_semantic_memory.py](../../tests/test_semantic_memory.py)、[test_storage_backends.py](../../tests/test_storage_backends.py)。表和 SQL 查询见[持久化后端](storage.md)。
