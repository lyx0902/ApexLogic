# 搜索与向量缓存

[返回索引](index.md)

## 职责

Redis 缓存减少相同请求的重复工作，不存放研究 checkpoint，不是长期记忆数据库，也不是当前系统的后台消息队列。

代码：[core/cache.py](../../core/cache.py)、[bge/embeddings.py](../../bge/embeddings.py)。Redis 关闭或不可用时，调用方仍可直接请求搜索与向量服务。

## 缓存键与有效性

| 类型 | 身份组成 | 默认 TTL |
| --- | --- | --- |
| 搜索 | namespace、工具 provider、绑定默认值后的调用参数 | 1800 秒 |
| 文本向量 | namespace、文本、模型、接口、凭证摘要、缓存版本 | 604800 秒 |

键前缀为 apexlogic:cache:v1，具体身份哈希化。向量缓存不把原始凭证直接放入键。仅缓存完全相同的请求身份，不做“意思差不多就复用”的语义结果缓存。

缓存值包含 fetched_at 和 value。读取时同时验证年龄与内容结构；Redis 中存在键不等于有效命中。空搜索结果不按正常有效结果缓存。

## 搜索请求流程

1. 检查研究主题、反馈、查询的时效关键词和强制刷新配置。
2. 若应刷新或未启用 Redis，直接调用原工具。
3. 查有效缓存，命中时为结果添加 cache_hit 和 cache_fetched_at。
4. 未命中时尝试 SET NX 获取 60 秒租约。
5. 未获锁者最多等待约 2 秒，每隔 0.05 秒检查结果，然后允许自行调用服务。
6. 锁拥有者调用搜索；Lua 校验所有权后写结果、释放锁。

它是有界并发合并，不保证永远只有一个外部调用。等待超时、租约过期或缓存故障都可能产生重复调用。锁失去所有权后不能覆盖新拥有者的结果。

## 向量流程

每个文本独立查缓存，缺失文本仍按批次请求，最后按原顺序重组。检查返回数量、有限数值、非零向量和维度。

如果命中向量和新返回向量的维度不一致，会刷新整批，避免拼接不兼容结果。APEXLOGIC_EMBED_CACHE_VERSION 可用于主动区分版本，但长期记忆的模型身份和数据库向量迁移是另一层机制。

APEXLOGIC_CACHE_FORCE_REFRESH=1 绕过搜索缓存和记忆免搜，不要求对相同文本重新计算向量。

## 故障旁路

连接和读写超时当前为 0.4 秒，Redis 客户端不自动重试；操作异常触发约 10 秒缓存旁路窗口。错误被统计，不输出可能包含凭证的异常原文。

缓存失败不会使搜索工具本身必然成功，旁路后仍可能遇到供应商错误。TTL 配置有 1 秒到 30 天的界限。

## 部署

在 .env 中加入：

```dotenv
APEXLOGIC_CACHE_ENABLED=1
APEXLOGIC_REDIS_HOST=127.0.0.1
APEXLOGIC_REDIS_PORT=6379
APEXLOGIC_REDIS_DB=0
APEXLOGIC_REDIS_PASSWORD=REPLACE_ME
APEXLOGIC_SEARCH_CACHE_TTL=1800
APEXLOGIC_EMBED_CACHE_TTL=604800
```

```bash
python -m pip install -r requirements-redis.txt
docker compose -f compose.redis.yml up -d --wait
python -m scripts.redis_admin
```

同时使用仓库 PostgreSQL 时，Compose 命令使用两个 -f 文件。Redis Compose 绑定本机、启用认证、256 MB 上限和 allkeys-lru，不启用磁盘持久化。

redis_admin 会写入一次隔离测试缓存、验证命中并清理测试键；它不是纯只读命令，也不会调用真实研究模型或搜索接口。

## 统计口径

- search.hits / misses：工具包装层缓存查询次数。
- search.external_calls：调用原搜索工具的次数；工具内部可能发出多个 HTTP 请求，不能直接等同网络请求总数。
- embedding.hits / misses：文本条数。
- embedding.external_calls：向量接口调用次数。
- waits / wait_timeouts：搜索请求的等待和超时。
- redis.errors / circuit_bypass：缓存故障与旁路。

统计由 ContextVar 隔离，在节点返回时累计进状态。强杀未提交节点会丢失该节点未保存的统计；图外记忆发布也不自动计入节点 cache_stats。测试见[测试与评测](evaluation.md)。
