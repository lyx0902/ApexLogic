# 持久化后端

[返回索引](index.md)

## 职责

提供任务仓库、记忆仓库、执行锁和 Checkpointer 的后端实现，不负责搜索或评审规则。

统一工厂：[core/storage.py](../../core/storage.py)。SQLite：[persistence.py](../../core/persistence.py)、[run_repository.py](../../core/run_repository.py)。PostgreSQL：[postgres.py](../../core/postgres.py)、[postgres_repository.py](../../core/postgres_repository.py)。

## 存储布局

| 对象 | SQLite | PostgreSQL |
| --- | --- | --- |
| runs、attempts | data/runs.sqlite | apexlogic |
| LangGraph checkpoint 及配套表 | data/checkpoints.sqlite | apexlogic_checkpoints |
| memory_items、memory_embeddings | data/memory.sqlite | apexlogic |
| memory_relations、memory_access_log | data/memory.sqlite | apexlogic |
| memory_publications、memory_publication_attempts | data/memory.sqlite | apexlogic |
| schema_migrations、memory_imports | 不使用同名迁移机制 | apexlogic |
| research_jobs、research_outbox | 未实现后台队列 | apexlogic |

APEXLOGIC_DATA_DIR 或 CLI --data-dir 可改变 SQLite 目录。PostgreSQL 记忆不依赖本地 SQLite 目录。页面历史在 appstats，导出在 reports，二者不属于 Checkpointer。

SQLite 采用 WAL 等设置支持本地执行。PostgreSQL 业务查询使用连接池；持有执行锁的专用会话与 checkpoint 写入绑定，避免锁失效后仍由另一连接继续提交节点。

## PostgreSQL 部署

当前 Compose 使用 pgvector/pgvector:0.8.6-pg17。数据库和用户为 apexlogic，宿主端口默认 5432；示例可选择 5433 与原生服务隔离。

在 .env 中设置，替换密码占位值：

```dotenv
APEXLOGIC_POSTGRES_PORT=5433
APEXLOGIC_POSTGRES_PASSWORD=REPLACE_ME
APEXLOGIC_POSTGRES_DSN="host=127.0.0.1 port=5433 dbname=apexlogic user=apexlogic password=REPLACE_ME"
```

```bash
python -m pip install -r requirements-postgres.txt
docker compose -f compose.postgres.yml up -d --wait
python -m scripts.postgres_admin init
python -m scripts.postgres_admin check
```

成功后设置 APEXLOGIC_STORAGE_BACKEND=postgres 并重启应用。已有数据卷的账号密码由数据库保存，单纯改 Compose 环境变量不会自动改已有数据库角色密码。

## 初始化与迁移

init 使用迁移锁和 checksum：

- 版本 1 读取 migrations/postgres/001_initial.sql。
- 版本 2 使用 memory/publication_log.py 中的 SCHEMA 新增发布尝试日志。
- 版本 3 读取 migrations/postgres/003_background.sql，新增调度与 outbox 表。
- 最后调用 PostgresSaver.setup 初始化 checkpoint 结构。

已应用迁移校验和不同会报错，不能通过编辑旧 SQL 让现有数据库悄悄接受新结构。部署文件必须包含版本 1 SQL；当前仓库忽略规则涉及 migrations，打包时应检查该文件是否实际包含。

check 验证基本 schema、pgvector 与 checkpoint 表，并不等于检查每一张表的数据完整性或所有迁移均已应用。

## SQLite 记忆导入

```bash
python -m scripts.postgres_admin import-memory --source data/memory.sqlite
python -m scripts.postgres_admin import-memory --source data/memory.sqlite --apply
```

第一条读取并预览，第二条提交到 PostgreSQL。导入使用来源摘要回执保证相同输入重放幂等。目前导入 memory_items、memory_embeddings、memory_publications、memory_relations、memory_access_log 五张表，不包含 memory_publication_attempts 的历史尝试日志；表清单以 scripts/postgres_admin.py 的 TABLES 为准。

这不是全项目数据搬迁：不会将旧任务和 checkpoint 转换成 PostgreSQL 任务。切换后端后，应回到原后端恢复旧任务。

## 严格只读观察

PyCharm 连接 host=127.0.0.1、port=实际端口、database/user=apexlogic，使用本地配置的密码；勾选 apexlogic 与 apexlogic_checkpoints 两个 schema。

以下为 PostgreSQL 查询示例：

```sql
BEGIN READ ONLY;
SELECT run_id, topic, status, termination_reason, completed_at
FROM apexlogic.runs ORDER BY created_at DESC LIMIT 10;

SELECT id, title, status, observed_at, valid_until
FROM apexlogic.memory_items ORDER BY observed_at DESC LIMIT 20;

SELECT run_id, status, item_ids, skip_reason
FROM apexlogic.memory_publications ORDER BY updated_at DESC LIMIT 10;

SELECT attempt_id, started_at, details::jsonb
FROM apexlogic.memory_publication_attempts
ORDER BY started_at DESC LIMIT 20;
ROLLBACK;
```

不要把业务表查询条数等同于本次新增条数；记忆去重和多个发布回执会关联已有证据。恢复行为见[执行与恢复](execution.md)，证据字段语义见[来源记忆](memory.md)。
