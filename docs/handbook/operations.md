# 使用与配置

[返回索引](index.md)

## 职责

提供启动、参数生效规则、UI / CLI 操作与统计解释。数据库安装只链接[持久化后端](storage.md)，Redis 安装只链接[缓存](cache.md)，避免维护两份部署说明。

## 本地启动

建议 Python 3.11+。在项目根目录建立并激活虚拟环境，然后：

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

首次将 .env.example 复制为 .env，填入模型与 BGE 服务凭证、地址和模型名，使用 Tavily 时填入其 API Key。删除复制后的 HTTP_PROXY=http:// 和 HTTPS_PROXY=http:// 占位项，或者替换为真实代理地址。

SQLite 为默认持久化后端，无需 Docker；数据写在项目 data 目录，或 APEXLOGIC_DATA_DIR 指定目录。

## 配置来源与生效时机

[run_config.py](../../core/run_config.py) 负责快照、校验和 ContextVar 绑定。节点通过 setting 读取配置，不临时修改进程全局环境。

| 类别 | 例子 | 生效规则 |
| --- | --- | --- |
| 研究快照 | 模型名 / URL、检索预算、BGE 参数、评审阈值 | 新任务创建时保存；恢复保持原值 |
| 记忆策略快照 | memory、memory_first、namespace | 同上；旧任务缺字段走兼容默认 |
| 任务时间 | research_as_of | 创建时冻结，恢复不变成当前日期 |
| 密钥 | DEEPSEEK_API_KEY、BGE_EMBED_API_KEY 等 | 不保存到快照，执行时读取当前环境 |
| 连接、缓存与并发运行参数 | PostgreSQL DSN、Redis 开关 / 连接 / TTL、强制刷新、检索线程与全局名额、排队容量 | 当前环境；应用重启后使用新配置 |

变更 .env 后要体验新的研究策略，应重启应用并新建任务，不直接改历史配置和哈希。

## 常用参数定位

| 功能 | 参数 | 规则的主文档 |
| --- | --- | --- |
| 评审 | MAX_REVISIONS、REVIEWER_PASS_THRESHOLD、REVIEWER_ALLOW_DEGRADED_PASS | [写作与评审](writing-review.md) |
| 搜索 | SEARCH_QUERY_BUDGET、GRAPH_EXPAND_QUERIES、AQD_*、MAX_HOPS | [检索编排](retrieval.md) |
| 广搜并发 | APEXLOGIC_BROAD_MAX_WORKERS、APEXLOGIC_BROAD_*_MAX_INFLIGHT | [检索编排](retrieval.md) |
| AQD、IRCoT 与图扩展并发 | APEXLOGIC_AQD_MAX_WORKERS、APEXLOGIC_IRCOT_MAX_WORKERS、APEXLOGIC_GRAPH_MAX_WORKERS | [检索编排](retrieval.md) |
| 后台容量 | APEXLOGIC_GLOBAL_*_MAX_INFLIGHT、APEXLOGIC_GLOBAL_*_PER_MINUTE、APEXLOGIC_QUEUE_MAX_PENDING、APEXLOGIC_PROVIDER_WAIT_SECONDS | [执行与恢复](execution.md) |
| 报告追问 | APEXLOGIC_FOLLOWUP_MAX_PENDING、APEXLOGIC_GLOBAL_FOLLOWUP_MAX_INFLIGHT | [操作识别与报告追问](intent-and-followup.md) |
| 排序 | BGE_RETRIEVER_*、BGE_RERANKER_*、BGE_EMBED_*、BGE_RERANK_* | [排序](ranking.md) |
| 记忆内容 | MEMORY_ENABLED、MEMORY_TOP_K、MEMORY_MIN_SCORE、MEMORY_TTL_DAYS、MEMORY_CHAR_BUDGET | [来源记忆](memory.md) |
| 免搜策略 | MEMORY_FIRST_MODE | [记忆优先调度](memory-policy.md) |
| 持久化 | APEXLOGIC_STORAGE_BACKEND、APEXLOGIC_POSTGRES_DSN | [存储](storage.md) |
| 缓存 | APEXLOGIC_CACHE_ENABLED、APEXLOGIC_*_CACHE_TTL、APEXLOGIC_CACHE_FORCE_REFRESH | [缓存](cache.md) |

MAX_REVISIONS 的示例值为 4，未配置回退 3；GRAPH_EXPAND_QUERIES 示例为 4，Researcher 未配置回退 2。不能把示例文件所有值都当作代码默认值。

MEMORY_FIRST_MODE 和 Redis 参数需按需加入 .env；不要假设示例文件包含全部可用开关。

## 后台 Worker（PostgreSQL）

先安装 PostgreSQL 和 Redis 可选依赖，运行 `python -m scripts.postgres_admin init` 应用后台调度版本 3、来源速率版本 4 与追问版本 5 迁移。`compose.redis.yml` 的 Redis 是可淘汰缓存，**不能**作为任务队列；另用 `compose.queue.yml` 启动启用 AOF、`noeviction` 和独立数据卷的 Redis。设置 `.env` 中的 `APEXLOGIC_QUEUE_REDIS_PASSWORD`、`APEXLOGIC_QUEUE_REDIS_URL`（默认端口 6380），URL 密码须与 Compose 密码相同。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-redis.txt
docker compose -f compose.postgres.yml -f compose.queue.yml up -d --wait
.\.venv\Scripts\python.exe -m scripts.postgres_admin init
.\.venv\Scripts\python.exe worker.py
```

另一个终端运行 `streamlit run app.py`，选择 PostgreSQL 后提交。关闭 UI 不会结束 Worker；重新打开可查看调度状态、最近节点、checkpoint 和报告。关闭 Worker 会留下租约，重启后最迟在租约过期并完成一次扫描时接管。取消先写入调度记录，最早在下一个节点边界生效。Worker 与 UI 都需要相同的 PostgreSQL 连接配置；研究密钥仅需 Worker 使用。

Worker 完成后会写入 `appstats/run_<run_id>.json`。历史记录按完成时间排序；如果文件缺失或损坏，PostgreSQL 已完成任务仍会列在历史列表，打开时用已保存的 checkpoint 补建并显示相同的历史详情，不重跑研究。`APEXLOGIC_HISTORY_DIR` 若设置，相对路径按项目根目录解析，UI 和 Worker 使用同一个目录。

PostgreSQL 已完成任务的详情和历史页还有“报告追问与研究操作”。输入追问后，页面只写入 PostgreSQL 追问记录，Worker 在后台从原任务 checkpoint 读取 S/R 证据生成回答；关闭或重开 Streamlit 不会丢失排队进度和已完成问答。明确的取消、恢复、查看指令沿用现有任务调度。更新、改写、核验现在可识别但尚未接通对应执行；不会偷偷按普通追问处理。SQLite 旧任务仍可查看和按旧路径恢复，但没有后台追问。

可启动多个 `worker.py` 进程。PostgreSQL 会话锁控制同时执行的研究任务和各搜索源的在途调用；Worker 强杀后连接关闭即释放名额。逐来源的分钟请求额度记录在 PostgreSQL，强杀后仍保留窗口内的已发请求，避免恢复时突发超额。队列满时 UI 明确提示，取消排队任务可腾出名额；符合领取条件的任务按创建时间排队，UI 显示查询时的等待队列位置。所有 Worker 应使用相同的全局上限配置。若 Redis 不可用，已启动的 Worker 会继续扫描 PostgreSQL；新 Worker 仍需连接队列实例后启动。跨进程公平性是任务领取顺序，不保证外部请求排队公平。

## CLI 与导出

```bash
python main.py --topic "你的研究主题"
python main.py --list-runs
python main.py --status RUN_ID
python main.py --resume RUN_ID --storage-backend postgres
python export_report.py --run-id RUN_ID --storage-backend postgres --output-mode both
```

RUN_ID 替换为实际任务身份，并使用原后端。main.py 输出模式为 user / debug；导出支持 user / debug / both / user_only。使用 export_report.py --topic 会新运行研究，--run-id 则读取已完成任务，不重新调用研究模型。

--status 和导出可能触发状态元数据校正；严格只读查询用数据库只读事务。具体原因见[执行与恢复](execution.md)。

## Streamlit 中看什么

- Researcher：计划与实际查询、搜索源配额、AQD 逐题资料、IRCoT 补搜。
- Writer：草稿和修订映射；模型返回的推理文本是展示内容，不是完整内部计算的观测。
- Reviewer：评分、降级模式、反馈与路由。
- 任务状态：下一节点、执行尝试、已记录耗时。
- 完成结果：报告引用、下载、记忆与缓存统计、记忆发布尝试历史。

render_aqd_subquestion 同时供实时页和历史页使用。旧 JSON 没有逐题资料字段时只能展示已有信息，不会自动补采或重跑旧研究。

## 常见误读

| 现象 | 应如何解释 |
| --- | --- |
| 搜索命中低 | 检查实际 query、provider、参数、TTL 和时效旁路；同主题不等于相同请求 |
| 向量命中高 | 表示复用了文本向量，不能据此认定省去了联网搜索 |
| 召回多、入选少 | 查看相关性、预算与排序；召回不是引用 |
| 完成报告但记忆未发布 | 检查 skipped 原因或失败阶段，不先重跑研究 |
| 仍显示 running | 可能活跃或上次强杀未校正，结合锁、checkpoint 和结束记录判断 |

这类诊断只解释当前记录，不保证历史缺失信息可以恢复。
