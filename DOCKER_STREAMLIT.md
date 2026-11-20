# Streamlit Docker 启动说明

新增的 `streamlit` 服务属于现有 Compose 项目 `apexlogic`。已有 PostgreSQL、缓存 Redis、队列 Redis 的 Compose 文件、容器和数据卷保持原样。

## 启动与访问

首次构建或代码更新后，在本项目目录执行：

```powershell
docker compose -p apexlogic -f compose.streamlit.yml up -d --build --no-deps streamlit
```

平时在 Docker Desktop 中启动 `apexlogic` 分组即可，页面地址为：

```text
http://localhost:8501
```

服务使用 `restart: unless-stopped`。Docker 引擎重启时会恢复此前运行的容器；手动停止过的容器需要再次启动。容器启动入口会等待数据库与队列就绪，再沿用页面的 Worker 管理规则预启动 Worker，并每 5 秒继续协调，关闭网页也能补齐退出的 Worker。数据库中已经保存的目标数量优先，包括用户主动设置的 0；页面仍可调整数量。快速重启时，旧 Worker 心跳记录可能需要约 20 秒失效，新 Worker 才会补齐。

仅重启新增服务：

```powershell
docker compose -p apexlogic -f compose.streamlit.yml restart streamlit
```

查看启动日志和状态：

```powershell
docker compose -p apexlogic -f compose.streamlit.yml logs --tail 100 streamlit
docker compose -p apexlogic -f compose.streamlit.yml ps streamlit
```

此配置只声明新服务，Compose 可能提示同项目其他容器未出现在当前文件中。不要添加 `--remove-orphans`，也不需要对原来三个服务执行 `up` 或 `down`。

## 网络与代理

实际验证发现，当前 Docker Host 网络能访问已发布的数据库端口，却不能通过回环地址直达 Windows 代理。因此应用使用普通 bridge 网络并复用 `apexlogic_default`，以确保 8501 端口映射正常；无需修改 Docker Desktop 全局网络设置。

宿主机 `.env` 原样读取，仅在新增容器的进程环境中转换原有本机服务地址：

| 原本机配置 | 容器内目标 |
| --- | --- |
| PostgreSQL `127.0.0.1:5433` | `postgres:5432` |
| 缓存 Redis `127.0.0.1:6379` | `redis:6379` |
| 队列 Redis `127.0.0.1:6380` | `queue:6379` |

只有 loopback 地址会被转换；远程服务地址保持原样，认证信息与其他选项保留。此转换不修改 `.env` 和历史任务配置快照。

`HTTP_PROXY` 和 `HTTPS_PROXY` 继续使用 `http://127.0.0.1:7890`。容器内的轻量 TCP 转发把该端口连接到 `host.docker.internal:7890`，Windows 代理需要保持运行。这个转发在 Streamlit 容器内部，不会增加第五个容器，也不向宿主机发布 7890。健康检查直接连接本地 Streamlit，不经过代理。

镜像构建代理默认使用 `http://host.docker.internal:7890`；可通过 `APEXLOGIC_BUILD_PROXY` 改变。它与应用运行时的回环代理配置分开。

## 配置和数据

- `.env` 作为容器运行时配置读取，不复制到镜像。
- `data/`、`appstats/`、`reports/` 分别挂载到容器同名目录。
- 数据库与 checkpoint 继续保存在现有 PostgreSQL。
- Worker 仍由现有管理器创建，是 Streamlit 容器里的独立 Python 进程。
- 关闭网页不结束后台研究；停止 Streamlit 容器会结束其中的进程，未完成任务依赖原有 checkpoint 和调度机制恢复。
- 应用容器不会自动执行数据库迁移。代码版本需要新增迁移时，按项目原有迁移流程操作。
- 容器运行期以普通用户执行。需要更新源代码时重新构建；Windows `.venv` 不进入镜像。

`requirements-docker.txt` 固定了当前本地环境的直接依赖版本，并引用原有三份依赖文件；宿主机依赖不变。

## 验证

本地网络适配与真实 TCP 转发测试：

```powershell
.\.venv\Scripts\python.exe scripts/test_container_streamlit.py
```

容器级验证需要检查：页面健康与渲染、三项服务认证连接、回环代理 HTTPS 请求、Worker 心跳以及重启后恢复服务。检索与模型付费调用不属于简单启动验收，执行完整研究时再使用你的正常任务。

### 本机验收结果（2026-11-20）

| 检查项 | 实际结果 |
| --- | --- |
| Compose 配置校验与镜像构建 | 通过；`pip check` 无依赖冲突 |
| 容器分组与端口 | `apexlogic-streamlit-1`，项目标签 `apexlogic`，`127.0.0.1:8501 → 8501` |
| 页面访问 | 宿主机首页和健康接口均返回 HTTP 200；浏览器显示完整界面及 2 个 Worker |
| 页面运行测试 | 容器内 Streamlit AppTest 渲染通过，0 个异常、0 个错误 |
| PostgreSQL / pgvector | 认证连接正常，向量扩展版本 0.8.6；可读取已有完成任务的 checkpoint |
| 缓存 / 队列 Redis | 均通过认证 PING |
| 代理 | 保留 HTTP/HTTPS 代理 `127.0.0.1:7890`，容器内访问 `https://example.com` 返回 200 |
| Worker | 2 个在线、0 个忙碌，心跳正常，无启动错误 |
| 重启新增容器 | 页面健康检查恢复，2 个 Worker 恢复且心跳刷新 |
| 挂载与运行用户 | 三个挂载目录可写；运行 UID 为 10001；镜像 `/app` 中没有 `.env` |
| 网络适配与 TCP 转发 | 4 项本地测试通过，包含 256 KiB 数据双向转发与半关闭 |
| 离线节点恢复 | 宿主机和容器内均通过：Reviewer 中断后只重跑 Reviewer，已完成的 Researcher / Writer 不重跑 |
| 原有基础设施 | 三个容器的 ID、启动时间、重启计数、端口、挂载、配置哈希均与操作前一致；原三份 Compose 和 `.env` 文件哈希不变 |

本次没有提交新的研究任务，也没有调用搜索或模型接口；离线恢复测试使用临时 SQLite 和模拟节点，不会修改正式研究数据。正式任务队列仍为 9 条 completed、5 条 cancelled。数据库中增加或更新的 Worker 目标与心跳属于正常服务运行记录。
