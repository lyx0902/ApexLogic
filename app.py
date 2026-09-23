"""
ApexLogic 深度研究引擎 — Streamlit 可视化界面
"""

from __future__ import annotations

from contextlib import closing
from itertools import chain

from dotenv import load_dotenv
load_dotenv()  # 必须在任何读取 os.getenv 的模块导入前执行

import os
import re
import traceback
from datetime import datetime
from uuid import uuid4

import streamlit as st

from export_report import _inject_citation_hyperlinks
from core.history import (load_history_list, load_or_rebuild_run,
                          merge_completed_runs, save_run_to_history, snapshot_for_event)
from core.runner import ResearchRunner
from core.intent import classify_operation
from core.conversation import ConversationRepository

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexLogic",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS（主题自适应：不硬编码深色背景，跟随 Streamlit Light/Dark 主题）─────────
st.markdown(
    """
<style>
/* Sidebar：仅保留右边框，背景色由 Streamlit 主题控制 */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(30, 58, 95, 0.35);
}
/* Primary button glow */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #0055cc 0%, #003388 100%);
    border: none;
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 0.06em;
    transition: box-shadow 0.2s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 16px rgba(0, 100, 255, 0.55);
}
/* Expander card */
div[data-testid="stExpander"] {
    border: 1px solid rgba(30, 58, 95, 0.35);
    border-radius: 8px;
    margin-bottom: 8px;
}
/* Metric card：仅边框，无背景，跟随主题 */
div[data-testid="stMetric"] {
    border: 1px solid rgba(30, 58, 95, 0.35);
    border-radius: 6px;
    padding: 6px 10px;
}
/* Inline code */
code {
    color: #00a884;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── 通用辅助函数 ───────────────────────────────────────────────────────────────

def extract_think(text: str) -> tuple[list[str], str]:
    """分离 <think>...</think> 块，返回 (思维链列表, 去标签正文)。"""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinks = pattern.findall(text)
    clean = pattern.sub("", text).strip()
    return thinks, clean


def _format_elapsed(seconds: float) -> str:
    """将秒数格式化为 'X分Y秒' 或 'Y秒'。"""
    total_s = int(seconds)
    m, s = divmod(total_s, 60)
    return f"{m}分{s}秒" if m > 0 else f"{s}秒"


def render_live_timer(
    placeholder: "st.delta_generator.DeltaGenerator",
    start_time: datetime,
    stop_seconds: float | None = None,
) -> None:
    """渲染前端秒级计时器；stop_seconds 不为 None 时停止并显示最终时长。"""
    timer_id = f"apex_timer_{int(start_time.timestamp() * 1000)}"
    start_ms = int(start_time.timestamp() * 1000)
    if stop_seconds is None:
        html = f"""
<div style="font-size:0.88rem;color:rgba(49,51,63,0.6);margin-bottom:4px;">运行总时长</div>
<div id="{timer_id}" style="font-size:1.55rem;font-weight:700;">0秒</div>
<script>
(function() {{
  const key = "{timer_id}";
  const startMs = {start_ms};
  const format = (sec) => {{
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m > 0 ? `${{m}}分${{s}}秒` : `${{s}}秒`;
  }};
  if (!window.__apexTimerHandles) window.__apexTimerHandles = {{}};
  if (window.__apexTimerHandles[key]) clearInterval(window.__apexTimerHandles[key]);
  const tick = () => {{
    const el = document.getElementById(key);
    if (!el) return;
    const sec = Math.max(0, Math.floor((Date.now() - startMs) / 1000));
    el.innerText = format(sec);
  }};
  tick();
  window.__apexTimerHandles[key] = setInterval(tick, 1000);
}})();
</script>
"""
    else:
        total_s = int(stop_seconds)
        m, s = divmod(total_s, 60)
        display = f"{m}分{s}秒" if m > 0 else f"{s}秒"
        html = f"""
<div style="font-size:0.88rem;color:rgba(49,51,63,0.6);margin-bottom:4px;">运行总时长</div>
<div id="{timer_id}" style="font-size:1.55rem;font-weight:700;">{display}</div>
<script>
(function() {{
  const key = "{timer_id}";
  if (window.__apexTimerHandles && window.__apexTimerHandles[key]) {{
    clearInterval(window.__apexTimerHandles[key]);
    delete window.__apexTimerHandles[key];
  }}
}})();
</script>
"""
    placeholder.markdown(html, unsafe_allow_html=True)


def render_references(contexts: list) -> None:
    """渲染 Reranker Top-10 参考资料列表。"""
    if not contexts:
        st.info("暂无参考资料信息。")
        return
    for i, ctx in enumerate(contexts, 1):
        if isinstance(ctx, dict):
            title = ctx.get("title", "未知标题")
            raw_url = str(ctx.get("url", "") or "").strip()
            # 兼容历史结构：仅当 source 本身是链接时才回退使用。
            source_fallback = str(ctx.get("source", "") or "").strip()
            url = raw_url if raw_url else (source_fallback if source_fallback.startswith(("http://", "https://")) else "")
            score = ctx.get("bge_reranker_score")
            summary = ctx.get("core_summary", "")
            score_str = f" · 相关度 `{score:.4f}`" if score is not None else ""
            if url:
                st.markdown(f"**{i}.** [{title}]({url}){score_str}")
            else:
                st.markdown(f"**{i}.** **{title}**{score_str}")
            if summary:
                st.caption(summary[:180])
        elif isinstance(ctx, str):
            st.markdown(f"**{i}.** {ctx}")


# ── 历史记录辅助函数 ────────────────────────────────────────────────────────────

def _history_label(record: dict) -> str:
    """生成历史记录的侧边栏展示标签。"""
    data = record["data"]
    try:
        ts = datetime.fromisoformat(data["timestamp"]).astimezone().strftime("%m-%d %H:%M")
    except Exception:
        ts = "??-??"
    topic_short = data["topic"][:14] + ("…" if len(data["topic"]) > 14 else "")
    score = data.get("weighted_score")
    score_text = f"{score:.2f}分" if isinstance(score, (int, float)) else "待打开"
    ok = "✅" if data.get("is_satisfactory") else ("⚠️ 有限结论" if data.get("answer_status") == "limited" else "❌")
    return f"{ts} · {topic_short} · {score_text} {ok}"


def render_publication_attempts(state):
    attempts = state.get("memory_publication_attempts", [])
    if state.get("memory_publication_log_error") or state.get("memory_publication", {}).get("attempt_log_error"):
        st.warning("记忆发布诊断日志未完整保存，请检查数据库；不影响已保存的研究报告。")
    if not attempts:
        fallback = state.get("memory_publication", {}).get("latest_attempt")
        attempts = [fallback] if fallback else []
    if not attempts:
        return
    stages = {"prepare_evidence":"准备证据", "check_existing_vectors":"检查已有向量",
              "read_evidence":"读取证据", "embedding_request":"向量接口请求",
              "write_vector":"写入向量", "finish_publication":"完成发布"}
    triggers = {"research_completed":"研究结束自动发布", "reopen_completed":"打开完成结果重试", "direct":"直接调用"}
    with st.expander("记忆发布尝试历史（最近 50 次）"):
        for attempt in attempts:
            done = attempt.get('completed_before', 0) + attempt.get('completed_this_attempt', 0)
            status = attempt.get('status')
            if status == 'running':
                status = '尚未记录结束（可能执行中或已中断）'
            st.caption(f"{attempt.get('started_at')} · {triggers.get(attempt.get('trigger'), '其他')} · {status}")
            st.write(f"阶段：{stages.get(attempt.get('stage'), attempt.get('stage'))}；向量完成：{done}/{attempt.get('total') if attempt.get('total') is not None else '未知'}；本次新增：{attempt.get('completed_this_attempt', 0)}")
            st.json(attempt)


def render_aqd_subquestion(sub):
    question = sub.get("question", "")
    st.markdown(f"**子问题 {sub.get('id', '?')}：** {question}")
    st.caption("规划查询：" + sub.get("search_query", ""))
    if sub.get("skipped"):
        st.caption("已有记忆覆盖，省去对应搜索" if sub.get("search_skipped") else "与已有查询重复，跳过独立补搜")
    elif sub.get("search_unavailable"):
        st.caption("本题未执行补搜：DuckDuckGo 工具不可用")
    elif sub.get("new_docs") is None:
        st.caption("旧记录未保存逐题资料，结果仅有汇总统计")
    else:
        st.caption(f"本题返回 {sub.get('new_docs', 0)} 条资料（去重前）")
    for record in sub.get("search_records", []) + sub.get("matched_search_records", []):
        status = "失败：" + record.get("error", "unknown") if record.get("status") == "failed" else "完成"
        st.caption(f"{record.get('provider')} · {record.get('query')} · 请求上限 {record.get('requested_results')} · {status}")
    for label, docs in (("搜索资料", sub.get("retrieved_docs", [])), ("召回记忆", sub.get("memory_docs", []))):
        if not docs:
            continue
        st.markdown(f"**{label}**")
        for doc in docs:
            title = doc.get("title") or "未知标题"
            url = doc.get("url") or ""
            st.markdown(f"- [{title}]({url})" if url else f"- {title}")
            if doc.get("search_provider"):
                st.caption(f"来源：{doc['search_provider']} · 查询：{doc.get('query', '')}")
            if "selected" in doc:
                selection = f"入选普通写作上下文 [{doc.get('citation_id')}]" if doc['selected'] else "未入选普通写作上下文"
                st.caption(selection + (" · 搜索缓存命中" if doc.get('cache_hit') else ""))
            if doc.get("excerpt"):
                st.text(doc["excerpt"])


def render_research_operations(runner: ResearchRunner, run_id: str,
                               max_revisions: int, pass_threshold: float) -> None:
    """Show durable report operations and versioned results for one PG run."""
    from core.background import BackgroundScheduler
    from core.service_limits import QueueFullError

    st.markdown("---")
    st.subheader("💬 报告追问与研究操作")
    st.caption("追问和改写使用已保存报告；更新与核验会检索新来源。新报告版本单独保存，原报告不变。")
    try:
        repo = ConversationRepository(runner)
    except Exception as exc:
        st.warning(f"研究操作暂不可用（{type(exc).__name__}）；请检查 PostgreSQL 版本 6 迁移和连接。")
        return
    key = f"research_operation_{run_id}"
    token_key = f"research_operation_token_{run_id}"
    if token_key not in st.session_state:
        st.session_state[token_key] = uuid4().hex
    command = st.text_area("输入追问或操作", key=key,
                           placeholder="例如：报告中 HNSW 与 IVF 的主要差异是什么？")
    mode = st.selectbox("操作类型", ["自动识别", "追问", "更新", "改写", "核验"],
                        key=f"research_operation_mode_{run_id}")
    if st.button("提交研究操作", key=f"submit_operation_{run_id}"):
        if mode == "自动识别":
            decision = classify_operation(command, selected_run_id=run_id)
        else:
            from core.intent import Operation
            decision = (Operation("clarify", clarification="请先输入操作内容。")
                        if not command.strip() else
                        Operation({"追问": "follow_up", "更新": "update", "改写": "rewrite",
                                   "核验": "verify"}[mode], target_run_id=run_id))
        if decision.intent == "clarify":
            st.info(decision.clarification)
        else:
            try:
                if decision.intent in {"follow_up", "update", "rewrite", "verify"}:
                    repo.submit(decision.target_run_id, command,
                                idempotency_key=st.session_state[token_key],
                                request={"intent": decision.intent,
                                         "time_scope": decision.time_scope,
                                         "constraints": decision.constraints,
                                         "output_format": decision.output_format})
                    st.session_state[token_key] = uuid4().hex
                    if decision.target_run_id != run_id:
                        st.session_state.active_run_id = decision.target_run_id
                        st.session_state.view_history = False
                    st.rerun()
                elif decision.intent == "new_research":
                    record = BackgroundScheduler(runner).submit(
                        decision.topic, max_revisions=max_revisions,
                        pass_threshold=pass_threshold, output_mode="user")
                    st.session_state.active_run_id = record["run_id"]
                    st.session_state.view_history = False
                    st.rerun()
                else:
                    target = decision.target_run_id
                    runner._record(target)
                    if decision.intent == "cancel":
                        BackgroundScheduler(runner).cancel(target)
                    elif decision.intent == "resume":
                        BackgroundScheduler(runner).enqueue(target)
                    st.session_state.active_run_id = target
                    st.session_state.view_history = False
                    st.rerun()
            except QueueFullError as exc:
                st.warning(str(exc))
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"操作未提交（{type(exc).__name__}）。请检查任务状态与后台配置。")

    try:
        turns = repo.list(run_id)
    except Exception as exc:
        st.warning(f"暂时无法读取操作记录（{type(exc).__name__}）。")
        return
    try:
        versions = repo.list_versions(run_id)
    except Exception as exc:
        st.warning(f"暂时无法读取报告版本（{type(exc).__name__}）。")
        versions = []
    version_by_turn = {item["operation_turn_id"]: item for item in versions}
    source_state = {}
    if any(turn["status"] == "completed" for turn in turns):
        try:
            source_state = runner.peek(run_id)["state"]
        except Exception as exc:
            st.warning(f"暂时无法读取追问引用链接（{type(exc).__name__}），回答正文仍可查看。")
    for turn in turns:
        kind = (turn.get("request_json") or {}).get("intent", "follow_up")
        label = {"follow_up": "追问", "update": "更新", "rewrite": "改写", "verify": "核验"}.get(kind, "操作")
        st.markdown(f"**{label}：** {turn['question']}")
        if turn["status"] == "completed":
            fresh = (turn.get("result_json") or {}).get("fresh_sources") or []
            if kind in {"update", "verify"} and fresh:
                counts = {source: sum(str(item.get("source") or "").lower() == source
                                      for item in fresh) for source in ("duckduckgo", "tavily")}
                st.caption(f"本次新来源：DuckDuckGo {counts['duckduckgo']} 条 · Tavily {counts['tavily']} 条")
                if not all(counts.values()):
                    st.info("本次只有一个检索服务返回可保存来源，跨服务对照不足。")
            st.markdown(_inject_citation_hyperlinks(turn.get("answer") or "",
                source_state.get("retrieved_context", []),
                list(source_state.get("reasoning_contexts", [])) + fresh))
            version = version_by_turn.get(turn["turn_id"])
            if version:
                with st.expander("查看此报告版本", expanded=False):
                    st.caption(f"版本 {version['version_id'][:8]} · 原报告 checkpoint {version['source_checkpoint_id']}")
                    st.markdown(_inject_citation_hyperlinks(version["report_markdown"],
                        source_state.get("retrieved_context", []),
                        list(source_state.get("reasoning_contexts", [])) +
                        list(version.get("source_manifest") or [])))
                    st.download_button("下载此版本 Markdown", version["report_markdown"],
                                       file_name=f"apexlogic-{run_id[:8]}-{version['version_id'][:8]}.md",
                                       key=f"download_version_{version['version_id']}")
        elif turn["status"] == "failed":
            st.warning(f"{label}处理失败；原研究报告未受影响。")
        else:
            st.caption(f"{label}状态：" + ("等待 Worker" if turn["status"] == "queued" else "正在处理"))
    if turns and any(turn["status"] in {"queued", "running"} for turn in turns):
        if st.button("刷新操作状态", key=f"refresh_followup_{run_id}"):
            st.rerun()


def show_history_view(data: dict) -> None:
    """在主区域渲染历史记录详情。"""
    ts_str = ""
    try:
        ts_str = datetime.fromisoformat(data["timestamp"]).astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    st.markdown("## 📋 历史记录查看")
    st.caption(f"研究时间：{ts_str}　｜　主题：{data['topic']}")

    # 研究主题单独一行，避免与其它指标挤在同一行
    st.metric("研究主题", data["topic"][:90] + ("…" if len(data["topic"]) > 90 else ""))
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("实际迭代轮数", data.get("iterations_done", "—"))
    hc2.metric("加权总分", f"{data.get('weighted_score', 0):.2f}")
    hc3.metric("运行总时长", _format_elapsed(data.get("elapsed_seconds", 0)))
    pass_threshold = data.get("pass_threshold")
    hc4.metric(
        "通过阈值",
        f"{float(pass_threshold):.1f}" if pass_threshold is not None else "—",
    )
    st.markdown("---")

    # 研究报告
    st.markdown("## 📄 研究报告")
    report = data.get("final_report", "")
    if report:
        _, clean = extract_think(report)
        clean = _inject_citation_hyperlinks(
            clean,
            data.get("references", []),
            data.get("reasoning_references", []),
        )
        st.markdown(clean)
        safe_t = re.sub(r"[^\w\u4e00-\u9fff]", "_", data["topic"])[:20]
        st.download_button(
            "⬇️ 下载 Markdown 报告",
            data=clean,
            file_name=f"ApexLogic_{safe_t}_历史.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning("该记录无最终报告。")

    # 参考资料
    run_metadata = data.get("run_metadata", {})
    with st.expander("📚 参考资料", expanded=False):
        refs = data.get("references", [])
        if refs:
            for i, ref in enumerate(refs, 1):
                title = ref.get("title", "未知标题")
                url = ref.get("url", "")
                score = ref.get("score")
                summary = ref.get("summary", "")
                score_str = f" · 相关度 `{score:.4f}`" if score is not None else ""
                if url:
                    st.markdown(f"**{i}.** [{title}]({url}){score_str}")
                else:
                    st.markdown(f"**{i}.** **{title}**{score_str}")
                if summary:
                    st.caption(summary[:180])
        else:
            st.info("该记录无参考资料。")

        if run_metadata.get("cache_stats"):
            with st.expander("历史缓存统计"):
                st.json(run_metadata["cache_stats"])
        render_publication_attempts(run_metadata)
        if run_metadata.get("memory_first"):
            with st.expander("历史记忆优先检索"):
                st.json(run_metadata["memory_first"])
        if run_metadata.get("memory_stats") or run_metadata.get("memory_publication"):
            with st.expander("历史记忆使用记录"):
                st.json({k: run_metadata.get(k) for k in ("memory_stats", "memory_used_ids", "memory_publication")})

    # ── IRCoT 推理链参考文献（历史记录）────────────────────────────────────
    h_reasoning_enabled = run_metadata.get("reasoning_enabled", False)
    h_reasoning_contexts = run_metadata.get("reasoning_contexts", [])

    if h_reasoning_enabled and h_reasoning_contexts:
        with st.expander("🧠 IRCoT 推理链参考文献", expanded=False):
            # 简略统计
            h_reasoning_chains = run_metadata.get("reasoning_chains", [])
            h_stat_col1, h_stat_col2, h_stat_col3 = st.columns(3)
            h_stat_col1.metric("推理跳数", len(h_reasoning_chains))
            h_stat_col2.metric("推理链文档数", len(h_reasoning_contexts))

            # 统计报告中的推理链引用
            h_final_draft = data.get("final_report", "")
            h_r_citations = len(re.findall(r'\[R\d+\]', h_final_draft))
            h_chain_citations = len(re.findall(r'\[推理链\d+\]', h_final_draft))
            h_stat_col3.metric("报告中引用次数", h_r_citations + h_chain_citations)

            st.caption(
                "以下文档由IRCoT多跳推理主动发现，独立保存在推理链通道中，"
                "不经过BGE筛选，确保推理链补搜的关键信息不被丢弃。"
            )

            # 展示推理链文档列表
            for idx, doc in enumerate(h_reasoning_contexts, 1):
                title = doc.get("title", "未知标题")
                url = doc.get("url", "")
                summary = doc.get("summary", "")

                if url:
                    st.markdown(f"**[R{idx}]** [{title}]({url})")
                else:
                    st.markdown(f"**[R{idx}]** {title}")

                if summary:
                    st.caption(summary)

    # 运行期警告（如有）
    errors = data.get("errors", [])
    if errors:
        with st.expander(
            f"⚠️ 运行期警告（共 {data.get('errors_count', len(errors))} 条）",
            expanded=False,
        ):
            for err in errors:
                st.warning(err)

    # ── 各阶段运行元数据（默认折叠，与直播 UI 完全一致）──────────────────────────
    run_metadata = data.get("run_metadata", {})
    h_snapshots = run_metadata.get("iteration_snapshots", [])
    if h_snapshots:
        with st.expander("🔬 各阶段运行元数据（点击展开查看详细过程）", expanded=False):
            _threshold = data.get("pass_threshold") or 0.0
            for snap in h_snapshots:
                snode = snap.get("node")
                sit = snap.get("iteration", "?")

                # ── Researcher ────────────────────────────────────────────────
                if snode == "researcher":
                    st.markdown(f"### 🔍 检索阶段（第 {sit} 轮）")

                    h_queries: list = snap.get("search_queries", [])
                    if h_queries:
                        st.markdown("**📋 本轮使用的检索词：**")
                        for q in h_queries:
                            st.markdown(f"- `{q}`")

                    h_mab: dict = snap.get("mab_state", {}) or {}
                    if h_mab:
                        with st.expander("📊 MAB 预算分配（Thompson Sampling）", expanded=False):
                            st.json(h_mab)

                    h_ctx_len = snap.get("retrieved_context_count", 0)
                    h_aqd: dict = snap.get("query_plan", {}) or {}
                    h_ircot: dict = snap.get("iterative_retrieval_summary", {}) or {}
                    h_sqs = snap.get("source_quality_summary", {}) or {}
                    h_dedup = (
                        h_sqs.get("bge_summary", {}).get("dedup_total", 0)
                        if isinstance(h_sqs, dict)
                        else 0
                    )

                    hmc1, hmc2, hmc3, hmc4 = st.columns(4)
                    hmc1.metric("去重后总检索数", h_dedup)
                    hmc2.metric("命中上下文数", h_ctx_len)
                    hmc3.metric(
                        "AQD 子问题数",
                        h_aqd.get("sub_questions_count", 0) if h_aqd.get("enabled") else "—",
                    )
                    hmc4.metric(
                        "IRCoT 跳数",
                        h_ircot.get("hops_executed", 0) if h_ircot.get("enabled") else "—",
                    )

                    if h_aqd.get("enabled"):
                        h_sub_results = h_aqd.get("sub_results", [])
                        h_total_new = h_aqd.get("total_new_docs", 0)
                        with st.expander(
                            f"🧩 AQD 查询分解详情（{len(h_sub_results)} 个子问题，返回 {h_total_new} 条资料（去重前））",
                            expanded=False,
                        ):
                            for sub in h_sub_results:
                                render_aqd_subquestion(sub)

                    if h_ircot.get("enabled"):
                        h_hop_summaries = h_ircot.get("hop_summaries", [])
                        h_total_gap = h_ircot.get("gap_contexts_added", 0)
                        h_hops_done = h_ircot.get("hops_executed", 0)
                        with st.expander(
                            f"🔁 IRCoT 迭代推理检索详情（{h_hops_done} 跳，共补搜 {h_total_gap} 条）",
                            expanded=False,
                        ):
                            for i, hop_s in enumerate(h_hop_summaries):
                                hop_num = hop_s.get("hop", i + 1)
                                hop_status = hop_s.get("status", "")
                                st.markdown(f"**第 {hop_num} 跳**")
                                gap_queries_list = hop_s.get("gap_queries", [])
                                if gap_queries_list:
                                    st.markdown("**补充查询：**")
                                    for gq in gap_queries_list:
                                        st.markdown(f"- `{gq}`")
                                hop_docs = hop_s.get("retrieved_docs", [])
                                if hop_docs:
                                    st.markdown("**补搜资料：**")
                                    for doc in hop_docs:
                                        doc_title = doc.get("title", "") or "未知标题"
                                        doc_url = doc.get("url", "")
                                        if doc_url:
                                            st.markdown(f"- [{doc_title}]({doc_url})")
                                        else:
                                            st.markdown(f"- {doc_title}")
                                elif hop_status == "no_new_gaps":
                                    st.caption("无新增信息缺口，迭代在此跳终止。")

                                # 展示本跳推理链
                                h_reasoning_full = hop_s.get("reasoning_full", "")
                                h_reasoning_preview = hop_s.get("reasoning_preview", "")
                                if h_reasoning_full:
                                    with st.expander("💭 本跳推理链", expanded=False):
                                        st.markdown(h_reasoning_full)
                                elif h_reasoning_preview:
                                    st.caption(f"💭 推理预览：{h_reasoning_preview}")

                                if i < len(h_hop_summaries) - 1:
                                    st.markdown("---")

                            # 展示最终总结推理链（如果有）
                            h_final_summary_hop = None
                            for hop_s in h_hop_summaries:
                                if hop_s.get("status") == "final_summary":
                                    h_final_summary_hop = hop_s
                                    break

                            if h_final_summary_hop:
                                st.markdown("")  # 空行分隔
                                with st.expander("🎯 最终总结推理链", expanded=False):
                                    st.caption(
                                        "💡 基于最后一跳补搜的文档生成的总结性推理，"
                                        "确保所有检索到的信息都被充分利用。"
                                    )
                                    h_final_reasoning = h_final_summary_hop.get("reasoning_full", "")
                                    if h_final_reasoning:
                                        st.markdown(h_final_reasoning)
                                    else:
                                        st.caption("（无最终总结内容）")

                            # 展示完整推理链记录（增强版：显示跨轮记忆）
                            h_reasoning_chains = run_metadata.get("reasoning_chains", [])
                            if h_reasoning_chains:
                                st.markdown("")  # 空行分隔
                                with st.expander(
                                    f"📜 完整推理链记录（共 {len(h_reasoning_chains)} 跳，跨轮累积）",
                                    expanded=False
                                ):
                                    st.caption(
                                        "💡 推理链在多轮检索中累积，每轮从上一轮的最后结论继续推理，"
                                        "避免重复推理已知信息。"
                                    )

                                    # 判断当前轮次和本轮新增的跳数
                                    h_current_iter = snap.get("iteration", 1)
                                    h_hops_this_round = h_ircot.get("hops_executed", 0)

                                    for i, chain in enumerate(h_reasoning_chains, 1):
                                        # 判断是否为历史轮次的推理链
                                        is_from_previous = (
                                            h_current_iter > 1 and
                                            i <= len(h_reasoning_chains) - h_hops_this_round
                                        )

                                        if is_from_previous:
                                            st.markdown(f"**第 {i} 跳推理（继承自历史轮次）** 🔗")
                                        else:
                                            st.markdown(f"**第 {i} 跳推理（本轮新增）** ✨")

                                        st.markdown(chain)
                                        if i < len(h_reasoning_chains):
                                            st.markdown("---")

                    st.markdown("---")

                # ── Writer ────────────────────────────────────────────────────
                elif snode == "writer":
                    st.markdown(f"### ✍️ 起草阶段（第 {sit} 轮）")

                    h_draft: str = snap.get("draft", "")
                    h_thinks, h_clean_draft = extract_think(h_draft)

                    if h_thinks:
                        with st.expander("💭 DeepSeek 思维链（Chain of Thought）", expanded=False):
                            for t in h_thinks:
                                st.text(t[:2000])

                    if h_clean_draft:
                        st.markdown("**📝 草稿预览（前 600 字）：**")
                        h_preview = h_clean_draft[:600]
                        if len(h_clean_draft) > 600:
                            h_preview += "\n\n*… [ 完整报告见上方「研究报告」区域 ] …*"
                        st.markdown(h_preview)
                    else:
                        st.warning("草稿内容为空。")

                    st.markdown("---")

                # ── Reviewer ──────────────────────────────────────────────────
                elif snode == "reviewer":
                    st.markdown(f"### 🧐 评审阶段（第 {sit} 轮）")

                    h_is_ok: bool = snap.get("is_satisfactory", False)
                    if h_is_ok:
                        st.success("✅ 评审通过！报告质量达标。")
                    elif snap.get("answer_status") == "limited":
                        st.warning("Reviewer 接受有限结论；报告仍有未解决问题。")
                    else:
                        st.warning("本轮评审未通过，修订意见已记录。")

                    h_review: dict = snap.get("review_result", {}) or {}
                    h_scores: dict = h_review.get("scores", {})
                    h_weighted: float = float(h_review.get("weighted_score", 0.0))

                    if h_scores:
                        st.markdown("**📊 四维评分：**")
                        h_score_cols = st.columns(5)
                        for idx, (key, label, weight) in enumerate([
                            ("S1", "事实准确性", "35%"),
                            ("S2", "逻辑完整性", "25%"),
                            ("S3", "信息覆盖", "25%"),
                            ("S4", "结论可执行", "15%"),
                        ]):
                            val = h_scores.get(key, "—")
                            h_score_cols[idx].metric(
                                f"{label}（{weight}）",
                                f"{val}/10" if val != "—" else "—",
                            )
                        h_delta = round(h_weighted - float(_threshold), 2)
                        h_score_cols[4].metric(
                            f"加权总分（阈值 {_threshold}）",
                            f"{h_weighted:.2f}",
                            delta=f"{h_delta:+.2f}",
                            delta_color="normal",
                        )

                    h_feedback: str = snap.get("critique_feedback", "")
                    if h_feedback:
                        st.markdown("**💬 评审意见：**")
                        st.warning(h_feedback)

                    h_directives: dict = snap.get("revision_directives", {}) or {}
                    if h_directives:
                        with st.expander("📋 结构化修订指令", expanded=False):
                            st.json(h_directives)

                    h_route: str = snap.get("next_route", "")
                    route_labels = {
                        "end": "📄 输出最终报告",
                        "researcher": "🔍 → Researcher（补充检索）",
                        "writer": "✍️ → Writer（修订草稿）",
                    }
                    st.info(f"**路由决策：** {route_labels.get(h_route, h_route or '未知')}")



# ── Session state 初始化 ───────────────────────────────────────────────────────
if "view_history" not in st.session_state:
    st.session_state.view_history = False
if "history_data" not in st.session_state:
    st.session_state.history_data = None

storage_choices = ["sqlite", "postgres"]
storage_default = os.getenv("APEXLOGIC_STORAGE_BACKEND", "sqlite")
selected_storage = st.sidebar.selectbox("任务存储", storage_choices,
    index=storage_choices.index(storage_default) if storage_default in storage_choices else 0,
    format_func=lambda value: "SQLite（本地 / 旧任务）" if value == "sqlite" else "PostgreSQL")
try:
    runner = ResearchRunner(backend=selected_storage)
    run_list = runner.repository.list()
except Exception as exc:
    st.error(f"无法打开任务存储（{type(exc).__name__}）。PostgreSQL 请先按 docs/postgres-migration.md 配置并初始化；SQLite 旧任务可切换后端查看。")
    st.stop()
history_list = load_history_list()
if selected_storage == "postgres":
    history_list = merge_completed_runs(history_list, run_list)
resume_run_id = None
inspect_run_id = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ApexLogic 参数配置")
    st.markdown("---")

    topic: str = st.text_input(
        "请输入研究主题",
        placeholder="例如：大型语言模型的推理能力",
    )
    max_revisions: int = st.slider("最大反思轮数", min_value=1, max_value=5, value=3)
    pass_threshold: float = st.slider("报告通过阈值", min_value=1.0, max_value=10.0, value=7.5, step=0.5)

    st.markdown("---")
    start_btn: bool = st.button(
        "🚀 开始深度研究",
        type="primary",
        use_container_width=True,
        disabled=not topic.strip(),
    )

    if run_list:
        st.markdown("---")
        st.markdown("**研究任务 · 断点恢复**")
        by_id = {r["run_id"]: r for r in run_list}
        selected_run = st.selectbox("选择研究任务", [None] + list(by_id),
            format_func=lambda key: "— 选择任务 —" if key is None else
                f"{by_id[key]['topic'][:24]} · {by_id[key]['status']} · {key[:8]}")
        if selected_run:
            if st.button("查看任务状态 / 已有结果"):
                inspect_run_id = selected_run
            if st.button("继续研究 / 打开完成结果"):
                resume_run_id = selected_run
        st.caption("列表为最近记录；查看任务可核对进度。恢复沿用任务原配置。")

    # ── 历史记录区 ──────────────────────────────────────────────────────────────
    if history_list:
        st.markdown("---")
        st.markdown("**📚 历史记录**")
        history_choices = {
            f"{_history_label(entry)} · {index + 1}": entry
            for index, entry in enumerate(history_list)
        }
        selected_history = st.selectbox("历史记录列表", [None] + list(history_choices),
            format_func=lambda choice: "— 选择历史记录 —" if choice is None else choice,
            label_visibility="collapsed")
        if selected_history is not None:
            if st.button("📖 查看此次记录", use_container_width=True):
                entry = history_choices[selected_history]
                try:
                    data = (load_or_rebuild_run(runner, entry["data"]["run_id"])
                            if entry["file"] is None else entry["data"])
                except Exception as exc:
                    st.error(f"无法从 checkpoint 重建历史记录（{type(exc).__name__}）。")
                    st.stop()
                st.session_state.view_history = True
                st.session_state.history_data = data
                st.rerun()

    showing_task = (inspect_run_id is not None or
                    (selected_storage == "postgres" and
                     any(item["run_id"] == st.session_state.get("active_run_id") for item in run_list)))
    if st.session_state.view_history or showing_task:
        if st.button("← 返回新研究", use_container_width=True):
            st.session_state.view_history = False
            st.session_state.history_data = None
            st.session_state.active_run_id = None
            st.rerun()

    st.markdown("---")
    st.caption(
        "**ApexLogic 多智能体深度研究系统**  \n"
        "Thompson Sampling MAB · Graph Expand  \n"
        "AQD · IRCoT · BGE 两阶段精筛  \n\n"
        "运行命令：`streamlit run app.py`"
    )

if (selected_storage == "postgres" and not (start_btn or resume_run_id or inspect_run_id)
        and not st.session_state.view_history):
    active = st.session_state.get("active_run_id")
    if active and any(item["run_id"] == active for item in run_list):
        inspect_run_id = active


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("# 🔬 ApexLogic 多智能体深度研究引擎")
st.caption(
    "LangGraph 多智能体循环：Researcher → Writer → Reviewer "
    "· 四层检索优化 · BGE 两阶段精筛 "
)
st.caption("©️南京理工大学计算机科学与技术22级刘宇翔")
st.markdown("---")


# ── 历史记录浏览模式（优先于新研究，但 start_btn 可覆盖）────────────────────────
if inspect_run_id and not start_btn and not resume_run_id:
    try:
        info = runner.peek(inspect_run_id) if selected_storage == "postgres" else runner.inspect(inspect_run_id)
        item = info["record"]
        st.subheader(item["topic"])
        st.code(item["run_id"])
        if selected_storage == "postgres":
            from core.background import BackgroundScheduler
            scheduler = BackgroundScheduler(runner)
            st.session_state.active_run_id = inspect_run_id
            job = scheduler.status(inspect_run_id)
            if job:
                st.write("后台调度：", job["status"] + ("（等待节点边界取消）" if job["cancel_requested"] else ""))
                if job.get("queue_position"):
                    st.write("等待队列位置：", job["queue_position"])
                st.write("最近完成节点：", job["last_node"] or "尚未完成节点")
                st.write("调度更新时间：", job["updated_at"])
                if job.get("last_error"):
                    if job["last_error"].startswith("HistoryExport:"):
                        st.warning("历史快照导出失败；打开已完成任务时将从 checkpoint 补建，不会重新研究。")
                    else:
                        st.warning(f"Worker 上次尝试：{job['last_error']}，将自动重试。")
                if job["status"] in {"queued", "running"} and not job["cancel_requested"]:
                    if st.button("取消后台研究"):
                        scheduler.cancel(inspect_run_id)
                        st.rerun()
            if st.button("刷新后台进度"):
                st.rerun()
        st.write("研究状态：", item["status"])
        st.write("待执行节点：", info["next"] or "无")
        st.write("最近保存：", info["saved_at"] or "尚未开始")
        st.write("完成时间：", info["record"].get("completed_at") or "尚未完成")
        if item.get("last_error"):
            st.warning(item["last_error"])
        if item.get("termination_reason") == "limited":
            st.warning("研究已结束：Reviewer 接受的有限结论，仍有未解决的问题。")
        if item.get("termination_reason") == "max_revisions":
            st.warning("研究已结束，但未达到评审通过条件。")
        saved = info["state"]
        if selected_storage == "postgres" and item["status"] == "completed":
            try:
                historical_view = load_or_rebuild_run(runner, inspect_run_id)
            except Exception as exc:
                st.warning(f"历史详情暂时无法重建（{type(exc).__name__}），以下展示已保存报告。")
            else:
                show_history_view(historical_view)
                render_research_operations(runner, inspect_run_id, max_revisions, pass_threshold)
                st.stop()
        render_publication_attempts(saved)
        if saved.get("draft"):
            report_text = saved.get("final_report") or saved["draft"]
            st.markdown(_inject_citation_hyperlinks(report_text,
                saved.get("retrieved_context", []), saved.get("reasoning_contexts", [])))
            if item["status"] == "completed":
                st.download_button("下载 Markdown 报告", report_text,
                                   file_name=f"apexlogic_{inspect_run_id}.md", mime="text/markdown")
        if selected_storage == "postgres" and item["status"] == "completed":
            render_research_operations(runner, inspect_run_id, max_revisions, pass_threshold)
        with st.expander("执行尝试与已保存轨迹"):
            st.json({"attempts": info["attempts"], "trace": saved.get("execution_trace", [])})
    except Exception as exc:
        st.error(f"读取任务失败：{exc}")
    st.stop()

if st.session_state.view_history and not start_btn and not resume_run_id:
    if st.session_state.history_data:
        show_history_view(st.session_state.history_data)
        history_run_id = st.session_state.history_data.get("run_id")
        if (selected_storage == "postgres" and history_run_id and
                any(item["run_id"] == history_run_id and item["status"] == "completed"
                    for item in run_list)):
            render_research_operations(runner, history_run_id, max_revisions, pass_threshold)
    else:
        st.warning("历史记录数据丢失，请重新选择。")
    st.stop()

if selected_storage == "postgres" and (start_btn or resume_run_id):
    try:
        from core.background import BackgroundScheduler
        scheduler = BackgroundScheduler(runner)
        if resume_run_id and not start_btn:
            run_id = resume_run_id
            scheduler.enqueue(run_id)
        else:
            record = scheduler.submit(topic, max_revisions=max_revisions,
                                      pass_threshold=pass_threshold, output_mode="user")
            run_id = record["run_id"]
        st.session_state.active_run_id = run_id
        st.rerun()
    except Exception as exc:
        from core.service_limits import QueueFullError
        if isinstance(exc, QueueFullError):
            st.warning("后台等待队列已满，请稍后再提交或取消不需要的排队任务。")
        else:
            st.error(f"提交后台任务失败（{type(exc).__name__}）。请检查 PostgreSQL 迁移和 Worker 配置。")
    st.stop()


# ── 欢迎界面（未启动研究时）───────────────────────────────────────────────────────
if not start_btn and not resume_run_id:
    st.info(
        "👈 请在左侧侧边栏输入研究主题并点击「🚀 开始深度研究」，"
        "系统将自动启动多智能体深度研究流程并在此实时展示执行过程。"
    )
    if history_list:
        st.caption("也可在左侧侧边栏「历史记录」中查看以往的研究结果。")
    st.stop()

# 主题为空保险检查
if not resume_run_id and not topic.strip():
    st.warning("⚠️ 研究主题不能为空，请在侧边栏输入主题后重试。")
    st.stop()

# 进入新研究时清除历史浏览状态
st.session_state.view_history = False
st.session_state.history_data = None


# ── 导入核心模块 ───────────────────────────────────────────────────────────────
try:
    if resume_run_id and not start_btn:
        run_id = resume_run_id
        info = runner.inspect(run_id)
        if info["running"]:
            st.info("任务仍在其他窗口或进程执行，请稍后查看状态。")
            st.stop()
        run_record = info["record"]
    else:
        run_record = runner.create(topic, max_revisions=max_revisions,
                                   pass_threshold=pass_threshold, output_mode="user")
        run_id = run_record["run_id"]
    st.session_state.active_run_id = run_id
    topic = run_record["topic"]
    max_revisions = int(run_record["run_config"]["settings"]["MAX_REVISIONS"])
    pass_threshold = float(run_record["run_config"]["settings"]["REVIEWER_PASS_THRESHOLD"])
    prior_events = runner.history(run_id)
except Exception as exc:
    st.error(f"❌ 无法创建或恢复任务：{exc}")
    st.stop()

st.caption(f"任务 ID：{run_id}。进度保存在本地，重启后可从侧边栏继续。")
st.caption("研究截至时间：" + (run_record["run_config"].get("research_as_of") or "旧任务未记录"))


# ── 运行参数概览（研究主题单行 + 4 列其余参数）──────────
display_start_time = datetime.now()
st.metric("研究主题", topic[:90] + ("…" if len(topic) > 90 else ""))
c1, c2, c3, c4 = st.columns(4)
c1.metric("最大反思轮数", max_revisions)
c2.metric("通过阈值", f"{pass_threshold:.1f}")
c3.metric("启动时间", display_start_time.strftime("%H:%M:%S"))
timer_placeholder = c4.empty()
# run_start_time 和计时器将在 graph.stream() 开始前设定，精确计量执行时长
st.markdown("---")


# ── 状态占位符 ─────────────────────────────────────────────────────────────────
status_placeholder = st.empty()
status_placeholder.info("🚀 引擎启动，多智能体流水线正在初始化……")


# ── 主流式循环 ─────────────────────────────────────────────────────────────────
final_state: dict = {}
iteration_snapshots: list = []

# 计时器：在 graph.stream() 启动前精确计时，与页面加载耗时解耦
run_start_time = datetime.now()
render_live_timer(timer_placeholder, run_start_time)

try:
    with st.spinner("🤖 正在思考与执行中，请耐心等待……"), closing(runner.stream(run_id)) as live_events:
        if prior_events:
            st.caption("以下先展示已保存的执行过程，再继续未完成节点。")
        for event in chain(prior_events, live_events):
            full_state = event.state
            final_state = dict(full_state)
            if event.kind == "complete":
                continue
            node = event.node

            # ── Researcher ────────────────────────────────────────────────────
            if node == "researcher":
                current_iteration = full_state.get("revision_step", 0) + 1
                if current_iteration > 1:
                    st.markdown("---")
                status_placeholder.info(
                    f"🔍 第 {current_iteration} 轮 · Researcher 检索完成，Writer 正在起草……"
                )

                with st.expander(
                    f"🔍 检索阶段完成（第 {current_iteration} 轮）", expanded=True
                ):
                    queries: list = full_state.get("search_queries", [])
                    if queries:
                        st.markdown("**📋 本轮使用的检索词：**")
                        for q in queries:
                            st.markdown(f"- `{q}`")

                    mab: dict = full_state.get("mab_state", {})
                    if mab:
                        with st.expander(
                            "📊 MAB 预算分配（Thompson Sampling）", expanded=False
                        ):
                            st.json(mab)

                    ctx_len = len(full_state.get("retrieved_context", []))
                    aqd: dict = full_state.get("query_plan", {})
                    ircot: dict = full_state.get("iterative_retrieval_summary", {})
                    _sqs = full_state.get("source_quality_summary", {}) or {}
                    dedup_total = (
                        _sqs.get("bge_summary", {}).get("dedup_total", 0)
                        if isinstance(_sqs, dict)
                        else 0
                    )

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("去重后总检索数", dedup_total)
                    mc2.metric("命中上下文数", ctx_len)
                    mc3.metric(
                        "AQD 子问题数",
                        aqd.get("sub_questions_count", 0) if aqd.get("enabled") else "—",
                    )
                    mc4.metric(
                        "IRCoT 跳数",
                        ircot.get("hops_executed", 0) if ircot.get("enabled") else "—",
                    )

                    # ── AQD 详情 ────────────────────────────────────────────
                    if aqd.get("enabled"):
                        sub_results_list = aqd.get("sub_results", [])
                        total_new_docs = aqd.get("total_new_docs", 0)
                        with st.expander(
                            f"🧩 AQD 查询分解详情（{len(sub_results_list)} 个子问题，返回 {total_new_docs} 条资料（去重前））",
                            expanded=False,
                        ):
                            for sub in sub_results_list:
                                render_aqd_subquestion(sub)

                    # ── IRCoT 详情 ──────────────────────────────────────────
                    if ircot.get("enabled"):
                        hop_summaries_list = ircot.get("hop_summaries", [])
                        total_gap = ircot.get("gap_contexts_added", 0)
                        hops_done = ircot.get("hops_executed", 0)
                        with st.expander(
                            f"🔁 IRCoT 迭代推理检索详情（{hops_done} 跳，共补搜 {total_gap} 条）",
                            expanded=False,
                        ):
                            for i, hop_s in enumerate(hop_summaries_list):
                                hop_num = hop_s.get("hop", i + 1)
                                hop_status = hop_s.get("status", "")
                                st.markdown(f"**第 {hop_num} 跳**")

                                gap_queries_list = hop_s.get("gap_queries", [])
                                if gap_queries_list:
                                    st.markdown("**补充查询：**")
                                    for gq in gap_queries_list:
                                        st.markdown(f"- `{gq}`")

                                hop_docs = hop_s.get("retrieved_docs", [])
                                if hop_docs:
                                    st.markdown("**补搜资料：**")
                                    for doc in hop_docs:
                                        doc_title = doc.get("title", "") or "未知标题"
                                        doc_url = doc.get("url", "")
                                        if doc_url:
                                            st.markdown(f"- [{doc_title}]({doc_url})")
                                        else:
                                            st.markdown(f"- {doc_title}")
                                elif hop_status == "no_new_gaps":
                                    st.caption("无新增信息缺口，迭代在此跳终止。")

                                # 展示本跳推理链
                                reasoning_full = hop_s.get("reasoning_full", "")
                                reasoning_preview = hop_s.get("reasoning_preview", "")
                                if reasoning_full:
                                    with st.expander("💭 本跳推理链", expanded=False):
                                        st.markdown(reasoning_full)
                                elif reasoning_preview:
                                    st.caption(f"💭 推理预览：{reasoning_preview}")

                                if i < len(hop_summaries_list) - 1:
                                    st.markdown("---")

                            # 展示最终总结推理链（如果有）
                            final_summary_hop = None
                            for hop_s in hop_summaries_list:
                                if hop_s.get("status") == "final_summary":
                                    final_summary_hop = hop_s
                                    break

                            if final_summary_hop:
                                st.markdown("")  # 空行分隔
                                with st.expander("🎯 最终总结推理链", expanded=False):
                                    st.caption(
                                        "💡 基于最后一跳补搜的文档生成的总结性推理，"
                                        "确保所有检索到的信息都被充分利用。"
                                    )
                                    final_reasoning = final_summary_hop.get("reasoning_full", "")
                                    if final_reasoning:
                                        st.markdown(final_reasoning)
                                    else:
                                        st.caption("（无最终总结内容）")

                            # 展示完整推理链记录（增强版：显示跨轮记忆）
                            reasoning_chains = full_state.get("reasoning_chains", [])
                            if reasoning_chains:
                                st.markdown("")  # 空行分隔
                                with st.expander(
                                    f"📜 完整推理链记录（共 {len(reasoning_chains)} 跳，跨轮累积）",
                                    expanded=False
                                ):
                                    st.caption(
                                        "💡 推理链在多轮检索中累积，每轮从上一轮的最后结论继续推理，"
                                        "避免重复推理已知信息。"
                                    )

                                    # 判断当前轮次和本轮新增的跳数
                                    current_iteration = full_state.get("revision_step", 0) + 1
                                    hops_this_round = ircot.get("hops_executed", 0)

                                    for i, chain in enumerate(reasoning_chains, 1):
                                        # 判断是否为历史轮次的推理链
                                        is_from_previous = (
                                            current_iteration > 1 and
                                            i <= len(reasoning_chains) - hops_this_round
                                        )

                                        if is_from_previous:
                                            st.markdown(f"**第 {i} 跳推理（继承自历史轮次）** 🔗")
                                        else:
                                            st.markdown(f"**第 {i} 跳推理（本轮新增）** ✨")

                                        st.markdown(chain)
                                        if i < len(reasoning_chains):
                                            st.markdown("---")

                # 采集 Researcher 阶段快照（与直播展示字段完全一致）
                iteration_snapshots.append(snapshot_for_event(event))

            # ── Writer ────────────────────────────────────────────────────────
            elif node == "writer":
                current_iteration = full_state.get("revision_step", 0) + 1
                status_placeholder.info(
                    f"✍️ 第 {current_iteration} 轮 · Writer 起草完成，Reviewer 正在评审……"
                )

                with st.expander(
                    f"✍️ 起草阶段完成（第 {current_iteration} 轮）", expanded=True
                ):
                    draft: str = full_state.get("draft", "")
                    thinks, clean_draft = extract_think(draft)

                    if thinks:
                        with st.expander(
                            "💭 DeepSeek 思维链（Chain of Thought）", expanded=False
                        ):
                            for t in thinks:
                                st.text(t[:2000])

                    if clean_draft:
                        st.markdown("**📝 草稿预览（前 600 字）：**")
                        preview = clean_draft[:600]
                        if len(clean_draft) > 600:
                            preview += "\n\n*… [ 完整报告见下方 ] …*"
                        st.markdown(preview)
                    else:
                        st.warning("草稿内容为空，可能发生了异常。")

                # 采集 Writer 阶段快照（保存完整 draft，渲染时再 extract_think）
                iteration_snapshots.append(snapshot_for_event(event))

            # ── Reviewer ──────────────────────────────────────────────────────
            elif node == "reviewer":
                # reviewer.py 已将 revision_step +1，此值即为当前迭代轮次
                current_iteration = full_state.get("revision_step", 0)
                is_ok: bool = full_state.get("is_satisfactory", False)
                status_placeholder.info(
                    f"🧐 第 {current_iteration} 轮 · Reviewer 评审完成 — "
                    f"{'通过 ✅' if is_ok else ('有限结论 ⚠️' if full_state.get('answer_status') == 'limited' else '未通过，继续迭代 🔄')}"
                )

                with st.expander(
                    f"🧐 审查阶段完成（第 {current_iteration} 轮）", expanded=True
                ):
                    if is_ok:
                        st.success("✅ 评审通过！报告质量达标，即将输出最终报告。")
                    elif full_state.get("answer_status") == "limited":
                        st.warning("Reviewer 接受有限结论；报告仍有未解决问题。")
                    else:
                        st.warning("本轮评审未通过，修订意见已记录。")

                    # 降级评审提示：LLM 结构化评审不可用时不得静默放行
                    _degraded_review: dict = full_state.get("review_result", {}) or {}
                    if _degraded_review.get("degraded"):
                        st.warning(
                            "⚠️ 本轮为降级评审"
                            f"（{_degraded_review.get('review_mode', 'unknown')}）："
                            "LLM 结构化评审不可用，通过判定基于保守标准。"
                        )

                    # 四维评分 + 加权总分 vs 阈值
                    # review_result 结构：{"scores": {"S1": N, ...}, "weighted_score": N, ...}
                    review_result: dict = full_state.get("review_result", {})
                    scores_dict: dict = review_result.get("scores", {})
                    weighted: float = float(review_result.get("weighted_score", 0.0))

                    if scores_dict:
                        st.markdown("**📊 四维评分：**")
                        score_cols = st.columns(5)  # 4 维 + 1 总分列
                        for idx, (key, label, weight) in enumerate([
                            ("S1", "事实准确性", "35%"),
                            ("S2", "逻辑完整性", "25%"),
                            ("S3", "信息覆盖", "25%"),
                            ("S4", "结论可执行", "15%"),
                        ]):
                            val = scores_dict.get(key, "—")
                            score_cols[idx].metric(
                                f"{label}（{weight}）",
                                f"{val}/10" if val != "—" else "—",
                            )
                        delta_val = round(weighted - pass_threshold, 2)
                        score_cols[4].metric(
                            f"加权总分（阈值 {pass_threshold}）",
                            f"{weighted:.2f}",
                            delta=f"{delta_val:+.2f}",
                            delta_color="normal",
                        )

                    feedback: str = full_state.get("critique_feedback", "")
                    if feedback:
                        st.markdown("**💬 评审意见：**")
                        st.warning(feedback)

                    directives: dict = full_state.get("revision_directives", {})
                    if directives:
                        with st.expander("📋 结构化修订指令", expanded=False):
                            st.json(directives)

                    route: str = full_state.get("next_route", "")
                    route_labels = {
                        "end": "📄 输出最终报告",
                        "researcher": "🔍 → Researcher（补充检索）",
                        "writer": "✍️ → Writer（修订草稿）",
                    }
                    st.info(f"**路由决策：** {route_labels.get(route, route or '未知')}")

                # 采集 Reviewer 阶段快照
                iteration_snapshots.append(snapshot_for_event(event))

    if final_state.get("is_satisfactory"):
        status_placeholder.success("✅ 研究完成，报告已通过评审。")
    elif final_state.get("answer_status") == "limited":
        status_placeholder.warning("研究已结束：Reviewer 接受有限结论，但问题尚未完整解决。")
    else:
        status_placeholder.warning("研究已结束，但未达到评审通过条件。")

except Exception as exc:
    status_placeholder.error(f"❌ 流程异常中断：{exc}")
    st.error(f"运行出错：{exc}")
    st.info(f"任务 {run_id} 已保留已提交的进度，可在侧边栏选择后继续研究。")
    with st.expander("📋 错误详情", expanded=True):
        st.code(traceback.format_exc(), language="python")
    st.stop()


# ── 计时器：更新运行总时长 ──────────────────────────────────────────────────────
elapsed_seconds = (datetime.now() - run_start_time).total_seconds()
render_live_timer(timer_placeholder, run_start_time, stop_seconds=elapsed_seconds)
attempts = runner.repository.attempts(run_id)
known_seconds = sum(a["elapsed_seconds"] or 0 for a in attempts)
st.caption(f"已记录执行时间：{known_seconds:.1f} 秒；执行尝试：{len(attempts)} 次。强制退出的未记录时长不计入。")


cache_stats = final_state.get("cache_stats", {})
if cache_stats:
    st.caption(f"缓存：搜索命中 {cache_stats.get('search.hits', 0)} 次，"
               f"实际搜索调用 {cache_stats.get('search.external_calls', 0)} 次；"
               f"向量命中 {cache_stats.get('embedding.hits', 0)} 条。")
    with st.expander("缓存命中与降级统计"):
        st.caption("统计来自已完成节点；中断节点未提交的调用、节点外记忆发布不计入。")
        st.json(cache_stats)

memory_first = final_state.get("memory_first", {})
if memory_first and memory_first.get("mode") != "off":
    st.caption(f"记忆优先：{memory_first.get('covered', 0)} 个子问题具备可复用证据，"
               f"{memory_first.get('searches_skipped', 0)} 个子问题省去预计划搜索。")
    with st.expander("记忆优先检索与补搜原因"):
        st.caption("省去的是子问题的预计划搜索；IRCoT 仍可发现新缺口并补搜。")
        st.json(memory_first)

memory_stats = final_state.get("memory_stats", {})
memory_publication = final_state.get("memory_publication", {})
render_publication_attempts(final_state)
if memory_stats.get("enabled") or memory_publication:
    st.caption(f"跨任务记忆：召回 {memory_stats.get('recalled', 0)} 条，入选 {memory_stats.get('selected', 0)} 条，"
               f"报告引用 {len(final_state.get('memory_used_ids', []))} 条。")
    if memory_publication.get("status") == "failed":
        st.warning("研究已完成，但记忆发布失败。再次打开完成结果可重试发布，不会重新执行研究。")
    elif memory_publication.get("status") == "skipped":
        reasons = {"report_not_accepted": "报告尚未通过证据与质量审核", "no_reviewed_claims": "没有逐项核验的主张",
                   "only_time_sensitive_claims": "仅含时效性事实", "no_eligible_original_evidence": "没有满足原文、引用与来源要求的证据",
                   "legacy_empty_publication": "旧记录发布为空，没有实际入库"}
        st.caption("本次未入库：" + reasons.get(memory_publication.get("skip_reason"), "没有可发布证据"))
    elif memory_publication.get("status") == "completed":
        st.caption(f"记忆发布完成：本任务关联 {len(memory_publication.get('item_ids', []))} 条证据（可能含去重复用）。")
    if memory_stats.get("status") == "failed":
        st.warning("记忆召回暂不可用，本次已继续在线检索。")
    elif memory_stats.get("status") == "fresh_search_required":
        st.caption("该问题包含时效要求，已跳过历史记忆，使用在线检索。")
    with st.expander("记忆来源与发布详情"):
        st.json({"recall": memory_stats, "used_ids": final_state.get("memory_used_ids", []), "publication": memory_publication})
        st.json([{k: c.get(k) for k in ("citation_id", "memory_id", "memory_version", "url", "memory_observed_at", "memory_valid_until")}
                 for c in final_state.get("retrieved_context", []) if isinstance(c, dict) and c.get("memory_id")])


# ── 保存历史记录（失败不中断主流程）──────────────────────────────────────────────
try:
    save_run_to_history(topic, max_revisions, pass_threshold, final_state, known_seconds, iteration_snapshots,
                        completed_at=runner.repository.get(run_id)["completed_at"])
except Exception as exc:
    st.warning(f"历史 JSON 导出失败，checkpoint 中的研究结果仍可读取：{exc}")


# ── 运行期警告 ────────────────────────────────────────────────────────────────
run_errors: list = final_state.get("errors", [])
if run_errors:
    with st.expander(f"⚠️ 运行期警告（共 {len(run_errors)} 条）", expanded=False):
        for err in run_errors:
            st.warning(err)


# ── 最终研究报告 ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📄 最终研究报告")

final_draft: str = final_state.get("final_report") or final_state.get("draft", "")

if final_draft:
    _, clean_report = extract_think(final_draft)
    clean_report = _inject_citation_hyperlinks(
        clean_report,
        final_state.get("retrieved_context", []),
        final_state.get("reasoning_contexts", []),
    )
    st.markdown(clean_report)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\u4e00-\u9fff]", "_", topic)[:30]
    st.download_button(
        label="⬇️ 下载 Markdown 报告",
        data=clean_report,
        file_name=f"ApexLogic_{safe_topic}_{ts_str}.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    st.warning("⚠️ 未生成最终报告，请检查运行期警告或错误日志。")


# ── 参考资料 Top 10 ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📚 参考资料（BGE Reranker Top 10）")
render_references(final_state.get("retrieved_context", [])[:10])


# ── IRCoT 推理链参考文献 ──────────────────────────────────────────────────────
reasoning_enabled = final_state.get("reasoning_enabled", False)
reasoning_contexts = final_state.get("reasoning_contexts", [])

if reasoning_enabled and reasoning_contexts:
    st.markdown("---")
    st.markdown("## 🧠 IRCoT 推理链参考文献")

    # 简略统计
    reasoning_chains = final_state.get("reasoning_chains", [])
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("推理跳数", len(reasoning_chains))
    stat_col2.metric("推理链文档数", len(reasoning_contexts))

    # 统计最终报告中的推理链引用
    final_draft = final_state.get("final_report") or final_state.get("draft", "")
    r_citations = len(re.findall(r'\[R\d+\]', final_draft))
    chain_citations = len(re.findall(r'\[推理链\d+\]', final_draft))
    stat_col3.metric("报告中引用次数", r_citations + chain_citations)

    st.caption(
        "以下文档由IRCoT多跳推理主动发现，独立保存在推理链通道中，"
        "不经过BGE筛选，确保推理链补搜的关键信息不被丢弃。"
    )

    # 展示推理链文档列表（标记为 [R1][R2]...）
    for idx, ctx in enumerate(reasoning_contexts[:15], 1):  # 最多展示15条
        if isinstance(ctx, dict):
            title = ctx.get("title", "未知标题")
            raw_url = str(ctx.get("url", "") or "").strip()
            source_fallback = str(ctx.get("source", "") or "").strip()
            url = raw_url if raw_url else (
                source_fallback if source_fallback.startswith(("http://", "https://")) else ""
            )
            summary = ctx.get("core_summary", "") or ctx.get("content", "")

            if url:
                st.markdown(f"**[R{idx}]** [{title}]({url})")
            else:
                st.markdown(f"**[R{idx}]** {title}")

            if summary:
                st.caption(summary[:200])
        elif isinstance(ctx, str):
            st.markdown(f"**[R{idx}]** {ctx}")
