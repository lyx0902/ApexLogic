"""
ApexLogic 深度研究引擎 — Streamlit 可视化界面
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # 必须在任何读取 os.getenv 的模块导入前执行

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

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

# ── 常量 ───────────────────────────────────────────────────────────────────────
APPSTATS_DIR = Path("appstats")

# ── 通用辅助函数 ───────────────────────────────────────────────────────────────

def extract_think(text: str) -> tuple[list[str], str]:
    """分离 <think>...</think> 块，返回 (思维链列表, 去标签正文)。"""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinks = pattern.findall(text)
    clean = pattern.sub("", text).strip()
    return thinks, clean


def detect_node(full: dict, prev: dict) -> str:
    """通过比对两次全量 State 快照推断刚完成的节点名称。

    优先级：
      1. revision_step 增大 → reviewer（唯一递增该字段的节点）
      2. draft 变化（非空）  → writer（唯一修改该字段的节点）
      3. 其余默认            → researcher
    """
    if full.get("revision_step", 0) > prev.get("revision_step", 0):
        return "reviewer"
    curr_draft = full.get("draft", "")
    if curr_draft and curr_draft != prev.get("draft", ""):
        return "writer"
    return "researcher"


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

def save_run_to_history(
    topic: str,
    max_revisions: int,
    pass_threshold: float,
    final_state: dict,
    elapsed_seconds: float,
    iteration_snapshots: list | None = None,
) -> None:
    """将本次研究结果序列化到 appstats/ 目录中。"""
    APPSTATS_DIR.mkdir(exist_ok=True)
    ts = datetime.now()
    safe_topic = re.sub(r"[^\w\u4e00-\u9fff]", "_", topic)[:20]
    filename = APPSTATS_DIR / f"run_{ts.strftime('%Y%m%d_%H%M%S')}_{safe_topic}.json"

    review_result = final_state.get("review_result", {})
    contexts = final_state.get("retrieved_context", [])[:10]

    record = {
        "timestamp": ts.isoformat(),
        "topic": topic,
        "max_revisions": max_revisions,
        "pass_threshold": pass_threshold,
        "iterations_done": final_state.get("revision_step", 0),
        "is_satisfactory": bool(final_state.get("is_satisfactory", False)),
        "weighted_score": float(review_result.get("weighted_score", 0.0)),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "final_report": final_state.get("final_report") or final_state.get("draft", ""),
        "references": [
            {
                "title": ctx.get("title", "") if isinstance(ctx, dict) else str(ctx),
                "url": (
                    str(ctx.get("url", "") or "").strip()
                    if isinstance(ctx, dict)
                    else ""
                ),
                "score": ctx.get("bge_reranker_score") if isinstance(ctx, dict) else None,
                "summary": (ctx.get("core_summary", "")[:200] if isinstance(ctx, dict) else ""),
            }
            for ctx in contexts
        ],
        "errors_count": len(final_state.get("errors", [])),
        "errors": final_state.get("errors", [])[:10],
        "run_metadata": {
            "iteration_snapshots": iteration_snapshots or [],
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_history_list() -> list[dict]:
    """加载 appstats/ 中最近 20 条历史记录（倒序）。"""
    if not APPSTATS_DIR.exists():
        return []
    files = sorted(APPSTATS_DIR.glob("run_*.json"), reverse=True)[:20]
    records: list[dict] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            records.append({"file": str(f), "data": data})
        except Exception:
            pass
    return records


def _history_label(record: dict) -> str:
    """生成历史记录的侧边栏展示标签。"""
    data = record["data"]
    try:
        ts = datetime.fromisoformat(data["timestamp"]).strftime("%m-%d %H:%M")
    except Exception:
        ts = "??-??"
    topic_short = data["topic"][:14] + ("…" if len(data["topic"]) > 14 else "")
    score = data.get("weighted_score", 0.0)
    ok = "✅" if data.get("is_satisfactory") else "❌"
    return f"{ts} · {topic_short} · {score:.2f}分 {ok}"


def show_history_view(data: dict) -> None:
    """在主区域渲染历史记录详情。"""
    ts_str = ""
    try:
        ts_str = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    st.markdown("## 📋 历史记录查看")
    st.caption(f"研究时间：{ts_str}　｜　主题：{data['topic']}")

    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    hc1.metric("研究主题", data["topic"][:18] + ("…" if len(data["topic"]) > 18 else ""))
    hc2.metric("实际迭代轮数", data.get("iterations_done", "—"))
    hc3.metric("加权总分", f"{data.get('weighted_score', 0):.2f}")
    hc4.metric("运行总时长", _format_elapsed(data.get("elapsed_seconds", 0)))
    pass_threshold = data.get("pass_threshold")
    hc5.metric(
        "通过阈值",
        f"{float(pass_threshold):.1f}" if pass_threshold is not None else "—",
    )
    st.markdown("---")

    # 研究报告
    st.markdown("## 📄 研究报告")
    report = data.get("final_report", "")
    if report:
        _, clean = extract_think(report)
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
    st.markdown("---")
    st.markdown("## 📚 参考资料")
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
                            f"🧩 AQD 查询分解详情（{len(h_sub_results)} 个子问题，共补搜 {h_total_new} 条）",
                            expanded=False,
                        ):
                            for sub in h_sub_results:
                                sq_id = sub.get("id", "?")
                                question = sub.get("question", "")
                                new_docs_count = sub.get("new_docs", 0)
                                skipped = sub.get("skipped", False)
                                retrieved_docs = sub.get("retrieved_docs", [])
                                if skipped:
                                    st.markdown(
                                        f"**子问题 {sq_id}：** {question}  \n"
                                        f"*（与已有查询高度重叠，已跳过）*"
                                    )
                                else:
                                    st.markdown(
                                        f"**子问题 {sq_id}：** {question} — 补搜 {new_docs_count} 条"
                                    )
                                    for doc in retrieved_docs:
                                        doc_title = doc.get("title", "") or "未知标题"
                                        doc_url = doc.get("url", "")
                                        if doc_url:
                                            st.markdown(f"&nbsp;&nbsp;- [{doc_title}]({doc_url})")
                                        else:
                                            st.markdown(f"&nbsp;&nbsp;- {doc_title}")

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
                                if i < len(h_hop_summaries) - 1:
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
                    else:
                        st.error("❌ 评审未通过，系统继续优化。")

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

                    st.markdown("---")


# ── Session state 初始化 ───────────────────────────────────────────────────────
if "view_history" not in st.session_state:
    st.session_state.view_history = False
if "history_data" not in st.session_state:
    st.session_state.history_data = None

# 提前加载历史列表（侧边栏和欢迎页均需要）
history_list = load_history_list()


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

    # ── 历史记录区 ──────────────────────────────────────────────────────────────
    if history_list:
        st.markdown("---")
        st.markdown("**📚 历史记录**")
        options = ["— 选择历史记录 —"] + [_history_label(r) for r in history_list]
        selected_label = st.selectbox(
            "历史记录列表", options, label_visibility="collapsed"
        )
        if selected_label != "— 选择历史记录 —":
            real_idx = options.index(selected_label) - 1
            if st.button("📖 查看此次记录", use_container_width=True):
                st.session_state.view_history = True
                st.session_state.history_data = history_list[real_idx]["data"]
                st.rerun()

    if st.session_state.view_history:
        if st.button("← 返回新研究", use_container_width=True):
            st.session_state.view_history = False
            st.session_state.history_data = None
            st.rerun()

    st.markdown("---")
    st.caption(
        "**ApexLogic 多智能体深度研究系统**  \n"
        "Thompson Sampling MAB · Graph Expand  \n"
        "AQD · IRCoT · BGE 两阶段精筛  \n\n"
        "运行命令：`streamlit run app.py`"
    )


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("# 🔬 ApexLogic 多智能体深度研究引擎")
st.caption(
    "LangGraph 多智能体循环：Researcher → Writer → Reviewer "
    "· 四层检索优化 · BGE 两阶段精筛 "
)
st.caption("©️南京理工大学计算机科学与技术22级刘宇翔")
st.markdown("---")


# ── 历史记录浏览模式（优先于新研究，但 start_btn 可覆盖）────────────────────────
if st.session_state.view_history and not start_btn:
    if st.session_state.history_data:
        show_history_view(st.session_state.history_data)
    else:
        st.warning("历史记录数据丢失，请重新选择。")
    st.stop()


# ── 欢迎界面（未启动研究时）───────────────────────────────────────────────────────
if not start_btn:
    st.info(
        "👈 请在左侧侧边栏输入研究主题并点击「🚀 开始深度研究」，"
        "系统将自动启动多智能体深度研究流程并在此实时展示执行过程。"
    )
    if history_list:
        st.caption("也可在左侧侧边栏「历史记录」中查看以往的研究结果。")
    st.stop()

# 主题为空保险检查
if not topic.strip():
    st.warning("⚠️ 研究主题不能为空，请在侧边栏输入主题后重试。")
    st.stop()

# 进入新研究时清除历史浏览状态
st.session_state.view_history = False
st.session_state.history_data = None


# ── 导入核心模块 ───────────────────────────────────────────────────────────────
try:
    from core.graph import compile_graph
    from core.state import create_initial_state
except ImportError as exc:
    st.error(f"❌ 核心模块导入失败，请确认依赖已安装：{exc}")
    st.stop()


# ── 运行参数概览（5 列：主题 / 轮数 / 通过阈值 / 启动时间 / 运行总时长）──────────
display_start_time = datetime.now()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("研究主题", topic[:16] + ("…" if len(topic) > 16 else ""))
c2.metric("最大反思轮数", max_revisions)
c3.metric("通过阈值", f"{pass_threshold:.1f}")
c4.metric("启动时间", display_start_time.strftime("%H:%M:%S"))
timer_placeholder = c5.empty()
# run_start_time 和计时器将在 graph.stream() 开始前设定，精确计量执行时长
st.markdown("---")


# ── 编译图 & 构建初始状态 ───────────────────────────────────────────────────────
# 将用户设置的通过阈值写入环境变量，reviewer_node 运行时会动态读取
os.environ["REVIEWER_PASS_THRESHOLD"] = str(pass_threshold)
try:
    graph = compile_graph(max_revisions)
    # output_mode="user" 保证最终报告干净；过程数据仍在 state 各字段中
    initial_state = create_initial_state(topic, output_mode="user")
except Exception as exc:
    st.error(f"❌ 图编译失败：{exc}")
    with st.expander("错误详情"):
        st.code(traceback.format_exc(), language="python")
    st.stop()


# ── 状态占位符 ─────────────────────────────────────────────────────────────────
status_placeholder = st.empty()
status_placeholder.info("🚀 引擎启动，多智能体流水线正在初始化……")


# ── 主流式循环 ─────────────────────────────────────────────────────────────────
prev_state: dict = {}
final_state: dict = dict(initial_state)
researcher_first_seen = False
iteration_snapshots: list = []

# 计时器：在 graph.stream() 启动前精确计时，与页面加载耗时解耦
run_start_time = datetime.now()
render_live_timer(timer_placeholder, run_start_time)

try:
    with st.spinner("🤖 正在思考与执行中，请耐心等待……"):

        # stream_mode="values"：LangGraph 内部 Reducer 合并 List 字段，
        # 每次 yield 为最新完整 State，无需手动 update()
        for full_state in graph.stream(initial_state, stream_mode="values"):
            node = detect_node(full_state, prev_state)
            final_state = dict(full_state)

            # ── Researcher ────────────────────────────────────────────────────
            if node == "researcher":
                current_iteration = full_state.get("revision_step", 0) + 1
                if not researcher_first_seen:
                    researcher_first_seen = True
                    status_placeholder.info(
                        f"🔍 第 {current_iteration} 轮 · Researcher 正在检索……"
                    )
                    prev_state = dict(full_state)
                    continue

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
                            f"🧩 AQD 查询分解详情（{len(sub_results_list)} 个子问题，共补搜 {total_new_docs} 条）",
                            expanded=False,
                        ):
                            for sub in sub_results_list:
                                sq_id = sub.get("id", "?")
                                question = sub.get("question", "")
                                new_docs_count = sub.get("new_docs", 0)
                                skipped = sub.get("skipped", False)
                                retrieved_docs = sub.get("retrieved_docs", [])
                                if skipped:
                                    st.markdown(
                                        f"**子问题 {sq_id}：** {question}  \n"
                                        f"*（与已有查询高度重叠，已跳过）*"
                                    )
                                else:
                                    st.markdown(
                                        f"**子问题 {sq_id}：** {question} — 补搜 {new_docs_count} 条"
                                    )
                                    for doc in retrieved_docs:
                                        doc_title = doc.get("title", "") or "未知标题"
                                        doc_url = doc.get("url", "")
                                        if doc_url:
                                            st.markdown(f"&nbsp;&nbsp;- [{doc_title}]({doc_url})")
                                        else:
                                            st.markdown(f"&nbsp;&nbsp;- {doc_title}")

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

                                if i < len(hop_summaries_list) - 1:
                                    st.markdown("---")

                # 采集 Researcher 阶段快照（与直播展示字段完全一致）
                iteration_snapshots.append({
                    "node": "researcher",
                    "iteration": current_iteration,
                    "search_queries": full_state.get("search_queries", []),
                    "mab_state": full_state.get("mab_state", {}),
                    "source_quality_summary": full_state.get("source_quality_summary", {}),
                    "query_plan": full_state.get("query_plan", {}),
                    "iterative_retrieval_summary": full_state.get("iterative_retrieval_summary", {}),
                    "retrieved_context_count": len(full_state.get("retrieved_context", [])),
                })

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
                iteration_snapshots.append({
                    "node": "writer",
                    "iteration": current_iteration,
                    "draft": full_state.get("draft", ""),
                })

            # ── Reviewer ──────────────────────────────────────────────────────
            elif node == "reviewer":
                # reviewer.py 已将 revision_step +1，此值即为当前迭代轮次
                current_iteration = full_state.get("revision_step", 0)
                is_ok: bool = full_state.get("is_satisfactory", False)
                status_placeholder.info(
                    f"🧐 第 {current_iteration} 轮 · Reviewer 评审完成 — "
                    f"{'通过 ✅' if is_ok else '未通过，继续迭代 🔄'}"
                )

                with st.expander(
                    f"🧐 审查阶段完成（第 {current_iteration} 轮）", expanded=True
                ):
                    if is_ok:
                        st.success("✅ 评审通过！报告质量达标，即将输出最终报告。")
                    else:
                        st.error("❌ 评审未通过，系统将根据反馈继续优化。")

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
                iteration_snapshots.append({
                    "node": "reviewer",
                    "iteration": current_iteration,
                    "review_result": full_state.get("review_result", {}),
                    "critique_feedback": full_state.get("critique_feedback", ""),
                    "revision_directives": full_state.get("revision_directives", {}),
                    "next_route": full_state.get("next_route", ""),
                    "is_satisfactory": bool(full_state.get("is_satisfactory", False)),
                })

            prev_state = dict(full_state)

    status_placeholder.success("✅ 运行结束！所有智能体节点执行完毕。")

except Exception as exc:
    status_placeholder.error(f"❌ 流程异常中断：{exc}")
    st.error(f"运行出错：{exc}")
    with st.expander("📋 错误详情", expanded=True):
        st.code(traceback.format_exc(), language="python")
    st.stop()


# ── 计时器：更新运行总时长 ──────────────────────────────────────────────────────
elapsed_seconds = (datetime.now() - run_start_time).total_seconds()
render_live_timer(timer_placeholder, run_start_time, stop_seconds=elapsed_seconds)


# ── 保存历史记录（失败不中断主流程）──────────────────────────────────────────────
try:
    save_run_to_history(topic, max_revisions, pass_threshold, final_state, elapsed_seconds, iteration_snapshots)
except Exception:
    pass


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
