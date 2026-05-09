"""
绘制毕业论文章节结构图。

输入：article.txt 中描述的 7 章结构。
输出：structure.png（论文章节流程图）。

布局参考：sample.png 风格——竖向流水线，章节标题为深蓝填充圆角矩形，
二级标题为浅色填充小圆角矩形，章节间用向下箭头连接。
第 4 章因二级标题较多，分两行排列。
第 7 章无二级标题，仅显示章节标题。
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CHAPTERS = [
    {
        "title": "第一章  绪论",
        "subs": [
            "1.1 研究背景",
            "1.2 研究意义",
            "1.3 研究问题",
            "1.4 本文主要工作",
            "1.5 论文章节安排",
        ],
        "rows": 1,
    },
    {
        "title": "第二章  相关研究工作",
        "subs": [
            "2.1 检索增强生成与结构化检索技术",
            "2.2 复杂查询规划与自适应资源分配",
            "2.3 迭代主动检索与多智能体反思",
            "2.4 有依据的文本生成与审查约束",
        ],
        "rows": 1,
    },
    {
        "title": "第三章  ApexLogic 系统架构与模型设计",
        "subs": [
            "3.1 系统设计理念与总体层次概述",
            "3.2 状态机流程与条件路由",
            "3.3 系统运行策略",
        ],
        "rows": 1,
    },
    {
        "title": "第四章  基于多智能体协作的深度研究方法设计",
        "subs": [
            "4.1 研究者节点：查询构造与多源搜索",
            "4.2 MAB 自适应预算分配",
            "4.3 图扩展：概念共现查询增强",
            "4.4 AQD：自适应查询分解",
            "4.5 IRCoT：交错推理与检索",
            "4.6 BGE 两阶段过滤",
            "4.7 写作者节点：结构化起草与修订",
            "4.8 审阅者节点：四维评分与路由",
        ],
        "rows": 2,
    },
    {
        "title": "第五章  实验结果与分析",
        "subs": [
            "5.1 实验环境与数据集",
            "5.2 测试评估框架",
            "5.3 各模型端到端测试结果与分析",
        ],
        "rows": 1,
    },
    {
        "title": "第六章  基于 Streamlit 的可视化交互界面设计与实现",
        "subs": [
            "6.1 可视化部署动机与技术选型",
            "6.2 零侵入流式状态追踪机制与数据持久化设计",
            "6.3 任务初始化配置",
            "6.4 智能体执行轨迹渲染",
            "6.5 依据可溯源设计",
        ],
        "rows": 1,
    },
    {
        "title": "第七章  总结与展望",
        "subs": [],
        "rows": 0,
    },
]

CONTAINER_WIDTH = 15.0
TITLE_HEIGHT = 0.75
SUB_HEIGHT = 0.62
ROW_GAP = 0.18
CHAPTER_PADDING = 0.22
CHAPTER_GAP = 0.55
SUB_INNER_GAP = 0.14

TITLE_BG = "#2E5C8A"
TITLE_FG = "white"
SUB_BG = "#F4F6F8"
SUB_EDGE = "#9FB4C7"
CHAPTER_EDGE = "#7A8FA4"

TITLE_FONTSIZE = 13
SUB_FONTSIZE = 9.5


def chapter_height(ch):
    if ch["rows"] == 0:
        return TITLE_HEIGHT
    return (
        TITLE_HEIGHT
        + ch["rows"] * SUB_HEIGHT
        + (ch["rows"] - 1) * ROW_GAP
        + 3 * CHAPTER_PADDING
    )


def split_rows(subs, rows):
    if rows <= 1:
        return [subs]
    half = (len(subs) + rows - 1) // rows
    return [subs[i * half : (i + 1) * half] for i in range(rows)]


def compute_widths(row_subs, available_width):
    n = len(row_subs)
    if n == 0:
        return []
    char_lens = [max(len(s), 4) for s in row_subs]
    total = sum(char_lens)
    widths = [available_width * cl / total for cl in char_lens]
    min_w = max(1.0, available_width / (n * 1.6))
    for j, w in enumerate(widths):
        if w < min_w:
            widths[j] = min_w
    excess = sum(widths) - available_width
    if excess > 0:
        scale = available_width / sum(widths)
        widths = [w * scale for w in widths]
    return widths


def draw():
    heights = [chapter_height(ch) for ch in CHAPTERS]
    total_height = sum(heights) + (len(CHAPTERS) - 1) * CHAPTER_GAP

    margin_x = 0.5
    margin_y = 0.4
    fig_w = CONTAINER_WIDTH + 2 * margin_x
    fig_h = total_height + 2 * margin_y

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    container_x_left = margin_x
    y_cursor = fig_h - margin_y

    for i, (ch, h) in enumerate(zip(CHAPTERS, heights)):
        y_top = y_cursor
        y_bottom = y_cursor - h

        if ch["rows"] == 0:
            box = FancyBboxPatch(
                (container_x_left, y_bottom),
                CONTAINER_WIDTH,
                TITLE_HEIGHT,
                boxstyle="round,pad=0,rounding_size=0.12",
                linewidth=1.0,
                edgecolor=TITLE_BG,
                facecolor=TITLE_BG,
            )
            ax.add_patch(box)
            ax.text(
                container_x_left + CONTAINER_WIDTH / 2,
                y_bottom + TITLE_HEIGHT / 2,
                ch["title"],
                ha="center",
                va="center",
                color=TITLE_FG,
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
            )
        else:
            container = FancyBboxPatch(
                (container_x_left, y_bottom),
                CONTAINER_WIDTH,
                h,
                boxstyle="round,pad=0,rounding_size=0.12",
                linewidth=1.0,
                edgecolor=CHAPTER_EDGE,
                facecolor="white",
            )
            ax.add_patch(container)

            title_y_top = y_top - CHAPTER_PADDING
            title_y_bottom = title_y_top - TITLE_HEIGHT
            title_box = FancyBboxPatch(
                (
                    container_x_left + CHAPTER_PADDING,
                    title_y_bottom,
                ),
                CONTAINER_WIDTH - 2 * CHAPTER_PADDING,
                TITLE_HEIGHT,
                boxstyle="round,pad=0,rounding_size=0.10",
                linewidth=0.5,
                edgecolor=TITLE_BG,
                facecolor=TITLE_BG,
            )
            ax.add_patch(title_box)
            ax.text(
                container_x_left + CONTAINER_WIDTH / 2,
                (title_y_top + title_y_bottom) / 2,
                ch["title"],
                ha="center",
                va="center",
                color=TITLE_FG,
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
            )

            rows_data = split_rows(ch["subs"], ch["rows"])
            inner_width = CONTAINER_WIDTH - 2 * CHAPTER_PADDING

            for row_idx, row_subs in enumerate(rows_data):
                row_y_top = (
                    title_y_bottom
                    - CHAPTER_PADDING
                    - row_idx * (SUB_HEIGHT + ROW_GAP)
                )
                row_y_bottom = row_y_top - SUB_HEIGHT

                n = len(row_subs)
                gap_total = SUB_INNER_GAP * (n - 1)
                available = inner_width - gap_total
                widths = compute_widths(row_subs, available)
                consumed = sum(widths) + gap_total
                row_x_start = (
                    container_x_left + (CONTAINER_WIDTH - consumed) / 2
                )

                x_cursor = row_x_start
                for sub_text, w in zip(row_subs, widths):
                    sub_box = FancyBboxPatch(
                        (x_cursor, row_y_bottom),
                        w,
                        SUB_HEIGHT,
                        boxstyle="round,pad=0,rounding_size=0.07",
                        linewidth=0.7,
                        edgecolor=SUB_EDGE,
                        facecolor=SUB_BG,
                    )
                    ax.add_patch(sub_box)
                    ax.text(
                        x_cursor + w / 2,
                        row_y_bottom + SUB_HEIGHT / 2,
                        sub_text,
                        ha="center",
                        va="center",
                        color="#1F2A36",
                        fontsize=SUB_FONTSIZE,
                    )
                    x_cursor += w + SUB_INNER_GAP

        if i < len(CHAPTERS) - 1:
            arrow_x = container_x_left + CONTAINER_WIDTH / 2
            ax.annotate(
                "",
                xy=(arrow_x, y_bottom - CHAPTER_GAP + 0.05),
                xytext=(arrow_x, y_bottom - 0.02),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#3B4A5A",
                    lw=1.4,
                    mutation_scale=15,
                ),
            )

        y_cursor = y_bottom - CHAPTER_GAP

    out_path = "structure.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    draw()
