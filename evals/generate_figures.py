"""
Figure generation script for ApexLogic paper.
Generates all data-driven figures using matplotlib.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

# Set style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Accuracy vs. Latency Scatter Plot (Pareto Frontier)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_accuracy_latency_scatter():
    """Accuracy vs. Latency scatter plot for all systems."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    datasets = ['Easy-50', 'Medium-150', 'Hard-100']

    # Data: (name, accuracy, latency, marker, color)
    data_easy = [
        ('doubao-seed1.8',        0.74, 48.61,  'o', '#4C72B0'),
        ('doubao-seed2.0mini',    0.70, 42.87,  'o', '#55A868'),
        ('doubao-seed2.0pro',     0.80, 41.04,  'o', '#C44E52'),
        ('qwen3-max',             0.68, 36.03,  'o', '#8172B2'),
        ('qwen3.6-plus',          0.74, 51.18,  'o', '#CCB974'),
        ('deepseek-V3.2(non-thinking)',   0.74, 22.48,  'o', '#64B5CD'),
        ('deepseek-V3.2(thinking)',     0.68, 27.78, 'o', '#FF6F00'),
        ('ApexLogic (Base)',      0.78, 201.42, 's', '#E84855'),
        ('ApexLogic (+AQD)',      0.76, 320.91, '^', '#E84855'),
        ('ApexLogic (+IRCoT)',    0.72, 313.43, 'D', '#E84855'),
        ('ApexLogic (Full)',      0.76, 337.17, 'P', '#E84855'),
    ]
    data_medium = [
        ('doubao-seed1.8',        0.613, 57.13,  'o', '#4C72B0'),
        ('doubao-seed2.0mini',    0.613, 49.47,  'o', '#55A868'),
        ('doubao-seed2.0pro',     0.727, 46.13,  'o', '#C44E52'),
        ('qwen3-max',             0.667, 27.84,  'o', '#8172B2'),
        ('qwen3.6-plus',          0.653, 56.24,  'o', '#CCB974'),
        ('deepseek-V3.2(non-thinking)',         0.593, 24.36,  'o', '#64B5CD'),
        ('deepseek-V3.2(thinking)', 0.61, 30.99, 'o', '#FF6F00'),
        ('ApexLogic (Base)',      0.720, 224.02, 's', '#E84855'),
        ('ApexLogic (+AQD)',      0.667, 295.13, '^', '#E84855'),
        ('ApexLogic (+IRCoT)',    0.660, 338.10, 'D', '#E84855'),
        ('ApexLogic (Full)',      0.707, 395.98, 'P', '#E84855'),
    ]
    data_hard = [
        ('doubao-seed1.8',        0.59, 55.41,  'o', '#4C72B0'),
        ('doubao-seed2.0mini',    0.58, 50.16,  'o', '#55A868'),
        ('doubao-seed2.0pro',     0.65, 44.92,  'o', '#C44E52'),
        ('qwen3-max',             0.60, 23.52,  'o', '#8172B2'),
        ('qwen3.6-plus',          0.64, 55.19,  'o', '#CCB974'),
        ('deepseek-V3.2(non-thinking)',         0.52, 22.76,  'o', '#64B5CD'),
        ('deepseek-V3.2(thinking)', 0.53, 28.21, 'o', '#FF6F00'),
        ('ApexLogic (Base)',      0.65, 225.56, 's', '#E84855'),
        ('ApexLogic (+AQD)',      0.64, 306.59, '^', '#E84855'),
        ('ApexLogic (+IRCoT)',    0.70, 326.33, 'D', '#E84855'),
        ('ApexLogic (Full)',      0.69, 396.32, 'P', '#E84855'),
    ]
    all_data = [data_easy, data_medium, data_hard]

    for ax, (title, data) in zip(axes, zip(datasets, all_data)):
        for name, acc, lat, marker, color in data:
            size = 120 if 'ApexLogic' in name else 70
            alpha = 0.9 if 'ApexLogic' in name else 0.7
            ax.scatter(lat, acc, marker=marker, c=color, s=size, alpha=alpha,
                      edgecolors='white', linewidths=0.5, zorder=3)

        # Label ApexLogic points
        for name, acc, lat, marker, color in data:
            if 'ApexLogic' in name:
                label = name.replace('ApexLogic ', '')
                ax.annotate(label, (lat, acc), textcoords='offset points',
                           xytext=(5, 3), fontsize=7.5, color='#B22222',
                           fontweight='bold')
            elif 'deepseek' in name:
                if 'non-thinking' in name:
                    short_label, offset = 'non-think', (5, -9)
                else:
                    short_label, offset = 'think', (5, 4)
                ax.annotate(short_label, (lat, acc), textcoords='offset points',
                           xytext=offset, fontsize=7, color=color,
                           fontweight='bold')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Avg. Latency per Question (s)')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0, 450)
        acc_vals = [d[1] for d in data]
        ax.set_ylim(min(acc_vals) - 0.05, max(acc_vals) + 0.05)

    # Legend: 7 commercial models + 1 ApexLogic patch, arranged 4 cols x 2 rows
    legend_handles = [
        mpatches.Patch(color='#4C72B0', label='doubao-seed1.8'),
        mpatches.Patch(color='#55A868', label='doubao-seed2.0mini'),
        mpatches.Patch(color='#C44E52', label='doubao-seed2.0pro'),
        mpatches.Patch(color='#8172B2', label='qwen3-max'),
        mpatches.Patch(color='#CCB974', label='qwen3.6-plus'),
        mpatches.Patch(color='#64B5CD', label='deepseek-V3.2 (non-thinking)'),
        mpatches.Patch(color='#FF6F00', label='deepseek-V3.2 (thinking)'),
        mpatches.Patch(color='#E84855', label='ApexLogic variants'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13),
               frameon=True, fancybox=True, fontsize=7.5,
               handlelength=1.4, handleheight=1.0,
               columnspacing=1.2, handletextpad=0.5)

    plt.suptitle('Accuracy vs. Latency Trade-off Across Difficulty Levels',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_accuracy_latency.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 & 2: Accuracy + Latency comparison tables (as figures for quick check)
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bar():
    """Bar chart for ablation study results."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    difficulties = ['Easy-50', 'Medium-150', 'Hard-100']
    systems = ['Base', '+AQD', '+IRCoT', 'Full\n(+AQD+IRCoT)']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    acc_data = {
        'Easy-50':    [0.78, 0.76, 0.72, 0.76],
        'Medium-150': [0.720, 0.667, 0.660, 0.707],
        'Hard-100':   [0.65, 0.64, 0.70, 0.69],
    }
    lat_data = {
        'Easy-50':    [201.42, 320.91, 313.43, 337.17],
        'Medium-150': [224.02, 295.13, 338.10, 395.98],
        'Hard-100':   [225.56, 306.59, 326.33, 396.32],
    }

    x = np.arange(len(systems))
    width = 0.6

    for ax, diff in zip(axes, difficulties):
        accs = acc_data[diff]
        lats = lat_data[diff]
        bars = ax.bar(x, accs, width, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

        # Add accuracy labels on bars
        for bar, acc, lat in zip(bars, accs, lats):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            # Latency label below bar
            ax.text(bar.get_x() + bar.get_width()/2., 0.002,
                   f'{lat:.0f}s', ha='center', va='bottom', fontsize=7,
                   color='#555', style='italic')

        ax.set_title(diff, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(systems, fontsize=8)
        ax.set_ylabel('Accuracy')
        min_acc = min(accs)
        ax.set_ylim(min_acc - 0.08, max(accs) + 0.04)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # Highlight best bar
        best_idx = accs.index(max(accs))
        bars[best_idx].set_edgecolor('#333')
        bars[best_idx].set_linewidth(2)

    plt.suptitle('Ablation Study: Accuracy by Module Configuration\n',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_ablation_bar.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig_e2e_comparison():
    """End-to-end accuracy comparison grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    systems = [
        'doubao\nseed1.8', 'doubao\nseed2.0mini', 'doubao\nseed2.0pro',
        'qwen3\nmax', 'qwen3.6\nplus', 'deepseek\nV3.2\n(non-thinking)','deepseek\nV3.2\n(thinking)', 'ApexLogic\n(Base)'
    ]
    easy   = [0.74, 0.70, 0.80, 0.68, 0.74, 0.74,0.68, 0.78]
    medium = [0.613, 0.613, 0.727, 0.667, 0.653, 0.593,0.610, 0.720]
    hard   = [0.59, 0.58, 0.65, 0.60, 0.64, 0.52, 0.53, 0.65]

    x = np.arange(len(systems))
    width = 0.25
    offset = [-width, 0, width]
    colors_diff = ['#4FC3F7', '#039BE5', '#01579B']
    labels_diff = ['Easy-50', 'Medium-150', 'Hard-100']

    for data, label, color, off in zip([easy, medium, hard], labels_diff, colors_diff, offset):
        bars = ax.bar(x + off, data, width, label=label, color=color, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.004,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    # Highlight ApexLogic
    for off, color in zip(offset, colors_diff):
        ax.bar(x[-1] + off, 0, width, color='none',
               edgecolor='#E84855', linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.45, 0.88)
    ax.legend(loc='lower right', frameon=True)
    ax.set_title('End-to-End Accuracy Comparison: ApexLogic vs. Commercial LLMs',
                fontsize=12, fontweight='bold')

    # Red box around ApexLogic
    ax.axvspan(x[-1] - 0.42, x[-1] + 0.42, alpha=0.08, color='#E84855', zorder=0)
    ax.text(x[-1], 0.455, 'ApexLogic', ha='center', fontsize=9,
           color='#E84855', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_e2e_comparison.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig_funnel():
    """BGE two-stage filtering funnel chart."""
    fig, ax = plt.subplots(figsize=(7, 5))

    stages = [
        'Multi-source\nBroad Search\n(DDG + ArXiv\n+ Tavily)',
        'After\nDeduplication',
        'BGE Retriever\n(Top-20)',
        'BGE Reranker\n(Top-10)',
    ]
    counts = [60, 45, 20, 10]
    colors_funnel = ['#7986CB', '#4DB6AC', '#FFB74D', '#E57373']

    y_positions = [3, 2, 1, 0]
    max_count = max(counts)

    for i, (stage, count, color, y) in enumerate(zip(stages, counts, colors_funnel, y_positions)):
        width_ratio = count / max_count
        bar_width = width_ratio * 0.7
        ax.barh(y, bar_width, height=0.6, left=(0.7 - bar_width)/2,
               color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        # Count label
        ax.text(0.7 + 0.02, y, f'~{count} docs', va='center',
               fontsize=10, fontweight='bold', color=color)
        # Stage label
        ax.text(-0.02, y, stage, va='center', ha='right', fontsize=9)

    # Arrows
    for y in [2.5, 1.5, 0.5]:
        ax.annotate('', xy=(0.35, y - 0.2), xytext=(0.35, y + 0.1),
                   arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    ax.set_xlim(-0.35, 1.0)
    ax.set_ylim(-0.5, 3.7)
    ax.axis('off')
    ax.set_title('BGE Two-Stage Retrieval Filtering Funnel\n(Typical counts per research iteration)',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_bge_funnel.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def fig_module_gain_heatmap():
    """Heatmap showing module gain by difficulty."""
    fig, ax = plt.subplots(figsize=(7, 4))

    modules = ['Base', '+AQD', '+IRCoT', '+AQD+IRCoT']
    difficulties = ['Easy-50', 'Medium-150', 'Hard-100']

    # Accuracy relative to Base (delta)
    base_acc = {'Easy-50': 0.78, 'Medium-150': 0.720, 'Hard-100': 0.65}
    acc_matrix = np.array([
        [0.78, 0.76, 0.72, 0.76],
        [0.720, 0.667, 0.660, 0.707],
        [0.65, 0.64, 0.70, 0.69],
    ])
    delta_matrix = acc_matrix - np.array([[0.78, 0.78, 0.78, 0.78],
                                           [0.720, 0.720, 0.720, 0.720],
                                           [0.65, 0.65, 0.65, 0.65]])

    im = ax.imshow(delta_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, fontsize=10)
    ax.set_yticks(range(len(difficulties)))
    ax.set_yticklabels(difficulties, fontsize=10)

    for i in range(len(difficulties)):
        for j in range(len(modules)):
            delta = delta_matrix[i, j]
            text = f'{delta:+.3f}\n({acc_matrix[i,j]:.3f})'
            color = 'white' if abs(delta) > 0.04 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Δ Accuracy vs. Base')
    ax.set_title('Module Contribution Heatmap\n(Δ accuracy relative to Base; absolute accuracy in parentheses)',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('System Configuration')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_module_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("Generating ApexLogic paper figures...")
    fig5_accuracy_latency_scatter()
    fig_ablation_bar()
    fig_e2e_comparison()
    fig_funnel()
    fig_module_gain_heatmap()
    print("\nAll figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
