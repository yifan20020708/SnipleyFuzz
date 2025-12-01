# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.2,
})

LINESTYLE_MAIN = '-'
LINESTYLE_ABL1 = '--'
LINESTYLE_ABL2 = ':'
MARK_MAIN = 'o'
MARK_ABL1 = 's'
MARK_ABL2 = '^'

t_plug = np.arange(0, 100, 10)

seed_SF_plug   = [2,2,3,4,5,6,7,8,8,9]      # SnipleyFuzz
seed_MS_plug   = [2,2,2,2,2,2,3,3,3,3]      # SnipleyFuzz-M-S
seed_MSN_plug  = [2,2,3,3,4,4,4,5,5,6]      # SnipleyFuzz-M-SN

resp_SF_plug   = [13,14,18,21,23,27,31,35,37,39]
resp_MS_plug   = [13,13,13,13,13,14,16,16,16,16]
resp_MSN_plug  = [13,14,14,15,16,16,19,19,23,26]

exp_SF_plug  = [None, None, 74, 39, 39, 37, 41, 36, 36, 37]
exp_MS_plug  = [None, None, None, None, None, None, 689, 689, 689, 689]
exp_MSN_plug = [None, None, 58, 58, 127, 127, 141, 141, 140, 115]

t_cam = np.arange(0, 100, 10)

seed_SF_cam   = [2,3,3,4,5,6,7,8,9,10]
seed_MS_cam   = [2,2,2,3,3,3,3,3,4,4]
seed_MSN_cam  = [2,2,3,4,4,4,5,6,6,7]

resp_SF_cam   = [10,12,15,17,22,27,32,39,46,52]
resp_MS_cam   = [10,10,10,11,11,11,11,11,12,12]
resp_MSN_cam  = [10,11,12,13,13,15,17,20,28,30]

exp_SF_cam  = [None, 14, 14, 18, 22, 19, 26, 23, 21, 19]
exp_MS_cam  = [None, None, None, 420, 420, 420, 420, 420, 582, 582]
exp_MSN_cam = [None, None, 39, 34, 34, 34, 102, 82, 82, 85]


def plot_time_series(ax, x, y_main, y_abl1, y_abl2, title, ylabel, vline=None,
                     colors=('#1f77b4', '#ff7f0e', '#2ca02c')):
    # 画线
    l1, = ax.plot(x, y_main, marker=MARK_MAIN, linestyle=LINESTYLE_MAIN,
                  color=colors[0], label='SnipleyFuzz')
    l2, = ax.plot(x, y_abl1, marker=MARK_ABL1, linestyle=LINESTYLE_ABL1,
                  color=colors[1], label='SnipleyFuzz-CMAB-based')
    l3, = ax.plot(x, y_abl2, marker=MARK_ABL2, linestyle=LINESTYLE_ABL2,
                  color=colors[2], label='SnipleyFuzz-priority-based')

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (min)')
    ax.grid(True, linestyle='--', alpha=0.35)

    y_lists = [y_main, y_abl1, y_abl2]
    lines   = [l1, l2, l3]

    # 先把 x 轴往右扩一点，留出标注空间
    x_min, x_max = x[0], x[-1]
    extra = (x_max - x_min) * 0.12   # 右侧多留 12% 的宽度
    ax.set_xlim(x_min, x_max + extra)

    x_label = x_max + extra * 0.4    # 数字放在右侧中间位置

    # 为了防止三个数字上下贴太近，加一点垂直偏移
    # 这里按 y 范围自适应一个小偏移
    y_all = [v for ys in y_lists for v in ys if v is not None]
    y_range = max(y_all) - min(y_all) if y_all else 1.0
    dy = y_range * 0.03

    for idx, (ys, line) in enumerate(zip(y_lists, lines)):
        if ys[-1] is None:
            continue
        y_last = ys[-1]
        # 不同曲线上下稍微错开一点
        offset = (idx - 1) * dy
        ax.text(
            x_label, y_last + offset,
            f'{y_last}',
            color=line.get_color(),
            va='center', ha='left',
            fontsize=9,
        )
    
    if vline is not None:
        for vline_min in vline:
            ymin, ymax = ax.get_ylim()
            ax.axvline(vline_min, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
            # 在虚线顶部画一个小箭头并文字标注
            ax.annotate(f"start {vline_min} min", xy=(vline_min, ymax), xytext=(vline_min+1, ymax*0.98),
                        fontsize=7, color='gray', rotation=90, va='top',
                        arrowprops=dict(arrowstyle='-|>', color='gray', lw=0.8, shrinkA=0, shrinkB=0))
            

fig, axes = plt.subplots(2, 3, figsize=(12, 5))

# Row 1: Smart Plug
plot_time_series(axes[0,0], t_plug, seed_SF_plug, seed_MS_plug, seed_MSN_plug,
                 '(a1) Smart Plug — Seed Pool Length', 'Seed Count')
plot_time_series(axes[0,1], t_plug, resp_SF_plug, resp_MS_plug, resp_MSN_plug,
                 '(a2) Smart Plug — Cumulative NRS Length', 'NRS cum Count')
plot_time_series(axes[0,2], t_plug, exp_SF_plug, exp_MS_plug, exp_MSN_plug,
                 '(a3) Smart Plug — Exploration Efficiency', 'Mutation Round / NRS', vline=[20, 60])

# Row 2: Smart Camera
plot_time_series(axes[1,0], t_cam, seed_SF_cam, seed_MS_cam, seed_MSN_cam,
                 '(b1) Smart Camera — Seed Pool Length', 'Seed Count')
plot_time_series(axes[1,1], t_cam, resp_SF_cam, resp_MS_cam, resp_MSN_cam,
                 '(b2) Smart Camera — Cumulative NRS Length', 'NRS cum Count')
plot_time_series(axes[1,2], t_cam, exp_SF_cam, exp_MS_cam, exp_MSN_cam,
                 '(b3) Smart Camera — Exploration Efficiency', 'Mutation Round / NRS', vline=[10, 20, 30])

# unified legend
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("/home/SnipleyFuzz/experiment/chart/ablation_combined.png", dpi=600, bbox_inches='tight')
plt.savefig("/home/SnipleyFuzz/experiment/chart/ablation_combined.pdf", bbox_inches='tight')
"Saved ablation_combined.[png|pdf]"