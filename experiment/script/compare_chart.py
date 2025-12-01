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

# 时间轴（分钟）
time_yeelight = np.arange(0, 70, 10)       
time_plug = np.arange(0, 100, 10)      
time_camera = np.arange(0, 100, 10)

# Yeelight 数据
seed_s = [2, 9, 15, 19, 24, 28, 31]
resp_s = [12, 66, 205, 316, 415, 520, 606]
exp_s_raw = [None, 32, 22, 16, 13, 11, 10]  # 原始 exp 数据
seed_z = [2, 5, 5, 5, 5, 5, 5]
resp_z = [12, 14, 14, 14, 14, 14, 14]
exp_z_raw = [None, 352, 352, 352, 352, 352, 352]

# Smart Plug 数据
seed_s_plug = [2, 2, 3, 4, 4, 5, 6, 6, 7, 9]
resp_s_plug = [15, 15, 17, 21, 24, 25, 29, 33, 35, 40]
exp_s_plug_raw = [None, None, 32, 27, 27, 19, 31, 31, 30, 26]
seed_z_plug = [2, 2, 3, 3, 4, 5, 6, 6, 7, 7]
resp_z_plug = [15, 15, 17, 19, 20, 21, 24, 26, 27, 27]
exp_z_plug_raw = [None, None, 36, 36, 52, 43, 45, 45, 42, 42]

# Smart Camera 数据
seed_s_cam = [2, 2, 3, 4, 4, 5, 6, 7, 8, 10]
resp_s_cam = [11, 11, 14, 16, 19, 29, 31, 38, 45, 56]
exp_s_cam_raw = [None, None, 81, 44, 44, 55, 42, 36, 35, 34]
seed_z_cam = [2, 2, 3, 3, 4, 5, 5, 6, 7, 8]
resp_z_cam = [11, 11, 13, 14, 15, 17, 18, 20, 22, 24]
exp_z_cam_raw = [None, None, 164, 164, 161, 110, 110, 140, 120, 108]


def plot_time_series(ax, x, y_main, y_abl1, title, ylabel, vline=None,
                     colors=('#1f77b4', '#ff7f0e', '#2ca02c')):
    # 画线
    l1, = ax.plot(x, y_main, marker=MARK_MAIN, linestyle=LINESTYLE_MAIN,
                  color=colors[0], label='SnipleyFuzz')
    l2, = ax.plot(x, y_abl1, marker=MARK_ABL1, linestyle=LINESTYLE_ABL1,
                  color=colors[1], label='Snipuzz')

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (min)')
    ax.grid(True, linestyle='--', alpha=0.35)

    # ------- 改进的数字标注：在右侧边缘按颜色对齐 -------
    y_lists = [y_main, y_abl1]
    lines   = [l1, l2]

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

fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))

plot_time_series(axes[0,0], time_yeelight, seed_s, seed_z,
                 '(a1) YLDP05YL — Seed Pool Length', 'Seed Count')
plot_time_series(axes[0,1], time_yeelight, resp_s, resp_z,
                 '(a2) YLDP05YL — Cumulative NRS Length', 'NRS cum Count')
plot_time_series(axes[0,2], time_yeelight, exp_s_raw, exp_z_raw,
                 '(a3) YLDP05YL — Exploration Efficiency', 'Mutation Round / NRS', vline=[10])

# Row 1: Smart Plug
plot_time_series(axes[1,0], time_plug, seed_s_plug, seed_z_plug,
                 '(b1) Smart Plug — Seed Pool Length', 'Seed Count')
plot_time_series(axes[1,1], time_plug, resp_s_plug, resp_z_plug,
                 '(b2) Smart Plug — Cumulative NRS Length', 'NRS cum Count')
plot_time_series(axes[1,2], time_plug, exp_s_plug_raw, exp_z_plug_raw,
                 '(b3) Smart Plug — Exploration Efficiency', 'Mutation Round / NRS', vline=[20])

# Row 2: Smart Camera
plot_time_series(axes[2,0], time_camera, seed_s_cam, seed_z_cam,
                 '(c1) Smart Camera — Seed Pool Length', 'Seed Count')
plot_time_series(axes[2,1], time_camera, resp_s_cam, resp_z_cam,
                 '(c2) Smart Camera — Cumulative NRS Length', 'NRS cum Count')
plot_time_series(axes[2,2], time_camera, exp_s_cam_raw, exp_z_cam_raw,
                 '(c3) Smart Camera — Exploration Efficiency', 'Mutation Round / NRS', vline=[20])

# unified legend
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("/home/SnipleyFuzz/experiment/chart/firmware_comparison.png", dpi=600, bbox_inches='tight')
plt.savefig("/home/SnipleyFuzz/experiment/chart/firmware_comparison.pdf", bbox_inches='tight')
