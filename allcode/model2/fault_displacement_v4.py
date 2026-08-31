"""
含逆冲分量的走滑断层位移场计算 V5
修改：只保留水平/垂直速度场图，各自单独导出为EPS
"""

import numpy as np
import matplotlib
matplotlib.rcParams['ps.fonttype'] = 3       # 字体嵌入，EPS投稿必须
matplotlib.rcParams['pdf.fonttype'] = 3
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 8

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# 1. 基础配置
# ============================================================
width = 200
height = 1000
origin = np.array([100, 500])
locking_depth = 15.0

x_coords = np.arange(width)
y_coords = np.arange(height)
X, Y = np.meshgrid(x_coords, y_coords)

# ============================================================
# 2. 断层参数
# ============================================================
fault_configs = [
    {
        'name': 'xshf',
        'dir': np.array([-3, 10]),
        'rate': 10.0,
        'color': 'red',
        'rake': 0.0,
        'dip': 90.0,
        'dip_direction': 1
    },
    {
        'name': 'anhf',
        'dir': np.array([0, -1]),
        'rate': 5.0,
        'color': 'blue',
        'rake': 0.0,
        'dip': 90.0,
        'dip_direction': 1
    },
    {
        'name': 'dlsf',
        'dir': np.array([2, -1]),
        'rate': 3.3,
        'color': 'green',
        'rake': 16.0,
        'dip': 50.0,
        'dip_direction': -1
    }
]

U_total = np.zeros_like(X, dtype=float)
V_total = np.zeros_like(Y, dtype=float)
W_total = np.zeros_like(X, dtype=float)
fault_L_max = {}
fault_fields = {}

# ============================================================
# 3. 计算
# ============================================================
for f in fault_configs:
    print(f"\n{'='*50}")
    print(f"  {f['name'].upper()}")
    print(f"{'='*50}")

    n = f['dir'] / np.linalg.norm(f['dir'])
    n_perp = np.array([-n[1], n[0]])
    dip_dir = f.get('dip_direction', 1)

    DX = X - origin[0]
    DY = Y - origin[1]
    L = DX * n[0] + DY * n[1]
    D = DX * n_perp[0] + DY * n_perp[1]
    D_signed = D * dip_dir

    corners = np.array([[0,0],[width-1,0],[0,height-1],[width-1,height-1]]) - origin
    L_corners = corners[:,0]*n[0] + corners[:,1]*n[1]
    L_max = np.max(L_corners[L_corners > 0])
    fault_L_max[f['name']] = L_max

    L_normalized = np.clip(L / L_max, 0, 1)
    apex = 0.8
    taper = np.where(L_normalized <= apex, L_normalized/apex, (1-L_normalized)/(1-apex))
    taper = np.where(L < 0, 0, taper)

    rake = np.radians(f.get('rake', 0.0))
    dip = np.radians(f.get('dip', 90.0))
    rate_strike = f['rate'] * np.cos(rake)
    rate_dip = f['rate'] * np.sin(rake)

    # 走滑
    fault_zone_width = locking_depth
    v_strike = (rate_strike / 2) * np.tanh(-D / fault_zone_width) * taper
    U_strike = v_strike * n[0]
    V_strike = v_strike * n[1]
    W_strike = np.zeros_like(v_strike)

    # 逆冲
    if np.abs(rate_dip) > 1e-10:
        dip_rad = max(dip, np.radians(10))
        W_fault = locking_depth / np.tan(dip_rad)
        transition_width = W_fault * 1.5
        decay_scale = 5.0 * W_fault

        x_norm = D_signed / transition_width
        tanh_val = np.tanh(x_norm)
        vertical_factor = tanh_val
        far_field_decay = np.exp(-np.abs(D_signed) / decay_scale)

        W_dip = rate_dip * np.sin(dip) * vertical_factor * far_field_decay * taper
        shortening = (rate_dip * np.cos(dip) / 2) * np.tanh(-D_signed / W_fault) * far_field_decay * taper
        U_dip = shortening * n_perp[0] * dip_dir
        V_dip = shortening * n_perp[1] * dip_dir
    else:
        U_dip = np.zeros_like(X)
        V_dip = np.zeros_like(Y)
        W_dip = np.zeros_like(X)
    

    U_fault = U_strike + U_dip 
    V_fault = V_strike + V_dip 
    W_fault = W_strike + W_dip

    U_total += U_fault
    V_total += V_fault
    W_total += W_fault

    fault_fields[f['name']] = {
        'u' : U_fault.copy(),
        'v' : V_fault.copy(),
        'w' : W_fault.copy(),
    }

# ============================================================
# 4. 断层线裁剪：只保留在图像范围 [0,width] x [0,height] 内的线段
# ============================================================
def clip_fault_line(origin, direction, L_max, x_min, x_max, y_min, y_max):
    """
    从 origin 沿 direction 延伸 L_max，
    使用参数化方法裁剪到矩形边界内，返回裁剪后的 (x, y) 端点。
    """
    n = direction / np.linalg.norm(direction)
    # 参数 t 范围：[0, L_max]
    t0, t1 = 0.0, L_max

    # 对每条边界裁剪
    # x_min: origin[0] + t*n[0] >= x_min  =>  t >= (x_min - origin[0]) / n[0]
    for axis, lo, hi in [(0, x_min, x_max), (1, y_min, y_max)]:
        d = n[axis]
        o = origin[axis]
        if abs(d) > 1e-10:
            t_lo = (lo - o) / d
            t_hi = (hi - o) / d
            if d > 0:
                t0 = max(t0, t_lo)
                t1 = min(t1, t_hi)
            else:
                t0 = max(t0, t_hi)
                t1 = min(t1, t_lo)
        else:
            # 平行于该轴，检查是否在范围内
            if not (lo <= o <= hi):
                return None  # 完全在范围外

    if t1 <= t0:
        return None  # 裁剪后无有效线段

    x0 = origin[0] + t0 * n[0]
    y0 = origin[1] + t0 * n[1]
    x1 = origin[0] + t1 * n[0]
    y1 = origin[1] + t1 * n[1]
    return ([x0, x1], [y0, y1])

# ============================================================
# 5. 断层全称标签（对照参考图）
# ============================================================
fault_labels = {
    'xshf': 'Xianshuihe fault',
    'anhf': 'Anninghe fault',
    'dlsf': 'Daliangshan fault'
}

def draw_faults(ax):
    """绘制裁剪后的断层线（实线，对照参考图去掉虚线）"""
    for f in fault_configs:
        result = clip_fault_line(
            origin, f['dir'], fault_L_max[f['name']],
            x_min=0, x_max=width, y_min=0, y_max=height
        )
        if result is not None:
            xs, ys = result
            ax.plot(xs, ys, color=f['color'], lw=2.0, ls='--',
                    label=fault_labels[f['name']])
    # 交汇点：空心圆，对照参考图
    ax.plot(origin[0], origin[1], 'o', ms=7,
            mfc='white', mec='black', mew=1.2, zorder=6,
            label=f'Junction({origin[0]},{origin[1]})')

def save_single_eps(data, cmap, vmin, vmax,
                    colorbar_label, panel_label,
                    filename, streamplot_data=None):
    """
    对照参考图风格绘图并导出 EPS：
    - 面板标签（G/H）在图像左上角外侧，粗体
    - 图例在图像内部左上角，含断层全称 + 交汇点
    - colorbar 竖排标签，与 axes 等高
    - 坐标轴：Distance(km)，刻度对齐参考图
    """
    # ---------- 尺寸计算 ----------
    ax_w   = 2.5    # axes 内容宽（inch）
    ax_h   = ax_w * (height / width)  # 严格保持 1:5 纵横比
    cb_w   = 0.20   # colorbar 宽
    pad_cb = 0.15   # axes 与 colorbar 间距
    # colorbar label 是竖排文字，约需 0.55 inch
    margin_l = 0.70  # ylabel + 刻度
    margin_b = 0.50  # xlabel + 刻度
    margin_t = 0.50  # 面板标签（在 axes 上方）
    margin_r = 0.75  # colorbar + 竖排 label

    fig_w = margin_l + ax_w + pad_cb + cb_w + margin_r
    fig_h = margin_b + ax_h + margin_t

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---------- axes 定位 ----------
    left_f   = margin_l / fig_w
    bottom_f = margin_b / fig_h
    w_f      = ax_w / fig_w
    h_f      = ax_h / fig_h
    ax = fig.add_axes([left_f, bottom_f, w_f, h_f])

    # ---------- 主图 ----------
    im = ax.imshow(data, origin='lower', cmap=cmap, aspect='auto',
                   extent=[0, width, 0, height], vmin=vmin, vmax=vmax)

    # 流线
    if streamplot_data is not None:
        U, V = streamplot_data
        ax.streamplot(x_coords, y_coords, U, V,
                      color='white', density=2.0, linewidth=0.7,
                      arrowsize=0.8)

    draw_faults(ax)

    # ---------- 坐标轴（对照参考图）----------
    ax.set_xlabel('Distance(km)', fontsize=9)
    ax.set_ylabel('Distance(km)', fontsize=9)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    # x 刻度：0, 50, 100, 150, 200
    ax.set_xticks([0, 50, 100, 150, 200])
    # y 刻度：0, 200, 400, 600, 800, 1000
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    ax.tick_params(labelsize=8, direction='out', length=4, width=0.8)
    # 四边都显示刻度线，只外侧有标签（对照参考图）
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)

    # ---------- 图例（内部左上）----------
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85,
              handlelength=1.8, borderpad=0.6, labelspacing=0.4,
              bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes)



    # ---------- colorbar ----------
    cb_left_f = (margin_l + ax_w + pad_cb) / fig_w
    cb_bot_f  = margin_b / fig_h
    cb_wf     = cb_w / fig_w
    cb_hf     = ax_h / fig_h
    cax = fig.add_axes([cb_left_f, cb_bot_f, cb_wf, cb_hf])

    cbar = fig.colorbar(im, cax=cax)
    # 竖排标签，对照参考图
    cbar.set_label(colorbar_label, fontsize=8, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5, pad=3, direction='out')
    # colorbar 刻度在右侧
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    # ---------- 导出 ----------
    fig.savefig(filename, format='eps', facecolor='white', dpi=600)
    print(f"Saved: {filename}")
    plt.close(fig)

# ============================================================
# 6. 分别导出两张图
# ============================================================
V_mag_h = np.sqrt(U_total**2 + V_total**2)

# 图A：水平速度场（面板标签 G，对照参考图）
save_single_eps(
    data=V_mag_h,
    cmap='viridis',
    vmin=V_mag_h.min(), vmax=V_mag_h.max(),
    colorbar_label='Displacement rate magnitude(mm/yr)',
    panel_label='G',
    filename='horizontal_velocity.eps',
    streamplot_data=(U_total, V_total)
)

# 图B：垂直速度场（面板标签 H）
w_max = max(np.abs(W_total).max(), 0.1)
save_single_eps(
    data=W_total,
    cmap='RdBu_r',
    vmin=-w_max, vmax=w_max,
    colorbar_label='Vertical rate(mm/yr)',
    panel_label='H',
    filename='vertical_velocity.eps',
    streamplot_data=None
)

# ============================================================
# 7. 保存CSV
# ============================================================
def save_badlands_lcol(filename, arr):
    np.savetxt(filename, arr.ravel(order='C'), fmt='%.6e')

save_badlands_lcol('total_u.csv', U_total)
save_badlands_lcol('total_v.csv', V_total)
save_badlands_lcol('total_w.csv', W_total)

save_badlands_lcol('xshf_u.csv', fault_fields['xshf']['u'])
save_badlands_lcol('xshf_v.csv', fault_fields['xshf']['v'])

save_badlands_lcol('anhf_u.csv', fault_fields['anhf']['u'])
save_badlands_lcol('anhf_v.csv', fault_fields['anhf']['v'])

save_badlands_lcol('dlsf_u.csv', fault_fields['dlsf']['u'])
save_badlands_lcol('dlsf_v.csv', fault_fields['dlsf']['v'])
save_badlands_lcol('dlsf_w.csv', fault_fields['dlsf']['w'])

print(f"W_total range = [{W_total.min():.6f}, {W_total.max():.6f}] mm/yr")
print("\nDone.")