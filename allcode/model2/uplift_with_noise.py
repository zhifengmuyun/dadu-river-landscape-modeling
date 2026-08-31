import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

# ===========================
# 1. 基础参数与网格设置
# ===========================
width, height = 200, 1000
origin = np.array([70, 400])  # 拐点/交汇点坐标
sigma_uplift_v = 15
sigma_uplift_h = 30           # 隆升带宽度参数 (km)
peak_rate_v = 0.69           # 最大隆升速率 (mm/a)
peak_rate_h = 0.45

# 生成坐标网格 (1000行 x 200列)
x_coords = np.arange(width)
y_coords = np.arange(height)
X, Y = np.meshgrid(x_coords, y_coords)

# ===========================
# 2. 噪声参数设置
# ===========================
add_noise = True              # 是否添加噪声
noise_type = 'fractal'        # 可选: 'simple', 'correlated', 'fractal'
noise_amplitude = 0.15        # 噪声幅度（相对于峰值速率的比例，0.15 = 15%）
correlation_length = 0.3        # 空间相关长度（km），需远小于sigma_uplift以产生锯齿感
random_seed = 42              # 随机种子（保证可重复）

# 噪声叠加方式
# 'multiplicative' - 乘法噪声，噪声随隆升值衰减（σ小时噪声弱）
# 'additive'       - 加法噪声，噪声有独立的空间包络，不受隆升σ束缚
noise_mode = 'additive'

# 加法噪声的独立包络宽度（仅 additive 模式有效）
# 建议取隆升σ的 2~3 倍，使噪声在隆升翼部仍有表现
sigma_noise_v = 25            # 纵向臂噪声包络宽度 (km)
sigma_noise_h = 60            # 横向臂噪声包络宽度 (km)

# ===========================
# 3. 噪声生成函数
# ===========================
def generate_noise(shape, noise_type, amplitude, corr_length, seed):
    """
    生成不同类型的随机噪声

    参数:
        shape: 输出数组形状
        noise_type: 噪声类型 ('simple', 'correlated', 'fractal')
        amplitude: 噪声幅度
        corr_length: 空间相关长度
        seed: 随机种子

    返回:
        noise: 噪声数组，范围约 [-amplitude, +amplitude]
    """
    np.random.seed(seed)

    if noise_type == 'simple':
        # 简单白噪声（无空间相关性）
        return amplitude * np.random.randn(*shape)

    elif noise_type == 'correlated':
        # 空间相关噪声（单尺度）
        white_noise = np.random.randn(*shape)
        corr_noise = gaussian_filter(white_noise, sigma=corr_length)
        corr_noise = corr_noise / np.max(np.abs(corr_noise))
        return amplitude * corr_noise

    elif noise_type == 'fractal':
        # 分形噪声（多尺度叠加，最真实）
        noise = np.zeros(shape)
        amp = 1.0
        total_amp = 0.0

        for i in range(4):  # 4层叠加
            scale = corr_length / (2 ** i)
            layer = gaussian_filter(np.random.randn(*shape), sigma=max(scale, 1))
            noise += amp * layer
            total_amp += amp
            amp *= 0.5  # 每层幅度减半

        noise = noise / total_amp
        noise = noise / np.max(np.abs(noise))
        return amplitude * noise

    return np.zeros(shape)

# ===========================
# 4. 计算点到线段的距离
# ===========================
p_top = np.array([70, 1000])   # 纵向臂端点
p_right = np.array([200, 400]) # 横向臂端点

def dist_to_segment(px, py, p1, p2):
    """计算点(px, py)到线段(p1, p2)的最短距离"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return np.sqrt((px-p1[0])**2 + (py-p1[1])**2)
    t = np.clip(((px - p1[0]) * dx + (py - p1[1]) * dy) / l2, 0, 1)
    return np.sqrt((px - (p1[0] + t * dx))**2 + (py - (p1[1] + t * dy))**2)

# 计算每个网格点到两条线段的距离
dist_v = dist_to_segment(X, Y, origin, p_top)
dist_h = dist_to_segment(X, Y, origin, p_right)

# ===== 纵向臂左侧高原修改 =====
# 高斯右侧（x > 中心线）保持不变，左侧（x < 中心线）改为平坦高原：
# 方法：只保留向右的 x 距离分量，左侧 x 分量截为 0，
#       这样左侧点的有效距离 ≈ 0，高斯输出恒等于峰值。
closest_y_on_seg = np.clip(Y, origin[1], p_top[1])   # 线段上最近点的 y 坐标
dx_from_line = X - origin[0]                          # x 方向位移（正=右/东）
dy_from_seg  = Y - closest_y_on_seg                   # y 方向位移（超出线段端点部分）
dx_plateau   = np.maximum(dx_from_line, 0)            # 左侧 x 分量截为 0
dist_v = np.sqrt(dx_plateau**2 + dy_from_seg**2)      # 覆盖原始 dist_v

# ===========================
# 5. 计算隆升场（高斯 + 噪声）
# ===========================
# 基础高斯隆升
W_vertical_arm = peak_rate_v * np.exp(-(dist_v**2 / (2 * sigma_uplift_v**2)))
W_horizontal_arm = peak_rate_h * np.exp(-(dist_h**2 / (2 * sigma_uplift_h**2)))

# ===== 横向臂空间限制 =====
# 只保留东侧一半（x > x_cutoff）
x_cutoff = 100  # 截断位置（km），只保留 x > 100 的部分
horizontal_mask = X >= x_cutoff
W_horizontal_arm = W_horizontal_arm * horizontal_mask

# 保存无噪声版本用于对比
W_v_clean = W_vertical_arm.copy()
W_h_clean = W_horizontal_arm.copy()

# 添加噪声
if add_noise:
    noise = generate_noise(
        W_vertical_arm.shape,
        noise_type,
        noise_amplitude,
        correlation_length,
        random_seed
    )

    if noise_mode == 'additive':
        # 加法噪声：噪声有独立的空间包络，不受隆升σ束缚
        # 包络比隆升高斯更宽，使噪声在隆升翼部仍然可见
        envelope_v = np.exp(-(dist_v**2 / (2 * sigma_noise_v**2)))
        envelope_h = np.exp(-(dist_h**2 / (2 * sigma_noise_h**2)))
        envelope_h = envelope_h * horizontal_mask  # 保持横向臂截断

        # noise 已经包含 noise_amplitude 缩放，不再重复乘
        W_vertical_arm = W_vertical_arm + peak_rate_v * noise * envelope_v
        W_horizontal_arm = W_horizontal_arm + peak_rate_h * noise * envelope_h
    else:
        # 乘法噪声（原方案）：噪声随隆升值衰减
        W_vertical_arm = W_vertical_arm * (1 + noise)
        W_horizontal_arm = W_horizontal_arm * (1 + noise)

    # 确保非负
    W_vertical_arm = np.maximum(W_vertical_arm, 0)
    W_horizontal_arm = np.maximum(W_horizontal_arm, 0)
else:
    noise = np.zeros_like(W_vertical_arm)

# ===========================
# 6. 保存为 CSV 文件
# ===========================
np.savetxt("uplift_v_arm_200k.csv", W_vertical_arm.flatten(), fmt='%.6e')
np.savetxt("uplift_h_arm_200k.csv", W_horizontal_arm.flatten(), fmt='%.6e')



# ===========================
# 7. 打印参数表格
# ===========================
print("\n" + "="*70)
print("隆起参数表")
print("="*70)
print(f"{'参数':<25} {'纵向臂 (Vertical)':<20} {'横向臂 (Horizontal)':<20}")
print("-"*70)
print(f"{'隆起中心线':<25} {'origin -> p_top':<20} {'origin -> p_right':<20}")
print(f"{'隆起宽度 sigma (km)':<25} {sigma_uplift_v:<20.1f} {sigma_uplift_h:<20.1f}")
print(f"{'最大速率 (mm/a)':<25} {peak_rate_v:<20.2f} {peak_rate_h:<20.2f}")
print("="*70)
print(f"\n基础公式: w(x,y) = A * exp(-d²/(2σ²))")
print(f"\n噪声设置:")
print(f"  - 是否添加噪声: {add_noise}")
print(f"  - 噪声类型: {noise_type}")
print(f"  - 噪声幅度: {noise_amplitude*100:.0f}%")
print(f"  - 空间相关长度: {correlation_length} km")
print(f"  - 噪声叠加方式: {noise_mode}")
if noise_mode == 'additive':
    print(f"  - 噪声包络宽度 (纵向): {sigma_noise_v} km")
    print(f"  - 噪声包络宽度 (横向): {sigma_noise_h} km")
print(f"  - 随机种子: {random_seed}")
print("\n成功保存以下文件：")
print("  - uplift_v_arm_200k.csv (纵向臂)")
print("  - uplift_h_arm_200k.csv (横向臂)")
print("  - uplift_combined_200k.csv (合并)")

# ===========================
# 8. 可视化 - 分开绘制纵向和横向臂
# ===========================
W_clean = W_v_clean + W_h_clean
W_noisy = W_vertical_arm + W_horizontal_arm

# 确定统一的颜色范围
vmin = 0.001
vmax = max(W_vertical_arm.max(), W_horizontal_arm.max(), W_noisy.max())

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (A) 纵向臂 - Vertical Arm
ax = axes[0, 0]
im = ax.imshow(W_vertical_arm, origin='lower', extent=[0, 200, 0, 1000],
               cmap='YlOrBr', norm=LogNorm(vmin=vmin, vmax=vmax))
ax.plot([origin[0], p_top[0]], [origin[1], p_top[1]], 'b--', linewidth=2.5, alpha=0.8)
ax.plot(origin[0], origin[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
ax.set_title(f'(A) Vertical Arm (max={peak_rate_v:.2f} mm/a)', fontsize=13, fontweight='bold')
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Distance North (km)', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.6, label='Uplift rate (mm/a)')

# (B) 横向臂 - Horizontal Arm
ax = axes[0, 1]
im = ax.imshow(W_horizontal_arm, origin='lower', extent=[0, 200, 0, 1000],
               cmap='YlOrBr', norm=LogNorm(vmin=vmin, vmax=vmax))
ax.plot([origin[0], p_right[0]], [origin[1], p_right[1]], 'r--', linewidth=2.5, alpha=0.8)
ax.plot(origin[0], origin[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
ax.axvline(x=x_cutoff, color='white', linestyle=':', linewidth=2, label=f'Cutoff x={x_cutoff}')
ax.set_title(f'(B) Horizontal Arm (max={peak_rate_h:.2f} mm/a, x≥{x_cutoff})', fontsize=13, fontweight='bold')
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Distance North (km)', fontsize=11)
ax.legend(loc='upper left')
plt.colorbar(im, ax=ax, shrink=0.6, label='Uplift rate (mm/a)')

# (C) 合并 - Combined
ax = axes[0, 2]
im = ax.imshow(W_noisy, origin='lower', extent=[0, 200, 0, 1000],
               cmap='YlOrBr', norm=LogNorm(vmin=vmin, vmax=vmax))
ax.plot([origin[0], p_top[0]], [origin[1], p_top[1]], 'b--', linewidth=2, alpha=0.8, label='Vertical')
ax.plot([origin[0], p_right[0]], [origin[1], p_right[1]], 'r--', linewidth=2, alpha=0.8, label='Horizontal')
ax.plot(origin[0], origin[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
ax.set_title('(C) Combined Uplift Field', fontsize=13, fontweight='bold')
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Distance North (km)', fontsize=11)
ax.legend(loc='upper right')
plt.colorbar(im, ax=ax, shrink=0.6, label='Uplift rate (mm/a)')

# (D) 噪声场
ax = axes[1, 0]
if add_noise:
    im = ax.imshow(noise, origin='lower', extent=[0, 200, 0, 1000],
                   cmap='RdBu_r', vmin=-noise_amplitude, vmax=noise_amplitude)
    ax.set_title(f'(D) Noise Field ({noise_type}, corr={correlation_length}km)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Noise amplitude')
else:
    ax.text(0.5, 0.5, 'No Noise Added', ha='center', va='center',
            transform=ax.transAxes, fontsize=16)
    ax.set_title('(D) Noise Field', fontsize=13, fontweight='bold')
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Distance North (km)', fontsize=11)

# (E) 纵向臂横剖面 (Y=700)
ax = axes[1, 1]
profile_y_v = 700
ax.plot(x_coords, W_v_clean[profile_y_v, :], 'b--', linewidth=2, label='Original', alpha=0.7)
ax.plot(x_coords, W_vertical_arm[profile_y_v, :], 'b-', linewidth=1.5, label='With noise')
ax.axvline(x=origin[0], color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Uplift rate (mm/a)', fontsize=11)
ax.set_title(f'(E) Vertical Arm Cross-section (Y={profile_y_v} km)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(max(0, origin[0] - 4*sigma_noise_v), min(width, origin[0] + 4*sigma_noise_v))

# (F) 横向臂横剖面 (Y=400)
ax = axes[1, 2]
profile_y_h = 400
ax.plot(x_coords, W_h_clean[profile_y_h, :], 'r--', linewidth=2, label='Original', alpha=0.7)
ax.plot(x_coords, W_horizontal_arm[profile_y_h, :], 'r-', linewidth=1.5, label='With noise')
ax.axvline(x=origin[0], color='gray', linestyle=':', alpha=0.5, label='Origin')
ax.axvline(x=x_cutoff, color='green', linestyle='--', alpha=0.8, label=f'Cutoff x={x_cutoff}')
ax.set_xlabel('Distance East (km)', fontsize=11)
ax.set_ylabel('Uplift rate (mm/a)', fontsize=11)
ax.set_title(f'(F) Horizontal Arm Cross-section (Y={profile_y_h} km)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('uplift_with_noise.png', dpi=150, bbox_inches='tight')
plt.savefig('uplift_with_noise.svg', format='svg', bbox_inches='tight')
print("\n可视化已保存为: uplift_with_noise.png 和 uplift_with_noise.svg")
plt.show()
