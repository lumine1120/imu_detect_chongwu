import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from matplotlib.widgets import Button

# 读取CSV文件
file_path = '/Users/lumine/Nutstore Files/我的坚果云/chongwu/运行版本1/data/imu_log_20251118_155301.csv'
df = pd.read_csv(file_path)

# 设置要分析的行范围（可以根据需要修改这些值）
start_row = 5000  # 起始行
end_row = 9000   # 结束行

# 提取指定范围的数据
df_range = df.iloc[start_row:end_row]

# 提取datetime, AccX, AccY, AccZ列
datetime = df_range['datetime'].values
acc_x = df_range['AccX'].values
acc_y = df_range['AccY'].values
acc_z = df_range['AccZ'].values

# 计算加速度的平方根（加速度向量的模）
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# 使用find_peaks找到峰值点
# 可以调整这些参数来优化峰值检测
peaks, properties = find_peaks(acc_magnitude, 
                               height=None,      # 最小峰值高度
                               distance=20,      # 峰之间的最小距离
                               prominence=0.5)   # 峰的突出度

# 创建一个列表来跟踪要保留的峰值点（初始所有峰值都保留）
peaks_to_keep = list(range(len(peaks)))

print(f"分析范围: 第 {start_row} 行到第 {end_row} 行")
print(f"找到 {len(peaks)} 个峰值点")
print(f"峰值位置索引: {peaks}")

# 更新图表的函数
def update_plots():
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # 获取当前保留的峰值
    current_peaks = peaks[peaks_to_keep]
    
    # 第一个图：加速度波形图并标记峰值点
    ax1.plot(range(len(acc_magnitude)), acc_magnitude, 'b-', linewidth=1, label='加速度幅值')
    if len(current_peaks) > 0:
        ax1.plot(current_peaks, acc_magnitude[current_peaks], 'rx', markersize=10, markeredgewidth=2, label='峰值点')
    ax1.set_xlabel('数据点索引', fontsize=12)
    ax1.set_ylabel('加速度幅值 (√(X²+Y²+Z²))', fontsize=12)
    ax1.set_title(f'加速度波形图及峰值检测 (行 {start_row}-{end_row}) [剩余 {len(current_peaks)} 个峰值]', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 添加峰值点的标注
    for i, peak_idx in enumerate(peaks_to_keep):
        peak = peaks[peak_idx]
        ax1.annotate(f'{peak_idx+1}', 
                    xy=(peak, acc_magnitude[peak]), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='red')
    
    # 第二个图：每个峰值点的值（折线图）
    if len(current_peaks) > 0:
        # 获取每个峰值点的幅值
        peak_values = acc_magnitude[current_peaks]
        
        # 绘制折线图
        ax2.plot(range(1, len(current_peaks) + 1), peak_values, 'g-o', linewidth=2, markersize=6, 
                 markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=1.5,
                 label='峰值幅值')
        
        ax2.set_xlabel('峰值序号', fontsize=12)
        ax2.set_ylabel('峰值幅值', fontsize=12)
        ax2.set_title('各峰值点的幅值变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 添加数值标签
        for i, value in enumerate(peak_values):
            ax2.text(i + 1, value, f'{value:.3f}', 
                    ha='center', 
                    va='bottom',
                    fontsize=7,
                    color='darkgreen')
    else:
        ax2.text(0.5, 0.5, '未找到峰值点或已全部删除', 
                ha='center', va='center', 
                transform=ax2.transAxes,
                fontsize=14)
        ax2.set_title('各峰值点的幅值变化', fontsize=14, fontweight='bold')
    
    # 第三个图：峰值幅值的平滑曲线
    if len(current_peaks) > 0:
        # 获取每个峰值点的幅值
        peak_values = acc_magnitude[current_peaks]
        
        # 对峰值数据进行平滑处理
        if len(peak_values) >= 5:
            # 使用Savitzky-Golay滤波器平滑
            # 增大window_length使曲线更平滑（原值为11，改为21）
            window_length = min(15, len(peak_values) if len(peak_values) % 2 == 1 else len(peak_values) - 1)
            if window_length < 5:
                window_length = 5
            # 降低polyorder使曲线更平滑（原值为3，改为2）
            polyorder = min(3, window_length - 1)
            smoothed_values = savgol_filter(peak_values, window_length, polyorder)
        elif len(peak_values) >= 3:
            # 使用简单移动平均
            window = 3
            smoothed_values = np.convolve(peak_values, np.ones(window)/window, mode='same')
        else:
            # 数据点太少，不平滑
            smoothed_values = peak_values
        
        # 对平滑后的数据使用find_peaks找峰值
        smoothed_peaks, _ = find_peaks(smoothed_values, 
                                       height=None,
                                       distance=5,       # 峰之间的最小距离
                                       prominence=0.1)   # 峰的突出度
        
        # 绘制原始数据和平滑后的数据
        ax3.plot(range(1, len(current_peaks) + 1), peak_values, 'g-o', linewidth=1, markersize=4, 
                 alpha=0.5, markerfacecolor='lightgreen', markeredgecolor='green', 
                 label='原始峰值幅值')
        ax3.plot(range(1, len(current_peaks) + 1), smoothed_values, 'b-', linewidth=2.5, 
                 label='平滑后峰值幅值')
        
        # 标记平滑曲线上的峰值点
        if len(smoothed_peaks) > 0:
            ax3.plot(smoothed_peaks + 1, smoothed_values[smoothed_peaks], 'r*', 
                    markersize=12, markeredgewidth=1.5, 
                    label=f'平滑曲线峰值点 ({len(smoothed_peaks)}个)')
            
            # 为每个峰值点添加标注
            for i, peak in enumerate(smoothed_peaks):
                ax3.annotate(f'{i+1}', 
                            xy=(peak + 1, smoothed_values[peak]), 
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8,
                            color='red',
                            fontweight='bold')
        
        ax3.set_xlabel('峰值序号', fontsize=12)
        ax3.set_ylabel('峰值幅值', fontsize=12)
        ax3.set_title('峰值幅值变化（平滑处理）', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, '未找到峰值点或已全部删除', 
                ha='center', va='center', 
                transform=ax3.transAxes,
                fontsize=14)
        ax3.set_title('峰值幅值变化（平滑处理）', fontsize=14, fontweight='bold')
    
    fig.canvas.draw()

# 鼠标点击事件处理函数
def on_click(event):
    global peaks_to_keep, peaks
    
    # 只响应左键点击，且点击在ax1上
    if event.button != 1 or event.inaxes != ax1:
        # 处理在第二个图上的点击（删除功能）
        if event.button == 1 and event.inaxes == ax2:
            if len(peaks_to_keep) == 0:
                print("没有可删除的峰值点")
                return
            
            click_x = event.xdata
            if click_x is None:
                return
            
            # 找到最近的峰值序号（1-based）
            peak_number = round(click_x)
            if 1 <= peak_number <= len(peaks_to_keep):
                idx_to_remove = peak_number - 1
                peak_to_remove = peaks_to_keep[idx_to_remove]
                print(f"删除峰值点 {peak_to_remove + 1} (索引: {peaks[peak_to_remove]})")
                peaks_to_keep.pop(idx_to_remove)
                update_plots()
        return
    
    # 点击在第一个图上
    click_x = event.xdata
    if click_x is None:
        return
    
    # 确保点击位置在有效范围内
    click_x = int(round(click_x))
    if click_x < 0 or click_x >= len(acc_magnitude):
        return
    
    # 找到最近的峰值点
    current_peaks = peaks[peaks_to_keep]
    
    if len(current_peaks) > 0:
        # 计算点击位置与每个峰值的距离（使用屏幕像素距离）
        # 将数据坐标转换为显示坐标
        min_pixel_distance = float('inf')
        nearest_idx = -1
        
        for i, peak_idx in enumerate(peaks_to_keep):
            peak_x = peaks[peak_idx]
            peak_y = acc_magnitude[peak_x]
            
            # 转换为显示坐标（像素）
            peak_display = ax1.transData.transform([(peak_x, peak_y)])[0]
            click_display = ax1.transData.transform([(click_x, event.ydata)])[0]
            
            # 计算欧氏距离
            pixel_dist = np.sqrt((peak_display[0] - click_display[0])**2 + 
                                (peak_display[1] - click_display[1])**2)
            
            if pixel_dist < min_pixel_distance:
                min_pixel_distance = pixel_dist
                nearest_idx = i
        
        # 如果点击距离足够近（像素距离小于20像素），则删除该点
        if min_pixel_distance < 20:
            peak_to_remove = peaks_to_keep[nearest_idx]
            print(f"删除峰值点 {peak_to_remove + 1} (索引: {peaks[peak_to_remove]})")
            peaks_to_keep.pop(nearest_idx)
            update_plots()
            return
    
    # 如果不是删除操作，则添加新峰值点
    # 定义搜索窗口大小（以点击位置为中心）
    window_size = 500  # 窗口大小
    half_window = window_size // 2
    
    # 计算窗口范围
    start_idx = max(0, click_x - half_window)
    end_idx = min(len(acc_magnitude), click_x + half_window)
    
    # 在窗口范围内使用find_peaks找峰值
    search_region = acc_magnitude[start_idx:end_idx]
    local_peaks, _ = find_peaks(search_region, 
                                height=None,
                                distance=20,
                                prominence=0.5)
    
    if len(local_peaks) == 0:
        # 如果窗口内没有找到峰值，就用最高点
        local_max_idx = np.argmax(search_region)
        new_peak_idx = start_idx + local_max_idx
        print(f"窗口内未找到明显峰值，使用最高点")
    else:
        # 找到距离点击位置最近的峰值
        distances_to_click = np.abs(local_peaks - (click_x - start_idx))
        nearest_peak_in_window = local_peaks[np.argmin(distances_to_click)]
        new_peak_idx = start_idx + nearest_peak_in_window
    
    # 检查这个位置是否已经是峰值点
    if new_peak_idx in peaks:
        peak_position = np.where(peaks == new_peak_idx)[0][0]
        if peak_position not in peaks_to_keep:
            # 如果这个峰值点之前被删除了，重新添加
            peaks_to_keep.append(peak_position)
            peaks_to_keep.sort()
            print(f"恢复峰值点 {peak_position + 1} (索引: {new_peak_idx}, 幅值: {acc_magnitude[new_peak_idx]:.4f})")
            update_plots()
        else:
            print(f"该位置已经是峰值点 {peak_position + 1}")
    else:
        # 添加新的峰值点
        # 将新峰值插入到peaks数组的正确位置（保持sorted）
        insert_pos = np.searchsorted(peaks, new_peak_idx)
        peaks = np.insert(peaks, insert_pos, new_peak_idx)
        
        # 更新peaks_to_keep索引（因为在insert_pos之后的索引都需要+1）
        peaks_to_keep = [idx + 1 if idx >= insert_pos else idx for idx in peaks_to_keep]
        peaks_to_keep.append(insert_pos)
        peaks_to_keep.sort()
        
        print(f"添加新峰值点 {insert_pos + 1} (索引: {new_peak_idx}, 幅值: {acc_magnitude[new_peak_idx]:.4f})")
        update_plots()

# 重置按钮回调函数
def reset_peaks(event):
    global peaks_to_keep
    peaks_to_keep = list(range(len(peaks)))
    print("已重置所有峰值点")
    update_plots()

# 保存按钮回调函数
def save_data(event):
    current_peaks = peaks[peaks_to_keep]
    
    if len(current_peaks) > 0:
        peaks_data = pd.DataFrame({
            '峰值序号': range(1, len(current_peaks) + 1),
            '数据点索引': current_peaks,
            '实际行号': current_peaks + start_row,
            '峰值幅值': acc_magnitude[current_peaks],
            'AccX': acc_x[current_peaks],
            'AccY': acc_y[current_peaks],
            'AccZ': acc_z[current_peaks]
        })
        peaks_data.to_csv('peaks_data.csv', index=False, encoding='utf-8-sig')
        print(f"\n峰值数据已保存为: peaks_data.csv (共 {len(current_peaks)} 个峰值)")
        
        plt.savefig('imu_peaks_analysis.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存为: imu_peaks_analysis.png")
    else:
        print("没有峰值点可保存")

# 创建图形
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
plt.subplots_adjust(bottom=0.06)

# 添加按钮
ax_reset = plt.axes([0.3, 0.02, 0.15, 0.04])
ax_save = plt.axes([0.55, 0.02, 0.15, 0.04])
btn_reset = Button(ax_reset, '重置峰值点')
btn_save = Button(ax_save, '保存数据')
btn_reset.on_clicked(reset_peaks)
btn_save.on_clicked(save_data)

# 绑定点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

# 初始绘图
update_plots()

# 打印初始峰值信息
print(f"\n各峰值点的幅值:")
for i, peak_idx in enumerate(peaks):
    print(f"峰{i+1} (索引{peak_idx}): 幅值={acc_magnitude[peak_idx]:.4f}")

print("\n=== 使用说明 ===")
print("• 点击红色峰值点附近可以删除该点")
print("• 点击没有峰值点的区域会自动找到附近最高点并添加为新峰值")
print("• 在第二个图（折线图）上点击可以删除对应的峰值点")
print("• 点击'重置峰值点'按钮可以恢复到初始检测的峰值点")
print("• 点击'保存数据'按钮可以保存当前的峰值点到CSV文件和图片")
print("================\n")

plt.show()
