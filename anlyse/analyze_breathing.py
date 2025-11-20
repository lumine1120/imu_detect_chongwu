import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

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

print(f"数据范围: 第 {start_row} 行到第 {end_row} 行")
print(f"数据点数: {len(acc_magnitude)}")

# 设计低通滤波器（提取呼吸频率）
# 呼吸频率通常在 0.1-0.5 Hz (6-30次/分钟)
# 假设采样频率为200Hz（根据你的设备调整）
sampling_rate = 200  # Hz，根据实际采样率调整

# 设计Butterworth低通滤波器
# 截止频率设置为0.5Hz，可以保留呼吸信号
cutoff_freq = 0.5  # Hz
nyquist_freq = sampling_rate / 2
normalized_cutoff = cutoff_freq / nyquist_freq

# 创建4阶Butterworth低通滤波器
order = 4
b, a = butter(order, normalized_cutoff, btype='low', analog=False)

# 应用滤波器（使用filtfilt进行零相位滤波，避免相位失真）
filtered_magnitude = filtfilt(b, a, acc_magnitude)

print(f"采样频率: {sampling_rate} Hz")
print(f"低通滤波截止频率: {cutoff_freq} Hz")
print(f"滤波器阶数: {order}")

# 在滤波后的数据上检测峰值（呼吸峰值）
breathing_peaks, _ = find_peaks(filtered_magnitude, 
                                distance=sampling_rate,  # 至少间隔1秒
                                prominence=0.02)         # 峰的突出度

print(f"检测到 {len(breathing_peaks)} 个呼吸峰值")

# 计算呼吸频率
if len(breathing_peaks) > 1:
    # 计算相邻峰值之间的时间间隔（单位：秒）
    peak_intervals = np.diff(breathing_peaks) / sampling_rate
    avg_interval = np.mean(peak_intervals)
    breathing_rate = 60 / avg_interval  # 转换为次/分钟
    print(f"平均呼吸间隔: {avg_interval:.2f} 秒")
    print(f"估计呼吸频率: {breathing_rate:.1f} 次/分钟")

# 创建图形，显示两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 第一个图：原始加速度数据
time_axis = np.arange(len(acc_magnitude)) / sampling_rate  # 转换为时间轴（秒）

ax1.plot(time_axis, acc_magnitude, 'b-', linewidth=1, alpha=0.7, label='原始加速度幅值')
ax1.set_xlabel('时间 (秒)', fontsize=12)
ax1.set_ylabel('加速度幅值 (√(X²+Y²+Z²))', fontsize=12)
ax1.set_title(f'原始加速度波形 (行 {start_row}-{end_row})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 第二个图：低通滤波后的数据（呼吸信号）
ax2.plot(time_axis, filtered_magnitude, 'g-', linewidth=2, label='低通滤波后（呼吸信号）')

# 标记呼吸峰值
if len(breathing_peaks) > 0:
    ax2.plot(breathing_peaks / sampling_rate, filtered_magnitude[breathing_peaks], 
            'r*', markersize=12, markeredgewidth=1.5, label=f'呼吸峰值 ({len(breathing_peaks)}个)')
    
    # 为每个峰值添加序号标注
    for i, peak in enumerate(breathing_peaks):
        ax2.annotate(f'{i+1}', 
                    xy=(peak / sampling_rate, filtered_magnitude[peak]), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='red',
                    fontweight='bold')

ax2.set_xlabel('时间 (秒)', fontsize=12)
ax2.set_ylabel('滤波后幅值', fontsize=12)
title_text = f'低通滤波后的呼吸信号 (截止频率: {cutoff_freq} Hz)'
if len(breathing_peaks) > 1:
    title_text += f' - 呼吸频率: {breathing_rate:.1f} 次/分钟'
ax2.set_title(title_text, fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()

# 保存图表
output_filename = 'breathing_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n图表已保存为: {output_filename}")

# 如果检测到呼吸峰值，保存数据
if len(breathing_peaks) > 0:
    breathing_data = pd.DataFrame({
        '峰值序号': range(1, len(breathing_peaks) + 1),
        '数据点索引': breathing_peaks,
        '实际行号': breathing_peaks + start_row,
        '时间(秒)': breathing_peaks / sampling_rate,
        '滤波后幅值': filtered_magnitude[breathing_peaks]
    })
    
    # 计算相邻峰值间隔
    if len(breathing_peaks) > 1:
        intervals = np.diff(breathing_peaks) / sampling_rate
        breathing_data['间隔(秒)'] = [np.nan] + list(intervals)
        breathing_data['频率(次/分)'] = breathing_data['间隔(秒)'].apply(
            lambda x: 60/x if pd.notna(x) and x > 0 else np.nan
        )
    
    breathing_data.to_csv('breathing_data.csv', index=False, encoding='utf-8-sig')
    print(f"呼吸峰值数据已保存为: breathing_data.csv")

print("\n=== 使用说明 ===")
print("• 第一个图显示原始加速度数据")
print("• 第二个图显示低通滤波后的呼吸信号")
print("• 红色星号标记检测到的呼吸峰值")
print("• 可以通过调整采样频率(sampling_rate)和截止频率(cutoff_freq)来优化结果")
print("• 正常呼吸频率: 12-20次/分钟（成人静息）")
print("================\n")

plt.show()
