import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
import os

SAMPLING_RATE_HZ=100  # 采样率 (Hz)
HR_WINDOW_SIZE=10     # 心率曲线的移动平均窗口大小 (以间隔数为单位)

# --- 1. 定义要读取的文件名 ---
# *** 重要提示：请将下面的 'your_file.csv' 替换为您自己的文件名 ***
filename = 'C:/Users/84998/Desktop/pet/imu_data_20251110_200532.csv' 

# --- 2. 检查文件是否存在 ---
if not os.path.exists(filename):
    print(f"错误：文件 '{filename}' 未找到。")
    print("请确保文件与脚本在同一目录，或提供正确的文件路径。")
    # 如果文件不存在，可以退出或进行错误处理
    # exit() 
else:
    try:
        # --- 3. 从CSV文件读取数据 ---
        print(f"Reading data from '{filename}'...")
        df = pd.read_csv(filename)
        df = df.iloc[5000:150000]
        # 检查所需列是否存在
        required_cols = ['AccX', 'AccY', 'AccZ']
        if not all(col in df.columns for col in required_cols):
            print(f"错误：CSV文件必须包含以下列: {required_cols}")
            print(f"找到的列: {df.columns.tolist()}")
        else:
            # --- 4. 计算幅值 ---
            # 幅值 = sqrt(AccX^2 + AccY^2 + AccZ^2)
            print("Calculating magnitude...")
            df['Magnitude'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
            
            magnitude_data = df['Magnitude'].values
            
            # ---  希尔伯特变换 (Hilbert Transform)检测包络 ---
            print("Calculating Hilbert Transform envelope...")
            # 希尔伯特变换在去除直流分量(均值)后效果最好
            mean_offset = np.mean(magnitude_data)
            analytic_signal = hilbert(magnitude_data - mean_offset)
            # 包络是解析信号的绝对值，再加上均值
            envelope_hilbert = np.abs(analytic_signal) + mean_offset
            # --- 5. 检测峰值 ---
            #
            # *** 调参提示 ***:
            # 您很可能需要根据您的数据特性调整 'prominence' 和 'distance'
            # 'prominence': 峰值的“突出程度”。较小的值会检测到更多（可能包括噪声）的峰值。
            # 'distance': 两个峰值之间的最小水平距离（以样本点为单位）。
            #
            print("Detecting peaks... (您可能需要调整 'prominence' 和 'distance' 参数)")
            peaks, properties = find_peaks(magnitude_data, prominence=0.005, distance=30)
            
            if len(peaks) < 2:
                print("未检测到足够的峰值（少于2个），无法计算间隔。")
                print("请尝试调低 find_peaks 函数中的 'prominence' 和 'distance' 参数。")
            else:
                print(f"Detected {len(peaks)} peaks.")
                
                # --- 6. 绘制幅值波形和标记的峰值 ---
                print("Plotting magnitude with peaks...")
                plt.figure(figsize=(15, 6))
                plt.plot(magnitude_data, label='Magnitude')
                plt.plot(peaks, magnitude_data[peaks], "x", markersize=8, label='Detected Peaks')
                # 绘制包络
                # plt.plot(envelope_hilbert, label='Envelope (Hilbert Transform)', color='red', linestyle='--', linewidth=2)
                plt.title('Magnitude Waveform with Detected Peaks')
                plt.xlabel('Sample Index')
                plt.ylabel('Magnitude')
                plt.legend()
                plt.grid(True)
                plt.savefig('magnitude_with_peaks.png') # 保存图像
                plt.close() # 关闭图形，以免在下一步中重叠
                
                # 画峰值点，观察包络
                print("Plotting envelope with peaks...")
                plt.figure(figsize=(15, 6))
                plt.plot(peaks, magnitude_data[peaks], label=' Peaks waveform')
                plt.title('Peaks Waveform')
                plt.xlabel('Sample Index')
                plt.ylabel('Magnitude')
                plt.legend()
                plt.grid(True)
                plt.savefig('magnitude_only_peaks_wqve.png') # 保存图像
                plt.close() # 关闭图形，以免在下一步中重叠

                # --- 7. 计算峰值间隔 ---
                # 间隔是连续峰值索引之间的差异
                peak_intervals = np.diff(peaks)
                
                # --- 8. 绘制峰值间隔的变化曲线 ---
                # X轴：每个间隔开始处的峰值索引
                print("Plotting peak intervals...")
                plt.figure(figsize=(15, 6))
                plt.plot(peaks[:-1], peak_intervals, linestyle='-')
                plt.title('Peak Interval Variation')
                plt.xlabel('Sample Index (Start of Interval)')
                plt.ylabel('Peak Interval (in samples)')
                plt.grid(True)
                plt.savefig('peak_intervals.png') # 保存图像
                plt.close() # 关闭图形
                
                # --- 7. (新) 计算峰值间隔 (单位：样本数) ---
                # intervals_samples 数组的长度会是 len(peaks) - 1
                intervals_samples = np.diff(peaks)
                
                if len(intervals_samples) == 0:
                     print("只检测到1个峰值，无法计算间隔。")
                     exit()

                # --- 8. (新) 计算心率 ---
                print("Calculating heart rate...")
                
                # 将间隔从“样本数”转换成“秒”
                intervals_seconds = intervals_samples / SAMPLING_RATE_HZ
                
                # 计算瞬时心率 (BPM)
                # 60秒 / 每次心跳的秒数 = 每分钟心跳次数 (BPM)
                instant_hr_bpm = 60.0 / intervals_seconds
                
                # 计算总体的平均心率
                overall_avg_bpm = np.mean(instant_hr_bpm)
                print(f"\n--- 总体平均心率: {overall_avg_bpm:.2f} BPM ---")
                
                # --- 9. (新) 计算心率曲线 (5个间隔的移动平均) ---
                if len(instant_hr_bpm) < HR_WINDOW_SIZE:
                    print(f"检测到的间隔数 ({len(instant_hr_bpm)}) 少于窗口数 ({HR_WINDOW_SIZE})，无法绘制移动平均心率曲线。")
                else:
                    print(f"Calculating {HR_WINDOW_SIZE}-interval moving average HR curve...")
                    
                    # 使用 pandas 的 rolling().mean() 来计算移动平均
                    # 注意：我们对瞬时心率数组 'instant_hr_bpm' 进行平滑
                    hr_series = pd.Series(instant_hr_bpm)
                    hr_curve_bpm = hr_series.rolling(window=HR_WINDOW_SIZE, center=True).mean()
                    
                    # 'center=True' 会导致开头和结尾有NaN，这在绘图时是合理的
                    # 'center=False' (默认) 会导致曲线有延迟
                    
                    # --- 10. (新) 绘制心率变化曲线 ---
                    
                    # X轴：心率计算是基于间隔的，每个间隔结束于一个峰值
                    # 'instant_hr_bpm[0]' 对应的是 'peaks[0]' 和 'peaks[1]' 之间的间隔
                    # 我们将该心率值绘制在第二个峰值的位置，即 peaks[1:]
                    hr_curve_x_axis = peaks[1:]
                    
                    print("Plotting 'heart_rate_curve.png'...")
                    plt.figure(figsize=(15, 6))
                    
                    # 绘制平滑后的心率曲线
                    # hr_curve_bpm.values 包含NaN，matplotlib会自动处理（不绘制NaN的部分）
                    plt.plot(hr_curve_x_axis, hr_curve_bpm.values, marker='o', linestyle='-', label=f'{HR_WINDOW_SIZE}-Interval Moving Avg HR')
                    
                    # (可选) 绘制瞬时心率（原始、未平滑），用于对比
                    plt.plot(hr_curve_x_axis, instant_hr_bpm, 'o', markersize=3, alpha=0.3, label='Instantaneous HR')
                    
                    # 绘制一条红线表示总体平均心率
                    # plt.axhline(y=overall_avg_bpm, color='r', linestyle='--', label=f'Overall Avg HR ({overall_avg_bpm:.2f} BPM)')
                    
                    plt.title('Heart Rate Variation Curve')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Heart Rate (BPM)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig('heart_rate_curve.png')
                    plt.close()
                print("\n脚本执行完毕。")
                print("已生成3个文件：")
                print("1. 'magnitude_with_peaks.png': 幅值波形及标记的峰值。")
                print("2. 'peak_intervals.png': 峰值间隔的变化曲线。")
                print("2. 'heart_rate_curve.png': 平均心率的变化曲线。")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")