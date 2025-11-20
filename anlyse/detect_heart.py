"""
心率检测模块
从队列读取加速度数据，通过峰值检测算法实时计算心率
"""
import numpy as np
import queue
from threading import Thread, Event
from collections import deque
from scipy import signal
import time


class HeartRateDetector:
    def __init__(self, data_queue, stop_event, sample_rate=200, window_size=10):
        """
        初始化心率检测器
        
        Args:
            data_queue: 数据队列C
            stop_event: 停止信号
            sample_rate: 采样率(Hz)，默认200
            window_size: 计算心率的时间窗口(秒)，默认10秒
        """
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # 缓存数据用于峰值检测
        self.buffer_size = sample_rate * window_size
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # 心率历史记录
        self.heart_rate_history = deque(maxlen=100)
        
        # 峰值检测参数
        self.min_peak_distance = int(sample_rate * 0.3)  # 最小峰值间隔(防止检测到同一个峰)
        self.min_peak_height_percentile = 60  # 峰值高度阈值(百分位数)
        
        # 统计信息
        self.data_count = 0
        self.last_update_time = time.time()
        self.update_interval = 2  # 每2秒更新一次心率
        
    def calculate_magnitude(self, accX, accY, accZ):
        """
        计算加速度的幅值
        
        Args:
            accX, accY, accZ: 三轴加速度
            
        Returns:
            加速度幅值
        """
        return np.sqrt(accX**2 + accY**2 + accZ**2)
    
    def butter_bandpass_filter(self, data, lowcut=0.5, highcut=5.0, order=4):
        """
        带通滤波器，用于滤除噪声和基线漂移
        
        Args:
            data: 输入数据
            lowcut: 低截止频率(Hz)
            highcut: 高截止频率(Hz)
            order: 滤波器阶数
            
        Returns:
            滤波后的数据
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 防止频率超出范围
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            print(f"滤波器错误: {e}")
            return data
    
    def detect_peaks(self, data):
        """
        检测数据中的峰值
        
        Args:
            data: 输入数据数组
            
        Returns:
            峰值的索引数组
        """
        if len(data) < self.min_peak_distance * 2:
            return np.array([])
        
        # 计算动态阈值
        threshold = np.percentile(data, self.min_peak_height_percentile)
        
        # 使用scipy的find_peaks函数
        peaks, properties = signal.find_peaks(
            data,
            height=threshold,
            distance=self.min_peak_distance,
            prominence=np.std(data) * 0.5  # 峰值突出度
        )
        
        return peaks
    
    def calculate_heart_rate(self, peak_indices, time_array):
        """
        根据峰值计算心率
        
        Args:
            peak_indices: 峰值索引数组
            time_array: 时间数组
            
        Returns:
            心率(BPM - 每分钟心跳数)
        """
        if len(peak_indices) < 2:
            return None
        
        # 计算相邻峰值之间的时间间隔
        peak_times = time_array[peak_indices]
        intervals = np.diff(peak_times)
        
        # 过滤异常间隔(太短或太长)
        # 假设心率范围: 30-200 BPM
        min_interval = 60.0 / 200  # 200 BPM对应的最小间隔
        max_interval = 60.0 / 30   # 30 BPM对应的最大间隔
        
        valid_intervals = intervals[
            (intervals >= min_interval) & (intervals <= max_interval)
        ]
        
        if len(valid_intervals) == 0:
            return None
        
        # 计算平均间隔
        avg_interval = np.median(valid_intervals)  # 使用中位数更稳健
        
        # 转换为BPM
        heart_rate = 60.0 / avg_interval
        
        return heart_rate
    
    def process_data(self):
        """
        处理数据并检测心率
        """
        print("心率检测线程启动...")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"分析窗口: {self.window_size} 秒")
        print(f"缓冲区大小: {self.buffer_size} 个数据点")
        print("-" * 60)
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=0.1)
                
                # 计算加速度幅值
                magnitude = self.calculate_magnitude(
                    data['AccX'],
                    data['AccY'],
                    data['AccZ']
                )
                
                # 添加到缓冲区
                self.acc_buffer.append(magnitude)
                self.time_buffer.append(self.data_count / self.sample_rate)
                self.data_count += 1
                
                # 当缓冲区填满且到达更新时间时，进行心率检测
                current_time = time.time()
                if (len(self.acc_buffer) >= self.buffer_size and 
                    current_time - self.last_update_time >= self.update_interval):
                    
                    # 转换为numpy数组
                    acc_data = np.array(self.acc_buffer)
                    time_data = np.array(self.time_buffer)
                    
                    # 应用带通滤波器
                    filtered_data = self.butter_bandpass_filter(acc_data)
                    
                    # 检测峰值
                    peaks = self.detect_peaks(filtered_data)
                    
                    # 计算心率
                    if len(peaks) >= 2:
                        heart_rate = self.calculate_heart_rate(peaks, time_data)
                        
                        if heart_rate is not None:
                            self.heart_rate_history.append(heart_rate)
                            
                            # 计算平滑后的心率(使用最近几次测量的平均值)
                            if len(self.heart_rate_history) >= 3:
                                smooth_hr = np.mean(list(self.heart_rate_history)[-3:])
                            else:
                                smooth_hr = heart_rate
                            
                            # 打印心率信息
                            print(f"\n[{data['datetime']}] 心率检测:")
                            print(f"  检测到峰值数量: {len(peaks)}")
                            print(f"  瞬时心率: {heart_rate:.1f} BPM")
                            print(f"  平滑心率: {smooth_hr:.1f} BPM")
                            print(f"  数据点数: {self.data_count}")
                            
                            # 显示心率趋势
                            if len(self.heart_rate_history) >= 5:
                                recent_hrs = list(self.heart_rate_history)[-5:]
                                hr_std = np.std(recent_hrs)
                                print(f"  心率稳定性: {hr_std:.2f} (标准差)")
                                
                                if hr_std < 2:
                                    print(f"  状态: 稳定 ✓")
                                elif hr_std < 5:
                                    print(f"  状态: 轻微波动 ~")
                                else:
                                    print(f"  状态: 波动较大 !")
                            
                            print("-" * 60)
                    
                    self.last_update_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"心率检测错误: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n心率检测线程结束")
        self.print_summary()
    
    def print_summary(self):
        """
        打印心率检测总结
        """
        print("\n" + "=" * 60)
        print("心率检测总结")
        print("=" * 60)
        print(f"总数据点数: {self.data_count}")
        
        if len(self.heart_rate_history) > 0:
            hr_array = np.array(self.heart_rate_history)
            print(f"检测次数: {len(self.heart_rate_history)}")
            print(f"平均心率: {np.mean(hr_array):.1f} BPM")
            print(f"最高心率: {np.max(hr_array):.1f} BPM")
            print(f"最低心率: {np.min(hr_array):.1f} BPM")
            print(f"心率标准差: {np.std(hr_array):.2f}")
            
            # 显示心率分布
            print(f"\n心率分布:")
            bins = [0, 60, 80, 100, 120, 200]
            labels = ["<60", "60-80", "80-100", "100-120", ">120"]
            hist, _ = np.histogram(hr_array, bins=bins)
            for label, count in zip(labels, hist):
                if count > 0:
                    print(f"  {label} BPM: {count} 次")
        else:
            print("未检测到有效心率")
        
        print("=" * 60)
    
    def start(self):
        """
        启动心率检测线程
        """
        thread = Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
        return thread


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    test_queue = queue.Queue()
    stop_event = Event()
    
    # 生成模拟心率数据(60 BPM)
    sample_rate = 200
    duration = 20  # 20秒
    t = np.linspace(0, duration, sample_rate * duration)
    
    # 模拟心率信号(60 BPM = 1 Hz)
    heart_rate_freq = 1.0  # Hz
    heart_signal = np.sin(2 * np.pi * heart_rate_freq * t)
    
    # 添加一些噪声和基线
    noise = np.random.randn(len(t)) * 0.1
    baseline = 1.0
    acc_magnitude = baseline + heart_signal + noise
    
    # 模拟三轴加速度
    for i in range(len(t)):
        mag = acc_magnitude[i]
        # 简单分解到三轴
        accX = mag * 0.577
        accY = mag * 0.577
        accZ = mag * 0.577
        
        data = {
            'datetime': f"{int(i/sample_rate):02d}:{(i%sample_rate)*5:03d}",
            'AccX': accX,
            'AccY': accY,
            'AccZ': accZ
        }
        test_queue.put(data)
    
    # 启动心率检测
    detector = HeartRateDetector(test_queue, stop_event, sample_rate=sample_rate, window_size=10)
    detector_thread = detector.start()
    
    # 等待处理完成
    detector_thread.join(timeout=30)
    stop_event.set()
    
    print("\n测试完成")
