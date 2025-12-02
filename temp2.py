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
    def __init__(self, data_queue, stop_event, sample_rate=100, window_size=10):
        """
        初始化心率检测器
        
        Args:
            data_queue: 数据队列
            stop_event: 停止信号
            sample_rate: 采样率(Hz)，默认100
            window_size: 计算心率的时间窗口(秒)，默认10秒
        """
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.last_print_time = time.time()
        # self.plot_queue=plot_queue
        # 缓存数据用于峰值检测
        self.buffer_size = sample_rate * window_size
        self.acc_buffer = []
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # 心率历史记录
        self.heart_rate_history = deque(maxlen=100)
        
        # 新增：心率计算相关参数（与DataRecorder兼容）
        self.hr_buffer = []           # 保存未处理的原始数据
        self.rt_hr = 0                # 实时心率
        self.hr_status = 'init'       # 心率状态
        self.hr_values = []           # 成功计算的心率值
        self.last_hr_update_count = 0  # 上次HR计算时的数据计数
        self.last_hr_output_count = 0  # 上次HR输出时的数据计数
        self.total_data_count = 0      # 总数据计数（用于模拟时间）
        
        # 统计信息
        self.data_count = 0
        self.last_update_count = 0    # 上次更新时的数据计数
        self.update_interval_count = 2 * sample_rate  # 每2秒更新一次心率（转换为数据点计数）
    
    def differentiator_filter_double(self, data, h):
        """
        二阶差分近似微分（与DataRecorder中的算法一致）
        """
        length = data.shape[0] - 6
        data_ans = data * 0.0
        res_data = data[3:length+3] * 4.0 + \
                   (data[4:length+4] + data[2:length+2]) - \
                   2.0 * (data[5:length+5] + data[1:length+1]) - \
                   (data[6:length+6] + data[0:length])
        res_data = res_data / 16.0 / h / h
        data_ans[3:-3] = res_data
        return data_ans
    
    def push_hr_samples(self, samples):
        """
        推入未处理的样本用于心率计算（与DataRecorder中的方法一致）
        """
        if not samples:
            return
        
        # 过滤有效的数值 - 安全版本
        clean = []
        for v in samples:
            try:
                val = float(v)
                if np.isfinite(val) and abs(val) < 100000:  # 避免极端值
                    clean.append(val)
            except (ValueError, TypeError):
                continue
        if not clean:
            return
        
        self.hr_buffer.extend(clean)
        if len(self.hr_buffer) > 2000:
            self.hr_buffer = self.hr_buffer[-2000:]
    
    def compute_hr(self):
        """
        使用自相关算法计算心率（基于数据量模拟时间检查）
        """
        # 计算模拟时间（基于数据量）
        simulated_time = self.total_data_count / self.sample_rate
        
        # 初始数据收集检查（前5秒）
        if simulated_time < 5:
            self.rt_hr = 0
            self.hr_status = 'collecting'
            return self.rt_hr

        # 5秒计算间隔检查（基于数据量）
        if (self.total_data_count - self.last_hr_update_count) < (5 * self.sample_rate):
            # print(f"[HR计算] 数据不足，当前数据量: {self.total_data_count}, last_hr_update_count: {self.last_hr_update_count}")
            return self.rt_hr

        self.last_hr_update_count = self.total_data_count
        
        if len(self.hr_buffer) < self.sample_rate * 5:
            print(f"[HR计算failed] 数据不足，当前数据量: {self.total_data_count}, last_hr_update_count: {self.last_hr_update_count}")
            return self.rt_hr

        # 心率计算(每5秒执行)
        current_hr = 0
        success = False
        
        try:
            # 使用自相关算法计算心率
            proc = HeartRateProcessor(self.hr_buffer[-int(self.sample_rate * 5):], sample_rate=self.sample_rate)
            proc.preprocess()
            proc.calculate_heart_rate(threshold=0.02, sampling_interval=1.0 / self.sample_rate)
            
            if proc.status == 'succeeded' and 35 <= proc.current_hr <= 200:
                current_hr = int(round(proc.current_hr))
                success = True
        except Exception as e:
            print(f"自相关心率计算错误: {e}")
        
        # 只有在成功计算时才添加到hr_values
        if success:
            self.hr_values.append(current_hr)
            print(f"[HR计算] 成功计算心率: {current_hr} BPM")
        else:
            print(f"[HR计算] 心率计算失败，放弃本次结果, {proc.current_hr}")
            return self.rt_hr

        # 保持最近10次记录
        if len(self.hr_values) > 10:
            self.hr_values = self.hr_values[-10:]

        # 前30秒不输出
        # if simulated_time < 30:
        #     self.rt_hr = 0
        #     self.hr_status = 'collecting'
        #     return self.rt_hr

        # 至少需要3个值
        if len(self.hr_values) < 3:
            self.rt_hr = 0
            self.hr_status = 'collecting'
            return self.rt_hr

        # 5秒输出间隔检查（基于数据量）
        if (self.total_data_count - self.last_hr_output_count) < (5 * self.sample_rate):
            return self.rt_hr

        self.last_hr_output_count = self.total_data_count

        # 加权平滑计算(每10秒更新输出)
        avg = np.mean(self.hr_values)
        # valid = [v for v in self.hr_values if abs(v - avg) / avg < 0.35]
        valid = [v for v in self.hr_values ]
        if valid:
            # 加权平滑:越近的数据权重越高
            n = len(valid)
            weights = np.arange(1, n + 1, dtype=float)
            weights = weights / np.sum(weights)
            weighted_hr = np.sum(np.array(valid) * weights)
            self.rt_hr = int(round(weighted_hr))
            self.hr_status = 'succeeded'
        else:
            self.rt_hr = int(round(np.median(self.hr_values)))
            self.hr_status = 'fallback'

        # if (len(self.hr_values) > 0):
        #     return self.hr_values[-1]
        return self.rt_hr


    def calculate_magnitude(self, accX, accY, accZ):
        """
        计算加速度的幅值
        """
        return np.sqrt(accX**2 + accY**2 + accZ**2)
    
    def butter_bandpass_filter(self, data, lowcut=0.5, highcut=5.0, order=4):
        """
        带通滤波器，用于滤除噪声和基线漂移
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            print(f"滤波器错误: {e}")
            return data
    
    def process_data(self):
        """
        处理数据并检测心率（使用新的相容性算法）
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
                self.total_data_count += 1  # 增加总数据计数
                
                # 当缓冲区填满且到达更新时间时，进行心率检测
                if (len(self.acc_buffer) > 10):
    
                    self.acc_buffer.pop(0)
                    
                    # 转换为numpy数组
                    acc_data = np.array(self.acc_buffer)
                    time_data = np.array(self.time_buffer)
                    
                    # 应用带通滤波器
                    # filtered_data = self.butter_bandpass_filter(acc_data)
                    filtered_data = acc_data
                    
                    # 使用二阶微分滤波（与DataRecorder一致）
                    try:
                        if len(filtered_data) > 6:
                            differentiated_data = self.differentiator_filter_double(filtered_data, 1/self.sample_rate)
                            valid_data = differentiated_data[3:-3]
                            
                            if len(valid_data) > 0:
                                # 将处理后的数据推入心率计算缓冲区
                                
                                self.push_hr_samples(valid_data[-1:])
                                # print("推入心率计算样本:", valid_data[-1:])
                                
                                # 计算心率（使用自相关算法）
                                heart_rate = self.compute_hr()
                                current_time = time.time()
                                if heart_rate is not None and heart_rate > 0 and (current_time - self.last_print_time) >= 1:
                                    self.heart_rate_history.append(heart_rate)
                                    
                                    # 打印心率信息
                                    print(f"\n[{data['datetime']}] 心率检测:")
                                    print(f"  使用自相关算法计算心率")
                                    print(f"  实时心率: {heart_rate} BPM")
                                    # print(f"  心率状态: {self.hr_status}")
                                    # print(f"  有效测量次数: {len(self.hr_values)}")
                                    # print(f"  数据点数: {self.data_count}")
                                    # print(f"  模拟时间: {self.total_data_count/self.sample_rate:.1f}秒")
                                    
                                    # 显示心率趋势
                                    # if len(self.heart_rate_history) >= 5:
                                    #     recent_hrs = list(self.heart_rate_history)[-5:]
                                    #     hr_std = np.std(recent_hrs)
                                    #     print(f"  心率稳定性: {hr_std:.2f} (标准差)")
                                        
                                    #     if hr_std < 2:
                                    #         print(f"  状态: 稳定 ✓")
                                    #     elif hr_std < 5:
                                    #         print(f"  状态: 轻微波动 ~")
                                    #     else:
                                    #         print(f"  状态: 波动较大 !")
                                    
                                    print("-" * 60)
                                    self.last_print_time = current_time
                    except Exception as e:
                        print(f"心率计算错误: {e}")
                
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
        print(f"心率计算算法: 自相关算法")
        print(f"模拟总时间: {self.total_data_count/self.sample_rate:.1f}秒")
        
        if len(self.hr_values) > 0:
            hr_array = np.array(self.hr_values)
            print(f"成功计算次数: {len(self.hr_values)}")
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
            print("未成功计算有效心率")
        
        print("=" * 60)
    
    def start(self):
        """
        启动心率检测线程
        """
        thread = Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
        return thread


# 从文档2中复制HeartRateProcessor类
class HeartRateProcessor:
    """
    处理心率信号：预处理、自相关与心率估计
    """
    def __init__(self, pos_node, sample_rate):
        self.pos_node = list(pos_node)
        self.pre_node = []
        self.norm_node = []
        self.autocorrelation_result = []
        self.max_peak = float('-inf')
        self.peak_index = None
        self.current_hr = 60
        self.last_hr = 60
        self.status = "failed"
        self.sample_rate = sample_rate

    def preprocess(self):
        if not any(self.pos_node):
            print("预处理错误: 输入数据为空")
            return
        # 限长到500点（对应100Hz下的5秒数据）
        if len(self.pos_node) > 500:
            self.pos_node = self.pos_node[-500:]
        # 剔除异常值
        self.pre_node = [0 if abs(float(x)) > 1500 else float(x) for x in self.pos_node]
        if not any(self.pre_node):
            print("预处理错误: 预处理后数据为空")
            return
        max_val = max(abs(x) for x in self.pre_node)
        if max_val == 0:
            return
        self.norm_node = [x / max_val for x in self.pre_node]


    def autocorrelation(self):
        signal = self.norm_node
        n = len(signal)
        if n == 0:
            self.autocorrelation_result = []
            return
        mean = np.mean(signal)
        var = np.var(signal)
        if var == 0:
            self.autocorrelation_result = []
            return
        autocorr = np.correlate(signal - mean, signal - mean, mode='full') / (var * n)
        self.autocorrelation_result = autocorr[n-1:]

    def find_peak(self):
        # 根据采样率动态计算搜索范围（心率范围: 50~133 BPM）
        # 公式: 延迟样本数 = 采样率 * 60 / 心率
        if not  np.any(self.autocorrelation_result):
            self.max_peak = float('-inf')
            self.peak_index = None
            return

        if self.sample_rate == 100:
            start, end = 45, 120
        elif self.sample_rate == 200:
            start, end = 90, 240
        else:
            # 动态计算：心率133 BPM -> start, 心率50 BPM -> end
            start = int(self.sample_rate * 60 / 133)
            end = int(self.sample_rate * 60 / 50)

        end = min(end, len(self.autocorrelation_result) - 1)
        if end <= start:
            self.max_peak = float('-inf')
            self.peak_index = None
            return
        sub_range = self.autocorrelation_result[start:end + 1]
        max_value = np.max(sub_range)
        max_index = int(np.argmax(sub_range)) + start
        self.max_peak = float(max_value)
        self.peak_index = max_index + 1

    def calculate_heart_rate(self, threshold, sampling_interval):
        # print("计算心率...")
        try:
            # print("self.norm_node",self.norm_node)
            
            if not any(self.norm_node):
                print("自相关计算错误 in calculate_heart_rate: 归一化数据为空")
                return
            self.autocorrelation()
            # print("self.autocorrelation_result",self.autocorrelation_result)
            if not np.any(self.autocorrelation_result):
                print("自相关计算错误 in calculate_heart_rate: 自相关结果为空")
                return
            self.find_peak()
        except Exception as e:
            print(f"自相关计算异常{e}")
            return
        if self.peak_index is not None:
            print(f"自相关峰值: {self.max_peak:.4f} at index {self.peak_index}")
        else:
            print("未找到自相关峰值")
        if self.autocorrelation_result[self.peak_index] > threshold:
            print("峰值超过阈值，心率计算成功")
        else:
            print(f"峰值:{self.autocorrelation_result[self.peak_index]}未超过阈值，心率计算失败")
        if self.peak_index is not None and self.autocorrelation_result[self.peak_index] > threshold:
            self.current_hr = 60 / (sampling_interval * (self.peak_index + 1))
            print(f"计算心率: {self.current_hr:.2f} BPM")
            self.last_hr = self.current_hr
            self.status = "succeeded"
        else:
            self.current_hr = -1
            self.status = "failed"


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