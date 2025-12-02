import time
from collections import deque

import numpy as np


class BreathingWaveDetector:
    """呼吸波形检测器- 峰顶是吸气"""

    def __init__(self,
                 buffer_size=1000,
                 smooth_window=85,
                 slope_window=25,
                 peak_threshold=100,
                 valley_threshold=30,
                 optimization_enabled=False,
                 history_size=5, # 表示用于判断新检测到的峰值/谷值是否有效时，参考的历史点数量（即最近5个历史峰/谷点）。
                 value_diff_peak_threshold=0.3, # unused
                 index_diff_peak_threshold=15, # unused
                 value_diff_valley_threshold=0.3,
                 index_diff_valley_threshold=15,
                 apnea_detection_enabled=True, # 是否开启呼吸暂停监测
                 apnea_window_size=800, # 统计最新apnea_window_size帧数据的最大值和最小值
                 apnea_threshold = 3.0,
                 rapid_filter_enabled = True,  # 启用快速波形过滤
                 rapid_time_threshold = 0.6,  # 快速波形的时间阈值（秒）
                 cooling_period = 3.0):  # 冷却期（秒）

        # 缓冲区和窗口参数
        self.buffer_size = buffer_size # 缓冲区大小
        self.smooth_window = smooth_window # 平滑窗口大小
        self.slope_window = slope_window # 斜率窗口大小
        self.peak_threshold = peak_threshold # 通过斜率判断出origin数据的峰值后，在改点的周围peak_threshold点寻找最大值【峰顶】
        self.valley_threshold = valley_threshold # 通过斜率判断出origin数据的峰值后，在改点的周围peak_threshold点寻找最大值【峰谷】

        # 优化参数
        self.optimization_enabled = optimization_enabled  # 是否开启优化
        self.history_size = history_size  # 历史记录大小
        self.value_diff_peak_threshold = value_diff_peak_threshold  # 值差异阈值
        self.index_diff_peak_threshold = index_diff_peak_threshold  # 索引差异阈值
        self.value_diff_valley_threshold = value_diff_valley_threshold  # 索引差异阈值
        self.index_diff_valley_threshold = index_diff_valley_threshold  # 索引差异阈值

        # 呼吸暂停检测参数
        self.apnea_detection_enabled = apnea_detection_enabled  # 是否监测呼吸暂停
        self.apnea_window_size = apnea_window_size  # 呼吸暂停检测窗口大小
        self.apnea_threshold = apnea_threshold  # 呼吸暂停阈值，最大值与最小值差小于此值视为呼吸暂停
        self.is_apnea_value = False

        # 历史记录队列
        self.peak_indexes_history = deque(maxlen=history_size)
        self.peak_values_history = deque(maxlen=history_size)
        self.valley_indexes_history = deque(maxlen=history_size)
        self.valley_values_history = deque(maxlen=history_size)

        # 数据存储
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.index = deque(maxlen=buffer_size)
        self.smoothed_data = deque(maxlen=buffer_size)
        self.slopes = deque(maxlen=buffer_size)

        self.index_num = 0

        # 存储峰值和谷值点
        self.peak_points = deque(maxlen=500)  # 是最终保留、对外展示的峰值点
        self.valley_points = deque(maxlen=500)
        self.peak_points_real = deque(maxlen=500)  #  用于存储“原始检测到的峰值点”，即在检测过程中未经过某些过滤（如快速波形过滤）前的峰值点。它主要用于内部逻辑，比如判断是否出现快速波形时，临时存储和回溯用。
        self.valley_points_real = deque(maxlen=500)
        self.peak_data = []  # 存储峰顶四元组数据
        self.valley_data = []  # 存储峰谷四元组数据

        # 就是短时间内有很多交替的吸气呼气点，会被过滤
        self.rapid_filter_enabled = rapid_filter_enabled
        self.rapid_time_threshold = rapid_time_threshold
        self.cooling_period = cooling_period
        self.cooling_end_time = 0  # 冷却期结束时间戳

        # 存储检测到的点（时间戳和类型）
        self.detection_history = deque(maxlen=10)  # 存储最近10个检测点

        # 呼吸周期与频率相关变量
        self.breath_intervals = deque(maxlen=20)  # 两个谷值之间的时间差（呼吸周期）
        self.breath_frequency_list = deque(maxlen=20)  # 呼吸频率队列
        self.inhale_ratio_list = deque(maxlen=20)      # 吸气占比队列
        self.exhale_ratio_list = deque(maxlen=20)      # 呼气占比队列

    # 假如原始数据有2000个点，我们的smooth_window是6，那么平滑后应该有1994个点，原始index为400，那么新进来的点取平滑后，对应的应该是原始的index - smooth_window // 2 = 396.
    # 意思就是新来一个点，它是原始数据，它在原始数据的index为400，我们计算平滑后，这个平滑其实是原始数据396的点。相当于延迟了4帧。只有再来4个点，计算的才是刚才这个index为400的平滑点。
    def moving_average(self, data, window_size):
        """计算移动平均"""
        if len(data) < window_size:
            return np.mean(data)
        return np.mean(list(data)[-window_size:])

    # 假如原始数据为[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],window_size = 2, 原来的数据长度为10，求斜率后长度为10 - 2*window = 6.
    # 因为window = 2，以3为中心，从[1, 5]计算（5-4）(4-3) (3-2) (2-1)的平均值为点3的斜率。
    # 因此会延迟2帧。新进来一个点5，计算其前面2 * window + 1个点得到的斜率，就说点3的斜率。
    # 实验结果是：2000个点，window_size = 5, 计算出来的斜率有1990个点。
    def calculate_slope(self, data, window_size):
        """计算斜率"""
        if len(data) < window_size * 2 + 1:
            return 0

        recent_data = list(data)[-(window_size * 2 + 1):]
        center_idx = window_size

        left_avg = np.mean(recent_data[:center_idx])
        right_avg = np.mean(recent_data[center_idx + 1:])

        slope = (right_avg - left_avg) / window_size
        return slope

    def format_timestamp(self, timestamp):
        """将时间戳格式化为可读的日期时间字符串"""
        try:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        except (ValueError, OSError):
            # 如果时间戳格式不正确，直接返回原值
            return str(timestamp)

    def is_valid_detection(self, detection_type, index_value, data_value):
        """验证检测到的峰值或谷值是否有效"""
        if not self.optimization_enabled:
            return True
        if detection_type == 'peak':
            return True

        history_indexes = self.peak_indexes_history if detection_type == 'peak' else self.valley_indexes_history
        history_values = self.peak_values_history if detection_type == 'peak' else self.valley_values_history

        # 如果历史记录为空，认为是有效的
        if not history_indexes or not history_values:
            return True

        # 计算与历史记录的加权平均差异
        # 最近的点权重更大
        weights = np.linspace(0.5, 1.0, len(history_indexes))
        weights = weights / np.sum(weights)

        # 计算索引的加权平均差异
        avg_index = np.sum([idx * w for idx, w in zip(history_indexes, weights)])
        index_diff = abs(index_value - avg_index)

        # 计算值的加权平均差异
        avg_value = np.sum([val * w for val, w in zip(history_values, weights)])
        value_diff = abs(data_value - avg_value) / (avg_value if avg_value != 0 else 1)

        return value_diff <= self.value_diff_valley_threshold

        # return index_diff <= self.index_diff_threshold and value_diff <= self.value_diff_threshold

    def is_apnea(self):
        """检测是否可能存在呼吸暂停"""
        if not self.apnea_detection_enabled or len(self.buffer) < self.apnea_window_size:
            return False

        # 获取最近的数据
        recent_data = list(self.buffer)[-self.apnea_window_size:]
        data_range = max(recent_data) - min(recent_data)
        
        # 如果数据波动很小，可能是呼吸暂停
        if data_range < self.apnea_threshold:
            return True
        return False

    def check_rapid_wave(self, current_type, current_timestamp, current_value):
        """
        检查当前点加上之前的点是否构成快速波形
        :param current_type: 当前检测点类型 ('peak' 或 'valley')
        :param current_timestamp: 当前检测点时间戳
        :param current_value: 当前检测点值
        :return: 是否需要过滤
        """
        # 获取最近的两个峰值和两个谷值
        if (len(self.peak_points_real) < 3 or len(self.valley_points_real) < 3):
            return
        latest_peaks = list(self.peak_points_real)[-2:] if len(self.peak_points_real) >= 2 else list(self.peak_points_real)
        latest_valleys = list(self.valley_points_real)[-2:] if len(self.valley_points_real) >= 2 else list(self.valley_points_real)

        # 构建5个点的列表（包括当前点）
        points = latest_peaks + latest_valleys
        if current_type == 'peak':
            points.append((current_timestamp, current_value))
        elif current_type == 'valley':
            points.append((current_timestamp, current_value))

        # 如果点数少于5，不需要过滤
        if len(points) < 5:
            return False

        # 按时间戳排序
        points.sort(key=lambda x: x[0])

        # 只取最近的5个点
        points = points[-5:]

        # 检查时间间隔是否都小于阈值
        time_valid = True
        for i in range(1, len(points)):
            if points[i][0] - points[i - 1][0] > self.rapid_time_threshold:
                time_valid = False
                break

        # 检查是否峰谷交替出现（通过检查值的升降交替）
        # alternating = True
        # for i in range(2, len(points)):
        #     # 如果三个点的变化模式相同（都是增加或都是减少），则不是交替的
        #     diff1 = points[i - 1][1] - points[i - 2][1]
        #     diff2 = points[i][1] - points[i - 1][1]
        #     if (diff1 > 0 and diff2 > 0) or (diff1 < 0 and diff2 < 0):
        #         alternating = False
        #         break

        # 如果满足过滤条件，从峰值和谷值队列中删除相应的点
        # if time_valid and alternating:
        if time_valid:
            # 直接删除最新的两个峰值点和两个谷值点
            for _ in range(min(2, len(self.peak_points))):
                self.peak_points.pop()
                self.peak_points_real.pop()

            for _ in range(min(2, len(self.valley_points))):
                self.valley_points.pop()
                self.valley_points_real.pop()

            return True

        return False

    def detect_peaks_valleys(self):
        """检测峰顶和峰谷"""
        if len(self.slopes) < 2 or len(self.smoothed_data) < self.slope_window:
            return None

        # 检测是否可能存在呼吸暂停
        if self.is_apnea():
            self.is_apnea_value = True
            return None

        self.is_apnea_value = False

        # 检查是否处于冷却期
        current_time = time.time()
        if current_time < self.cooling_end_time:
            return None

        slopes_list = list(self.slopes)
        smoothed_list = list(self.smoothed_data)
        buffer_list = list(self.buffer)
        timestamps_list = list(self.timestamps)

        check_index = len(slopes_list) - 2
        if check_index <= 0:
            return None

        current_slope = slopes_list[check_index]
        next_slope = slopes_list[check_index + 1]

        smooth_index = len(smoothed_list) - self.slope_window
        original_index = len(buffer_list) - (self.smooth_window // 2) - self.slope_window

        if original_index < 0 or smooth_index < 0 or original_index >= len(buffer_list):
            return None

        original_value = buffer_list[original_index]
        original_timestamp = timestamps_list[original_index] if original_index < len(timestamps_list) else 0
        smoothed_value = smoothed_list[smooth_index] if smooth_index < len(smoothed_list) else 0

        # 检测峰顶：斜率从正变负
        if current_slope > 0 and next_slope < 0:
            start_idx = max(0, original_index - self.peak_threshold)
            end_idx = min(len(buffer_list), original_index + self.peak_threshold + 1)
            local_max_idx = start_idx + np.argmax(buffer_list[start_idx:end_idx])
            local_max_value = buffer_list[local_max_idx]
            local_max_timestamp = timestamps_list[local_max_idx]

            # 验证检测到的峰顶是否有效
            if not self.is_valid_detection('peak', self.index[original_index], local_max_value):
                print(
                    f"\033[33m⚠️ 峰顶被优化算法过滤\033[0m: 索引={self.index[original_index]}, 值={local_max_value:.3f}")
                return None

            # 检查是否需要过滤快速波形
            if self.check_rapid_wave('peak', local_max_timestamp, local_max_value):
                print(f"\033[31m⚠️ 检测到快速波形，进入{self.cooling_period}秒冷却期\033[0m")
                self.cooling_end_time = current_time + self.cooling_period
                return None

            self.peak_points.append((local_max_timestamp, local_max_value))
            self.peak_points_real.append((local_max_timestamp, local_max_value))
            return {'type': 'peak', 'index': original_index, 'value': original_value, 'timestamp': original_timestamp}

        # 检测峰谷：斜率从负变正
        elif current_slope < 0 and next_slope > 0:
            start_idx = max(0, original_index - self.valley_threshold)
            end_idx = min(len(buffer_list), original_index + self.valley_threshold + 1)
            local_min_idx = start_idx + np.argmin(buffer_list[start_idx:end_idx])
            local_min_value = buffer_list[local_min_idx]
            local_min_timestamp = timestamps_list[local_min_idx]

            # 验证检测到的峰谷是否有效
            if not self.is_valid_detection('valley', self.index[original_index], local_min_value):
                print(
                    f"\033[33m⚠️ 峰谷被优化算法过滤\033[0m: 索引={self.index[original_index]}, 值={local_min_value:.3f}")
                return None

            # 检查是否需要过滤快速波形
            if self.check_rapid_wave('valley', local_min_timestamp, local_min_value):
                print(f"\033[31m⚠️ 检测到快速波形，进入{self.cooling_period}秒冷却期\033[0m")
                self.cooling_end_time = current_time + self.cooling_period
                return None

            self.valley_points.append((local_min_timestamp, local_min_value))
            self.valley_points_real.append((local_min_timestamp, local_min_value))
           # ----------- 用谷→峰→谷计算周期和比例 -----------
            if len(self.valley_points_real) >= 2 and len(self.peak_points_real) >= 1:
                V2_time, _ = self.valley_points_real[-1]   # 当前谷
                V1_time, _ = self.valley_points_real[-2]   # 上一个谷

                # 找最近一个峰，且在 V1 和 V2 之间
                peaks_between = [p for p in self.peak_points_real if V1_time < p[0] < V2_time]
                if peaks_between:
                    P_time, _ = peaks_between[-1]
                    cycle = V2_time - V1_time
                    inhale_dur = P_time - V1_time
                    exhale_dur = V2_time - P_time

                    if cycle > 0 and inhale_dur > 0 and exhale_dur > 0:
                        freq = 60.0 / cycle
                        inhale_ratio = inhale_dur / cycle
                        exhale_ratio = exhale_dur / cycle

                        self.breath_intervals.append(cycle)
                        self.breath_frequency_list.append(freq)
                        self.inhale_ratio_list.append(inhale_ratio)
                        self.exhale_ratio_list.append(exhale_ratio)
            # ---------------------------------------------------
            return {'type': 'valley', 'index': original_index, 'value': original_value, 'timestamp': original_timestamp}

        return None

    def detect(self, timestamp, value):
        """主检测函数"""
        self.buffer.append(value)
        self.timestamps.append(timestamp)
        self.index.append(self.index_num)
        self.index_num += 1
        # print(f"是否呼吸暂停: {self.is_apnea_value}")

        if len(self.buffer) >= self.smooth_window:
            smoothed_value = self.moving_average(self.buffer, self.smooth_window)
            self.smoothed_data.append(smoothed_value)

            if len(self.smoothed_data) >= self.slope_window * 2 + 1:
                slope = self.calculate_slope(self.smoothed_data, self.slope_window)
                self.slopes.append(slope)

                return self.detect_peaks_valleys()
        return None

    def reset(self):
        """重置检测器"""
        self.buffer.clear()
        self.timestamps.clear()
        self.index.clear()
        self.smoothed_data.clear()
        self.slopes.clear()
        self.peak_points.clear()
        self.valley_points.clear()
        self.peak_data.clear()
        self.valley_data.clear()
        self.peak_indexes_history.clear()
        self.peak_values_history.clear()
        self.valley_indexes_history.clear()
        self.valley_values_history.clear()
        self.breath_intervals.clear()
        self.breath_frequency_list.clear()
        self.inhale_ratio_list.clear()
        self.exhale_ratio_list.clear()

    def get_statistics(self):
        """获取统计信息"""
        # 最近两个的平均值
        breath_interval = np.mean(list(self.breath_intervals)[-2:]) if len(self.breath_intervals) > 0 else 0.0
        breath_freq = np.mean(list(self.breath_frequency_list)[-2:]) if len(self.breath_frequency_list) > 0 else 0.0
        inhale_ratio = np.mean(list(self.inhale_ratio_list)[-2:]) if len(self.inhale_ratio_list) > 0 else 0.0
        exhale_ratio = np.mean(list(self.exhale_ratio_list)[-2:]) if len(self.exhale_ratio_list) > 0 else 0.0

        return {
            # 'peak_count': len(self.peak_points),
            # 'valley_count': len(self.valley_points),
            # 'total_detections': len(self.peak_points) + len(self.valley_points),
            # 'optimization_enabled': self.optimization_enabled,
            # 'apnea_detection_enabled': self.apnea_detection_enabled,
            'breath_interval': breath_interval,
            'breath_frequency': breath_freq,
            'inhale_ratio': inhale_ratio,
            'exhale_ratio': exhale_ratio
        }

    def set_apnea_detection(self, enabled):
        """设置是否启用呼吸暂停检测"""
        self.apnea_detection_enabled = enabled
        print(f"呼吸暂停检测已{'启用' if enabled else '禁用'}")


from collections import deque
import numpy as np

class AirZeroCrossingDetector:
    """检测空气数据的过零点，并计算呼吸参数"""
    def __init__(self):
        self.peak_points = []    # 从负到正的过零点
        self.valley_points = []  # 从正到负的过零点
        self.prev_value = None   # 上一个值
        self.prev_timestamp = None  # 上一个时间戳

        # 呼吸参数队列（存最近20次）
        self.breath_intervals = deque(maxlen=20)       # 呼吸周期 (s)
        self.breath_frequency_list = deque(maxlen=20)  # 呼吸频率 (次/分)
        self.inhale_ratio_list = deque(maxlen=20)      # 吸气占比
        self.exhale_ratio_list = deque(maxlen=20)      # 呼气占比

    def detect(self, timestamp, value):
        """检测过零点，并在 valley→peak→valley 时更新呼吸参数"""
        if self.prev_value is None:
            self.prev_value = value
            self.prev_timestamp = timestamp
            return

        # 从负到正 → 峰点
        if self.prev_value <= 0 and value > 0:
            delta_v = value - self.prev_value
            if abs(delta_v) > 1e-7:
                factor = -self.prev_value / delta_v
                zero_crossing_time = self.prev_timestamp + factor * (timestamp - self.prev_timestamp)
                self.peak_points.append((zero_crossing_time, 0))

        # 从正到负 → 谷点
        elif self.prev_value >= 0 and value < 0:
            delta_v = value - self.prev_value
            if abs(delta_v) > 1e-7:
                factor = -self.prev_value / delta_v
                zero_crossing_time = self.prev_timestamp + factor * (timestamp - self.prev_timestamp)
                self.valley_points.append((zero_crossing_time, 0))

                # 如果已有 >=2 个 valley 且 >=1 个 peak → 计算呼吸参数
                if len(self.valley_points) >= 2 and len(self.peak_points) >= 1:
                    V1_time = self.valley_points[-2][0]
                    V2_time = self.valley_points[-1][0]

                    # 找最近的一个峰值，且在 (V1, V2) 之间
                    recent_peak = max([p for p in self.peak_points if V1_time < p[0] < V2_time], 
                                      key=lambda x: x[0], default=None)

                    if recent_peak:
                        P_time = recent_peak[0]
                        cycle = V2_time - V1_time
                        inhale_dur = P_time - V1_time
                        exhale_dur = V2_time - P_time

                        if cycle > 0 and inhale_dur > 0 and exhale_dur > 0:
                            freq = 60.0 / cycle
                            inhale_ratio = inhale_dur / cycle
                            exhale_ratio = exhale_dur / cycle

                            # 存入队列
                            self.breath_intervals.append(cycle)
                            self.breath_frequency_list.append(freq)
                            self.inhale_ratio_list.append(inhale_ratio)
                            self.exhale_ratio_list.append(exhale_ratio)

        # 更新状态
        self.prev_value = value
        self.prev_timestamp = timestamp

    def get_statistics(self):
        """获取统计信息（取最近2个的平均值）"""
        breath_interval = np.mean(list(self.breath_intervals)[-2:]) if self.breath_intervals else 0.0
        breath_freq = np.mean(list(self.breath_frequency_list)[-2:]) if self.breath_frequency_list else 0.0
        inhale_ratio = np.mean(list(self.inhale_ratio_list)[-2:]) if self.inhale_ratio_list else 0.0
        exhale_ratio = np.mean(list(self.exhale_ratio_list)[-2:]) if self.exhale_ratio_list else 0.0

        return {
            "breath_interval": breath_interval,
            "breath_frequency": breath_freq,
            "inhale_ratio": inhale_ratio,
            "exhale_ratio": exhale_ratio
        }

    def reset(self):
        """重置检测器状态"""
        self.peak_points.clear()
        self.valley_points.clear()
        self.prev_value = None
        self.prev_timestamp = None

        # 清空队列
        self.breath_intervals.clear()
        self.breath_frequency_list.clear()
        self.inhale_ratio_list.clear()
        self.exhale_ratio_list.clear()
