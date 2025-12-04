"""
呼吸检测器（无绘图版本）
Breath Detector without Plotting

仅负责计算呼吸率，不包含绘图功能
从 DataReader 的 data_queue_detect 读取数据，仅基于 AngX 计算呼吸率
"""

import threading
import queue
import time
from collections import deque
from typing import Deque, List, Tuple, Optional

import numpy as np
from datetime import datetime as dt


class BreathDetectorNoPlot:
    """
    呼吸检测器（无绘图版本）
    - 平滑：移动平均
    - 斜率：中心差分法计算斜率
    - 峰谷：斜率过零检测 + 局部搜索
    - 呼吸率：通过谷到谷的时间间隔计算
    """

    def __init__(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = 100,
        smooth_window: int = 10,
        slope_window: int = 10,
        peak_search_radius: int = 5,
        breath_rate_queue: Optional[queue.Queue] = None,
        breath_data_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.smooth_window = max(3, smooth_window)
        self.slope_window = max(2, slope_window)
        self.peak_search_radius = max(5, peak_search_radius)
        self.breath_rate_queue = breath_rate_queue
        self.breath_data_queue = breath_data_queue  # 用于推送原始数据和峰谷点

        # 数据缓存（原始 AngX、平滑、斜率、时间戳）
        self.angx: Deque[float] = deque(maxlen=2000)
        self.smoothed: Deque[float] = deque(maxlen=2000)
        self.slopes: Deque[float] = deque(maxlen=2000)
        self.ts: Deque[float] = deque(maxlen=2000)  # 以秒为单位的相对时间

        # 检测到的峰谷（索引、时间、值）
        self.peak_points: List[Tuple[int, float, float]] = []
        self.valley_points: List[Tuple[int, float, float]] = []

        # 呼吸率（通过 valley->valley 计算）
        self.breath_intervals: Deque[float] = deque(maxlen=10)
        self.latest_bpm: float = 0.0

        # 相对时间起点
        self._t0 = None
        self.current_datetime: str = ""
        # 样本计数
        self.data_count: int = 0

    # ============ 核心算法 ============
    def _moving_average(self, data: Deque[float], window_size: int) -> float:
        """移动平均"""
        if not data:
            return 0.0
        n = min(len(data), window_size)
        if n <= 1:
            return data[-1]
        return float(np.mean(list(data)[-n:]))

    def _calculate_slope_centered(self, data: Deque[float], window_size: int) -> float:
        """中心差分法计算斜率"""
        if len(data) < window_size * 2 + 1:
            return 0.0
        recent = list(data)[-(window_size * 2 + 1) :]
        center = window_size
        left_avg = float(np.mean(recent[:center]))
        right_avg = float(np.mean(recent[center + 1 :]))
        return (right_avg - left_avg) / float(window_size)

    def _detect_events(self):
        """检测峰谷事件"""
        if len(self.slopes) < 2 or len(self.smoothed) < (self.slope_window * 2 + 1):
            return
        
        s_prev = self.slopes[-2]
        s_curr = self.slopes[-1]
        idx_curr = len(self.smoothed) - 1

        angx_list = list(self.angx)
        ts_list = list(self.ts)

        origin_idx = idx_curr
        start = max(0, origin_idx - self.peak_search_radius)
        end = origin_idx + self.peak_search_radius + 1

        origin_indx = len(self.angx) - (self.smooth_window // 2) - self.slope_window

        if origin_indx < 0 or origin_indx >= len(self.angx):
            return

        # 峰：斜率正->负
        if s_prev > 0 and s_curr < 0:
            seg = angx_list[start:end]
            if seg:
                local_idx = int(np.argmax(seg)) + start
                base_start = max(0, self.data_count - len(self.angx))
                abs_idx = base_start + local_idx
                self.peak_points.append((abs_idx, ts_list[local_idx], angx_list[local_idx]))
                if len(self.peak_points) > 300:
                    self.peak_points = self.peak_points[-300:]

        # 谷：斜率负->正
        if s_prev < 0 and s_curr > 0:
            seg = angx_list[start:end]
            if seg:
                local_idx = int(np.argmin(seg)) + start
                base_start = max(0, self.data_count - len(self.angx))
                abs_idx = base_start + local_idx
                self.valley_points.append((abs_idx, ts_list[local_idx], angx_list[local_idx]))
                if len(self.valley_points) > 300:
                    self.valley_points = self.valley_points[-300:]
                
                # 用最近两个谷更新 BPM
                if len(self.valley_points) >= 2:
                    t1 = self.valley_points[-2][1]
                    t2 = self.valley_points[-1][1]
                    dt = max(1e-6, t2 - t1)
                    bpm = 60.0 / dt
                    self.breath_intervals.append(dt)
                    # 取最近2次平均
                    vals = list(self.breath_intervals)[-2:]
                    self.latest_bpm = int(round(np.mean([60.0 / max(1e-6, v) for v in vals])))
                    # 推送呼吸率到队列（带时间戳，毫秒级）
                    # 只有呼吸率在6-200之间才加入队列
                    if self.breath_rate_queue is not None and 6 <= self.latest_bpm <= 200:
                        try:
                            timestamp_ms = int(time.time() * 1000)
                            breath_data = {
                                "breath_value": self.latest_bpm,
                                "timestamp": timestamp_ms
                            }
                            self.breath_rate_queue.put_nowait(breath_data)
                        except queue.Full:
                            pass

    def _process_data(self):
        """数据处理线程"""
        print(f"[呼吸检测] 启动检测线程: sample_rate={self.sample_rate}, smooth_window={self.smooth_window}, slope_window={self.slope_window}")
        
        while not self.stop_event.is_set():
            try:
                # 每次最多读取一定数量
                max_read = 10
                read_cnt = 0
                
                while read_cnt < max_read and not self.stop_event.is_set():
                    try:
                        item = self.data_queue.get(timeout=0.1)
                    except queue.Empty:
                        break
                    
                    angx = float(item.get("AngX", 0.0))
                    self.angx.append(angx)
                    self.data_count += 1
                    
                    # 使用数据包中的时间戳
                    ts_str = str(item.get("datetime", "")).strip()
                    if ts_str:
                        self.current_datetime = ts_str
                    
                    try:
                        if not ts_str:
                            ts_dt = dt.now()
                        else:
                            # 支持毫秒精度
                            if "." in ts_str:
                                date_part, micro_part = ts_str.split(".", 1)
                                if len(micro_part) == 3:
                                    ts_str_parse = f"{date_part}.{micro_part}000"
                                else:
                                    ts_str_parse = ts_str
                            else:
                                ts_str_parse = ts_str
                            ts_dt = dt.strptime(ts_str_parse, "%Y-%m-%d %H:%M:%S.%f")
                        
                        if self._t0 is None:
                            self._t0 = ts_dt
                        rel_sec = (ts_dt - self._t0).total_seconds()
                    except Exception:
                        if self._t0 is None:
                            self._t0 = dt.now()
                        rel_sec = (dt.now() - self._t0).total_seconds()
                    
                    self.ts.append(rel_sec)

                    # 平滑与斜率
                    sm = self._moving_average(self.angx, self.smooth_window)
                    self.smoothed.append(sm)
                    slope = self._calculate_slope_centered(self.smoothed, self.slope_window)
                    self.slopes.append(slope)

                    # 检测峰谷 & 更新 bpm
                    self._detect_events()
                    
                    # 推送原始数据和峰谷点到可视化队列
                    if self.breath_data_queue is not None:
                        try:
                            # 只保留当前窗口内的峰谷点（最近1000个样本）
                            current_window_start = max(0, self.data_count - 1000)
                            visible_peaks = [(idx, val) for idx, t, val in self.peak_points 
                                           if idx >= current_window_start]
                            visible_valleys = [(idx, val) for idx, t, val in self.valley_points 
                                             if idx >= current_window_start]
                            
                            breath_vis_data = {
                                "angx": angx,
                                "peaks": visible_peaks,
                                "valleys": visible_valleys,
                            }
                            self.breath_data_queue.put_nowait(breath_vis_data)
                        except queue.Full:
                            pass
                    
                    read_cnt += 1
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[呼吸检测] 处理错误: {e}")
        
        print("[呼吸检测] 检测线程已停止")

    def start(self) -> threading.Thread:
        """启动呼吸检测器（返回线程对象）"""
        thread = threading.Thread(target=self._process_data, daemon=True)
        thread.start()
        print("[呼吸检测] 呼吸检测器已启动（无绘图模式）")
        return thread


if __name__ == "__main__":
    """测试代码"""
    import random
    
    # 创建测试队列和事件
    test_data_queue = queue.Queue()
    test_breath_queue = queue.Queue()
    test_stop_event = threading.Event()
    
    # 创建检测器
    detector = BreathDetectorNoPlot(
        data_queue=test_data_queue,
        stop_event=test_stop_event,
        sample_rate=100,
        smooth_window=50,
        slope_window=10,
        peak_search_radius=60,
        breath_rate_queue=test_breath_queue,
    )
    
    # 启动检测器
    detector_thread = detector.start()
    
    # 模拟数据生成
    print("生成模拟数据...")
    t = 0
    for i in range(1000):
        # 模拟呼吸波形（正弦波 + 噪声）
        angx = 10 * np.sin(2 * np.pi * 0.2 * t) + random.uniform(-0.5, 0.5)
        test_data_queue.put({
            "AngX": angx,
            "datetime": dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        })
        t += 0.01  # 100Hz
        time.sleep(0.01)
    
    # 读取结果
    time.sleep(2)
    print("\n检测结果:")
    while not test_breath_queue.empty():
        result = test_breath_queue.get()
        print(f"  呼吸率: {result['breath_value']} 次/分钟, 时间戳: {result['timestamp']}")
    
    # 停止检测器
    test_stop_event.set()
    detector_thread.join(timeout=2)
    print("测试完成")
