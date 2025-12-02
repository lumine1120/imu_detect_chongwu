import threading
import queue
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime as dt


class BreathDetector:
    """
    从 DataReader 的 data_queue_detect 读取数据，仅基于 AngX 计算呼吸率并绘图：
    - 不进行平滑：直接使用原始 AngX 计算斜率与峰谷
    - 斜率：左右窗口均值差/窗口（基于原始 AngX）
    - 峰谷：斜率过零 + 局部搜索
    - 绘图：只绘制原始 AngX，标注峰/谷，并显示最新呼吸率
    - 横轴：使用样本序号，窗口按点数 max_points 控制（与 DataPlotter 一致）
    """

    def __init__(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = 100,
        max_points: int = 2000,
        slope_window: int = 10,
        peak_search_radius: int = 10,
    ) -> None:
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.max_points = max_points
        self.slope_window = max(2, slope_window)
        self.peak_search_radius = max(5, peak_search_radius)

        # 数据缓存（原始 AngX、斜率、时间戳相对秒）
        self.angx: Deque[float] = deque(maxlen=max_points)
        self.slopes: Deque[float] = deque(maxlen=max_points)
        self.ts: Deque[float] = deque(maxlen=max_points)

        # 检测到的峰谷（绝对样本索引、相对时间、值）
        self.peak_points: List[Tuple[int, float, float]] = []
        self.valley_points: List[Tuple[int, float, float]] = []

        # 呼吸率（通过 valley->valley 计算）
        self.breath_intervals: Deque[float] = deque(maxlen=10)
        self.latest_bpm: float = 0.0

        # 绘图相关
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        (self.line_angx,) = self.ax.plot([], [], color="#1f77b4", lw=1.2, label="AngX")
        self.sc_peaks = self.ax.scatter([], [], c="red", marker="^", s=40, label="Peaks")
        self.sc_valleys = self.ax.scatter([], [], c="green", marker="v", s=40, label="Valleys")
        self.text_info = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
        )
        self.ax.set_title("BreathDetector3 - AngX (No Smoothing)")
        self.ax.set_xlabel("数据点")
        self.ax.set_ylabel("AngX (deg)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")

        # 动画更新间隔
        self.update_interval_ms = 10

        # 相对时间基准
        self._t0 = None
        self.current_datetime: str = ""
        # 样本计数（横轴）
        self.data_count: int = 0

    # --------- 算法 ---------
    def _calculate_slope_centered(self, data: Deque[float], window_size: int) -> float:
        # 左右窗口均值差 / window_size
        if len(data) < window_size * 2 + 1:
            return 0.0
        recent = list(data)[-(window_size * 2 + 1) :]
        center = window_size
        left_avg = float(np.mean(recent[:center]))
        right_avg = float(np.mean(recent[center + 1 :]))
        return (right_avg - left_avg) / float(window_size)

    def _detect_events(self):
        # 基于原始数据的斜率过零检测
        if len(self.slopes) < 2 or len(self.angx) < (self.slope_window * 2 + 1):
            return
        s_prev = self.slopes[-2]
        s_curr = self.slopes[-1]
        idx_curr = len(self.angx) - 1

        origin_idx = idx_curr
        start = max(0, origin_idx - self.peak_search_radius)
        end = origin_idx + self.peak_search_radius + 1
        angx_list = list(self.angx)
        ts_list = list(self.ts)

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
                if len(self.valley_points) >= 2:
                    t1 = self.valley_points[-2][1]
                    t2 = self.valley_points[-1][1]
                    dtv = max(1e-6, t2 - t1)
                    self.breath_intervals.append(dtv)
                    vals = list(self.breath_intervals)[-2:]
                    self.latest_bpm = float(np.mean([60.0 / max(1e-6, v) for v in vals]))

    # --------- 动画循环 ---------
    def _update_once(self):
        max_read = 10
        read_cnt = 0
        while read_cnt < max_read and not self.stop_event.is_set():
            try:
                item = self.data_queue.get_nowait()
            except queue.Empty:
                break

            angx = float(item.get("AngX", 0.0))
            self.angx.append(angx)
            self.data_count += 1

            ts_str = str(item.get("datetime", "")).strip()
            if ts_str:
                self.current_datetime = ts_str
            try:
                if not ts_str:
                    ts_dt = dt.now()
                else:
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

            # 斜率基于原始 AngX
            slope = self._calculate_slope_centered(self.angx, self.slope_window)
            self.slopes.append(slope)

            # 检测峰谷 & 更新 bpm
            self._detect_events()
            read_cnt += 1

        # 绘图（横轴为样本索引）
        n = len(self.angx)
        start_idx = max(0, self.data_count - n)
        x = list(range(start_idx, start_idx + n))
        y = list(self.angx)
        self.line_angx.set_data(x, y)

        x_min = max(0, self.data_count - self.max_points)
        x_max = max(self.max_points, self.data_count)
        self.ax.set_xlim(x_min, x_max)
        if y:
            ymin, ymax = min(y), max(y)
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
            self.ax.set_ylim(ymin - margin, ymax + margin)

        # 更新峰谷散点（当前窗口内）
        if self.peak_points:
            px_all = [p[0] for p in self.peak_points]
            py_all = [p[2] for p in self.peak_points]
            px, py = [], []
            for xi, yi in zip(px_all, py_all):
                if x_min <= xi <= x_max:
                    px.append(xi)
                    py.append(yi)
        else:
            px, py = [], []
        if self.valley_points:
            vx_all = [v[0] for v in self.valley_points]
            vy_all = [v[2] for v in self.valley_points]
            vx, vy = [], []
            for xi, yi in zip(vx_all, vy_all):
                if x_min <= xi <= x_max:
                    vx.append(xi)
                    vy.append(yi)
        else:
            vx, vy = [], []

        self.sc_peaks.set_offsets(np.empty((0, 2)))
        self.sc_valleys.set_offsets(np.empty((0, 2)))
        self.sc_peaks.set_offsets(np.column_stack([px, py]) if px else np.empty((0, 2)))
        self.sc_valleys.set_offsets(np.column_stack([vx, vy]) if vx else np.empty((0, 2)))

        # 文本信息
        if y:
            current_value = y[-1]
            avg_value = float(np.mean(y)) if len(y) else 0.0
            max_value = float(np.max(y)) if len(y) else 0.0
            min_value = float(np.min(y)) if len(y) else 0.0
        else:
            current_value = avg_value = max_value = min_value = 0.0
        self.text_info.set_text(
            f"当前时间: {self.current_datetime}\n"
            f"数据点: {n}\n"
            f"当前值: {current_value:.4f}\n"
            f"平均值: {avg_value:.4f}\n"
            f"最大值: {max_value:.4f}\n"
            f"最小值: {min_value:.4f}\n"
            f"BPM: {self.latest_bpm:.2f}"
        )

        return (self.line_angx, self.sc_peaks, self.sc_valleys, self.text_info)

    def start(self):
        print(
            f"启动 BreathDetector3(无平滑): sample_rate={self.sample_rate}, slope_window={self.slope_window}"
        )
        ani = FuncAnimation(
            self.fig,
            lambda frame: self._update_once(),
            interval=self.update_interval_ms,
            blit=True,
            cache_frame_data=False,
        )
        plt.tight_layout()
        try:
            plt.show()
        finally:
            self.stop_event.set()
