import threading
import queue
import time
from collections import deque
from typing import Deque, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime as dt

from detector import BreathingWaveDetector


class BreathDetector:
	"""
	从 DataReader 的检测队列读取数据，驱动 BreathingWaveDetector 进行峰谷检测与呼吸率计算；
	独立绘图线程基于时间戳（秒）绘制 AngX，并仅显示当前窗口内的峰/谷与呼吸率。
	"""

	def __init__(
		self,
		data_queue_detect: queue.Queue,
		stop_event: threading.Event,
		sample_rate: int = 100,
		max_points: int = 2000,
		smooth_window: int = 25,
		slope_window: int = 10,
	) -> None:
		self.data_queue = data_queue_detect
		self.stop_event = stop_event
		self.sample_rate = sample_rate
		self.max_points = max_points

		# 数据缓存
		self.angx: Deque[float] = deque(maxlen=max_points)
		self.ts: Deque[float] = deque(maxlen=max_points)  # 相对秒

		# 检测器
		self.detector = BreathingWaveDetector(
			buffer_size=max_points,
			smooth_window=max(3, smooth_window),
			slope_window=max(2, slope_window),
		)

		# 呼吸统计
		self.latest_bpm: float = 0.0

		# 绘图对象
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
		self.ax.set_title("BreathDetector2 - AngX with Peaks/Valleys")
		self.ax.set_xlabel("Time (s)")
		self.ax.set_ylabel("AngX (deg)")
		self.ax.grid(True, alpha=0.3)
		self.ax.legend(loc="upper right")

		self.update_interval_ms = int(1000 / max(1, self.sample_rate))

		# 时间起点（以首包时间为参考）
		self._t0 = None

		# 线程
		self.read_thread: Optional[threading.Thread] = None

	@staticmethod
	def _parse_ts_to_datetime(ts_str: str) -> dt:
		ts_str = (ts_str or "").strip()
		if not ts_str:
			return dt.now()
		if "." in ts_str:
			date_part, micro_part = ts_str.split(".", 1)
			if len(micro_part) == 3:
				ts_str = f"{date_part}.{micro_part}000"
		return dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

	def _feed_and_detect(self, item: dict):
		# 提取 AngX 与时间戳
		angx = float(item.get("AngX", 0.0))
		ts_dt = self._parse_ts_to_datetime(str(item.get("datetime", "")))
		if self._t0 is None:
			self._t0 = ts_dt
		rel_sec = (ts_dt - self._t0).total_seconds()

		# 入缓存
		self.angx.append(angx)
		self.ts.append(rel_sec)

		# 调用检测器（使用绝对秒时间戳，兼容 time.time() 类型）
		ret = self.detector.detect(timestamp=ts_dt.timestamp(), value=angx)

		# 更新呼吸率：优先使用 detect 返回，否则 get_statistics
		bpm = None
		if isinstance(ret, dict):
			bpm = ret.get("breath_frequency")
		if bpm is None:
			stats = self.detector.get_statistics()
			bpm = stats.get("breath_frequency", 0.0)
		try:
			self.latest_bpm = float(bpm or 0.0)
		except Exception:
			self.latest_bpm = 0.0

	def _update_plot(self):
		# 每帧尽量批量读取以减少开销
		max_read = 10
		read_cnt = 0
		while read_cnt < max_read and not self.stop_event.is_set():
			try:
				item = self.data_queue.get_nowait()
			except queue.Empty:
				break
			self._feed_and_detect(item)
			read_cnt += 1

		# 更新线条
		x = list(self.ts)
		y = list(self.angx)
		self.line_angx.set_data(x, y)

		# 计算时间窗口
		time_window = float(self.max_points) / max(1, self.sample_rate)
		if x:
			last_t = x[-1]
			x_min = max(0.0, last_t - time_window)
			x_max = max(time_window, last_t)
		else:
			x_min, x_max = 0.0, time_window
		self.ax.set_xlim(x_min, x_max)

		if y:
			ymin, ymax = min(y), max(y)
			margin = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
			self.ax.set_ylim(ymin - margin, ymax + margin)

		# 峰谷绘制（仅当前范围内）。BreathingWaveDetector 的点结构可能包含(时间戳, 值, ...)，取前两位。
		def _filter_points(points):
			xs, ys = [], []
			for p in list(points):
				if isinstance(p, (list, tuple)) and len(p) >= 2:
					t_abs, val = p[0], p[1]
					# 将绝对时间戳转换为相对秒
					if self._t0 is not None:
						t_rel = t_abs - self._t0.timestamp()
					else:
						t_rel = 0.0
					if x_min <= t_rel <= x_max:
						xs.append(t_rel)
						ys.append(val)
			return xs, ys

		px, py = _filter_points(self.detector.peak_points)
		vx, vy = _filter_points(self.detector.valley_points)

		# 先清空再设置，避免残留
		self.sc_peaks.set_offsets(np.empty((0, 2)))
		self.sc_valleys.set_offsets(np.empty((0, 2)))
		self.sc_peaks.set_offsets(np.column_stack([px, py]) if px else np.empty((0, 2)))
		self.sc_valleys.set_offsets(np.column_stack([vx, vy]) if vx else np.empty((0, 2)))

		# 信息文本
		n = len(self.angx)
		current_value = y[-1] if y else 0.0
		avg_value = float(np.mean(y)) if y else 0.0
		max_value = float(np.max(y)) if y else 0.0
		min_value = float(np.min(y)) if y else 0.0
		self.text_info.set_text(
			f"数据点: {n}\n"
			f"当前值: {current_value:.4f}\n"
			f"平均值: {avg_value:.4f}\n"
			f"最大值: {max_value:.4f}\n"
			f"最小值: {min_value:.4f}\n"
			f"BPM: {self.latest_bpm:.2f}"
		)

		return (
			self.line_angx,
			self.sc_peaks,
			self.sc_valleys,
			self.text_info,
		)

	def start(self):
		"""启动读取与绘图"""
		print(
			f"启动 BreathDetector2: sample_rate={self.sample_rate}, max_points={self.max_points}"
		)

		ani = FuncAnimation(
			self.fig,
			lambda frame: self._update_plot(),
			interval=self.update_interval_ms,
			blit=True,
			cache_frame_data=False,
		)
		plt.tight_layout()
		try:
			plt.show()
		finally:
			self.stop_event.set()
    