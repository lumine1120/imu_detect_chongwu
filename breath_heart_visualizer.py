"""
呼吸和心率双图可视化模块
Breath and Heart Rate Dual Visualization Module

从队列中读取原始传感器数据，实时绘制两个独立的图表：
1. 呼吸图：显示 AngX 原始波形、峰谷标记、呼吸率等参数
2. 心率图：显示加速度总量原始波形、当前心率等参数
3. 行为状态：在两个图表中都显示当前行为状态
"""

import threading
import queue
import time
from collections import deque
from typing import Optional, List, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime as dt


class BreathHeartVisualizer:
    """呼吸和心率双图可视化器 - 显示原始数据波形"""

    def __init__(
        self,
        breath_data_queue: queue.Queue,
        heart_data_queue: queue.Queue,
        breath_rate_queue: queue.Queue,
        heart_rate_queue: queue.Queue,
        action_queue: queue.Queue,
        stop_event: threading.Event,
        max_points: int = 1000,
        sample_rate: int = 100,
    ) -> None:
        """
        初始化可视化器
        
        参数:
            breath_data_queue: 呼吸原始数据队列（AngX）
            heart_data_queue: 心率原始数据队列（AccX, AccY, AccZ）
            breath_rate_queue: 呼吸率结果队列
            heart_rate_queue: 心率结果队列
            action_queue: 行为状态队列
            stop_event: 停止事件
            max_points: 波形最大显示点数
            sample_rate: 采样率
        """
        self.breath_data_queue = breath_data_queue
        self.heart_data_queue = heart_data_queue
        self.breath_rate_queue = breath_rate_queue
        self.heart_rate_queue = heart_rate_queue
        self.action_queue = action_queue
        self.stop_event = stop_event
        self.max_points = max_points
        self.sample_rate = sample_rate

        # 呼吸原始数据（AngX 波形）
        self.breath_angx: deque = deque(maxlen=max_points)
        self.breath_sample_idx: deque = deque(maxlen=max_points)
        self.breath_data_count = 0
        self.latest_breath_bpm: float = 0.0
        self.breath_peak_points: List[Tuple[int, float]] = []  # (sample_idx, value)
        self.breath_valley_points: List[Tuple[int, float]] = []

        # 心率原始数据（加速度总量波形）
        self.heart_acc_magnitude: deque = deque(maxlen=max_points)
        self.heart_sample_idx: deque = deque(maxlen=max_points)
        self.heart_data_count = 0
        self.latest_heart_bpm: float = 0.0

        # 行为状态
        self.current_action: str = "未知"
        self.action_map = {
            0: "未知",
            1: "静止",
            2: "行走",
            3: "跑步",
        }

        # 动画更新间隔 (毫秒)
        self.update_interval_ms = 50

        # 创建图形和子图
        self.fig, (self.ax_breath, self.ax_heart) = plt.subplots(2, 1, figsize=(14, 10))
        self.fig.suptitle("呼吸与心率原始波形监测", fontsize=14, fontweight="bold")

        # ==== 呼吸图设置（AngX 波形）====
        (self.line_breath,) = self.ax_breath.plot(
            [], [], color="#1f77b4", lw=1.2, label="AngX 波形"
        )
        self.sc_breath_peaks = self.ax_breath.scatter(
            [], [], c="red", marker="^", s=40, label="吸气峰", zorder=5
        )
        self.sc_breath_valleys = self.ax_breath.scatter(
            [], [], c="green", marker="v", s=40, label="呼气谷", zorder=5
        )
        self.ax_breath.set_title("呼吸波形 (AngX)")
        self.ax_breath.set_xlabel("样本点")
        self.ax_breath.set_ylabel("角度 X (度)")
        self.ax_breath.set_xlim(0, max_points)
        self.ax_breath.set_ylim(-50, 50)  # 初始Y轴范围，会动态调整
        self.ax_breath.grid(True, alpha=0.3)
        self.ax_breath.legend(loc="upper left")
        
        # 呼吸信息文本
        self.text_breath = self.ax_breath.text(
            0.02,
            0.95,
            "",
            transform=self.ax_breath.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
        )

        # ==== 心率图设置（加速度总量波形）====
        (self.line_heart,) = self.ax_heart.plot(
            [], [], color="#ff7f0e", lw=1.2, label="加速度总量"
        )
        self.ax_heart.set_title("心率波形 (加速度总量)")
        self.ax_heart.set_xlabel("样本点")
        self.ax_heart.set_ylabel("加速度总量 (m/s²)")
        self.ax_heart.set_xlim(0, max_points)
        self.ax_heart.set_ylim(0, 2)  # 初始Y轴范围，会动态调整
        self.ax_heart.grid(True, alpha=0.3)
        self.ax_heart.legend(loc="upper left")
        
        # 心率信息文本
        self.text_heart = self.ax_heart.text(
            0.02,
            0.95,
            "",
            transform=self.ax_heart.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6),
        )

    def _update_breath_data(self):
        """从队列中更新呼吸原始数据（AngX）和峰谷点"""
        max_read = 20
        read_count = 0
        
        while read_count < max_read:
            try:
                data = self.breath_data_queue.get_nowait()
                angx = data.get("angx", 0.0)
                self.breath_angx.append(angx)
                self.breath_sample_idx.append(self.breath_data_count)
                self.breath_data_count += 1
                
                # 更新峰谷点
                if "peaks" in data:
                    self.breath_peak_points = data["peaks"]
                if "valleys" in data:
                    self.breath_valley_points = data["valleys"]
                
                read_count += 1
            except queue.Empty:
                break
        
        # 更新呼吸率
        try:
            while True:
                rate_data = self.breath_rate_queue.get_nowait()
                self.latest_breath_bpm = rate_data.get("breath_value", 0)
        except queue.Empty:
            pass

    def _update_heart_data(self):
        """从队列中更新心率原始数据（加速度总量）"""
        max_read = 20
        read_count = 0
        
        while read_count < max_read:
            try:
                data = self.heart_data_queue.get_nowait()
                acc_x = data.get("AccX", 0.0)
                acc_y = data.get("AccY", 0.0)
                acc_z = data.get("AccZ", 0.0)
                
                # 计算加速度总量
                magnitude = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                self.heart_acc_magnitude.append(magnitude)
                self.heart_sample_idx.append(self.heart_data_count)
                self.heart_data_count += 1
                
                read_count += 1
            except queue.Empty:
                break
        
        # 更新心率
        try:
            while True:
                rate_data = self.heart_rate_queue.get_nowait()
                self.latest_heart_bpm = rate_data.get("heart_value", 0)
        except queue.Empty:
            pass

    def _update_action_data(self):
        """从队列中更新行为状态"""
        try:
            while True:
                data = self.action_queue.get_nowait()
                action_code = data.get("action", 0)
                self.current_action = self.action_map.get(action_code, "未知")
        except queue.Empty:
            pass

    def _update_once(self, frame):
        """单次更新函数（由动画调用）"""
        if self.stop_event.is_set():
            return ()

        # 更新所有数据
        self._update_breath_data()
        self._update_heart_data()
        self._update_action_data()

        # ==== 更新呼吸图（AngX 原始波形）====
        if self.breath_sample_idx and self.breath_angx:
            x_data = list(self.breath_sample_idx)
            y_data = list(self.breath_angx)
            self.line_breath.set_data(x_data, y_data)
            
            # 设置X轴范围（总是更新）
            x_min = max(0, self.breath_data_count - self.max_points)
            x_max = max(self.max_points, self.breath_data_count)
            self.ax_breath.set_xlim(x_min, x_max)
            
            # 动态调整Y轴（总是更新）
            if y_data:
                y_min = min(y_data)
                y_max = max(y_data)
                margin = max(1.0, (y_max - y_min) * 0.15)  # 至少1度的边距
                self.ax_breath.set_ylim(y_min - margin, y_max + margin)
            
            # 更新峰谷散点（只显示当前窗口内的）
            x_min = max(0, self.breath_data_count - self.max_points)
            x_max = self.breath_data_count
            
            peak_x, peak_y = [], []
            for idx, val in self.breath_peak_points:
                if x_min <= idx <= x_max:
                    peak_x.append(idx)
                    peak_y.append(val)
            
            valley_x, valley_y = [], []
            for idx, val in self.breath_valley_points:
                if x_min <= idx <= x_max:
                    valley_x.append(idx)
                    valley_y.append(val)
            
            self.sc_breath_peaks.set_offsets(np.column_stack([peak_x, peak_y]) if peak_x else np.empty((0, 2)))
            self.sc_breath_valleys.set_offsets(np.column_stack([valley_x, valley_y]) if valley_x else np.empty((0, 2)))
            
            # 计算统计信息
            current_val = y_data[-1] if y_data else 0
            avg_val = np.mean(y_data) if y_data else 0
            max_val = max(y_data) if y_data else 0
            min_val = min(y_data) if y_data else 0
            
            # 更新文本信息
            self.text_breath.set_text(
                f"当前行为: {self.current_action}\n"
                f"呼吸率: {self.latest_breath_bpm:.1f} 次/分钟\n"
                f"当前 AngX: {current_val:.2f}°\n"
                f"平均值: {avg_val:.2f}°\n"
                f"最大值: {max_val:.2f}°\n"
                f"最小值: {min_val:.2f}°\n"
                f"数据点数: {len(y_data)}"
            )

        # ==== 更新心率图（加速度总量原始波形）====
        if self.heart_sample_idx and self.heart_acc_magnitude:
            x_data = list(self.heart_sample_idx)
            y_data = list(self.heart_acc_magnitude)
            self.line_heart.set_data(x_data, y_data)
            
            # 设置X轴范围（总是更新）
            x_min = max(0, self.heart_data_count - self.max_points)
            x_max = max(self.max_points, self.heart_data_count)
            self.ax_heart.set_xlim(x_min, x_max)
            
            # 动态调整Y轴（总是更新）
            if y_data:
                y_min = min(y_data)
                y_max = max(y_data)
                margin = max(0.2, (y_max - y_min) * 0.15)  # 至少0.2的边距
                self.ax_heart.set_ylim(max(0, y_min - margin), y_max + margin)
            
            # 计算统计信息
            current_val = y_data[-1] if y_data else 0
            avg_val = np.mean(y_data) if y_data else 0
            max_val = max(y_data) if y_data else 0
            min_val = min(y_data) if y_data else 0
            
            # 更新文本信息
            self.text_heart.set_text(
                f"当前行为: {self.current_action}\n"
                f"心率: {self.latest_heart_bpm:.0f} BPM\n"
                f"当前加速度: {current_val:.3f} m/s²\n"
                f"平均值: {avg_val:.3f} m/s²\n"
                f"最大值: {max_val:.3f} m/s²\n"
                f"最小值: {min_val:.3f} m/s²\n"
                f"数据点数: {len(y_data)}"
            )

        return (self.line_breath, self.sc_breath_peaks, self.sc_breath_valleys, 
                self.text_breath, self.line_heart, self.text_heart)

    def start(self):
        """启动可视化"""
        print("=" * 60)
        print("启动呼吸与心率原始波形双图可视化")
        print("  - 上图: AngX 呼吸波形（带峰谷标记）")
        print("  - 下图: 加速度总量心率波形")
        print("=" * 60)
        
        ani = FuncAnimation(
            self.fig,
            self._update_once,
            interval=self.update_interval_ms,
            blit=True,
            cache_frame_data=False,
        )
        
        plt.tight_layout()
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n用户中断可视化")
        finally:
            self.stop_event.set()
            print("可视化已停止")


if __name__ == "__main__":
    """测试代码"""
    import random
    
    # 创建测试队列和事件
    breath_data_queue = queue.Queue()
    heart_data_queue = queue.Queue()
    breath_rate_queue = queue.Queue()
    heart_rate_queue = queue.Queue()
    action_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 模拟数据生成线程
    def generate_test_data():
        t = 0
        data_count = 0
        peaks = []
        valleys = []
        
        while not stop_event.is_set():
            # 生成模拟呼吸波形（AngX 正弦波）
            angx = 10 * np.sin(2 * np.pi * 0.2 * t) + random.uniform(-0.5, 0.5)
            breath_data = {
                "angx": angx,
            }
            
            # 每隔一段时间生成峰谷点
            if data_count % 50 == 0:
                if data_count % 100 == 0:
                    peaks.append((data_count, angx + random.uniform(0, 2)))
                else:
                    valleys.append((data_count, angx - random.uniform(0, 2)))
                breath_data["peaks"] = peaks[-10:]
                breath_data["valleys"] = valleys[-10:]
            
            breath_data_queue.put(breath_data)
            
            # 生成模拟心率波形（加速度总量）
            acc_magnitude = 1.0 + 0.1 * np.sin(2 * np.pi * 1.2 * t) + random.uniform(-0.05, 0.05)
            heart_data = {
                "AccX": acc_magnitude * 0.577,
                "AccY": acc_magnitude * 0.577,
                "AccZ": acc_magnitude * 0.577,
            }
            heart_data_queue.put(heart_data)
            
            # 每秒生成一次呼吸率和心率
            if data_count % 100 == 0:
                breath_rate_queue.put({"breath_value": random.uniform(12, 20)})
                heart_rate_queue.put({"heart_value": random.uniform(60, 100)})
            
            # 随机生成行为状态
            if random.random() < 0.01:
                action_queue.put({"action": random.choice([1, 2, 3])})
            
            t += 0.01
            data_count += 1
            time.sleep(0.01)
    
    # 启动数据生成线程
    data_thread = threading.Thread(target=generate_test_data, daemon=True)
    data_thread.start()
    
    # 创建并启动可视化器
    visualizer = BreathHeartVisualizer(
        breath_data_queue=breath_data_queue,
        heart_data_queue=heart_data_queue,
        breath_rate_queue=breath_rate_queue,
        heart_rate_queue=heart_rate_queue,
        action_queue=action_queue,
        stop_event=stop_event,
        max_points=1000,
        sample_rate=100,
    )
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\n测试中断")
    finally:
        stop_event.set()
        data_thread.join(timeout=1)
        print("测试完成")
