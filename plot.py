"""
数据可视化模块
从队列读取数据，计算加速度幅值并实时绘图
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
from threading import Event
from collections import deque
import time


class DataPlotter:
    def __init__(self, data_queue, stop_event, max_points=1000, sample_rate=200, source_type='csv', model=1):
        """
        初始化数据绘图器
        
        Args:
            data_queue: 数据队列
            stop_event: 停止信号
            max_points: 图表中显示的最大数据点数
            sample_rate: 采样率(Hz)，默认200，即每秒200个数据点
            source_type: 数据源类型，'csv' 或 'ble'
        """
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.max_points = max_points
        self.sample_rate = sample_rate
        self.source_type = source_type
        self.update_interval = 1000 / sample_rate  # 毫秒
        self.model = model  # 1: 仅总加速度; 2: 总加速度+AngX; 3: 总加速度+AngY; 4: 总加速度+AngZ; 5: 总加速度+AngX+AngY+AngZ
        
        # 使用deque存储数据，自动维护最大长度
        self.time_data = deque(maxlen=max_points)
        self.acc_magnitude = deque(maxlen=max_points)
        self.angX_data = deque(maxlen=max_points)
        self.angY_data = deque(maxlen=max_points)
        self.angZ_data = deque(maxlen=max_points)
        self.datetime_labels = deque(maxlen=max_points)
        
        self.current_datetime = ""
        self.data_count = 0
        
        # 初始化图表
        if self.model == 1:
            self.fig, self.ax_acc = plt.subplots(figsize=(12, 6))
            self.line_acc, = self.ax_acc.plot([], [], 'b-', linewidth=1.5)
            self.ax_acc.set_xlabel('数据点', fontsize=12)
            self.ax_acc.set_ylabel('加速度幅值 (√(AccX² + AccY² + AccZ²))', fontsize=12)
            self.ax_acc.set_title('实时加速度幅值', fontsize=14, fontweight='bold')
            self.ax_acc.grid(True, alpha=0.3)
            self.time_text = self.ax_acc.text(0.02, 0.95, '', transform=self.ax_acc.transAxes,
                                              fontsize=12, verticalalignment='top',
                                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            self.stats_text = self.ax_acc.text(0.02, 0.88, '', transform=self.ax_acc.transAxes,
                                               fontsize=10, verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        elif self.model in [2,3,4]:
            self.fig, (self.ax_acc, self.ax_ang) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            self.line_acc, = self.ax_acc.plot([], [], 'b-', linewidth=1.2)
            self.line_ang, = self.ax_ang.plot([], [], 'r-', linewidth=1.2)
            self.ax_acc.set_ylabel('加速度幅值', fontsize=12)
            self.ax_acc.set_title('实时加速度 + 角度', fontsize=14, fontweight='bold')
            ang_label = {2: 'AngX', 3: 'AngY', 4: 'AngZ'}[self.model]
            self.ax_ang.set_ylabel(f'{ang_label} (deg)', fontsize=12)
            self.ax_ang.grid(True, alpha=0.3)
            self.ax_acc.grid(True, alpha=0.3)
            self.time_text = self.ax_acc.text(0.02, 0.95, '', transform=self.ax_acc.transAxes,
                                              fontsize=11, verticalalignment='top',
                                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
            self.stats_text = self.ax_acc.text(0.02, 0.80, '', transform=self.ax_acc.transAxes,
                                               fontsize=10, verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
        elif self.model == 5:
            self.fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            self.ax_acc, self.ax_angX, self.ax_angY, self.ax_angZ = axes
            self.line_acc, = self.ax_acc.plot([], [], 'b-', linewidth=1.0)
            self.line_angX, = self.ax_angX.plot([], [], 'r-', linewidth=1.0)
            self.line_angY, = self.ax_angY.plot([], [], 'g-', linewidth=1.0)
            self.line_angZ, = self.ax_angZ.plot([], [], 'm-', linewidth=1.0)
            self.ax_acc.set_ylabel('加速度幅值', fontsize=11)
            self.ax_angX.set_ylabel('AngX (deg)', fontsize=11)
            self.ax_angY.set_ylabel('AngY (deg)', fontsize=11)
            self.ax_angZ.set_ylabel('AngZ (deg)', fontsize=11)
            self.ax_acc.set_title('实时加速度与姿态角', fontsize=14, fontweight='bold')
            for ax in axes:
                ax.grid(True, alpha=0.3)
            self.time_text = self.ax_acc.text(0.02, 0.90, '', transform=self.ax_acc.transAxes,
                                              fontsize=11, verticalalignment='top',
                                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
            self.stats_text = self.ax_acc.text(0.02, 0.75, '', transform=self.ax_acc.transAxes,
                                               fontsize=9, verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
    
    def calculate_magnitude(self, accX, accY, accZ):
        """
        计算加速度的幅值(平方根)
        
        Args:
            accX, accY, accZ: 三轴加速度
            
        Returns:
            加速度幅值
        """
        return np.sqrt(accX**2 + accY**2 + accZ**2)
    
    def update_plot(self, frame):
        """
        更新图表的回调函数
        每次最多从队列读取10个数据，减少IO操作次数
        """
        max_read = 10  # 每次最多读取10个数据
        read_count = 0  # 已读取计数
        
        while read_count < max_read and not self.stop_event.is_set():
            if self.data_queue.empty():
                break  # 队列空了就停止读取
            
            try:
                data = self.data_queue.get_nowait()  # 非阻塞读取
                
                # 计算加速度幅值
                magnitude = self.calculate_magnitude(
                    data.get('AccX', 0.0), 
                    data.get('AccY', 0.0), 
                    data.get('AccZ', 0.0)
                )
                
                # 添加数据
                self.time_data.append(self.data_count)
                self.acc_magnitude.append(magnitude)
                self.datetime_labels.append(data['datetime'])
                self.current_datetime = data['datetime']
                # 角度数据（可能为0或存在）
                self.angX_data.append(data.get('AngX', 0.0))
                self.angY_data.append(data.get('AngY', 0.0))
                self.angZ_data.append(data.get('AngZ', 0.0))
                self.data_count += 1
                read_count += 1  # 读取计数+1
                
            except queue.Empty:
                break  # 队列空则退出循环
        
        # 更新图表数据
        if len(self.time_data) > 0:
            # 公共 X 范围
            x_min = max(0, self.data_count - self.max_points)
            x_max = max(self.max_points, self.data_count)
            # 总加速度图
            if self.model == 1:
                self.line_acc.set_data(list(self.time_data), list(self.acc_magnitude))
                self.ax_acc.set_xlim(x_min, x_max)
                if len(self.acc_magnitude) > 0:
                    y_min = min(self.acc_magnitude)
                    y_max = max(self.acc_magnitude)
                    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                    self.ax_acc.set_ylim(y_min - y_margin, y_max + y_margin)
            elif self.model in [2,3,4]:
                ang_series = {2: self.angX_data, 3: self.angY_data, 4: self.angZ_data}[self.model]
                self.line_acc.set_data(list(self.time_data), list(self.acc_magnitude))
                self.line_ang.set_data(list(self.time_data), list(ang_series))
                self.ax_acc.set_xlim(x_min, x_max)
                self.ax_ang.set_xlim(x_min, x_max)
                # y 范围
                if len(self.acc_magnitude) > 0:
                    y_min = min(self.acc_magnitude)
                    y_max = max(self.acc_magnitude)
                    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                    self.ax_acc.set_ylim(y_min - y_margin, y_max + y_margin)
                if len(ang_series) > 0:
                    ay_min = min(ang_series)
                    ay_max = max(ang_series)
                    ay_margin = (ay_max - ay_min) * 0.1 if ay_max > ay_min else 1.0
                    self.ax_ang.set_ylim(ay_min - ay_margin, ay_max + ay_margin)
            elif self.model == 5:
                self.line_acc.set_data(list(self.time_data), list(self.acc_magnitude))
                self.line_angX.set_data(list(self.time_data), list(self.angX_data))
                self.line_angY.set_data(list(self.time_data), list(self.angY_data))
                self.line_angZ.set_data(list(self.time_data), list(self.angZ_data))
                self.ax_acc.set_xlim(x_min, x_max)
                self.ax_angX.set_xlim(x_min, x_max)
                self.ax_angY.set_xlim(x_min, x_max)
                self.ax_angZ.set_xlim(x_min, x_max)
                if len(self.acc_magnitude) > 0:
                    y_min = min(self.acc_magnitude)
                    y_max = max(self.acc_magnitude)
                    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                    self.ax_acc.set_ylim(y_min - y_margin, y_max + y_margin)
                # Each angle axis range
                for series, ax in [(self.angX_data, self.ax_angX), (self.angY_data, self.ax_angY), (self.angZ_data, self.ax_angZ)]:
                    if len(series) > 0:
                        ay_min = min(series)
                        ay_max = max(series)
                        ay_margin = (ay_max - ay_min) * 0.1 if ay_max > ay_min else 1.0
                        ax.set_ylim(ay_min - ay_margin, ay_max + ay_margin)

            # 时间与统计文本
            self.time_text.set_text(f'当前时间: {self.current_datetime}')
            if len(self.acc_magnitude) > 0:
                current_value = self.acc_magnitude[-1]
                avg_value = np.mean(self.acc_magnitude)
                max_value = np.max(self.acc_magnitude)
                min_value = np.min(self.acc_magnitude)
                self.stats_text.set_text(
                    f'数据点: {self.data_count}\n'
                    f'当前值: {current_value:.4f}\n'
                    f'平均值: {avg_value:.4f}\n'
                    f'最大值: {max_value:.4f}\n'
                    f'最小值: {min_value:.4f}'
                )

        # 返回依赖的艺术对象供 blit
        if self.model == 1:
            return self.line_acc, self.time_text, self.stats_text
        elif self.model in [2,3,4]:
            return self.line_acc, self.line_ang, self.time_text, self.stats_text
        else:  # model 5
            return self.line_acc, self.line_angX, self.line_angY, self.line_angZ, self.time_text, self.stats_text
    
    def start(self):
        """
        启动绘图
        """
        print("启动实时绘图 (model={})...".format(self.model))
        print(f"采样率: {self.sample_rate} Hz")
        print(f"更新间隔: {self.update_interval:.2f} ms")
        print(f"每次最多读取数据量: 10个")  # 新增提示
        
        # 创建动画
        ani = FuncAnimation(
            self.fig, 
            self.update_plot,
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 测试代码
    test_queue = queue.Queue()
    stop_event = Event()
    
    # 生成一些测试数据
    import threading
    def generate_test_data():
        for i in range(1000):
            if stop_event.is_set():
                break
            data = {
                'datetime': f"00:{i//60:02d}.{i%60}",
                'AccX': np.sin(i * 0.1) + np.random.randn() * 0.1,
                'AccY': np.cos(i * 0.1) + np.random.randn() * 0.1,
                'AccZ': np.sin(i * 0.05) + np.random.randn() * 0.1
            }
            test_queue.put(data)
            time.sleep(0.005)  # 200Hz
    
    # 启动数据生成线程
    data_thread = threading.Thread(target=generate_test_data)
    data_thread.daemon = True
    data_thread.start()
    
    # 启动绘图
    plotter = DataPlotter(test_queue, stop_event, max_points=500, sample_rate=200)
    plotter.start()