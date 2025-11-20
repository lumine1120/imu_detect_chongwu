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
    def __init__(self, data_queue, stop_event, max_points=1000, sample_rate=200, source_type='csv'):
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
        
        # 使用deque存储数据，自动维护最大长度
        self.time_data = deque(maxlen=max_points)
        self.acc_magnitude = deque(maxlen=max_points)
        self.datetime_labels = deque(maxlen=max_points)
        
        self.current_datetime = ""
        self.data_count = 0
        
        # 初始化图表
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5)
        
        self.ax.set_xlabel('数据点', fontsize=12)
        self.ax.set_ylabel('加速度幅值 (√(AccX² + AccY² + AccZ²))', fontsize=12)
        self.ax.set_title('实时加速度幅值', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # 添加文本显示当前时间
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                      fontsize=12, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加统计信息文本
        self.stats_text = self.ax.text(0.02, 0.88, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
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
                    data['AccX'], 
                    data['AccY'], 
                    data['AccZ']
                )
                
                # 添加数据
                self.time_data.append(self.data_count)
                self.acc_magnitude.append(magnitude)
                self.datetime_labels.append(data['datetime'])
                self.current_datetime = data['datetime']
                self.data_count += 1
                read_count += 1  # 读取计数+1
                
            except queue.Empty:
                break  # 队列空则退出循环
        
        # 更新图表数据
        if len(self.time_data) > 0:
            self.line.set_data(list(self.time_data), list(self.acc_magnitude))
            
            # 动态调整坐标轴范围
            self.ax.set_xlim(max(0, self.data_count - self.max_points), 
                            max(self.max_points, self.data_count))
            
            if len(self.acc_magnitude) > 0:
                y_min = min(self.acc_magnitude)
                y_max = max(self.acc_magnitude)
                y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # 更新时间文本（所有模式都显示）
            self.time_text.set_text(f'当前时间: {self.current_datetime}')
            
            # 更新统计信息
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
        
        return self.line, self.time_text, self.stats_text
    
    def start(self):
        """
        启动绘图
        """
        print("启动实时绘图...")
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