# coding:UTF-8
"""
行为检测器
Action Detector

基于加速度数据识别用户行为（静止、行走、跑步）
Identifies user actions based on acceleration data (stationary, walking, running)
"""

import queue
import threading
from collections import deque
import math
import time
from datetime import datetime


class ActionDetector:
    """行为检测器类 Action Detector Class"""
    
    # 行为状态常量 Action state constants
    # 0: 未知 (Unknown)
    # 1: 静止 (Stationary)
    # 2: 行走 (Walking)
    # 3: 跑步 (Running)
    STATE_UNKNOWN = 0
    STATE_STATIONARY = 1
    STATE_WALKING = 2
    STATE_RUNNING = 3
    
    def __init__(
        self, 
        data_queue: queue.Queue,
        stop_event: threading.Event,
        action_queue: queue.Queue = None,
        window_size: int = 200,
        stationary_threshold: tuple = (0.8, 1.2),
        walking_threshold: tuple = (0.5, 2.0),
        running_threshold: float = 2.0,
        confirm_frames: int = 10
    ):
        """
        初始化行为检测器
        
        参数:
            data_queue: 加速度数据队列（从DataReader传入）
            stop_event: 停止信号事件
            action_queue: 行为结果队列（传递给ResultMonitor）
            window_size: 滑动窗口大小，用于保存最近N个数据点
            stationary_threshold: 静止状态的加速度范围 (min, max)
            walking_threshold: 行走状态的加速度范围 (min, max)
            running_threshold: 跑步状态的最小加速度
        """
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.action_queue = action_queue
        self.window_size = window_size
        self.stationary_range = stationary_threshold
        self.walking_range = walking_threshold
        self.running_threshold = running_threshold
        # 抖动过滤：状态变更需未来N帧确认
        self.confirm_frames = max(1, int(confirm_frames))
        
        # 当前状态
        self.current_state = self.STATE_UNKNOWN
        
        # 统计信息
        self.state_change_count = 0

        # 待确认的目标状态与计数
        self._pending_state = None
        self._pending_count = 0
        
    def _calculate_total_acceleration(self, acc_x: float, acc_y: float, acc_z: float) -> float:
        return math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    def _analyze_action_state(self, recent_acc: deque) -> str:
        """
        分析行为状态
        
        参数:
            recent_acc: 最近的加速度数据列表
            
        返回:
            行为状态字符串
        
        判断规则:
            1. 90%以上数据在 0.8-1.2 范围 → 静止
            2. 30%以上数据 > 2.0 → 跑步
            3. 其他情况 → 行走
        """
        if len(recent_acc) == 0:
            return self.STATE_UNKNOWN
            
        total_count = len(recent_acc)
        
        # 统计静止范围的数据个数 (0.8-1.2)
        stationary_count = sum(
            1 for a in recent_acc 
            if self.stationary_range[0] <= a <= self.stationary_range[1]
        )
        
        # 判断是否为静止：90%以上数据在 0.8-1.2 范围
        if stationary_count / total_count >= 0.90:
            return self.STATE_STATIONARY
        
        # 统计跑步范围的数据个数 (>2.0)
        running_count = sum(
            1 for a in recent_acc 
            if a > self.running_threshold
        )
        
        # 如果跑步数据占比较大，判断为跑步
        if running_count / total_count >= 0.3:  # 30%以上数据>2.0，判断为跑步
            return self.STATE_RUNNING
        
        # 否则判断为行走
        return self.STATE_WALKING
    
    def _push_action_result(self, action: str, timestamp: str = None):
        """
        将行为检测结果推送到队列
        
        参数:
            action: 检测到的行为
            timestamp: 时间戳（可选）
        """
        if self.action_queue is None:
            return
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        action_data = {
            "action": action,
            "timestamp": timestamp
        }
        
        try:
            self.action_queue.put_nowait(action_data)
        except queue.Full:
            # 队列满时丢弃旧数据
            try:
                self.action_queue.get_nowait()
                self.action_queue.put_nowait(action_data)
            except:
                pass
    
    def _detection_loop(self):
        """
        检测循环（在独立线程中运行）
        """
        # 保存最近N个加速度数据
        recent_acc = deque(maxlen=self.window_size)
        
        # 用于计数，每200次打印一次
        data_count = 0
        
        print("[行为检测] 检测线程已启动...")
        
        while not self.stop_event.is_set():
            try:
                # 从队列中获取数据，超时1秒
                data = self.data_queue.get(timeout=1)
                
                if data is None:
                    continue
                
                # 从字典中提取三轴加速度
                acc_x = data.get('AccX')
                acc_y = data.get('AccY')
                acc_z = data.get('AccZ')
                
                if acc_x is not None and acc_y is not None and acc_z is not None:
                    # 计算加速度总大小（向量模）
                    total_acc = self._calculate_total_acceleration(acc_x, acc_y, acc_z)
                    recent_acc.append(total_acc)
                    data_count += 1
                
                # 至少需要一半的窗口大小才能开始分析
                min_data_points = max(5, self.window_size // 2)
                
                if len(recent_acc) >= min_data_points:
                    # 分析最新状态（可能抖动）
                    observed_state = self._analyze_action_state(recent_acc)

                    # 每200次数据打印一次当前行为（观察值）
                    if data_count % 200 == 0:
                        action_text = {0: "未知", 1: "静止", 2: "行走", 3: "跑步"}.get(observed_state, "未知")
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[行为检测] 时间: {current_time} | 观察行为: {action_text} ({observed_state}) | 当前状态: {self.current_state}")

                    # 抖动过滤逻辑：
                    # 1) 若观察到不同于当前状态，则启动/更新待确认状态计数
                    # 2) 若观察回到当前状态，则取消待确认
                    # 3) 若观察到另一新状态（非待确认目标），重置待确认为该新状态
                    # 4) 待确认计数达到阈值时，正式切换状态并入队

                    if observed_state == self.current_state:
                        # 观察回到当前状态，取消待确认
                        self._pending_state = None
                        self._pending_count = 0
                    else:
                        # 如果没有待确认，或观察到不同的目标状态，则重置待确认为该新状态
                        if self._pending_state is None or self._pending_state != observed_state:
                            self._pending_state = observed_state
                            self._pending_count = 1
                        else:
                            # 同一待确认目标，计数+1
                            self._pending_count += 1

                        # 达到确认帧数阈值，执行状态切换与入队
                        if self._pending_count >= self.confirm_frames:
                            self.current_state = self._pending_state
                            self.state_change_count += 1
                            self._push_action_result(self.current_state)
                            # 切换完成后清空待确认
                            self._pending_state = None
                            self._pending_count = 0
                        
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"[行为检测] 检测循环错误: {e}")
        
        print("[行为检测] 检测线程已停止")
    
    def start(self) -> threading.Thread:
        """
        启动行为检测器
        
        返回:
            检测线程对象
        """
        thread = threading.Thread(target=self._detection_loop, daemon=True)
        thread.start()
        print("[行为检测] 行为检测器已启动")
        return thread
    
    def get_current_state(self) -> str:
        """获取当前行为状态"""
        return self.current_state
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        返回:
            统计信息字典
        """
        return {
            'current_state': self.current_state,
            'state_change_count': self.state_change_count,
            'queue_size': self.data_queue.qsize(),
            'is_running': not self.stop_event.is_set()
        }


# 使用示例
if __name__ == "__main__":
    import random
    
    # 创建测试队列和事件
    test_data_queue = queue.Queue()
    test_action_queue = queue.Queue()
    test_stop_event = threading.Event()
    
    # 创建行为检测器
    detector = ActionDetector(
        data_queue=test_data_queue,
        stop_event=test_stop_event,
        action_queue=test_action_queue
    )
    
    # 启动检测器
    detector_thread = detector.start()
    
    # 模拟数据生成
    print("模拟静止状态...")
    for i in range(15):
        test_data_queue.put({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'AccX': random.uniform(-0.1, 0.1),
            'AccY': random.uniform(-0.1, 0.1),
            'AccZ': random.uniform(0.9, 1.1)
        })
        time.sleep(0.1)
    
    print("模拟行走状态...")
    for i in range(15):
        test_data_queue.put({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'AccX': random.uniform(0.5, 1.0),
            'AccY': random.uniform(0.2, 0.5),
            'AccZ': random.uniform(0.8, 1.5)
        })
        time.sleep(0.1)
    
    print("模拟跑步状态...")
    for i in range(15):
        test_data_queue.put({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'AccX': random.uniform(1.5, 2.5),
            'AccY': random.uniform(0.5, 1.5),
            'AccZ': random.uniform(1.0, 2.0)
        })
        time.sleep(0.1)
    
    # 读取结果
    time.sleep(1)
    print("\n检测结果:")
    while not test_action_queue.empty():
        result = test_action_queue.get()
        print(f"  {result}")
    
    # 打印统计信息
    print(f"\n统计信息: {detector.get_statistics()}")
    
    # 停止检测器
    test_stop_event.set()
    detector_thread.join(timeout=2)
    print("测试完成")
