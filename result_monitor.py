"""
结果监控模块
从心率和呼吸队列读取检测结果并打印输出，同时定期上传到服务器
"""
import threading
import queue
import time
import requests
from typing import Optional


class ResultMonitor:
    """
    监控心率和呼吸检测结果的类
    功能：
    1. 从队列读取数据并打印到控制台
    2. 本地缓存数据，每20秒上传到服务器
    """
    
    def __init__(
        self,
        heart_rate_queue: queue.Queue,
        breath_rate_queue: queue.Queue,
        stop_event: threading.Event,
        heart_upload_url: Optional[str] = None,
        breath_upload_url: Optional[str] = None,
        upload_interval: Optional[int] = None,
        action_queue: Optional[queue.Queue] = None,
        # 新增：分别控制上传间隔与行为上传策略
        heart_breath_upload_interval: int = 4,
        action_upload_url: Optional[str] = None,
        action_upload_mode: str = "on_event",  # 'on_event' | 'interval'
        action_upload_interval: int = 10
    ):
        """
        初始化结果监控器
        
        Args:
            heart_rate_queue: 心率结果队列（包含 {"heart_value": xx, "timestamp": xx}）
            breath_rate_queue: 呼吸率结果队列（包含 {"breath_value": xx, "timestamp": xx}）
            stop_event: 停止信号事件
            heart_upload_url: 心率数据上传URL（可选）
            breath_upload_url: 呼吸数据上传URL（可选）
            upload_interval: 兼容旧参数：统一上传间隔（秒）。若提供，则用于覆盖 heart_breath_upload_interval
            action_queue: 行为检测结果队列（包含 {"action": xx, "timestamp": xx}）（可选）
            heart_breath_upload_interval: 心率/呼吸上传间隔（秒），默认10秒
            action_upload_url: 行为上传URL（可选）
            action_upload_mode: 行为上传模式：'on_event'（有新数据立刻上传，默认）或 'interval'（定时上传）
            action_upload_interval: 行为定时上传间隔（秒），默认10秒
        """
        self.heart_rate_queue = heart_rate_queue
        self.breath_rate_queue = breath_rate_queue
        self.action_queue = action_queue
        self.stop_event = stop_event
        self.heart_upload_url = heart_upload_url
        self.breath_upload_url = breath_upload_url
        # 兼容旧的 upload_interval；若传入则覆盖心/呼吸的独立间隔
        self.hb_upload_interval = (
            upload_interval if upload_interval is not None else heart_breath_upload_interval
        )
        # 行为上传配置
        self.action_upload_url = action_upload_url
        # 不再使用 action_upload_mode,两种模式同时启用
        self.action_upload_interval = action_upload_interval
        
        # 本地缓存队列（用于累积数据后上传）
        self.heart_data_cache = []
        self.breath_data_cache = []
        self.action_data_cache = []  # 行为数据缓存
        
        # 最新的检测结果（用于控制台显示）
        self.latest_heart_rate = None
        self.latest_breath_rate = None
        self.latest_heart_timestamp = None
        self.latest_breath_timestamp = None
        self.latest_action = None
        self.latest_action_timestamp = None
        
        # 统计信息
        self.heart_rate_count = 0
        self.breath_rate_count = 0
        self.action_count = 0
        self.heart_upload_count = 0
        self.breath_upload_count = 0
        self.action_upload_count = 0
        self.heart_upload_failed = 0
        self.breath_upload_failed = 0
        self.action_upload_failed = 0
        
    def _read_queues(self):
        """从队列读取数据并缓存到本地"""
        # 读取心率队列
        try:
            while not self.heart_rate_queue.empty():
                heart_data = self.heart_rate_queue.get_nowait()
                if heart_data and isinstance(heart_data, dict):
                    heart_value = heart_data.get("heart_value")
                    timestamp = heart_data.get("timestamp")
                    if heart_value is not None and heart_value > 0:
                        # 更新最新值（用于显示）
                        self.latest_heart_rate = heart_value
                        self.latest_heart_timestamp = timestamp
                        self.heart_rate_count += 1
                        
                        # 根据当前行为状态设置置信度：静止(1)时为1.0，其他状态为0.0
                        # 状态值: 0=未知, 1=静止, 2=行走, 3=跑步
                        confidence = 1.0 if self.latest_action == 1 else 0.0
                        
                        # 添加到缓存（用于上传），timestamp转换为毫秒时间戳
                        upload_heart_data = {
                            "value": heart_value,
                            "timestamp": str(timestamp) if timestamp is not None else None,
                            "source": "imu",
                            "confidence": confidence
                        }
                        self.heart_data_cache.append(upload_heart_data)
        except queue.Empty:
            pass
        
        # 读取呼吸率队列
        try:
            while not self.breath_rate_queue.empty():
                breath_data = self.breath_rate_queue.get_nowait()
                if breath_data and isinstance(breath_data, dict):
                    breath_value = breath_data.get("breath_value")
                    timestamp = breath_data.get("timestamp")
                    if breath_value is not None and breath_value > 0:
                        # 更新最新值（用于显示）
                        self.latest_breath_rate = breath_value
                        self.latest_breath_timestamp = timestamp
                        self.breath_rate_count += 1
                        
                        # 根据当前行为状态设置置信度：静止(1)时为1.0，其他状态为0.0
                        # 状态值: 0=未知, 1=静止, 2=行走, 3=跑步
                        confidence = 1.0 if self.latest_action == 1 else 0.0
                        
                        # 添加到缓存（用于上传），timestamp转换为毫秒时间戳
                        upload_breath_data = {
                            "value": breath_value,
                            "timestamp": str(timestamp) if timestamp is not None else None,
                            "source": "imu",
                            "confidence": confidence
                        }
                        self.breath_data_cache.append(upload_breath_data)
        except queue.Empty:
            pass
        
        # 读取行为检测队列
        # 状态值: 0=未知, 1=静止, 2=行走, 3=跑步
        if self.action_queue:
            try:
                while not self.action_queue.empty():
                    action_data = self.action_queue.get_nowait()
                    if action_data and isinstance(action_data, dict):
                        # 兼容读取：优先使用 'value'，回退到 'action'
                        action = action_data.get("value", action_data.get("action"))
                        timestamp = action_data.get("timestamp")
                        if action is not None:
                            # 更新最新值并打印
                            self.latest_action = action
                            self.latest_action_timestamp = timestamp
                            self.action_count += 1
                            # 立即打印行为变化（转换为可读文本）
                            action_text = {0: "未知", 1: "静止", 2: "行走", 3: "跑步"}.get(action, "未知")
                            print(f"\n[行为检测] 时间: {timestamp} | 行为: {action_text} ({action})")

                            # 模式1：状态切换立即上报（事件触发）
                            if self.action_upload_url:
                                self._upload_action_data_immediate(action, timestamp)
            except queue.Empty:
                pass
    
    def _print_status(self):
        """打印当前状态"""
        heart_str = f"{self.latest_heart_rate:.0f} bpm" if self.latest_heart_rate else "-- bpm"
        breath_str = f"{self.latest_breath_rate:.1f} bpm" if self.latest_breath_rate else "-- bpm"
        # 状态值转换为文本: 0=未知, 1=静止, 2=行走, 3=跑步
        action_text = {0: "未知", 1: "静止", 2: "行走", 3: "跑步"}.get(self.latest_action, "--") if self.latest_action is not None else "--"
        
        print(f"\r[监控] 心率: {heart_str:>10} | 呼吸率: {breath_str:>10} | 行为: {action_text:>6} | "
              f"心率:{self.heart_rate_count:>3}次 | 呼吸:{self.breath_rate_count:>3}次 | 行为:{self.action_count:>3}次 | "
              f"缓存:心{len(self.heart_data_cache):>2}/呼{len(self.breath_data_cache):>2} | "
              f"已上传:心{self.heart_upload_count:>2}/呼{self.breath_upload_count:>2} | "
              f"失败:心{self.heart_upload_failed:>2}/呼{self.breath_upload_failed:>2}", 
              end='', flush=True)
    
    def _upload_heart_data(self):
        """上传心率数据到服务器（异步发送，不等待响应）"""
        if not self.heart_upload_url or len(self.heart_data_cache) == 0:
            return
        
        # 准备上传数据（使用新的JSON格式）
        upload_data = {
            "deviceState": "on_body",
            "data": self.heart_data_cache.copy()
        }
        count = len(self.heart_data_cache)
        
        # 在独立线程中发送请求
        def send_request():
            try:
                print(f"\n[心率发送] 数据: {upload_data}")
                headers = {"X-Device-Id": "9ae374d4-f3a4-4896-aa0d-7ce2044163f0"}
                response = requests.post(
                    self.heart_upload_url,
                    json=upload_data,
                    headers=headers,
                    timeout=5
                )
                print(f"[心率响应] 状态码: {response.status_code}, 内容: {response.text}")
            except Exception as e:
                print(f"\n[心率异常] {str(e)}")
        
        threading.Thread(target=send_request, daemon=True).start()
        
        # 立即清空缓存并更新计数
        self.heart_data_cache.clear()
        self.heart_upload_count += count
    
    def _upload_breath_data(self):
        """上传呼吸数据到服务器（异步发送，不等待响应）"""
        if not self.breath_upload_url or len(self.breath_data_cache) == 0:
            return
        
        # 准备上传数据（使用新的JSON格式）
        upload_data = {
            "deviceState": "on_body",
            "data": self.breath_data_cache.copy()
        }
        count = len(self.breath_data_cache)
        
        # 在独立线程中发送请求
        def send_request():
            try:
                print(f"\n[呼吸发送] 数据: {upload_data}")
                headers = {"X-Device-Id": "9ae374d4-f3a4-4896-aa0d-7ce2044163f0"}
                response = requests.post(
                    self.breath_upload_url,
                    json=upload_data,
                    headers=headers,
                    timeout=5
                )
                print(f"[呼吸响应] 状态码: {response.status_code}, 内容: {response.text}")
            except Exception as e:
                print(f"\n[呼吸异常] {str(e)}")
        
        threading.Thread(target=send_request, daemon=True).start()
        
        # 立即清空缓存并更新计数
        self.breath_data_cache.clear()
        self.breath_upload_count += count
    
    def run(self):
        """运行监控线程的主循环"""
        print("结果监控器启动...")
        if self.heart_upload_url:
            print(f"  心率上传URL: {self.heart_upload_url}")
        if self.breath_upload_url:
            print(f"  呼吸上传URL: {self.breath_upload_url}")
        print(f"  心/呼吸上传间隔: {self.hb_upload_interval} 秒")
        if self.action_upload_url:
            print(f"  行为上传URL: {self.action_upload_url}")
            print(f"  行为上传模式: 状态切换立即上报 + 定时上报(间隔{self.action_upload_interval}秒)")
        
        last_print_time = time.time()
        last_hb_upload_time = time.time()
        last_action_upload_time = time.time()
        print_interval = 1.0  # 每秒打印一次
        
        try:
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # 读取队列数据
                self._read_queues()
                
                # 定期打印状态
                if current_time - last_print_time >= print_interval:
                    # self._print_status()
                    last_print_time = current_time
                
                # 定期上传 心率/呼吸 数据
                if current_time - last_hb_upload_time >= self.hb_upload_interval:
                    self._upload_heart_data()
                    self._upload_breath_data()
                    last_hb_upload_time = current_time

                # 模式2：定时上报最新状态
                if self.action_upload_url:
                    if current_time - last_action_upload_time >= self.action_upload_interval:
                        self._upload_action_data_periodic()
                        last_action_upload_time = current_time
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n结果监控器被用户中断")
        finally:
            # 程序结束前上传剩余数据
            print("\n正在上传剩余数据...")
            self._upload_heart_data()
            self._upload_breath_data()
            if self.action_upload_url and self.latest_action is not None:
                self._upload_action_data_periodic()
            print("结果监控器已停止")
    
    def start(self) -> threading.Thread:
        """
        启动监控线程
        
        Returns:
            启动的线程对象
        """
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

    def _upload_action_data_immediate(self, action: int, timestamp: int):
        """模式1：状态切换时立即上报（异步发送，不等待响应）
        
        Args:
            action: 行为状态值 (0=未知, 1=静止, 2=行走, 3=跑步)
            timestamp: 时间戳
        """
        if not self.action_upload_url:
            return

        upload_data = {
            "deviceState": "on_body",
            "data": [{
                "value": int(action),
                "timestamp": str(timestamp) if timestamp is not None else None,
                "source": "imu",
                "confidence": 1.0
            }]
        }

        def send_request():
            try:
                print(f"\n[行为立即上报] 数据: {upload_data}")
                headers = {"X-Device-Id": "9ae374d4-f3a4-4896-aa0d-7ce2044163f0"}
                response = requests.post(
                    self.action_upload_url,
                    json=upload_data,
                    headers=headers,
                    timeout=5
                )
                print(f"[行为响应] 状态码: {response.status_code}, 内容: {response.text}")
            except Exception as e:
                print(f"\n[行为异常] {str(e)}")

        threading.Thread(target=send_request, daemon=True).start()
        self.action_upload_count += 1

    def _upload_action_data_periodic(self):
        """模式2：定时上报最新状态（异步发送，不等待响应）
        
        即使队列为空，也会上报当前最新的行为状态
        """
        if not self.action_upload_url:
            return

        # 如果没有任何行为数据，跳过
        if self.latest_action is None:
            return

        # 使用最新的行为状态和时间戳
        upload_data = {
            "deviceState": "on_body",
            "data": [{
                "value": int(self.latest_action),
                "timestamp": str(self.latest_action_timestamp or int(time.time() * 1000)),
                "source": "imu",
                "confidence": 1.0
            }]
        }

        def send_request():
            try:
                action_text = {0: "未知", 1: "静止", 2: "行走", 3: "跑步"}.get(self.latest_action, "未知")
                print(f"\n[行为定时上报] 当前状态: {action_text}, 数据: {upload_data}")
                headers = {"X-Device-Id": "9ae374d4-f3a4-4896-aa0d-7ce2044163f0"}
                response = requests.post(
                    self.action_upload_url,
                    json=upload_data,
                    headers=headers,
                    timeout=5
                )
                print(f"[行为响应] 状态码: {response.status_code}, 内容: {response.text}")
            except Exception as e:
                print(f"\n[行为异常] {str(e)}")

        threading.Thread(target=send_request, daemon=True).start()
        self.action_upload_count += 1
