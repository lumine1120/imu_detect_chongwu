"""
数据日志记录模块
从队列读取数据并保存到CSV文件
"""
import csv
import queue
import threading
import time
import os
from datetime import datetime


class DataLogger:
    def __init__(self, data_queue, stop_event, log_file=None, data_dir='data'):
        """
        初始化数据日志记录器
        
        Args:
            data_queue: 数据队列
            stop_event: 停止信号
            log_file: 日志文件路径（可选，默认自动生成）
            data_dir: 数据存储目录
        """
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.data_dir = data_dir
        self.log_file_handle = None
        self.log_writer = None
        self.log_thread = None
        self.data_count = 0
        
        # 确保data目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")
        
        # 如果未指定文件名，自动生成
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.data_dir, f"imu_log_{timestamp}.csv")
        else:
            self.log_file = os.path.join(self.data_dir, log_file)
    
    def start(self):
        """启动日志记录线程"""
        try:
            # 打开CSV文件并写入表头
            self.log_file_handle = open(self.log_file, 'w', newline='', encoding='utf-8')
            self.log_writer = csv.writer(self.log_file_handle)
            
            # 写入CSV表头
            self.log_writer.writerow([
                'datetime', 'AccX', 'AccY', 'AccZ',
                'AsX', 'AsY', 'AsZ',
                'AngX', 'AngY', 'AngZ',  # 原角度/可复用
                'GyroX', 'GyroY', 'GyroZ',
                'Roll', 'Pitch', 'Yaw'
            ])
            self.log_file_handle.flush()
            
            # 启动日志写入线程
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()
            
            print(f"数据日志记录已启动，文件: {self.log_file}")
            return self.log_thread
            
        except Exception as e:
            print(f"启动日志记录失败: {e}")
            return None
    
    def _log_worker(self):
        """日志写入工作线程"""
        print("日志记录线程启动...")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取数据（超时0.1秒）
                data = self.data_queue.get(timeout=0.1)
                
                # 准备写入的数据行
                log_row = [
                    data.get('datetime', ''),
                    data.get('AccX', ''),
                    data.get('AccY', ''),
                    data.get('AccZ', ''),
                    data.get('AsX', ''),
                    data.get('AsY', ''),
                    data.get('AsZ', ''),
                    data.get('AngX', ''),
                    data.get('AngY', ''),
                    data.get('AngZ', ''),
                    data.get('GyroX', ''),
                    data.get('GyroY', ''),
                    data.get('GyroZ', ''),
                    data.get('Roll', ''),
                    data.get('Pitch', ''),
                    data.get('Yaw', '')
                ]
                
                # 写入CSV文件
                if self.log_writer:
                    self.log_writer.writerow(log_row)
                    self.data_count += 1
                    
                    # 每100条数据刷新一次文件
                    if self.data_count % 100 == 0:
                        self.log_file_handle.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"日志写入错误: {e}")
        
        # 清空剩余队列数据
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                log_row = [
                    data.get('datetime', ''),
                    data.get('AccX', ''),
                    data.get('AccY', ''),
                    data.get('AccZ', ''),
                    data.get('AsX', ''),
                    data.get('AsY', ''),
                    data.get('AsZ', ''),
                    data.get('AngX', ''),
                    data.get('AngY', ''),
                    data.get('AngZ', ''),
                    data.get('GyroX', ''),
                    data.get('GyroY', ''),
                    data.get('GyroZ', ''),
                    data.get('Roll', ''),
                    data.get('Pitch', ''),
                    data.get('Yaw', '')
                ]
                if self.log_writer:
                    self.log_writer.writerow(log_row)
                    self.data_count += 1
            except:
                break
        
        print(f"\n日志记录线程结束，共记录 {self.data_count} 条数据")
        self._close()
    
    def _close(self):
        """关闭日志文件"""
        if self.log_file_handle:
            try:
                self.log_file_handle.flush()
                self.log_file_handle.close()
                print(f"数据日志已保存: {self.log_file}")
            except Exception as e:
                print(f"关闭日志文件错误: {e}")
    
    def stop(self):
        """停止日志记录"""
        if self.log_thread and self.log_thread.is_alive():
            self.log_thread.join(timeout=2.0)


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    test_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 创建日志记录器
    logger = DataLogger(test_queue, stop_event)
    logger_thread = logger.start()
    
    # 生成测试数据
    print("生成测试数据...")
    for i in range(100):
        data = {
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'AccX': round(np.random.randn(), 3),
            'AccY': round(np.random.randn(), 3),
            'AccZ': round(np.random.randn(), 3),
            'AsX': round(np.random.randn() * 10, 3),
            'AsY': round(np.random.randn() * 10, 3),
            'AsZ': round(np.random.randn() * 10, 3),
            'AngX': round(np.random.randn() * 180, 3),
            'AngY': round(np.random.randn() * 180, 3),
            'AngZ': round(np.random.randn() * 180, 3)
        }
        test_queue.put(data)
        time.sleep(0.01)
    
    # 停止日志记录
    print("停止日志记录...")
    stop_event.set()
    logger.stop()
    
    print("测试完成")
