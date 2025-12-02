# coding:UTF-8
"""
数据记录模块 Data Recording Module
实时记录IMU传感器数据到CSV文件
Records IMU sensor data to CSV file in real-time
"""

import csv
import threading
import time
import os
from datetime import datetime
from device_model import DeviceModel


class DataRecorder:
    """
    数据记录器类 Data Recorder Class
    用于实时记录设备数据到CSV文件，定期刷新到磁盘
    Records device data to CSV file in real-time with periodic disk flushing
    """
    
    def __init__(self, device, output_dir="data", flush_interval=5.0, buffer_size=1000,
                 split_by='time', split_interval=3600, max_rows_per_file=360000):
        """
        初始化数据记录器
        
        参数 Parameters:
            device: DeviceModel实例 DeviceModel instance
            output_dir: 输出目录 Output directory
            flush_interval: 刷新到磁盘的时间间隔(秒) Flush interval to disk (seconds)
            buffer_size: 缓冲区大小，达到此大小时也会刷新 Buffer size, flush when reached
            split_by: 文件拆分方式 'time'(按时间) 或 'rows'(按行数) Split method
            split_interval: 按时间拆分时的间隔(秒)，默认3600秒=1小时 Time interval for splitting (seconds)
            max_rows_per_file: 按行数拆分时每个文件的最大行数，默认360000行=1小时 Max rows per file
        """
        self.device = device
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self.buffer_size = buffer_size
        self.is_recording = False
        self.recording_thread = None
        self.csv_file = None
        self.csv_writer = None
        self.file_path = None
        self.write_count = 0
        self.last_flush_time = time.time()
        
        # 文件拆分相关参数 File splitting parameters
        self.split_by = split_by  # 'time' 或 'rows'
        self.split_interval = split_interval  # 时间间隔(秒)
        self.max_rows_per_file = max_rows_per_file  # 每个文件最大行数
        self.current_file_rows = 0  # 当前文件已写入行数
        self.file_start_time = None  # 当前文件开始时间
        self.file_index = 1  # 文件序号
        self.base_filename = None  # 基础文件名
        
        # 创建输出目录 Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def start_recording(self, filename=None):
        """
        开始记录数据 Start recording data
        
        参数 Parameters:
            filename: 自定义文件名，如果为None则使用时间戳 Custom filename, use timestamp if None
        """
        if self.is_recording:
            print("Recording is already in progress!")
            return
        
        # 生成基础文件名 Generate base filename
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_filename = f"imu_data_{timestamp_str}"
        else:
            # 移除扩展名作为基础文件名
            self.base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # 重置文件序号和计数器
        self.file_index = 1
        self.current_file_rows = 0
        self.file_start_time = time.time()
        
        # 打开第一个文件
        self._open_new_file()
        
        self.is_recording = True
        self.write_count = 0
        self.last_flush_time = time.time()
        
        # 启动记录线程 Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        print(f"Started recording to: {self.file_path}")
        print(f"Flush interval: {self.flush_interval}s, Buffer size: {self.buffer_size}")
        print(f"Split method: {self.split_by}, Split interval: {self.split_interval if self.split_by == 'time' else self.max_rows_per_file}")
    
    def _open_new_file(self):
        """
        打开新的CSV文件 Open a new CSV file
        """
        # 如果有旧文件，先关闭
        if self.csv_file:
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            self.csv_file.close()
            print(f"Closed file: {self.file_path} (Total rows: {self.current_file_rows})")
        
        # 生成新文件名
        if self.file_index == 1:
            filename = f"{self.base_filename}.csv"
        else:
            timestamp_str2 = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.base_filename}_part{self.file_index}_{timestamp_str2}.csv"
        
        self.file_path = os.path.join(self.output_dir, filename)
        
        # 打开新文件
        self.csv_file = open(self.file_path, 'w', newline='', encoding='utf-8')
        
        # 定义CSV表头 Define CSV header
        fieldnames = [
            'timestamp', 'datetime',
            'AccX', 'AccY', 'AccZ',
            'AsX', 'AsY', 'AsZ',
            'AngX', 'AngY', 'AngZ',
            'HX', 'HY', 'HZ',
            'Q0', 'Q1', 'Q2', 'Q3'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
        # 刷新表头
        self.csv_file.flush()
        try:
            os.fsync(self.csv_file.fileno())
        except OSError as e:
            print(f"Warning: fsync failed ({e}), but data has been flushed to buffer")
        
        # 重置计数器
        self.current_file_rows = 0
        self.file_start_time = time.time()
        
        print(f"Opened new file: {self.file_path}")
    
    def _should_split_file(self):
        """
        检查是否需要拆分文件 Check if file should be split
        """
        if self.split_by == 'time':
            # 按时间拆分
            elapsed = time.time() - self.file_start_time
            return elapsed >= self.split_interval
        elif self.split_by == 'rows':
            # 按行数拆分
            return self.current_file_rows >= self.max_rows_per_file
        return False
    
    def _recording_loop(self):
        """
        记录循环，从队列中获取数据并写入文件
        Recording loop, get data from queue and write to file
        """
        while self.is_recording:
            try:
                # 从队列获取数据，设置超时避免阻塞
                # Get data from queue with timeout to avoid blocking
                if not self.device.all_data_queue.empty():
                    data = self.device.all_data_queue.get(timeout=0.1)
                    
                    # 添加可读的日期时间字段
                    # Add human-readable datetime field
                    data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    
                    # 检查是否需要拆分文件
                    if self._should_split_file():
                        self.file_index += 1
                        self._open_new_file()
                    
                    # 写入CSV Write to CSV
                    self.csv_writer.writerow(data)
                    self.write_count += 1
                    self.current_file_rows += 1
                    
                    # 检查是否需要刷新到磁盘 Check if need to flush to disk
                    current_time = time.time()
                    time_elapsed = current_time - self.last_flush_time
                    
                    # 条件1: 达到缓冲区大小 Condition 1: buffer size reached
                    # 条件2: 达到时间间隔 Condition 2: time interval reached
                    if self.write_count >= self.buffer_size or time_elapsed >= self.flush_interval:
                        self.csv_file.flush()
                        os.fsync(self.csv_file.fileno())
                        print(f"Flushed {self.write_count} records to disk (elapsed: {time_elapsed:.1f}s, current file rows: {self.current_file_rows})")
                        self.write_count = 0
                        self.last_flush_time = current_time
                else:
                    # 队列为空，短暂休眠 Queue is empty, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error in recording loop: {e}")
                time.sleep(0.1)
    
    def stop_recording(self):
        """
        停止记录数据 Stop recording data
        """
        if not self.is_recording:
            print("No recording in progress!")
            return
        
        self.is_recording = False
        
        # 等待记录线程结束 Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        
        # 最后一次刷新并关闭文件 Final flush and close file
        if self.csv_file:
            # 写入队列中剩余的数据 Write remaining data in queue
            while not self.device.all_data_queue.empty():
                try:
                    data = self.device.all_data_queue.get_nowait()
                    data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self.csv_writer.writerow(data)
                    self.write_count += 1
                except:
                    break
            
            self.csv_file.flush()
            try:
                os.fsync(self.csv_file.fileno())
            except OSError as e:
                print(f"Warning: fsync failed ({e}), but data has been flushed to buffer")
            self.csv_file.close()
            print(f"Recording stopped. Final flush: {self.write_count} records")
            print(f"Data saved to: {self.file_path}")
        
        self.csv_file = None
        self.csv_writer = None
    
    def get_recording_status(self):
        """
        获取记录状态 Get recording status
        
        返回 Returns:
            dict: 包含记录状态信息的字典 Dictionary containing recording status info
        """
        return {
            'is_recording': self.is_recording,
            'file_path': self.file_path if self.is_recording else None,
            'pending_records': self.device.all_data_queue.qsize(),
            'records_since_last_flush': self.write_count,
            'time_since_last_flush': time.time() - self.last_flush_time if self.is_recording else 0
        }


# 示例使用 Example Usage
if __name__ == "__main__":
    import asyncio
    import bleak
    
    async def main():
        # 扫描设备 Scan for devices
        print("Scanning for BLE devices...")
        devices = await bleak.BleakScanner.discover(timeout=5.0)
        
        target_device = None
        for device in devices:
            print(f"Found device: {device.name} - {device.address}")
            # 根据设备名称或地址查找目标设备 Find target device by name or address
            if device.name and "WT901BLE" in device.name:
                target_device = device
                break
        
        if target_device is None:
            print("Target device not found!")
            return
        
        print(f"Connecting to: {target_device.name}")
        
        # 创建设备模型（不传递callback_method，使用默认打印）
        # Create device model (no callback_method, use default print)
        device = DeviceModel("IMU_Sensor", target_device)
        
        # 创建数据记录器 Create data recorder
        # flush_interval=5: 每5秒刷新一次
        # buffer_size=1000: 或者每1000条数据刷新一次
        # split_by='time': 按时间拆分文件
        # split_interval=3600: 每1小时拆分一个新文件 (12小时录制 = 约12个文件)
        # 如果想按行数拆分: split_by='rows', max_rows_per_file=360000 (1小时约360000行)
        recorder = DataRecorder(
            device, 
            output_dir="recorded_data", 
            flush_interval=5.0, 
            buffer_size=1000,
            split_by='time',        # 'time' 或 'rows'
            split_interval=3600,    # 1小时 = 3600秒
            max_rows_per_file=360000  # 备用：按行数拆分时使用
        )
        
        # 开始记录 Start recording
        recorder.start_recording()
        
        # 在后台打开设备 Open device in background
        device_task = asyncio.create_task(device.openDevice())
        
        try:
            # 持续记录直到手动终止 Keep recording until manually stopped
            print("Recording... Press Ctrl+C to stop")
            
            # 等待设备连接建立 Wait for device connection to establish
            max_wait = 30  # 最多等待30秒
            wait_count = 0
            while not device.isOpen and wait_count < max_wait:
                await asyncio.sleep(0.5)
                wait_count += 0.5
            
            if not device.isOpen:
                print("Failed to open device within timeout!")
                return
            
            print("Device connected successfully! Recording in progress...")
            
            # 使用无限循环，只有在设备断开或手动终止时才停止
            # Use infinite loop, only stop when device disconnects or manually terminated
            while device.isOpen:
                await asyncio.sleep(1)
                # 可选：每隔一段时间显示记录状态
                # Optional: display recording status periodically
                if int(time.time()) % 300 == 0:  # 每5分钟显示一次
                    status = recorder.get_recording_status()
                    print(f"Still recording... Pending: {status['pending_records']}, "
                          f"File: {status['file_path']}")
                    await asyncio.sleep(1)  # 避免重复打印
            
            print("Device disconnected!")
            
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            # 停止记录 Stop recording
            recorder.stop_recording()
            
            # 关闭设备 Close device
            device.closeDevice()
            
            # 等待设备任务结束 Wait for device task to finish
            try:
                await asyncio.wait_for(device_task, timeout=5.0)
            except:
                pass
            
            print("Done!")
    
    # 运行主程序 Run main program
    asyncio.run(main())
