"""
基于recv.py的蓝牙数据读取器
使用通知方式从蓝牙设备获取ACC数据，兼容main.py的DataReader接口
支持recv.py的20字节合并数据包格式
"""

import asyncio
import queue
import threading
import time
import struct
from datetime import datetime
from bleak import BleakScanner, BleakClient


class RecvDataReader:
    """
    基于recv.py的蓝牙数据读取器
    通过BLE通知方式获取加速度数据
    """
    
    # 蓝牙特征UUID（与recv.py保持一致）
    CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
    
    def __init__(self, device_name, max_queue_size=2000, log_queue=None):
        """
        初始化RecvDataReader
        
        Args:
            device_name: 蓝牙设备名称（例如："TOUCH"）
            max_queue_size: 数据队列最大长度
            log_queue: 日志记录队列（可选，用于数据记录）
        """
        self.device_name = device_name
        self.log_queue = log_queue
        self.max_queue_size = max_queue_size
        
        # 创建四个专用队列，与read.py保持一致
        self.data_queue_detect = queue.Queue(maxsize=max_queue_size)  # 供通用检测模块使用
        self.data_queue_plot = queue.Queue(maxsize=max_queue_size)    # 供绘图模块使用
        self.data_queue_heart = queue.Queue(maxsize=max_queue_size)   # 供心率检测专用
        self.data_queue_breath = queue.Queue(maxsize=max_queue_size)  # 供呼吸检测专用
        
        self.stop_event = threading.Event()
        self.read_thread = None
        self.ble_client = None
        self.ble_device = None
        
        # 用于统计
        self.packet_count = 0
        self.last_timestamp = None
    
    def start(self):
        """启动数据读取"""
        self.read_thread = threading.Thread(target=self._ble_worker, daemon=True)
        self.read_thread.start()
        print(f"RecvDataReader已启动，目标设备: {self.device_name}")
    
    def _ble_worker(self):
        """蓝牙工作线程（运行异步事件循环）"""
        # Windows兼容性处理
        import sys
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 创建新的事件循环并运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_ble_main())
        except Exception as e:
            print(f"蓝牙连接错误: {e}")
        finally:
            loop.close()
    
    async def _async_ble_main(self):
        """异步蓝牙主流程"""
        print(f"正在扫描设备 '{self.device_name}' ...")
        
        # 查找设备
        try:
            self.ble_device = await BleakScanner.find_device_by_filter(
                lambda d, adv: d.name and self.device_name.lower() in d.name.lower(),
                timeout=10.0
            )
        except Exception as e:
            print(f"设备扫描失败: {e}")
            self.stop_event.set()
            return
        
        if not self.ble_device:
            print(f"未找到设备: {self.device_name}")
            self.stop_event.set()
            return
        
        print(f"找到设备: {self.ble_device.name} [{self.ble_device.address}]")
        
        # 连接设备
        try:
            async with BleakClient(self.ble_device) as client:
                self.ble_client = client
                print(f"已连接到 {self.ble_device.name}")
                
                # 订阅通知
                await client.start_notify(self.CHAR_UUID, self._notification_handler)
                print(f"已订阅特征: {self.CHAR_UUID}")
                print("正在接收数据... (等待停止信号)")
                
                # 保持连接直到收到停止信号
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                
                # 取消订阅
                await client.stop_notify(self.CHAR_UUID)
                print("已取消通知订阅")
                
        except Exception as e:
            print(f"连接或订阅失败: {e}")
            self.stop_event.set()
        finally:
            self.ble_client = None
            print("蓝牙连接已关闭")
    
    def _notification_handler(self, sender, data: bytearray):
        """
        蓝牙通知处理回调函数
        解析20字节合并数据包并分发到队列
        
        Args:
            sender: 发送方特征
            data: 接收到的字节数据
        """
        if self.stop_event.is_set():
            return
        
        b = bytes(data)
        
        # 解析合并数据包
        parsed = self._parse_combined_packet(b)
        
        if parsed:
            # 提取加速度数据（mm/s² -> g，转换为与其他DataReader一致的单位）
            # 1g ≈ 9800 mm/s²
            acc_x = parsed['accel']['x'] / 980.0
            acc_y = parsed['accel']['y'] / 980.0
            acc_z = parsed['accel']['z'] / 980.0

            # print(f"已经收到数据 acc_x {acc_x:.3f}g, acc_y {acc_y:.3f}g, acc_z {acc_z:.3f}g")
            
            # 构造数据字典，与read.py保持一致的格式
            data_dict = {
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'AccX': acc_x,
                'AccY': acc_y,
                'AccZ': acc_z
            }
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            
            # 分发数据到两个专用队列
            self._distribute_data(data_dict)
            
            # 统计
            self.packet_count += 1
            
            # 可选：打印调试信息（每100个数据包打印一次）
            if self.packet_count % 2000 == 0:
                print(f"已接收 {self.packet_count} 个数据包 | "
                      f"最新数据: AccX={acc_x:.3f}g, AccY={acc_y:.3f}g, AccZ={acc_z:.3f}g")
        else:
            # 解析失败，打印警告
            if len(b) != 20:
                print(f"警告: 数据包长度不正确 (期望20字节，实际{len(b)}字节)")
    
    def _parse_combined_packet(self, data: bytes):
        """
        解析20字节合并数据包（与recv.py保持一致）
        
        格式:
        [0-3]   ts_ms       (uint32_t) 时间戳，毫秒，小端序
        [4]     btn         (uint8_t)  按钮状态 0/1
        [5-6]   adc         (uint16_t) ADC值
        [7-8]   acc_x       (int16_t)  加速度X轴 (mm/s²)
        [9-10]  acc_y       (int16_t)  加速度Y轴 (mm/s²)
        [11-12] acc_z       (int16_t)  加速度Z轴 (mm/s²)
        [13-14] gyr_x       (int16_t)  陀螺仪X轴 (mrad/s)
        [15-16] gyr_y       (int16_t)  陀螺仪Y轴 (mrad/s)
        [17-18] gyr_z       (int16_t)  陀螺仪Z轴 (mrad/s)
        [19]    padding     (1 byte)   保留
        
        Returns:
            dict: 包含解析后的所有字段，或 None 如果格式错误
        """
        if len(data) != 20:
            return None
        
        try:
            # 解包：'<' = 小端序
            # 'I' = uint32_t (4 bytes)
            # 'B' = uint8_t (1 byte)
            # 'H' = uint16_t (2 bytes)
            # '6h' = 6x int16_t (12 bytes)
            # 'B' = uint8_t (1 byte padding)
            ts_ms, btn, adc, ax, ay, az, gx, gy, gz, _ = struct.unpack('<IBH6hB', data)
            
            return {
                'timestamp_ms': ts_ms,
                'button': btn,
                'adc': adc,
                'accel': {'x': ax, 'y': ay, 'z': az},  # mm/s²
                'gyro': {'x': gx, 'y': gy, 'z': gz}    # mrad/s
            }
        except struct.error as e:
            print(f"数据包解析错误: {e}")
            return None
    
    def _distribute_data(self, data):
        """
        将数据同时分发到所有专用队列（与read.py保持一致）
        队列满时自动丢弃最旧的数据，保证队列始终接收最新数据
        
        Args:
            data: 数据字典，包含datetime, AccX, AccY, AccZ
        """
        # 分发到检测队列
        try:
            self.data_queue_detect.put_nowait(data)
        except queue.Full:
            self.data_queue_detect.get_nowait()  # 丢弃最旧数据
            self.data_queue_detect.put_nowait(data)  # 放入新数据
        
        # 分发到心率检测队列
        try:
            self.data_queue_heart.put_nowait(data)
        except queue.Full:
            self.data_queue_heart.get_nowait()  # 丢弃最旧数据
            self.data_queue_heart.put_nowait(data)  # 放入新数据
        
        # 分发到呼吸检测队列
        try:
            self.data_queue_breath.put_nowait(data)
        except queue.Full:
            self.data_queue_breath.get_nowait()  # 丢弃最旧数据
            self.data_queue_breath.put_nowait(data)  # 放入新数据
        
        # 分发到绘图队列
        try:
            self.data_queue_plot.put_nowait(data)
        except queue.Full:
            self.data_queue_plot.get_nowait()  # 丢弃最旧数据
            self.data_queue_plot.put_nowait(data)  # 放入新数据
        
        # 分发到日志队列（如果存在）
        if self.log_queue:
            try:
                self.log_queue.put_nowait(data)
            except queue.Full:
                self.log_queue.get_nowait()  # 丢弃最旧数据
                self.log_queue.put_nowait(data)  # 放入新数据
    
    def get_data_queue_detect(self):
        """获取检测专用数据队列（供temp2.py使用）"""
        return self.data_queue_detect
    
    def get_data_queue_plot(self):
        """获取绘图专用数据队列（供plot.py使用）"""
        return self.data_queue_plot
    
    def get_data_queue_heart(self):
        """获取心率检测专用数据队列"""
        return self.data_queue_heart
    
    def get_data_queue_breath(self):
        """获取呼吸检测专用数据队列"""
        return self.data_queue_breath
    
    def stop(self):
        """停止数据读取并释放资源"""
        print("正在停止RecvDataReader...")
        self.stop_event.set()
        
        # 等待线程结束
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3.0)
        
        # 清空队列
        self._clear_queues()
        
        print(f"RecvDataReader已停止，共接收 {self.packet_count} 个数据包")
    
    def _clear_queues(self):
        """清空所有专用队列"""
        for q in [self.data_queue_detect, self.data_queue_plot]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break


if __name__ == "__main__":
    """测试代码"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python recv_data_reader.py DEVICE_NAME")
        print("示例: python recv_data_reader.py TOUCH")
        sys.exit(1)
    
    device_name = sys.argv[1]
    reader = RecvDataReader(device_name)
    reader.start()
    
    # 监控队列状态
    def monitor_queues():
        while not reader.stop_event.is_set():
            time.sleep(3)
            print(f"队列状态 - 检测队列: {reader.data_queue_detect.qsize()}, "
                  f"绘图队列: {reader.data_queue_plot.qsize()}")
            
            # 从队列读取并显示数据示例
            try:
                data = reader.data_queue_detect.get(timeout=0.1)
                print(f"数据示例: {data['datetime']} | "
                      f"AccX={data['AccX']:.3f}g, "
                      f"AccY={data['AccY']:.3f}g, "
                      f"AccZ={data['AccZ']:.3f}g")
                # 放回队列
                reader.data_queue_detect.put_nowait(data)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"队列读取错误: {e}")
    
    monitor_thread = threading.Thread(target=monitor_queues, daemon=True)
    monitor_thread.start()
    
    try:
        while not reader.stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        reader.stop()
