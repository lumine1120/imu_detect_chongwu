import queue
import threading
import time
import csv
from datetime import datetime
import asyncio
import bleak
from device_model import DeviceModel


class DataReader:
    def __init__(self, source_type, source_param, max_queue_size=2000, log_queue=None):
        """
        初始化数据读取器
        :param source_type: 数据来源类型，'csv' 或 'ble'
        :param source_param: 数据来源参数，CSV文件路径或蓝牙设备MAC地址
        :param max_queue_size: 数据队列最大长度
        :param log_queue: 日志记录队列（可选，用于数据记录）
        """
        self.source_type = source_type
        self.source_param = source_param
        self.log_queue = log_queue
        
        # 仅维护两个专用队列，移除原self.data_queue
        self.data_queue_detect = queue.Queue(maxsize=max_queue_size)  # 供检测模块（temp2.py）使用
        self.data_queue_plot = queue.Queue(maxsize=max_queue_size)    # 供绘图模块（plot.py）使用

        self.stop_event = threading.Event()
        self.read_thread = None
        self.ble_device = None
        self.ble_process_thread = None

    def start(self):
        """启动数据读取"""
        if self.source_type == 'csv':
            self.read_thread = threading.Thread(target=self._read_from_csv, daemon=True)
            self.read_thread.start()
        elif self.source_type == 'ble':
            self.read_thread = threading.Thread(target=self._setup_ble_connection, daemon=True)
            self.read_thread.start()
        else:
            raise ValueError("数据源类型必须是 'csv' 或 'ble'")

    def _read_from_csv(self):
        """从CSV文件读取数据，同时分发到两个专用队列"""
        try:
            with open(self.source_param, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                print(f"开始从CSV文件读取数据: {self.source_param}")
                
                for row in reader:
                    if self.stop_event.is_set():
                        break
                    
                    # 确保数据格式统一
                    data = {
                        'datetime': row.get('datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]),
                        'AccX': float(row['AccX']),
                        'AccY': float(row['AccY']),
                        'AccZ': float(row['AccZ'])
                    }
                    
                    # 同时放入两个专用队列
                    self._distribute_data(data)
                    
                    # 模拟实时数据读取速度
                    time.sleep(0.005)
            
            print("CSV数据读取完成")
        
        except Exception as e:
            print(f"CSV读取错误: {str(e)}")
        finally:
            self.stop_event.set()

    def _setup_ble_connection(self):
        """建立蓝牙连接并读取数据"""
        async def ble_worker():
            # 查找蓝牙设备
            print(f"开始搜索蓝牙设备: {self.source_param}")
            ble_device = None
            try:
                ble_device = await bleak.BleakScanner.find_device_by_address(
                    self.source_param, 
                    timeout=20.0
                )
            except Exception as e:
                print(f"蓝牙设备搜索失败: {str(e)}")
                return

            if not ble_device:
                print(f"未找到蓝牙设备: {self.source_param}")
                return

            print(f"找到蓝牙设备: {ble_device.name} ({ble_device.address})")
            
            # 创建设备实例
            self.ble_device = DeviceModel(
                deviceName=f"BLE_{ble_device.address}",
                BLEDevice=ble_device,
                callback_method=self._ble_data_callback,
                log_queue=self.log_queue
            )

            # 启动蓝牙数据处理线程
            self.ble_process_thread = threading.Thread(
                target=self._process_ble_data, 
                daemon=True
            )
            self.ble_process_thread.start()

            # 连接设备并保持连接
            try:
                await self.ble_device.openDevice()
            except Exception as e:
                print(f"蓝牙连接错误: {str(e)}")
                self.stop()

        # 运行异步蓝牙任务
        asyncio.run(ble_worker())

    def _process_ble_data(self):
        """处理蓝牙设备数据，添加时间戳并分发到两个队列"""
        print("开始处理蓝牙数据")
        while not self.stop_event.is_set() and self.ble_device:
            try:
                # 从设备的加速度队列获取数据
                acc_data = self.ble_device.acc_data_queue.get(
                    block=True, 
                    timeout=1.0
                )
                
                # 添加时间戳，保持数据格式统一
                data = {
                    'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    'AccX': acc_data['AccX'],
                    'AccY': acc_data['AccY'],
                    'AccZ': acc_data['AccZ']
                }
                
                # 分发到两个专用队列
                self._distribute_data(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"蓝牙数据处理错误: {str(e)}")
                break

        print("蓝牙数据处理线程结束")

    def _distribute_data(self, data):
        """将数据同时分发到两个专用队列（非阻塞模式避免阻塞）"""
        # 分发到检测队列
        try:
            self.data_queue_detect.put(data, block=True, timeout=0.05)
        except queue.Full:
            # 队列满时丢弃最旧数据，确保获取最新数据
            try:
                self.data_queue_detect.get_nowait()
                self.data_queue_detect.put_nowait(data)
            except:
                pass
        
        # 分发到绘图队列
        try:
            self.data_queue_plot.put(data, block=True, timeout=0.05)
        except queue.Full:
            try:
                self.data_queue_plot.get_nowait()
                self.data_queue_plot.put_nowait(data)
            except:
                pass
        
        # # 分发到日志队列（如果存在）
        # if self.log_queue:
        #     try:
        #         self.log_queue.put(data, block=True, timeout=0.05)
        #     except queue.Full:
        #         try:
        #             self.log_queue.get_nowait()
        #             self.log_queue.put_nowait(data)
        #         except:
        #             pass

    def _ble_data_callback(self, device_model):
        """蓝牙设备数据更新回调（可选）"""
        pass

    # 专用队列获取方法
    def get_data_queue_detect(self):
        """获取检测专用数据队列（供temp2.py使用）"""
        return self.data_queue_detect

    def get_data_queue_plot(self):
        """获取绘图专用数据队列（供plot.py使用）"""
        return self.data_queue_plot

    def stop(self):
        """停止数据读取并释放资源"""
        self.stop_event.set()
        
        # 关闭蓝牙设备
        if self.ble_device:
            self.ble_device.closeDevice()
        
        # 等待线程结束
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        if self.ble_process_thread and self.ble_process_thread.is_alive():
            self.ble_process_thread.join(timeout=2.0)
        
        # 清空两个队列释放资源
        self._clear_queues()
        
        print("数据读取已停止")

    def _clear_queues(self):
        """清空所有专用队列"""
        for q in [self.data_queue_detect, self.data_queue_plot]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python read.py [source_type] [source_param]")
        print("示例: python read.py csv data.csv")
        print("示例: python read.py ble FA:8B:D4:D0:45:04")
        sys.exit(1)
    
    reader = DataReader(sys.argv[1], sys.argv[2])
    reader.start()
    
    try:
        # 测试队列数据同步性
        def test_queue_sync():
            """测试两个队列的数据一致性"""
            count = 0
            while not reader.stop_event.is_set():
                time.sleep(2)
                count += 1
                if count % 3 == 0:
                    print(f"\n队列状态 - 检测队列: {reader.data_queue_detect.qsize()}, "
                          f"绘图队列: {reader.data_queue_plot.qsize()}")
                    
                    # 验证数据一致性
                    try:
                        data_detect = reader.data_queue_detect.get_nowait()
                        data_plot = reader.data_queue_plot.get_nowait()
                        
                        if (data_detect['AccX'] == data_plot['AccX'] and
                            data_detect['AccY'] == data_plot['AccY'] and
                            data_detect['AccZ'] == data_plot['AccZ']):
                            print(f"数据一致性验证通过: AccX={data_detect['AccX']}")
                        else:
                            print("数据一致性验证失败!")
                        
                        # 放回数据（不影响正常使用）
                        reader.data_queue_detect.put_nowait(data_detect)
                        reader.data_queue_plot.put_nowait(data_plot)
                    except queue.Empty:
                        print("队列暂时无数据")
                    except Exception as e:
                        print(f"队列测试错误: {str(e)}")
        
        test_thread = threading.Thread(target=test_queue_sync, daemon=True)
        test_thread.start()
        
        while not reader.stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        reader.stop()