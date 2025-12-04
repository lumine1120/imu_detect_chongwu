import queue
import threading
import time
import csv
from datetime import datetime
import asyncio
import bleak
from device_model import DeviceModel
import socket


class DataReader:
    def __init__(self, source_type, source_param, max_queue_size=2000, log_queue=None, enable_plot_queue: bool = True):
        """
        初始化数据读取器
        :param source_type: 数据来源类型，'csv'、'ble' 或 'tcp'
        :param source_param: 数据来源参数，CSV文件路径、蓝牙设备MAC地址，或 TCP 监听参数('端口' / 'host:port')
        :param max_queue_size: 数据队列最大长度
        :param log_queue: 日志记录队列（可选，用于数据记录）
        """
        self.source_type = source_type
        self.source_param = source_param
        self.log_queue = log_queue
        self.enable_plot_queue = enable_plot_queue
        self.count_frame = 0
        
        # 维护五个专用队列：检测、绘图、心率、呼吸、行为
        self.data_queue_detect = queue.Queue(maxsize=max_queue_size)  # 供通用检测/算法模块使用
        self.data_queue_plot = queue.Queue(maxsize=max_queue_size) if enable_plot_queue else None   # 供绘图模块（plot.py）使用（可禁用）
        self.data_queue_heart = queue.Queue(maxsize=max_queue_size)  # 供心率检测专用
        self.data_queue_breath = queue.Queue(maxsize=max_queue_size)  # 供呼吸检测专用
        self.data_queue_action = queue.Queue(maxsize=max_queue_size)  # 供行为检测专用

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
        elif self.source_type == 'tcp':
            self.read_thread = threading.Thread(target=self._start_tcp_server, daemon=True)
            self.read_thread.start()
        else:
            raise ValueError("数据源类型必须是 'csv'、'ble' 或 'tcp'")

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
                    # 追加角度列（若存在）
                    try:
                        data['AngX'] = float(row.get('AngX', 0))
                        data['AngY'] = float(row.get('AngY', 0))
                        data['AngZ'] = float(row.get('AngZ', 0))
                    except Exception:
                        data['AngX'] = 0.0
                        data['AngY'] = 0.0
                        data['AngZ'] = 0.0
                    
                    # 同时放入两个专用队列
                    self._distribute_data(data)
                    
                    # 模拟实时数据读取速度
                    time.sleep(0.005) # 0.005
            
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
                # 从设备模型中补充角度（若存在）
                angx = self.ble_device.get('AngX')
                angy = self.ble_device.get('AngY')
                angz = self.ble_device.get('AngZ')
                if angx is None:
                    angx = 0.0
                if angy is None:
                    angy = 0.0
                if angz is None:
                    angz = 0.0
                data['AngX'] = angx
                data['AngY'] = angy
                data['AngZ'] = angz
                
                # 分发到两个专用队列
                self._distribute_data(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"蓝牙数据处理错误: {str(e)}")
                break

        print("蓝牙数据处理线程结束")

    def _distribute_data(self, data):
        """将数据同时分发到所有专用队列，队列满时自动丢弃最旧数据"""
        # 统一并补齐时间戳字段
        try:
            ts = data.get('datetime')
            if not ts or not isinstance(ts, str) or len(ts) < 12:
                data['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except Exception:
            data['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

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
        
        # 分发到行为检测队列
        try:
            self.data_queue_action.put_nowait(data)
        except queue.Full:
            self.data_queue_action.get_nowait()  # 丢弃最旧数据
            self.data_queue_action.put_nowait(data)  # 放入新数据
        
        # 分发到绘图队列
        if self.data_queue_plot is not None:
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

    # ============ TCP 读取（作为服务器等待芯片连接）============
    def _start_tcp_server(self):
        """TCP服务器读取芯片数据。
        支持帧格式：
          3列  AccX,AccY,AccZ
          6列  AccX,AccY,AccZ,GyroX,GyroY,GyroZ
          9列  AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Roll,Pitch,Yaw
        分隔：逗号/空格/Tab 任意。"""
        host = '0.0.0.0'
        port = 1122
        try:
            param = str(self.source_param).strip()
            if ':' in param:
                host_part, port_part = param.split(':', 1)
                host = host_part.strip() or '0.0.0.0'
                port = int(port_part)
            else:
                port = int(param)
        except Exception:
            print(f"TCP 参数解析失败，回退到默认 0.0.0.0:9000 (param={self.source_param})")

        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv_sock.bind((host, port))
            srv_sock.listen(1)
            srv_sock.settimeout(1.0)
            print(f"TCP 服务器监听 {host}:{port}，等待芯片连接…")
        except Exception as e:
            print(f"TCP 服务器启动失败: {e}")
            self.stop_event.set()
            try:
                srv_sock.close()
            except:
                pass
            return

        client_sock = None
        client_file = None
        try:
            while not self.stop_event.is_set():
                if client_sock is None:
                    try:
                        client_sock, addr = srv_sock.accept()
                        client_sock.settimeout(1.0)
                        client_file = client_sock.makefile('r')
                        print(f"芯片已连接: {addr}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"接受连接失败: {e}")
                        time.sleep(0.5)
                        continue

                try:
                    line = client_file.readline()
                    if not line:
                        print("芯片断开连接")
                        client_file.close()
                        client_sock.close()
                        client_file = None
                        client_sock = None
                        continue
                    line = line.strip()
                    self.count_frame += 1
                    parts = [p for p in line.replace('\t', ' ').replace(',', ' ').split(' ') if p]
                    if len(parts) < 3:
                        continue
                    ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
                    gx = gy = gz = None
                    roll = pitch = yaw = None
                    if len(parts) >= 6:
                        gx, gy, gz = float(parts[3]), float(parts[4]), float(parts[5])
                    if len(parts) >= 9:
                        roll, pitch, yaw = float(parts[6]), float(parts[7]), float(parts[8])
                    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    # if gx is not None and roll is not None:
                    #     print(f"count:{self.count_frame} ts:{ts} Acc=({ax:.5f},{ay:.5f},{az:.5f}) Gyro=({gx:.5f},{gy:.5f},{gz:.5f}) RPY=({roll:.2f},{pitch:.2f},{yaw:.2f})")
                    # elif gx is not None:
                    #     print(f"count:{self.count_frame} ts:{ts} Acc=({ax:.5f},{ay:.5f},{az:.5f}) Gyro=({gx:.5f},{gy:.5f},{gz:.5f})")
                    # else:
                    #     print(f"count:{self.count_frame} ts:{ts} Acc=({ax:.5f},{ay:.5f},{az:.5f})")

                    data = {
                        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'AccX': ax,
                        'AccY': ay,
                        'AccZ': az
                    }
                    if gx is not None:
                        data['GyroX'] = gx
                        data['GyroY'] = gy
                        data['GyroZ'] = gz
                    if roll is not None:
                        data['Roll'] = roll
                        data['Pitch'] = pitch
                        data['Yaw'] = yaw
                        # 映射到统一的角度字段供绘图使用
                        data['AngX'] = roll
                        data['AngY'] = pitch
                        data['AngZ'] = yaw
                    else:
                        # 若无姿态信息，提供默认0值，保证绘图通用
                        data.setdefault('AngX', 0.0)
                        data.setdefault('AngY', 0.0)
                        data.setdefault('AngZ', 0.0)
                        # print(data)
                    self._distribute_data(data)
                except ValueError:
                    continue
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"TCP 读取错误: {e}")
                    try:
                        if client_file:
                            client_file.close()
                    except:
                        pass
                    try:
                        if client_sock:
                            client_sock.close()
                    except:
                        pass
                    client_file = None
                    client_sock = None
                    time.sleep(0.2)
                    continue
        finally:
            try:
                if client_file:
                    client_file.close()
            except:
                pass
            try:
                if client_sock:
                    client_sock.close()
            except:
                pass
            try:
                srv_sock.close()
            except:
                pass

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
    
    def get_data_queue_heart(self):
        """获取心率检测专用数据队列"""
        return self.data_queue_heart
    
    def get_data_queue_breath(self):
        """获取呼吸检测专用数据队列"""
        return self.data_queue_breath
    
    def get_data_queue_action(self):
        """获取行为检测专用数据队列"""
        return self.data_queue_action

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