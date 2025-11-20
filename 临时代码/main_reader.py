from read import DataReader
from recv_data_reader import RecvDataReader
from plot import DataPlotter
from temp2 import HeartRateDetector  # 假设temp2.py是心率检测模块
from data_logger import DataLogger
import time
import queue
# main.py 中添加监控线程
import threading


if __name__ == "__main__":
    sample_rate = 100  # 采样率100Hz
    logging = True  # 控制是否记录日志，True为记录，False为不记录
    
    # 选择数据来源：'csv' 或 'ble'
    # source_type = 'ble'  # 切换这里选择数据源
    # source_param = 'FA:8B:D4:D0:45:04'  # 蓝牙MAC地址
    # source_type = 'csv'
    # source_param = r"C:\\path\\to\\your\\file.csv"  # CSV文件路径
    # 新增: TCP 读取（作为服务器监听端口，芯片作为客户端连接）
    source_type = 'tcp'
    source_param = '0.0.0.0:9000'  # 监听本机 9000 端口，可改为 '9000'

    
    # 其他配置示例：
    # source_type = 'ble'
    # source_param = 'FA:8B:D4:D0:45:04'
    # 
    # source_type = 'recv'
    # source_param = 'BACC'
    
    # 创建日志队列（如果需要记录）
    log_queue = queue.Queue(maxsize=2000) if logging else None
    
    # 根据数据源类型初始化对应的数据读取器
    if source_type == 'recv':
        # 使用RecvDataReader（基于recv.py的通知方式）
        data_reader = RecvDataReader(source_param, log_queue=log_queue)
        sample_rate = 200
    else:
        # 使用原有的DataReader（支持csv和ble）
        data_reader = DataReader(source_type, source_param, log_queue=log_queue)
    
    data_reader.start()

    # def monitor_queues():
    #     while not data_reader.stop_event.is_set():
    #         print(f"检测队列: {data_reader.data_queue_detect.qsize()}, 绘图队列: {data_reader.data_queue_plot.qsize()}")
    #         time.sleep(1)

    # threading.Thread(target=monitor_queues, daemon=True).start()
    
    # 初始化绘图器
    plotter = DataPlotter(
        data_queue=data_reader.get_data_queue_plot(),
        stop_event=data_reader.stop_event,
        max_points=1000,
        sample_rate=sample_rate,
        source_type=source_type
    )
    
    # 初始化心率检测器（temp2.py功能）
    heart_detector = HeartRateDetector(
        data_queue=data_reader.get_data_queue_detect(),
        stop_event=data_reader.stop_event,
        sample_rate=sample_rate
    )
    heart_thread = heart_detector.start()
    
    # 初始化数据记录器（如果启用logging）
    data_logger = None
    logger_thread = None
    if logging:
        data_logger = DataLogger(
            data_queue=log_queue,
            stop_event=data_reader.stop_event
        )
        logger_thread = data_logger.start()
    
    # 启动绘图
    try:
        plotter.start()
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        data_reader.stop()
        heart_thread.join()
        if logger_thread:
            logger_thread.join()