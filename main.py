from read import DataReader
from plot import DataPlotter
from temp2 import HeartRateDetector  # 假设temp2.py是心率检测模块
import time
# main.py 中添加监控线程
import threading


if __name__ == "__main__":
    sample_rate = 100  # 采样率100Hz
    # 选择数据来源：'csv' 或 'ble'
    source_type = 'ble'  # 切换这里选择数据源
    source_param = 'FA:8B:D4:D0:45:04'  # 蓝牙MAC地址或CSV文件路径
    # source_type = 'csv'
    # source_param = "E:\A_work\IMU测心率\imu_data_20251110_200532.csv"
    
    # 初始化数据读取器
    data_reader = DataReader(source_type, source_param)
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
    
    # 启动绘图
    try:
        plotter.start()
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        data_reader.stop()
        heart_thread.join()