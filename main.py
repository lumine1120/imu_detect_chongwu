from read import DataReader
from recv_data_reader import RecvDataReader
from plot import DataPlotter
from breath_detector import BreathDetector
from heart_detector import HeartRateDetector  # 假设temp2.py是心率检测模块
from data_logger import DataLogger
from result_monitor import ResultMonitor
from action_detector import ActionDetector
import time
import queue
import threading

# 各来源的参数常量
CSV_FILE = r"/Users/lumine/Nutstore Files/我的坚果云/chongwu/tcp/呼吸算法开发/data/labuladuo.csv"
CSV_FILE_ALT = r"/Users/lumine/code/chongwu/呼吸算法开发/data/imu_log_20251128_152724_test.csv"  # 可切换备用
BLE_ADDRESS = 'FA:8B:D4:D0:45:04'
RECV_PARAM = 'BACC'
TCP_PARAM = '0.0.0.0:1122'  # 或端口 '9000'

# 数据上传URL配置（可选，设置为None则不上传）


HEART_UPLOAD_URL = "https://api.qyrix.3drx.top/upload/stats/heartrate"
BREATH_UPLOAD_URL = "https://api.qyrix.3drx.top/upload/stats/breath"
ACTION_UPLOAD_URL = "https://api.qyrix.3drx.top/upload/stats/action"
# HEART_UPLOAD_URL = "http://10.29.176.68:8787/upload/stats/heartrate"
# BREATH_UPLOAD_URL = "http://10.29.176.68:8787/upload/stats/breath"
# ACTION_UPLOAD_URL = "http://10.29.176.68:8787/upload/stats/action"
UPLOAD_INTERVAL = 5  # 上传间隔（秒）

def run(source_type: str, plt_model: int, logging: bool=True ,sample_rate: int=100):
    """
    运行主流程。
    参数:
        source_type: 'csv' | 'ble' | 'recv' | 'tcp'
        plt_model: DataPlotter model (1~5)
        logging: 是否记录数据日志（注意: csv 回放强制关闭日志）
    """
    # sample_rate = 100  # 默认采样率

    # 选择 source_param 常量
    if source_type == 'csv':
        source_param = CSV_FILE  # 使用备用 CSV 文件，可根据需要切换
    elif source_type == 'ble':
        source_param = BLE_ADDRESS
    elif source_type == 'recv':
        source_param = RECV_PARAM
        sample_rate = 200  # recv 模式可设更高采样率
    elif source_type == 'tcp':
        source_param = TCP_PARAM
    else:
        raise ValueError(f"不支持的 source_type: {source_type}")

    # CSV 回放强制不记录日志
    if source_type == 'csv':
        logging = False

    log_queue = queue.Queue(maxsize=2000) if logging else None

    if source_type == 'recv':
        data_reader = RecvDataReader(source_param, log_queue=log_queue)
    else:
        data_reader = DataReader(source_type, source_param, log_queue=log_queue, enable_plot_queue=True)

    data_reader.start()

    plotter = DataPlotter(
        data_queue=data_reader.get_data_queue_plot(),
        stop_event=data_reader.stop_event,
        max_points=500,
        sample_rate=sample_rate,
        source_type=source_type,
        model=plt_model
    )

    heart_detector = HeartRateDetector(
        data_queue=data_reader.get_data_queue_detect(),
        stop_event=data_reader.stop_event,
        sample_rate=sample_rate
    )
    heart_thread = heart_detector.start()

    data_logger = None
    logger_thread = None
    if logging:
        data_logger = DataLogger(
            data_queue=log_queue,
            stop_event=data_reader.stop_event
        )
        logger_thread = data_logger.start()

    try:
        plotter.start()
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        data_reader.stop()
        heart_thread.join()
        if logger_thread:
            logger_thread.join()


def run_breath_detector(source_type: str, logging: bool = True, sample_rate: int = 100):
    """
    使用 BreathDetector 进行呼吸检测与绘图：
    - 仅消费 DataReader 的 data_queue_detect
    - 禁用 DataReader 的绘图队列创建（enable_plot_queue=False）
    - 数据记录逻辑与 run() 一致
    """
    # 选择参数
    if source_type == 'csv':
        source_param = CSV_FILE
        logging = False  # CSV 回放不记录日志
    elif source_type == 'ble':
        source_param = BLE_ADDRESS
    elif source_type == 'recv':
        source_param = RECV_PARAM
        sample_rate = 200
    elif source_type == 'tcp':
        source_param = TCP_PARAM
    else:
        raise ValueError(f"不支持的 source_type: {source_type}")

    log_queue = queue.Queue(maxsize=2000) if logging else None

    if source_type == 'recv':
        data_reader = RecvDataReader(source_param, log_queue=log_queue)
    else:
        data_reader = DataReader(source_type, source_param, log_queue=log_queue, enable_plot_queue=False)

    data_reader.start()

    breath_detector = BreathDetector(
        data_queue=data_reader.get_data_queue_detect(),
        stop_event=data_reader.stop_event,
        sample_rate=sample_rate,
        max_points=500,
        smooth_window=50,
        slope_window=10,
        peak_search_radius=60,
    )

    data_logger = None
    logger_thread = None
    if logging:
        data_logger = DataLogger(
            data_queue=log_queue,
            stop_event=data_reader.stop_event
        )
        logger_thread = data_logger.start()

    try:
        breath_detector.start()  # 阻塞直到窗口关闭或被中断
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        data_reader.stop()
        if logger_thread:
            logger_thread.join()


def run_both_heart_breath(source_type: str, logging: bool = True, sample_rate: int = 100):
    """
    同时检测心率和呼吸：
    - 心率: 基于总加速度 (使用 HeartRateDetector)
    - 呼吸: 基于角度X (使用 BreathDetector)
    - 实时绘制呼吸图表
    - 将检测结果推送到队列并打印
    
    参数:
        source_type: 'csv' | 'ble' | 'recv' | 'tcp'
        logging: 是否记录数据日志
        sample_rate: 采样率
    """
    # 选择参数
    if source_type == 'csv':
        source_param = CSV_FILE
        logging = False  # CSV 回放不记录日志
    elif source_type == 'ble':
        source_param = BLE_ADDRESS
    elif source_type == 'recv':
        source_param = RECV_PARAM
        sample_rate = 200
    elif source_type == 'tcp':
        source_param = TCP_PARAM
    else:
        raise ValueError(f"不支持的 source_type: {source_type}")

    log_queue = queue.Queue(maxsize=2000) if logging else None
    heart_rate_queue = queue.Queue(maxsize=1000)
    breath_rate_queue = queue.Queue(maxsize=1000)
    action_queue = queue.Queue(maxsize=1000)

    # 创建数据读取器
    if source_type == 'recv':
        data_reader = RecvDataReader(source_param, log_queue=log_queue)
    else:
        data_reader = DataReader(source_type, source_param, log_queue=log_queue, enable_plot_queue=False)

    data_reader.start()

    # 创建心率检测器 (后台线程运行，使用独立的心率队列)
    heart_detector = HeartRateDetector(
        data_queue=data_reader.get_data_queue_heart(),
        stop_event=data_reader.stop_event,
        sample_rate=sample_rate,
        heart_rate_queue=heart_rate_queue
    )
    heart_thread = heart_detector.start()

    # 创建呼吸检测器 (主线程运行并绘图，使用独立的呼吸队列)
    breath_detector = BreathDetector(
        data_queue=data_reader.get_data_queue_breath(),
        stop_event=data_reader.stop_event,
        sample_rate=sample_rate,
        max_points=500,
        smooth_window=50,
        slope_window=10,
        peak_search_radius=60,
        breath_rate_queue=breath_rate_queue,
    )

    # 创建行为检测器 (后台线程运行，使用加速度数据)
    action_detector = ActionDetector(
        data_queue=data_reader.get_data_queue_action(),
        stop_event=data_reader.stop_event,
        action_queue=action_queue,
        window_size=200,
        stationary_threshold=(0.8, 1.2),
        walking_threshold=(0.5, 2.0),
        running_threshold=2.0
    )
    action_thread = action_detector.start()

    # 创建结果监控器（带网络上传功能）行为检测）
    result_monitor = ResultMonitor(
        heart_rate_queue=heart_rate_queue,
        breath_rate_queue=breath_rate_queue,
        stop_event=data_reader.stop_event,
        heart_upload_url=HEART_UPLOAD_URL,
        breath_upload_url=BREATH_UPLOAD_URL,
        action_upload_url=ACTION_UPLOAD_URL,
        upload_interval=UPLOAD_INTERVAL,
        action_queue=action_queue
    )
    monitor_thread = result_monitor.start()

    # 创建数据记录器
    data_logger = None
    logger_thread = None
    if logging:
        data_logger = DataLogger(
            data_queue=log_queue,
            stop_event=data_reader.stop_event
        )
        logger_thread = data_logger.start()

    try:
        breath_detector.start()  # 阻塞直到窗口关闭或被中断
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        data_reader.stop()
        heart_thread.join()
        action_thread.join()
        monitor_thread.join()
        if logger_thread:
            logger_thread.join()


if __name__ == "__main__":
    # plt_model 1: 仅总加速度; 2: 总加速度+AngX; 3: 总加速度+AngY; 4: 总加速度+AngZ; 5: 总加速度+AngX+AngY+AngZ
    # 示例调用: CSV 回放 + 四图 (model=5)
    # run(source_type='tcp', plt_model=1, logging=True, sample_rate=100)
    # 示例调用：使用 BreathDetector（CSV 回放，不记录日志）
    # run_breath_detector(source_type='csv', logging=True, sample_rate=100)
    # 示例调用：同时检测心率和呼吸
    run_both_heart_breath(source_type='csv', logging=True, sample_rate=100)

