"""
配置文件示例
复制此文件为 config.py 并修改相关配置
"""

# ============ 数据源配置 ============
# CSV 文件路径
CSV_FILE = r"/Users/lumine/code/chongwu/呼吸算法开发/data/imu_log_20251128_152724_test.csv"

# 蓝牙设备地址
BLE_ADDRESS = 'FA:8B:D4:D0:45:04'

# RECV 参数
RECV_PARAM = 'BACC'

# TCP 监听参数
TCP_PARAM = '0.0.0.0:1122'

# ============ 网络上传配置 ============
# 心率数据上传URL（设置为 None 则不上传）
HEART_UPLOAD_URL = "http://your-server.com/api/heart"

# 呼吸数据上传URL（设置为 None 则不上传）
BREATH_UPLOAD_URL = "http://your-server.com/api/breath"

# 上传间隔（秒）
UPLOAD_INTERVAL = 20

# ============ 示例：真实服务器配置 ============
# HEART_UPLOAD_URL = "https://api.example.com/v1/health/heart"
# BREATH_UPLOAD_URL = "https://api.example.com/v1/health/breath"

# ============ 示例：本地测试服务器 ============
# HEART_UPLOAD_URL = "http://localhost:8000/api/heart"
# BREATH_UPLOAD_URL = "http://localhost:8000/api/breath"

# ============ 示例：禁用上传 ============
# HEART_UPLOAD_URL = None
# BREATH_UPLOAD_URL = None
