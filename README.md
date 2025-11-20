# 宠物心率监测系统

基于 IMU（加速度传感器）数据的实时心率检测与可视化系统。

## 📁 文件说明

### 🎯 核心文件

- **`main.py`** - 主程序入口，整合数据读取、心率检测、可视化和日志记录等功能
- **`device_model.py`** - 蓝牙设备数据模型，定义设备通信协议和数据解析规则

### 📡 数据读取模块

- **`read.py`** - 通用数据读取器，支持从 CSV 文件或蓝牙设备读取加速度数据
- **`recv.py`** - 蓝牙数据接收基础模块，使用 Bleak 库实现 BLE 通信
- **`recv_data_reader.py`** - 基于 recv.py 的蓝牙数据读取器，兼容 main.py 接口
- **`device_run.py`** - 蓝牙设备扫描和连接管理，包含运动检测功能

### 💓 心率检测模块

- **`detect_heart.py`** - 心率检测模块，通过峰值检测算法从加速度数据计算心率
- **`temp2.py`** - 心率检测模块的另一版本，提供不同的检测算法实现

### 📊 数据可视化模块

- **`plot.py`** - 实时数据可视化，绘制加速度幅值和心率趋势图
- **`plot_csv_range.py`** - 读取 CSV 文件指定行数范围并绘制静态加速度幅值图
- **`plot_csv_range_animation.py`** - 带交互控件的动画版本，支持拖动滑块、播放/暂停/重置
- **`plot_csv_auto_animation.py`** - 纯自动播放动画版本，无交互控件，模拟实时数据流

### 📝 数据处理与分析

- **`data_logger.py`** - 数据日志记录器，从队列读取数据并保存到 CSV 文件
- **`process.py`** - 数据处理工具，提供数据清洗和预处理功能
- **`analyze_imu_data.py`** - IMU 数据分析工具，支持信号处理和自相关分析

### 📂 目录说明

- **`data/`** - 存放 IMU 数据 CSV 文件
- **`录长时间数据/`** - 长时间数据记录相关脚本

## 🚀 快速开始

### 运行主程序
```bash
python main.py
```

### 绘制静态图表
```bash
python plot_csv_range.py
```

### 播放动画（带控件）
```bash
python plot_csv_range_animation.py
```

### 播放自动动画
```bash
python plot_csv_auto_animation.py
```

### 分析 IMU 数据
```bash
python analyze_imu_data.py
```

## 📋 系统要求

- Python 3.7+
- 主要依赖：numpy, pandas, matplotlib, scipy, bleak

## 💡 使用说明

1. **数据来源配置**：在 `main.py` 中可以选择数据源（CSV 文件或蓝牙设备）
2. **采样率设置**：默认采样率为 200Hz，可根据实际设备调整
3. **动画速度调节**：在动画脚本中通过 `speed_factor` 参数调整播放速度
4. **数据分析**：使用 analyze_imu_data.py 对历史数据进行深度分析

## 📊 数据格式

CSV 文件格式：
- `datetime` - 时间戳
- `AccX` - X 轴加速度
- `AccY` - Y 轴加速度
- `AccZ` - Z 轴加速度

## 🔧 版本信息

- 采样率：约 200 帧/秒
- 心率检测范围：40-180 BPM
- 数据窗口：可调节（默认 10 秒）
