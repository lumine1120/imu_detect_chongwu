# 蓝牙设备切换使用说明

本系统现在支持连接两种不同类型的蓝牙设备，可以灵活切换数据源。

## 支持的数据源类型

### 1. CSV文件 (`source_type = 'csv'`)
从CSV文件读取历史数据，用于离线分析和测试。

**配置示例：**
```python
source_type = 'csv'
source_param = "E:\A_work\IMU测心率\imu_data_20251110_200532.csv"
```

### 2. BLE设备 - DeviceModel方式 (`source_type = 'ble'`)
使用原有的`DeviceModel`类连接蓝牙设备，通过MAC地址连接。

**特点：**
- 基于BleakScanner的MAC地址查找
- 使用DeviceModel封装
- 适合已知MAC地址的设备

**配置示例：**
```python
source_type = 'ble'
source_param = 'FA:8B:D4:D0:45:04'  # 蓝牙MAC地址
```

### 3. RECV设备 - 通知方式 (`source_type = 'recv'`)
**新增！** 基于`recv.py`的通知方式连接蓝牙设备，通过设备名称连接。

**特点：**
- 基于设备名称模糊匹配查找
- 使用BLE Notify特性接收数据
- 解析20字节合并数据包格式
- 支持加速度和陀螺仪数据（本系统只使用加速度）
- 适合TOUCH等recv.py支持的设备

**配置示例：**
```python
source_type = 'recv'
source_param = 'TOUCH'  # 蓝牙设备名称
```

## 在main.py中切换数据源

只需修改`main.py`中的两个参数：

```python
# ============ 数据源配置 ============
source_type = 'recv'     # 选择：'csv', 'ble', 或 'recv'
source_param = 'TOUCH'   # 根据source_type填写对应参数
```

### 示例1：使用CSV文件
```python
source_type = 'csv'
source_param = "E:\A_work\IMU测心率\imu_data_20251110_200532.csv"
```

### 示例2：使用原有BLE设备（DeviceModel方式）
```python
source_type = 'ble'
source_param = 'FA:8B:D4:D0:45:04'
```

### 示例3：使用RECV蓝牙设备（新增方式）
```python
source_type = 'recv'
source_param = 'TOUCH'
```

## 技术细节

### RecvDataReader特点
1. **通知机制**：使用BLE Notify特性实时接收数据
2. **数据解析**：自动解析20字节合并包格式
3. **单位转换**：将mm/s²转换为g单位，与其他数据源保持一致
4. **接口兼容**：完全兼容原DataReader接口，无需修改其他模块

### 数据包格式
RecvDataReader解析的20字节数据包格式：
```
[0-3]   timestamp_ms  (uint32_t) 时间戳（毫秒）
[4]     button        (uint8_t)  按钮状态
[5-6]   adc           (uint16_t) ADC值
[7-12]  accel_xyz     (3×int16_t) 加速度 (mm/s²)
[13-18] gyro_xyz      (3×int16_t) 陀螺仪 (mrad/s)
[19]    padding       (1 byte)   保留字节
```

### 单位转换
- **原始数据**：mm/s²（毫米每秒平方）
- **转换后**：g（重力加速度单位）
- **转换公式**：g = mm/s² / 9800

## 独立测试

### 测试RecvDataReader
```powershell
python recv_data_reader.py TOUCH
```

### 测试原有DataReader
```powershell
# CSV模式
python read.py csv "E:\A_work\IMU测心率\imu_data_20251110_200532.csv"

# BLE模式
python read.py ble FA:8B:D4:D0:45:04
```

## 完整运行流程

1. **确认设备类型**：确定要连接的蓝牙设备类型
   - 如果是通过MAC地址连接的设备 → 使用 `'ble'`
   - 如果是通过设备名称连接的recv类型设备 → 使用 `'recv'`

2. **修改main.py配置**：
   ```python
   source_type = 'recv'
   source_param = 'TOUCH'
   ```

3. **运行程序**：
   ```powershell
   python main.py
   ```

4. **观察输出**：
   - 设备扫描和连接信息
   - 数据接收统计
   - 心率检测结果
   - 实时绘图

## 故障排查

### 找不到设备
```
未找到设备: TOUCH
```
**解决方法：**
1. 确认设备已开启蓝牙
2. 检查设备名称是否正确（支持部分匹配，不区分大小写）
3. 增加扫描超时时间（在RecvDataReader中修改timeout参数）

### 连接失败
```
连接或订阅失败: ...
```
**解决方法：**
1. 确认设备未被其他程序占用
2. 检查设备是否支持特征UUID: `12345678-1234-5678-1234-56789abcdef1`
3. 重启蓝牙设备

### 数据包解析错误
```
警告: 数据包长度不正确
```
**解决方法：**
1. 确认设备发送的是20字节合并数据包格式
2. 检查设备固件版本是否与recv.py兼容
3. 查看原始十六进制数据（在RecvDataReader中启用调试输出）

## 文件说明

- **recv_data_reader.py**：新创建的RecvDataReader类，基于recv.py实现
- **read.py**：原有的DataReader类，支持CSV和BLE（DeviceModel方式）
- **main.py**：主程序，已更新支持三种数据源切换
- **recv.py**：原始蓝牙接收脚本（参考用）

## 注意事项

1. **Windows兼容性**：程序已处理Windows下的asyncio事件循环策略
2. **队列大小**：默认最大2000，可根据需要调整
3. **数据丢失**：队列满时会自动丢弃最旧数据，确保实时性
4. **日志记录**：设置`logging=True`可记录所有数据到CSV文件
