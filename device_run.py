import asyncio
import bleak
import device_model
from motion_detector import MotionDetector

# 扫描到的设备 Scanned devices
devices = []
# 蓝牙设备 BLEDevice
BLEDevice = None
# 运动检测器 Motion detector
motion_detector = None


# 扫描蓝牙设备并过滤名称
# Scan Bluetooth devices and filter names
async def scan():
    global devices
    global BLEDevice
    find = []
    print("Searching for Bluetooth devices......")
    try:
        devices = await bleak.BleakScanner.discover(timeout=20.0)
        print("Search ended")
        for d in devices:
            if d.name is not None and "WT" in d.name:
                find.append(d)
                print(d)
        if len(find) == 0:
            print("No devices found in this search!")
        else:
            # user_input = input("Please enter the Mac address you want to connect to (e.g. DF:E9:1F:2C:BD:59)：")
            user_input = "FA:8B:D4:D0:45:04"
            for d in devices:
                if d.address == user_input:
                    BLEDevice = d
                    break
    except Exception as ex:
        print("Bluetooth search failed to start")
        print(ex)


# 指定MAC地址搜索并连接设备
# Specify MAC address to search and connect devices
async def scanByMac(device_mac):
    global BLEDevice
    print("Searching for Bluetooth devices......")
    BLEDevice = await bleak.BleakScanner.find_device_by_address(device_mac, timeout=20)


# 数据更新时会调用此方法 This method will be called when data is updated
def updateData(DeviceModel):
    # 直接打印出设备数据字典 Directly print out the device data dictionary
    print(DeviceModel.deviceData)


# 状态变化回调函数 State change callback function
def on_motion_state_change(state, avg_acc, recent_data):
    """当运动状态改变时调用 Called when motion state changes"""
    print(f"\n[State Change Detected]")
    print(f"  New State: {state}")
    print(f"  Average Acceleration: {avg_acc:.3f} g")


if __name__ == '__main__':
    # 方式一：广播搜索和连接蓝牙设备
    # Method 1:Broadcast search and connect Bluetooth devices
    asyncio.run(scan())

    # # 方式二：指定MAC地址搜索并连接设备
    # # Method 2: Specify MAC address to search and connect devices
    # asyncio.run(scanByMac("C6:46:21:41:0B:BD"))

    if BLEDevice is not None:
        # 创建设备 Create device
        device = device_model.DeviceModel("MyBle5.0", BLEDevice, updateData)
        
        # 创建运动检测器并关联设备的队列
        # Create motion detector and link it with device's queue
        motion_detector = MotionDetector(device.acc_data_queue)
        motion_detector.set_state_change_callback(on_motion_state_change)
        motion_detector.start()
        
        # 开始连接设备 Start connecting devices
        try:
            asyncio.run(device.openDevice())
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            # 停止运动检测器 Stop motion detector
            if motion_detector:
                motion_detector.stop()
            print("Program terminated")
    else:
        print("This BLEDevice was not found!!")
