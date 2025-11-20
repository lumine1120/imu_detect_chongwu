#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio, datetime, sys, struct
from bleak import BleakScanner, BleakClient

CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

# --------------- 工具函数 ---------------

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")

# --------------- 解析函数 ---------------

def parse_combined_packet(data: bytes):
    """
    解析20字节合并数据包

    格式:
    [0-3]   ts_ms_le    (uint32_t) 时间戳，毫秒，小端序
    [4]     btn         (uint8_t)  按钮状态 0/1
    []
    [5-6]   acc_x       (int16_t)  加速度X轴 (mm/s²)
    [7-8]   acc_y       (int16_t)  加速度Y轴 (mm/s²)
    [9-10]  acc_z       (int16_t)  加速度Z轴 (mm/s²)
    [11-12] gyr_x       (int16_t)  陀螺仪X轴 (mrad/s)
    [13-14] gyr_y       (int16_t)  陀螺仪Y轴 (mrad/s)
    [15-16] gyr_z       (int16_t)  陀螺仪Z轴 (mrad/s)
    [17-19] padding     (3 bytes)  保留

    返回: dict 包含解析后的所有字段，或 None 如果格式错误
    """
    if len(data) != 20:
        return None

    try:
        # '<' = 小端序
        # 'I' = uint32_t (4 bytes)
        # 'B' = uint8_t (1 byte)
        # '6h' = 6x int16_t (12 bytes)
        # '3B' = 3x uint8_t (3 bytes padding)
        ts_ms, btn, adc, ax, ay, az, gx, gy, gz, _ = struct.unpack('<IBH6hB', data)

        return {
            'timestamp_ms': ts_ms,
            'button': btn,
            'adc': adc, 
            'accel': {'x': ax, 'y': ay, 'z': az},  # mm/s²
            'gyro': {'x': gx, 'y': gy, 'z': gz}    # mrad/s
        }
    except struct.error:
        return None

def parse_legacy_packet(data: bytes):
    """
    解析旧版3包格式（兼容性，不再推荐使用）

    返回: (kind, ts, payload, desc) 或 None
    """
    if not data or len(data) < 5:
        return None

    pkt_type = data[0]

    # 类型A/B/C: [type(1)] [ts_le(4)] [payload...]
    if pkt_type in (0xA, 0xB, 0xC):
        ts = int.from_bytes(data[1:5], "little")

        if pkt_type == 0xA and len(data) >= 6:
            btn = data[5]
            return ("A", ts, {"btn": btn}, f"type=A ts={ts}ms btn={btn}")

        if pkt_type == 0xB and len(data) >= 11:
            acc = [int.from_bytes(data[5+2*i:7+2*i], "little", signed=True) for i in range(3)]
            return ("B", ts, {"acc_mg": acc}, f"type=B ts={ts}ms acc_mg={acc}")

        if pkt_type == 0xC and len(data) >= 11:
            gyr = [int.from_bytes(data[5+2*i:7+2*i], "little", signed=True) for i in range(3)]
            return ("C", ts, {"gyr_mdps": gyr}, f"type=C ts={ts}ms gyr_mdps={gyr}")

    return None

# --------------- 通知处理 ---------------

last_ts = None  # 用于计算时间间隔

def handle_notify(_, data: bytearray):
    global last_ts

    b = bytes(data)
    hexstr = b.hex()

    # 尝试解析新版20字节合并包
    parsed = parse_combined_packet(b)
    if parsed:
        ts = parsed['timestamp_ms']
        btn = parsed['button']
        adc = parsed['adc']
        acc = parsed['accel']
        gyr = parsed['gyro']

        # 计算时间间隔
        delta_str = ""
        if last_ts is not None:
            delta_ms = ts - last_ts
            delta_str = f" Δt={delta_ms}ms"
        last_ts = ts

        # 打印解析后的数据
        print(f"[{now()}] PACK ts={ts}ms{delta_str} | "
              f"btn={btn} | "
              f"adc={adc} | "
              f"acc=({acc['x']:6d}, {acc['y']:6d}, {acc['z']:6d}) mm/s² | "
              f"gyr=({gyr['x']:6d}, {gyr['y']:6d}, {gyr['z']:6d}) mrad/s")

        # 可选：打印原始十六进制数据（调试用）
        # print(f"  raw={hexstr}")
        return

    # 尝试解析旧版3包格式（向后兼容）
    legacy = parse_legacy_packet(b)
    if legacy:
        kind, ts, payload, desc = legacy
        print(f"[{now()}] LEGACY {desc} | raw={hexstr}")
        print(f"  WARNING: Device using deprecated 3-packet format!")
        return

    # 未知格式
    print(f"[{now()}] UNKNOWN | len={len(b)} raw={hexstr}")

# --------------- BLE 主流程 ---------------

async def main(name: str):
    print(f"[{now()}] Scanning for device '{name}' ...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: d.name and name.lower() in d.name.lower(), timeout=10.0
    )
    if not dev:
        print(f"[{now()}] Device not found")
        return

    print(f"[{now()}] Found {dev.name} [{dev.address}]")

    try:
        async with BleakClient(dev) as client:
            print(f"[{now()}] Connected")

            # 可选：列出服务/特征（调试用）
            # for s in client.services:
            #     print(f"  Service: {s.uuid}")
            #     for c in s.characteristics:
            #         print(f"    Char: {c.uuid} ({','.join(sorted(c.properties))})")

            await client.start_notify(CHAR_UUID, handle_notify)
            print(f"[{now()}] Subscribed to {CHAR_UUID}")
            print(f"[{now()}] Receiving data... (Ctrl+C to stop)\n")

            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n[{now()}] Stopping ...")

    except Exception as e:
        print(f"[{now()}] Connect/subscribe failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recv.py DEVICE_NAME")
        print("Example: python recv.py TOUCH")
        sys.exit(1)

    # Windows兼容性处理
    if sys.platform.startswith("win"):
        import asyncio.windows_events
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(sys.argv[1]))
