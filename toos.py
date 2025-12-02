from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


def clip_csv_by_time(start_time: str, end_time: str, file_path: str, output_path: Optional[str] = None) -> str:
    """
    裁剪指定 CSV 文件在时间范围内的数据并复制到新的 CSV。

    参数:
        start_time: 开始时间字符串，例如 "2025-11-24 14:20:42.200"。
        end_time: 结束时间字符串，例如 "2025-11-24 14:20:42.600"。
        file_path: 源 CSV 文件路径，文件需包含表头且包含名为 "datetime" 的时间列，格式如 "%Y-%m-%d %H:%M:%S.%f" 或无毫秒的 "%Y-%m-%d %H:%M:%S"。
        output_path: 输出 CSV 文件路径；如未提供则在源文件同目录生成形如 "<name>_clip_<start>_<end>.csv" 的文件。

    返回:
        生成的新 CSV 文件的绝对路径字符串。

    行为:
        - 读取源文件首行表头，要求存在 "datetime" 列。
        - 解析每行的 datetime 字段，保留 [start_time, end_time] 闭区间内的所有行。
        - 保留原始列顺序与表头，写入到新的文件。
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")

    # 解析时间，支持带毫秒或不带毫秒
    def parse_ts(s: str) -> datetime:
        s = s.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析时间: {s}")

    ts_start = parse_ts(start_time)
    ts_end = parse_ts(end_time)
    if ts_start > ts_end:
        raise ValueError("开始时间不能晚于结束时间")

    # 输出路径默认规则
    if output_path is None:
        safe_start = start_time.replace(":", "").replace(" ", "_").replace(".", "")
        safe_end = end_time.replace(":", "").replace(" ", "_").replace(".", "")
        out_name = f"{src.stem}_clip_{safe_start}_{safe_end}{src.suffix}"
        out = src.with_name(out_name)
    else:
        out = Path(output_path)
        if out.is_dir():
            safe_start = start_time.replace(":", "").replace(" ", "_").replace(".", "")
            safe_end = end_time.replace(":", "").replace(" ", "_").replace(".", "")
            out = out / f"{src.stem}_clip_{safe_start}_{safe_end}{src.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    # 读取与写入
    count_in = 0
    count_out = 0
    with src.open("r", encoding="utf-8", newline="") as fin, out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("CSV 文件缺少表头")
        if "datetime" not in reader.fieldnames:
            raise ValueError("CSV 文件缺少 'datetime' 列")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            dt_str = (row.get("datetime") or "").strip()
            if not dt_str:
                continue
            try:
                ts = parse_ts(dt_str)
            except ValueError:
                # 跳过不可解析的时间行
                continue
            count_in += 1
            if ts_start <= ts <= ts_end:
                writer.writerow(row)
                count_out += 1

    # 简单的无数据保护：如果输出为空，仍返回路径但提醒调用者
    # 调用方可根据需要决定是否删除空文件
    return str(out.resolve())


if __name__ == "__main__":
    inputFilePath=r"/Users/lumine/code/chongwu/呼吸算法开发/data/imu_log_20251128_152724.csv"
    outputFilePath=r"/Users/lumine/code/chongwu/呼吸算法开发/data/imu_log_20251128_152724_test.csv"
    clip_csv_by_time("2025-11-28 15:36:25.756", "2025-11-28 15:40:40.000", inputFilePath, outputFilePath)
    pass
    # 简易命令行用法示例（可选）：
    # python toos.py "2025-11-24 14:20:42.200" "2025-11-24 14:20:42.600" "data/imu_log_20251124_142040.csv"
    # import sys    
    # if len(sys.argv) >= 4:
    #     start_time_arg = sys.argv[1]
    #     end_time_arg = sys.argv[2]
    #     file_arg = sys.argv[3]
    #     out_arg = sys.argv[4] if len(sys.argv) >= 5 else None
    #     result = clip_csv_by_time(start_time_arg, end_time_arg, file_arg, out_arg)
    #     print(result)
    # else:
    #     print("用法: python toos.py <开始时间> <结束时间> <源CSV路径> [输出目录或文件]")
