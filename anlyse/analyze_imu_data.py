"""
IMU数据分析工具
从CSV文件读取IMU数据，计算加速度幅值，并进行信号处理和自相关分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class IMUDataAnalyzer:
    def __init__(self, csv_file, num_rows=None, sample_rate=100):
        """
        初始化IMU数据分析器
        
        Args:
            csv_file: CSV文件路径
            num_rows: 读取的行数，None表示读取全部
            sample_rate: 采样率(Hz)，默认100
        """
        self.csv_file = csv_file
        self.num_rows = num_rows
        self.sample_rate = sample_rate
        self.data = None
        self.acc_magnitude = None
        self.filtered_data = None
        self.autocorr_result = None
        
    def load_data(self):
        """
        从CSV文件加载数据
        """
        print(f"正在读取文件: {self.csv_file}")
        
        # 读取指定行数的数据
        if self.num_rows:
            self.data = pd.read_csv(self.csv_file, nrows=self.num_rows)
            print(f"已读取 {self.num_rows} 行数据")
        else:
            self.data = pd.read_csv(self.csv_file)
            print(f"已读取全部 {len(self.data)} 行数据")
        
        # 检查必要的列是否存在
        required_cols = ['AccX', 'AccY', 'AccZ']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        print(f"数据时间范围: {self.data['datetime'].iloc[0]} 到 {self.data['datetime'].iloc[-1]}")
        
    def calculate_magnitude(self):
        """
        计算加速度幅值：sqrt(AccX^2 + AccY^2 + AccZ^2)
        """
        print("正在计算加速度幅值...")
        self.acc_magnitude = np.sqrt(
            self.data['AccX']**2 + 
            self.data['AccY']**2 + 
            self.data['AccZ']**2
        )
        print(f"幅值范围: {self.acc_magnitude.min():.4f} 到 {self.acc_magnitude.max():.4f}")
        
    def differentiator_filter_double(self, data, h):
        """
        二阶差分近似微分（与temp2.py中的算法一致）
        
        Args:
            data: 输入数据
            h: 采样间隔
        
        Returns:
            滤波后的数据
        """
        length = data.shape[0] - 6
        data_ans = data * 0.0
        res_data = data[3:length+3] * 4.0 + \
                   (data[4:length+4] + data[2:length+2]) - \
                   2.0 * (data[5:length+5] + data[1:length+1]) - \
                   (data[6:length+6] + data[0:length])
        res_data = res_data / 16.0 / h / h
        data_ans[3:-3] = res_data
        return data_ans
    
    def apply_filter(self):
        """
        应用二阶差分滤波器
        """
        print("正在应用二阶差分滤波器...")
        
        # 转换为numpy数组
        data_array = self.acc_magnitude.values
        
        # 应用滤波器
        h = 1.0 / self.sample_rate  # 采样间隔
        self.filtered_data = self.differentiator_filter_double(data_array, h)
        
        print(f"滤波后数据范围: {self.filtered_data.min():.4f} 到 {self.filtered_data.max():.4f}")
        
    def calculate_autocorrelation(self):
        """
        计算自相关（与temp2.py中的算法一致）
        """
        print("正在计算自相关...")
        
        # 使用滤波后的有效数据（去除前后3个点）
        signal = self.filtered_data[3:-3]
        
        # 归一化处理
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            print("警告: 信号全为零，无法计算自相关")
            self.autocorr_result = np.zeros_like(signal)
            return
            
        norm_signal = signal / max_val
        
        # 计算自相关
        n = len(norm_signal)
        mean = np.mean(norm_signal)
        var = np.var(norm_signal)
        
        if var == 0:
            print("警告: 信号方差为零，无法计算自相关")
            self.autocorr_result = np.zeros(n)
            return
            
        autocorr = np.correlate(norm_signal - mean, norm_signal - mean, mode='full') / (var * n)
        self.autocorr_result = autocorr[n-1:]  # 只保留正延迟部分
        
        # 查找峰值（在45-120的延迟范围内，对应50-133 BPM @100Hz）
        start, end = 45, min(120, len(self.autocorr_result) - 1)
        if end > start:
            sub_range = self.autocorr_result[start:end + 1]
            max_value = np.max(sub_range)
            max_index = int(np.argmax(sub_range)) + start
            
            # 根据峰值位置估算心率
            estimated_hr = 60 / ((max_index + 1) / self.sample_rate)
            print(f"自相关峰值: {max_value:.4f} at index {max_index}")
            print(f"估算心率: {estimated_hr:.1f} BPM")
        
    def plot_results(self):
        """
        绘制三个图：原始信号、滤波后信号、自相关
        """
        print("正在生成图表...")
        
        # 创建时间轴
        time_axis = np.arange(len(self.acc_magnitude)) / self.sample_rate
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 图1: 原始加速度幅值
        axes[0].plot(time_axis, self.acc_magnitude, 'b-', linewidth=0.5)
        axes[0].set_title('原始加速度幅值信号', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间 (秒)', fontsize=12)
        axes[0].set_ylabel('加速度幅值 (g)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 图2: 二阶差分滤波后的信号
        axes[1].plot(time_axis, self.filtered_data, 'r-', linewidth=0.5)
        axes[1].set_title('二阶差分滤波后信号', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('时间 (秒)', fontsize=12)
        axes[1].set_ylabel('滤波后幅值', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 图3: 自相关
        lag_axis = np.arange(len(self.autocorr_result))
        axes[2].plot(lag_axis, self.autocorr_result, 'g-', linewidth=1)
        axes[2].set_title('自相关函数', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('延迟 (样本数)', fontsize=12)
        axes[2].set_ylabel('自相关系数', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # 标记心率相关的延迟范围 (45-120样本，对应50-133 BPM @100Hz)
        axes[2].axvline(x=45, color='orange', linestyle='--', alpha=0.5, label='50 BPM')
        axes[2].axvline(x=120, color='orange', linestyle='--', alpha=0.5, label='133 BPM')
        axes[2].legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_file = Path(self.csv_file).stem + '_analysis.png'
        output_path = Path(self.csv_file).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        
        plt.show()
        
    def analyze(self):
        """
        执行完整的分析流程
        """
        print("=" * 60)
        print("IMU数据分析")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算加速度幅值
        self.calculate_magnitude()
        
        # 3. 应用滤波器
        self.apply_filter()
        
        # 4. 计算自相关
        self.calculate_autocorrelation()
        
        # 5. 绘制结果
        self.plot_results()
        
        print("=" * 60)
        print("分析完成！")
        print("=" * 60)


def main():
    """
    主函数：示例用法
    """
    # 配置参数
    csv_file = r"data\imu_log_20251114_235200.csv"  # CSV文件路径
    num_rows = 500  # 读取行数，设置为None读取全部数据
    sample_rate = 200  # 采样率，根据实际情况调整

    # 创建分析器并执行分析
    analyzer = IMUDataAnalyzer(csv_file, num_rows=num_rows, sample_rate=sample_rate)
    analyzer.analyze()


if __name__ == "__main__":
    main()
