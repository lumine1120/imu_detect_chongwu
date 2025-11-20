"""
读取CSV文件指定行数范围的数据并绘制加速度幅值图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_csv_acc_range(csv_file, start_row, end_row):
    """
    读取CSV文件指定行数范围并绘制加速度幅值
    
    Args:
        csv_file: CSV文件路径
        start_row: 开始行数（从1开始计数，不包括表头）
        end_row: 结束行数（从1开始计数，包含该行）
    """
    print(f"读取CSV文件: {csv_file}")
    print(f"行数范围: {start_row} 到 {end_row}")
    
    # 读取CSV文件
    # skiprows跳过表头后的前start_row-1行
    # nrows读取end_row-start_row+1行
    df = pd.read_csv(
        csv_file,
        skiprows=range(1, start_row),  # 跳过表头后的前start_row-1行
        nrows=end_row - start_row + 1   # 读取指定行数
    )
    
    print(f"成功读取 {len(df)} 行数据")
    print(f"列名: {list(df.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['AccX', 'AccY', 'AccZ']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 计算加速度幅值（平方根）
    df['Magnitude'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
    
    # 创建数据点索引
    data_points = np.arange(start_row, end_row + 1)
    
    # 计算统计信息
    mean_val = df['Magnitude'].mean()
    max_val = df['Magnitude'].max()
    min_val = df['Magnitude'].min()
    std_val = df['Magnitude'].std()
    
    print(f"\n统计信息:")
    print(f"  平均值: {mean_val:.4f}")
    print(f"  最大值: {max_val:.4f}")
    print(f"  最小值: {min_val:.4f}")
    print(f"  标准差: {std_val:.4f}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制加速度幅值
    ax.plot(data_points, df['Magnitude'].values, 'b-', linewidth=1.5, label='加速度幅值')
    
    # 绘制平均值线
    ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1, 
               label=f'平均值 = {mean_val:.4f}', alpha=0.7)
    
    # 设置标签和标题
    ax.set_xlabel('数据行数', fontsize=12)
    ax.set_ylabel('加速度幅值 √(AccX² + AccY² + AccZ²)', fontsize=12)
    ax.set_title(f'加速度幅值图 (行 {start_row} 到 {end_row})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 添加统计信息文本框
    stats_text = (
        f'数据行数: {start_row} - {end_row}\n'
        f'总数据点: {len(df)}\n'
        f'平均值: {mean_val:.4f}\n'
        f'最大值: {max_val:.4f}\n'
        f'最小值: {min_val:.4f}\n'
        f'标准差: {std_val:.4f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 如果有datetime列，显示时间范围
    if 'datetime' in df.columns:
        start_time = df['datetime'].iloc[0]
        end_time = df['datetime'].iloc[-1]
        time_text = f'时间范围:\n{start_time}\n至\n{end_time}'
        ax.text(0.98, 0.98, time_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"\n时间范围:")
        print(f"  开始: {start_time}")
        print(f"  结束: {end_time}")
    
    plt.tight_layout()
    plt.show()
    
    return df


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    # CSV文件路径
    csv_file = r"C:\\Users\\mmwave\\Nutstore\\1\\我的坚果云\\chongwu\\运行版本1\\data\\imu_log_20251117_190357_NEW.csv"
    
    # 指定读取的行数范围（从1开始计数，不包括表头）
    start_row = 12000   # 开始行数
    end_row = 13000 # 结束行数
    # =================================================
    
    try:
        # 读取并绘制数据
        df = plot_csv_acc_range(csv_file, start_row, end_row)
        
        # 可选：保存处理后的数据
        # output_file = "processed_data.csv"
        # df.to_csv(output_file, index=False)
        # print(f"\n处理后的数据已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
