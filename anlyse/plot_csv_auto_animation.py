"""
读取CSV文件指定行数范围的数据并以动画方式逐渐绘制加速度幅值图
采样率约200帧/秒，动画会模拟实时绘制效果
纯自动播放版本，无交互控件
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class AccelerationAnimator:
    """加速度数据动画绘制类（自动播放版）"""
    
    def __init__(self, csv_file, start_row, end_row, 
                 fps=30, speed_factor=10):
        """
        初始化动画绘制器
        
        Args:
            csv_file: CSV文件路径
            start_row: 开始行数（从1开始计数，不包括表头）
            end_row: 结束行数（从1开始计数，包含该行）
            fps: 动画帧率（默认30fps）
            speed_factor: 播放速度倍数（默认10倍速，即1秒动画显示10秒数据）
        """
        self.csv_file = csv_file
        self.start_row = start_row
        self.end_row = end_row
        self.fps = fps
        self.speed_factor = speed_factor
        
        # 加载数据
        self._load_data()
        
        # 计算每帧显示的数据点数（假设采样率200Hz）
        self.sampling_rate = 200  # Hz
        self.points_per_frame = int(self.sampling_rate * self.speed_factor / self.fps)
        if self.points_per_frame < 1:
            self.points_per_frame = 1
        
        print(f"动画参数:")
        print(f"  帧率: {self.fps} fps")
        print(f"  播放速度: {self.speed_factor}x")
        print(f"  每帧显示: {self.points_per_frame} 个数据点")
        print(f"  预计动画时长: {len(self.df) / (self.points_per_frame * self.fps):.1f} 秒")
        
    def _load_data(self):
        """加载CSV数据"""
        print(f"读取CSV文件: {self.csv_file}")
        print(f"行数范围: {self.start_row} 到 {self.end_row}")
        
        # 读取CSV文件
        self.df = pd.read_csv(
            self.csv_file,
            skiprows=range(1, self.start_row),
            nrows=self.end_row - self.start_row + 1
        )
        
        print(f"成功读取 {len(self.df)} 行数据")
        print(f"列名: {list(self.df.columns)}")
        
        # 检查必要的列是否存在
        required_columns = ['AccX', 'AccY', 'AccZ']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
        
        # 计算加速度幅值
        self.df['Magnitude'] = np.sqrt(
            self.df['AccX']**2 + self.df['AccY']**2 + self.df['AccZ']**2
        )
        
        # 创建数据点索引
        self.data_points = np.arange(self.start_row, self.end_row + 1)
        
        # 计算统计信息
        self.mean_val = self.df['Magnitude'].mean()
        self.max_val = self.df['Magnitude'].max()
        self.min_val = self.df['Magnitude'].min()
        self.std_val = self.df['Magnitude'].std()
        
        print(f"\n统计信息:")
        print(f"  平均值: {self.mean_val:.4f}")
        print(f"  最大值: {self.max_val:.4f}")
        print(f"  最小值: {self.min_val:.4f}")
        print(f"  标准差: {self.std_val:.4f}")
        
    def _init_plot(self):
        """初始化图表"""
        self.line.set_data([], [])
        self.mean_line.set_data([], [])
        self.progress_text.set_text('')
        return self.line, self.mean_line, self.progress_text
    
    def _update(self, frame):
        """更新动画帧"""
        # 计算当前应显示到第几个数据点
        end_idx = min((frame + 1) * self.points_per_frame, len(self.df))
        
        if end_idx == 0:
            return self.line, self.mean_line, self.progress_text
        
        # 更新数据线
        x_data = self.data_points[:end_idx]
        y_data = self.df['Magnitude'].values[:end_idx]
        self.line.set_data(x_data, y_data)
        
        # 更新平均值线
        self.mean_line.set_data([self.data_points[0], self.data_points[end_idx-1]], 
                                [self.mean_val, self.mean_val])
        
        # 更新进度文本
        progress = (end_idx / len(self.df)) * 100
        elapsed_time = end_idx / self.sampling_rate  # 实际经过的时间（秒）
        self.progress_text.set_text(
            f'进度: {progress:.1f}%\n'
            f'已显示: {end_idx}/{len(self.df)} 点\n'
            f'时间: {elapsed_time:.2f}秒'
        )
        
        return self.line, self.mean_line, self.progress_text
    
    def animate(self):
        """开始动画"""
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        
        # 初始化空线条
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label='加速度幅值')
        self.mean_line, = self.ax.plot([], [], 'r--', linewidth=1, 
                                       label=f'平均值 = {self.mean_val:.4f}', alpha=0.7)
        
        # 设置坐标轴范围
        self.ax.set_xlim(self.data_points[0], self.data_points[-1])
        y_margin = (self.max_val - self.min_val) * 0.1
        self.ax.set_ylim(self.min_val - y_margin, self.max_val + y_margin)
        
        # 设置标签和标题
        self.ax.set_xlabel('数据行数', fontsize=12)
        self.ax.set_ylabel('加速度幅值 √(AccX² + AccY² + AccZ²)', fontsize=12)
        self.ax.set_title(
            f'加速度幅值动画 (行 {self.start_row} 到 {self.end_row}) - {self.speed_factor}x速度', 
            fontsize=14, fontweight='bold'
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=10)
        
        # 添加统计信息文本框
        stats_text = (
            f'数据行数: {self.start_row} - {self.end_row}\n'
            f'总数据点: {len(self.df)}\n'
            f'平均值: {self.mean_val:.4f}\n'
            f'最大值: {self.max_val:.4f}\n'
            f'最小值: {self.min_val:.4f}\n'
            f'标准差: {self.std_val:.4f}\n'
            f'采样率: ~{self.sampling_rate} Hz'
        )
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 添加进度文本
        self.progress_text = self.ax.text(
            0.98, 0.02, '', transform=self.ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
        
        # 如果有datetime列，显示时间范围
        if 'datetime' in self.df.columns:
            start_time = self.df['datetime'].iloc[0]
            end_time = self.df['datetime'].iloc[-1]
            time_text = f'时间范围:\n{start_time}\n至\n{end_time}'
            self.ax.text(0.98, 0.98, time_text, transform=self.ax.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            print(f"\n时间范围:")
            print(f"  开始: {start_time}")
            print(f"  结束: {end_time}")
        
        # 计算总帧数
        total_frames = int(np.ceil(len(self.df) / self.points_per_frame))
        
        # 创建动画
        self.anim = FuncAnimation(
            self.fig, 
            self._update,
            init_func=self._init_plot,
            frames=total_frames,
            interval=1000/self.fps,  # 毫秒
            blit=True,
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def save_animation(self, filename='animation.mp4', dpi=100):
        """
        保存动画为视频文件
        
        Args:
            filename: 输出文件名
            dpi: 视频分辨率
        
        注意: 需要安装 ffmpeg 或 pillow
        """
        print(f"正在保存动画到: {filename}")
        print("这可能需要一些时间...")
        
        self.anim.save(filename, writer='ffmpeg', fps=self.fps, dpi=dpi)
        print(f"动画已保存!")


def plot_csv_acc_animation(csv_file, start_row, end_row, 
                          fps=30, speed_factor=10, save_video=False):
    """
    读取CSV文件指定行数范围并以动画方式绘制加速度幅值
    
    Args:
        csv_file: CSV文件路径
        start_row: 开始行数（从1开始计数，不包括表头）
        end_row: 结束行数（从1开始计数，包含该行）
        fps: 动画帧率（默认30fps）
        speed_factor: 播放速度倍数（默认10倍速）
        save_video: 是否保存为视频文件
    
    Returns:
        处理后的DataFrame
    """
    animator = AccelerationAnimator(csv_file, start_row, end_row, fps, speed_factor)
    df = animator.animate()
    
    if save_video:
        animator.save_animation('acceleration_animation.mp4')
    
    return df


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    # CSV文件路径
    csv_file = "/Users/lumine/Nutstore Files/我的坚果云/chongwu/运行版本1/data/imu_log_20251117_190357_NEW.csv"
    
    # 指定读取的行数范围（从1开始计数，不包括表头）
    start_row = 12000   # 开始行数
    end_row = 13000     # 结束行数
    
    # 动画参数
    fps = 30            # 动画帧率
    speed_factor = 1   # 播放速度倍数（10倍速表示1秒动画显示10秒数据）
    save_video = False  # 是否保存为视频文件
    # =================================================
    
    try:
        # 读取并绘制动画
        df = plot_csv_acc_animation(
            csv_file, 
            start_row, 
            end_row, 
            fps=fps,
            speed_factor=speed_factor,
            save_video=save_video
        )
        
        print("\n动画播放完成!")
        print("提示：")
        print("  - 可以调整 speed_factor 参数来改变播放速度")
        print("  - speed_factor=1 表示实时速度（1秒动画=1秒数据）")
        print("  - speed_factor=10 表示10倍速（1秒动画=10秒数据）")
        print("  - 设置 save_video=True 可以保存为视频文件（需要安装ffmpeg）")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
