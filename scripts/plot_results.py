#!/usr/bin/env python3
"""
YOLO26n训练结果可视化脚本
支持单实验和多实验对比的mAP曲线绘制
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional

def read_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    读取YOLO训练结果CSV文件
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        DataFrame或None（如果文件不存在）
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Cannot find file {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # 验证必要的列是否存在
        required_columns = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns in {csv_path}: {missing_columns}")
            return None
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def plot_single_experiment(csv_path: str, save_dir: Optional[str] = None) -> bool:
    """
    绘制单个实验的mAP曲线图
    
    Args:
        csv_path: CSV文件路径
        save_dir: 保存目录，默认为CSV所在目录
        
    Returns:
        是否成功
    """
    df = read_csv_data(csv_path)
    if df is None:
        return False
    
    epochs = df['epoch']
    map50 = df['metrics/mAP50(B)']
    map50_95 = df['metrics/mAP50-95(B)']
    
    # 找出最大值及对应epoch
    max_map50 = map50.max()
    max_map50_epoch = epochs.iloc[map50.idxmax()]
    max_map50_95 = map50_95.max()
    max_map50_95_epoch = epochs.iloc[map50_95.idxmax()]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制曲线
    plt.plot(epochs, map50, 'b-o', label=f'mAP50 (Max: {max_map50:.4f} @ Epoch {max_map50_epoch})', markersize=4)
    plt.plot(epochs, map50_95, 'r-s', label=f'mAP50-95 (Max: {max_map50_95:.4f} @ Epoch {max_map50_95_epoch})', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.title('YOLO26n Training Results', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '0.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    
    # 打印统计信息
    print(f"\nTraining Statistics:")
    print(f"Total epochs: {len(epochs)}")
    print(f"mAP50 Max(Epoch {max_map50_epoch}): {max_map50:.4f} {map50_95.iloc[map50.idxmax()]:.4f}")
    print(f"mAP50-95 Max(Epoch {max_map50_95_epoch}): {map50.iloc[map50_95.idxmax()]:.4f} {max_map50_95:.4f}")
    print(f"Final mAP50: {map50.iloc[-1]:.4f}")
    print(f"Final mAP50-95: {map50_95.iloc[-1]:.4f}")
    
    plt.close()
    return True

def plot_multiple_experiments(experiments: Dict[str, str], save_dir: str = './comparison', 
                             metric: str = 'mAP50-95') -> bool:
    """
    绘制多个实验的mAP对比曲线图
    
    Args:
        experiments: 字典，键为实验名称，值为CSV文件路径
                    例如: {'Exp1': 'path/to/results1.csv', 'Exp2': 'path/to/results2.csv'}
        save_dir: 图片保存目录
        metric: 要对比的指标，可选 'mAP50' 或 'mAP50-95'
        
    Returns:
        是否成功
    """
    if not experiments:
        print("Error: No experiments provided")
        return False
    
    # 定义颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    plt.figure(figsize=(12, 7))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    success_count = 0
    max_values = {}
    
    for idx, (exp_name, csv_path) in enumerate(experiments.items()):
        df = read_csv_data(csv_path)
        if df is None:
            continue
        
        epochs = df['epoch']
        if metric == 'mAP50':
            values = df['metrics/mAP50(B)']
            metric_label = 'mAP50'
        else:
            values = df['metrics/mAP50-95(B)']
            metric_label = 'mAP50-95'
        
        # 选择颜色和标记
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 找出最大值
        max_val = values.max()
        max_epoch = epochs.iloc[values.idxmax()]
        max_values[exp_name] = (max_val, max_epoch)
        
        # 绘制曲线
        plt.plot(epochs, values, color=color, marker=marker, 
                label=f'{exp_name} (Max: {max_val:.4f} @ Epoch {max_epoch})',
                markersize=4, linewidth=1.5)
        success_count += 1
    
    if success_count == 0:
        print("Error: Failed to load any experiment data")
        plt.close()
        return False
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'Multiple Experiments Comparison - {metric_label}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'comparison_{metric.lower().replace("-", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved to: {save_path}")
    
    # 打印对比统计
    print(f"\n{'='*60}")
    print(f"Experiments Comparison ({metric_label}):")
    print(f"{'='*60}")
    for exp_name, (max_val, max_epoch) in sorted(max_values.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{exp_name:20s}: Max={max_val:.4f} @ Epoch {max_epoch:3d}")
    print(f"{'='*60}")
    
    plt.close()
    return True

if __name__ == "__main__":
    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # ========== 示例1: 单实验绘图 ==========
    # csv_file = os.path.join(project_root, "runs/detect/visdrone/yolo26m-20260412-225419/results.csv")
    # plot_single_experiment(csv_file)
    
    # ========== 示例2: 多实验对比 ==========
    experiments = {
        '26m-ms0.5': os.path.join(project_root, "runs/detect/visdrone/yolo26m-20260411-221639/results.csv"),
        '26m-ms0': os.path.join(project_root, "runs/detect/visdrone/yolo26m-20260412-225419/results.csv"),
        '26m-p2-ms0': os.path.join(project_root, "runs/detect/visdrone/yolo26m-p2-20260412-234046/results.csv"),
        '26m-p2-ms0-1024': os.path.join(project_root, "runs/detect/visdrone/yolo26m-p2-b4-s1024-ms0.0-20260413-163054/results.csv"),
    }
    
    # 对比mAP50-95
    plot_multiple_experiments(experiments, save_dir='./', metric='mAP50-95')