#!/usr/bin/env python3
"""
YOLO26n训练结果可视化脚本
绘制epoch、mAP50、mAP50-95的曲线图
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_yolo_results(csv_path):
    """
    读取YOLO训练结果CSV文件并绘制mAP曲线图
    
    Args:
        csv_path: CSV文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find file {csv_path}")
        return False
    
    # 读取数据
    df = pd.read_csv(csv_path)
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
    output_dir = os.path.dirname(csv_path)
    save_path = os.path.join(output_dir, '0.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    
    # 打印统计信息
    print(f"\nTraining Statistics:")
    print(f"Total epochs: {len(epochs)}")
    print(f"mAP50 Max(Epoch {max_map50_epoch}): {max_map50:.4f} {map50_95.iloc[map50.idxmax()]:.4f}")
    print(f"mAP50-95 Max(Epoch {max_map50_95_epoch}): {map50.iloc[map50_95.idxmax()]:.4f} {max_map50_95:.4f}")
    print(f"Final mAP50: {map50.iloc[-1]:.4f}")
    print(f"Final mAP50-95: {map50_95.iloc[-1]:.4f}")
    
    return True

if __name__ == "__main__":
    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_file = os.path.join(project_root, "runs/detect/visdrone/yolo26m-20260411-221639/results.csv")
    # 调用绘图函数
    plot_yolo_results(csv_file)
