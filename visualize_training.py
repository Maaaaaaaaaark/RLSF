#!/usr/bin/env python3
"""
RLSF训练过程可视化工具

该脚本读取训练日志并生成可视化图表，用于分析：
1. 学习曲线（回报和成本）
2. δ值的自适应变化
3. 不确定性统计
4. 违约率控制
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='可视化RLSF训练结果')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='训练日志目录路径')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='输出图表保存目录')
    parser.add_argument('--compare_baseline', action='store_true',
                        help='是否与基线对比')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='基线日志目录路径')
    return parser.parse_args()

def load_training_log(log_dir):
    """加载训练日志数据"""
    data = {}
    
    # 读取回报和成本
    return_file = os.path.join(log_dir, 'eval_return.txt')
    cost_file = os.path.join(log_dir, 'eval_cost.txt')
    
    if os.path.exists(return_file):
        with open(return_file, 'r') as f:
            data['returns'] = [float(line.strip()) for line in f if line.strip()]
    
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            data['costs'] = [float(line.strip()) for line in f if line.strip()]
    
    # 尝试读取改进版特有的指标
    delta_file = os.path.join(log_dir, 'delta_values.txt')
    uncertainty_file = os.path.join(log_dir, 'uncertainty_stats.txt')
    
    if os.path.exists(delta_file):
        with open(delta_file, 'r') as f:
            data['deltas'] = [float(line.strip()) for line in f if line.strip()]
    
    if os.path.exists(uncertainty_file):
        with open(uncertainty_file, 'r') as f:
            data['uncertainties'] = [float(line.strip()) for line in f if line.strip()]
    
    return data

def plot_learning_curves(data, baseline_data=None, output_dir='./visualizations'):
    """绘制学习曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 回报曲线
    if 'returns' in data:
        steps = np.arange(len(data['returns']))
        axes[0].plot(steps, data['returns'], label='Improved RLSF', linewidth=2, alpha=0.8)
        
        if baseline_data and 'returns' in baseline_data:
            baseline_steps = np.arange(len(baseline_data['returns']))
            axes[0].plot(baseline_steps, baseline_data['returns'], 
                        label='Baseline RLSF', linewidth=2, alpha=0.8, linestyle='--')
        
        axes[0].set_xlabel('Evaluation Step', fontsize=12)
        axes[0].set_ylabel('Average Return', fontsize=12)
        axes[0].set_title('Learning Curve - Return', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # 成本曲线
    if 'costs' in data:
        steps = np.arange(len(data['costs']))
        axes[1].plot(steps, data['costs'], label='Improved RLSF', 
                    linewidth=2, alpha=0.8, color='red')
        
        if baseline_data and 'costs' in baseline_data:
            baseline_steps = np.arange(len(baseline_data['costs']))
            axes[1].plot(baseline_steps, baseline_data['costs'], 
                        label='Baseline RLSF', linewidth=2, alpha=0.8, 
                        linestyle='--', color='orange')
        
        # 添加成本阈值线
        axes[1].axhline(y=25, color='black', linestyle=':', 
                       label='Cost Limit', linewidth=1.5)
        
        axes[1].set_xlabel('Evaluation Step', fontsize=12)
        axes[1].set_ylabel('Average Cost', fontsize=12)
        axes[1].set_title('Learning Curve - Cost', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 学习曲线已保存到: {output_path}")
    plt.close()

def plot_delta_dynamics(data, output_dir='./visualizations'):
    """绘制δ值动态变化"""
    if 'deltas' not in data:
        print("⚠️  未找到δ值数据，跳过绘制")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = np.arange(len(data['deltas']))
    deltas = data['deltas']
    
    # 绘制δ值曲线
    ax.plot(steps, deltas, linewidth=2, color='blue', alpha=0.8, label='δ value')
    
    # 添加移动平均线
    window_size = min(50, len(deltas) // 10)
    if window_size > 1:
        moving_avg = np.convolve(deltas, np.ones(window_size)/window_size, mode='valid')
        avg_steps = np.arange(window_size-1, len(deltas))
        ax.plot(avg_steps, moving_avg, linewidth=2, color='red', 
               alpha=0.6, label=f'Moving Average (window={window_size})')
    
    # 添加参考线
    ax.axhline(y=0.1, color='green', linestyle='--', 
              label='Initial δ', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('δ Value', fontsize=12)
    ax.set_title('Adaptive Bias Correction - δ Dynamics', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'delta_dynamics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ δ值动态图已保存到: {output_path}")
    plt.close()

def plot_uncertainty_stats(data, output_dir='./visualizations'):
    """绘制不确定性统计"""
    if 'uncertainties' not in data:
        print("⚠️  未找到不确定性数据，跳过绘制")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = np.arange(len(data['uncertainties']))
    uncertainties = data['uncertainties']
    
    # 不确定性随时间变化
    axes[0].plot(steps, uncertainties, linewidth=2, color='purple', alpha=0.8)
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Mean Uncertainty', fontsize=12)
    axes[0].set_title('Uncertainty Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 不确定性分布（直方图）
    axes[1].hist(uncertainties, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(uncertainties), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(uncertainties):.4f}')
    axes[1].axvline(x=np.median(uncertainties), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(uncertainties):.4f}')
    axes[1].set_xlabel('Uncertainty Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'uncertainty_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 不确定性统计图已保存到: {output_path}")
    plt.close()

def plot_performance_comparison(data, baseline_data, output_dir='./visualizations'):
    """绘制性能对比图"""
    if not baseline_data:
        print("⚠️  未提供基线数据，跳过对比图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算最终性能指标
    metrics = ['Final Return', 'Final Cost', 'Convergence Speed']
    
    improved_values = []
    baseline_values = []
    
    # 最终回报
    if 'returns' in data and 'returns' in baseline_data:
        improved_final_return = np.mean(data['returns'][-10:])
        baseline_final_return = np.mean(baseline_data['returns'][-10:])
        improved_values.append(improved_final_return)
        baseline_values.append(baseline_final_return)
    
    # 最终成本
    if 'costs' in data and 'costs' in baseline_data:
        improved_final_cost = np.mean(data['costs'][-10:])
        baseline_final_cost = np.mean(baseline_data['costs'][-10:])
        # 成本越低越好，所以取负值用于可视化
        improved_values.append(-improved_final_cost)
        baseline_values.append(-baseline_final_cost)
    
    # 收敛速度（达到阈值所需步数的倒数）
    if 'returns' in data and 'returns' in baseline_data:
        threshold = np.mean(baseline_data['returns'][-10:]) * 0.9
        
        improved_convergence = next((i for i, r in enumerate(data['returns']) 
                                    if r >= threshold), len(data['returns']))
        baseline_convergence = next((i for i, r in enumerate(baseline_data['returns']) 
                                    if r >= threshold), len(baseline_data['returns']))
        
        # 收敛越快越好，所以取倒数
        improved_values.append(1000.0 / max(improved_convergence, 1))
        baseline_values.append(1000.0 / max(baseline_convergence, 1))
    
    # 绘制对比柱状图
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, improved_values, width, label='Improved RLSF', 
                   color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline RLSF', 
                   color='orange', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value (Normalized)', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 性能对比图已保存到: {output_path}")
    plt.close()

def generate_summary_report(data, baseline_data=None, output_dir='./visualizations'):
    """生成文本摘要报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("RLSF训练结果摘要报告")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # 改进版性能
    report_lines.append("【改进版RLSF】")
    if 'returns' in data:
        final_return = np.mean(data['returns'][-10:])
        return_std = np.std(data['returns'][-10:])
        report_lines.append(f"  最终回报: {final_return:.2f} ± {return_std:.2f}")
    
    if 'costs' in data:
        final_cost = np.mean(data['costs'][-10:])
        cost_std = np.std(data['costs'][-10:])
        report_lines.append(f"  最终成本: {final_cost:.2f} ± {cost_std:.2f}")
    
    if 'deltas' in data:
        final_delta = np.mean(data['deltas'][-100:])
        delta_std = np.std(data['deltas'][-100:])
        report_lines.append(f"  最终δ值: {final_delta:.4f} ± {delta_std:.4f}")
    
    if 'uncertainties' in data:
        mean_uncertainty = np.mean(data['uncertainties'])
        report_lines.append(f"  平均不确定性: {mean_uncertainty:.4f}")
    
    report_lines.append("")
    
    # 基线性能
    if baseline_data:
        report_lines.append("【基线RLSF】")
        if 'returns' in baseline_data:
            baseline_return = np.mean(baseline_data['returns'][-10:])
            baseline_return_std = np.std(baseline_data['returns'][-10:])
            report_lines.append(f"  最终回报: {baseline_return:.2f} ± {baseline_return_std:.2f}")
        
        if 'costs' in baseline_data:
            baseline_cost = np.mean(baseline_data['costs'][-10:])
            baseline_cost_std = np.std(baseline_data['costs'][-10:])
            report_lines.append(f"  最终成本: {baseline_cost:.2f} ± {baseline_cost_std:.2f}")
        
        report_lines.append("")
        
        # 改进幅度
        report_lines.append("【改进幅度】")
        if 'returns' in data and 'returns' in baseline_data:
            improvement = ((final_return - baseline_return) / abs(baseline_return)) * 100
            report_lines.append(f"  回报提升: {improvement:+.2f}%")
        
        if 'costs' in data and 'costs' in baseline_data:
            cost_reduction = ((baseline_cost - final_cost) / baseline_cost) * 100
            report_lines.append(f"  成本降低: {cost_reduction:+.2f}%")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印到控制台
    print('\n'.join(report_lines))
    print(f"\n✅ 摘要报告已保存到: {report_path}")

def main():
    args = parse_args()
    
    print("=" * 60)
    print("RLSF训练结果可视化工具")
    print("=" * 60)
    
    # 加载数据
    print(f"\n📂 加载训练日志: {args.log_dir}")
    data = load_training_log(args.log_dir)
    
    baseline_data = None
    if args.compare_baseline and args.baseline_dir:
        print(f"📂 加载基线日志: {args.baseline_dir}")
        baseline_data = load_training_log(args.baseline_dir)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成可视化
    print(f"\n📊 生成可视化图表...")
    plot_learning_curves(data, baseline_data, args.output_dir)
    plot_delta_dynamics(data, args.output_dir)
    plot_uncertainty_stats(data, args.output_dir)
    
    if baseline_data:
        plot_performance_comparison(data, baseline_data, args.output_dir)
    
    # 生成摘要报告
    print(f"\n📝 生成摘要报告...")
    generate_summary_report(data, baseline_data, args.output_dir)
    
    print(f"\n✅ 所有可视化已完成！输出目录: {args.output_dir}")

if __name__ == '__main__':
    main()

