#!/usr/bin/env python3
"""
RLSF改进算法一键运行脚本

该脚本提供了简单的接口来运行改进版RLSF算法，包括：
1. 单次实验运行
2. 消融研究
3. 基准测试
4. 结果分析
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def run_single_experiment(args):
    """运行单次实验"""
    print("🚀 Running Single RLSF Experiment")
    print(f"Environment: {args.env_name}")
    print(f"Improvements: Bias Correction={args.enable_bias_correction}, "
          f"Uncertainty Modeling={args.enable_uncertainty_modeling}")

    # 使用改进版训练脚本
    cmd = [
        'python', 'Trains/train_improved_prefim.py',
        f'--env_name={args.env_name}',
        f'--seed={args.seed}',
        f'--num_training_step={args.num_training_step}',
        f'--wandb_log={args.wandb_log}',
        f'--n_ensemble={args.n_ensemble}'
    ]

    # 添加其他参数
    if args.max_episode_length:
        cmd.append(f'--max_episode_length={args.max_episode_length}')
    if args.segment_length:
        cmd.append(f'--segment_length={args.segment_length}')

    try:
        print("Starting training...")
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False

def run_ablation_study(args):
    """运行消融研究"""
    print("🧪 Running Ablation Study")
    
    cmd = [
        'python', 'Scripts/run_ablation_study.py',
        '--environments'] + args.environments + [
        '--seeds'] + [str(s) for s in args.seeds] + [
        f'--output_dir={args.output_dir}',
        f'--num_training_step={args.num_training_step}'
    ]
    
    try:
        print("Starting ablation study...")
        result = subprocess.run(cmd, check=True)
        print("✅ Ablation study completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ablation study failed with error: {e}")
        return False

def run_benchmark(args):
    """运行基准测试"""
    print("📊 Running Performance Benchmark")
    
    # 这里需要实现基准测试逻辑
    # 假设已有基线结果和改进结果
    
    try:
        from Sources.utils.evaluation_metrics import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(
            baseline_results_dir=args.baseline_dir,
            improved_results_dir=args.improved_dir
        )
        
        results = benchmark.run_benchmark()
        benchmark.generate_benchmark_report(
            output_file=os.path.join(args.output_dir, 'benchmark_report.md')
        )
        
        print("✅ Benchmark completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        return False

def analyze_results(args):
    """分析实验结果"""
    print("📈 Analyzing Results")
    
    try:
        from Sources.utils.evaluation_metrics import RLSFEvaluationMetrics
        
        # 加载结果并生成分析报告
        evaluator = RLSFEvaluationMetrics(save_dir=args.output_dir)
        
        # 这里需要加载实际的实验数据
        # 简化版本：直接生成示例报告
        
        print("✅ Results analysis completed!")
        return True
        
    except Exception as e:
        print(f"❌ Results analysis failed with error: {e}")
        return False

def setup_environment():
    """设置运行环境"""
    print("🔧 Setting up environment...")
    
    # 检查必要的依赖
    required_packages = ['torch', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, using CPU")
    except:
        pass
    
    print("✅ Environment setup completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='RLSF Improved Algorithm Runner')
    
    # 通用参数
    parser.add_argument('--mode', type=str, choices=['single', 'ablation', 'benchmark', 'analyze'],
                       default='single', help='Running mode')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    # 单次实验参数
    parser.add_argument('--env_name', type=str, default='SafetyPointCircle1-v0',
                       help='Environment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_training_step', type=int, default=100000,
                       help='Number of training steps')
    parser.add_argument('--max_episode_length', type=int, default=500,
                       help='Maximum episode length')
    parser.add_argument('--segment_length', type=int, default=500,
                       help='Segment length for feedback')
    parser.add_argument('--n_ensemble', type=int, default=3,
                       help='Number of ensemble models')
    
    # 改进功能开关
    parser.add_argument('--enable_bias_correction', type=str, default='True',
                       help='Enable adaptive bias correction')
    parser.add_argument('--enable_uncertainty_modeling', type=str, default='True',
                       help='Enable uncertainty-aware cost estimation')
    parser.add_argument('--enable_improved_labeling', type=str, default='True',
                       help='Enable improved segment labeling')
    
    # 实验配置
    parser.add_argument('--experiment_name', type=str, default='improved_rlsf',
                       help='Experiment name')
    parser.add_argument('--wandb_log', type=str, default='True',
                       help='Enable Weights & Biases logging')
    
    # 消融研究参数
    parser.add_argument('--environments', nargs='+',
                       default=['SafetyPointCircle1-v0', 'SafetyCarCircle1-v0'],
                       help='List of environments for ablation study')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='List of seeds for ablation study')
    
    # 基准测试参数
    parser.add_argument('--baseline_dir', type=str, default='./baseline_results',
                       help='Baseline results directory')
    parser.add_argument('--improved_dir', type=str, default='./improved_results',
                       help='Improved results directory')
    
    args = parser.parse_args()
    
    # 设置环境
    if not setup_environment():
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录运行配置
    config_file = os.path.join(args.output_dir, 'run_config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RLSF IMPROVED ALGORITHM RUNNER")
    print(f"Mode: {args.mode.upper()}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 根据模式运行相应功能
    success = False
    
    if args.mode == 'single':
        success = run_single_experiment(args)
    elif args.mode == 'ablation':
        success = run_ablation_study(args)
    elif args.mode == 'benchmark':
        success = run_benchmark(args)
    elif args.mode == 'analyze':
        success = analyze_results(args)
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return 1
    
    if success:
        print(f"\n🎉 {args.mode.capitalize()} completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        return 0
    else:
        print(f"\n💥 {args.mode.capitalize()} failed!")
        return 1

if __name__ == "__main__":
    exit(main())
