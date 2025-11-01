#!/usr/bin/env python3
"""
RLSF改进算法消融研究脚本

该脚本系统性地评估各个改进组件的贡献：
1. 基线RLSF
2. 仅偏差校正
3. 仅不确定性建模  
4. 完整改进版本
"""

import os
import sys
import subprocess
import json
import numpy as np
from datetime import datetime
import argparse

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AblationStudy:
    """消融研究管理器"""
    
    def __init__(self, base_config, environments, seeds, output_dir):
        self.base_config = base_config
        self.environments = environments
        self.seeds = seeds
        self.output_dir = output_dir
        self.results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def define_experiment_variants(self):
        """定义实验变体"""
        variants = {
            'baseline': {
                'enable_bias_correction': 'False',
                'enable_uncertainty_modeling': 'False',
                'enable_improved_labeling': 'False',
                'experiment_name': 'baseline_rlsf'
            },
            'bias_correction_only': {
                'enable_bias_correction': 'True',
                'enable_uncertainty_modeling': 'False',
                'enable_improved_labeling': 'False',
                'experiment_name': 'bias_correction_only'
            },
            'uncertainty_only': {
                'enable_bias_correction': 'False',
                'enable_uncertainty_modeling': 'True',
                'enable_improved_labeling': 'False',
                'experiment_name': 'uncertainty_only'
            },
            'improved_labeling_only': {
                'enable_bias_correction': 'False',
                'enable_uncertainty_modeling': 'False',
                'enable_improved_labeling': 'True',
                'experiment_name': 'improved_labeling_only'
            },
            'bias_correction_uncertainty': {
                'enable_bias_correction': 'True',
                'enable_uncertainty_modeling': 'True',
                'enable_improved_labeling': 'False',
                'experiment_name': 'bias_correction_uncertainty'
            },
            'full_improved': {
                'enable_bias_correction': 'True',
                'enable_uncertainty_modeling': 'True',
                'enable_improved_labeling': 'True',
                'experiment_name': 'full_improved'
            }
        }
        return variants
    
    def run_single_experiment(self, variant_name, variant_config, env_name, seed):
        """运行单个实验"""
        print(f"Running {variant_name} on {env_name} with seed {seed}")
        
        # 构建命令行参数
        cmd = [
            'python', 'Trains/train_prefim.py',
            f'--env_name={env_name}',
            f'--seed={seed}',
            f'--wandb_log=True',
            f'--project_name=RLSF_Ablation_Study',
            f'--run_name={variant_name}_{env_name}_seed{seed}'
        ]
        
        # 添加基础配置
        for key, value in self.base_config.items():
            cmd.append(f'--{key}={value}')
        
        # 添加变体特定配置
        for key, value in variant_config.items():
            cmd.append(f'--{key}={value}')

            
        # --- 在这里添加 ---
        # 确保评估间隔小于总步数，例如设为总步数接近结束时
        # 从 base_config 获取 num_training_step 的值
        num_steps = int(self.base_config.get('num_training_step', 50000)) 
        # 设置评估间隔略小于总步数，例如 90% 的位置，确保至少评估一次
        eval_interval_for_ablation = int(num_steps * 0.9) 
        # 或者设置一个足够小的固定值，比如 10000
        # eval_interval_for_ablation = 10000 
        cmd.append(f'--eval_interval={eval_interval_for_ablation}')

        
        # 设置输出目录
        output_subdir = os.path.join(self.output_dir, variant_name, env_name, f'seed_{seed}')
        os.makedirs(output_subdir, exist_ok=True)
        cmd.append(f'--weight_path={output_subdir}')
        
        try:
            # 运行实验
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            
            if result.returncode == 0:
                print(f"✓ {variant_name} on {env_name} seed {seed} completed successfully")
                return True, output_subdir
            else:
                print(f"✗ {variant_name} on {env_name} seed {seed} failed")
                print(f"Error: {result.stderr}")
                return False, None
                
        except subprocess.TimeoutExpired:
            print(f"✗ {variant_name} on {env_name} seed {seed} timed out")
            return False, None
        except Exception as e:
            print(f"✗ {variant_name} on {env_name} seed {seed} error: {e}")
            return False, None
    
    def collect_results(self, variant_name, env_name, seed, output_dir):
        """收集实验结果"""
        try:
            # 查找结果文件
            metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # 存储结果
                key = f"{variant_name}_{env_name}"
                if key not in self.results:
                    self.results[key] = []
                
                self.results[key].append({
                    'seed': seed,
                    'metrics': metrics,
                    'output_dir': output_dir
                })
                
                return True
            else:
                print(f"Warning: No metrics file found for {variant_name}_{env_name}_seed{seed}")
                return False
                
        except Exception as e:
            print(f"Error collecting results for {variant_name}_{env_name}_seed{seed}: {e}")
            return False
    
    def run_full_study(self):
        """运行完整的消融研究"""
        variants = self.define_experiment_variants()
        
        print("Starting Ablation Study")
        print(f"Variants: {list(variants.keys())}")
        print(f"Environments: {self.environments}")
        print(f"Seeds: {self.seeds}")
        print(f"Total experiments: {len(variants) * len(self.environments) * len(self.seeds)}")
        
        start_time = datetime.now()
        
        for variant_name, variant_config in variants.items():
            print(f"\n--- Running variant: {variant_name} ---")
            
            for env_name in self.environments:
                for seed in self.seeds:
                    success, output_dir = self.run_single_experiment(
                        variant_name, variant_config, env_name, seed
                    )
                    
                    if success and output_dir:
                        self.collect_results(variant_name, env_name, seed, output_dir)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nAblation study completed in {duration}")
        print(f"Results collected for {len(self.results)} variant-environment combinations")
        
        # 保存原始结果
        results_file = os.path.join(self.output_dir, 'raw_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results
    
    def analyze_results(self):
        """分析消融研究结果"""
        if not self.results:
            print("No results to analyze")
            return
        
        analysis = {}
        
        for key, experiments in self.results.items():
            variant_name, env_name = key.split('_', 1)
            
            if len(experiments) == 0:
                continue
            
            # 计算统计量
            metrics_list = [exp['metrics'] for exp in experiments]
            
            # 提取关键指标
            key_metrics = ['final_reward', 'final_cost', 'convergence_episode', 
                          'safety_performance_ratio']
            
            variant_analysis = {}
            for metric in key_metrics:
                values = []
                for metrics in metrics_list:
                    if 'sample_efficiency' in metrics and metric in metrics['sample_efficiency']:
                        values.append(metrics['sample_efficiency'][metric])
                
                if values:
                    variant_analysis[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            analysis[key] = variant_analysis
        
        # 保存分析结果
        analysis_file = os.path.join(self.output_dir, 'ablation_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # 生成对比报告
        self.generate_comparison_report(analysis)
        
        return analysis
    
    def generate_comparison_report(self, analysis):
        """生成对比报告"""
        report_file = os.path.join(self.output_dir, 'ablation_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# RLSF Ablation Study Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Configuration\n")
            f.write(f"- Environments: {', '.join(self.environments)}\n")
            f.write(f"- Seeds: {self.seeds}\n")
            f.write(f"- Variants: {len(self.define_experiment_variants())}\n\n")
            
            f.write("## Results Summary\n\n")
            
            # 按环境分组显示结果
            for env_name in self.environments:
                f.write(f"### {env_name}\n\n")
                f.write("| Variant | Final Reward | Final Cost | Convergence Episode | Safety-Performance Ratio |\n")
                f.write("|---------|--------------|------------|--------------------|--------------------------|\n")
                
                for variant_name in self.define_experiment_variants().keys():
                    key = f"{variant_name}_{env_name}"
                    if key in analysis:
                        metrics = analysis[key]
                        reward = metrics.get('final_reward', {}).get('mean', 'N/A')
                        cost = metrics.get('final_cost', {}).get('mean', 'N/A')
                        convergence = metrics.get('convergence_episode', {}).get('mean', 'N/A')
                        ratio = metrics.get('safety_performance_ratio', {}).get('mean', 'N/A')
                        
                        f.write(f"| {variant_name} | {reward:.2f} | {cost:.2f} | {convergence:.0f} | {ratio:.2f} |\n")
                
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("1. **Bias Correction Impact**: [To be filled based on results]\n")
            f.write("2. **Uncertainty Modeling Impact**: [To be filled based on results]\n")
            f.write("3. **Improved Labeling Impact**: [To be filled based on results]\n")
            f.write("4. **Combined Effects**: [To be filled based on results]\n\n")
        
        print(f"Ablation report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Run RLSF Ablation Study')
    parser.add_argument('--environments', nargs='+', 
                       default=['SafetyPointCircle1-v0', 'SafetyCarCircle1-v0'],
                       help='List of environments to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='List of random seeds')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Output directory for results')
    parser.add_argument('--num_training_step', type=int, default=50000,
                       help='Number of training steps per experiment')
    
    args = parser.parse_args()
    
    # 基础配置
    base_config = {
        'max_episode_length': 500,
        'segment_length': 500,
        'num_training_step': args.num_training_step,
        'gamma': 0.99,
        'cost_gamma': 0.99,
        'n_ensemble': 3,
        'batch_size': 4096,
        'lr_clfs': 0.001,
        'enable_detailed_logging': 'True'
    }
    
    # 创建消融研究实例
    study = AblationStudy(
        base_config=base_config,
        environments=args.environments,
        seeds=args.seeds,
        output_dir=args.output_dir
    )
    
    # 运行研究
    results = study.run_full_study()
    
    # 分析结果
    analysis = study.analyze_results()
    
    print(f"\nAblation study completed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
