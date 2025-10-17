import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional

class RLSFEvaluationMetrics:
    """
    RLSF改进算法的评估指标计算器
    
    用于量化分析自适应偏差校正和不确定性建模的效果
    """
    
    def __init__(self, save_dir: str = "./evaluation_results"):
        self.save_dir = save_dir
        self.metrics_history = defaultdict(list)
        self.episode_data = []
        
    def update_episode_metrics(self, episode_data: Dict):
        """
        更新单个episode的指标
        
        Args:
            episode_data: 包含episode信息的字典
                - reward: episode总回报
                - cost: episode总成本
                - violation_rate: 违约率
                - predicted_cost: 预测成本
                - true_cost: 真实成本
                - uncertainty: 平均不确定性
                - confidence: 平均置信度
                - delta: 当前偏差校正参数
        """
        self.episode_data.append(episode_data)
        
        # 更新历史指标
        for key, value in episode_data.items():
            self.metrics_history[key].append(value)
    
    def compute_bias_correction_effectiveness(self, window_size: int = 100) -> Dict:
        """
        计算偏差校正的有效性指标
        
        Returns:
            effectiveness_metrics: 偏差校正效果指标
        """
        if len(self.episode_data) < window_size:
            return {}
        
        recent_data = self.episode_data[-window_size:]
        
        # 计算预测偏差
        predicted_costs = [d.get('predicted_cost', 0) for d in recent_data]
        true_costs = [d.get('true_cost', 0) for d in recent_data]
        
        if not predicted_costs or not true_costs:
            return {}
        
        bias = np.mean(predicted_costs) - np.mean(true_costs)
        bias_variance = np.var(np.array(predicted_costs) - np.array(true_costs))
        
        # 计算偏差校正的稳定性
        deltas = [d.get('delta', 0) for d in recent_data]
        delta_stability = 1.0 / (1.0 + np.std(deltas))  # 标准差越小，稳定性越高
        
        # 计算违约率控制效果
        violation_rates = [d.get('violation_rate', 0) for d in recent_data]
        target_violation_rate = 0.05
        violation_control_error = abs(np.mean(violation_rates) - target_violation_rate)
        
        return {
            'bias_estimate': bias,
            'bias_variance': bias_variance,
            'delta_stability': delta_stability,
            'violation_control_error': violation_control_error,
            'mean_delta': np.mean(deltas),
            'delta_range': np.max(deltas) - np.min(deltas)
        }
    
    def compute_uncertainty_modeling_effectiveness(self, window_size: int = 100) -> Dict:
        """
        计算不确定性建模的有效性指标
        
        Returns:
            uncertainty_metrics: 不确定性建模效果指标
        """
        if len(self.episode_data) < window_size:
            return {}
        
        recent_data = self.episode_data[-window_size:]
        
        # 提取不确定性相关数据
        uncertainties = [d.get('uncertainty', 0) for d in recent_data]
        confidences = [d.get('confidence', 1) for d in recent_data]
        predicted_costs = [d.get('predicted_cost', 0) for d in recent_data]
        true_costs = [d.get('true_cost', 0) for d in recent_data]
        
        if not uncertainties or not confidences:
            return {}
        
        # 计算不确定性校准度（uncertainty calibration）
        # 高不确定性的预测应该有更高的错误率
        uncertainty_bins = np.linspace(0, 1, 11)
        calibration_error = 0.0
        
        for i in range(len(uncertainty_bins) - 1):
            bin_mask = (np.array(uncertainties) >= uncertainty_bins[i]) & \
                      (np.array(uncertainties) < uncertainty_bins[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_uncertainty = np.mean(np.array(uncertainties)[bin_mask])
                bin_errors = np.abs(np.array(predicted_costs)[bin_mask] - 
                                  np.array(true_costs)[bin_mask])
                bin_error_rate = np.mean(bin_errors)
                
                # 理想情况下，不确定性应该与错误率正相关
                calibration_error += abs(bin_uncertainty - bin_error_rate)
        
        calibration_error /= (len(uncertainty_bins) - 1)
        
        # 计算置信度与准确性的相关性
        confidence_accuracy_corr = np.corrcoef(
            confidences, 
            1.0 - np.abs(np.array(predicted_costs) - np.array(true_costs))
        )[0, 1] if len(confidences) > 1 else 0.0
        
        return {
            'mean_uncertainty': np.mean(uncertainties),
            'uncertainty_variance': np.var(uncertainties),
            'mean_confidence': np.mean(confidences),
            'calibration_error': calibration_error,
            'confidence_accuracy_correlation': confidence_accuracy_corr,
            'high_uncertainty_ratio': np.mean(np.array(uncertainties) > 0.5)
        }
    
    def compute_sample_efficiency_metrics(self) -> Dict:
        """
        计算样本效率指标
        
        Returns:
            efficiency_metrics: 样本效率指标
        """
        if len(self.episode_data) < 100:
            return {}
        
        rewards = self.metrics_history['reward']
        costs = self.metrics_history['cost']
        
        # 计算收敛速度（达到90%最终性能所需的episode数）
        final_reward = np.mean(rewards[-50:])  # 最后50个episode的平均回报
        target_reward = 0.9 * final_reward
        
        convergence_episode = len(rewards)
        for i, reward in enumerate(rewards):
            if reward >= target_reward:
                convergence_episode = i
                break
        
        # 计算学习曲线的平滑度（方差）
        reward_smoothness = 1.0 / (1.0 + np.var(rewards))
        
        # 计算安全性-性能权衡
        safety_performance_ratio = np.mean(rewards) / (np.mean(costs) + 1e-6)
        
        return {
            'convergence_episode': convergence_episode,
            'convergence_ratio': convergence_episode / len(rewards),
            'reward_smoothness': reward_smoothness,
            'safety_performance_ratio': safety_performance_ratio,
            'final_reward': final_reward,
            'final_cost': np.mean(costs[-50:])
        }
    
    def generate_comparison_report(self, baseline_metrics: Dict, 
                                 improved_metrics: Dict) -> Dict:
        """
        生成改进前后的对比报告
        
        Args:
            baseline_metrics: 基线方法的指标
            improved_metrics: 改进方法的指标
            
        Returns:
            comparison_report: 对比报告
        """
        report = {}
        
        for category in ['bias_correction', 'uncertainty_modeling', 'sample_efficiency']:
            if category in baseline_metrics and category in improved_metrics:
                baseline_cat = baseline_metrics[category]
                improved_cat = improved_metrics[category]
                
                category_comparison = {}
                for metric_name in baseline_cat.keys():
                    if metric_name in improved_cat:
                        baseline_val = baseline_cat[metric_name]
                        improved_val = improved_cat[metric_name]
                        
                        # 计算相对改进
                        if baseline_val != 0:
                            relative_improvement = (improved_val - baseline_val) / abs(baseline_val)
                        else:
                            relative_improvement = float('inf') if improved_val > 0 else 0
                        
                        category_comparison[metric_name] = {
                            'baseline': baseline_val,
                            'improved': improved_val,
                            'absolute_improvement': improved_val - baseline_val,
                            'relative_improvement': relative_improvement
                        }
                
                report[category] = category_comparison
        
        return report
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        绘制训练曲线
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 回报曲线
        axes[0, 0].plot(self.metrics_history['reward'])
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 成本曲线
        axes[0, 1].plot(self.metrics_history['cost'])
        axes[0, 1].set_title('Episode Cost')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cost')
        
        # 违约率曲线
        if 'violation_rate' in self.metrics_history:
            axes[0, 2].plot(self.metrics_history['violation_rate'])
            axes[0, 2].axhline(y=0.05, color='r', linestyle='--', label='Target')
            axes[0, 2].set_title('Violation Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Violation Rate')
            axes[0, 2].legend()
        
        # 偏差校正参数
        if 'delta' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['delta'])
            axes[1, 0].set_title('Bias Correction Parameter (δ)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('δ')
        
        # 不确定性
        if 'uncertainty' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['uncertainty'])
            axes[1, 1].set_title('Average Uncertainty')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Uncertainty')
        
        # 置信度
        if 'confidence' in self.metrics_history:
            axes[1, 2].plot(self.metrics_history['confidence'])
            axes[1, 2].set_title('Average Confidence')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_metrics(self, filename: str = "evaluation_metrics.json"):
        """
        保存评估指标到文件
        """
        metrics_summary = {
            'bias_correction': self.compute_bias_correction_effectiveness(),
            'uncertainty_modeling': self.compute_uncertainty_modeling_effectiveness(),
            'sample_efficiency': self.compute_sample_efficiency_metrics(),
            'episode_count': len(self.episode_data)
        }
        
        with open(f"{self.save_dir}/{filename}", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        return metrics_summary

class PerformanceBenchmark:
    """
    性能基准测试工具

    用于系统性地比较不同RLSF变体的性能
    """

    def __init__(self, baseline_results_dir: str, improved_results_dir: str):
        self.baseline_dir = baseline_results_dir
        self.improved_dir = improved_results_dir
        self.benchmark_results = {}

    def load_experiment_results(self, results_dir: str) -> Dict:
        """加载实验结果"""
        results = {}

        if not os.path.exists(results_dir):
            print(f"Warning: Results directory {results_dir} does not exist")
            return results

        # 遍历所有实验子目录
        for env_dir in os.listdir(results_dir):
            env_path = os.path.join(results_dir, env_dir)
            if not os.path.isdir(env_path):
                continue

            env_results = []
            for seed_dir in os.listdir(env_path):
                seed_path = os.path.join(env_path, seed_dir)
                metrics_file = os.path.join(seed_path, 'evaluation_metrics.json')

                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        env_results.append(metrics)
                    except Exception as e:
                        print(f"Error loading {metrics_file}: {e}")

            if env_results:
                results[env_dir] = env_results

        return results

    def compute_statistical_significance(self, baseline_values: List[float],
                                       improved_values: List[float]) -> Dict:
        """计算统计显著性"""
        from scipy import stats

        if len(baseline_values) < 2 or len(improved_values) < 2:
            return {'significant': False, 'p_value': 1.0, 'test': 'insufficient_data'}

        # 使用Welch's t-test（不假设等方差）
        t_stat, p_value = stats.ttest_ind(improved_values, baseline_values, equal_var=False)

        # 计算效应大小（Cohen's d）
        pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) +
                             (len(improved_values) - 1) * np.var(improved_values, ddof=1)) /
                            (len(baseline_values) + len(improved_values) - 2))

        cohens_d = (np.mean(improved_values) - np.mean(baseline_values)) / pooled_std if pooled_std > 0 else 0

        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
            'test': 'welch_t_test'
        }

    def run_benchmark(self) -> Dict:
        """运行完整的基准测试"""
        print("Loading baseline results...")
        baseline_results = self.load_experiment_results(self.baseline_dir)

        print("Loading improved results...")
        improved_results = self.load_experiment_results(self.improved_dir)

        # 找到共同的环境
        common_envs = set(baseline_results.keys()) & set(improved_results.keys())

        if not common_envs:
            print("No common environments found between baseline and improved results")
            return {}

        print(f"Benchmarking on environments: {list(common_envs)}")

        benchmark_results = {}

        for env_name in common_envs:
            print(f"\nBenchmarking {env_name}...")

            baseline_env_results = baseline_results[env_name]
            improved_env_results = improved_results[env_name]

            env_benchmark = self.benchmark_single_environment(
                baseline_env_results, improved_env_results, env_name
            )

            benchmark_results[env_name] = env_benchmark

        self.benchmark_results = benchmark_results
        return benchmark_results

    def benchmark_single_environment(self, baseline_results: List[Dict],
                                   improved_results: List[Dict],
                                   env_name: str) -> Dict:
        """对单个环境进行基准测试"""

        # 定义要比较的关键指标
        key_metrics = {
            'final_reward': ('sample_efficiency', 'final_reward'),
            'final_cost': ('sample_efficiency', 'final_cost'),
            'convergence_episode': ('sample_efficiency', 'convergence_episode'),
            'safety_performance_ratio': ('sample_efficiency', 'safety_performance_ratio'),
            'bias_estimate': ('bias_correction', 'bias_estimate'),
            'mean_uncertainty': ('uncertainty_modeling', 'mean_uncertainty'),
            'calibration_error': ('uncertainty_modeling', 'calibration_error')
        }

        env_benchmark = {}

        for metric_name, (category, metric_key) in key_metrics.items():
            # 提取基线值
            baseline_values = []
            for result in baseline_results:
                if category in result and metric_key in result[category]:
                    baseline_values.append(result[category][metric_key])

            # 提取改进值
            improved_values = []
            for result in improved_results:
                if category in result and metric_key in result[category]:
                    improved_values.append(result[category][metric_key])

            if not baseline_values or not improved_values:
                continue

            # 计算基本统计量
            baseline_stats = {
                'mean': np.mean(baseline_values),
                'std': np.std(baseline_values),
                'count': len(baseline_values)
            }

            improved_stats = {
                'mean': np.mean(improved_values),
                'std': np.std(improved_values),
                'count': len(improved_values)
            }

            # 计算改进程度
            if baseline_stats['mean'] != 0:
                relative_improvement = (improved_stats['mean'] - baseline_stats['mean']) / abs(baseline_stats['mean'])
            else:
                relative_improvement = float('inf') if improved_stats['mean'] > 0 else 0

            # 计算统计显著性
            significance = self.compute_statistical_significance(baseline_values, improved_values)

            env_benchmark[metric_name] = {
                'baseline': baseline_stats,
                'improved': improved_stats,
                'absolute_improvement': improved_stats['mean'] - baseline_stats['mean'],
                'relative_improvement': relative_improvement,
                'significance': significance
            }

        return env_benchmark

    def generate_benchmark_report(self, output_file: str = "benchmark_report.md"):
        """生成基准测试报告"""
        if not self.benchmark_results:
            print("No benchmark results to report")
            return

        with open(output_file, 'w') as f:
            f.write("# RLSF Performance Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")

            # 计算总体改进统计
            all_improvements = []
            significant_improvements = 0
            total_comparisons = 0

            for env_name, env_results in self.benchmark_results.items():
                for metric_name, metric_results in env_results.items():
                    if 'relative_improvement' in metric_results:
                        all_improvements.append(metric_results['relative_improvement'])
                        total_comparisons += 1
                        if metric_results['significance']['significant']:
                            significant_improvements += 1

            if all_improvements:
                f.write(f"- **Average Relative Improvement**: {np.mean(all_improvements):.2%}\n")
                f.write(f"- **Statistically Significant Improvements**: {significant_improvements}/{total_comparisons} ({significant_improvements/total_comparisons:.1%})\n")
                f.write(f"- **Best Single Improvement**: {np.max(all_improvements):.2%}\n")
                f.write(f"- **Worst Single Change**: {np.min(all_improvements):.2%}\n\n")

            # 详细结果
            f.write("## Detailed Results\n\n")

            for env_name, env_results in self.benchmark_results.items():
                f.write(f"### {env_name}\n\n")
                f.write("| Metric | Baseline | Improved | Abs. Δ | Rel. Δ | Significant | Effect Size |\n")
                f.write("|--------|----------|----------|--------|--------|-------------|-------------|\n")

                for metric_name, metric_results in env_results.items():
                    baseline_mean = metric_results['baseline']['mean']
                    improved_mean = metric_results['improved']['mean']
                    abs_improvement = metric_results['absolute_improvement']
                    rel_improvement = metric_results['relative_improvement']
                    significant = "✓" if metric_results['significance']['significant'] else "✗"
                    effect_size = metric_results['significance'].get('effect_size', 'N/A')

                    f.write(f"| {metric_name} | {baseline_mean:.3f} | {improved_mean:.3f} | "
                           f"{abs_improvement:+.3f} | {rel_improvement:+.1%} | {significant} | {effect_size} |\n")

                f.write("\n")

        print(f"Benchmark report saved to: {output_file}")

# 导入必要的库（用于统计测试）
try:
    from scipy import stats
    from datetime import datetime
    import os
    import json
except ImportError:
    print("Warning: Some optional dependencies not available. Install scipy for statistical tests.")
    stats = None
