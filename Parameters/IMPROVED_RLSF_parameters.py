import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

# 继承原有PREFIM参数
from .PREFIM_parameters import *

# 新增：自适应偏差校正参数
bias_correction_group = parser.add_argument_group('Adaptive_Bias_Correction')

bias_correction_group.add_argument('--enable_bias_correction', type=str, default='True',
                                 help='Enable adaptive bias correction mechanism')
bias_correction_group.add_argument('--bias_window_size', type=int, default=1000,
                                 help='Window size for bias estimation')
bias_correction_group.add_argument('--initial_delta', type=float, default=0.1,
                                 help='Initial bias correction parameter')
bias_correction_group.add_argument('--adaptation_rate', type=float, default=0.03,
                                 help='Learning rate for delta adaptation')
bias_correction_group.add_argument('--min_delta', type=float, default=0.0,
                                 help='Minimum value for delta')
bias_correction_group.add_argument('--max_delta', type=float, default=0.5,
                                 help='Maximum value for delta')
bias_correction_group.add_argument('--target_violation_rate', type=float, default=0.03,
                                 help='Target violation rate for adaptive control')

# 新增：不确定性建模参数
uncertainty_group = parser.add_argument_group('Uncertainty_Modeling')

uncertainty_group.add_argument('--enable_uncertainty_modeling', type=str, default='True',
                             help='Enable uncertainty-aware cost estimation')
uncertainty_group.add_argument('--uncertainty_penalty', type=float, default=0.2,
                             help='Penalty coefficient for uncertainty')
uncertainty_group.add_argument('--exploration_bonus', type=float, default=0.02,
                             help='Exploration bonus for high uncertainty regions')
uncertainty_group.add_argument('--confidence_threshold', type=float, default=0.8,
                             help='Threshold for high confidence predictions')
uncertainty_group.add_argument('--uncertainty_method', type=str, default='ensemble',
                             choices=['ensemble', 'mc_dropout', 'bayesian'],
                             help='Method for uncertainty quantification')

# 新增：改进的segment标签参数
segment_group = parser.add_argument_group('Improved_Segment_Labeling')

segment_group.add_argument('--enable_improved_labeling', type=str, default='True',
                         help='Enable improved segment labeling strategy')
segment_group.add_argument('--labeling_confidence_threshold', type=float, default=0.7,
                         help='Confidence threshold for segment labeling')
segment_group.add_argument('--conservative_labeling', type=str, default='True',
                         help='Use conservative labeling for uncertain states')

# 新增：评估和监控参数
evaluation_group = parser.add_argument_group('Evaluation_Monitoring')

evaluation_group.add_argument('--enable_detailed_logging', type=str, default='True',
                            help='Enable detailed logging for evaluation')
evaluation_group.add_argument('--log_uncertainty_stats', type=str, default='True',
                            help='Log uncertainty statistics')
evaluation_group.add_argument('--log_bias_correction_stats', type=str, default='True',
                            help='Log bias correction statistics')
evaluation_group.add_argument('--evaluation_interval', type=int, default=1000,
                            help='Interval for detailed evaluation')
evaluation_group.add_argument('--save_evaluation_plots', type=str, default='True',
                            help='Save evaluation plots')

# 新增：实验配置参数
experiment_group = parser.add_argument_group('Experiment_Configuration')

experiment_group.add_argument('--experiment_name', type=str, default='improved_rlsf',
                            help='Name of the experiment')
experiment_group.add_argument('--baseline_comparison', type=str, default='False',
                            help='Run baseline comparison experiment')
experiment_group.add_argument('--ablation_study', type=str, default='False',
                            help='Run ablation study')
experiment_group.add_argument('--save_models_interval', type=int, default=10000,
                            help='Interval for saving model checkpoints')

args, _unknown = parser.parse_known_args()

# 类型转换函数
def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 转换字符串参数为布尔值
enable_bias_correction = str_to_bool(args.enable_bias_correction)
enable_uncertainty_modeling = str_to_bool(args.enable_uncertainty_modeling)
enable_improved_labeling = str_to_bool(args.enable_improved_labeling)
enable_detailed_logging = str_to_bool(args.enable_detailed_logging)
log_uncertainty_stats = str_to_bool(args.log_uncertainty_stats)
log_bias_correction_stats = str_to_bool(args.log_bias_correction_stats)
save_evaluation_plots = str_to_bool(args.save_evaluation_plots)
baseline_comparison = str_to_bool(args.baseline_comparison)
ablation_study = str_to_bool(args.ablation_study)
conservative_labeling = str_to_bool(args.conservative_labeling)

# 偏差校正配置
bias_correction_config = {
    'enabled': enable_bias_correction,
    'window_size': args.bias_window_size,
    'initial_delta': args.initial_delta,
    'adaptation_rate': args.adaptation_rate,
    'min_delta': args.min_delta,
    'max_delta': args.max_delta,
    'target_violation_rate': args.target_violation_rate
}

# 不确定性建模配置
uncertainty_config = {
    'enabled': enable_uncertainty_modeling,
    'uncertainty_penalty': args.uncertainty_penalty,
    'exploration_bonus': args.exploration_bonus,
    'confidence_threshold': args.confidence_threshold,
    'method': args.uncertainty_method
}

# 改进标签配置
labeling_config = {
    'enabled': enable_improved_labeling,
    'confidence_threshold': args.labeling_confidence_threshold,
    'conservative': conservative_labeling
}

# 评估配置
evaluation_config = {
    'detailed_logging': enable_detailed_logging,
    'log_uncertainty': log_uncertainty_stats,
    'log_bias_correction': log_bias_correction_stats,
    'evaluation_interval': args.evaluation_interval,
    'save_plots': save_evaluation_plots
}

# 实验配置
experiment_config = {
    'name': args.experiment_name,
    'baseline_comparison': baseline_comparison,
    'ablation_study': ablation_study,
    'save_models_interval': args.save_models_interval
}

# 配置验证函数
def validate_config():
    """验证配置参数的合理性"""
    errors = []
    
    # 验证偏差校正参数
    if bias_correction_config['enabled']:
        if not (0 <= bias_correction_config['initial_delta'] <= 1):
            errors.append("initial_delta should be in [0, 1]")
        if not (0 < bias_correction_config['adaptation_rate'] <= 0.1):
            errors.append("adaptation_rate should be in (0, 0.1]")
        if bias_correction_config['min_delta'] >= bias_correction_config['max_delta']:
            errors.append("min_delta should be less than max_delta")
    
    # 验证不确定性建模参数
    if uncertainty_config['enabled']:
        if not (0 <= uncertainty_config['uncertainty_penalty'] <= 1):
            errors.append("uncertainty_penalty should be in [0, 1]")
        if not (0 <= uncertainty_config['exploration_bonus'] <= 1):
            errors.append("exploration_bonus should be in [0, 1]")
        if not (0 < uncertainty_config['confidence_threshold'] < 1):
            errors.append("confidence_threshold should be in (0, 1)")
    
    # 验证集成参数
    if n_ensemble < 3 and uncertainty_config['enabled']:
        errors.append("n_ensemble should be at least 3 for uncertainty modeling")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# 配置摘要函数
def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("IMPROVED RLSF CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"Environment: {env_name}")
    print(f"Seed: {seed}")
    print(f"Training Steps: {num_training_step}")
    print(f"Ensemble Size: {n_ensemble}")
    
    print("\n--- Bias Correction ---")
    for key, value in bias_correction_config.items():
        print(f"{key}: {value}")
    
    print("\n--- Uncertainty Modeling ---")
    for key, value in uncertainty_config.items():
        print(f"{key}: {value}")
    
    print("\n--- Improved Labeling ---")
    for key, value in labeling_config.items():
        print(f"{key}: {value}")
    
    print("\n--- Evaluation ---")
    for key, value in evaluation_config.items():
        print(f"{key}: {value}")
    
    print("=" * 60)

# 自动验证配置
if __name__ == "__main__":
    try:
        validate_config()
        print("Configuration validation passed!")
        print_config_summary()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        exit(1)

# 导出配置字典（供其他模块使用）
IMPROVED_RLSF_CONFIG = {
    'bias_correction': bias_correction_config,
    'uncertainty_modeling': uncertainty_config,
    'improved_labeling': labeling_config,
    'evaluation': evaluation_config,
    'experiment': experiment_config
}
