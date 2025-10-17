#!/usr/bin/env python3
"""
测试RLSF改进功能的简单脚本

该脚本验证自适应偏差校正和不确定性建模模块是否正常工作
"""

import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch

print("=" * 60)
print("RLSF改进功能测试")
print("=" * 60)

# 测试1: 导入改进模块
print("\n[测试1] 导入改进模块...")
try:
    from Sources.algo.adaptive_bias_corrector import AdaptiveBiasCorrector, SegmentLevelBiasCorrector
    print("✅ AdaptiveBiasCorrector 导入成功")
    print("✅ SegmentLevelBiasCorrector 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

try:
    from Sources.algo.uncertainty_aware_cost_estimator import UncertaintyAwareCostEstimator
    print("✅ UncertaintyAwareCostEstimator 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 自适应偏差校正器
print("\n[测试2] 测试自适应偏差校正器...")
try:
    corrector = AdaptiveBiasCorrector(
        window_size=100,
        initial_delta=0.1,
        adaptation_rate=0.01
    )
    
    # 模拟训练过程
    for i in range(10):
        training_progress = i / 10.0
        violation_rate = 0.08 - i * 0.003  # 模拟违约率下降
        
        delta = corrector.compute_adaptive_delta(
            training_progress=training_progress,
            current_violation_rate=violation_rate,
            target_violation_rate=0.05
        )
        
        # 模拟成本数据（使用张量以兼容update_statistics接口）
        predicted_cost = torch.tensor([np.random.rand()], dtype=torch.float32)
        true_cost = torch.tensor([np.random.rand()], dtype=torch.float32)
        corrector.update_statistics(predicted_cost, true_cost, violation_rate)

    print(f"✅ 自适应偏差校正器测试通过")
    print(f"   最终δ值: {corrector.delta:.4f}")
    print(f"   偏差估计: {corrector.bias_estimate:.4f}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 不确定性感知成本估计器
print("\n[测试3] 测试不确定性感知成本估计器...")
try:
    # 创建简单的分类器模拟
    class MockClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 1)
        
        def forward(self, states, actions):
            # 简单的前向传播
            x = torch.cat([states, actions], dim=-1)
            return self.fc(x)
    
    # 创建集成分类器
    n_ensemble = 3
    classifiers = [MockClassifier() for _ in range(n_ensemble)]
    
    # 创建不确定性估计器
    estimator = UncertaintyAwareCostEstimator(
        n_ensemble=n_ensemble,
        uncertainty_penalty=0.1,
        exploration_bonus=0.05
    )
    
    # 模拟预测
    batch_size = 32
    state_dim = 8
    action_dim = 2
    
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    # 计算集成预测和不确定性
    mean_probs, uncertainty, individual_probs = estimator.compute_ensemble_predictions(
        classifiers, states, actions
    )
    
    print(f"✅ 不确定性感知成本估计器测试通过")
    print(f"   平均预测概率: {mean_probs.mean().item():.4f}")
    print(f"   平均不确定性: {uncertainty.mean().item():.4f}")
    print(f"   不确定性范围: [{uncertainty.min().item():.4f}, {uncertainty.max().item():.4f}]")
    
    # 测试不确定性感知成本计算
    costs, confidence = estimator.compute_uncertainty_aware_costs(
        mean_probs, uncertainty, class_prob=0.5
    )
    
    print(f"   平均成本: {costs.mean().item():.4f}")
    print(f"   平均置信度: {confidence.mean().item():.4f}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: Segment级别偏差校正器
print("\n[测试4] 测试Segment级别偏差校正器...")
try:
    segment_corrector = SegmentLevelBiasCorrector(
        segment_length=100,
        confidence_threshold=0.7
    )

    # 模拟segment数据（状态+动作）
    segment_len = 100
    state_dim, action_dim = 8, 2
    segment_states = torch.randn(segment_len, state_dim)
    segment_actions = torch.randn(segment_len, action_dim)
    segment_label = 1  # 危险segment

    # 使用简单的Mock分类器集成
    classifier_ensemble = [MockClassifier() for _ in range(3)]

    improved_labels, confidence_scores = segment_corrector.improved_segment_labeling(
        segment_states, segment_actions, classifier_ensemble, segment_label
    )

    print(f"✅ Segment级别偏差校正器测试通过")
    print(f"   原始标签: {segment_label}")
    print(f"   校正后的标签数量: {improved_labels.numel()}")
    print(f"   高置信度标签比例: {improved_labels.float().mean().item():.2%}")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 集成测试
print("\n[测试5] 集成测试...")
try:
    # 模拟完整的训练循环
    corrector = AdaptiveBiasCorrector(window_size=50)
    estimator = UncertaintyAwareCostEstimator(n_ensemble=3)
    
    classifiers = [MockClassifier() for _ in range(3)]
    
    for epoch in range(5):
        # 模拟一个epoch的数据
        states = torch.randn(10, 8)
        actions = torch.randn(10, 2)
        
        # 计算不确定性感知的成本
        mean_probs, uncertainty, _ = estimator.compute_ensemble_predictions(
            classifiers, states, actions
        )
        costs, confidence = estimator.compute_uncertainty_aware_costs(
            mean_probs, uncertainty, class_prob=0.5
        )
        
        # 应用偏差校正
        training_progress = epoch / 5.0
        violation_rate = 0.06 - epoch * 0.002
        delta = corrector.compute_adaptive_delta(training_progress, violation_rate)
        corrected_costs = corrector.apply_bias_correction(costs, uncertainty)
        
        # 更新统计
        for i in range(len(costs)):
            pc = costs[i].unsqueeze(0)
            tc = torch.tensor([np.random.rand()], dtype=torch.float32)
            corrector.update_statistics(pc, tc, violation_rate)
    
    print(f"✅ 集成测试通过")
    print(f"   最终δ值: {corrector.delta:.4f}")
    print(f"   最终偏差估计: {corrector.bias_estimate:.4f}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！RLSF改进功能正常工作")
print("=" * 60)

print("\n📋 功能摘要:")
print("  ✅ 自适应偏差校正 - 动态调整δ参数")
print("  ✅ 不确定性建模 - 集成学习量化不确定性")
print("  ✅ Segment级别校正 - 改进标签策略")
print("  ✅ 集成工作流 - 所有组件协同工作")

print("\n🚀 下一步:")
print("  1. 在虚拟环境中安装所有依赖:")
print("     pip install torch numpy matplotlib scipy wandb safety-gymnasium")
print("  2. 运行完整训练:")
print("     python Trains/train_improved_prefim.py --env_name SafetyPointCircle1-v0")
print("  3. 或使用一键脚本:")
print("     python run_improved_rlsf.py --mode single --env_name SafetyPointCircle1-v0")

