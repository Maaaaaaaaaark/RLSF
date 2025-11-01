import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class UncertaintyAwareCostEstimator:
    """
    不确定性感知的成本估计器
    
    核心功能：
    1. 基于集成学习量化预测不确定性
    2. 根据置信度调整成本估计
    3. 提供探索-利用平衡机制
    """
    
    def __init__(self, n_ensemble, uncertainty_penalty=0.1, 
                 exploration_bonus=0.05, confidence_threshold=0.8):
        self.n_ensemble = n_ensemble
        self.uncertainty_penalty = uncertainty_penalty
        self.exploration_bonus = exploration_bonus
        self.confidence_threshold = confidence_threshold
        
        # 不确定性统计
        self.uncertainty_history = []
        self.prediction_history = []
    
    def compute_ensemble_predictions(self, classifiers: List[nn.Module], 
                                   states: torch.Tensor, 
                                   actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算集成预测和不确定性
        
        Args:
            classifiers: 分类器集成列表
            states: 状态 [batch_size, state_dim]
            actions: 动作 [batch_size, action_dim]
            
        Returns:
            mean_probs: 平均预测概率 [batch_size]
            uncertainty: 预测不确定性 [batch_size]
            individual_probs: 各个分类器的预测 [n_ensemble, batch_size]
        """
        individual_predictions = []
        
        with torch.no_grad():
            for clf in classifiers:
                logits = clf(states, actions)
                probs = torch.sigmoid(logits)
                individual_predictions.append(probs.squeeze())
        
        # 堆叠所有预测
        individual_probs = torch.stack(individual_predictions, dim=0)  # [n_ensemble, batch_size]
        
        # 计算均值和不确定性
        mean_probs = torch.mean(individual_probs, dim=0)
        
        # 使用标准差作为不确定性度量
        uncertainty = torch.std(individual_probs, dim=0)
        
        # 归一化不确定性到[0,1]范围
        uncertainty = uncertainty / 0.5  # 最大标准差为0.5（当预测在0和1之间均匀分布时）
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
        
        return mean_probs, uncertainty, individual_probs
    
    def compute_uncertainty_aware_costs(self, mean_probs: torch.Tensor, 
                                      uncertainty: torch.Tensor,
                                      class_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算不确定性感知的成本估计 (已修正逻辑)
        
        Args:
            mean_probs: 平均预测概率 [batch_size]
            uncertainty: 不确定性分数 [batch_size]
            class_prob: 分类阈值
            
        Returns:
            costs: 调整后的成本 [batch_size]
            confidence: 预测置信度 [batch_size]
        """
        # 计算置信度（不确定性的反面）
        confidence = 1.0 - uncertainty
        
        # 1. 基础二值分类 (所有状态的成本都以此为基准)
        base_costs = (mean_probs > class_prob).float()
        
        # 2. 复制基础成本，准备进行调整
        uncertainty_adjusted_costs = base_costs.clone()
        
        # 3. 找出低置信度区域
        low_conf_mask = confidence <= self.confidence_threshold
        
        # 4. 在低置信度区域，应用探索激励（降低成本）
        if low_conf_mask.any():
            # 探索激励：不确定性越高，激励越大
            # self.exploration_bonus 应该是一个正值
            exploration_incentive = self.exploration_bonus * uncertainty[low_conf_mask]
            
            # 从基础成本中减去激励，使探索这些区域的“成本更低”
            adjusted_costs_low_conf = base_costs[low_conf_mask] - exploration_incentive
            
            # 将调整后的成本应用回张量，并确保成本不为负
            uncertainty_adjusted_costs[low_conf_mask] = torch.clamp(adjusted_costs_low_conf, 0.0, 1.0)
        
        # 5. 高置信度区域的成本保持不变（即 base_costs），因为我们从 base_costs.clone() 开始
        
        return uncertainty_adjusted_costs, confidence
    
    def compute_epistemic_uncertainty(self, individual_probs: torch.Tensor) -> torch.Tensor:
        """
        计算认识不确定性（模型不确定性）
        
        Args:
            individual_probs: 各分类器预测 [n_ensemble, batch_size]
            
        Returns:
            epistemic_uncertainty: 认识不确定性 [batch_size]
        """
        # 使用互信息作为认识不确定性的度量
        mean_probs = torch.mean(individual_probs, dim=0)
        
        # 计算平均熵（总不确定性）
        mean_entropy = -mean_probs * torch.log(mean_probs + 1e-8) - \
                      (1 - mean_probs) * torch.log(1 - mean_probs + 1e-8)
        
        # 计算期望熵（偶然不确定性）
        individual_entropy = -individual_probs * torch.log(individual_probs + 1e-8) - \
                           (1 - individual_probs) * torch.log(1 - individual_probs + 1e-8)
        expected_entropy = torch.mean(individual_entropy, dim=0)
        
        # 认识不确定性 = 总不确定性 - 偶然不确定性
        epistemic_uncertainty = mean_entropy - expected_entropy
        
        return epistemic_uncertainty
    
    def adaptive_exploration_strategy(self, states: torch.Tensor, 
                                    actions: torch.Tensor,
                                    uncertainty: torch.Tensor,
                                    training_progress: float) -> torch.Tensor:
        """
        自适应探索策略
        
        Args:
            states: 状态
            actions: 动作  
            uncertainty: 不确定性分数
            training_progress: 训练进度 [0,1]
            
        Returns:
            exploration_bonus: 探索奖励 [batch_size]
        """
        # 探索强度随训练进度递减
        exploration_strength = self.exploration_bonus * (1.0 - training_progress)
        
        # 基于不确定性的探索奖励
        exploration_bonus = exploration_strength * uncertainty
        
        # 添加新颖性奖励（基于状态访问频率）
        # 这里可以结合现有的hash-based novelty detection
        
        return exploration_bonus
    
    def update_uncertainty_statistics(self, uncertainty: torch.Tensor, 
                                    predictions: torch.Tensor):
        """更新不确定性统计信息"""
        self.uncertainty_history.extend(uncertainty.cpu().numpy())
        self.prediction_history.extend(predictions.cpu().numpy())
        
        # 保持历史记录在合理范围内
        if len(self.uncertainty_history) > 10000:
            self.uncertainty_history = self.uncertainty_history[-5000:]
            self.prediction_history = self.prediction_history[-5000:]
    
    def get_uncertainty_statistics(self) -> dict:
        """获取不确定性统计信息"""
        if len(self.uncertainty_history) == 0:
            return {}
        
        uncertainty_array = np.array(self.uncertainty_history)
        return {
            'mean_uncertainty': np.mean(uncertainty_array),
            'std_uncertainty': np.std(uncertainty_array),
            'high_uncertainty_ratio': np.mean(uncertainty_array > 0.5),
            'sample_count': len(uncertainty_array)
        }

class BayesianCostClassifier(nn.Module):
    """
    贝叶斯成本分类器（可选的更高级实现）
    """
    
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 dropout_rate=0.1, n_samples=10):
        super().__init__()
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        input_size = state_shape[0] + action_shape[0]
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_units:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states, actions, training=True):
        """
        前向传播，支持Monte Carlo Dropout进行不确定性估计
        """
        input_tensor = torch.cat([states, actions], dim=-1)
        
        if training or self.training:
            return self.network(input_tensor)
        else:
            # 推理时使用MC Dropout
            self.train()  # 启用dropout
            predictions = []
            
            for _ in range(self.n_samples):
                pred = self.network(input_tensor)
                predictions.append(pred)
            
            self.eval()  # 恢复eval模式
            
            predictions = torch.stack(predictions, dim=0)
            mean_pred = torch.mean(predictions, dim=0)
            uncertainty = torch.std(predictions, dim=0)
            
            return mean_pred, uncertainty
