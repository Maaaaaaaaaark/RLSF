import torch
import numpy as np
from collections import deque
import torch.nn.functional as F

class AdaptiveBiasCorrector:
    """
    自适应偏差校正器，用于动态调整RLSF中的成本高估偏差
    
    核心思想：
    1. 基于历史性能数据估计偏差程度
    2. 根据训练进度和策略安全性动态调整校正强度
    3. 使用滑动窗口统计真实成本与预测成本的差异
    """
    
    def __init__(self, window_size=1000, initial_delta=0.1, 
                 adaptation_rate=0.01, min_delta=0.0, max_delta=0.5):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_delta = min_delta
        self.max_delta = max_delta
        
        # 当前偏差校正参数
        self.delta = initial_delta
        
        # 历史数据缓存
        self.cost_predictions = deque(maxlen=window_size)
        self.true_costs = deque(maxlen=window_size)
        self.violation_rates = deque(maxlen=window_size)
        
        # 性能统计
        self.bias_estimate = 0.0
        self.confidence = 0.0
        
    def update_statistics(self, predicted_costs, true_costs, violation_rate):
        """
        更新统计信息
        
        Args:
            predicted_costs: 预测的成本值 [batch_size]
            true_costs: 真实的成本值 [batch_size] 
            violation_rate: 当前违约率 (scalar)
        """
        # 更新历史数据
        self.cost_predictions.extend(predicted_costs.cpu().numpy())
        self.true_costs.extend(true_costs.cpu().numpy())
        self.violation_rates.append(violation_rate)
        
        # 计算偏差估计
        if len(self.cost_predictions) >= 100:  # 最小样本数
            pred_array = np.array(self.cost_predictions)
            true_array = np.array(self.true_costs)
            
            # 计算系统性偏差
            self.bias_estimate = np.mean(pred_array - true_array)
            
            # 计算置信度（基于样本数和方差）
            bias_variance = np.var(pred_array - true_array)
            self.confidence = min(1.0, len(self.cost_predictions) / self.window_size) * \
                            np.exp(-bias_variance)  # 方差越大置信度越低
    
    def compute_adaptive_delta(self, training_progress, current_violation_rate, 
                                target_violation_rate=0.03): #
            """
            计算自适应偏差校正参数 (已修正逻辑)
            
            Args:
                training_progress: 训练进度 [0, 1]
                current_violation_rate: 当前违约率
                target_violation_rate: 目标违约率
                
            Returns:
                delta: 偏差校正参数
            """
            # 基于违约率的调整
            # (修正) 如果违规率过高 (error > 0)，我们应该 *减少* delta (即 *增加* 惩罚)
            # 因此 violation_error 的权重应该是 *负* 的。
            violation_error = current_violation_rate - target_violation_rate
            
            # 基于偏差估计的调整
            # (修正) 如果偏差过高 (bias > 0)，意味着我们高估了成本，
            # 我们应该 *增加* delta 来 *抵消* 这个高估。
            # 因此 bias_factor 的权重应该是 *正* 的。
            bias_factor = np.tanh(self.bias_estimate * 5)
            
            # 综合调整 (修正了 violation_error 的符号)
            delta_adjustment = self.adaptation_rate * (
                -1.0 * violation_error +      # (修正) 违约率高 -> 负向调整
                0.3 * bias_factor           # (保持) 偏差高 -> 正向调整
            )
            
            # (可选) 移除 progress_factor，因为它会导致 delta 仅因时间推移而增长
            
            # 更新delta
            self.delta = np.clip(
                self.delta + delta_adjustment,
                self.min_delta, 
                self.max_delta
            )
            
            return self.delta
    def apply_bias_correction(self, raw_costs, uncertainty_scores=None):
        """
        应用偏差校正
        
        Args:
            raw_costs: 原始成本预测 [batch_size]
            uncertainty_scores: 不确定性分数 [batch_size] (可选)
            
        Returns:
            corrected_costs: 校正后的成本 [batch_size]
        """
        if uncertainty_scores is not None:
            # 基于不确定性的自适应校正
            # 不确定性高的地方校正力度小，不确定性低的地方校正力度大
            adaptive_delta = self.delta * (1.0 - uncertainty_scores)
        else:
            adaptive_delta = self.delta
        
        # 应用校正：降低成本估计以减少过度保守
        corrected_costs = raw_costs - adaptive_delta
        
        # 确保成本在合理范围内
        corrected_costs = torch.clamp(corrected_costs, 0.0, 1.0)
        
        return corrected_costs
    
    def get_correction_info(self):
        """获取校正器状态信息"""
        return {
            'delta': self.delta,
            'bias_estimate': self.bias_estimate,
            'confidence': self.confidence,
            'sample_count': len(self.cost_predictions)
        }

class SegmentLevelBiasCorrector:
    """
    Segment-Level偏差校正器，从根本上改进segment labeling策略
    """
    
    def __init__(self, segment_length, confidence_threshold=0.7):
        self.segment_length = segment_length
        self.confidence_threshold = confidence_threshold
    
    def improved_segment_labeling(self, segment_states, segment_actions, 
                                classifier_ensemble, true_segment_label):
        """
        改进的segment标签策略，减少标签噪声
        
        Args:
            segment_states: segment中的状态 [segment_length, state_dim]
            segment_actions: segment中的动作 [segment_length, action_dim]
            classifier_ensemble: 分类器集成
            true_segment_label: 真实的segment标签 (0 or 1)
            
        Returns:
            improved_labels: 改进的state-level标签 [segment_length]
            confidence_scores: 置信度分数 [segment_length]
        """
        # 获取每个状态的预测
        state_predictions = []
        state_uncertainties = []
        
        for clf in classifier_ensemble:
            with torch.no_grad():
                logits = clf(segment_states, segment_actions)
                probs = torch.sigmoid(logits)
                state_predictions.append(probs)
        
        # 计算集成预测和不确定性
        ensemble_probs = torch.stack(state_predictions, dim=0)
        mean_probs = torch.mean(ensemble_probs, dim=0)
        uncertainty = torch.std(ensemble_probs, dim=0)
        
        # 改进的标签分配策略
        improved_labels = torch.zeros_like(mean_probs)
        confidence_scores = 1.0 - uncertainty
        
        if true_segment_label == 1:  # unsafe segment
            # 对于unsafe segment，只有高置信度预测为unsafe的状态才标记为unsafe
            high_conf_unsafe = (mean_probs > 0.5) & (confidence_scores > self.confidence_threshold)
            improved_labels[high_conf_unsafe] = 1.0
        else:  # safe segment
            # 对于safe segment，只有高置信度预测为safe的状态才标记为safe
            high_conf_safe = (mean_probs <= 0.5) & (confidence_scores > self.confidence_threshold)
            improved_labels[~high_conf_safe] = 1.0  # 不确定的状态标记为unsafe（保守策略）
        
        return improved_labels.squeeze(), confidence_scores.squeeze()
