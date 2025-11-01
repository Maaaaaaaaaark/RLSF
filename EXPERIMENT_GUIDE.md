# RLSF改进算法实验指南

## 📋 实验设计

本指南提供了完整的实验流程，用于验证RLSF算法改进的有效性。

## 🎯 实验目标

1. **验证自适应偏差校正的有效性**
   - δ参数能否自动调整到合理范围
   - 是否减少了过度保守行为
   - 违约率控制是否更精确

2. **验证不确定性建模的价值**
   - 不确定性估计是否准确
   - 是否提高了探索效率
   - 是否改善了样本效率

3. **对比基线性能**
   - 与原始RLSF算法对比
   - 与其他安全RL算法对比（如CPO、PPO-Lagrangian）

## 🔬 实验设置

### 实验1：功能验证测试

**目的**: 验证改进模块是否正常工作

```bash
# 运行功能测试
python test_improvements.py
```

**预期结果**:
- ✅ 所有模块导入成功
- ✅ 自适应偏差校正器正常工作
- ✅ 不确定性估计器正常工作
- ✅ 集成测试通过

### 实验2：单环境快速测试

**目的**: 在单个环境上快速验证改进效果

```bash
# 改进版RLSF (1000步快速测试)
python Trains/train_improved_prefim.py \
    --env_name SafetyPointCircle1-v0 \
    --seed 0 \
    --num_training_step 1000 \
    --wandb_log False

# 对比：原始RLSF
python Trains/train_prefim.py \
    --env_name SafetyPointCircle1-v0 \
    --seed 0 \
    --num_training_step 1000 \
    --wandb_log False
```

**观察指标**:
- 训练过程中的δ值变化
- 不确定性统计
- 违约率控制情况
- 收敛速度

### 实验3：完整训练对比

**目的**: 完整训练周期的性能对比

```bash
# 改进版RLSF (完整训练)
python run_improved_rlsf.py \
    --mode single \
    --env_name SafetyPointCircle1-v0 \
    --num_training_step 100000 \
    --seed 0 \
    --wandb_log True
```

**评估指标**:
- 最终回报 (Final Return)
- 最终成本 (Final Cost)
- 成功率 (Success Rate)
- 收敛episode数
- 训练时间

### 实验4：消融研究

**目的**: 分析各个改进组件的独立贡献

```bash
python run_improved_rlsf.py \
    --mode ablation \
    --environments SafetyPointCircle1-v0 \
    --seeds 0 1 2 \
    --num_training_step 50000
```

**实验变体**:
1. **baseline**: 原始RLSF
2. **bias_correction_only**: 仅自适应偏差校正
3. **uncertainty_only**: 仅不确定性建模
4. **improved_labeling_only**: 仅改进标签策略
5. **bias_correction_uncertainty**: 偏差校正+不确定性
6. **full_improved**: 所有改进

**分析维度**:
- 各组件对性能的独立贡献
- 组件间的协同效应
- 不同环境下的表现差异

### 实验5：多环境泛化测试

**目的**: 验证改进在不同环境下的泛化能力

```bash
# 测试多个Safety Gymnasium环境
for env in SafetyPointCircle1-v0 SafetyCarCircle1-v0 SafetyPointGoal1-v0
do
    python run_improved_rlsf.py \
        --mode single \
        --env_name $env \
        --num_training_step 100000 \
        --seed 0
done
```

**环境列表**:
- `SafetyPointCircle1-v0`: 点机器人避障
- `SafetyCarCircle1-v0`: 车辆避障
- `SafetyPointGoal1-v0`: 目标导航
- `SafetyAntCircle1-v0`: 四足机器人

### 实验6：参数敏感性分析

**目的**: 分析关键参数对性能的影响

```bash
# 测试不同的δ初始值
for delta in 0.05 0.1 0.15 0.2
do
    python Trains/train_improved_prefim.py \
        --env_name SafetyPointCircle1-v0 \
        --initial_delta $delta \
        --seed 0
done

# 测试不同的集成数量
for n_ensemble in 2 3 5 7
do
    python Trains/train_improved_prefim.py \
        --env_name SafetyPointCircle1-v0 \
        --n_ensemble $n_ensemble \
        --seed 0
done
```

**参数范围**:
- `initial_delta`: [0.05, 0.1, 0.15, 0.2]
- `adaptation_rate`: [0.005, 0.01, 0.02, 0.05]
- `n_ensemble`: [2, 3, 5, 7]
- `uncertainty_penalty`: [0.05, 0.1, 0.15, 0.2]

## 📊 数据收集

### 训练过程指标

在训练过程中记录以下指标：

1. **性能指标**
   - `episode_return`: 每个episode的累积回报
   - `episode_cost`: 每个episode的累积成本
   - `success_rate`: 成功率（成本≤阈值）

2. **偏差校正指标**
   - `delta`: 当前δ值
   - `bias_estimate`: 估计的系统偏差
   - `violation_rate`: 当前违约率
   - `delta_stability`: δ的变化率

3. **不确定性指标**
   - `mean_uncertainty`: 平均不确定性
   - `high_uncertainty_ratio`: 高不确定性状态比例
   - `confidence_accuracy`: 置信度-准确性相关性

4. **效率指标**
   - `convergence_episode`: 收敛所需episode数
   - `sample_efficiency`: 样本效率
   - `training_time`: 训练时间

### 评估指标

在评估阶段记录：

1. **最终性能**
   - `final_return`: 最终平均回报
   - `final_cost`: 最终平均成本
   - `final_success_rate`: 最终成功率

2. **稳定性**
   - `return_std`: 回报标准差
   - `cost_std`: 成本标准差
   - `success_rate_std`: 成功率标准差

3. **安全性**
   - `max_cost`: 最大单次成本
   - `cost_violation_count`: 违约次数
   - `safety_margin`: 安全边际

## 📈 结果分析

### 可视化

创建以下可视化图表：

1. **学习曲线**
   ```python
   # 回报和成本随训练步数的变化
   plt.plot(steps, returns, label='Return')
   plt.plot(steps, costs, label='Cost')
   ```

2. **δ值动态**
   ```python
   # δ值随训练进度的自适应变化
   plt.plot(steps, delta_values)
   plt.axhline(y=optimal_delta, linestyle='--', label='Optimal')
   ```

3. **不确定性分布**
   ```python
   # 不确定性的分布和演化
   plt.hist(uncertainties, bins=50)
   ```

4. **消融研究对比**
   ```python
   # 不同变体的性能对比
   plt.bar(variants, performances)
   ```

### 统计检验

使用统计检验验证改进的显著性：

```python
from scipy import stats

# t检验
t_stat, p_value = stats.ttest_ind(improved_returns, baseline_returns)

# Mann-Whitney U检验（非参数）
u_stat, p_value = stats.mannwhitneyu(improved_returns, baseline_returns)

# 效应量（Cohen's d）
cohens_d = (mean_improved - mean_baseline) / pooled_std
```

**显著性水平**: p < 0.05

## 📝 实验记录模板

### 实验日志

```markdown
## 实验 [编号]

**日期**: YYYY-MM-DD
**实验者**: [姓名]

### 配置
- 环境: SafetyPointCircle1-v0
- 种子: 0, 1, 2
- 训练步数: 100000
- 改进功能: 全部启用

### 结果
- 平均回报: XXX ± YYY
- 平均成本: XXX ± YYY
- 成功率: XX%
- 收敛episode: XXX

### 观察
- δ值稳定在 [X.XX, X.XX] 范围
- 不确定性随训练下降
- 违约率控制在目标范围内

### 结论
[实验结论]

### 问题和改进
[遇到的问题和改进建议]
```

## 🎯 成功标准

改进被认为成功，如果满足以下条件：

1. **性能提升**
   - 平均回报提升 ≥ 10%
   - 或收敛速度提升 ≥ 15%
   - 或样本效率提升 ≥ 20%

2. **安全性维持**
   - 违约率 ≤ 目标违约率 + 5%
   - 最大成本不显著增加

3. **稳定性**
   - 性能标准差不显著增加
   - δ值在合理范围内稳定

4. **统计显著性**
   - p-value < 0.05
   - Cohen's d > 0.5 (中等效应量)

## 🔧 故障排除

### 常见问题

1. **δ值不稳定**
   - 减小 `adaptation_rate`
   - 增大 `window_size`

2. **不确定性过高**
   - 增加 `n_ensemble`
   - 检查分类器训练是否充分

3. **违约率控制不佳**
   - 调整 `target_violation_rate`
   - 检查成本函数定义

4. **收敛缓慢**
   - 增加 `exploration_bonus`
   - 检查奖励函数设计

## 📚 参考资料

- 原始RLSF论文
- `TECHNICAL_IMPROVEMENTS.md`: 技术细节
- `USAGE_GUIDE.md`: 使用指南
- `IMPROVEMENTS_README.md`: 改进说明

## 🚀 下一步

完成实验后：

1. **整理结果**: 汇总所有实验数据
2. **撰写报告**: 分析改进效果
3. **发布代码**: 开源改进实现
4. **撰写论文**: 发表研究成果

---

**最后更新**: 2025-10-14
**版本**: 1.0.0

