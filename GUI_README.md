# IMPROVED RLSF 图形界面使用指南

## 简介

本 GUI 为 IMPROVED RLSF 算法提供了一个交互式的训练监控界面，基于 Streamlit 构建。通过图形界面，您可以：

- 可视化配置训练参数（无需手动编写命令行）
- 实时监控训练进度和评估指标
- 查看训练日志和错误信息
- 管理模型权重和训练历史
- 检查系统环境和依赖

## 安装依赖

在已激活的虚拟环境中安装 GUI 所需的额外依赖：

```powershell
# 激活虚拟环境（如果尚未激活）
.\myenv\Scripts\Activate.ps1

# 安装依赖
pip install streamlit plotly
```

或者重新安装完整的 requirements.txt：

```powershell
pip install -r requirements.txt
```

## 启动界面

在 `RLSF` 目录下运行：

```powershell
cd RLSF
streamlit run gui_app.py
```

界面将自动在浏览器中打开（默认地址：http://localhost:8501）

## 界面功能说明

### 1. 侧边栏 - 训练配置

#### 预设模板
- **快速测试**：500步，适合快速验证功能
- **短程训练**：5000步，适合初步实验
- **完整训练**：100000步，适合正式训练
- **自定义**：手动配置所有参数

#### 基础配置
- **环境**：选择 Safety Gymnasium 环境（如 SafetyPointCircle1-v0）
- **训练步数**：总训练步数
- **随机种子**：用于结果复现

#### 设备配置
- **设备**：选择 cuda（GPU）或 cpu
- 界面会自动检测 CUDA 可用性并显示 GPU 信息

#### 并行配置
- **训练环境数**：并行训练的环境数量
- **评估环境数**：并行评估的环境数量
- **批次大小**：训练批次大小（建议 1024 以避免 GAE 维度问题）
- ⚠️ Windows 系统建议环境数 ≤ 4

#### WandB 日志
- **启用 WandB**：是否记录到 Weights & Biases
- **API Key**：WandB API 密钥（可选，避免控制台交互）

#### 改进功能
- ✅ 自适应偏差校正（默认启用）
- ✅ 不确定性感知成本估计（默认启用）
- ✅ 改进的片段标注（默认启用）

#### 训练控制
- **▶️ 开始训练**：启动训练进程
- **⏹️ 停止训练**：终止训练进程

### 2. 主区域 - 多标签页

#### 📊 概览
- **训练进度条**：显示当前训练进度百分比
- **指标卡片**：显示最新评估指标
  - R：平均回报
  - C：平均代价
  - SR：成功率
  - V：评估值
  - maxV：历史最佳值
- **改进算法指标**：
  - δ（delta）：偏差校正参数
  - bias_estimate：偏差估计值

#### 📈 实时曲线
- 四个子图实时显示训练曲线：
  - 平均回报随时间变化
  - 平均代价随时间变化
  - 成功率随时间变化
  - 评估值随时间变化
- 支持交互式缩放和悬停查看数值

#### 📝 日志
- 实时显示训练控制台输出
- 错误信息自动高亮显示（红色）
- 显示最近 200 行日志

#### 💾 模型权重
- 列出所有已保存的模型权重目录
- 显示每个权重的：
  - 路径
  - 文件大小
  - 文件数量
  - 最后修改时间
- 按修改时间倒序排列

#### 📜 历史记录
- 显示最近 10 次训练运行的记录
- 每条记录包含：
  - 运行时间戳
  - 训练配置参数
  - 最终评估指标
- 可用于复现之前的实验

#### 🔧 系统检查
- **CUDA 信息**：显示 CUDA 可用性和 GPU 设备名
- **依赖包版本**：显示关键包的版本信息
  - torch
  - gymnasium
  - safety-gymnasium
  - wandb
  - streamlit

## 使用流程

### 快速开始

1. **启动界面**
   ```powershell
   cd RLSF
   streamlit run gui_app.py
   ```

2. **选择预设模板**
   - 在侧边栏选择"快速测试"

3. **开始训练**
   - 点击"▶️ 开始训练"按钮

4. **监控训练**
   - 切换到"📊 概览"查看实时指标
   - 切换到"📈 实时曲线"查看训练曲线
   - 切换到"📝 日志"查看详细日志

5. **停止训练**（可选）
   - 点击"⏹️ 停止训练"按钮

### 自定义训练

1. **选择"自定义"模板**

2. **配置参数**
   - 选择环境（如 SafetyPointCircle1-v0）
   - 设置训练步数（如 5000）
   - 选择设备（cuda 或 cpu）
   - 设置并行环境数（Windows 建议 2）
   - 设置批次大小（建议 1024）

3. **可选：启用 WandB**
   - 勾选"启用 WandB"
   - 输入 API Key（如果有）

4. **开始训练**

## 配置持久化

- 界面会自动保存您的配置到 `RLSF/.gui_config.json`
- 下次启动时会自动加载上次的配置
- 训练历史会保存到 `RLSF/.gui_runs.json`（最多保留 50 条）

## 注意事项

### Windows 系统
- 建议并行环境数 ≤ 4，避免 AsyncVectorEnv 多进程问题
- 训练过程中可能出现 AsyncVectorEnv 析构警告，可忽略

### 批次大小
- 建议设置 batch_size=1024 或更大，避免 GAE 计算时的张量维度错误
- 如果遇到 "expand" 相关错误，尝试增大 batch_size

### WandB 日志
- 如果启用 WandB 但未提供 API Key，训练脚本会尝试使用内置的默认 Key
- 建议在 GUI 中输入您自己的 API Key 以避免冲突

### 日志解析
- 界面通过正则表达式解析训练日志中的指标
- 如果日志格式发生变化，可能需要更新 `gui_utils.py` 中的解析规则

## 故障排除

### 界面无法启动
```powershell
# 检查 streamlit 是否安装
pip show streamlit

# 重新安装
pip install streamlit plotly
```

### 训练无法启动
- 检查"📝 日志"标签页中的错误信息
- 检查"🔧 系统检查"标签页中的依赖版本
- 确保在正确的目录下运行（RLSF 目录）

### CUDA 不可用
- 检查是否安装了 GPU 版本的 PyTorch
- 运行系统检查查看 CUDA 状态
- 如果没有 GPU，选择 cpu 设备

### 指标不更新
- 确保训练正在运行（状态为 🟢 RUNNING）
- 界面每 2 秒自动刷新一次
- 检查日志中是否有评估输出

## 技术细节

### 后台进程管理
- 使用 `subprocess.Popen` 启动训练脚本
- 独立线程异步读取 stdout/stderr
- 线程安全的日志队列缓冲

### 日志解析
- 正则表达式匹配评估指标：`[Eval] R: xx, C: xx, SR: xx, V: xx, maxV: xx`
- 正则表达式匹配训练进度：`train: xx.xx%`
- 错误检测关键字：ERROR, Error, Traceback, Exception, Failed

### 配置文件格式
```json
{
  "env_name": "SafetyPointCircle1-v0",
  "num_training_step": 1000,
  "device_name": "cuda",
  "num_envs": 2,
  "eval_num_envs": 2,
  "batch_size": 1024,
  "seed": 0,
  "wandb_log": false,
  "wandb_api_key": ""
}
```

## 未来扩展

可能的增强功能：
- 模型权重加载和删除
- 训练暂停/恢复
- 超参数网格搜索
- 多实验对比
- WandB 深度集成（拉取在线指标）
- 自定义评估脚本

## 反馈与支持

如遇到问题或有改进建议，请查看：
- 主文档：`USAGE_GUIDE.md`
- 训练脚本：`Trains/train_improved_prefim.py`
- GUI 工具：`gui_utils.py`

