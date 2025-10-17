"""
IMPROVED RLSF 训练图形界面
使用 Streamlit 构建的交互式训练监控界面
"""

import streamlit as st
import time
import os
from gui_utils import TrainingProcess, LogParser, ConfigManager, SystemChecker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="IMPROVED RLSF Training GUI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if "process" not in st.session_state:
    st.session_state.process = TrainingProcess()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None
if "training_status" not in st.session_state:
    st.session_state.training_status = "idle"
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "config" not in st.session_state:
    st.session_state.config = ConfigManager.load_config()

# 预设模板
PRESETS = {
    "快速测试": {
        "num_training_step": 5000,
        "num_envs": 2,
        "eval_num_envs": 2,
        "batch_size": 1024,
        "buffer_size": 1000,  # 更频繁更新（5000/1000≈5次）
        "wandb_log": False,
        "eval_interval": 50,
        "num_eval_episodes": 6,
        "max_episode_length": 200,
        "lr_actor": 5e-05,
        "lr_critic": 5e-05,
        "lr_cost_critic": 5e-05,
        "uncertainty_penalty": 0.5,
        "exploration_bonus": 0.01,
        "adaptation_rate": 0.03,
        "target_violation_rate": 0.03,
        "env_cost_limit": 25.0,
        "n_ensemble": 3,
        "segment_length": 200
    },
    "短程训练": {
        "num_training_step": 5000,
        "num_envs": 2,
        "eval_num_envs": 2,
        "batch_size": 1024,
        "wandb_log": False,
        "eval_interval": 1000
    },
    "完整训练": {
        "num_training_step": 100000,
        "num_envs": 4,
        "eval_num_envs": 4,
        "batch_size": 2048,
        "wandb_log": True,
        "eval_interval": 5000
    }
}

# 环境列表
ENVIRONMENTS = [
    "SafetyPointCircle1-v0",
    "SafetyPointGoal1-v0",
    "SafetyCarCircle1-v0",
    "SafetyCarGoal1-v0",
    "SafetyAntCircle1-v0"
]

# ==================== 侧边栏配置 ====================
st.sidebar.title("🎛️ 训练配置")

# 预设模板选择
preset = st.sidebar.selectbox("预设模板", ["自定义"] + list(PRESETS.keys()))
if preset != "自定义":
    for key, value in PRESETS[preset].items():
        st.session_state.config[key] = value

# 基础配置
st.sidebar.subheader("基础配置")
st.session_state.config["env_name"] = st.sidebar.selectbox(
    "环境", ENVIRONMENTS,
    index=ENVIRONMENTS.index(st.session_state.config["env_name"])
    if st.session_state.config["env_name"] in ENVIRONMENTS else 0
)

st.session_state.config["num_training_step"] = st.sidebar.number_input(
    "训练步数", min_value=100, max_value=10000000,
    value=st.session_state.config["num_training_step"], step=100
)

st.session_state.config["seed"] = st.sidebar.number_input(
    "随机种子", min_value=0, max_value=9999,
    value=st.session_state.config["seed"]
)

# 设备配置
cuda_available, cuda_info = SystemChecker.check_cuda()
default_device = "cuda" if cuda_available else "cpu"
st.session_state.config["device_name"] = st.sidebar.selectbox(
    "设备", ["cuda", "cpu"],
    index=0 if st.session_state.config["device_name"] == "cuda" else 1
)
if cuda_available:
    st.sidebar.success(f"✅ CUDA 可用: {cuda_info}")
else:
    st.sidebar.warning(f"⚠️ {cuda_info}")

# 并行配置
st.sidebar.subheader("并行配置")
st.session_state.config["num_envs"] = st.sidebar.number_input(
    "训练环境数", min_value=1, max_value=20,
    value=st.session_state.config["num_envs"]
)
st.session_state.config["eval_num_envs"] = st.sidebar.number_input(
    "评估环境数", min_value=1, max_value=50,
    value=st.session_state.config["eval_num_envs"]
)
if os.name == 'nt' and (st.session_state.config["num_envs"] > 4 or st.session_state.config["eval_num_envs"] > 4):
    st.sidebar.warning("⚠️ Windows 建议环境数 ≤ 4 以避免多进程问题")

# 评估间隔（步）——数值越小评估越频繁
st.session_state.config["eval_interval"] = st.sidebar.number_input(
    "评估间隔(步)", min_value=50, max_value=200000,
    value=st.session_state.config.get("eval_interval", 1000), step=50
)
st.session_state.config["batch_size"] = st.sidebar.number_input(
    "批次大小", min_value=128, max_value=8192,
    value=st.session_state.config["batch_size"], step=128
)

# WandB配置
st.sidebar.subheader("WandB 日志")
st.session_state.config["wandb_log"] = st.sidebar.checkbox(
    "启用 WandB", value=st.session_state.config["wandb_log"]
)
if st.session_state.config["wandb_log"]:
    st.session_state.config["wandb_api_key"] = st.sidebar.text_input(
        "API Key", value=st.session_state.config.get("wandb_api_key", ""),
        type="password"
    )

# 改进功能（只读显示）
st.sidebar.subheader("改进功能")
st.sidebar.info("✅ 自适应偏差校正")
st.sidebar.info("✅ 不确定性感知成本估计")
st.sidebar.info("✅ 改进的片段标注")

# 控制按钮
st.sidebar.subheader("训练控制")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ 开始训练", disabled=st.session_state.training_status == "running"):
        ConfigManager.save_config(st.session_state.config)
        if st.session_state.process.start(st.session_state.config):
            st.session_state.training_status = "running"
            st.session_state.logs = []
            st.session_state.metrics_history = []
            st.rerun()

with col2:
    if st.button("⏹️ 停止训练", disabled=st.session_state.training_status != "running"):
        st.session_state.process.stop()
        st.session_state.training_status = "stopped"
        st.rerun()

# ==================== 主区域 ====================
st.title("🚀 IMPROVED RLSF 训练监控")

# 状态指示器
status_colors = {
    "idle": "🔵",
    "running": "🟢",
    "stopped": "🟡",
    "completed": "✅",
    "error": "🔴"
}
st.markdown(f"### 状态: {status_colors.get(st.session_state.training_status, '⚪')} {st.session_state.training_status.upper()}")

# 进程状态快照
if st.session_state.training_status in ("running", "stopped", "completed", "error"):
    status = st.session_state.process.get_status()
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("PID", status.get("pid"))
    with col_b:
        st.metric("Alive", "Yes" if status.get("alive") else "No")
    with col_c:
        st.metric("Uptime(s)", f"{(status.get('uptime_sec') or 0):.0f}")
    with col_d:
        st.metric("Last Log(s)", f"{(status.get('sec_since_last_log') or 0):.0f}")


# 标签页
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 概览", "📈 实时曲线", "📝 日志", "💾 模型权重", "📜 历史记录", "🔧 系统检查"
])

# ==================== Tab 1: 概览 ====================
with tab1:
    # 进度条
    progress_bar = st.progress(st.session_state.progress / 100.0)
    st.text(f"训练进度: {st.session_state.progress:.2f}%")

    # 指标卡片
    if st.session_state.latest_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("平均回报 (R)", f"{st.session_state.latest_metrics['return']:.2f}")
        with col2:
            st.metric("平均代价 (C)", f"{st.session_state.latest_metrics['cost']:.2f}")
        with col3:
            st.metric("成功率 (SR)", f"{st.session_state.latest_metrics['success_rate']:.2f}")
        with col4:
            st.metric("评估值 (V)", f"{st.session_state.latest_metrics['value']:.2f}")
        with col5:
            st.metric("最佳值 (maxV)", f"{st.session_state.latest_metrics['max_value']:.2f}")
    else:
        st.info("等待评估数据...")

    # 改进算法指标
    st.subheader("改进算法指标")
    if st.session_state.latest_metrics and "delta" in st.session_state.latest_metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("偏差校正参数 (δ)", f"{st.session_state.latest_metrics.get('delta', 0.0):.4f}")
        with col2:
            st.metric("偏差估计", f"{st.session_state.latest_metrics.get('bias_estimate', 0.0):.4f}")
    else:
        st.info("改进指标将在训练过程中显示")

# ==================== Tab 2: 实时曲线 ====================
with tab2:
    if st.session_state.metrics_history:
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("平均回报", "平均代价", "成功率", "评估值")
        )

        steps = list(range(len(st.session_state.metrics_history)))
        returns = [m["return"] for m in st.session_state.metrics_history]
        costs = [m["cost"] for m in st.session_state.metrics_history]
        srs = [m["success_rate"] for m in st.session_state.metrics_history]
        values = [m["value"] for m in st.session_state.metrics_history]

        fig.add_trace(go.Scatter(x=steps, y=returns, mode='lines+markers', name='Return'), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=costs, mode='lines+markers', name='Cost'), row=1, col=2)
        fig.add_trace(go.Scatter(x=steps, y=srs, mode='lines+markers', name='Success Rate'), row=2, col=1)

        fig.add_trace(go.Scatter(x=steps, y=values, mode='lines+markers', name='Value'), row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待训练数据...")


# ==================== Tab 3: 日志 ====================
with tab3:
    st.subheader("训练日志")
    # 诊断提示
    if st.session_state.training_status == "running":
        status = st.session_state.process.get_status()
        if status.get("sec_since_last_log") and status["sec_since_last_log"] > 60:
            st.warning("超过 60 秒无新日志：这通常出现在评估阶段或训练静默阶段。若 CPU 持续接近 0% 且长时间无输出，可尝试点击‘停止训练’后重新开始。")

    log_container = st.container()
    with log_container:
        if st.session_state.logs:
            for log in st.session_state.logs[-200:]:  # 显示最近200行
                if LogParser.is_error(log):
                    st.error(log.strip())
                else:
                    st.text(log.strip())
        else:
            st.info("暂无日志")

# ==================== Tab 4: 模型权重 ====================
with tab4:
    st.subheader("已保存的模型权重")
    weights = SystemChecker.list_weight_dirs()
    if weights:
        for w in weights[:20]:  # 显示最近20个
            with st.expander(f"📁 {w['path']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("大小", f"{w['size_mb']:.2f} MB")
                col2.metric("文件数", w['file_count'])
                col3.text(f"修改时间: {w['modified']}")
    else:
        st.info("暂无保存的权重")

# ==================== Tab 5: 历史记录 ====================
with tab5:
    st.subheader("训练历史")
    history = ConfigManager.load_run_history()
    if history:
        for i, run in enumerate(reversed(history[-10:])):  # 显示最近10条
            with st.expander(f"Run {len(history)-i}: {run['timestamp'][:19]}"):
                st.json(run['config'])
                if run.get('final_metrics'):
                    st.json(run['final_metrics'])
    else:
        st.info("暂无历史记录")

# ==================== Tab 6: 系统检查 ====================
with tab6:
    st.subheader("系统环境")

    # CUDA信息
    cuda_available, cuda_info = SystemChecker.check_cuda()
    if cuda_available:
        st.success(f"✅ CUDA 可用: {cuda_info}")
    else:
        st.warning(f"⚠️ {cuda_info}")

    # 包版本
    st.subheader("依赖包版本")
    versions = SystemChecker.get_package_versions()
    for pkg, ver in versions.items():
        st.text(f"{pkg}: {ver}")

# ==================== 自动更新逻辑 ====================
if st.session_state.training_status == "running":
    # 读取新日志
    new_logs = st.session_state.process.get_logs()
    st.session_state.logs.extend(new_logs)

    # 解析日志
    for log in new_logs:
        # 解析评估指标
        metrics = LogParser.parse_eval_metrics(log)
        if metrics:
            st.session_state.latest_metrics = metrics
            st.session_state.metrics_history.append(metrics)

        # 解析进度
        progress = LogParser.parse_progress(log)
        if progress is not None:
            st.session_state.progress = progress

        # 解析改进指标
        improved = LogParser.parse_improved_metrics(log)
        if improved and st.session_state.latest_metrics:
            st.session_state.latest_metrics.update(improved)

        # 检测错误
        if LogParser.is_error(log):
            st.session_state.training_status = "error"

    # 检查进程状态
    if not st.session_state.process.is_alive():
        st.session_state.training_status = "completed"
        # 进程已结束，将进度条置为 100%
        try:
            if st.session_state.progress < 100:
                st.session_state.progress = 100.0
        except Exception:
            st.session_state.progress = 100.0
        if st.session_state.latest_metrics:
            ConfigManager.add_run_history(st.session_state.config, st.session_state.latest_metrics)

    # 自动刷新
    time.sleep(2)
    st.rerun()

