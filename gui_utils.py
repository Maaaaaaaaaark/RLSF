"""
GUI工具模块 - 用于IMPROVED RLSF训练界面
包含进程管理、日志解析、配置管理等功能
"""

import subprocess
import threading
import queue
import re
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class TrainingProcess:
    """管理训练子进程和日志读取"""

    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir
        self.process: Optional[subprocess.Popen] = None
        self.log_queue = queue.Queue()
        self.log_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.pid: Optional[int] = None
        self.start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None

    def start(self, config: Dict) -> bool:
        """启动训练进程"""
        if self.is_running:
            return False

        # 选择Python可执行文件：优先使用项目根目录下的 myenv 虚拟环境
        python_exec = None
        try:
            from sys import executable as sys_exec
            python_exec = sys_exec
        except Exception:
            python_exec = "python"
        try:
            base_dir = Path(__file__).resolve().parent  # RLSF 目录
            root_dir = base_dir.parent  # 仓库根目录
            myenv_python = root_dir / "myenv" / "Scripts" / "python.exe"
            if os.name == 'nt' and myenv_python.exists():
                python_exec = str(myenv_python)
        except Exception:
            pass

        # 构建命令行参数
        cmd = [python_exec, "Trains/train_improved_prefim.py"]
        cmd.extend(["--env_name", config["env_name"]])
        cmd.extend(["--num_training_step", str(config["num_training_step"])])
        cmd.extend(["--device_name", config["device_name"]])
        cmd.extend(["--num_envs", str(config["num_envs"])])
        cmd.extend(["--eval_num_envs", str(config["eval_num_envs"])])
        cmd.extend(["--seed", str(config["seed"])])
        cmd.extend(["--wandb_log", "True" if config["wandb_log"] else "False"])

        # 可选：评估相关频率/规模
        if config.get("eval_interval") is not None:
            cmd.extend(["--eval_interval", str(config["eval_interval"])])
        if config.get("num_eval_episodes") is not None:
            cmd.extend(["--num_eval_episodes", str(config["num_eval_episodes"])])
        if config.get("max_episode_length") is not None:
            cmd.extend(["--max_episode_length", str(config["max_episode_length"])])


        if config["batch_size"] is not None:
            cmd.extend(["--batch_size", str(config["batch_size"])])

        # buffer_size 控制 rollout_length，影响更新频率
        if config.get("buffer_size") is not None:
            cmd.extend(["--buffer_size", str(config["buffer_size"])])

        # 传递可选的高级超参数（若在 config 中提供）
        for key, flag in [
            ("lr_actor", "--lr_actor"),
            ("lr_critic", "--lr_critic"),
            ("lr_cost_critic", "--lr_cost_critic"),
            ("uncertainty_penalty", "--uncertainty_penalty"),
            ("exploration_bonus", "--exploration_bonus"),
            ("adaptation_rate", "--adaptation_rate"),
            ("target_violation_rate", "--target_violation_rate"),
            ("env_cost_limit", "--env_cost_limit"),
            ("n_ensemble", "--n_ensemble"),
            ("segment_length", "--segment_length"),
        ]:
            if key in config and config[key] is not None:
                cmd.extend([flag, str(config[key])])

        try:
            # 设置环境变量
            env = os.environ.copy()
            if config["wandb_log"] and config.get("wandb_api_key"):
                env["WANDB_API_KEY"] = config["wandb_api_key"]

            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',  # 明确指定 UTF-8 编码，避免 Windows GBK 问题
                errors='replace',  # 遇到无法解码的字符时替换为 �，而不是抛出异常
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

            # 记录进程信息
            self.pid = self.process.pid
            self.start_time = time.time()
            self.last_log_time = self.start_time

            # 启动日志读取线程
            self.is_running = True
            self.log_thread = threading.Thread(target=self._read_logs, daemon=True)
            self.log_thread.start()

            return True
        except Exception as e:
            self.log_queue.put(f"ERROR: Failed to start training: {str(e)}\n")
            return False

    def _read_logs(self):
        """后台线程读取日志"""
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                self.log_queue.put(line)
                self.last_log_time = time.time()
            self.process.stdout.close()
        self.is_running = False

    def stop(self):
        """停止训练进程"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False

    def get_logs(self, max_lines: int = 500) -> List[str]:
        """获取最新日志"""
        logs = []
        while not self.log_queue.empty() and len(logs) < max_lines:
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def get_status(self) -> Dict:
        """返回子进程状态快照"""
        alive = self.is_alive()
        now = time.time()
        return {
            "pid": self.pid,
            "alive": alive,
            "uptime_sec": (now - self.start_time) if self.start_time else None,
            "sec_since_last_log": (now - self.last_log_time) if self.last_log_time else None,
        }

    def is_alive(self) -> bool:
        """检查进程是否存活"""
        return self.is_running and self.process is not None and self.process.poll() is None


class LogParser:
    """解析训练日志中的指标"""

    # 正则表达式模式
    # 支持负数与科学计数法，例如 -1.23、4.5e-3
    EVAL_PATTERN = re.compile(
        r"\[Eval\]\s+R:\s+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s+"
        r"C:\s+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s+"
        r"SR:\s+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s+"
        r"V:\s+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s+"
        r"maxV:\s+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
    )
    PROGRESS_PATTERN = re.compile(r'train:\s+([\d.]+)%')
    # 改进指标：更宽松，支持负数/科学计数
    IMPROVED_DELTA_PATTERN = re.compile(r'improved/delta[:\s]+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)')
    IMPROVED_BIAS_PATTERN = re.compile(r'improved/bias_estimate[:\s]+(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)')

    @staticmethod
    def parse_eval_metrics(log_line: str) -> Optional[Dict]:
        """解析评估指标"""
        match = LogParser.EVAL_PATTERN.search(log_line)
        if match:
            return {
                "return": float(match.group(1)),
                "cost": float(match.group(2)),
                "success_rate": float(match.group(3)),
                "value": float(match.group(4)),
                "max_value": float(match.group(5)),
                "timestamp": datetime.now().isoformat()
            }
        return None

    @staticmethod
    def parse_progress(log_line: str) -> Optional[float]:
        """解析训练进度"""
        match = LogParser.PROGRESS_PATTERN.search(log_line)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def parse_improved_metrics(log_line: str) -> Dict:
        """解析改进算法指标"""
        metrics = {}
        delta_match = LogParser.IMPROVED_DELTA_PATTERN.search(log_line)
        bias_match = LogParser.IMPROVED_BIAS_PATTERN.search(log_line)

        if delta_match:
            metrics["delta"] = float(delta_match.group(1))
        if bias_match:
            metrics["bias_estimate"] = float(bias_match.group(1))

        return metrics

    @staticmethod
    def is_error(log_line: str) -> bool:
        """检测是否为错误日志"""
        error_keywords = ["ERROR", "Error", "Traceback", "Exception", "Failed"]
        return any(keyword in log_line for keyword in error_keywords)

    @staticmethod
    def read_metric_file(filepath: str) -> List[float]:
        """读取指标文件（如eval_return.txt）"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return [float(line.strip()) for line in f if line.strip()]
        except Exception:
            pass
        return []


class ConfigManager:
    """配置和历史记录管理"""

    CONFIG_FILE = ".gui_config.json"
    HISTORY_FILE = ".gui_runs.json"

    @staticmethod
    def get_default_config() -> Dict:
        """获取默认配置"""
        return {
            "env_name": "SafetyPointCircle1-v0",
            "num_training_step": 1000,
            "device_name": "cuda",
            "num_envs": 2,
            "eval_num_envs": 2,
            "batch_size": 1024,
            "seed": 0,
            "wandb_log": False,
            "wandb_api_key": "",
            "eval_interval": 1000,
            "num_eval_episodes": None,
            "max_episode_length": None,
            "enable_bias_correction": True,
            "enable_uncertainty_modeling": True,
            "enable_improved_labeling": True,
            # extended hyperparameters passed from GUI (optional)
            "lr_actor": None,
            "lr_critic": None,
            "lr_cost_critic": None,
            "uncertainty_penalty": None,
            "exploration_bonus": None,
            "adaptation_rate": None,
            "target_violation_rate": None,
            "env_cost_limit": None,
            "n_ensemble": None,
            "segment_length": None,
            "max_episode_length": None,
            "num_eval_episodes": None,
            "buffer_size": None
        }

    @staticmethod
    def load_config() -> Dict:
        """加载配置"""
        try:
            cfg_path = Path(ConfigManager.CONFIG_FILE)
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return ConfigManager.get_default_config()

    @staticmethod
    def save_config(config: Dict):
        """保存配置"""
        try:
            cfg_path = Path(ConfigManager.CONFIG_FILE)
            parent = str(cfg_path.parent)
            if parent not in ("", "."):
                os.makedirs(parent, exist_ok=True)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save config: {e}")

    @staticmethod
    def add_run_history(config: Dict, metrics: Dict):
        """添加运行历史"""
        try:
            history = []
            hist_path = Path(ConfigManager.HISTORY_FILE)
            if hist_path.exists():
                with open(hist_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            history.append({
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "final_metrics": metrics
            })

            # 只保留最近50条
            history = history[-50:]

            hist_path = Path(ConfigManager.HISTORY_FILE)
            parent = str(hist_path.parent)
            if parent not in ("", "."):
                os.makedirs(parent, exist_ok=True)
            with open(hist_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save run history: {e}")

    @staticmethod
    def load_run_history() -> List[Dict]:
        """加载运行历史"""
        try:
            if os.path.exists(ConfigManager.HISTORY_FILE):
                with open(ConfigManager.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []


class SystemChecker:
    """系统环境检查"""

    @staticmethod
    def check_cuda() -> Tuple[bool, str]:
        """检查CUDA可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return True, device_name
            else:
                return False, "CUDA not available"
        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_package_versions() -> Dict[str, str]:
        """获取关键包版本"""
        packages = {}
        try:
            import torch
            packages["torch"] = torch.__version__
        except:
            packages["torch"] = "Not installed"

        try:
            import gymnasium
            packages["gymnasium"] = gymnasium.__version__
        except:
            packages["gymnasium"] = "Not installed"

        try:
            import safety_gymnasium
            packages["safety_gymnasium"] = safety_gymnasium.__version__
        except:
            packages["safety_gymnasium"] = "Not installed"

        try:
            import wandb
            packages["wandb"] = wandb.__version__
        except:
            packages["wandb"] = "Not installed"

        try:
            import streamlit
            packages["streamlit"] = streamlit.__version__
        except:
            packages["streamlit"] = "Not installed"

        return packages

    @staticmethod
    def list_weight_dirs(base_path: str = "weights") -> List[Dict]:
        """列出权重目录"""
        weight_dirs = []
        try:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if files:  # 只列出包含文件的目录
                        rel_path = os.path.relpath(root, base_path)
                        size = sum(os.path.getsize(os.path.join(root, f)) for f in files)
                        mtime = max(os.path.getmtime(os.path.join(root, f)) for f in files)
                        weight_dirs.append({
                            "path": rel_path,
                            "size_mb": size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            "file_count": len(files)
                        })
        except Exception as e:
            print(f"Error listing weights: {e}")
        return sorted(weight_dirs, key=lambda x: x["modified"], reverse=True)

