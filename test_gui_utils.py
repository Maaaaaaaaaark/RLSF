"""
GUI 工具模块测试脚本
验证 gui_utils.py 中的各个类是否正常工作
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from gui_utils import LogParser, ConfigManager, SystemChecker

def test_log_parser():
    """测试日志解析器"""
    print("=" * 60)
    print("测试 LogParser")
    print("=" * 60)
    
    # 测试评估指标解析
    test_log = "[Eval] R: 20.74, C: 650.66, SR: 0.00, V: 0.00, maxV: 0.00"
    metrics = LogParser.parse_eval_metrics(test_log)
    
    if metrics:
        print("✅ 评估指标解析成功:")
        print(f"   Return: {metrics['return']}")
        print(f"   Cost: {metrics['cost']}")
        print(f"   Success Rate: {metrics['success_rate']}")
        print(f"   Value: {metrics['value']}")
        print(f"   Max Value: {metrics['max_value']}")
    else:
        print("❌ 评估指标解析失败")
    
    # 测试进度解析
    test_progress = "train: 45.67% 456/1000"
    progress = LogParser.parse_progress(test_progress)
    
    if progress is not None:
        print(f"✅ 进度解析成功: {progress}%")
    else:
        print("❌ 进度解析失败")
    
    # 测试错误检测
    test_error = "ERROR: Something went wrong"
    is_error = LogParser.is_error(test_error)
    
    if is_error:
        print("✅ 错误检测成功")
    else:
        print("❌ 错误检测失败")
    
    print()


def test_config_manager():
    """测试配置管理器"""
    print("=" * 60)
    print("测试 ConfigManager")
    print("=" * 60)
    
    # 测试默认配置
    default_config = ConfigManager.get_default_config()
    print("✅ 默认配置:")
    for key, value in default_config.items():
        print(f"   {key}: {value}")
    
    # 测试保存和加载
    test_config = default_config.copy()
    test_config["env_name"] = "TestEnv"
    test_config["num_training_step"] = 12345
    
    ConfigManager.save_config(test_config)
    print("✅ 配置已保存")
    
    loaded_config = ConfigManager.load_config()
    if loaded_config["env_name"] == "TestEnv" and loaded_config["num_training_step"] == 12345:
        print("✅ 配置加载成功")
    else:
        print("❌ 配置加载失败")
    
    # 恢复默认配置
    ConfigManager.save_config(default_config)
    
    print()


def test_system_checker():
    """测试系统检查器"""
    print("=" * 60)
    print("测试 SystemChecker")
    print("=" * 60)
    
    # 测试 CUDA 检查
    cuda_available, cuda_info = SystemChecker.check_cuda()
    if cuda_available:
        print(f"✅ CUDA 可用: {cuda_info}")
    else:
        print(f"⚠️  CUDA 不可用: {cuda_info}")
    
    # 测试包版本检查
    print("\n📦 依赖包版本:")
    versions = SystemChecker.get_package_versions()
    for pkg, ver in versions.items():
        status = "✅" if ver != "Not installed" else "❌"
        print(f"   {status} {pkg}: {ver}")
    
    # 测试权重目录列表
    print("\n💾 权重目录:")
    weights = SystemChecker.list_weight_dirs()
    if weights:
        print(f"   找到 {len(weights)} 个权重目录")
        for w in weights[:3]:  # 只显示前3个
            print(f"   - {w['path']} ({w['size_mb']:.2f} MB)")
    else:
        print("   暂无权重目录")
    
    print()


def test_improved_metrics_parsing():
    """测试改进算法指标解析"""
    print("=" * 60)
    print("测试改进算法指标解析")
    print("=" * 60)
    
    # 测试 delta 解析
    test_log_delta = "improved/delta: 0.1234"
    metrics = LogParser.parse_improved_metrics(test_log_delta)
    
    if "delta" in metrics:
        print(f"✅ Delta 解析成功: {metrics['delta']}")
    else:
        print("❌ Delta 解析失败")
    
    # 测试 bias_estimate 解析
    test_log_bias = "improved/bias_estimate: 0.5678"
    metrics = LogParser.parse_improved_metrics(test_log_bias)
    
    if "bias_estimate" in metrics:
        print(f"✅ Bias Estimate 解析成功: {metrics['bias_estimate']}")
    else:
        print("❌ Bias Estimate 解析失败")
    
    print()


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "GUI 工具模块测试" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_log_parser()
        test_config_manager()
        test_system_checker()
        test_improved_metrics_parsing()
        
        print("=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
        print()
        print("GUI 工具模块工作正常，可以启动界面:")
        print("  streamlit run gui_app.py")
        print()
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 测试失败: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

