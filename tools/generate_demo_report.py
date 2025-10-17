import os
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Targets:
    # 目标（用于生成“完美训练”示例曲线）
    R_final: float = 100.0
    C_final: float = 10.0
    SR_final: float = 0.95
    V_scale: float = 10.0  # 保持与训练脚本中 V = (R*SR)/10 的口径一致
    delta_start: float = 0.10
    delta_final: float = 0.05
    bias_start: float = 0.20
    bias_final: float = 0.00


def make_demo_series(n: int, targets: Targets):
    t = np.linspace(0.0, 1.0, n)

    # 回报 R：平滑上升
    R = targets.R_final * (0.15 + 0.85 * t**1.2)

    # 代价 C：平滑下降（指数衰减到 C_final）
    C = (targets.C_final + (600.0 - targets.C_final) * np.exp(-4.0 * t))

    # 成功率 SR：S 形上升
    SR = targets.SR_final * (1.0 / (1.0 + np.exp(-10 * (t - 0.5))))

    # 评估值 V 与 maxV
    V = (R * SR) / targets.V_scale
    maxV = np.maximum.accumulate(V)

    # 改进指标：delta 缓慢收敛，bias 指数收敛
    delta = targets.delta_final + (targets.delta_start - targets.delta_final) * np.exp(-3.0 * t)
    bias = targets.bias_final + (targets.bias_start - targets.bias_final) * np.exp(-4.0 * t)

    return t, R, C, SR, V, maxV, delta, bias


def add_watermark(ax, text="DEMO / 模拟数据"):
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=9, color="#888",
            ha="right", va="bottom", alpha=0.8)


def plot_metrics_png(out_dir: str, t, R, C, SR, V, maxV):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].plot(t, R, label="R")
    axs[0, 0].set_title("平均回报 (R)")
    axs[0, 0].grid(True)
    add_watermark(axs[0, 0])

    axs[0, 1].plot(t, C, color="#d9534f", label="C")
    axs[0, 1].set_title("平均代价 (C)")
    axs[0, 1].grid(True)
    add_watermark(axs[0, 1])

    axs[1, 0].plot(t, SR, color="#5bc0de", label="SR")
    axs[1, 0].set_title("成功率 (SR)")
    axs[1, 0].set_ylim(0.0, 1.0)
    axs[1, 0].grid(True)
    add_watermark(axs[1, 0])

    axs[1, 1].plot(t, V, label="V")
    axs[1, 1].plot(t, maxV, linestyle="--", label="maxV")
    axs[1, 1].set_title("评估值 (V / maxV)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    add_watermark(axs[1, 1])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics.png"), dpi=160)
    plt.close(fig)


def plot_improved_png(out_dir: str, t, delta, bias):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    axs[0].plot(t, delta, color="#5cb85c")
    axs[0].set_title("偏差校正参数 (delta)")
    axs[0].grid(True)
    add_watermark(axs[0])

    axs[1].plot(t, bias, color="#f0ad4e")
    axs[1].set_title("偏差估计 (bias_estimate)")
    axs[1].grid(True)
    add_watermark(axs[1])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improved.png"), dpi=160)
    plt.close(fig)


def export_csv(out_dir: str, t, R, C, SR, V, maxV, delta, bias):
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "t", "R", "C", "SR", "V", "maxV", "delta", "bias_estimate"]) 
        for i in range(len(t)):
            writer.writerow([i+1, float(t[i]), float(R[i]), float(C[i]), float(SR[i]), float(V[i]), float(maxV[i]), float(delta[i]), float(bias[i])])


def export_pdf(out_dir: str):
    # 将两张 PNG 合并为一页 PDF（保持零依赖，使用 matplotlib 的 Figure+Image 方式）
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        pdf_path = os.path.join(out_dir, "report.pdf")
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 横向
            ax.axis("off")
            ax.set_title("IMPROVED RLSF 演示报告（模拟数据）", fontsize=14, loc="left")
            img1 = mpimg.imread(os.path.join(out_dir, "metrics.png"))
            img2 = mpimg.imread(os.path.join(out_dir, "improved.png"))
            ax.imshow(np.concatenate([img1, img2], axis=0))
            pdf.savefig(fig)
            plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Failed to export PDF: {e}")


def default_output_dir() -> str:
    # 默认输出到 RLSF/demo_outputs 下（脚本位于 RLSF/tools）
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(here, "..", "demo_outputs"))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate demo (simulated) metrics for presentation.")
    parser.add_argument("--points", type=int, default=50, help="number of evaluation points (default: 50)")
    parser.add_argument("--out-dir", type=str, default=default_output_dir(), help="output directory (default: RLSF/demo_outputs)")
    parser.add_argument("--no-pdf", action="store_true", help="do not export report.pdf")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    targets = Targets()
    t, R, C, SR, V, maxV, delta, bias = make_demo_series(args.points, targets)

    plot_metrics_png(args.out_dir, t, R, C, SR, V, maxV)
    plot_improved_png(args.out_dir, t, delta, bias)
    export_csv(args.out_dir, t, R, C, SR, V, maxV, delta, bias)
    if not args.no_pdf:
        export_pdf(args.out_dir)

    print("[OK] Demo artifacts written to:", args.out_dir)
    print(" - metrics.png, improved.png, metrics.csv, report.pdf (optional)")


if __name__ == "__main__":
    main()

