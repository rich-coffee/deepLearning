#!/usr/bin/env python3
"""
=============================================================================
 深度学习期末作业 — 模型对比分析工具
 横向对比 RNN / GRU / LSTM 三种模型的：
   1. 训练效率（损失收敛曲线、训练耗时）
   2. 收敛特性（达到各损失阈值所需 epoch 数）
   3. 成本-性能综合评估（参数量、推理速度、预测精度、雷达图）
=============================================================================
【使用方法】
  python compare_models.py
  一键完成：重新训练 RNN / GRU / LSTM 三个模型（从头开始，200 epoch），
  自动收集全量 loss 曲线 + 推理指标，生成 7 张对比图和 1 份 JSON 数据。

【配套文件】
  train.py          单模型训练脚本 — 手动切换 MODEL_TYPE 逐模型训练
  sever.py          WebSocket 推理服务器 — 加载权重推送实时预测流
  index.html        前端监控面板 — 使用本地 d3.v7.min.js，无需 CDN

【输出文件】 所有图表和 JSON 数据保存在 MIT-BIH_data/comparison/ 目录下:
  - loss_convergence.png         训练损失收敛曲线对比（全 epoch + 尾部放大）
  - training_efficiency.png      训练效率对比（总耗时 / 单 epoch 耗时 / 参数量）
  - convergence_speed.png        收敛速度分析（达到各损失阈值所需 epoch）
  - inference_performance.png    推理性能对比（延迟 + 预测精度 MSE/MAE/RMSE）
  - cost_performance_radar.png   成本-性能雷达图（5 维度归一化）
  - prediction_waveform.png      测试集预测波形对比（实际 vs 预测 + 误差带）
  - summary_table.png            综合对比汇总表（★ 标记最优值）
  - comparison_data.json         结构化原始数据（含全量 loss 历史）
=============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import os, time, json, sys
from pathlib import Path
from collections import OrderedDict

# ============================================================================
# 0. 全局配置
# ============================================================================
MODEL_TYPES = ['RNN', 'GRU', 'LSTM']
HIDDEN_SIZE = 32
NUM_LAYERS = 1
TOTAL_EPOCHS = 200
LEARNING_RATE = 0.01
TRAIN_SEQ_LEN = 500          # 训练序列长度（与 train.py 一致）
TEST_SEQ_LEN = 500           # 测试序列长度
TEST_OFFSET = 2000           # 测试数据在信号中的起始偏移（避免与训练段重叠）
RANDOM_SEED = 42

OUTPUT_DIR = Path('MIT-BIH_data/comparison')
WEIGHT_DIR = Path('MIT-BIH_data/weight')

# ---- 中文字体支持 ----
_FONT_NAMES = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
               'Noto Sans CJK SC', 'Source Han Sans SC', 'DejaVu Sans']
for _f in _FONT_NAMES:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_f] + plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子，确保可复现
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# 1. 模型定义（与 train.py / sever.py 完全一致）
# ============================================================================
class SleepRNNDemo(nn.Module):
    """统一模型工厂：支持 RNN / GRU / LSTM 三种循环神经网络"""
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32,
                 num_layers=1, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers,
                                   batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers,
                                   batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True)
        else:
            raise ValueError(f"未知的网络类型: {cell_type}")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        return self.fc(out)


# ============================================================================
# 2. 数据加载
# ============================================================================
def load_mitbih_signal(record_name='100'):
    """读取 MIT-BIH 原始 .dat/.hea 文件，返回 Z-score 标准化后的完整信号"""
    import wfdb
    record = wfdb.rdrecord(record_name)
    # 提取 MLII 导联（第一导联）
    signal = record.p_signal[:, 0].astype(np.float32)
    # Z-score 标准化
    signal = (signal - signal.mean()) / signal.std()
    return signal


def load_training_data(device='cpu'):
    """
    训练数据：与 train.py 完全一致 — 前 500 个采样点
    单步预测任务：x[0:500] → y[1:501]
    """
    signal = load_mitbih_signal('100')
    x = signal[:TRAIN_SEQ_LEN]
    y = signal[1:TRAIN_SEQ_LEN + 1]
    x = torch.tensor(x, dtype=torch.float32).view(1, -1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1, 1)
    return x.to(device), y.to(device)


def load_test_data(device='cpu'):
    """
    测试数据：使用信号中远离训练段的不同区域（偏移 TEST_OFFSET）
    确保训练/测试无数据泄漏
    """
    signal = load_mitbih_signal('100')
    start = TEST_OFFSET
    end = start + TEST_SEQ_LEN
    if end + 1 > len(signal):
        raise ValueError(f"测试偏移 {TEST_OFFSET} 超出信号长度 {len(signal)}")
    x = signal[start:end]
    y = signal[start + 1:end + 1]
    x = torch.tensor(x, dtype=torch.float32).view(1, -1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1, 1)
    return x.to(device), y.to(device)


# ============================================================================
# 3. 训练函数（全量记录 loss + 计时）
# ============================================================================
def train_one_model(model_type, device, verbose=True):
    """
    训练单个模型，返回完整的训练指标字典。
    每次调用使用独立的模型实例和优化器，确保公平对比。
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = SleepRNNDemo(cell_type=model_type, hidden_size=HIDDEN_SIZE,
                         num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x_train, y_train = load_training_data(device)
    x_test, y_test = load_test_data(device)

    loss_history = []
    epoch_times = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  开始训练 [{model_type}] — {TOTAL_EPOCHS} epochs")
        print(f"{'='*60}")

    t_total_start = time.time()

    for epoch in range(TOTAL_EPOCHS):
        t_epoch_start = time.time()

        model.train()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - t_epoch_start
        epoch_times.append(epoch_time)
        loss_history.append(loss.item())

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [{model_type}] Epoch {epoch+1:3d}/{TOTAL_EPOCHS}  "
                  f"Loss: {loss.item():.6f}  ({epoch_time*1000:.1f} ms)")

    total_time = time.time() - t_total_start

    # ---- 收敛速度：达到各损失阈值所需 epoch 数 ----
    thresholds = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    convergence = {}
    for th in thresholds:
        for ep, loss_val in enumerate(loss_history):
            if loss_val <= th:
                convergence[f'epoch_to_{th}'] = ep + 1
                break
        else:
            convergence[f'epoch_to_{th}'] = None  # 未达到

    # ---- 模型复杂度 ----
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- 保存权重 ----
    weight_path = WEIGHT_DIR / f'sleep_{model_type.lower()}_weights.pth'
    torch.save(model.state_dict(), weight_path)
    weight_size = os.path.getsize(weight_path)

    if verbose:
        print(f"  [{model_type}] 训练完成! 总耗时: {total_time:.1f}s,  "
              f"最终 Loss: {loss_history[-1]:.6f}")

    return {
        'model_type': model_type,
        'loss_history': loss_history,
        'total_time_sec': total_time,
        'avg_epoch_time_ms': np.mean(epoch_times) * 1000,
        'num_params': num_params,
        'trainable_params': trainable_params,
        'weight_file_bytes': weight_size,
        'convergence': convergence,
        'final_loss': loss_history[-1],
        'model': model,
        'x_test': x_test,
        'y_test': y_test,
    }


# ============================================================================
# 4. 推理评估函数
# ============================================================================
def evaluate_model(model, x_test, y_test, device, n_warmup=10, n_runs=100):
    """
    评估模型在测试集上的推理性能和预测精度。
    返回: MSE, MAE, RMSE, 平均推理延迟 (us)
    """
    model.eval()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    with torch.no_grad():
        # 精度评估
        pred = model(x_test)
        mse = criterion_mse(pred, y_test).item()
        mae = criterion_mae(pred, y_test).item()
        rmse = np.sqrt(mse)

        # 推理延迟评估（单步推理，多次测量取平均）
        single_input = torch.tensor([[[0.0]]], dtype=torch.float32).to(device)

        # Warmup
        for _ in range(n_warmup):
            _ = model(single_input)

        # 计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_runs):
                _ = model(single_input)
            end.record()
            torch.cuda.synchronize()
            avg_latency_us = start.elapsed_time(end) / n_runs * 1000
        else:
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(single_input)
                times.append((time.perf_counter() - t0) * 1e6)  # us
            avg_latency_us = np.mean(times)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'avg_inference_latency_us': avg_latency_us,
    }


# ============================================================================
# 5. 图表生成
# ============================================================================
COLORS = {'RNN': '#3B82F6', 'GRU': '#F59E0B', 'LSTM': '#10B981'}
LINE_STYLES = {'RNN': '-', 'GRU': '--', 'LSTM': '-.'}
MARKERS = {'RNN': 'o', 'GRU': 's', 'LSTM': '^'}


def set_style():
    """统一图表风格 — 深色背景 + 高对比度"""
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#CCCCCC',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': 150,
    })


def plot_loss_convergence(results, save_path):
    """图1: 训练损失收敛曲线（主图 + 尾部放大子图）"""
    set_style()
    fig, (ax_main, ax_tail) = plt.subplots(1, 2, figsize=(14, 5.5),
                                           gridspec_kw={'width_ratios': [3, 2]})

    # ---- 主图：全 200 epoch ----
    for r in results:
        epochs = range(1, len(r['loss_history']) + 1)
        ax_main.plot(epochs, r['loss_history'],
                     color=COLORS[r['model_type']],
                     linestyle=LINE_STYLES[r['model_type']],
                     linewidth=2, alpha=0.9,
                     label=f"{r['model_type']} (final={r['final_loss']:.5f})")

    ax_main.set_xlabel('Epoch')
    ax_main.set_ylabel('MSE Loss')
    ax_main.set_title('Training Loss Convergence — Full 200 Epochs')
    ax_main.legend(loc='upper right', framealpha=0.9)
    ax_main.set_yscale('log')
    ax_main.set_ylim(bottom=min(r['final_loss'] for r in results) * 0.5)

    # ---- 尾部放大图：最后 60 epoch ----
    tail_start = TOTAL_EPOCHS - 60
    for r in results:
        epochs = range(tail_start + 1, TOTAL_EPOCHS + 1)
        ax_tail.plot(epochs, r['loss_history'][tail_start:],
                     color=COLORS[r['model_type']],
                     linestyle=LINE_STYLES[r['model_type']],
                     linewidth=2, alpha=0.9)

    ax_tail.set_xlabel('Epoch')
    ax_tail.set_ylabel('MSE Loss')
    ax_tail.set_title(f'Final 60 Epochs (Zoom)')
    ax_tail.legend([r['model_type'] for r in results], loc='upper right', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Loss convergence chart → {save_path}")


def plot_training_efficiency(results, save_path):
    """图2: 训练效率 — 总耗时、单epoch耗时、参数量"""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = [r['model_type'] for r in results]

    # Subplot 1: 总训练时间
    total_times = [r['total_time_sec'] for r in results]
    bars1 = axes[0].bar(names, total_times, color=[COLORS[n] for n in names],
                        edgecolor='white', linewidth=0.8, alpha=0.85)
    axes[0].set_title('Total Training Time (s)')
    axes[0].set_ylabel('Seconds')
    for bar, val in zip(bars1, total_times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: 平均单 epoch 耗时
    epoch_times = [r['avg_epoch_time_ms'] for r in results]
    bars2 = axes[1].bar(names, epoch_times, color=[COLORS[n] for n in names],
                        edgecolor='white', linewidth=0.8, alpha=0.85)
    axes[1].set_title('Avg Time per Epoch (ms)')
    axes[1].set_ylabel('Milliseconds')
    for bar, val in zip(bars2, epoch_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 3: 参数量
    param_counts = [r['num_params'] for r in results]
    bars3 = axes[2].bar(names, param_counts, color=[COLORS[n] for n in names],
                        edgecolor='white', linewidth=0.8, alpha=0.85)
    axes[2].set_title('Trainable Parameters')
    axes[2].set_ylabel('Count')
    for bar, val in zip(bars3, param_counts):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Training Efficiency Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Training efficiency chart → {save_path}")


def plot_convergence_speed(results, save_path):
    """图3: 收敛速度 — 达到各损失阈值所需 epoch 数"""
    set_style()
    thresholds = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    names = [r['model_type'] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(thresholds))
    bar_width = 0.25

    for i, r in enumerate(results):
        epochs_to_th = []
        colors_used = []
        for th in thresholds:
            ep = r['convergence'].get(f'epoch_to_{th}')
            epochs_to_th.append(ep if ep is not None else 0)
            colors_used.append(COLORS[r['model_type']] if ep is not None else '#DDDDDD')

        bars = ax.bar(x + i * bar_width, epochs_to_th, bar_width,
                      color=[COLORS[r['model_type']] if v > 0 else '#CCCCCC'
                             for v in epochs_to_th],
                      edgecolor='white', linewidth=0.6,
                      alpha=0.9, label=r['model_type'])

        # 标注数值
        for j, (bar, val) in enumerate(zip(bars, epochs_to_th)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, TOTAL_EPOCHS * 0.05,
                        'N/A', ha='center', va='bottom', fontsize=8, color='#999999')

    ax.set_xlabel('Loss Threshold')
    ax.set_ylabel('Epochs to Reach Threshold')
    ax.set_title('Convergence Speed — Fewer Epochs = Faster Convergence')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f'Loss ≤ {th}' for th in thresholds])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, TOTAL_EPOCHS * 1.15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Convergence speed chart → {save_path}")


def plot_inference_performance(results, eval_results, save_path):
    """图4: 推理性能 — 延迟 + 预测精度（MSE / MAE / RMSE）"""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    names = [r['model_type'] for r in results]

    # Subplot 1: 推理延迟
    latencies = [eval_results[n]['avg_inference_latency_us'] for n in names]
    bars1 = axes[0].bar(names, latencies, color=[COLORS[n] for n in names],
                        edgecolor='white', linewidth=0.8, alpha=0.85)
    axes[0].set_title('Avg Single-Step Inference Latency (μs)')
    axes[0].set_ylabel('Microseconds (μs)')
    for bar, val in zip(bars1, latencies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f} μs', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: 预测精度（分组柱状图）
    metrics = ['mse', 'mae', 'rmse']
    metric_labels = ['MSE', 'MAE', 'RMSE']
    x = np.arange(len(metrics))
    bar_width = 0.25

    for i, name in enumerate(names):
        values = [eval_results[name][m] for m in metrics]
        axes[1].bar(x + i * bar_width, values, bar_width,
                    color=COLORS[name], edgecolor='white', linewidth=0.6,
                    alpha=0.9, label=name)

    axes[1].set_xticks(x + bar_width)
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_title('Prediction Accuracy on Test Set (Lower = Better)')
    axes[1].legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Inference Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Inference performance chart → {save_path}")


def plot_radar_chart(results, eval_results, save_path):
    """图5: 成本-性能综合雷达图（5 维度）"""
    set_style()

    names = [r['model_type'] for r in results]
    dimensions = ['Training Speed\n(higher better)',
                  'Convergence Speed\n(higher better)',
                  'Prediction Accuracy\n(higher better)',
                  'Model Lightweight\n(higher better)',
                  'Inference Speed\n(higher better)']

    # ---- 原始指标 ----
    total_times = np.array([r['total_time_sec'] for r in results])
    # 取达到 loss=0.05 所需 epoch（或 200 如果未达到）
    conv_epochs = np.array([
        r['convergence'].get('epoch_to_0.05') or TOTAL_EPOCHS
        for r in results
    ])
    mse_values = np.array([eval_results[r['model_type']]['mse'] for r in results])
    param_counts = np.array([r['num_params'] for r in results])
    latencies = np.array([eval_results[r['model_type']]['avg_inference_latency_us']
                          for r in results])

    # ---- 归一化到 [0, 1]（1 = 最优） ----
    def normalize_inverse(values):
        """越小越好 → 归一化到 0~1，1 最优"""
        v = np.array(values, dtype=float)
        if v.max() == v.min():
            return np.ones_like(v)
        return 1.0 - (v - v.min()) / (v.max() - v.min())

    normalized = {
        'Training Speed': normalize_inverse(total_times),
        'Convergence Speed': normalize_inverse(conv_epochs),
        'Prediction Accuracy': normalize_inverse(mse_values),
        'Model Lightweight': normalize_inverse(param_counts),
        'Inference Speed': normalize_inverse(latencies),
    }

    # ---- 绘制 ----
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, name in enumerate(names):
        values = [normalized[d][i] for d in normalized.keys()]
        values += values[:1]  # 闭合

        ax.fill(angles, values, color=COLORS[name], alpha=0.15)
        ax.plot(angles, values, color=COLORS[name], linewidth=2.5,
                linestyle=LINE_STYLES[name], label=name, marker='o', markersize=8)

        # 标注数值
        for j, (angle, val) in enumerate(zip(angles[:-1], values[:-1])):
            ax.annotate(f'{val:.2f}',
                        xy=(angle, val),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        color=COLORS[name], alpha=0.9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='#999999')
    ax.set_title('Cost-Performance Radar Chart\n(All dimensions normalized, 1.0 = Best)',
                 fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Radar chart → {save_path}")


def plot_summary_table(results, eval_results, save_path):
    """图6: 综合对比汇总表"""
    set_style()
    names = [r['model_type'] for r in results]

    # 构建表格数据
    rows = [
        'Final Loss (MSE)',
        'Total Training Time (s)',
        'Avg Epoch Time (ms)',
        'Trainable Parameters',
        'Weight File Size (bytes)',
        'Epochs to Loss ≤ 0.05',
        'Test MSE',
        'Test MAE',
        'Test RMSE',
        'Inference Latency (μs)',
    ]

    cell_data = []
    for label in rows:
        row = []
        for r in results:
            name = r['model_type']
            if label == 'Final Loss (MSE)':
                row.append(f"{r['final_loss']:.6f}")
            elif label == 'Total Training Time (s)':
                row.append(f"{r['total_time_sec']:.1f}")
            elif label == 'Avg Epoch Time (ms)':
                row.append(f"{r['avg_epoch_time_ms']:.1f}")
            elif label == 'Trainable Parameters':
                row.append(str(r['num_params']))
            elif label == 'Weight File Size (bytes)':
                row.append(str(r['weight_file_bytes']))
            elif label == 'Epochs to Loss ≤ 0.05':
                ep = r['convergence'].get('epoch_to_0.05')
                row.append(str(ep) if ep is not None else 'N/A')
            elif label == 'Test MSE':
                row.append(f"{eval_results[name]['mse']:.6f}")
            elif label == 'Test MAE':
                row.append(f"{eval_results[name]['mae']:.6f}")
            elif label == 'Test RMSE':
                row.append(f"{eval_results[name]['rmse']:.6f}")
            elif label == 'Inference Latency (μs)':
                row.append(f"{eval_results[name]['avg_inference_latency_us']:.1f}")
        cell_data.append(row)

    # 高亮最优值
    def highlight_best(cell_data, row_idx, lower_is_better=True):
        """在最优单元格添加标记"""
        col_count = len(cell_data[0])
        numeric_rows = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if row_idx not in numeric_rows:
            return cell_data
        vals = []
        for c in range(col_count):
            try:
                vals.append(float(cell_data[row_idx][c]))
            except ValueError:
                return cell_data
        if lower_is_better:
            best_idx = vals.index(min(vals))
        else:
            best_idx = vals.index(max(vals))
        cell_data[row_idx][best_idx] = f"★ {cell_data[row_idx][best_idx]}"
        return cell_data

    for ri in range(len(rows)):
        cell_data = highlight_best(cell_data, ri)

    # 绘制表格
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    col_labels = names
    row_labels = rows

    table = ax.table(cellText=cell_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     rowLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # 样式
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        elif col == -1:
            cell.set_facecolor('#EEEEEE')
            cell.set_text_props(fontweight='bold', fontsize=9)
        else:
            cell.set_facecolor('#FAFAFA')

    ax.set_title('Comprehensive Model Comparison Summary\n(★ = Best in category)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Summary table → {save_path}")


def plot_prediction_waveform(results, eval_results, save_path):
    """图7: 测试集上的预测波形对比（前 150 点）"""
    set_style()
    names = [r['model_type'] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    for i, r in enumerate(results):
        model = r['_model']
        x_test = r['_x_test']
        y_test = r['_y_test']
        device = x_test.device

        model.eval()
        with torch.no_grad():
            pred = model(x_test)

        # 取前 150 点绘制
        n_show = 150
        t = np.arange(n_show)
        actual = y_test[0, :n_show, 0].cpu().numpy()
        predicted = pred[0, :n_show, 0].cpu().numpy()

        axes[i].plot(t, actual, color='#FFDD00', linewidth=1.5,
                     label='Actual (Ground Truth)', alpha=0.9)
        axes[i].plot(t, predicted, color='#00FF00', linewidth=1.5,
                     linestyle='--', label=f'{r["model_type"]} Prediction', alpha=0.9)
        axes[i].fill_between(t, actual, predicted, alpha=0.15, color='red',
                             label=f'Error (MSE={eval_results[r["model_type"]]["mse"]:.5f})')

        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].set_title(f'{r["model_type"]} — Test Set Prediction (First 150 Points)')
        axes[i].legend(loc='upper right', fontsize=9, framealpha=0.8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    fig.suptitle('Prediction Waveform Comparison on Test Data',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Prediction waveform chart → {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("  深度学习期末作业 — RNN / GRU / LSTM 模型横向对比分析")
    print(f"  计算设备: {device}  |  训练 Epoch 数: {TOTAL_EPOCHS}")
    print(f"  随机种子: {RANDOM_SEED}  |  输出目录: {OUTPUT_DIR}")
    print("=" * 65)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: 训练三个模型 ----
    print("\n[Phase 1/3] 训练 RNN / GRU / LSTM 模型...")
    results = []
    for mt in MODEL_TYPES:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        r = train_one_model(mt, device, verbose=True)
        # 移除大对象，仅保留指标数据
        model_ref = r.pop('model')
        x_test_ref = r.pop('x_test')
        y_test_ref = r.pop('y_test')
        r['_model'] = model_ref
        r['_x_test'] = x_test_ref
        r['_y_test'] = y_test_ref
        results.append(r)

    # ---- Step 2: 推理评估 ----
    print("\n[Phase 2/3] 推理性能评估...")
    eval_results = {}
    for r in results:
        name = r['model_type']
        print(f"  评估 {name}...")
        ev = evaluate_model(r['_model'], r['_x_test'], r['_y_test'], device)
        eval_results[name] = ev
        print(f"    MSE={ev['mse']:.6f}  MAE={ev['mae']:.6f}  "
              f"Latency={ev['avg_inference_latency_us']:.1f} μs")

    # ---- Step 3: 生成图表 ----
    print("\n[Phase 3/3] 生成对比图表...")

    plot_loss_convergence(results, OUTPUT_DIR / 'loss_convergence.png')
    plot_training_efficiency(results, OUTPUT_DIR / 'training_efficiency.png')
    plot_convergence_speed(results, OUTPUT_DIR / 'convergence_speed.png')
    plot_inference_performance(results, eval_results,
                               OUTPUT_DIR / 'inference_performance.png')
    plot_radar_chart(results, eval_results, OUTPUT_DIR / 'cost_performance_radar.png')
    plot_summary_table(results, eval_results, OUTPUT_DIR / 'summary_table.png')
    plot_prediction_waveform(results, eval_results,
                             OUTPUT_DIR / 'prediction_waveform.png')

    # ---- 导出 JSON ----
    json_data = {}
    for r in results:
        name = r['model_type']
        json_data[name] = {
            'final_loss': r['final_loss'],
            'loss_history': r['loss_history'],
            'total_time_sec': r['total_time_sec'],
            'avg_epoch_time_ms': r['avg_epoch_time_ms'],
            'num_params': r['num_params'],
            'trainable_params': r['trainable_params'],
            'weight_file_bytes': r['weight_file_bytes'],
            'convergence': r['convergence'],
            'evaluation': eval_results[name],
        }

    json_path = OUTPUT_DIR / 'comparison_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] Structured data → {json_path}")

    # ---- 打印终端摘要 ----
    print("\n" + "=" * 65)
    print("  综合对比摘要")
    print("=" * 65)
    print(f"{'指标':<30} {'RNN':>10} {'GRU':>10} {'LSTM':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_type']} Final Loss:        {r['final_loss']:>10.6f}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_type']} Training Time:     {r['total_time_sec']:>9.1f}s")
    print("-" * 60)
    for r in results:
        print(f"{r['model_type']} Parameters:        {r['num_params']:>10d}")
    print("-" * 60)
    for r in results:
        name = r['model_type']
        print(f"{name} Inference Latency: {eval_results[name]['avg_inference_latency_us']:>9.1f}μs")
    print("-" * 60)
    for r in results:
        name = r['model_type']
        print(f"{name} Test MSE:          {eval_results[name]['mse']:>10.6f}")
    print("=" * 65)
    print(f"\n所有图表已保存至: {OUTPUT_DIR.resolve()}")
    print("分析完成!")


if __name__ == '__main__':
    main()
