"""
=============================================================================
 Data Loading Module — MIT-BIH Physiological Signal I/O & Sliding-Window Sampler

 Provides a complete pipeline from raw PhysioNet-format files to PyTorch
 DataLoader objects suitable for training recurrent models.

 How to use:
   from data_loader import load_mitbih_signal, get_dataloaders

   # Quick: load the full normalized signal
   signal = load_mitbih_signal('100')

   # Full pipeline: sliding windows → train/val DataLoaders
   train_loader, val_loader, stats = get_dataloaders(
       seq_len=500, batch_size=64, val_split=0.2
   )

   # Run unit tests:  python data_loader.py

 Pipeline steps:
   1. load_mitbih_signal() — reads .dat/.hea via wfdb, extracts MLII lead,
      applies Z-score normalization.
   2. create_sequences() — sliding-window cut with configurable stride
      (default stride=seq_len/4 → 75% overlap). Produces (x, y) pairs
      where y is the one-step-ahead target.
   3. SignalDataset + get_dataloaders() — wraps sequences as a PyTorch
      Dataset, splits by time order (80/20 default), returns DataLoaders.

 Design rationale for time-ordered split:
   For time-series forecasting, random shuffling across the full dataset
   leaks future information into training. We split chronologically: the
   first 80% of sequences train, the last 20% validate. This mirrors the
   real-world constraint that a deployed model can only use past data.

 Research significance:
   Full-dataset utilization (~650k samples, yielding thousands of training
   sequences after windowing) is critical for learning meaningful temporal
   representations. Subsampling to a few hundred points — a common shortcut
   in tutorial code — produces models that memorize a single waveform shape
   and fail to generalize. The 75% overlapping window strategy also ensures
   the model sees every possible phase alignment of the signal, improving
   robustness to temporal shift.
=============================================================================
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def load_mitbih_signal(record_name='100'):
    """
    读取 MIT-BIH 原始 .dat/.hea 文件，返回 Z-score 标准化后的完整信号

    参数:
        record_name: 不含扩展名的记录名（默认 '100'）
    返回:
        signal: numpy 数组 (total_length,)，已 Z-score 标准化
    """
    import wfdb

    record = wfdb.rdrecord(record_name)
    # 提取 MLII 导联（第一导联 — 最常用于心律失常分析）
    signal = record.p_signal[:, 0].astype(np.float32)
    # Z-score 标准化，加速模型收敛
    signal = (signal - signal.mean()) / signal.std()
    return signal


def create_sequences(signal, seq_len=500, stride=None):
    """
    滑窗切割：将一维信号转换为监督学习序列

    每一条序列: x = signal[i : i+seq_len], y = signal[i+1 : i+seq_len+1]
    即单步预测任务 — 用当前采样点预测下一个采样点

    参数:
        signal:  一维 numpy 数组，Z-score 标准化后的完整信号
        seq_len: 每条序列的长度（默认 500，与模型输入匹配）
        stride:  滑窗步长（默认 seq_len//4，即 75% 重叠）
    返回:
        x: numpy 数组 (num_sequences, seq_len, 1)
        y: numpy 数组 (num_sequences, seq_len, 1)
    """
    if stride is None:
        stride = max(1, seq_len // 4)

    total_len = len(signal)
    # 每个滑窗需要 seq_len+1 个点（x 取前 seq_len 个，y 取后 seq_len 个）
    num_sequences = (total_len - seq_len - 1) // stride + 1

    x_list, y_list = [], []
    for i in range(num_sequences):
        start = i * stride
        end = start + seq_len
        x_list.append(signal[start:end])
        y_list.append(signal[start + 1:end + 1])

    x = np.array(x_list, dtype=np.float32).reshape(num_sequences, seq_len, 1)
    y = np.array(y_list, dtype=np.float32).reshape(num_sequences, seq_len, 1)

    return x, y


class SignalDataset(Dataset):
    """PyTorch Dataset 封装：将滑窗产物包装为可迭代的数据集"""

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(seq_len=500, batch_size=64, val_split=0.2, stride=None):
    """
    加载全量 MIT-BIH 数据，生成训练集和验证集 DataLoader

    数据划分策略：
      - 按时间顺序切分（不打乱时间序列），前 (1-val_split) 用于训练，后 val_split 用于验证
      - 这是时序预测的标准做法，避免未来信息泄漏到训练中

    参数:
        seq_len:    序列长度（默认 500）
        batch_size: 批量大小（默认 64）
        val_split:  验证集比例（默认 0.2）
        stride:     滑窗步长（默认 seq_len//4）

    返回:
        train_loader: 训练集 DataLoader（shuffle=True，打乱序列顺序）
        val_loader:   验证集 DataLoader（shuffle=False）
        stats:        字典，包含 'num_train', 'num_val', 'total_samples', 'signal_mean', 'signal_std'
    """
    # 加载完整信号
    signal = load_mitbih_signal('100')

    # 滑窗切割
    x, y = create_sequences(signal, seq_len=seq_len, stride=stride)

    total_sequences = len(x)
    # 按时间顺序切分：前部训练，后部验证
    split_idx = int(total_sequences * (1 - val_split))
    # 确保验证集至少有一个 batch
    if split_idx >= total_sequences:
        split_idx = total_sequences - max(1, total_sequences // 10)
    if split_idx < 1:
        split_idx = 1

    x_train, y_train = x[:split_idx], y[:split_idx]
    x_val, y_val = x[split_idx:], y[split_idx:]

    train_dataset = SignalDataset(x_train, y_train)
    val_dataset = SignalDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)

    stats = {
        'total_samples': total_sequences,
        'num_train': len(train_dataset),
        'num_val': len(val_dataset),
    }

    return train_loader, val_loader, stats


# ==========================================
# 单元测试：验证数据加载流程
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  数据加载模块单元测试")
    print("=" * 60)

    # 测试信号加载
    signal = load_mitbih_signal('100')
    print(f"\n[信号加载]")
    print(f"  总长度: {len(signal):,} 个采样点")
    print(f"  采样率: 360 Hz → 约 {len(signal)/360/60:.1f} 分钟")
    print(f"  均值: {signal.mean():.6f} (Z-score 后应接近 0)")
    print(f"  标准差: {signal.std():.6f} (Z-score 后应接近 1)")
    print(f"  值域: [{signal.min():.4f}, {signal.max():.4f}]")

    # 测试滑窗切割
    seq_len = 500
    stride = seq_len // 4  # 125
    x, y = create_sequences(signal, seq_len=seq_len, stride=stride)
    print(f"\n[滑窗切割] seq_len={seq_len}, stride={stride}")
    print(f"  生成序列数: {len(x):,}")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    # 验证 y[i] ≈ 下一时间步
    assert abs(x[0, -1, 0] - y[0, -2, 0]) < 1e-4, "单步预测对齐检查失败"
    print(f"  单步对齐检查: x[0,-1]={x[0,-1,0]:.4f}, y[0,-2]={y[0,-2,0]:.4f} [OK]")

    # 测试 DataLoader
    train_loader, val_loader, stats = get_dataloaders(
        seq_len=seq_len, batch_size=64, val_split=0.2, stride=stride
    )
    print(f"\n[DataLoader]")
    print(f"  总序列数: {stats['total_samples']:,}")
    print(f"  训练集: {stats['num_train']:,} 条序列 ({stats['num_train']/stats['total_samples']*100:.1f}%)")
    print(f"  验证集: {stats['num_val']:,} 条序列 ({stats['num_val']/stats['total_samples']*100:.1f}%)")
    print(f"  训练 batch 数: {len(train_loader)}")
    print(f"  验证 batch 数: {len(val_loader)}")

    # 检查一个 batch
    xb, yb = next(iter(train_loader))
    print(f"  单个 batch x shape: {xb.shape}, y shape: {yb.shape}")

    print(f"\n{'=' * 60}")
    print("  数据加载模块测试全部通过！")
    print(f"{'=' * 60}")
