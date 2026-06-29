"""
=============================================================================
 Shared Model Module — Unified RNN / GRU / LSTM Factory

 A single model class (SleepRNNDemo) that supports all three recurrent cell
 types via a string parameter. Shared by train.py and sever.py to avoid code
 duplication and guarantee that the inference server loads exactly the same
 architecture used during training.

 How to use:
   from models import SleepRNNDemo

   model = SleepRNNDemo(cell_type='LSTM', hidden_size=64, num_layers=2)
   # cell_type: 'RNN' | 'GRU' | 'LSTM'

   # Unit tests:  python models.py

 Architecture:
   RNN core (1-2 layers) → Dropout → small MLP head (hidden → 16 → 1)
   The MLP head provides additional non-linear capacity beyond a single
   linear projection, improving prediction fidelity for non-trivial signals.

   Key parameters:
     hidden_size   — dimensionality of the recurrent state (default 64).
     num_layers    — stacked RNN depth (default 2). Dropout is applied
                      between layers when num_layers ≥ 2.
     dropout       — regularization strength (default 0.1).
     bidirectional — if True, processes the sequence in both directions
                      (not recommended for causal/online prediction).

 Research significance:
   A uniform factory pattern makes architecture comparison rigorous. By
   changing a single string ('RNN' → 'GRU' → 'LSTM') while keeping all
   other hyperparameters identical, researchers can isolate the effect of
   the recurrent cell type on forecasting accuracy. The MLP output head
   also serves as a simple ablation point: removing it (replace with a
   single Linear) quantifies how much the extra non-linearity matters.
=============================================================================
"""
import torch
import torch.nn as nn


class SleepRNNDemo(nn.Module):
    """
    统一模型工厂 — 通过 cell_type 参数切换 RNN / GRU / LSTM

    参数:
        cell_type:   'RNN' | 'GRU' | 'LSTM'
        input_size:  输入特征维度（默认 1，单导联信号）
        hidden_size: 隐藏层维度（默认 64，提升容量）
        num_layers:  堆叠层数（默认 2，增强深度）
        dropout:     Dropout 比率（默认 0.1，仅 num_layers≥2 时生效）
        bidirectional: 是否使用双向 RNN（默认 False）
        output_size: 输出维度（默认 1，预测下一个采样点）
    """

    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.1, bidirectional=False, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # RNN dropout 仅在 num_layers >= 2 时有效
        rnn_dropout = dropout if num_layers > 1 else 0.0

        # 三种核心时序组件
        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=rnn_dropout,
                                   bidirectional=bidirectional)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=rnn_dropout,
                                   bidirectional=bidirectional)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=rnn_dropout,
                                    bidirectional=bidirectional)
        else:
            raise ValueError("未知的网络类型！请选择 'RNN', 'GRU' 或 'LSTM'")

        # 方向数：双向 ×2
        self.direction_factor = 2 if bidirectional else 1
        fc_input = hidden_size * self.direction_factor

        # Dropout 层（应用于 RNN 输出后、MLP 前）
        self.dropout = nn.Dropout(dropout)

        # 小 MLP 输出头：替代单层 Linear，增强非线性拟合能力
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # 输出头用一半 dropout
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        """
        前向传播 — LSTM 需要双隐状态 (h, c)，RNN/GRU 只需 h

        参数:
            x: Tensor (batch, seq_len, input_size)
        返回:
            out: Tensor (batch, seq_len, output_size) — 每个时间步的预测
        """
        device = x.device
        batch_size = x.size(0)
        directions = self.direction_factor

        h0 = torch.zeros(self.num_layers * directions, batch_size,
                         self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * directions, batch_size,
                             self.hidden_size).to(device)
            rnn_out, _ = self.rnn_core(x, (h0, c0))
        else:
            rnn_out, _ = self.rnn_core(x, h0)

        # Dropout 正则化后送入输出 MLP
        rnn_out = self.dropout(rnn_out)
        return self.fc(rnn_out)


# ==========================================
# 单元测试：验证模型可正常实例化与前向传播
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  模型模块单元测试")
    print("=" * 60)

    for cell_type in ['RNN', 'GRU', 'LSTM']:
        model = SleepRNNDemo(cell_type=cell_type)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 模拟小批量输入 (batch=4, seq_len=100, input_size=1)
        x = torch.randn(4, 100, 1)
        with torch.no_grad():
            out = model(x)

        print(f"\n[{cell_type}]")
        print(f"  参数量: {total_params:,} (可训练: {trainable:,})")
        print(f"  输入 shape: {x.shape} → 输出 shape: {out.shape}")
        assert out.shape == (4, 100, 1), f"输出 shape 错误: {out.shape}"

    print(f"\n{'=' * 60}")
    print("  所有模型测试通过！")
    print(f"{'=' * 60}")
