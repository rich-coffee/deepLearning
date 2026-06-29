"""
=============================================================================
 Training Script — RNN / GRU / LSTM for Physiological Signal Prediction

 Trains a recurrent neural network on the full MIT-BIH Arrhythmia Database
 Record 100 (~650,000 samples) to predict the next signal value from a
 sliding window of past values. This is a single-step forecasting task:
 given x[t-seq_len : t], predict x[t+1].

 How to use:
   python train.py

   Edit the configuration block below (MODEL_TYPE, SEQ_LEN, BATCH_SIZE, etc.)
   to experiment with different architectures and hyperparameters.

   MODEL_TYPE: 'RNN' | 'GRU' | 'LSTM'

 Key training features:
   - Full-dataset training — uses all ~650k sampling points from Record 100,
     sliding-windowed into thousands of training sequences.
   - Time-ordered train/val split (80/20) — prevents future information leakage.
   - Early stopping with patience — terminates when validation loss plateaus.
   - ReduceLROnPlateau scheduler — automatically decays learning rate.
   - Gradient clipping (max norm = 1.0) — prevents gradient explosion in deep RNNs.
   - Checkpoint resume — interrupted runs automatically continue from last epoch.
   - Best-model export — saves the weight with lowest validation loss, not just
     the final epoch.

 Research significance:
   Training on a real, publicly available physiological benchmark (MIT-BIH)
   rather than synthetic data ensures the learned dynamics reflect actual
   biological signal characteristics. The single-step prediction task serves
   as a baseline for evaluating how well different recurrent architectures
   (vanilla RNN, GRU, LSTM) capture the short-term temporal structure of
   human ECG signals. The best-model export + checkpointing workflow also
   makes the training reproducible and robust to HPC preemption.
=============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys

from models import SleepRNNDemo
from data_loader import get_dataloaders

# ==========================================
# 配置：通过修改变量一键切换 RNN / GRU / LSTM
# 可选值: 'RNN' | 'GRU' | 'LSTM'
# ==========================================
MODEL_TYPE = 'GRU'

# ---- 数据配置 ----
SEQ_LEN = 500                # 序列长度（滑动窗口大小，单位：采样点）
BATCH_SIZE = 64              # 小批量大小
VAL_SPLIT = 0.2              # 验证集比例（按时间顺序切分）
STRIDE = 125                 # 滑窗步长（= seq_len/4，75% 重叠保证数据充分利用）

# ---- 模型超参数 ----
HIDDEN_SIZE = 64             # 隐藏层维度
NUM_LAYERS = 2               # RNN 堆叠层数
DROPOUT = 0.1                # Dropout 正则化比率

# ---- 训练超参数 ----
LEARNING_RATE = 0.001        # 初始学习率（Adam）
TOTAL_EPOCHS = 300           # 最大训练轮数
EARLY_STOP_PATIENCE = 20     # 早停耐心值（验证损失连续不改善则停止）
GRAD_CLIP = 1.0              # 梯度裁剪最大范数
LR_PATIENCE = 10             # 学习率调度耐心值
LR_FACTOR = 0.5              # 学习率衰减因子

# ---- 路径配置 ----
CHECKPOINT_PATH = f'sleep_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'MIT-BIH_data\\weight\\sleep_{MODEL_TYPE.lower()}_weights.pth'
BEST_WEIGHTS_PATH = f'MIT-BIH_data\\weight\\sleep_{MODEL_TYPE.lower()}_best.pth'


# ----------------- 1. 信号捕获与安全性配置（HPC 共享集群友好） -----------------
# 当作业调度器（如 SLURM）回收资源时，捕获信号并优雅保存进度

def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})！正在紧急保存当前进度...")
    save_checkpoint(epoch, model, optimizer,
                    train_loss, val_loss, lr_scheduler,
                    path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型的进度已安全保存，程序优雅退出。")
    sys.exit(0)


signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss,
                    scheduler=None, path=CHECKPOINT_PATH):
    """保存完整训练快照，支持断点续训"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"-> 检查点已保存至: {path}")


# ----------------- 2. 工具函数 -----------------

def compute_loss(model, data_loader, criterion, device):
    """在给定 DataLoader 上计算平均损失"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
    return total_loss / total_samples if total_samples > 0 else float('inf')


# ----------------- 3. 主训练逻辑 — 断点续训 + 早停 + 自动导出 -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"  Physiological Signal Prediction — Training 【{MODEL_TYPE}】 Model")
    print(f"  计算设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    print("=" * 65)

    # ---- 加载数据 ----
    print("\n[数据] 加载 MIT-BIH Record 100 全量数据...")
    train_loader, val_loader, data_stats = get_dataloaders(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT, stride=STRIDE
    )
    print(f"  总序列数: {data_stats['total_samples']:,}")
    print(f"  训练集:   {data_stats['num_train']:,} 条 ({len(train_loader)} batches)")
    print(f"  验证集:   {data_stats['num_val']:,} 条 ({len(val_loader)} batches)")
    print(f"  序列长度: {SEQ_LEN} 点 ({SEQ_LEN/360:.1f}s @ 360Hz)")
    print(f"  Batch 大小: {BATCH_SIZE}")

    # ---- 实例化模型 ----
    print(f"\n[模型] 构建 {MODEL_TYPE} (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT})")
    model = SleepRNNDemo(
        cell_type=MODEL_TYPE,
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}  (可训练: {trainable_params:,})")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    train_loss = 0.0
    val_loss = 0.0

    # ---- 断点续训 ----
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[恢复] 发现【{MODEL_TYPE}】的历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint.get('train_loss', 0.0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 恢复最佳验证损失（从最佳权重文件推断）
        if os.path.exists(BEST_WEIGHTS_PATH):
            best_checkpoint = torch.load(BEST_WEIGHTS_PATH, map_location='cpu')
            # 用当前验证集评估最佳模型以获取基准
            model.load_state_dict(best_checkpoint)
            best_val_loss = compute_loss(model, val_loader, criterion, device)
            # 恢复当前模型
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  已恢复最佳验证损失基准: {best_val_loss:.6f}")
        print(f"成功恢复！从第 {start_epoch} 个 Epoch 继续。")

    # ---- 确保输出目录存在 ----
    os.makedirs('MIT-BIH_data\\weight', exist_ok=True)

    # ---- 训练循环 ----
    print(f"\n[训练] 开始训练（最多 {TOTAL_EPOCHS} epoch，早停耐心={EARLY_STOP_PATIENCE}）")
    print("-" * 65)

    try:
        for epoch in range(start_epoch, TOTAL_EPOCHS):
            # ----- 训练阶段 -----
            model.train()
            epoch_train_loss = 0.0
            train_samples = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()

                epoch_train_loss += loss.item() * batch_x.size(0)
                train_samples += batch_x.size(0)

            train_loss = epoch_train_loss / train_samples

            # ----- 验证阶段 -----
            val_loss = compute_loss(model, val_loader, criterion, device)

            # ----- 学习率调度 -----
            current_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(val_loss)

            # ----- 日志输出 -----
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                marker = ""
                if val_loss < best_val_loss:
                    marker = " ★"
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    # 保存最佳模型
                    torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
                print(f"[{MODEL_TYPE}] Epoch {epoch+1:3d}/{TOTAL_EPOCHS} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}{marker}")

            # ----- 早停检查 -----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0
                torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n[早停] 验证损失连续 {EARLY_STOP_PATIENCE} 轮未改善，停止训练。")
                    break

            # ----- 定期保存检查点 -----
            if (epoch + 1) % 20 == 0:
                save_checkpoint(epoch, model, optimizer,
                                train_loss, val_loss, lr_scheduler,
                                path=CHECKPOINT_PATH)

        # ---- 训练完成 ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[完成] {MODEL_TYPE} 模型训练结束！")
        print(f"  最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
        print(f"  训练总轮数:   {epoch + 1}")

        # 导出部署权重（使用最佳模型）
        if os.path.exists(BEST_WEIGHTS_PATH):
            model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
            print(f"  已加载最佳模型权重用于部署导出。")

        torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
        print(f"  部署权重已保存至: {FINAL_WEIGHTS_PATH}")
        # 清理检查点文件
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print(f"  已清理检查点文件。")

    except Exception as e:
        print(f"\n[错误] 训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer,
                       train_loss, val_loss, lr_scheduler,
                       path=CHECKPOINT_PATH)
        raise
