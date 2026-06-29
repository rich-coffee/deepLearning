"""
=============================================================================
 WebSocket Real-time Inference Server for Physiological Signal Prediction

 Loads a trained RNN/GRU/LSTM model and streams single-step predictions to a
 browser-based oscilloscope panel via WebSocket. Designed as a lightweight
 edge-computing telemetry terminal for research prototyping.

 How to use:
   1. Train a model first:  python train.py
   2. Start this server:    python sever.py
   3. Open index.html in a browser to view the live waveform display.

 Configuration (top of file):
   MODEL_TYPE  — 'RNN' | 'GRU' | 'LSTM', must match the trained weights.
   WEIGHTS_PATH — path to the .pth checkpoint (auto-derived from MODEL_TYPE).
   DEMO_SIGNAL_START / DEMO_SIGNAL_LENGTH — which segment of the MIT-BIH
     record to stream; use an offset far from the training split to
     demonstrate generalization.

 Key features:
   - Real physiological data — streams from MIT-BIH Record 100 ECG.
   - Graceful degradation — if weights or data files are missing, falls back
     to untrained weights or synthetic signals rather than crashing.
   - SCADA-style JSON telemetry frames with simulated edge latency.

 Research significance:
   This server closes the sense→predict→visualize loop. By streaming a held-out
   segment of real physiological data through a trained recurrent model, it
   provides qualitative, frame-by-frame insight into the model's dynamic
   behavior — something aggregate metrics (MSE, MAE) cannot capture. Phase
   errors, amplitude drift, and transient failures become immediately visible
   on the oscilloscope, guiding further model refinement.
=============================================================================
"""
import torch
import numpy as np
import asyncio
import json
import time
import random
import sys
import os

from models import SleepRNNDemo
from data_loader import load_mitbih_signal

# ==========================================
# 工业现场配置 — SCADA 遥测终端参数
# ==========================================
MODEL_TYPE = 'LSTM'  # 可选: 'RNN' | 'GRU' | 'LSTM'，与训练脚本保持一致
WEIGHTS_PATH = f'MIT-BIH_data\\weight\\sleep_{MODEL_TYPE.lower()}_weights.pth'
HOST = "127.0.0.1"
PORT = 8765

# 推理演示使用的信号段（MIT-BIH Record 100 中取一段未参与训练的区间作为演示）
DEMO_SIGNAL_START = 500000   # 信号起始位置（远离默认训练区间）
DEMO_SIGNAL_LENGTH = 3000    # 演示数据点数


# 1. WebSocket 实时遥测数据流 — 使用真实 MIT-BIH 生理信号
async def stream_data(websocket):
    print(f"\n[终端接入] 客户端已连接。开始下发 {MODEL_TYPE} 遥测数据...")

    # 初始化模型并加载预训练权重
    model = SleepRNNDemo(cell_type=MODEL_TYPE)
    model.eval()

    weights_loaded = False
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        weights_loaded = True
        print(f"[系统] 成功加载 MIT-BIH 预训练权重: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"[警告] 未找到 {WEIGHTS_PATH}，将使用未训练的初始权重进行仿真演示。")

    # 加载真实 MIT-BIH 生理信号作为演示数据流
    try:
        full_signal = load_mitbih_signal('100')
        # 截取信号的一个片段（避免与训练数据重合，展示泛化能力）
        start = min(DEMO_SIGNAL_START, len(full_signal) - DEMO_SIGNAL_LENGTH - 1)
        test_signal = full_signal[start:start + DEMO_SIGNAL_LENGTH + 1]
        signal_source = f"MIT-BIH Record 100 (offset={start})"
    except Exception:
        # 极端情况：数据文件不存在，回退到合成信号
        print("[警告] 无法读取 MIT-BIH 数据文件，使用合成信号作为回退方案。")
        t = np.linspace(0, 100, DEMO_SIGNAL_LENGTH + 1)
        test_signal = np.sin(t) + 0.5 * np.cos(t * 2.5) + np.random.normal(0, 0.05, t.shape)
        signal_source = "合成信号 (sin+cos+noise)"

    print(f"[数据源] {signal_source}")
    if weights_loaded:
        print(f"[推理] 使用训练好的 {MODEL_TYPE} 模型进行单步预测")
    else:
        print(f"[推理] 使用未训练的 {MODEL_TYPE} 模型 — 预测结果仅供参考")

    try:
        with torch.no_grad():
            for i in range(len(test_signal) - 1):
                start_time = time.time()

                # 单步推理：输入当前采样点，预测下一采样点
                input_point = torch.tensor([[[test_signal[i]]]], dtype=torch.float32)
                pred_point = model(input_point).item()
                actual_point = test_signal[i + 1]

                # 模拟工业边缘计算延迟（真实计算耗时 + 网络传输抖动）
                calc_time = (time.time() - start_time) * 1000
                simulated_latency = calc_time + random.uniform(5.0, 15.0)

                # 打包工业标准 JSON 遥测帧
                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": MODEL_TYPE,
                    "ch1_actual": float(actual_point),       # 真实波形值
                    "ch2_predict": float(pred_point),        # 神经网络预测值
                    "error_abs": abs(float(actual_point) - float(pred_point)),  # 绝对误差
                    "latency_ms": round(simulated_latency, 2) # 链路延迟
                }
                await websocket.send(json.dumps(payload))

                # 采样率控制：~33ms/帧 ≈ 30 FPS（工业遥测典型刷新率）
                await asyncio.sleep(0.03)
    except Exception as e:
        print(f"[断开] 客户端连接中断或发生异常: {e}")


async def main():
    import websockets
    async with websockets.serve(stream_data, HOST, PORT):
        print("=============================================")
        print(f"  [SYS] 工业级边缘计算遥测终端已启动")
        print(f"  [SYS] 当前挂载计算核心: {MODEL_TYPE} 神经网络")
        print(f"  [SYS] 数据来源: MIT-BIH 心律失常数据库 Record 100 (真实生理信号)")
        print(f"  [SYS] 监听端口: ws://{HOST}:{PORT}")
        print("=============================================")
        print("等待前端监控面板接入...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
