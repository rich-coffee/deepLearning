#  ECG  Oscilloscope — Real-Time Physiological Signal Prediction & Visualization

> **RNN / GRU / LSTM  single-step forecasting + browser-based oscilloscope monitoring**
>
> Trains recurrent neural networks on the full MIT-BIH Arrhythmia Database Record 100 and streams predictions to a D3.js real-time HMI panel via WebSocket.

---

##  Project Structure

```
deepLearning/
├── train.py                       # Training script: full MIT-BIH data → RNN/GRU/LSTM
├── models.py                      # Shared model factory: SleepRNNDemo (used by train.py & sever.py)
├── data_loader.py                 # Data pipeline: sliding-window sampler + PyTorch DataLoader
├── sever.py                       # WebSocket inference server: streams predictions to browser
├── index.html                     # Frontend panel: D3.js oscilloscope + telemetry readouts
├── d3.v7.min.js                   # D3.js v7 local copy (273 KB), no CDN required
├── requirements.txt               # Python dependencies
├── .gitignore
├── 100.dat / 100.hea / 100.atr    # MIT-BIH Arrhythmia Database Record 100 (~30 min ECG)
└── MIT-BIH_data/
    ├── weight/
    │   ├── sleep_rnn_weights.pth    # RNN trained weights
    │   ├── sleep_gru_weights.pth    # GRU trained weights
    │   └── sleep_lstm_weights.pth   # LSTM trained weights
    └── result/
        ├── MIT-BIH数据RNN训练结果.png
        ├── MIT-BIH数据GRU训练结果.png
        └── MIT-BIH数据LSTM训练结果.png
```

---

##  Quick Start

### Prerequisites

- Python 3.10+
- A modern browser (Chrome / Edge / Firefox)

### 1. Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows

# ---- CPU-only (no NVIDIA GPU required) ----
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ---- GPU (CUDA 12.4, recommended for training) ----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Train a model

```bash
python train.py
```

Edit the configuration block at the top of [train.py](train.py) to switch architectures or tune hyperparameters:

```python
MODEL_TYPE = 'LSTM'   # 'RNN' | 'GRU' | 'LSTM'
```

**Training features:**
- **Full dataset** — uses all ~650,000 samples from Record 100, sliding-windowed into thousands of sequences
- **Mini-batch training** — default batch_size=64, GPU-compatible
- **Validation** — 80/20 chronological split, evaluated every epoch
- **Early stopping** — terminates when validation loss plateaus (patience=20)
- **Learning rate scheduling** — ReduceLROnPlateau with automatic decay
- **Gradient clipping** — max norm = 1.0, prevents explosion in deep RNNs
- **Checkpoint resume** — interrupted runs automatically continue from last epoch
- **Best-model export** — saves the weight with lowest validation loss

### 3. Start the WebSocket inference server

```bash
python sever.py
```

Expected output:

```
=============================================
  [SYS] 工业级边缘计算遥测终端已启动
  [SYS] 当前挂载计算核心: LSTM 神经网络
  [SYS] 数据来源: MIT-BIH 心律失常数据库 Record 100 (真实生理信号)
  [SYS] 监听端口: ws://127.0.0.1:8765
=============================================
```

### 4. Open the monitoring panel

Open `index.html` directly in a browser. The panel auto-connects via WebSocket; the status indicator changes from **SYS OFFLINE** to **DATA LINK ACTV** and waveforms begin rendering in real time.

> D3.js is bundled locally (`d3.v7.min.js`) — no internet connection is required to open the panel.

---

##  How to Use

### Switching model architectures

1. Set `MODEL_TYPE` in [sever.py](sever.py) (must match the trained weights):
   ```python
   MODEL_TYPE = 'GRU'  # 'RNN' | 'GRU' | 'LSTM'
   ```
2. Restart `python sever.py`.
3. Refresh the browser to observe prediction differences across architectures.

### Tuning hyperparameters

Edit the configuration block in [train.py](train.py) (lines ~36–60):

```python
MODEL_TYPE = 'LSTM'          # architecture
SEQ_LEN = 500                # sliding-window length (samples)
BATCH_SIZE = 64              # mini-batch size
HIDDEN_SIZE = 64             # recurrent state dimension
NUM_LAYERS = 2               # stacked RNN depth
DROPOUT = 0.1                # dropout ratio
LEARNING_RATE = 0.001        # initial learning rate (Adam)
TOTAL_EPOCHS = 300           # maximum epochs
EARLY_STOP_PATIENCE = 20     # early-stopping patience
```

### Running unit tests

Each Python module includes a `__main__` block that validates its core functionality:

```bash
python data_loader.py   # verify signal loading, windowing, DataLoader
python models.py        # verify RNN/GRU/LSTM instantiation & forward pass
```

---

##  Research Significance

### Why real-time visualization matters

Scalar loss metrics (MSE, MAE) provide a single number that summarizes model performance, but they obscure the *dynamics* of prediction error. A model that achieves low MSE may still exhibit:

- **Phase lag** — the prediction tracks the correct shape but is consistently shifted in time.
- **Amplitude drift** — the prediction envelope decays or explodes relative to the ground truth.
- **Transient blindness** — the model fails during sudden waveform changes (e.g., arrhythmic beats) while performing well on regular rhythms.

The oscilloscope panel makes these failure modes immediately visible, enabling qualitative debugging that complements quantitative evaluation.

### Why full-dataset training

Using all ~650,000 samples from MIT-BIH Record 100 (rather than subsampling a few hundred points) produces models that learn genuine temporal structure rather than memorizing a single waveform snippet. The 75%-overlap sliding window ensures the model sees the signal at every possible phase alignment, improving robustness to temporal shift.

### Why a unified model factory

By keeping all hyperparameters identical and changing only the cell type (`RNN` → `GRU` → `LSTM`), researchers can isolate the effect of the recurrent architecture on forecasting accuracy. The MLP output head (hidden → 16 → 1) also serves as an ablation point: replace it with a single `nn.Linear` to quantify how much the extra non-linearity contributes.

### Why a real physiological benchmark

MIT-BIH is one of the most widely cited datasets in biomedical signal processing. Training on real ECG data — rather than synthetic sine waves — ensures that the learned dynamics reflect actual biological signal characteristics, making the results relevant to real-world applications such as anomaly detection, compression, and denoising.

---

##  Data Source

> **All training data comes from real physiological signals.**

- Data files `100.dat`, `100.hea`, `100.atr` are from the **MIT-BIH Arrhythmia Database** ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/)), one of the most authoritative public benchmarks in physiological signal processing.
- Record 100 contains approximately 30 minutes of two-lead ambulatory ECG, sampled at 360 Hz, totaling ~650,000 sample points.
- Training uses all 650,000 points; the inference server streams a held-out segment to demonstrate generalization.
- The MLII lead (most commonly used for arrhythmia analysis) is extracted and Z-score normalized.

---

##  Technology Stack

| Layer | Technology | Role |
|-------|------------|------|
| Deep learning | PyTorch | RNN / GRU / LSTM training & inference |
| Data I/O | wfdb | Parse PhysioNet-format MIT-BIH files |
| Real-time streaming | WebSocket (websockets library) | Low-latency server→browser data push |
| Waveform rendering | D3.js v7 (local copy) | 150-point sliding-window SVG rendering |
| Frontend layout | Flexbox + CSS3 | Responsive oscilloscope + telemetry panel |

---

##  Notes

- Start `sever.py` before opening `index.html`, or the panel will display **SYS OFFLINE**.
- `index.html` uses a local `d3.v7.min.js` — no internet connection is needed.
- Training requires the MIT-BIH data files (`100.dat`, `100.hea`, `100.atr`) in the project root.
- Full-dataset training takes ~3–5 minutes on CPU; a GPU significantly reduces wall-clock time.
- A virtual environment is recommended to avoid dependency conflicts.
