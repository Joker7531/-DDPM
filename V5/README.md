# UAR-ACSSNet: 单通道 EEG 去伪影/重建系统

> **Unified Artifact Removal with Axis-Conditioned Selective Scan Network**
> 
> 基于 PyTorch 的端到端单通道 EEG 信号去噪与重建框架

---

## 📋 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [数据集格式](#数据集格式)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [训练与验证](#训练与验证)
- [推理与部署](#推理与部署)
- [技术细节](#技术细节)
- [可复现性](#可复现性)

---

## 概述

UAR-ACSSNet 是一个专为单通道 EEG 信号去伪影设计的深度学习模型。它结合了时域和时频域信息，通过轴向条件选择性扫描（Axis-Conditioned Selective Scan）机制，实现高保真的信号重建和自适应置信度估计。

**核心创新点：**
1. **时域-时频双分支架构**：1D U-Net 负责时域重建，2D 时频编码器提取频域特征
2. **ACSSBlock（轴向条件选择性扫描块）**：可替代的 Mamba-like 扫描机制，沿时间和频率轴聚合信息
3. **FiLM 跨域融合**：将时频特征通过 Feature-wise Linear Modulation 调制时域解码器
4. **自适应置信图**：输出逐样本点的置信度，支持加权损失

---

## 核心特性

- ✅ **端到端可训练**：从原始 EEG 到去噪信号，无需手工特征提取
- ✅ **时频联合建模**：STFT 固定参数（fs=500Hz, 1-100Hz），确保可复现
- ✅ **可替代扫描机制**：ACSSBlock 内部使用简化的 depthwise conv 模拟扫描，未来可替换为真实 Mamba/SSM
- ✅ **灵活的数据集支持**：支持变长信号、滑窗切片、Z-score 归一化
- ✅ **完善的损失函数**：Charbonnier 重建损失 + 置信图正则（TV + 熵）+ 一致性损失接口
- ✅ **完整的训练框架**：包含 train/val/test 数据加载、梯度裁剪、学习率调度、模型保存

---

## 模型架构

### 整体结构

```
输入 x_raw (B, 1, L)
    ├─ 时域分支: 1D U-Net
    │   ├─ Encoder (4 层下采样)
    │   ├─ Bottleneck (残差块)
    │   └─ Decoder (4 层上采样 + skip connections)
    │       └─ FiLM 调制 (由时频特征生成 α, β)
    │
    ├─ 时频分支: STFT + SpecEncoder2D + ACSSStack
    │   ├─ STFT (固定参数: n_fft=512, hop=64, win=156)
    │   │   └─ 选择 1-100 Hz bins → (B, F_sel, T)
    │   ├─ SpecEncoder2D: (B, F_sel, T) → (B, C, T, F)
    │   └─ ACSSStack (K 层 ACSSBlock2D)
    │       └─ 每层: Axis Summary → Gate → Selective Scan → Mixture → Residual
    │
    └─ 融合 & 输出
        ├─ FiLM Generator: 从时频特征生成调制参数
        ├─ 重建输出: y_hat (B, 1, L)
        └─ 置信图: w (B, 1, L) ∈ [0, 1]
```

### ACSSBlock2D 详解

**输入/输出**: `(B, C, T, F)`  
**包含模块**:

1. **Axis Summary（证据提取）**
   ```
   频轴摘要: (B,C,T,F) --pool F--> (B,2C,T)  [mean+std]
   时轴摘要: (B,C,T,F) --pool T--> (B,2C,F)
   ```

2. **Axis-conditioned Gate（轴向自适应门控）**
   ```
   s_f (B,2C,T) --MLP--> g_freq (B,1,T) ∈ [0,1]
   ```

3. **Selective Scan Mixture**
   ```
   U_freq = ScanFreq(X)  # 沿频率轴扫描
   U_time = ScanTime(X)  # 沿时间轴扫描
   Y = g * U_freq + (1-g) * U_time
   ```
   
   **注**：`ScanFreq/ScanTime` 当前使用 depthwise Conv1D 模拟，接口设计便于替换为真实 Mamba/SSM

4. **Residual + Norm**
   ```
   out = X + Proj(Y)
   ```

### FiLM 调制机制

从时频分支提取的特征 `X_tf (B, C, T, F)` 经过频率维 pooling 得到 `m(t) (B, C, T)`，插值到时域长度 `L` 后生成：

```
α, β (B, C_dec, L)
H' = α ⊙ H + β  # 对 U-Net decoder 的特征进行调制
```

应用于 decoder 的前两层（对应 2 个最高分辨率层），实现跨域信息融合。

---

## 项目结构

```
cDDPM/V5/
├── datasets/
│   ├── __init__.py
│   ├── eeg_pair_dataset.py    # EEGPairDataset 类
│   ├── build_loaders.py       # 数据加载器构建函数
│   └── transforms.py          # 数据增强变换
├── signal_processing/
│   ├── __init__.py
│   └── stft_utils.py          # STFTProcessor (固定参数 STFT)
├── models/
│   ├── __init__.py
│   └── uar_acssnet.py         # 完整模型实现
│       ├── ResidualBlock1D, DownBlock1D, UpBlock1D
│       ├── UNet1D (时域主干)
│       ├── DepthwiseScan1D, ScanFreq, ScanTime (可替代扫描)
│       ├── ACSSBlock2D (核心模块)
│       ├── SpecEncoder2D (时频编码器)
│       ├── FiLMGenerator1D (跨域融合)
│       └── UAR_ACSSNet (完整模型)
├── train/
│   ├── __init__.py
│   ├── losses.py              # 损失函数
│   │   ├── CharbonnierLoss, HuberLoss
│   │   ├── ConfidenceRegularization (TV + 熵)
│   │   ├── ConsistencyLoss (一致性损失接口)
│   │   └── compute_losses (总损失计算)
│   └── min_train.py           # 训练入口
│       ├── train_one_epoch, validate
│       ├── train (完整训练循环)
│       └── main_minimal_example (随机数据测试)
├── configs/
│   ├── __init__.py
│   └── default.py             # 默认配置
├── inference_file.py          # 🆕 文件级推理脚本
├── visualize_inference.py     # 🆕 推理结果可视化
├── test_inference.py          # 🆕 推理功能测试
├── example_inference_api.py   # 🆕 Python API 使用示例
├── main.py                    # 训练主入口
├── INFERENCE_README.md        # 🆕 推理完整文档
└── README.md                  # 本文档
```

---

## 数据集格式

### 目录结构（固定）

```
Dataset/
├── train/
│   ├── raw/       # 原始带伪影信号
│   │   ├── 0001.npy
│   │   ├── 0002.npy
│   │   └── ...
│   └── clean/     # 干净参考信号
│       ├── 0001.npy
│       ├── 0002.npy
│       └── ...
├── val/
│   ├── raw/
│   └── clean/
└── test/
    ├── raw/
    └── clean/
```

### 文件格式要求

- **格式**: `.npy` (NumPy array)
- **Shape**: `(L,)` 或 `(1, L)` （代码会自动统一为 `(1, L)`）
- **Dtype**: 推荐 `float32`
- **配对**: `raw/` 和 `clean/` 下的文件名必须一一对应

### 数据集特性支持

- ✅ **变长信号**: 自动零填充到 `segment_length`（记录在 `meta['is_padded']`）
- ✅ **滑窗切片**: 通过 `stride` 参数生成确定性切片（用于 val/test）
- ✅ **随机裁剪**: train 模式支持随机裁剪（设置 `random_crop=True`）
- ✅ **Z-score 归一化**: 逐样本归一化（可选 `"zscore_per_sample"` 或 `"none"`）

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
conda create -n eeg_denoise python=3.9
conda activate eeg_denoise

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy
```

### 2. 测试模块（无需数据集）

```bash
# 测试 STFT 处理器
cd cDDPM/V5
python -m signal.stft_utils

# 测试模型前向传播
python -m models.uar_acssnet

# 测试损失函数
python -m train.losses

# 测试最小训练循环（随机数据）
python -m train.min_train
```

**预期输出**:
- STFT 频率 bin 验证（1-100 Hz）
- 模型参数量统计
- Shape 和范围断言通过
- 训练 2 epoch（每 epoch 5 batch）完成

### 3. 使用真实数据集训练

```python
import sys
sys.path.append("cDDPM/V5")

from configs import get_default_config, print_config
from datasets import build_dataloaders
from models import UAR_ACSSNet
from train import train, set_seed
import torch
from pathlib import Path

# 设置随机种子
set_seed(42)

# 加载配置
cfg = get_default_config()
cfg["dataset_root"] = "../../Dataset"  # 修改为实际路径
cfg["batch_size"] = 16
cfg["num_epochs"] = 50
print_config(cfg)

# 构建数据加载器
loaders = build_dataloaders(
    root=cfg["dataset_root"],
    batch_size=cfg["batch_size"],
    segment_length=cfg["segment_length"],
    val_stride=cfg["val_stride"],
    test_stride=cfg["test_stride"],
    normalize=cfg["normalize"],
    num_workers=cfg["num_workers"],
    pin_memory=cfg["pin_memory"],
)

# 创建模型
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
model = UAR_ACSSNet(
    segment_length=cfg["segment_length"],
    unet_base_ch=cfg["unet_base_ch"],
    unet_levels=cfg["unet_levels"],
    spec_channels=cfg["spec_channels"],
    acss_depth=cfg["acss_depth"],
    num_freq_bins=cfg["num_freq_bins"],
    dropout=cfg["dropout"],
).to(device)

# 优化器和调度器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg["num_epochs"],
    eta_min=cfg["learning_rate"] * 0.01,
)

# 训练
save_dir = Path(cfg["save_dir"])
save_dir.mkdir(exist_ok=True)

train(
    model=model,
    train_loader=loaders["train"],
    val_loader=loaders["val"],
    optimizer=optimizer,
    scheduler=scheduler,
    cfg=cfg,
    device=device,
    save_dir=save_dir,
)
```

---

## 配置说明

### 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_root` | str | `"../../Dataset"` | 数据集根目录 |
| `segment_length` | int | `2048` | 输入信号长度 |
| `normalize` | str | `"zscore_per_sample"` | 归一化方式 (`"none"` / `"zscore_per_sample"`) |
| `batch_size` | int | `16` | Batch size |
| `val_stride` | int | `1024` | 验证集滑窗步长 |

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `unet_base_ch` | int | `32` | U-Net 基础通道数 |
| `unet_levels` | int | `4` | U-Net 编码器层数 |
| `spec_channels` | int | `64` | 谱图编码器输出通道数 |
| `acss_depth` | int | `3` | ACSSBlock 堆叠层数 |
| `num_freq_bins` | int | `103` | STFT 频率 bin 数量（自动计算） |
| `dropout` | float | `0.1` | Dropout 比例 |

**参数量估算** (默认配置):
- U-Net: ~1.2M
- SpecEncoder + ACSS: ~0.5M
- FiLM + Confidence: ~0.3M
- **总计**: ~2M 参数

### 损失配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `recon_weight` | float | `1.0` | 重建损失权重 |
| `conf_reg_weight` | float | `0.1` | 置信图正则权重 |
| `tv_weight` | float | `0.01` | TV 平滑正则 |
| `entropy_weight` | float | `0.01` | 熵正则（防止退化） |
| `use_weighted_recon` | bool | `False` | 是否使用置信图加权重建损失 |

**损失函数形式**:
```
L_total = λ_recon * L_recon + λ_conf_reg * L_conf_reg

L_recon = Charbonnier(y_hat, x_clean)  # 可选加权

L_conf_reg = λ_tv * TV(w) + λ_ent * Entropy(w)
  TV(w) = mean(|w[t+1] - w[t]|)
  Entropy(w) = -mean(w*log(w) + (1-w)*log(1-w))
```

### 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_epochs` | int | `50` | 训练 epoch 数 |
| `learning_rate` | float | `1e-4` | 初始学习率 |
| `weight_decay` | float | `1e-5` | 权重衰减 |
| `grad_clip` | float | `1.0` | 梯度裁剪阈值 |
| `scheduler_type` | str | `"cosine"` | 学习率调度器类型 |

---

## 训练与验证

### 训练日志示例

```
============================================================
Starting training for 50 epochs
============================================================

[Epoch 1/50] Training...
  Epoch 1 [0/100] Loss: 0.3245 | Recon: 0.3123 | ConfReg: 0.0122
  Epoch 1 [10/100] Loss: 0.2987 | Recon: 0.2876 | ConfReg: 0.0111
  ...

[Epoch 1/50] Validating...

[Epoch 1/50] Summary:
  Time: 45.23s
  Train Loss: 0.287654 | Recon: 0.276432
  Val   Loss: 0.254321 | Recon: 0.243210
  LR: 0.000100
  ✓ Saved best model to checkpoints/best_model.pth

...
```

### 推理使用

```python
import torch
from models import UAR_ACSSNet

# 加载模型
checkpoint = torch.load("checkpoints/best_model.pth")
model = UAR_ACSSNet(**checkpoint['cfg'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
x_raw = torch.randn(1, 1, 2048)  # (B, 1, L)
with torch.no_grad():
    outputs = model(x_raw)

y_hat = outputs["y_hat"]  # 重建信号 (1, 1, 2048)
w = outputs["w"]          # 置信图 (1, 1, 2048)
```

---

## 技术细节

### STFT 参数（固定，不可更改）

| 参数 | 值 | 说明 |
|------|-----|------|
| `fs` | 500 Hz | 采样率 |
| `n_fft` | 512 | FFT 点数 |
| `hop_length` | 64 | 帧移 |
| `win_length` | 156 | 窗长 |
| `window` | Hann | 窗函数 |
| 频率分辨率 | ~0.977 Hz | `fs / n_fft` |
| 频率范围 | 1–100 Hz | 选择 bin [1, 102]（共 103 bins） |

**频率 bin 计算**:
```python
df = 500 / 512 ≈ 0.977 Hz
k_min = ceil(1.0 / df) = 2
k_max = floor(100.0 / df) = 102
num_bins = 102 - 2 + 1 = 103
```

### Shape 流程追踪

假设输入 `x_raw (4, 1, 2048)`:

```
1. STFT:
   (4, 1, 2048) → stft → (4, 257, T)  [T ≈ 35]
   → select bins [2:103] → (4, 103, 35)

2. SpecEncoder2D:
   (4, 103, 35) → (4, 64, 35, 103)  [permute to (B,C,T,F)]

3. ACSSStack (depth=3):
   (4, 64, 35, 103) → ACSSBlock × 3 → (4, 64, 35, 103)

4. FiLM:
   (4, 64, 35, 103) → pool F → (4, 64, 35)
   → interpolate to L → (4, 64, 2048)
   → generate α, β for decoder layers

5. U-Net:
   (4, 1, 2048) + FiLM → (4, 1, 2048)

6. Confidence:
   (4, 64, 35, 103) → pool & head → (4, 1, 35)
   → interpolate to L → (4, 1, 2048)
   → sigmoid → [0, 1]
```

### 内存与速度估算

**单机单卡 (RTX 3090 24GB)**:
- Batch size 16, L=2048: ~4GB
- 训练速度: ~150 samples/s
- 单 epoch (10k samples): ~70s

---

## 可复现性

### 随机种子设置

所有随机性均通过 `set_seed(42)` 固定：
```python
from train import set_seed
set_seed(42)
```

包含:
- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed`
- CUDA `torch.cuda.manual_seed_all`
- cuDNN `deterministic=True, benchmark=False`

### Dtype 与设备

- **Dtype**: 所有计算使用 `float32`
- **设备**: 自动检测 CUDA 或 CPU
- **AMP**: 未启用（可自行添加）

### 断言检查

代码中包含大量 shape 和范围断言：
```python
assert y_hat.shape == (B, 1, L)
assert (w >= 0).all() and (w <= 1).all()
assert S.shape[1] == num_freq_bins
```

运行时会自动验证，确保数据流正确。

---

## 扩展指南

### 1. 替换为真实 Mamba/SSM

当前 `ScanFreq/ScanTime` 使用简化实现（depthwise conv）。替换步骤：

```python
# 在 models/uar_acssnet.py 中

# 原实现
class ScanFreq(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scan = DepthwiseScan1D(channels, kernel_size=5)
    ...

# 替换为 Mamba
from mamba_ssm import Mamba

class ScanFreq(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4)
    
    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B*T, C, F)
        # Mamba expects (B, L, D)
        x_in = x_reshaped.permute(0, 2, 1)  # (B*T, F, C)
        out = self.mamba(x_in)  # (B*T, F, C)
        out = out.permute(0, 2, 1).reshape(B, T, C, F).permute(0, 2, 1, 3)
        return out
```

### 2. 添加数据增强

在 `datasets/eeg_pair_dataset.py` 中添加增强：

```python
def augment(self, x: np.ndarray) -> np.ndarray:
    # 时间平移
    shift = np.random.randint(-100, 100)
    x = np.roll(x, shift, axis=-1)
    
    # 幅值缩放
    scale = np.random.uniform(0.9, 1.1)
    x = x * scale
    
    return x
```

### 3. 多 GPU 训练

```python
from torch.nn.parallel import DataParallel

model = UAR_ACSSNet(...)
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model.to(device)
```

---

## 推理与部署

### 文件级推理

训练完成后，使用 `inference_file.py` 对新数据进行降噪：

#### 单文件推理

```bash
python inference_file.py \
    --checkpoint output_V5/checkpoints/best_model.pth \
    --input data/noisy_signal.npy \
    --output results/denoised_signal.npy \
    --segment_length 2048 \
    --stride 1024
```

#### 批量推理（目录）

```bash
python inference_file.py \
    --checkpoint output_V5/checkpoints/best_model.pth \
    --input data/raw_signals/ \
    --output results/denoised/ \
    --pattern "*.npy" \
    --batch_size 32
```

#### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 模型文件路径（必需）|
| `--input` | - | 输入文件或目录（必需）|
| `--output` | - | 输出文件或目录（必需）|
| `--segment_length` | 2048 | 分割片段长度 |
| `--stride` | 1024 | 滑窗步长（建议为segment_length/2）|
| `--normalize` | zscore | 归一化方法（zscore/minmax/none）|
| `--batch_size` | 32 | 批处理大小 |
| `--save_format` | npy | 保存格式（npy/npz/txt）|
| `--device` | cuda | 设备（cuda/cpu）|

**长信号处理**：自动使用滑窗分割 → 批量推理 → 重叠平均重建，保持信号完整性。

### 结果可视化

使用 `visualize_inference.py` 比较原始和降噪信号：

```bash
python visualize_inference.py \
    --raw data/test_001_raw.npy \
    --denoised results/test_001_denoised.npy \
    --clean data/test_001_clean.npy \
    --spectral \
    --save comparison.png
```

### 快速测试

运行测试脚本验证推理功能：

```bash
python test_inference.py
```

该脚本会：
1. 检查模型和数据文件
2. 执行单文件推理测试
3. 执行批量推理测试
4. 验证输出文件并显示统计信息

**详细文档**: 查看 [`INFERENCE_README.md`](INFERENCE_README.md) 获取完整推理指南。

---

## 常见问题

### Q1: STFT 频率 bin 不匹配？

**A**: 检查 `num_freq_bins` 是否与 STFT 配置一致。默认 `fs=500, n_fft=512, freq_range=[1,100]` 对应 **103 bins**。

### Q2: 内存溢出？

**A**: 减小 `batch_size` 或 `segment_length`。推荐配置：
- 8GB GPU: batch_size=8, L=2048
- 16GB GPU: batch_size=16, L=2048
- 24GB GPU: batch_size=32, L=2048

### Q3: 置信图 `w` 全为 0.5？

**A**: 可能是正则权重过大导致退化。尝试：
- 降低 `entropy_weight` (0.01 → 0.001)
- 增加 `conf_reg_weight` 的训练 epoch 延迟

### Q4: 验证损失不下降？

**A**: 检查：
1. 数据集是否正确配对（raw/clean 文件名一致）
2. 归一化是否合理（建议使用 `zscore_per_sample`）
3. 学习率是否过大（降低到 `1e-5` 试试）

### Q5: 推理时显存不足？

**A**: 减小推理批处理大小或分割长度：
```bash
--batch_size 8 --segment_length 1024
```

---

## 引用

如果本项目对你的研究有帮助，欢迎引用（示例）：

```bibtex
@misc{uar_acssnet2026,
  title={UAR-ACSSNet: Unified Artifact Removal with Axis-Conditioned Selective Scan},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourrepo/uar-acssnet}}
}
```

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请提交 Issue 或联系：`your.email@example.com`

---

**Happy Training! 🚀**
