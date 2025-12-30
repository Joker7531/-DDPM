# SpectrogramUNet for EEG Signal Denoising

## 概述

基于 2D U-Net 的 STFT 域 EEG 信号去噪系统。

## 文件结构

```
cDDPM/V2/
├── model.py          # SpectrogramUNet 和 DoubleConv 类
├── dataset.py        # 数据集加载器（支持 STFT 变换）
├── train.py          # 训练脚本
├── inference.py      # 推理和评估脚本
├── config.yaml       # 配置文件
└── README.md         # 本文件
```

## 模型架构

### SpectrogramUNet

```
输入: [Batch, 2, 103, Time]
  ├─ Channel 0: STFT 实部
  └─ Channel 1: STFT 虚部

编码器 (4层):
  ├─ Level 1: DoubleConv(2→64)   → MaxPool
  ├─ Level 2: DoubleConv(64→128)  → MaxPool
  ├─ Level 3: DoubleConv(128→256) → MaxPool
  └─ Level 4: DoubleConv(256→512) → MaxPool

瓶颈层:
  └─ DoubleConv(512→1024)

解码器 (4层):
  ├─ Level 4: TransposeConv → Concat(512) → DoubleConv(1024→512)
  ├─ Level 3: TransposeConv → Concat(256) → DoubleConv(512→256)
  ├─ Level 2: TransposeConv → Concat(128) → DoubleConv(256→128)
  └─ Level 1: TransposeConv → Concat(64)  → DoubleConv(128→64)

输出层:
  └─ Conv2d(1x1): 64 → 2 channels

输出: [Batch, 2, 103, Time]
```

### DoubleConv 模块

```
Input → Conv2d(3x3, pad=1)
     → BatchNorm2d
     → LeakyReLU(0.1)
     → Conv2d(3x3, pad=1)
     → BatchNorm2d
     → LeakyReLU(0.1)
     → Output
```

## 数据格式

### 输入数据结构

```
dataset/
├── train/
│   ├── raw/           # 原始含噪信号 (.npy)
│   │   ├── subj001_raw.npy
│   │   └── ...
│   └── clean/         # 干净信号 (.npy)
│       ├── subj001_clean.npy
│       └── ...
├── val/
│   ├── raw/
│   └── clean/
├── test/
│   ├── raw/
│   └── clean/
└── metadata.csv       # 元数据文件
```

## 使用方法

### 1. 测试模型架构

```bash
python model.py
```

输出示例:
```
======================================================================
Testing SpectrogramUNet
======================================================================

Model Architecture:
  - Total parameters: 31,037,122
  - Trainable parameters: 31,037,122

Input Shape: [B=4, C=2, F=103, T=156]
Output Shape: [4, 2, 103, 156]

✓ All assertions passed!
```

### 2. 训练模型

```bash
python train.py
```

训练过程:
- 自动加载 `dataset/` 中的数据
- 每 10 个 epoch 保存检查点
- 保存最佳模型到 `checkpoints/best.pth`
- TensorBoard 日志保存到 `logs/`

查看训练日志:
```bash
tensorboard --logdir=logs
```

### 3. 运行推理

```bash
# 在测试集上评估
python inference.py --model checkpoints/best.pth --split test

# 指定保存目录
python inference.py --model checkpoints/best.pth --split test --save-dir results/test
```

### 4. 自定义推理

```python
from inference import Inferencer
import numpy as np

# 加载模型
inferencer = Inferencer('checkpoints/best.pth', device='cuda')

# 对时域信号去噪
raw_signal = np.load('your_signal.npy')
denoised_signal = inferencer.denoise_signal(raw_signal)

# 保存结果
np.save('denoised_signal.npy', denoised_signal)
```

## 配置参数

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_channels` | 2 | 输入通道数（实部+虚部） |
| `out_channels` | 2 | 输出通道数 |
| `base_channels` | 64 | 基础通道数 |
| `depth` | 4 | U-Net 深度 |

### STFT 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fs` | 500 | 采样率 (Hz) |
| `nperseg` | 512 | STFT 窗口长度 |
| `noverlap` | 64 | 重叠点数 |
| `nfft` | 512 | FFT 长度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批次大小 |
| `num_epochs` | 200 | 训练轮数 |
| `learning_rate` | 1e-4 | 学习率 |
| `weight_decay` | 1e-5 | 权重衰减 |
| `mse_weight` | 1.0 | MSE 损失权重 |
| `l1_weight` | 0.5 | L1 损失权重 |

## 评估指标

模型使用以下指标评估性能:

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **SNR**: 信噪比 (dB)
- **Correlation**: 皮尔逊相关系数

## 核心技术细节

### 1. Pad-Crop 机制

为了处理奇数频率维度（103），模型在 forward 方法中实现了自动 padding 和 cropping:

```python
# 1. 计算 padding (使维度可被 16 整除)
divisor = 2 ** depth  # 16 for depth=4
target_height = ((103 + 15) // 16) * 16  # 112

# 2. Padding (使用反射填充避免边界伪影)
x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

# 3. 通过 U-Net
output_padded = self.model(x_padded)

# 4. Crop 回原始尺寸
output = output_padded[:, :, :103, :original_width]
```

### 2. 跳跃连接处理

解码器中处理潜在的尺寸不匹配:

```python
if x_dec.shape != skip.shape:
    diff_h = skip.shape[2] - x_dec.shape[2]
    diff_w = skip.shape[3] - x_dec.shape[3]
    x_dec = F.pad(x_dec, [diff_w//2, diff_w-diff_w//2,
                          diff_h//2, diff_h-diff_h//2])
```

### 3. 组合损失函数

```python
total_loss = mse_weight * MSE(pred, target) + l1_weight * L1(pred, target)
```

- MSE: 像素级重建精度
- L1: 促进稀疏性，减少过拟合

## 性能优化建议

1. **预计算 STFT**: 如果多次训练，可以预先计算 STFT 并保存，使用 `PrecomputedSTFTDataset`
2. **混合精度训练**: 使用 `torch.cuda.amp` 加速训练
3. **数据增强**: 添加时间平移、幅度缩放等增强
4. **正则化**: 尝试 Dropout 或更强的权重衰减

**最后更新**: 2025-12-30
