## V4 文件夹结构

```
V4/
├── dataset.py    # 数据加载与预处理
├── model.py      # SpectrogramNAFNet 架构
├── losses.py     # 复合损失函数
└── train.py      # 训练主循环
```

## 各模块功能概述

### 1. dataset.py
- **`STFTSlicingDataset`**: 滑动窗口切片数据集
  - 支持 `.npy` 文件输入 `[2, 257, T]`
  - 频域裁剪至索引 `1:104` (共103个频点)
  - 抗恒等映射归一化（Log变换 + Z-score，保持相位比例）
  - 残差目标计算: `Noise = Raw - Clean`
- **`get_dataloaders()`**: 获取训练/验证/测试数据加载器

### 2. model.py
- **`NAFBlock`**: 无激活函数的残差块
  - LayerNorm2d → Conv1x1升维 → DepthwiseConv3x3 → SimpleGate → SCA → Conv1x1降维 → Dropout → Residual
- **`SpectrogramNAFNet`**: U-Net结构主干网络
  - 4层编码器 + 4个瓶颈NAFBlock + 4层解码器
  - 基础通道数: 32
  - 自动填充奇数维度（F=103 → 112）并裁剪回原尺寸

### 3. losses.py
- **`CompositeLoss`**: 双域约束复合损失
  - $L_{noise}$: 实虚部 L1 损失
  - $L_{reconstruct}$: Log-Magnitude L1 损失
  - 总损失: $L = w_1 \cdot L_{noise} + w_2 \cdot L_{reconstruct}$
- **`PSNRMetric`**: PSNR 指标计算器

### 4. train.py
- **`Trainer`**: 完整训练流程
  - AdamW 优化器 (lr=1e-3, weight_decay=1e-2)
  - CosineAnnealingWarmRestarts 调度器
  - 混合精度训练 (AMP)
  - 检查点管理与日志记录

## 使用示例

```bash
python train.py \
    --raw_dir /path/to/raw \
    --clean_dir /path/to/clean \
    --output_dir outputs \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```
归一化：
1. magnitude = |S|
2. phase = S / |S|  (单位相位向量，保持不变)
3. log_mag = log1p(magnitude)
4. norm_log_mag = (log_mag - mean) / std  (Z-score)
5. S_norm = norm_log_mag * phase

反归一化：
1. norm_log_mag = |S_norm| * sign  (恢复符号)
2. log_mag = norm_log_mag * std + mean
3. magnitude = expm1(log_mag)
4. S = magnitude * phase