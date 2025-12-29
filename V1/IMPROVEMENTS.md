# 模型改进说明

## 📋 改进概览

本次更新针对**低频漂移**和**突变噪声**问题进行了全面优化。

---

## 🎯 改进 1: 低频漂移和积分误差抑制

### **问题**
- 扩散模型在长时间迭代过程中容易产生积分误差累积
- 导致重建信号出现明显的低频漂移和DC偏移

### **解决方案**

#### 1.1 显式均值损失 (Mean Loss)
**位置**: `loss.py` - `MeanLoss`

```python
class MeanLoss(nn.Module):
    """惩罚重建信号的均值偏离零点"""
    def forward(self, y_pred):
        mean_values = torch.mean(y_pred, dim=-1)
        loss = torch.mean(torch.abs(mean_values))
        return self.weight * loss
```

**作用**:
- 直接惩罚输出信号的非零均值
- 强制模型学习零均值信号
- 权重: `0.5`

#### 1.2 数据输入端归零
**位置**: `dataset.py` - `_instance_normalize()`

```python
# 归一化后显式强制均值为零
normalized_raw = normalized_raw - np.mean(normalized_raw)
normalized_clean = normalized_clean - np.mean(normalized_clean)
```

**作用**:
- 从数据源头确保输入信号零均值
- 避免训练时引入DC偏移
- 配合推理时的baseline correction形成闭环

---

## 🎯 改进 2: 突变噪声抑制

### **问题**
- EEG信号中存在肌电等突变噪声
- 纯L1/L2损失对高频突变不敏感
- 可能产生过度平滑或保留伪影

### **解决方案**

#### 2.1 总变分损失 (Total Variation Loss)
**位置**: `loss.py` - `TotalVariationLoss`

```python
class TotalVariationLoss(nn.Module):
    """平滑信号，抑制突变噪声"""
    def forward(self, y):
        diff = y[:, :, 1:] - y[:, :, :-1]  # 一阶差分
        tv_loss = torch.mean(torch.abs(diff))
        return self.weight * tv_loss
```

**作用**:
- 惩罚信号的剧烈变化
- 鼓励平滑过渡
- 权重: `0.01` (温和约束)

#### 2.2 梯度差异损失 (Gradient Difference Loss)
**位置**: `loss.py` - `GradientDifferenceLoss`

```python
class GradientDifferenceLoss(nn.Module):
    """确保预测信号的梯度接近真值"""
    def forward(self, y_pred, y_true):
        grad_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        grad_true = y_true[:, :, 1:] - y_true[:, :, :-1]
        grad_loss = torch.mean(torch.abs(grad_pred - grad_true))
        return self.weight * grad_loss
```

**作用**:
- 保持信号的时间动态特性
- 避免过度平滑导致细节丢失
- 权重: `0.05`

#### 2.3 条件Dropout
**位置**: `model.py` - `ConditionalDiffWave`

```python
self.condition_dropout_layer = nn.Dropout(0.1)

# 训练时应用
if self.training and self.condition_dropout > 0:
    condition = self.condition_dropout_layer(condition)
```

**作用**:
- 在训练时随机丢弃10%的条件信息
- 迫使模型学习更鲁棒的去噪策略
- 提高对突变噪声的泛化能力
- 仅在训练时生效，推理时完全使用条件

---

## 📊 损失函数组合

### **最终损失函数** (`diffusion.py`)

```python
total_loss = (
    noise_loss +           # L1噪声预测损失
    0.1 * stft_loss +      # 多分辨率频域损失
    mean_loss_val +        # 均值损失 (0.5)
    tv_loss_val +          # 总变分损失 (0.01)
    grad_loss_val          # 梯度损失 (0.05)
)
```

### **权重设计原理**

| 损失项 | 权重 | 作用 |
|--------|------|------|
| Noise L1 | 1.0 | 主要优化目标 |
| STFT | 0.1 | 频域约束 |
| Mean | 0.5 | 强抑制DC漂移 |
| TV | 0.01 | 温和平滑 |
| Gradient | 0.05 | 保持动态 |

---

## 🚀 使用方法

### **训练**
```bash
# 新模型将自动使用所有改进
python train_amp.py \
    --data_dir Dataset \
    --output_dir output_improved \
    --batch_size 24 \
    --epochs 300
```

### **推理**
```bash
# 推理时的基线校正已内置
python inference.py \
    --model output_improved/checkpoints/model_best.pt \
    --input data.npy \
    --sampling_timesteps 100
```

---

## 📈 预期效果

### **低频漂移改善**
- ✅ DC偏移: **消除**
- ✅ 0.5Hz以下分量: **显著减少**
- ✅ 长信号稳定性: **大幅提升**

### **突变噪声改善**
- ✅ 肌电伪影: **有效抑制**
- ✅ 信号平滑度: **适度提升**
- ✅ 细节保留: **良好平衡**

### **模型鲁棒性**
- ✅ 条件依赖: **降低过拟合风险**
- ✅ 泛化能力: **提升**
- ✅ 边缘情况: **更稳定**

---

## 🔍 监控指标

训练日志中新增的损失项：
```json
{
  "noise_l1": 0.1234,
  "stft_total": 0.0567,
  "mean_loss": 0.0089,   // 新增：应逐渐趋近0
  "tv_loss": 0.0012,     // 新增：适度值即可
  "grad_loss": 0.0234,   // 新增：与clean梯度接近
  "total": 0.2136
}
```

---

## ⚠️ 注意事项

1. **兼容性**: 需要重新训练模型，旧checkpoint不兼容
2. **计算开销**: 新增损失项约增加5-10%训练时间
3. **超参数**: 可根据具体数据调整各损失权重
4. **验证**: 建议对比改进前后的频谱分析

---

## 📝 版本信息

- **版本**: V1.1
- **更新日期**: 2025-12-29
- **向后兼容**: 否（需重新训练）
