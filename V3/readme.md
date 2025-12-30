V3 核心改进
组件	V2	V3 (残差版本)
目标	直接预测 Clean	预测 Noise = Raw - Clean
推理	Clean = Model(Raw)	Clean = Raw - Model(Raw)
预处理	无	Log + InstanceNorm
Loss	L1 + LogMag	NoiseLoss + CleanLoss + LogMagLoss
base_channels	64	32
文件结构


关键技术点
残差学习: 网络学习噪声模式比学习干净信号更容易
Log压缩: log(1+|x|)*sign(x) 处理STFT大动态范围
InstanceNorm: 每个样本独立归一化，适合变化的EEG特征
三项Loss: 同时监督噪声预测和信号重建