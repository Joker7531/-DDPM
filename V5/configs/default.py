"""
默认配置
"""


def get_default_config():
    """
    返回默认配置字典
    """
    cfg = {
        # ==================
        # 数据配置
        # ==================
        "dataset_root": "../../Dataset",  # 数据集根目录（需根据实际情况修改）
        "segment_length": 2048,           # 输入信号长度
        "normalize": "zscore_per_sample", # 归一化方式
        "batch_size": 16,                 # batch size
        "num_workers": 4,                 # DataLoader workers
        "pin_memory": True,               # 是否 pin memory
        
        # 数据增强（可选）
        "train_stride": 512,             # train 滑窗步长（None 表示随机裁剪）
        "val_stride": 1024,               # val 滑窗步长（建议 segment_length // 2）
        "test_stride": 1024,              # test 滑窗步长
        
        # ==================
        # 模型配置
        # ==================
        # U-Net
        "unet_base_ch": 32,               # U-Net 基础通道数
        "unet_levels": 4,                 # U-Net 编码器层数
        
        # 时频分支
        "spec_channels": 64,              # 谱图编码器输出通道数
        "acss_depth": 3,                  # ACSSBlock 堆叠层数
        "num_freq_bins": 101,             # STFT 频率 bin 数量（1-100Hz @ fs=500, n_fft=512）
        
        # 其他
        "dropout": 0.1,                   # dropout 比例
        
        # ==================
        # 损失配置
        # ==================
        "charbonnier_eps": 1e-6,          # Charbonnier loss epsilon
        "use_weighted_recon": False,      # 是否使用置信图加权重建损失
        
        # 置信图正则
        "tv_weight": 0.01,                # TV 平滑正则权重
        "entropy_weight": 0.01,           # 熵正则权重
        
        # 损失权重
        "recon_weight": 1.0,              # 重建损失权重
        "conf_reg_weight": 1.0,           # 置信图正则权重（增大到1.0以防止w退化）
        "consistency_weight": 0.0,        # 一致性损失权重（默认不使用）
        "consistency_type": "l1",         # 一致性损失类型
        
        # ==================
        # 训练配置
        # ==================
        "num_epochs": 50,                 # 训练 epoch 数
        "learning_rate": 1e-4,            # 初始学习率
        "weight_decay": 1e-4,             # 权重衰减（增大到1e-4以增强正则化）
        "grad_clip": 1.0,                 # 梯度裁剪（0 表示不裁剪）
        "early_stop_patience": 20,        # Early stopping 耐心值（验证损失不降低的最大 epoch 数）
        
        # 学习率调度器
        "use_scheduler": True,            # 是否使用学习率调度器
        "scheduler_type": "cosine",       # 调度器类型: "cosine", "step", "plateau"
        
        # 日志与保存
        "log_interval": 10,               # 打印日志间隔（batches）
        "save_dir": "checkpoints",        # 模型保存目录
        
        # 快速测试（可选）
        "max_train_batches": None,        # 最大训练 batch 数（None 表示全部）
        "max_val_batches": None,          # 最大验证 batch 数
        
        # ==================
        # 其他
        # ==================
        "seed": 42,                       # 随机种子
        "device": "cuda",                 # 设备: "cuda" 或 "cpu"
    }
    
    return cfg


def print_config(cfg: dict):
    """打印配置"""
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    
    for key, value in cfg.items():
        print(f"  {key:30s}: {value}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    cfg = get_default_config()
    print_config(cfg)
