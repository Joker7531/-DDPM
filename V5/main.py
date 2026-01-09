"""
完整训练脚本 - 使用真实数据集
"""
import sys
import argparse
from pathlib import Path
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from configs import get_default_config, print_config
from datasets import build_dataloaders
from models import UAR_ACSSNet
from train import train, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train UAR-ACSSNet")
    
    # 数据配置
    # 默认路径：相对于脚本文件向上两级到 3_ICA，再到 Dataset
    default_dataset = str(Path(__file__).parent.parent / "Dataset")
    parser.add_argument("--dataset_root", type=str, default=default_dataset,
                        help="数据集根目录")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--segment_length", type=int, default=2048,
                        help="输入信号长度")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # 模型配置
    parser.add_argument("--unet_base_ch", type=int, default=32,
                        help="U-Net 基础通道数")
    parser.add_argument("--unet_levels", type=int, default=4,
                        help="U-Net 编码器层数")
    parser.add_argument("--spec_channels", type=int, default=64,
                        help="谱图编码器输出通道数")
    parser.add_argument("--acss_depth", type=int, default=3,
                        help="ACSSBlock 堆叠层数")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比例")
    
    # 训练配置
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="训练 epoch 数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    
    # 其他
    parser.add_argument("--save_dir", type=str, default="output_V5/checkpoints",
                        help="模型保存目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    
    # 快速测试
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式（少量 epoch 和 batch）")
    
    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()
    
    # 加载默认配置
    cfg = get_default_config()
    
    # 更新配置
    for key, value in vars(args).items():
        if key in cfg:
            cfg[key] = value
    
    # 快速测试模式
    if args.quick_test:
        print("\n⚡ Quick Test Mode Enabled")
        cfg["num_epochs"] = 2
        cfg["max_train_batches"] = 5
        cfg["max_val_batches"] = 3
        cfg["log_interval"] = 1
    
    # 打印配置
    print_config(cfg)
    
    # 设置随机种子
    set_seed(cfg["seed"])
    
    # 设备
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 构建数据加载器
    print("Loading dataset...")
    loaders = build_dataloaders(
        root=cfg["dataset_root"],
        batch_size=cfg["batch_size"],
        segment_length=cfg["segment_length"],
        train_stride=cfg.get("train_stride"),
        val_stride=cfg.get("val_stride", 1024),
        test_stride=cfg.get("test_stride", 1024),
        normalize=cfg["normalize"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        return_meta=False,
    )
    
    # 创建模型
    print("\nBuilding model...")
    model = UAR_ACSSNet(
        segment_length=cfg["segment_length"],
        unet_base_ch=cfg["unet_base_ch"],
        unet_levels=cfg["unet_levels"],
        spec_channels=cfg["spec_channels"],
        acss_depth=cfg["acss_depth"],
        num_freq_bins=cfg["num_freq_bins"],
        dropout=cfg["dropout"],
        baseline_mode=cfg.get("baseline_mode", False),
    ).to(device)
    
    if cfg.get("baseline_mode", False):
        print("🔹 Baseline Mode: Using pure U-Net (no FiLM/ACSS)")
    else:
        print("🔹 Full Mode: UAR-ACSSNet with FiLM modulation")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    
    # 学习率调度器
    scheduler = None
    if cfg.get("use_scheduler", True):
        if cfg.get("scheduler_type", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["num_epochs"],
                eta_min=cfg["learning_rate"] * 0.01,
            )
        elif cfg["scheduler_type"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg["num_epochs"] // 3,
                gamma=0.1,
            )
    
    # 创建保存目录
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 训练
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
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {save_dir / 'best_model.pth'}\n")


if __name__ == "__main__":
    main()
