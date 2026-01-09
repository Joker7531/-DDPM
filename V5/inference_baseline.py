"""
Baseline模型单样本推理脚本
从验证集加载样本，使用训练好的模型进行推理并可视化结果
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import UAR_ACSSNet
from datasets import build_dataloaders
from configs import get_default_config


def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    Returns:
        model: 加载好的模型
        cfg: 配置字典
    """
    print(f"\n📦 Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint获取配置（如果有）或使用默认配置
    if 'cfg' in ckpt:
        cfg = ckpt['cfg']
        print("✓ Using config from checkpoint")
    else:
        cfg = get_default_config()
        print("✓ Using default config")
    
    # 创建模型
    model = UAR_ACSSNet(
        segment_length=cfg.get("segment_length", 2048),
        unet_base_ch=cfg.get("unet_base_ch", 32),
        unet_levels=cfg.get("unet_levels", 4),
        spec_channels=cfg.get("spec_channels", 64),
        acss_depth=cfg.get("acss_depth", 3),
        num_freq_bins=cfg.get("num_freq_bins", 101),
        dropout=cfg.get("dropout", 0.0),
        baseline_mode=cfg.get("baseline_mode", True),
    ).to(device)
    
    # 加载权重
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
    print(f"✓ Best val loss: {ckpt.get('val_loss', 'unknown'):.6f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    return model, cfg


def inference_and_visualize(
    model,
    val_loader,
    device='cuda',
    num_samples=4,
    save_path="inference_results.png"
):
    """
    推理并可视化结果
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        num_samples: 可视化样本数量
        save_path: 保存路径
    """
    print(f"\n🔍 Running inference on {num_samples} samples...")
    
    # 获取一个batch
    batch = next(iter(val_loader))
    
    # 解析batch（根据数据集返回格式）
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x_raw, x_clean = batch
        else:
            x_raw, x_clean, _ = batch
    else:
        x_raw = batch['raw']
        x_clean = batch['clean']
    
    x_raw = x_raw.to(device)
    x_clean = x_clean.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(x_raw)
        y_hat = outputs['y_hat']
        w = outputs.get('w', None)
    
    # 计算MSE
    mse = torch.mean((y_hat - x_clean) ** 2, dim=-1).cpu().numpy()
    
    print(f"✓ Inference completed")
    print(f"  - Input shape: {x_raw.shape}")
    print(f"  - Output shape: {y_hat.shape}")
    print(f"  - MSE range: [{mse.min():.6f}, {mse.max():.6f}]")
    print(f"  - MSE mean: {mse.mean():.6f}")
    
    # 可视化
    num_samples = min(num_samples, x_raw.shape[0])
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 转换为numpy
        raw_np = x_raw[i, 0].cpu().numpy()
        clean_np = x_clean[i, 0].cpu().numpy()
        pred_np = y_hat[i, 0].cpu().numpy()
        
        # 绘制
        time_axis = np.arange(len(raw_np))
        ax.plot(time_axis, raw_np, 'k', alpha=0.4, linewidth=1, label='Raw (Noisy)')
        ax.plot(time_axis, clean_np, 'g', linewidth=1.5, label='Ground Truth (Clean)')
        ax.plot(time_axis, pred_np, 'r--', linewidth=1.5, label='Prediction (Denoised)')
        
        # 标题和标签
        sample_mse = mse[i, 0]
        ax.set_title(f"Sample {i+1} - MSE: {sample_mse:.6f}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (samples)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        info_text = (
            f"Raw: μ={raw_np.mean():.3f}, σ={raw_np.std():.3f}\n"
            f"Clean: μ={clean_np.mean():.3f}, σ={clean_np.std():.3f}\n"
            f"Pred: μ={pred_np.mean():.3f}, σ={pred_np.std():.3f}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # 如果有confidence map，额外保存
    if w is not None and not model.baseline_mode:
        save_confidence_map(w, x_raw, num_samples, save_path.replace('.png', '_confidence.png'))


def save_confidence_map(w, x_raw, num_samples, save_path):
    """
    保存confidence map可视化
    
    Args:
        w: confidence map (B, 1, L)
        x_raw: 输入信号
        num_samples: 样本数量
        save_path: 保存路径
    """
    num_samples = min(num_samples, w.shape[0])
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 2 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        w_np = w[i, 0].cpu().numpy()
        raw_np = x_raw[i, 0].cpu().numpy()
        time_axis = np.arange(len(w_np))
        
        # 双y轴：信号 + confidence
        ax2 = ax.twinx()
        ax.plot(time_axis, raw_np, 'k', alpha=0.3, linewidth=0.5, label='Raw Signal')
        ax2.plot(time_axis, w_np, 'b-', linewidth=1.5, label='Confidence Map')
        ax2.fill_between(time_axis, 0, w_np, alpha=0.3, color='blue')
        
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude', color='k')
        ax2.set_ylabel('Confidence w(t)', color='b')
        ax2.set_ylim([0, 1])
        
        ax.set_title(f"Sample {i+1} - Confidence Map (mean={w_np.mean():.3f}, std={w_np.std():.3f})")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confidence map saved to: {save_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference with trained baseline model')
    parser.add_argument('--checkpoint', type=str, default='output_V5/checkpoints/best_model.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--dataset_root', type=str, default='../../Dataset',
                        help='Dataset root directory')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='inference_baseline_vis.png',
                        help='Output visualization file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Please train the model first or specify correct checkpoint path")
        return
    
    model, cfg = load_model(checkpoint_path, device)
    
    # 更新数据集路径
    cfg['dataset_root'] = args.dataset_root
    cfg['batch_size'] = max(4, args.num_samples)  # 至少加载足够的样本
    
    # 构建数据加载器
    print(f"\n📊 Loading validation dataset from: {cfg['dataset_root']}")
    _, val_loader, _ = build_dataloaders(
        dataset_root=cfg['dataset_root'],
        batch_size=cfg['batch_size'],
        segment_length=cfg['segment_length'],
        train_stride=cfg.get('train_stride'),
        val_stride=cfg.get('val_stride', 1024),
        test_stride=cfg.get('test_stride', 1024),
        normalize=cfg['normalize'],
        num_workers=0,  # 单线程避免潜在问题
        pin_memory=False,
        return_meta=False,
    )
    
    print(f"✓ Validation loader ready (batch_size={cfg['batch_size']})")
    
    # 推理并可视化
    inference_and_visualize(
        model=model,
        val_loader=val_loader,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output
    )
    
    print(f"\n🎉 Inference completed successfully!")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()
