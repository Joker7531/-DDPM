"""
最小训练入口
包含 train_one_epoch, validate 和主训练循环
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .losses import compute_losses


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        cfg: 配置字典
        device: 设备
        epoch: 当前 epoch
        max_batches: 最大 batch 数（用于快速测试）
    
    Returns:
        metrics: {"loss": ..., "recon": ..., ...}
    """
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_freq = 0.0
    total_conf_reg = 0.0
    total_conf_reg_weighted = 0.0
    total_tv = 0.0
    total_boundary = 0.0
    total_consistency = 0.0
    conf_weight_value = 0.0
    
    # 用于详细统计
    w_means = []
    w_stds = []
    
    num_batches = 0
    
    # 创建进度条
    total_iters = min(len(train_loader), max_batches) if max_batches else len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=total_iters, desc=f"Epoch {epoch}", ncols=120)
    
    # 是否打印详细信息（每10个epoch或第一个epoch）
    verbose = (epoch == 1 or epoch % 10 == 0)
    
    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # 解析 batch
        if len(batch) == 2:
            x_raw, x_clean = batch
        else:
            x_raw, x_clean, meta = batch
        
        x_raw = x_raw.to(device)
        x_clean = x_clean.to(device)
        
        # 前向传播
        outputs = model(x_raw)
        
        # 更新配置中的当前epoch（用于warm-up调度）
        cfg["_current_epoch"] = epoch
        
        # 计算损失
        losses = compute_losses(
            batch=(x_raw, x_clean),
            outputs=outputs,
            cfg=cfg,
            use_consistency=False,
        )
        
        loss = losses["total"]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选）
        if cfg.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        
        # 梯度诊断（verbose 模式且第一个 batch）
        if verbose and batch_idx == 0:
            print("\n  === Gradient Diagnostics ===")
            # 检查 confidence_head 参数
            for name, param in model.named_parameters():
                if 'confidence_head' in name:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"    {name}: requires_grad={param.requires_grad}, grad_norm={grad_norm:.6f}")
                    else:
                        print(f"    {name}: requires_grad={param.requires_grad}, grad=None")
            
            # 检查 w 的统计
            if 'w' in outputs:
                w_data = outputs['w']
                print(f"    w: mean={w_data.mean().item():.4f}, std={w_data.std().item():.4f}, "
                      f"min={w_data.min().item():.4f}, max={w_data.max().item():.4f}")
        
        optimizer.step()
        
        # 累积统计
        total_loss += loss.item()
        total_recon += losses["recon"].item()
        total_freq += losses.get("freq", torch.tensor(0.0)).item()
        total_conf_reg += losses["conf_reg"].item()
        total_conf_reg_weighted += losses["conf_reg_weighted"].item()
        total_tv += losses["tv"].item()
        total_boundary += losses["boundary_penalty"].item()
        total_consistency += losses["consistency"].item()
        conf_weight_value = losses["conf_weight"]
        num_batches += 1
        
        # 收集w统计
        if "w" in outputs:
            w_means.append(outputs["w"].mean().item())
            w_stds.append(outputs["w"].std().item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{losses["recon"].item():.4f}',
            'tv': f'{losses["tv"].item():.4f}',
            'bnd': f'{losses["boundary_penalty"].item():.4f}'
        })
    
    # 平均指标
    metrics = {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "freq": total_freq / num_batches,
        "conf_reg": total_conf_reg / num_batches,
        "conf_reg_weighted": total_conf_reg_weighted / num_batches,
        "tv": total_tv / num_batches,
        "boundary_penalty": total_boundary / num_batches,
        "consistency": total_consistency / num_batches,
        "conf_weight": conf_weight_value,
        "w_mean": np.mean(w_means) if w_means else 0.0,
        "w_std": np.mean(w_stds) if w_stds else 0.0,
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    验证
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        cfg: 配置字典
        device: 设备
        max_batches: 最大 batch 数（用于快速测试）
    
    Returns:
        metrics: {"loss": ..., "recon": ..., ...}
    """
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_freq = 0.0
    total_conf_reg = 0.0
    total_conf_reg_weighted = 0.0
    total_tv = 0.0
    total_boundary = 0.0
    conf_weight_value = 0.0
    
    w_means = []
    w_stds = []
    
    num_batches = 0
    
    # 创建进度条
    total_iters = min(len(val_loader), max_batches) if max_batches else len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=total_iters, desc="Validation", ncols=120, leave=False)
    
    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # 解析 batch
        if len(batch) == 2:
            x_raw, x_clean = batch
        else:
            x_raw, x_clean, meta = batch
        
        x_raw = x_raw.to(device)
        x_clean = x_clean.to(device)
        
        # 前向传播
        outputs = model(x_raw)
        
        # 计算损失
        losses = compute_losses(
            batch=(x_raw, x_clean),
            outputs=outputs,
            cfg=cfg,
            use_consistency=False,
        )
        
        # 累积统计
        total_loss += losses["total"].item()
        total_recon += losses["recon"].item()
        total_freq += losses.get("freq", torch.tensor(0.0)).item()
        total_conf_reg += losses["conf_reg"].item()
        total_conf_reg_weighted += losses["conf_reg_weighted"].item()
        total_tv += losses["tv"].item()
        total_boundary += losses["boundary_penalty"].item()
        conf_weight_value = losses["conf_weight"]
        num_batches += 1
        
        # 收集w统计
        if "w" in outputs:
            w_means.append(outputs["w"].mean().item())
            w_stds.append(outputs["w"].std().item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses["total"].item():.4f}',
            'recon': f'{losses["recon"].item():.4f}'
        })
    
    # 平均指标
    metrics = {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "freq": total_freq / num_batches,
        "conf_reg": total_conf_reg / num_batches,
        "conf_reg_weighted": total_conf_reg_weighted / num_batches,
        "tv": total_tv / num_batches,
        "boundary_penalty": total_boundary / num_batches,
        "conf_weight": conf_weight_value,
        "w_mean": np.mean(w_means) if w_means else 0.0,
        "w_std": np.mean(w_stds) if w_stds else 0.0,
    }
    
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: dict,
    device: torch.device,
    save_dir: Optional[Path] = None,
):
    """
    完整训练循环
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        cfg: 配置字典
        device: 设备
        save_dir: 模型保存目录（可选）
    """
    num_epochs = cfg.get("num_epochs", 10)
    best_val_loss = float("inf")
    patience = cfg.get("early_stop_patience", 20)  # Early stopping 耐心值
    no_improve_count = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"Early stopping patience: {patience} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, cfg, device, epoch,
            max_batches=cfg.get("max_train_batches", None),
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, cfg, device,
            max_batches=cfg.get("max_val_batches", None),
        )
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # 打印统计
        epoch_time = time.time() - epoch_start
        
        lr_str = f" | LR: {scheduler.get_last_lr()[0]:.2e}" if scheduler is not None else ""
        
        # 每10个epoch或第一个epoch打印详细信息
        if epoch == 1 or epoch % 10 == 0:
            warmup_status = f" [Conf Warmup: {epoch}/{cfg.get('conf_warmup_epochs', 0)}, Weight={train_metrics['conf_weight']:.2f}]" if epoch <= cfg.get('conf_warmup_epochs', 0) else ""
            print(f"\n{'='*120}")
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s){warmup_status}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Recon={train_metrics['recon']:.4f}, Freq={train_metrics['freq']:.4f}, "
                  f"ConfReg={train_metrics['conf_reg_weighted']:.6f} (raw={train_metrics['conf_reg']:.4f}, TV={train_metrics['tv']:.6f}, Boundary={train_metrics['boundary_penalty']:.4f})")
            print(f"         w_mean={train_metrics['w_mean']:.3f}, w_std={train_metrics['w_std']:.4f}")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, Recon={val_metrics['recon']:.4f}, Freq={val_metrics['freq']:.4f}, "
                  f"ConfReg={val_metrics['conf_reg_weighted']:.6f} (raw={val_metrics['conf_reg']:.4f}, TV={val_metrics['tv']:.6f}, Boundary={val_metrics['boundary_penalty']:.4f}){lr_str}")
            print(f"         w_mean={val_metrics['w_mean']:.3f}, w_std={val_metrics['w_std']:.4f}")
            print(f"{'='*120}")
        else:
            # 简洁显示
            print(f"\n{'='*120}")
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Train: {train_metrics['loss']:.4f} ({train_metrics['recon']:.4f}) | "
                  f"Val: {val_metrics['loss']:.4f} ({val_metrics['recon']:.4f}){lr_str}")
            print(f"{'='*120}")
        
        # 保存最佳模型和 Early Stopping
        if save_dir is not None:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                no_improve_count = 0
                save_path = save_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'cfg': cfg,
                }, save_path)
                print(f"  ✓ Saved best model to {save_path} (val_loss improved)")
            else:
                no_improve_count += 1
                print(f"  No improvement for {no_improve_count}/{patience} epochs")
                
                # Early stopping
                if no_improve_count >= patience:
                    print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                    print(f"  Best validation loss: {best_val_loss:.6f}")
                    break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")


def main_minimal_example():
    """
    最小可运行示例（使用随机数据）
    """
    print("\n" + "="*60)
    print("Minimal Training Example (Random Data)")
    print("="*60 + "\n")
    
    # 设置随机种子
    set_seed(42)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 配置
    from ..configs.default import get_default_config
    cfg = get_default_config()
    
    # 模型
    from ..models import UAR_ACSSNet
    model = UAR_ACSSNet(
        segment_length=cfg["segment_length"],
        unet_base_ch=cfg["unet_base_ch"],
        unet_levels=cfg["unet_levels"],
        spec_channels=cfg["spec_channels"],
        acss_depth=cfg["acss_depth"],
        num_freq_bins=cfg["num_freq_bins"],
        dropout=cfg["dropout"],
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["num_epochs"],
        eta_min=cfg["learning_rate"] * 0.01,
    )
    
    # 创建随机数据加载器（模拟真实数据集）
    from torch.utils.data import TensorDataset
    
    def make_dummy_loader(num_samples, batch_size):
        x_raw = torch.randn(num_samples, 1, cfg["segment_length"])
        x_clean = torch.randn(num_samples, 1, cfg["segment_length"])
        ds = TensorDataset(x_raw, x_clean)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader
    
    train_loader = make_dummy_loader(num_samples=64, batch_size=8)
    val_loader = make_dummy_loader(num_samples=32, batch_size=8)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # 训练
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # 快速测试：只训练几个 epoch，每个 epoch 几个 batch
    cfg["num_epochs"] = 2
    cfg["max_train_batches"] = 5
    cfg["max_val_batches"] = 3
    cfg["log_interval"] = 1
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        save_dir=save_dir,
    )
    
    print("\n✓ Minimal training example completed!\n")


if __name__ == "__main__":
    train()
