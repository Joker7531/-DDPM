"""
损失函数模块
包含重建损失、门控/置信正则、一致性损失、频域损失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ================================
# 1) 重建损失
# ================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (平滑 L1)
    L = sqrt(x^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """
        Args:
            pred: (B, 1, L)
            target: (B, 1, L)
            weight: (B, 1, L)  可选的样本权重（置信图）
        
        Returns:
            loss: scalar
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        
        if weight is not None:
            # 加权损失（w 越小越不强迫拟合）
            loss = loss * weight
        
        return loss.mean()


class FrequencyDomainLoss(nn.Module):
    """
    频域损失：在STFT域计算重建误差
    使用与模型相同的STFT参数
    """
    def __init__(
        self,
        fs: int = 500,
        n_fft: int = 512,
        hop_length: int = 64,
        win_length: int = 156,
        loss_type: str = "l1",  # "l1", "l2", "charbonnier"
        eps: float = 1e-6,
    ):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.loss_type = loss_type
        self.eps = eps
        
        # 使用Hann窗
        self.register_buffer('window', torch.hann_window(win_length))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, L) 预测时域信号
            target: (B, 1, L) 目标时域信号
        
        Returns:
            loss: scalar
        """
        B, C, L = pred.shape
        assert C == 1, "Expected single channel"
        
        # Squeeze channel维度
        pred_1d = pred.squeeze(1)  # (B, L)
        target_1d = target.squeeze(1)  # (B, L)
        
        # 确保window在正确的设备上
        window = self.window.to(pred.device)
        
        # 计算STFT (返回复数)
        pred_stft = torch.stft(
            pred_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )  # (B, freq_bins, time_frames)
        
        target_stft = torch.stft(
            target_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_stft)  # (B, F, T)
        target_mag = torch.abs(target_stft)
        
        # 计算损失
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_mag, target_mag)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_mag, target_mag)
        elif self.loss_type == "charbonnier":
            diff = pred_mag - target_mag
            loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber Loss
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """
        Args:
            pred: (B, 1, L)
            target: (B, 1, L)
            weight: (B, 1, L)  可选的样本权重
        
        Returns:
            loss: scalar
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        if weight is not None:
            loss = loss * weight
        
        return loss.mean()


# ================================
# 2) 置信图正则
# ================================

class ConfidenceRegularization(nn.Module):
    """
    置信图 w 的正则化
    包含:
        - TV (Total Variation) 平滑正则
        - 边界惩罚（惩罚 w 太接近 0 或 1，鼓励在中间范围）
    """
    def __init__(self, tv_weight: float = 0.01, boundary_weight: float = 0.1):
        super().__init__()
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight  # 边界惩罚权重
    
    def tv_loss(self, w: torch.Tensor) -> torch.Tensor:
        """
        Total Variation Loss（一阶差分平滑）
        Args:
            w: (B, 1, L)
        """
        diff = w[:, :, 1:] - w[:, :, :-1]
        return torch.abs(diff).mean()
    
    def boundary_penalty(self, w: torch.Tensor) -> torch.Tensor:
        """
        边界惩罚：惩罚 w 太接近 0 或 1
        使用负对数似然鼓励 w 在 (0, 1) 内部
        Args:
            w: (B, 1, L)  范围应该在 [0, 1]
        Returns:
            loss: -log(w) - log(1-w) 的均值（在边界处趋于无穷）
        """
        eps = 1e-6
        w_clamp = torch.clamp(w, eps, 1 - eps)
        # 同时惩罚接近0和接近1：-log(w) - log(1-w)
        # 这在 w=0.5 时最小，在 w→0 或 w→1 时趋于无穷
        boundary_loss = -torch.log(w_clamp) - torch.log(1 - w_clamp)
        return boundary_loss.mean()
    
    def forward(self, w: torch.Tensor) -> tuple:
        """
        Args:
            w: (B, 1, L)  置信图
        
        Returns:
            total_reg: scalar
            tv: scalar (单独返回tv值)
            boundary_pen: scalar (单独返回边界惩罚值)
        """
        tv = self.tv_loss(w)
        boundary_pen = self.boundary_penalty(w)
        
        total_reg = self.tv_weight * tv + self.boundary_weight * boundary_pen
        return total_reg, tv, boundary_pen


# ================================
# 3) 一致性损失（接口预留）
# ================================

class ConsistencyLoss(nn.Module):
    """
    一致性损失：同一样本不同视图的输出应保持一致
    
    使用场景:
        - 数据增强：对同一样本施加不同噪声/扰动
        - 多视图训练：期望模型对同一信号的不同版本给出一致的重建
    """
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y1: (B, 1, L)  视图1的重建
            y2: (B, 1, L)  视图2的重建
        
        Returns:
            loss: scalar
        """
        if self.loss_type == "l1":
            return F.l1_loss(y1, y2)
        elif self.loss_type == "l2":
            return F.mse_loss(y1, y2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


# ================================
# 4) 总损失计算函数
# ================================

def compute_losses(
    batch: tuple,
    outputs: Dict[str, torch.Tensor],
    cfg: dict,
    use_consistency: bool = False,
    outputs_aug: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    计算总损失
    
    Args:
        batch: (x_raw, x_clean) 或 (x_raw, x_clean, meta)
        outputs: 模型输出，包含 {"y_hat", "w", ...}
        cfg: 配置字典，包含损失权重等
        use_consistency: 是否使用一致性损失
        outputs_aug: 可选的增强样本输出（用于一致性损失）
    
    Returns:
        losses: {
            "total": total_loss,
            "recon": recon_loss,
            "conf_reg": conf_reg_loss,
            "consistency": consistency_loss,  # 如果使用
        }
    """
    # 解析 batch
    if len(batch) == 2:
        x_raw, x_clean = batch
    else:
        x_raw, x_clean, meta = batch
    
    # 提取输出
    y_hat = outputs["y_hat"]
    w = outputs["w"]
    
    # 1) 时域重建损失
    recon_criterion = CharbonnierLoss(eps=cfg.get("charbonnier_eps", 1e-6))
    
    # Warm-up期间禁用weighted_recon，避免w接近0导致loss消失
    current_epoch = cfg.get("_current_epoch", 0)
    warmup_epochs = cfg.get("conf_warmup_epochs", 0)
    use_weighted = cfg.get("use_weighted_recon", False) and (current_epoch >= warmup_epochs)
    
    if use_weighted:
        # 使用置信图加权
        # w 的语义：接近 0.5 表示"不确定"，接近边界（0或1）表示"确定"
        # 这里我们直接用 w 作为权重：w 大 → 更重视该位置的重建
        # 边界惩罚会自然地让 w 在 (0,1) 内分布，模型学习哪些位置需要更多关注
        recon_loss = recon_criterion(y_hat, x_clean, weight=w)
    else:
        recon_loss = recon_criterion(y_hat, x_clean, weight=None)
    
    # 1.5) 频域重建损失（可选）
    freq_loss = torch.tensor(0.0, device=y_hat.device)
    if cfg.get("use_freq_loss", False):
        freq_criterion = FrequencyDomainLoss(
            fs=cfg.get("stft_fs", 500),
            n_fft=cfg.get("stft_n_fft", 512),
            hop_length=cfg.get("stft_hop", 64),
            win_length=cfg.get("stft_win", 156),
            loss_type=cfg.get("freq_loss_type", "l1"),
            eps=cfg.get("charbonnier_eps", 1e-6),
        )
        freq_loss = freq_criterion(y_hat, x_clean)
    
    # 2) 置信图正则
    conf_reg_criterion = ConfidenceRegularization(
        tv_weight=cfg.get("tv_weight", 0.01),
        boundary_weight=cfg.get("boundary_weight", 0.1),
    )
    conf_reg_loss, tv_loss, boundary_penalty = conf_reg_criterion(w)
    
    # 3) 一致性损失（可选）
    consistency_loss = torch.tensor(0.0, device=y_hat.device)
    if use_consistency and outputs_aug is not None:
        consistency_criterion = ConsistencyLoss(loss_type=cfg.get("consistency_type", "l1"))
        y_hat_aug = outputs_aug["y_hat"]
        consistency_loss = consistency_criterion(y_hat, y_hat_aug)
    
    # Warm-up调度：前N个epoch置信图正则权重为0
    warmup_epochs = cfg.get("conf_warmup_epochs", 0)
    conf_weight = 0.0 if current_epoch < warmup_epochs else cfg.get("conf_reg_weight", 0.1)
    
    # 总损失
    total_loss = (
        cfg.get("recon_weight", 1.0) * recon_loss +
        cfg.get("freq_loss_weight", 0.1) * freq_loss +
        conf_weight * conf_reg_loss +
        cfg.get("consistency_weight", 0.0) * consistency_loss
    )
    
    losses = {
        "total": total_loss,
        "recon": recon_loss,
        "freq": freq_loss,
        "conf_reg": conf_reg_loss,
        "conf_reg_weighted": conf_weight * conf_reg_loss,  # 实际贡献
        "tv": tv_loss,
        "boundary_penalty": boundary_penalty,
        "consistency": consistency_loss,
        "conf_weight": conf_weight,  # 当前权重
    }
    
    return losses


# ================================
# 5) 测试
# ================================

def test_losses():
    """测试损失函数"""
    print("\n=== Testing Loss Functions ===\n")
    
    B, C, L = 4, 1, 2048
    
    # 模拟数据
    x_raw = torch.randn(B, C, L)
    x_clean = torch.randn(B, C, L)
    y_hat = torch.randn(B, C, L)
    w = torch.sigmoid(torch.randn(B, C, L))  # [0, 1]
    
    batch = (x_raw, x_clean)
    outputs = {"y_hat": y_hat, "w": w}
    
    # 配置
    cfg = {
        "charbonnier_eps": 1e-6,
        "use_weighted_recon": False,
        "tv_weight": 0.01,
        "entropy_weight": 0.01,
        "recon_weight": 1.0,
        "conf_reg_weight": 0.1,
        "consistency_weight": 0.0,
    }
    
    # 计算损失
    losses = compute_losses(batch, outputs, cfg, use_consistency=False)
    
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
    
    # 测试加权重建
    cfg["use_weighted_recon"] = True
    losses_weighted = compute_losses(batch, outputs, cfg, use_consistency=False)
    print("\nWeighted Recon Loss:")
    print(f"  recon: {losses_weighted['recon'].item():.6f}")
    
    # 测试一致性损失
    outputs_aug = {"y_hat": y_hat + 0.1 * torch.randn_like(y_hat), "w": w}
    cfg["consistency_weight"] = 0.5
    losses_cons = compute_losses(batch, outputs, cfg, use_consistency=True, outputs_aug=outputs_aug)
    print("\nWith Consistency Loss:")
    for k, v in losses_cons.items():
        print(f"  {k}: {v.item():.6f}")
    
    print("\n✓ All tests passed!\n")


if __name__ == "__main__":
    test_losses()
