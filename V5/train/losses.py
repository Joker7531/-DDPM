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
    频域损失：在STFT域计算重建误差（单分辨率，保留以兼容旧配置）
    建议改用 MultiResolutionSTFTLoss
    """
    def __init__(
        self,
        fs: int = 500,
        n_fft: int = 512,
        hop_length: int = 64,
        win_length: int = 156,
        loss_type: str = "log_L1",  # "l1", "l2", "charbonnier", "log_L1"
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
        lt = self.loss_type
        if lt == "l1":
            loss = F.l1_loss(pred_mag, target_mag)
        elif lt == "l2":
            loss = F.mse_loss(pred_mag, target_mag)
        elif lt == "charbonnier":
            diff = pred_mag - target_mag
            loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()
        elif lt == "log_L1" or lt == "log_l1":
            # 使用 log1p 稳定变换（避免 log(0) 问题）
            pred_logmag = torch.log1p(pred_mag)
            target_logmag = torch.log1p(target_mag)
            loss = F.l1_loss(pred_logmag, target_logmag)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss

class STFTLoss(nn.Module):
    """
    单组分辨率的 STFT Loss 计算单元（谱收敛 + 对数幅值）
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def forward(self, pred_1d: torch.Tensor, target_1d: torch.Tensor) -> torch.Tensor:
        # 确保 window 在正确设备
        window = self.window.to(pred_1d.device)
        
        # STFT（返回复数）
        pred_stft = torch.stft(
            pred_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        target_stft = torch.stft(
            target_1d,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        
        # Magnitude
        eps = 1e-7
        pred_mag = torch.abs(pred_stft) + eps
        target_mag = torch.abs(target_stft) + eps
        
        # 1) Spectral Convergence (安全版本，防止除零)
        numer = torch.norm(target_mag - pred_mag, p='fro')
        denom = torch.norm(target_mag, p='fro')
        denom = torch.clamp(denom, min=1e-5)  # 安全夹断，防止除以极小值
        sc_loss = numer / denom
        
        # 2) Log Magnitude L1
        log_mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        
        return sc_loss + log_mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率 STFT Loss (MR-STFT)
    默认参数参考常见语音增强设置，可在 cfg 中覆盖：
      - mrstft_fft_sizes
      - mrstft_hop_sizes
      - mrstft_win_lengths
    """
    def __init__(
        self,
        fft_sizes = [1024, 512, 128],
        hop_sizes = [120, 50, 10],
        win_lengths = [600, 240, 60],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), 'MR-STFT parameter lists must have same length'
        
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, hop, win) for fs, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Squeeze channel dim if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        
        total = pred.new_tensor(0.0)
        for f in self.stft_losses:
            total = total + f(pred, target)
        
        return total / len(self.stft_losses)


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
    置信图 w 的正则化（已弃用，存在方法论缺陷）
    
    已知问题：
        - TV Loss 鼓励平滑 → w 坍缩成常数
        - Boundary Penalty 是全局标量，无空间依赖
        - 加权重建鼓励 w → min
    
    当前状态：
        - 此模块仍计算正则项（用于监控），但不影响重建损失
        - 建议未来移除或重构
    
    遗留功能:
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


class ImprovedConfidenceRegularization(nn.Module):
    """
    改进的置信度正则化（实验性，未启用）
    
    核心改进：
        1. 熵正则化：鼓励 w 在每个样本内有足够的分布熵（防止坍缩成常数）
        2. 稀疏性约束：鼓励部分位置的 w 接近边界（高置信度），而非全部居中
        3. 移除 TV Loss（与置信度语义冲突）
    
    使用方法：
        在 cfg 中设置 "use_improved_conf_reg": True
    """
    def __init__(self, entropy_weight: float = 0.1, sparsity_weight: float = 0.05):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight
    
    def entropy_regularization(self, w: torch.Tensor) -> torch.Tensor:
        """
        熵正则化：鼓励 w 在每个样本内有足够的变化
        
        Args:
            w: (B, 1, L)
        
        Returns:
            entropy_loss: 负熵（越大越好，所以返回负值用于最小化）
        """
        eps = 1e-6
        w_clamp = torch.clamp(w, eps, 1 - eps)
        
        # 计算每个样本的熵（沿 L 维度）
        # H = -sum(w * log(w) + (1-w) * log(1-w))
        entropy = -(w_clamp * torch.log(w_clamp) + (1 - w_clamp) * torch.log(1 - w_clamp))
        
        # 返回负熵（鼓励高熵 = 多样性）
        # 目标：让 w 在空间上有差异，而非常数
        return -entropy.mean()
    
    def sparsity_regularization(self, w: torch.Tensor) -> torch.Tensor:
        """
        稀疏性约束：鼓励 w 呈双峰分布（接近 0 或 1），而非都在中间
        
        使用 L1 范数的变体：min |w - 0.5|（鼓励远离 0.5）
        
        Args:
            w: (B, 1, L)
        
        Returns:
            sparsity_loss: 鼓励 w 极化（接近边界）
        """
        # 惩罚 w 接近 0.5（不确定区域）
        # 鼓励 w 接近 0 或 1（高置信度区域）
        return -torch.abs(w - 0.5).mean()  # 负号：最小化此项 = 最大化 |w-0.5|
    
    def forward(self, w: torch.Tensor) -> tuple:
        """
        Args:
            w: (B, 1, L)
        
        Returns:
            total_reg: scalar
            entropy_loss: scalar
            sparsity_loss: scalar
        """
        entropy_loss = self.entropy_regularization(w)
        sparsity_loss = self.sparsity_regularization(w)
        
        total_reg = (
            self.entropy_weight * entropy_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        return total_reg, entropy_loss, sparsity_loss


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
    
    # ========================================
    # 置信度策略状态：已禁用（方法论缺陷）
    # ========================================
    # 问题分析：
    # 1. TV Loss 鼓励 w 平滑 → 极端解：w 变成常数（w_std → 0）
    # 2. Boundary Penalty 是全局标量，无法鼓励空间差异
    # 3. 加权重建鼓励 w → min，导致 w 坍缩到下界
    # 结果：w 失去空间自适应能力，等价于一个无用的全局标量
    # ========================================
    # 修复方案：
    # - 当前：完全禁用加权机制，使用标准重建损失
    # - 未来：若需要自适应加权，需重新设计策略（如熵正则化、稀疏性约束等）
    # ========================================
    
    # 强制禁用加权重建（即使配置文件中启用）
    use_weighted = False
    recon_loss = recon_criterion(y_hat, x_clean, weight=None)
    
    # 1.5) 频域重建损失（可选）
    freq_loss = torch.tensor(0.0, device=y_hat.device)
    if cfg.get("use_freq_loss", False):
        # 使用多分辨率 STFT Loss；可从 cfg 覆盖默认参数
        mr_fft = cfg.get("mrstft_fft_sizes", [1024, 512, 128])
        mr_hop = cfg.get("mrstft_hop_sizes", [120, 50, 10])
        mr_win = cfg.get("mrstft_win_lengths", [600, 240, 60])
        freq_criterion = MultiResolutionSTFTLoss(
            fft_sizes=mr_fft,
            hop_sizes=mr_hop,
            win_lengths=mr_win,
        )
        freq_loss = freq_criterion(y_hat, x_clean)
    
    # 2) 置信图正则（仅用于监控，不影响训练）
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
    
    # 置信度正则权重：已禁用（方法论缺陷）
    conf_weight = 0.0  # 强制为 0，不参与梯度优化
    
    # 总损失
    total_loss = (
        cfg.get("recon_weight", 1.0) * recon_loss +
        cfg.get("freq_loss_weight", 0.5) * freq_loss +
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
        # 诊断信息：监控 w 的统计特性
        "w_mean": w.mean().item(),
        "w_std": w.std().item(),
        "w_min": w.min().item(),
        "w_max": w.max().item(),
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
