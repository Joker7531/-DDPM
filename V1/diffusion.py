"""
高斯扩散过程 (Gaussian Diffusion)
实现DDPM的前向加噪和反向去噪逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm

from model import ConditionalDiffWave
from loss import MultiResolutionSTFTLoss


class GaussianDiffusion(nn.Module):
    """
    高斯扩散过程包装类
    
    Args:
        model: 噪声预测网络
        timesteps: 扩散步数 T
        beta_start: beta schedule起始值
        beta_end: beta schedule结束值
        loss_type: 损失类型 ('l1', 'l2', 'stft', 'hybrid')
    """
    
    def __init__(
        self,
        model: ConditionalDiffWave,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = 'hybrid'
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        # 创建STFT损失（用于hybrid模式）
        if loss_type in ['stft', 'hybrid']:
            self.stft_loss = MultiResolutionSTFTLoss()
        
        # 定义beta schedule (线性调度)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # 计算alpha相关的系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为buffer（不参与梯度更新，但会随模型保存）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 前向扩散过程的系数: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 后验分布的系数（用于采样）
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        从系数张量中提取特定时间步的值，并reshape以便广播
        
        Args:
            a: 系数张量 [T]
            t: 时间步索引 [B]
            x_shape: 目标形状 [B, C, L]
            
        Returns:
            提取的系数 [B, 1, 1]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散过程: 从x_0采样x_t
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        
        Args:
            x_start: 干净信号 x_0 [B, 1, 2048]
            t: 时间步 [B]
            noise: 可选的噪声（用于确定性采样）[B, 1, 2048]
            
        Returns:
            x_t: 带噪信号 [B, 1, 2048]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从预测的噪声反推x_0
        
        x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        
        Args:
            x_t: 当前带噪信号 [B, 1, 2048]
            t: 时间步 [B]
            noise: 预测的噪声 [B, 1, 2048]
            
        Returns:
            x_0_pred: 预测的干净信号 [B, 1, 2048]
        """
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t
        
        return x_0_pred
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        训练步骤：计算损失
        
        Args:
            x_start: 干净的EEG信号 [B, 1, 2048]
            condition: 条件信号（raw EEG）[B, 1, 2048]
            t: 随机采样的时间步 [B]
            noise: 可选的噪声 [B, 1, 2048]
            
        Returns:
            loss: 总损失
            loss_dict: 损失详情字典
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 前向加噪: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # 模型预测噪声
        predicted_noise = self.model(x_noisy, t, condition)
        
        loss_dict = {}
        
        # 计算噪声预测损失
        if self.loss_type == 'l1':
            noise_loss = F.l1_loss(predicted_noise, noise)
            total_loss = noise_loss
            loss_dict['noise_l1'] = noise_loss.item()
            
        elif self.loss_type == 'l2':
            noise_loss = F.mse_loss(predicted_noise, noise)
            total_loss = noise_loss
            loss_dict['noise_l2'] = noise_loss.item()
            
        elif self.loss_type == 'stft':
            # 从预测的噪声反推x_0
            x_0_pred = self.predict_x0_from_noise(x_noisy, t, predicted_noise)
            stft_loss, stft_dict = self.stft_loss(x_0_pred, x_start)
            total_loss = stft_loss
            loss_dict.update(stft_dict)
            loss_dict['stft_total'] = stft_loss.item()
            
        elif self.loss_type == 'hybrid':
            # 混合损失: L1 + STFT
            noise_loss = F.l1_loss(predicted_noise, noise)
            
            # 从预测的噪声反推x_0
            x_0_pred = self.predict_x0_from_noise(x_noisy, t, predicted_noise)
            stft_loss, stft_dict = self.stft_loss(x_0_pred, x_start)
            
            # 加权组合
            total_loss = noise_loss + 0.1 * stft_loss  # STFT loss权重可调
            
            loss_dict['noise_l1'] = noise_loss.item()
            loss_dict['stft_total'] = stft_loss.item()
            loss_dict.update(stft_dict)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        t_index: int,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        单步去噪采样: p(x_{t-1} | x_t)
        
        Args:
            x: 当前状态 x_t [B, 1, 2048]
            t: 时间步标量
            t_index: 时间步索引（用于提取系数）
            condition: 条件信号 [B, 1, 2048]
            
        Returns:
            x_{t-1}: 去噪后的信号 [B, 1, 2048]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 创建时间步张量
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = self.model(x, t_tensor, condition)
        
        # 提取系数
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t_tensor, x.shape)
        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
        )
        
        # 计算均值: mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_cumprod_t) * epsilon_theta)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            # 最后一步不添加噪声
            return model_mean
        else:
            # 添加后验方差的噪声
            posterior_variance_t = self._extract(self.posterior_variance, t_tensor, x.shape)
            noise = torch.randn_like(x)
            
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        完整的采样过程: 从纯高斯噪声逐步去噪
        
        Args:
            condition: 条件信号（raw EEG）[B, 1, 2048]
            return_intermediates: 是否返回中间步骤
            
        Returns:
            denoised: 去噪后的干净信号 [B, 1, 2048]
            或 (denoised, intermediates) 如果 return_intermediates=True
        """
        device = condition.device
        batch_size = condition.shape[0]
        shape = condition.shape
        
        # 从纯高斯噪声开始
        x = torch.randn(shape, device=device)
        
        intermediates = []
        
        # 逐步去噪: T -> 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps, leave=False):
            x = self.p_sample(x, i, i, condition)
            
            if return_intermediates and i % 100 == 0:
                intermediates.append(x.cpu())
        
        if return_intermediates:
            return x, intermediates
        else:
            return x


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ConditionalDiffWave(
        in_channels=2,
        out_channels=1,
        residual_channels=128,  # 减小以加快测试
        num_layers=10,  # 减少层数以加快测试
        dilation_cycle=10
    ).to(device)
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=100,  # 减少步数以加快测试
        loss_type='hybrid'
    ).to(device)
    
    print(f"Device: {device}")
    print(f"Timesteps: {diffusion.timesteps}")
    
    # 测试训练损失
    batch_size = 2
    x_clean = torch.randn(batch_size, 1, 2048).to(device)
    x_raw = torch.randn(batch_size, 1, 2048).to(device)
    t = torch.randint(0, diffusion.timesteps, (batch_size,)).to(device)
    
    loss, loss_dict = diffusion.p_losses(x_clean, x_raw, t)
    
    print("\n=== Training Test ===")
    print(f"Loss: {loss.item():.6f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试采样
    print("\n=== Sampling Test ===")
    with torch.no_grad():
        denoised = diffusion.sample(x_raw[:1])  # 只采样1个以节省时间
    
    print(f"Input condition shape: {x_raw[:1].shape}")
    print(f"Denoised output shape: {denoised.shape}")
    print("Sampling completed successfully!")
