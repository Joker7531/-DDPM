"""
EEG数据增强模块
包含随机相位/极性翻转、幅度缩放、时移等增强方法
"""
import torch
import numpy as np
from typing import Tuple, Optional


class EEGAugmentation:
    """
    EEG数据增强类
    支持多种增强方法的组合
    """
    def __init__(
        self,
        flip_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        time_shift_range: int = 100,
        add_noise_prob: float = 0.0,
        noise_std: float = 0.01,
    ):
        """
        Args:
            flip_prob: 极性翻转概率
            scale_range: 幅度缩放范围 (min_scale, max_scale)
            time_shift_range: 时移最大范围（样本点数，±range）
            add_noise_prob: 添加噪声概率
            noise_std: 噪声标准差（相对于信号标准差）
        """
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.time_shift_range = time_shift_range
        self.add_noise_prob = add_noise_prob
        self.noise_std = noise_std
    
    def __call__(
        self,
        x_raw: torch.Tensor,
        x_clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对raw和clean信号应用相同的增强（保持配对关系）
        
        Args:
            x_raw: (C, L) 原始信号
            x_clean: (C, L) 干净信号
        
        Returns:
            aug_raw, aug_clean: 增强后的信号
        """
        # 1. 随机极性翻转（相位反转）
        if torch.rand(1).item() < self.flip_prob:
            x_raw = -x_raw
            x_clean = -x_clean
        
        # 2. 随机幅度缩放
        if self.scale_range[0] < self.scale_range[1]:
            scale = torch.FloatTensor(1).uniform_(
                self.scale_range[0],
                self.scale_range[1]
            ).item()
            x_raw = x_raw * scale
            x_clean = x_clean * scale
        
        # 3. 随机时移
        if self.time_shift_range > 0:
            shift = torch.randint(
                -self.time_shift_range,
                self.time_shift_range + 1,
                (1,)
            ).item()
            
            if shift != 0:
                x_raw = self._time_shift(x_raw, shift)
                x_clean = self._time_shift(x_clean, shift)
        
        # 4. 可选：添加随机噪声（仅对raw）
        if torch.rand(1).item() < self.add_noise_prob:
            signal_std = x_raw.std()
            noise = torch.randn_like(x_raw) * (signal_std * self.noise_std)
            x_raw = x_raw + noise
        
        return x_raw, x_clean
    
    def _time_shift(self, x: torch.Tensor, shift: int) -> torch.Tensor:
        """
        时域平移（循环移位）
        
        Args:
            x: (C, L) 输入信号
            shift: 移位量（正数右移，负数左移）
        
        Returns:
            shifted: (C, L) 移位后的信号
        """
        return torch.roll(x, shifts=shift, dims=-1)


class RandomPolarity:
    """随机极性翻转"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, x_raw: torch.Tensor, x_clean: torch.Tensor):
        if torch.rand(1).item() < self.prob:
            return -x_raw, -x_clean
        return x_raw, x_clean


class RandomScale:
    """随机幅度缩放"""
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, x_raw: torch.Tensor, x_clean: torch.Tensor):
        scale = torch.FloatTensor(1).uniform_(
            self.scale_range[0],
            self.scale_range[1]
        ).item()
        return x_raw * scale, x_clean * scale


class RandomTimeShift:
    """随机时移（循环移位）"""
    def __init__(self, max_shift: int = 100):
        self.max_shift = max_shift
    
    def __call__(self, x_raw: torch.Tensor, x_clean: torch.Tensor):
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return x_raw, x_clean
        
        x_raw_shifted = torch.roll(x_raw, shifts=shift, dims=-1)
        x_clean_shifted = torch.roll(x_clean, shifts=shift, dims=-1)
        return x_raw_shifted, x_clean_shifted


class Compose:
    """组合多个增强方法"""
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x_raw: torch.Tensor, x_clean: torch.Tensor):
        for t in self.transforms:
            x_raw, x_clean = t(x_raw, x_clean)
        return x_raw, x_clean


def get_train_augmentation(cfg: dict) -> Optional[EEGAugmentation]:
    """
    根据配置构建训练数据增强
    
    Args:
        cfg: 配置字典
    
    Returns:
        augmentation: 数据增强对象或None
    """
    if not cfg.get("use_augmentation", False):
        return None
    
    return EEGAugmentation(
        flip_prob=cfg.get("aug_flip_prob", 0.5),
        scale_range=cfg.get("aug_scale_range", (0.8, 1.2)),
        time_shift_range=cfg.get("aug_time_shift", 100),
        add_noise_prob=cfg.get("aug_noise_prob", 0.0),
        noise_std=cfg.get("aug_noise_std", 0.01),
    )


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    x_raw = torch.randn(1, 2048)
    x_clean = torch.randn(1, 2048)
    
    print("Original:")
    print(f"  x_raw: mean={x_raw.mean():.3f}, std={x_raw.std():.3f}")
    print(f"  x_clean: mean={x_clean.mean():.3f}, std={x_clean.std():.3f}")
    
    # 测试增强
    aug = EEGAugmentation(
        flip_prob=0.5,
        scale_range=(0.8, 1.2),
        time_shift_range=100,
    )
    
    aug_raw, aug_clean = aug(x_raw, x_clean)
    
    print("\nAugmented:")
    print(f"  aug_raw: mean={aug_raw.mean():.3f}, std={aug_raw.std():.3f}")
    print(f"  aug_clean: mean={aug_clean.mean():.3f}, std={aug_clean.std():.3f}")
    
    # 测试组合
    print("\nTesting Compose:")
    composed_aug = Compose([
        RandomPolarity(prob=0.5),
        RandomScale(scale_range=(0.9, 1.1)),
        RandomTimeShift(max_shift=50),
    ])
    
    aug_raw2, aug_clean2 = composed_aug(x_raw, x_clean)
    print(f"  aug_raw2: mean={aug_raw2.mean():.3f}, std={aug_raw2.std():.3f}")
