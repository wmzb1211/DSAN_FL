# privacy.py
import torch
import numpy as np
from collections import defaultdict
import math


class AutoClipper:
    """NIPS 2023自动裁剪核心模块"""

    def __init__(self, gamma=0.01, R_init=1.0, momentum=0.9):
        self.gamma = gamma  # 稳定化因子（公式(6)）
        self.R = R_init  # 动态裁剪阈值（初始值）
        self.momentum = momentum  # 阈值动量更新系数
        self.ema_norm = None  # 梯度范数指数移动平均

    def clip(self, grad):
        """执行自动裁剪（公式(5)）"""
        grad_norm = torch.norm(grad).item()

        # 首次运行初始化EMA
        if self.ema_norm is None:
            self.ema_norm = grad_norm
        else:
            self.ema_norm = self.momentum * self.ema_norm + (1 - self.momentum) * grad_norm

        # 动态调整R（算法1第5步）
        self.R = (1 - self.momentum) * self.R + self.momentum * grad_norm

        # 自动裁剪公式
        denominator = np.sqrt(grad_norm ** 2 + (self.gamma * self.R) ** 2)
        scaling = self.R / denominator
        return grad * scaling

    def get_current_R(self):
        """获取当前动态阈值（用于隐私会计）"""
        return self.R


class DPOptimizer:
    """支持自动裁剪的差分隐私优化器"""

    def __init__(self, optimizer, noise_multiplier, clipper, lr_scheduler=None):
        """
        Args:
            optimizer: 基础优化器（Adam/SGD）
            noise_multiplier (float): 噪声系数σ
            clipper (AutoClipper): 自动裁剪器实例
        """
        self.optimizer = optimizer
        self.sigma = noise_multiplier
        self.clipper = clipper
        self.lr_scheduler = lr_scheduler

    def step(self):
        # 梯度裁剪与加噪
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 应用自动裁剪
                clipped_grad = self.clipper.clip(p.grad.data)

                # 添加高斯噪声（公式(7)）
                noise = torch.randn_like(clipped_grad) * self.sigma * self.clipper.R
                p.grad.data = clipped_grad + noise

        # 参数更新
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

class DPOptimizerWithoutClipper:
    def __init__(self, optimizer, noise_multiplier, max_grad_norm, lr_scheduler=None):
        """
        Args:
            optimizer: 基础优化器（Adam/SGD）
            noise_multiplier (float): 噪声系数σ
        """
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
    def step(self):
        # 梯度裁剪
        total_norm = 0.0
        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)

        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        # 添加噪声
        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

class PrivacyAccountant:
    """支持自动裁剪的隐私会计（RDP+MA）"""

    def __init__(self, delta=1e-5, max_epsilon=8.0, use_rdp=True):
        self.delta = float(delta)
        self.max_epsilon = max_epsilon
        self.use_rdp = use_rdp

        # RDP参数
        self.rdp_alphas = [1 + x / 10.0 for x in range(1, 100)]
        self.rdp_eps = defaultdict(float)

        # MA参数
        self.ma_lambdas = list(range(1, 65))
        self.ma_log_mgf = np.zeros(len(self.ma_lambdas))

        # 动态裁剪追踪
        self.current_R = 1.0  # 初始值，会被AutoClipper更新

    def update(self, batch_size, dataset_size, steps, noise_multiplier, current_R):
        """
        Args:
            current_R (float): 当前自动裁剪阈值（来自AutoClipper）
        """
        self.current_R = current_R
        q = batch_size / dataset_size

        if self.use_rdp:
            self._update_rdp(q, steps, noise_multiplier)
        else:
            self._update_ma(q, steps, noise_multiplier)

    def _update_rdp(self, q, steps, sigma):
        """RDP会计（考虑动态R）"""
        for alpha in self.rdp_alphas:
            sensitivity = self.current_R  # 动态敏感度
            rdp_per_step = (alpha * (sensitivity ** 2) * q ** 2) / (2 * sigma ** 2)
            self.rdp_eps[alpha] += rdp_per_step * steps

    def _update_ma(self, q, steps, sigma):
        """矩会计（考虑动态R）"""
        for _ in range(steps):
            for i, lambd in enumerate(self.ma_lambdas):
                term = (q ** 2 * lambd * (lambd + 1) * self.current_R ** 2) / (2 * sigma ** 2)
                self.ma_log_mgf[i] += term

    def _rdp_to_dp(self):
        min_epsilon = float('inf')
        for alpha in self.rdp_alphas:
            if alpha <= 1:
                continue
            epsilon = self.rdp_eps[alpha] + (math.log(1 / self.delta) / (alpha - 1))
            min_epsilon = min(min_epsilon, epsilon)
        return min_epsilon

    def _ma_to_dp(self):
        min_epsilon = float('inf')
        for i, lambd in enumerate(self.ma_lambdas):
            if lambd <= 0:
                continue
            epsilon = (self.ma_log_mgf[i] - math.log(self.delta)) / lambd
            min_epsilon = min(min_epsilon, epsilon)
        return min_epsilon

    def get_epsilon(self):
        if self.use_rdp:
            eps = self._rdp_to_dp()
        else:
            eps = self._ma_to_dp()
        return min(eps, self.max_epsilon), self.delta

    def is_budget_exhausted(self):
        current_eps, _ = self.get_epsilon()
        return current_eps >= self.max_epsilon

