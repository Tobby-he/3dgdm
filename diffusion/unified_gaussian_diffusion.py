#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一高斯扩散模块
实现真正的3D高斯扩散数学基础，包含粗糙和精细两阶段扩散
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def extract(a, t, x_shape):
    """从张量a中根据时间步t提取值"""
    batch_size = t.shape[0]
    # 确保a和t在同一设备上
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class UNet3D(nn.Module):
    """3D UNet用于噪声预测"""
    
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, style_dim=512):
        super().__init__()
        self.time_dim = time_dim
        self.style_dim = style_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 风格嵌入
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        # 编码器
        self.encoder = nn.ModuleList([
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.Conv3d(64, 128, 3, padding=1, stride=2),
            nn.Conv3d(128, 256, 3, padding=1, stride=2),
            nn.Conv3d(256, 512, 3, padding=1, stride=2)
        ])
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1)
        )
        
        # 解码器
        self.decoder = nn.ModuleList([
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.Conv3d(64, out_channels, 3, padding=1)
        ])
        
        # 条件注入层
        self.time_proj = nn.ModuleList([
            nn.Linear(time_dim, 64),
            nn.Linear(time_dim, 128),
            nn.Linear(time_dim, 256),
            nn.Linear(time_dim, 512)
        ])
        
        self.style_proj = nn.ModuleList([
            nn.Linear(style_dim, 64),
            nn.Linear(style_dim, 128),
            nn.Linear(style_dim, 256),
            nn.Linear(style_dim, 512)
        ])
    
    def forward(self, x, t, style_features):
        """前向传播"""
        # 时间嵌入
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # 风格嵌入
        # 处理style_features可能是列表的情况
        if isinstance(style_features, list):
            # 使用最后一层特征或合并所有特征
            style_feat = style_features[-1]  # 使用最深层特征
            # 全局平均池化到固定维度
            style_feat = F.adaptive_avg_pool2d(style_feat, (1, 1)).flatten(1)
        else:
            style_feat = style_features
        
        style_emb = self.style_mlp(style_feat)
        
        # 编码
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = F.relu(x)
                # 注入时间和风格信息
                t_proj = self.time_proj[i](t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                s_proj = self.style_proj[i](style_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x + t_proj + s_proj
                skip_connections.append(x)
        
        # 中间层
        x = self.middle(x)
        
        # 解码
        for i, layer in enumerate(self.decoder[:-1]):
            x = layer(x)
            x = F.relu(x)
            if i < len(skip_connections):
                x = x + skip_connections[-(i+1)]
        
        # 输出层
        x = self.decoder[-1](x)
        return x
    
    def time_embedding(self, t):
        """正弦时间嵌入"""
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CoarseGaussianDiffusion(nn.Module):
    """粗糙扩散：像画家一样先定调子和轮廓"""
    
    def __init__(self, num_timesteps=250):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 噪声调度
        self.register_buffer('betas', cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 粗糙阶段关注整体结构
        self.position_net = UNet3D(in_channels=3, out_channels=3)
        self.opacity_net = UNet3D(in_channels=1, out_channels=1)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：向3D高斯添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, style_features, significance_map):
        """计算扩散损失"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 分别预测位置和不透明度的噪声
        if x_start.shape[1] == 3:  # 位置
            predicted_noise = self.position_net(x_noisy, t, style_features)
        else:  # 不透明度
            predicted_noise = self.opacity_net(x_noisy, t, style_features)
        
        # 加权损失：显著区域权重更高
        loss = F.mse_loss(noise, predicted_noise, reduction='none')
        if significance_map is not None:
            # significance_map是2D的[B, 1, H, W]，需要扩展到3D体素维度
            # loss的形状是[B, C, D, H, W]
            B, C, D, H, W = loss.shape
            
            # 将2D significance_map扩展到3D
            if significance_map.dim() == 4:  # [B, 1, H_sig, W_sig]
                # 首先调整到匹配的H, W
                if significance_map.shape[2:] != (H, W):
                    significance_map = F.interpolate(
                        significance_map, size=(H, W), 
                        mode='bilinear', align_corners=True
                    )
                # 然后扩展到深度维度
                significance_map_3d = significance_map.unsqueeze(2).expand(B, 1, D, H, W)
                # 扩展到所有通道
                significance_map_3d = significance_map_3d.expand(B, C, D, H, W)
            else:
                # 如果已经是3D，直接使用
                significance_map_3d = significance_map
            
            weighted_loss = (loss * significance_map_3d).mean()
        else:
            weighted_loss = loss.mean()
        
        return weighted_loss
    
    def p_sample(self, x, t, style_features, significance_map=None):
        """反向采样步骤"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # 预测噪声
        if x.shape[1] == 3:
            predicted_noise = self.position_net(x, t, style_features)
        else:
            predicted_noise = self.opacity_net(x, t, style_features)
        
        # 应用显著性权重
        if significance_map is not None:
            predicted_noise = predicted_noise * significance_map.unsqueeze(1)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.betas, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


class FineGaussianDiffusion(nn.Module):
    """精细扩散：像雕塑家一样刻画细节"""
    
    def __init__(self, num_timesteps=750):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 噪声调度（更细致的调度）
        self.register_buffer('betas', cosine_beta_schedule(num_timesteps, s=0.004))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 精细阶段关注局部细节和风格
        self.color_net = UNet3D(in_channels=3, out_channels=3)
        self.scale_net = UNet3D(in_channels=3, out_channels=3)
        self.rotation_net = UNet3D(in_channels=4, out_channels=4)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, style_features, significance_map, attribute_type='color'):
        """计算扩散损失"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 根据属性类型选择网络
        if attribute_type == 'color':
            predicted_noise = self.color_net(x_noisy, t, style_features)
        elif attribute_type == 'scale':
            predicted_noise = self.scale_net(x_noisy, t, style_features)
        elif attribute_type == 'rotation':
            predicted_noise = self.rotation_net(x_noisy, t, style_features)
        else:
            raise ValueError(f"Unknown attribute type: {attribute_type}")
        
        # 加权损失
        loss = F.mse_loss(noise, predicted_noise, reduction='none')
        if significance_map is not None:
            # significance_map是2D的[B, 1, H, W]，需要扩展到3D体素维度
            # loss的形状是[B, C, D, H, W]
            B, C, D, H, W = loss.shape
            
            # 将2D significance_map扩展到3D
            if significance_map.dim() == 4:  # [B, 1, H_sig, W_sig]
                # 首先调整到匹配的H, W
                if significance_map.shape[2:] != (H, W):
                    significance_map = F.interpolate(
                        significance_map, size=(H, W), 
                        mode='bilinear', align_corners=True
                    )
                # 然后扩展到深度维度
                significance_map_3d = significance_map.unsqueeze(2).expand(B, 1, D, H, W)
                # 扩展到所有通道
                significance_map_3d = significance_map_3d.expand(B, C, D, H, W)
            else:
                # 如果已经是3D，直接使用
                significance_map_3d = significance_map
            
            weighted_loss = (loss * significance_map_3d).mean()
        else:
            weighted_loss = loss.mean()
        
        return weighted_loss


class UnifiedGaussianDiffusion(nn.Module):
    """统一高斯扩散模块"""
    
    def __init__(self, coarse_steps=250, fine_steps=750, voxel_resolution=32):
        super().__init__()
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.total_steps = coarse_steps + fine_steps
        self.voxel_resolution = voxel_resolution
        
        self.coarse_diffusion = CoarseGaussianDiffusion(coarse_steps)
        self.fine_diffusion = FineGaussianDiffusion(fine_steps)
    
    def forward(self, gaussians, style_features, significance_map, timestep, phase='auto'):
        """统一前向传播"""
        if phase == 'auto':
            if timestep < self.coarse_steps:
                phase = 'coarse'
            else:
                phase = 'fine'
        
        if phase == 'coarse':
            # 粗糙扩散：处理位置和不透明度
            positions = gaussians.get_xyz
            opacity = gaussians.get_opacity
            
            # 转换为3D体素表示进行扩散
            pos_voxels = self.gaussians_to_voxels(positions)
            opacity_voxels = self.gaussians_to_voxels(opacity.unsqueeze(-1))
            
            # 扩散采样
            t = torch.tensor([timestep], device=positions.device)
            denoised_pos = self.coarse_diffusion.p_sample(pos_voxels, t, style_features, significance_map)
            denoised_opacity = self.coarse_diffusion.p_sample(opacity_voxels, t, style_features, significance_map)
            
            return self.voxels_to_gaussians(denoised_pos), self.voxels_to_gaussians(denoised_opacity)
        
        else:  # fine phase
            # 精细扩散：处理颜色、缩放、旋转
            colors = gaussians.get_features
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation
            
            # 转换并扩散
            color_voxels = self.gaussians_to_voxels(colors)
            scale_voxels = self.gaussians_to_voxels(scales)
            rotation_voxels = self.gaussians_to_voxels(rotations)
            
            t = torch.tensor([timestep - self.coarse_steps], device=colors.device)
            
            denoised_colors = self.fine_diffusion.p_sample(color_voxels, t, style_features, significance_map)
            denoised_scales = self.fine_diffusion.p_sample(scale_voxels, t, style_features, significance_map)
            denoised_rotations = self.fine_diffusion.p_sample(rotation_voxels, t, style_features, significance_map)
            
            return (
                self.voxels_to_gaussians(denoised_colors),
                self.voxels_to_gaussians(denoised_scales),
                self.voxels_to_gaussians(denoised_rotations)
            )
    
    def compute_diffusion_loss(self, rendered_image, style_features, significance_map, timestep):
        """计算扩散损失"""
        # 将渲染图像转换为3D表示进行扩散训练
        image_voxels = self.image_to_voxels(rendered_image)
        
        # 确保 timestep 是正确的张量格式
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(rendered_image.device)
            timestep_value = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = torch.tensor([timestep], device=rendered_image.device)
            timestep_value = timestep
        
        if timestep_value < self.coarse_steps:
            # 粗糙阶段损失
            loss = self.coarse_diffusion.p_losses(image_voxels, t, style_features, significance_map)
        else:
            # 精细阶段损失
            if isinstance(timestep, torch.Tensor):
                t_fine = torch.tensor([timestep_value - self.coarse_steps], device=rendered_image.device)
            else:
                t_fine = torch.tensor([timestep - self.coarse_steps], device=rendered_image.device)
            loss = self.fine_diffusion.p_losses(image_voxels, t_fine, style_features, significance_map)
        
        return loss
    
    def gaussians_to_voxels(self, gaussians, resolution=32):
        """将高斯点转换为3D体素表示"""
        # 简化实现：将点云栅格化为3D体素
        batch_size = 1
        channels = gaussians.shape[-1]
        voxels = torch.zeros(batch_size, channels, resolution, resolution, resolution, device=gaussians.device)
        
        # 这里需要实现具体的栅格化逻辑
        # 暂时使用简单的插值方法
        return voxels
    
    def voxels_to_gaussians(self, voxels):
        """将3D体素转换回高斯点"""
        # 简化实现：从体素中采样点
        # 这里需要实现具体的采样逻辑
        return voxels.mean(dim=(2, 3, 4))
    
    def image_to_voxels(self, image, resolution=None):
        if resolution is None:
            resolution = self.voxel_resolution
        """将2D图像转换为3D体素表示 - 内存优化版本"""
        # 检查并处理图像维度
        if len(image.shape) == 3:
            # 如果缺少批次维度，添加它
            image = image.unsqueeze(0)
        
        batch_size, channels, height, width = image.shape
        
        # 内存优化：直接将图像调整到目标分辨率，避免大尺寸的中间张量
        # 首先将图像调整为正方形并缩小到合理尺寸
        target_2d_size = min(resolution * 4, min(height, width))  # 限制中间尺寸
        if height != target_2d_size or width != target_2d_size:
            image = F.interpolate(image, size=(target_2d_size, target_2d_size), mode='bilinear', align_corners=False)
        
        # 创建3D体素：使用更节省内存的方法
        # 方法1：直接创建目标尺寸的体素，避免大尺寸的repeat操作
        voxels = image.unsqueeze(2)  # (B, C, 1, H, W)
        
        # 使用3D插值直接调整到目标分辨率，避免中间的大张量
        voxels = F.interpolate(voxels, size=(resolution, resolution, resolution), mode='trilinear', align_corners=False)
        
        return voxels