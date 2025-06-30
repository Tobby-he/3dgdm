#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应风格特征场模块
根据扩散阶段动态调整风格强度，实现渐进式风格迁移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19
import math


class DiffusionStageEncoder(nn.Module):
    """扩散阶段编码器：将时间步编码为特征"""
    
    def __init__(self, max_timesteps=1000, embed_dim=128):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.embed_dim = embed_dim
        
        # 位置编码
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, timesteps):
        """编码时间步"""
        # 正弦位置编码
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.time_embedding(emb)


class StyleIntensityController(nn.Module):
    """风格强度控制器：根据扩散阶段调整风格强度"""
    
    def __init__(self, time_embed_dim=128, style_dim=512):
        super().__init__()
        
        # 时间条件网络
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        # 风格强度预测网络
        self.intensity_predictor = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim // 2),
            nn.SiLU(),
            nn.Linear(style_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的强度值
        )
        
        # 多层级强度控制 - 对应VGG各层的通道数
        vgg_channels = [64, 128, 256, 512, 512]  # VGG19各层通道数
        self.layer_controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_embed_dim + channels, style_dim),
                nn.SiLU(),
                nn.Linear(style_dim, 1),
                nn.Sigmoid()
            ) for channels in vgg_channels
        ])
        
    def forward(self, time_embed, style_features):
        """计算风格强度"""
        # 时间条件投影
        time_proj = self.time_proj(time_embed)
        
        # 全局风格强度
        if isinstance(style_features, list):
            # 如果是多层特征，使用最后一层
            global_style_feat = style_features[-1].mean(dim=[2, 3])  # 全局平均池化
        else:
            global_style_feat = style_features.mean(dim=[2, 3])
        
        combined_feat = torch.cat([time_proj, global_style_feat], dim=-1)
        global_intensity = self.intensity_predictor(combined_feat)
        
        # 层级风格强度
        layer_intensities = []
        if isinstance(style_features, list):
            for i, (controller, style_feat) in enumerate(zip(self.layer_controllers, style_features)):
                layer_feat = style_feat.mean(dim=[2, 3])  # [B, C]
                layer_input = torch.cat([time_embed, layer_feat], dim=-1)
                layer_intensity = controller(layer_input)
                layer_intensities.append(layer_intensity)
        else:
            # 如果只有一个特征，为所有层使用相同的强度
            for controller in self.layer_controllers:
                layer_feat = style_features.mean(dim=[2, 3])
                layer_input = torch.cat([time_embed, layer_feat], dim=-1)
                layer_intensity = controller(layer_input)
                layer_intensities.append(layer_intensity)
        
        return global_intensity, layer_intensities


class AdaptiveStyleExtractor(nn.Module):
    """自适应风格提取器"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        # VGG19特征提取器
        vgg = vgg19(pretrained=pretrained).features
        
        # 分层提取
        self.layer1 = vgg[:4]   # relu1_2
        self.layer2 = vgg[4:9]  # relu2_2
        self.layer3 = vgg[9:18] # relu3_4
        self.layer4 = vgg[18:27] # relu4_4
        self.layer5 = vgg[27:36] # relu5_4
        
        # 冻结预训练权重
        for param in self.parameters():
            param.requires_grad = False
        
        # 自适应特征调制器
        self.feature_modulators = nn.ModuleList([
            FeatureModulator(64),   # layer1
            FeatureModulator(128),  # layer2
            FeatureModulator(256),  # layer3
            FeatureModulator(512),  # layer4
            FeatureModulator(512),  # layer5
        ])
        
    def forward(self, style_image, layer_intensities=None):
        """提取自适应风格特征"""
        features = []
        x = style_image
        
        # 逐层提取和调制
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        
        for i, layer in enumerate(layers):
            x = layer(x)
            
            # 如果提供了层级强度，进行特征调制
            if layer_intensities is not None:
                intensity = layer_intensities[i]
                x = self.feature_modulators[i](x, intensity)
            
            features.append(x)
        
        return features


class FeatureModulator(nn.Module):
    """特征调制器：根据强度调制特征"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # 强度调制网络
        self.intensity_modulator = nn.Sequential(
            nn.Linear(1, channels),
            nn.Sigmoid()
        )
        
    def forward(self, features, intensity):
        """特征调制"""
        B, C, H, W = features.shape
        
        # 通道注意力
        channel_weights = self.channel_attention(features)
        
        # 强度调制
        intensity_weights = self.intensity_modulator(intensity).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 组合调制
        modulated_features = features * channel_weights * intensity_weights
        
        return modulated_features


class ProgressiveStyleField(nn.Module):
    """渐进式风格场：构建3D风格特征场"""
    
    def __init__(self, style_dim=512, field_resolution=64):
        super().__init__()
        self.style_dim = style_dim
        self.field_resolution = field_resolution
        
        # 3D风格场生成网络
        self.field_generator = nn.Sequential(
            nn.Linear(style_dim + 3, 256),  # style + xyz
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, style_dim)
        )
        
        # 空间编码器
        self.spatial_encoder = SpatialEncoder()
        
        # 风格一致性网络
        self.consistency_net = nn.Sequential(
            nn.Conv3d(style_dim, style_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(style_dim // 2, style_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(style_dim // 4, style_dim, 3, padding=1)
        )
        
    def forward(self, style_features, gaussian_positions, global_intensity):
        """生成渐进式风格场"""
        # 全局风格特征
        if isinstance(style_features, list):
            global_style = style_features[-1].mean(dim=[2, 3])  # [B, style_dim]
        else:
            global_style = style_features.mean(dim=[2, 3])
        
        # 应用全局强度
        global_style = global_style * global_intensity
        
        # 为每个高斯点生成风格特征
        N = gaussian_positions.shape[0]
        B = global_style.shape[0]
        
        # 空间编码
        spatial_codes = self.spatial_encoder(gaussian_positions)  # [N, 3]
        
        # 扩展全局风格特征
        expanded_style = global_style.unsqueeze(1).expand(B, N, -1)  # [B, N, style_dim]
        
        # 组合空间和风格信息
        combined_input = torch.cat([
            expanded_style.reshape(-1, self.style_dim),  # [B*N, style_dim]
            spatial_codes.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)  # [B*N, 3]
        ], dim=-1)  # [B*N, style_dim + 3]
        
        # 生成局部风格特征
        local_style_features = self.field_generator(combined_input)  # [B*N, style_dim]
        local_style_features = local_style_features.reshape(B, N, self.style_dim)
        
        return local_style_features
    
    def generate_3d_field(self, style_features, global_intensity, field_size=(64, 64, 64)):
        """生成完整的3D风格场"""
        D, H, W = field_size
        
        # 创建3D网格
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        z = torch.linspace(-1, 1, D)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
        grid_points = grid_points.to(style_features.device if isinstance(style_features, torch.Tensor) else style_features[0].device)
        
        # 生成风格场
        with torch.no_grad():
            field_features = self.forward(style_features, grid_points, global_intensity)
        
        # 重塑为3D场
        field_3d = field_features.reshape(-1, self.style_dim, D, H, W)
        
        # 应用一致性约束
        field_3d = self.consistency_net(field_3d)
        
        return field_3d


class SpatialEncoder(nn.Module):
    """空间位置编码器"""
    
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        
    def forward(self, positions):
        """位置编码"""
        # 正弦位置编码
        encoded = [positions]
        
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * positions))
            encoded.append(torch.cos(freq * positions))
        
        return torch.cat(encoded, dim=-1)


class AdaptiveStyleFeatureField(nn.Module):
    """自适应风格特征场主模块"""
    
    def __init__(self, max_timesteps=1000, style_dim=512, field_resolution=64):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.style_dim = style_dim
        
        # 扩散阶段编码器
        self.stage_encoder = DiffusionStageEncoder(max_timesteps, embed_dim=128)
        
        # 风格强度控制器
        self.intensity_controller = StyleIntensityController(time_embed_dim=128, style_dim=style_dim)
        
        # 自适应风格提取器
        self.style_extractor = AdaptiveStyleExtractor()
        
        # 渐进式风格场
        self.style_field = ProgressiveStyleField(style_dim, field_resolution)
        
        # 风格一致性损失权重
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, style_image, timesteps, gaussian_positions=None, return_field=False):
        """前向传播"""
        # 编码扩散阶段
        time_embed = self.stage_encoder(timesteps)
        
        # 提取初始风格特征
        initial_style_features = self.style_extractor(style_image)
        
        # 计算风格强度
        global_intensity, layer_intensities = self.intensity_controller(
            time_embed, initial_style_features
        )
        
        # 提取自适应风格特征
        adaptive_style_features = self.style_extractor(style_image, layer_intensities)
        
        # 如果需要生成风格场
        if gaussian_positions is not None:
            style_field_features = self.style_field(
                adaptive_style_features, gaussian_positions, global_intensity
            )
            
            if return_field:
                # 生成完整3D场
                field_3d = self.style_field.generate_3d_field(
                    adaptive_style_features, global_intensity
                )
                return adaptive_style_features, style_field_features, field_3d, global_intensity
            
            return adaptive_style_features, style_field_features, global_intensity
        
        return adaptive_style_features, global_intensity
    
    def compute_consistency_loss(self, style_field_features):
        """计算风格一致性损失"""
        if style_field_features.dim() == 3:  # [B, N, style_dim]
            B, N, C = style_field_features.shape
            
            # 计算相邻点的特征差异
            # 简化处理：随机采样相邻点对
            num_pairs = min(1000, N // 2)
            indices = torch.randperm(N)[:num_pairs]
            
            feat1 = style_field_features[:, indices]
            feat2 = style_field_features[:, indices + 1]
            
            # L2一致性损失
            consistency_loss = F.mse_loss(feat1, feat2)
            
            return self.consistency_weight * consistency_loss
        
        return torch.tensor(0.0, device=style_field_features.device)
    
    def get_style_intensity_schedule(self, total_timesteps=1000):
        """获取风格强度调度"""
        timesteps = torch.arange(total_timesteps)
        time_embed = self.stage_encoder(timesteps)
        
        # 使用虚拟风格特征计算强度调度
        dummy_style = torch.randn(1, 512)  # 虚拟风格特征
        
        with torch.no_grad():
            intensities = []
            for t_emb in time_embed:
                global_intensity, _ = self.intensity_controller(
                    t_emb.unsqueeze(0), dummy_style
                )
                intensities.append(global_intensity.item())
        
        return np.array(intensities)
    
    def visualize_style_field(self, style_image, timestep, field_size=(32, 32, 32)):
        """可视化风格场"""
        with torch.no_grad():
            timesteps = torch.tensor([timestep])
            _, _, field_3d, _ = self.forward(
                style_image, timesteps, return_field=True
            )
            
            # 返回3D场的切片用于可视化
            mid_slice = field_3d[0, :, field_size[0]//2, :, :]
            return mid_slice