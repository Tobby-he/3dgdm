#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的显著性注意力模块
实现真正的显著性建模：判断哪部分重要，结合内容和风格的交叉注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import math


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用VGG19作为骨干网络
        vgg = vgg19(pretrained=pretrained).features
        
        # 分层提取特征
        self.layer1 = vgg[:4]   # relu1_2
        self.layer2 = vgg[4:9]  # relu2_2
        self.layer3 = vgg[9:18] # relu3_4
        self.layer4 = vgg[18:27] # relu4_4
        self.layer5 = vgg[27:36] # relu5_4
        
        # 冻结预训练权重
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """提取多尺度特征"""
        features = []
        
        x = self.layer1(x)
        features.append(x)  # 1/1
        
        x = self.layer2(x)
        features.append(x)  # 1/2
        
        x = self.layer3(x)
        features.append(x)  # 1/4
        
        x = self.layer4(x)
        features.append(x)  # 1/8
        
        x = self.layer5(x)
        features.append(x)  # 1/16
        
        return features


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 投影层
        self.content_proj_q = nn.Linear(dim, dim)
        self.style_proj_k = nn.Linear(dim, dim)
        self.style_proj_v = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, content_feat, style_feat):
        """交叉注意力计算"""
        B, C, H, W = content_feat.shape
        
        # 重塑为序列格式
        content_seq = content_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        style_seq = style_feat.flatten(2).transpose(1, 2)      # [B, HW, C]
        
        # 多头注意力
        attended_feat, attention_weights = self.multi_head_cross_attention(
            content_seq, style_seq, style_seq
        )
        
        # 残差连接和层归一化
        attended_feat = self.norm1(attended_feat + content_seq)
        
        # 前馈网络
        ffn_out = self.ffn(attended_feat)
        attended_feat = self.norm2(attended_feat + ffn_out)
        
        # 重塑回特征图格式
        attended_feat = attended_feat.transpose(1, 2).reshape(B, C, H, W)
        attention_weights = attention_weights.mean(dim=1)  # 平均多头注意力权重
        
        return attended_feat, attention_weights
    
    def multi_head_cross_attention(self, q, k, v):
        """多头交叉注意力（内存优化版本）"""
        B, N_q, C = q.shape
        B, N_k, C = k.shape
        B, N_v, C = v.shape
        
        # 内存优化：如果序列太长，使用分块处理
        max_seq_len = 256  # 进一步减小最大序列长度限制以适应4GB GPU
        
        if N_q > max_seq_len or N_k > max_seq_len:
            return self._chunked_attention(q, k, v, max_seq_len)
        
        # 投影
        q = self.content_proj_q(q).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.style_proj_k(k).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.style_proj_v(v).reshape(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力权重
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        
        return out, attn
    
    def _chunked_attention(self, q, k, v, chunk_size):
        """分块注意力计算以节省内存"""
        B, N_q, C = q.shape
        B, N_k, C = k.shape
        
        # 进一步减小chunk_size以适应4GB GPU
        chunk_size = min(chunk_size, 128)  # 进一步减小到128
        
        # 投影
        q = self.content_proj_q(q).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.style_proj_k(k).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.style_proj_v(v).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 分块处理查询
        outputs = []
        attentions = []
        
        for i in range(0, N_q, chunk_size):
            end_i = min(i + chunk_size, N_q)
            q_chunk = q[:, :, i:end_i, :]
            
            # 计算注意力权重
            attn_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            
            # 应用注意力权重
            out_chunk = (attn_chunk @ v).transpose(1, 2).reshape(B, end_i - i, C)
            
            outputs.append(out_chunk.detach())
            attentions.append(attn_chunk.mean(dim=1).detach())  # 平均多头权重
            
            # 清理中间变量
            del attn_chunk, out_chunk, q_chunk
            torch.cuda.empty_cache()
        
        # 拼接结果
        out = torch.cat(outputs, dim=1)
        out = self.out_proj(out)
        attn = torch.cat(attentions, dim=1)
        
        return out, attn


class SignificancePredictor(nn.Module):
    """显著性预测器"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度显著性预测
        self.significance_heads = nn.ModuleList([
            nn.Conv2d(hidden_dim // 2, 1, 1),  # 细节显著性
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),  # 局部显著性
            nn.Conv2d(hidden_dim // 2, 1, 5, padding=2),  # 全局显著性
        ])
        
        # 最终融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = SpatialAttention()
        
    def forward(self, fused_features):
        """预测显著性图"""
        # 特征融合
        x = self.fusion_conv(fused_features)
        
        # 多尺度显著性预测
        significance_maps = []
        for head in self.significance_heads:
            sig_map = head(x)
            significance_maps.append(sig_map)
        
        # 拼接多尺度结果
        multi_scale_sig = torch.cat(significance_maps, dim=1)
        
        # 最终显著性图
        significance_map = self.final_conv(multi_scale_sig)
        
        # 应用空间注意力
        significance_map = self.spatial_attention(significance_map)
        
        return significance_map


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """空间注意力计算"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sigmoid(self.conv(spatial_input))
        return x * spatial_weight


class GaussianSignificanceMapper(nn.Module):
    """高斯显著性映射器：将2D显著性映射到3D高斯点"""
    
    def __init__(self):
        super().__init__()
        # 深度估计网络
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 3D投影网络
        self.projection_mlp = nn.Sequential(
            nn.Linear(4, 64),  # [x, y, depth, significance]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, significance_map_2d, rendered_image, gaussians_xyz):
        """将2D显著性映射到3D高斯点"""
        B, C, H, W = significance_map_2d.shape
        N = gaussians_xyz.shape[0]  # 高斯点数量
        
        # 估计深度
        depth_map = self.depth_estimator(rendered_image)
        
        # 将高斯点投影到图像平面
        # 这里简化处理，实际需要相机参数
        projected_coords = self.project_gaussians_to_image(gaussians_xyz, H, W)
        
        # 从2D显著性图中采样
        sampled_significance = F.grid_sample(
            significance_map_2d, 
            projected_coords.unsqueeze(0).unsqueeze(0),
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        ).squeeze()
        
        # 从深度图中采样
        sampled_depth = F.grid_sample(
            depth_map,
            projected_coords.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze()
        
        # 构建3D特征
        features_3d = torch.cat([
            projected_coords,  # [N, 2]
            sampled_depth.unsqueeze(-1),  # [N, 1]
            sampled_significance.unsqueeze(-1)  # [N, 1]
        ], dim=-1)  # [N, 4]
        
        # 预测3D显著性
        gaussian_significance = self.projection_mlp(features_3d).squeeze(-1)  # [N]
        
        return gaussian_significance
    
    def project_gaussians_to_image(self, gaussians_xyz, height, width):
        """将3D高斯点投影到图像平面"""
        # 简化的正交投影
        # 实际应用中需要使用相机内外参数
        x_coords = gaussians_xyz[:, 0]
        y_coords = gaussians_xyz[:, 1]
        
        # 归一化到[-1, 1]范围
        x_norm = 2.0 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1.0
        y_norm = 2.0 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1.0
        
        projected_coords = torch.stack([x_norm, y_norm], dim=-1)
        return projected_coords


class EnhancedSignificanceAttention(nn.Module):
    """增强的显著性注意力主模块"""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 多尺度特征提取
        self.content_extractor = MultiScaleFeatureExtractor()
        self.style_extractor = MultiScaleFeatureExtractor()
        
        # 特征对齐网络
        self.feature_align = nn.ModuleList([
            nn.Conv2d(64, feature_dim, 1),   # layer1
            nn.Conv2d(128, feature_dim, 1),  # layer2
            nn.Conv2d(256, feature_dim, 1),  # layer3
            nn.Conv2d(512, feature_dim, 1),  # layer4
            nn.Conv2d(512, feature_dim, 1),  # layer5
        ])
        
        # 交叉注意力融合
        self.cross_attention = CrossAttentionFusion(feature_dim)
        
        # 显著性预测
        self.significance_predictor = SignificancePredictor(feature_dim)
        
        # 高斯映射器
        self.gaussian_mapper = GaussianSignificanceMapper()
        
        # 多尺度融合权重
        self.scale_weights = nn.Parameter(torch.ones(5) / 5)
        
    def forward(self, rendered_image, style_image, gaussians_xyz=None):
        """前向传播"""
        # 确保输入图像格式正确
        if rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)
        if style_image.dim() == 3:
            style_image = style_image.unsqueeze(0)
        
        # 提取多尺度特征
        content_features = self.content_extractor(rendered_image)
        style_features = self.style_extractor(style_image)
        
        # 特征对齐和融合
        fused_features_list = []
        attention_weights_list = []
        
        for i, (content_feat, style_feat) in enumerate(zip(content_features, style_features)):
            # 对齐特征维度
            content_aligned = self.feature_align[i](content_feat)
            style_aligned = self.feature_align[i](style_feat)
            
            # 内存优化：限制最大分辨率
            target_size = content_features[0].shape[2:]
            max_resolution = 64  # 进一步限制最大分辨率为64x64以节省内存
            
            if target_size[0] > max_resolution or target_size[1] > max_resolution:
                # 计算缩放比例
                scale_factor = min(max_resolution / target_size[0], max_resolution / target_size[1])
                target_size = (int(target_size[0] * scale_factor), int(target_size[1] * scale_factor))
            
            if content_aligned.shape[2:] != target_size:
                content_aligned = F.interpolate(content_aligned, size=target_size, mode='bilinear', align_corners=True)
                style_aligned = F.interpolate(style_aligned, size=target_size, mode='bilinear', align_corners=True)
            
            # 交叉注意力融合
            fused_feat, attn_weights = self.cross_attention(content_aligned, style_aligned)
            
            fused_features_list.append(fused_feat)
            attention_weights_list.append(attn_weights)
        
        # 多尺度特征加权融合
        weighted_features = sum(w * feat for w, feat in zip(self.scale_weights, fused_features_list))
        
        # 预测2D显著性图
        significance_map_2d = self.significance_predictor(weighted_features)
        
        # 如果提供了高斯点坐标，映射到3D
        if gaussians_xyz is not None:
            gaussian_significance = self.gaussian_mapper(
                significance_map_2d, rendered_image, gaussians_xyz
            )
            return significance_map_2d, gaussian_significance
        
        return significance_map_2d
    
    def get_attention_visualization(self, rendered_image, style_image):
        """获取注意力可视化"""
        with torch.no_grad():
            significance_map = self.forward(rendered_image, style_image)
            
            # 上采样到原图尺寸
            if significance_map.shape[2:] != rendered_image.shape[2:]:
                significance_map = F.interpolate(
                    significance_map, 
                    size=rendered_image.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            return significance_map