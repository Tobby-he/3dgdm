#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-GDM 模块测试脚本
用于验证各个模块是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion
    from diffusion.enhanced_significance_attention import EnhancedSignificanceAttention
    from diffusion.adaptive_style_feature_field import AdaptiveStyleFeatureField
    print("✓ 成功导入所有3D-GDM模块")
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    sys.exit(1)

def test_unified_gaussian_diffusion():
    """测试统一高斯扩散模块"""
    print("\n=== 测试统一高斯扩散模块 ===")
    
    try:
        # 初始化模型
        diffusion = UnifiedGaussianDiffusion(
            timesteps=100,  # 减少时间步用于测试
            coarse_ratio=0.7,
            unet_channels=[32, 64, 128]  # 减少通道数
        )
        
        # 创建测试数据
        batch_size = 1
        num_gaussians = 1000
        
        # 模拟高斯参数
        gaussians = {
            'xyz': torch.randn(num_gaussians, 3),
            'opacity': torch.randn(num_gaussians, 1),
            'scaling': torch.randn(num_gaussians, 3),
            'rotation': torch.randn(num_gaussians, 4)
        }
        
        # 测试前向扩散
        timesteps = torch.randint(0, 100, (batch_size,))
        noisy_gaussians = diffusion.add_noise(gaussians, timesteps)
        
        print(f"✓ 前向扩散测试通过")
        print(f"  - 原始高斯点数: {num_gaussians}")
        print(f"  - 时间步: {timesteps.item()}")
        print(f"  - 噪声高斯形状: {noisy_gaussians['xyz'].shape}")
        
        # 测试噪声预测（简化版）
        # 这里我们跳过完整的UNet测试，因为需要更多的依赖
        print("✓ 统一高斯扩散模块基础功能正常")
        
    except Exception as e:
        print(f"✗ 统一高斯扩散模块测试失败: {e}")
        return False
    
    return True

def test_enhanced_significance_attention():
    """测试增强显著性注意力模块"""
    print("\n=== 测试增强显著性注意力模块 ===")
    
    try:
        # 初始化模型
        significance_attention = EnhancedSignificanceAttention(feature_dim=256)  # 减少特征维度
        
        # 创建测试图像
        batch_size = 1
        height, width = 128, 128  # 减少图像尺寸
        
        rendered_image = torch.randn(batch_size, 3, height, width)
        style_image = torch.randn(batch_size, 3, height, width)
        
        # 创建测试高斯位置
        num_gaussians = 500
        gaussian_positions = torch.randn(num_gaussians, 3)
        
        # 测试显著性计算
        significance_map_2d, gaussian_significance = significance_attention(
            rendered_image, style_image, gaussian_positions
        )
        
        print(f"✓ 显著性注意力测试通过")
        print(f"  - 输入图像尺寸: {rendered_image.shape}")
        print(f"  - 2D显著性图尺寸: {significance_map_2d.shape}")
        print(f"  - 3D高斯显著性尺寸: {gaussian_significance.shape}")
        print(f"  - 显著性值范围: [{gaussian_significance.min():.3f}, {gaussian_significance.max():.3f}]")
        
        # 测试注意力可视化
        attention_vis = significance_attention.get_attention_visualization(
            rendered_image, style_image
        )
        print(f"✓ 注意力可视化功能正常: {attention_vis.shape}")
        
    except Exception as e:
        print(f"✗ 增强显著性注意力模块测试失败: {e}")
        return False
    
    return True

def test_adaptive_style_feature_field():
    """测试自适应风格特征场模块"""
    print("\n=== 测试自适应风格特征场模块 ===")
    
    try:
        # 初始化模型
        style_field = AdaptiveStyleFeatureField(
            max_timesteps=100,
            style_dim=256,  # 减少风格维度
            field_resolution=32  # 减少场分辨率
        )
        
        # 创建测试数据
        batch_size = 1
        height, width = 128, 128
        
        style_image = torch.randn(batch_size, 3, height, width)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        # 创建高斯位置
        num_gaussians = 300
        gaussian_positions = torch.randn(num_gaussians, 3)
        
        # 测试风格特征提取
        style_features, style_field_features, style_intensity = style_field(
            style_image, timesteps, gaussian_positions
        )
        
        print(f"✓ 自适应风格特征场测试通过")
        print(f"  - 风格图像尺寸: {style_image.shape}")
        print(f"  - 时间步: {timesteps.item()}")
        print(f"  - 风格特征数量: {len(style_features)}")
        print(f"  - 风格场特征尺寸: {style_field_features.shape}")
        print(f"  - 风格强度: {style_intensity.item():.3f}")
        
        # 测试风格强度调度
        intensity_schedule = style_field.get_style_intensity_schedule(total_timesteps=100)
        print(f"✓ 风格强度调度功能正常: 长度={len(intensity_schedule)}")
        
        # 测试一致性损失
        consistency_loss = style_field.compute_consistency_loss(style_field_features)
        print(f"✓ 一致性损失计算正常: {consistency_loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ 自适应风格特征场模块测试失败: {e}")
        return False
    
    return True

def test_integration():
    """测试模块集成"""
    print("\n=== 测试模块集成 ===")
    
    try:
        # 初始化所有模块
        diffusion = UnifiedGaussianDiffusion(
            timesteps=50,
            coarse_ratio=0.7,
            unet_channels=[32, 64]
        )
        
        significance_attention = EnhancedSignificanceAttention(feature_dim=128)
        
        style_field = AdaptiveStyleFeatureField(
            max_timesteps=50,
            style_dim=128,
            field_resolution=16
        )
        
        # 创建测试数据
        batch_size = 1
        height, width = 64, 64
        num_gaussians = 200
        
        rendered_image = torch.randn(batch_size, 3, height, width)
        style_image = torch.randn(batch_size, 3, height, width)
        gaussian_positions = torch.randn(num_gaussians, 3)
        timesteps = torch.randint(0, 50, (batch_size,))
        
        # 模拟完整流程
        print("1. 计算显著性注意力...")
        significance_map, gaussian_significance = significance_attention(
            rendered_image, style_image, gaussian_positions
        )
        
        print("2. 生成自适应风格特征...")
        style_features, style_field_features, style_intensity = style_field(
            style_image, timesteps, gaussian_positions
        )
        
        print("3. 模拟高斯参数...")
        gaussians = {
            'xyz': gaussian_positions,
            'opacity': torch.randn(num_gaussians, 1),
            'scaling': torch.randn(num_gaussians, 3),
            'rotation': torch.randn(num_gaussians, 4)
        }
        
        print("4. 添加扩散噪声...")
        noisy_gaussians = diffusion.add_noise(gaussians, timesteps)
        
        print("✓ 模块集成测试通过")
        print(f"  - 所有模块协同工作正常")
        print(f"  - 数据流转无误")
        
    except Exception as e:
        print(f"✗ 模块集成测试失败: {e}")
        return False
    
    return True

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 测试内存使用情况 ===")
    
    try:
        import psutil
        import gc
        
        # 获取初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建较大的测试数据
        significance_attention = EnhancedSignificanceAttention(feature_dim=512)
        
        batch_size = 2
        height, width = 256, 256
        num_gaussians = 2000
        
        rendered_image = torch.randn(batch_size, 3, height, width)
        style_image = torch.randn(batch_size, 3, height, width)
        gaussian_positions = torch.randn(num_gaussians, 3)
        
        # 运行测试
        significance_map, gaussian_significance = significance_attention(
            rendered_image, style_image, gaussian_positions
        )
        
        # 获取峰值内存
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # 清理内存
        del significance_attention, rendered_image, style_image
        del gaussian_positions, significance_map, gaussian_significance
        gc.collect()
        
        print(f"✓ 内存使用测试完成")
        print(f"  - 初始内存: {initial_memory:.1f} MB")
        print(f"  - 峰值内存: {peak_memory:.1f} MB")
        print(f"  - 内存增长: {memory_increase:.1f} MB")
        
        if memory_increase > 2000:  # 2GB
            print(f"⚠ 警告: 内存使用较高，建议优化")
        
    except ImportError:
        print("⚠ psutil未安装，跳过内存测试")
    except Exception as e:
        print(f"✗ 内存测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始3D-GDM模块测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行测试
    tests = [
        ("统一高斯扩散", test_unified_gaussian_diffusion),
        ("增强显著性注意力", test_enhanced_significance_attention),
        ("自适应风格特征场", test_adaptive_style_feature_field),
        ("模块集成", test_integration),
        ("内存使用", test_memory_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 总结
    print(f"\n{'='*50}")
    print(f"测试总结")
    print(f"{'='*50}")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！3D-GDM模块工作正常。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)