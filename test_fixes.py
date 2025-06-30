#!/usr/bin/env python3
"""
测试所有修复是否正确工作
"""

import torch
import torch.nn.functional as F
from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion
from diffusion.enhanced_significance_attention import EnhancedSignificanceAttention
from diffusion.adaptive_style_feature_field import AdaptiveStyleFeatureField

def test_dimension_fixes():
    """测试维度修复"""
    print("=== 测试维度修复 ===")
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同维度的图像
    image_3d = torch.randn(3, 256, 256).to(device)  # 缺少批次维度
    image_4d = torch.randn(1, 3, 256, 256).to(device)  # 有批次维度
    style_image = torch.randn(3, 256, 256).to(device)
    timestep = torch.randint(0, 1000, (1,)).to(device)
    
    try:
        # 测试 UnifiedGaussianDiffusion
        print("测试 UnifiedGaussianDiffusion...")
        diffusion = UnifiedGaussianDiffusion().to(device)
        
        # 测试 image_to_voxels 方法
        voxels_3d = diffusion.image_to_voxels(image_3d)
        voxels_4d = diffusion.image_to_voxels(image_4d)
        print(f"3D图像转体素: {image_3d.shape} -> {voxels_3d.shape}")
        print(f"4D图像转体素: {image_4d.shape} -> {voxels_4d.shape}")
        
        # 测试 EnhancedSignificanceAttention
        print("\n测试 EnhancedSignificanceAttention...")
        attention = EnhancedSignificanceAttention().to(device)
        
        sig_map_3d = attention(image_3d, style_image)
        sig_map_4d = attention(image_4d, style_image)
        print(f"3D图像显著性: {image_3d.shape} -> {sig_map_3d.shape}")
        print(f"4D图像显著性: {image_4d.shape} -> {sig_map_4d.shape}")
        
        # 测试 AdaptiveStyleFeatureField
        print("\n测试 AdaptiveStyleFeatureField...")
        style_field = AdaptiveStyleFeatureField().to(device)
        
        style_features_3d, intensity_3d = style_field(style_image, timestep)
        style_features_4d, intensity_4d = style_field(image_4d, timestep)
        print(f"3D风格特征: {style_image.shape} -> {len(style_features_3d)} 层特征")
        print(f"4D风格特征: {image_4d.shape} -> {len(style_features_4d)} 层特征")
        
        print("\n✅ 所有维度测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timestep_handling():
    """测试时间步处理"""
    print("\n=== 测试时间步处理 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = UnifiedGaussianDiffusion().to(device)
    
    # 测试数据
    image = torch.randn(1, 3, 64, 64).to(device)
    style_features = [torch.randn(1, 64, 32, 32).to(device)]
    significance_map = torch.randn(1, 1, 64, 64).to(device)
    
    try:
        # 测试不同格式的时间步
        timestep_int = 500
        timestep_tensor = torch.tensor([500]).to(device)
        timestep_scalar = torch.tensor(500).to(device)
        
        print("测试整数时间步...")
        loss1 = diffusion.compute_diffusion_loss(image, style_features, significance_map, timestep_int)
        print(f"整数时间步损失: {loss1.item():.6f}")
        
        print("测试张量时间步...")
        loss2 = diffusion.compute_diffusion_loss(image, style_features, significance_map, timestep_tensor)
        print(f"张量时间步损失: {loss2.item():.6f}")
        
        print("测试标量张量时间步...")
        loss3 = diffusion.compute_diffusion_loss(image, style_features, significance_map, timestep_scalar)
        print(f"标量张量时间步损失: {loss3.item():.6f}")
        
        print("\n✅ 时间步处理测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 时间步测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试所有修复...\n")
    
    success1 = test_dimension_fixes()
    success2 = test_timestep_handling()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！修复成功！")
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查。")