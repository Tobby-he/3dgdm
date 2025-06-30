import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion

def quick_verification():
    """快速验证修复是否正确"""
    print("=== 快速验证修复 ===")
    
    try:
        # 创建扩散模型实例
        diffusion = UnifiedGaussianDiffusion(
            num_timesteps=1000,
            coarse_steps=500,
            voxel_resolution=32
        )
        print("✓ UnifiedGaussianDiffusion 初始化成功")
        
        # 模拟渲染器输出 (C, H, W) - 这是导致原始错误的格式
        rendered_image = torch.randn(3, 800, 800)
        print(f"输入图像形状: {rendered_image.shape}")
        
        # 测试 image_to_voxels
        voxels = diffusion.image_to_voxels(rendered_image, resolution=32)
        print(f"体素形状: {voxels.shape}")
        
        # 验证形状是否正确
        expected_shape = (1, 3, 32, 32, 32)
        if voxels.shape == expected_shape:
            print("✓ image_to_voxels 修复成功")
        else:
            print(f"✗ image_to_voxels 仍有问题，期望 {expected_shape}，得到 {voxels.shape}")
            return False
        
        # 测试 voxels_to_gaussians
        gaussians = diffusion.voxels_to_gaussians(voxels)
        print(f"高斯形状: {gaussians.shape}")
        
        expected_gaussian_shape = (1, 3)
        if gaussians.shape == expected_gaussian_shape:
            print("✓ voxels_to_gaussians 工作正常")
        else:
            print(f"✗ voxels_to_gaussians 有问题，期望 {expected_gaussian_shape}，得到 {gaussians.shape}")
            return False
        
        # 测试完整的扩散损失计算流程
        style_features = torch.randn(1, 256)
        significance_map = torch.randn(1, 1, 800, 800)
        timestep = 100
        
        print("\n测试完整的扩散损失计算...")
        loss = diffusion.compute_diffusion_loss(rendered_image, style_features, significance_map, timestep)
        print(f"扩散损失: {loss.item():.6f}")
        print("✓ 完整的扩散损失计算成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_verification()
    if success:
        print("\n🎉 所有修复验证通过！训练应该可以正常进行了。")
    else:
        print("\n❌ 仍有问题需要解决。")