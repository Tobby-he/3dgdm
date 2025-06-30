import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion

def test_renderer_output_format():
    """测试渲染器输出格式的处理"""
    print("=== 测试渲染器输出格式处理 ===")
    
    # 创建UnifiedGaussianDiffusion实例
    diffusion = UnifiedGaussianDiffusion(
        num_timesteps=1000,
        coarse_steps=500,
        voxel_resolution=32
    )
    
    # 模拟渲染器输出：(C, H, W) 格式，这是实际的渲染器输出格式
    rendered_image = torch.randn(3, 800, 800)  # 典型的渲染器输出
    print(f"模拟渲染器输出形状: {rendered_image.shape}")
    
    try:
        # 调用image_to_voxels方法
        voxels = diffusion.image_to_voxels(rendered_image, resolution=32)
        print(f"转换后体素形状: {voxels.shape}")
        print(f"期望形状: (1, 3, 32, 32, 32)")
        
        if voxels.shape == (1, 3, 32, 32, 32):
            print("✓ 成功: 渲染器输出格式处理正确")
            
            # 测试voxels_to_gaussians
            gaussians = diffusion.voxels_to_gaussians(voxels)
            print(f"高斯输出形状: {gaussians.shape}")
            print(f"期望形状: (1, 3)")
            
            if gaussians.shape == (1, 3):
                print("✓ 成功: 完整的图像->体素->高斯转换链正常")
            else:
                print("✗ 失败: voxels_to_gaussians输出形状不正确")
        else:
            print("✗ 失败: image_to_voxels输出形状不正确")
            
    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("=== 测试完成 ===")

def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 测试边缘情况 ===")
    
    diffusion = UnifiedGaussianDiffusion(
        num_timesteps=1000,
        coarse_steps=500,
        voxel_resolution=32
    )
    
    # 测试非正方形图像
    non_square_image = torch.randn(3, 600, 800)
    print(f"\n非正方形图像: {non_square_image.shape}")
    
    try:
        voxels = diffusion.image_to_voxels(non_square_image, resolution=32)
        print(f"输出体素形状: {voxels.shape}")
        if voxels.shape == (1, 3, 32, 32, 32):
            print("✓ 成功: 非正方形图像处理正确")
        else:
            print("✗ 失败: 非正方形图像处理不正确")
    except Exception as e:
        print(f"✗ 错误: {str(e)}")
    
    # 测试已有批次维度的图像
    batched_image = torch.randn(2, 3, 512, 512)
    print(f"\n批次图像: {batched_image.shape}")
    
    try:
        voxels = diffusion.image_to_voxels(batched_image, resolution=32)
        print(f"输出体素形状: {voxels.shape}")
        if voxels.shape == (2, 3, 32, 32, 32):
            print("✓ 成功: 批次图像处理正确")
        else:
            print("✗ 失败: 批次图像处理不正确")
    except Exception as e:
        print(f"✗ 错误: {str(e)}")
    
    print("=== 边缘情况测试完成 ===")

if __name__ == "__main__":
    test_renderer_output_format()
    test_edge_cases()
    print("\n所有测试完成！")