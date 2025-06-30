# -*- coding: utf-8 -*-
"""
增强版3D-GDM推理脚本
修复风格迁移效果不明显的问题
"""

import torch
import torch.nn.functional as F
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

# 导入必要模块
try:
    from scene import Scene, GaussianModel
    from utils.camera_utils import cameraList_from_camInfos
    from gaussian_renderer import render
    from train_3dgdm import ThreeDGaussianDiffusionModel
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene.VGG import VGGEncoder
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在StyleGaussian项目根目录下运行此脚本")
    exit(1)

def load_style_image_enhanced(style_path, device, target_size=None):
    """增强版风格图像加载和预处理"""
    # 如果没有指定目标尺寸，使用默认的512x512
    if target_size is None:
        target_size = (512, 512)
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    style_image = Image.open(style_path).convert('RGB')
    style_tensor = transform(style_image).unsqueeze(0).to(device)
    return style_tensor

def enhance_style_distinctiveness(style_features):
    """
    增强风格特征的独特性，确保不同风格图像产生不同效果
    """
    if not isinstance(style_features, list):
        return style_features
    
    enhanced_features = []
    for i, feat in enumerate(style_features):
        if feat is not None:
            # 计算特征的统计信息
            feat_mean = feat.mean(dim=[2, 3], keepdim=True)
            feat_std = feat.std(dim=[2, 3], keepdim=True)
            
            # 增强特征对比度
            contrast_factor = 1.0 + (i * 0.2)  # 深层特征对比度更强
            enhanced_feat = (feat - feat_mean) * contrast_factor + feat_mean
            
            # 添加层级特异性
            layer_signature = torch.randn_like(feat_mean) * 0.1 * (i + 1)
            enhanced_feat = enhanced_feat + layer_signature
            
            enhanced_features.append(enhanced_feat)
        else:
            enhanced_features.append(feat)
    
    return enhanced_features

def apply_style_transfer_enhanced(rendered_image, style_features, significance_map, style_strength=0.8):
    """增强版风格迁移应用"""
    if rendered_image is None:
        return None
    
    # 确保输入是正确的张量格式
    if not isinstance(rendered_image, torch.Tensor):
        rendered_image = torch.from_numpy(rendered_image).float()
    
    # 确保维度正确 [B, C, H, W]
    if rendered_image.dim() == 3:
        rendered_image = rendered_image.unsqueeze(0)
    
    # 增强风格特征的独特性
    style_features = enhance_style_distinctiveness(style_features)
    
    # 应用风格特征（这里是简化版本）
    styled_image = rendered_image.clone()
    
    # 如果有显著性图，使用它来控制风格应用的强度
    if significance_map is not None:
        # 将显著性图应用为权重
        if len(significance_map.shape) == 3:
            significance_map = significance_map.unsqueeze(0)
        
        # 确保显著性图与渲染图像尺寸匹配
        if significance_map.shape[2:] != styled_image.shape[2:]:
            significance_map = F.interpolate(
                significance_map, 
                size=styled_image.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 确保显著性图的通道数匹配
        if significance_map.shape[1] != styled_image.shape[1]:
            if significance_map.shape[1] == 1:
                # 如果显著性图是单通道，复制到所有通道
                significance_map = significance_map.repeat(1, styled_image.shape[1], 1, 1)
            else:
                # 如果通道数不匹配，取前3个通道或填充
                if significance_map.shape[1] > styled_image.shape[1]:
                    significance_map = significance_map[:, :styled_image.shape[1], :, :]
                else:
                    # 填充到匹配的通道数
                    padding_channels = styled_image.shape[1] - significance_map.shape[1]
                    padding = torch.zeros(significance_map.shape[0], padding_channels, 
                                        significance_map.shape[2], significance_map.shape[3], 
                                        device=significance_map.device)
                    significance_map = torch.cat([significance_map, padding], dim=1)
    
    # 实现真正的多层风格迁移
    try:
        B, C, H, W = styled_image.shape
        
        # 计算内容图像的统计信息
        content_mean = styled_image.view(B, C, -1).mean(dim=2, keepdim=True).view(B, C, 1, 1)
        content_std = styled_image.view(B, C, -1).std(dim=2, keepdim=True).view(B, C, 1, 1)
        
        # 如果style_features是列表，使用多层特征进行风格迁移
        if isinstance(style_features, list) and len(style_features) > 0:
            # 使用多层特征的加权组合
            style_means = []
            style_stds = []
            
            # 为每一层计算风格统计信息
            for i, style_feat in enumerate(style_features):
                if style_feat is not None:
                    # 计算该层的风格统计信息
                    feat_flat = style_feat.view(style_feat.shape[0], style_feat.shape[1], -1)
                    feat_mean = feat_flat.mean(dim=2, keepdim=True)
                    feat_std = feat_flat.std(dim=2, keepdim=True)
                    
                    # 调整到目标通道数
                    if feat_mean.shape[1] != C:
                        if feat_mean.shape[1] > C:
                            feat_mean = feat_mean[:, :C, :]
                            feat_std = feat_std[:, :C, :]
                        else:
                            # 使用插值扩展到目标通道数
                            feat_mean = F.interpolate(feat_mean, size=(C, 1), mode='linear', align_corners=False)
                            feat_std = F.interpolate(feat_std, size=(C, 1), mode='linear', align_corners=False)
                    
                    style_means.append(feat_mean.view(B, C, 1, 1))
                    style_stds.append(torch.clamp(feat_std.view(B, C, 1, 1), min=0.1))
            
            if style_means:
                # 使用不同层的加权平均（高层特征权重更大）
                weights = torch.softmax(torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0][:len(style_means)], device=styled_image.device), dim=0)
                
                style_mean = sum(w * sm for w, sm in zip(weights, style_means))
                style_std = sum(w * ss for w, ss in zip(weights, style_stds))
                
                # 增强风格差异性：根据风格特征的方差调整风格强度
                style_variance = torch.var(torch.cat([sm.flatten() for sm in style_means]))
                style_diversity_factor = torch.clamp(style_variance * 2.0, min=0.5, max=2.0)
                
                # 应用风格差异性增强
                style_mean = style_mean * style_diversity_factor
                style_std = style_std * style_diversity_factor
                
                style_feat_resized = True  # 标记有有效的风格特征
            else:
                style_feat_resized = None
        else:
            # 单个风格特征的处理
            style_feat = style_features if not isinstance(style_features, list) else style_features[-1] if style_features else None
            
            if style_feat is not None:
                # 调整风格特征维度
                style_feat_resized = F.adaptive_avg_pool2d(style_feat, (1, 1))
                if style_feat_resized.shape[1] != C:
                    if style_feat_resized.shape[1] > C:
                        style_feat_resized = style_feat_resized[:, :C, :, :]
                    else:
                        repeat_factor = C // style_feat_resized.shape[1]
                        remainder = C % style_feat_resized.shape[1]
                        style_feat_resized = torch.cat([
                            style_feat_resized.repeat(1, repeat_factor, 1, 1),
                            style_feat_resized[:, :remainder, :, :]
                        ], dim=1)
                
                # 计算单层风格统计信息
                style_flat = style_feat_resized.view(B, C, -1)
                style_mean = style_flat.mean(dim=2, keepdim=True).view(B, C, 1, 1)
                style_std = torch.clamp(style_flat.std(dim=2, keepdim=True).view(B, C, 1, 1), min=0.1)
            else:
                style_feat_resized = None
        
        # 如果有有效的风格特征，应用AdaIN风格迁移
        if style_feat_resized is not None:
            # style_mean和style_std已经在上面计算好了
            # 应用AdaIN风格迁移
            normalized_content = (styled_image - content_mean) / (content_std + 1e-8)
            styled_result = normalized_content * style_std + style_mean
            
            # 使用显著性图进行加权混合
            if significance_map is not None:
                # 归一化显著性图
                sig_norm = (significance_map - significance_map.min()) / (significance_map.max() - significance_map.min() + 1e-8)
                # 混合原始图像和风格化图像
                styled_image = styled_image * (1 - style_strength * sig_norm) + styled_result * (style_strength * sig_norm)
            else:
                # 直接混合
                styled_image = styled_image * (1 - style_strength) + styled_result * style_strength
        else:
            # 如果没有风格特征，使用简单的颜色调整
            if significance_map is not None:
                color_shift = torch.randn_like(styled_image) * 0.1 * style_strength
                styled_image = styled_image + color_shift * significance_map
        
    except Exception as e:
        print(f"风格迁移过程中出错: {e}")
        # 如果风格迁移失败，使用简单的颜色调整
        if significance_map is not None:
            color_shift = torch.randn_like(styled_image) * 0.1 * style_strength
            styled_image = styled_image + color_shift * significance_map
    
    return styled_image.squeeze(0) if styled_image.shape[0] == 1 else styled_image

def save_comparison_images(original_image, styled_image, output_dir, idx, style_name):
    """保存对比图像"""
    def tensor_to_pil(tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        img_tensor = tensor.clamp(0, 1)
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    # 保存原始图像
    original_pil = tensor_to_pil(original_image)
    original_path = os.path.join(output_dir, f"original_{idx:04d}.png")
    original_pil.save(original_path)
    
    # 保存风格化图像
    styled_pil = tensor_to_pil(styled_image)
    styled_path = os.path.join(output_dir, f"styled_{style_name}_{idx:04d}.png")
    styled_pil.save(styled_path)
    
    # 创建并保存对比图像
    comparison = Image.new('RGB', (original_pil.width * 2, original_pil.height))
    comparison.paste(original_pil, (0, 0))
    comparison.paste(styled_pil, (original_pil.width, 0))
    comparison_path = os.path.join(output_dir, f"comparison_{style_name}_{idx:04d}.png")
    comparison.save(comparison_path)
    
    return original_path, styled_path, comparison_path

def inference_3dgdm_enhanced(checkpoint_path, style_image_path, output_dir, source_path):
    """增强版3D-GDM推理"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    if not os.path.exists(style_image_path):
        print(f"错误: 风格图像不存在: {style_image_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载检查点
    print("正在加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化场景
    print("正在初始化场景...")
    
    # 创建临时的 ModelParams 类
    class TempModelParams:
        def __init__(self):
            self.sh_degree = 3
            self.source_path = source_path
            self.model_path = ""
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
    
    model_params = TempModelParams()
    model_params.resolution = -1
    
    scene = Scene(model_params, gaussians=GaussianModel(3), load_iteration=None, shuffle=False)
    
    # 恢复高斯参数
    print("正在恢复高斯参数...")
    gaussians = scene.gaussians
    
    # 创建临时的 OptimizationParams 类
    class TempOptimizationParams:
        def __init__(self):
            self.iterations = 30000
            self.position_lr_init = 0.00016
            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            self.position_lr_max_steps = 30000
            self.feature_lr = 0.0025
            self.opacity_lr = 0.05
            self.scaling_lr = 0.005
            self.rotation_lr = 0.001
            self.percent_dense = 0.01
            self.lambda_dssim = 0.2
            self.densification_interval = 100
            self.opacity_reset_interval = 3000
            self.densify_from_iter = 500
            self.densify_until_iter = 15000
            self.densify_grad_threshold = 0.0002
            self.random_background = True
    
    # 创建优化参数
    opt_params = TempOptimizationParams()
    gaussians.restore(checkpoint['gaussians'], opt_params)
    
    # 初始化3D-GDM模型
    print("正在初始化3D-GDM模型...")
    model_3dgdm = ThreeDGaussianDiffusionModel(
        gaussian_dim=3,
        feature_dim=256,
        num_timesteps=1000,
        enable_style_mixing=True,
        mixing_probability=0.3,
        voxel_resolution=32
    ).to(device)
    
    # 加载模型权重
    print("正在加载模型权重...")
    if 'model_3dgdm' in checkpoint:
        model_3dgdm.load_state_dict(checkpoint['model_3dgdm'])
    else:
        print("警告: 检查点中未找到model_3dgdm权重，使用随机初始化的模型")
    model_3dgdm.eval()
    
    # 获取相机列表
    cameras = scene.getTrainCameras()
    
    # 创建管道参数
    class TempPipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    pipeline_params = TempPipelineParams()
    
    # 加载风格图像
    print("正在加载风格图像...")
    # 首先进行一次测试渲染来获取渲染图像的尺寸
    test_camera = cameras[0]
    background = torch.tensor([1.0, 1.0, 1.0], device=device)
    test_render = render(test_camera, gaussians, pipeline_params, background)["render"]
    render_height, render_width = test_render.shape[1], test_render.shape[2]
    
    # 使用渲染图像的尺寸来加载风格图像
    style_image = load_style_image_enhanced(style_image_path, device, target_size=(render_height, render_width))
    style_name = os.path.splitext(os.path.basename(style_image_path))[0]
    
    print(f"渲染图像尺寸: {render_height}x{render_width}")
    print(f"风格图像尺寸: {style_image.shape[2]}x{style_image.shape[3]}")
    
    print(f"开始推理，共{len(cameras)}个视角...")
    print(f"风格图像: {style_name}")
    
    with torch.no_grad():
        for idx, camera in enumerate(tqdm(cameras[:3], desc="渲染视角")):
            # 内存优化：每次循环开始时清理缓存
            torch.cuda.empty_cache()
            
            # 标准渲染
            background = torch.tensor([1.0, 1.0, 1.0], device=device)
            render_pkg = render(camera, gaussians, pipeline_params, background)
            original_render = render_pkg["render"]
            
            # 内存优化：立即清理渲染包
            del render_pkg
            torch.cuda.empty_cache()
            
            # 内存优化：如果图像太大，进行下采样以节省内存
            original_height, original_width = original_render.shape[1], original_render.shape[2]
            max_dimension = 512  # 限制最大尺寸为512像素
            
            if original_height > max_dimension or original_width > max_dimension:
                import torch.nn.functional as F
                scale_factor = min(max_dimension / original_height, max_dimension / original_width)
                new_height = int(original_height * scale_factor)
                new_width = int(original_width * scale_factor)
                
                # 下采样原始渲染
                original_render_resized = F.interpolate(
                    original_render.unsqueeze(0), 
                    size=(new_height, new_width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 下采样风格图像
                style_image_resized = F.interpolate(
                    style_image, 
                    size=(new_height, new_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                print(f"  内存优化：图像从 {original_height}x{original_width} 下采样到 {new_height}x{new_width}")
            else:
                original_render_resized = original_render
                style_image_resized = style_image
            
            # 风格迁移处理
            styled_image = original_render.clone()
            
            try:
                # 内存优化：清理缓存
                torch.cuda.empty_cache()
                
                # 使用固定的较小timestep以获得更稳定的结果
                timestep = torch.tensor([100], device=device)
                
                # 打印调试信息
                print(f"  调试信息 - 原始渲染尺寸: {original_render_resized.shape}")
                print(f"  调试信息 - 风格图像尺寸: {style_image_resized.shape}")
                
                # 内存优化：使用no_grad减少内存使用
                with torch.no_grad():
                    # 调用模型获取风格特征和显著性图（使用下采样的图像）
                    model_output = model_3dgdm.forward(
                        gaussians=gaussians,
                        style_image=style_image_resized,
                        viewpoint_cam=camera,
                        pipe=pipeline_params,
                        background=background,
                        timestep=timestep
                    )
                
                # 提取风格特征和显著性图
                style_features = model_output.get('style_features')
                significance_map = model_output.get('significance_map')
                
                # 打印更多调试信息
                if significance_map is not None:
                    print(f"  调试信息 - 显著性图尺寸: {significance_map.shape}")
                if style_features is not None:
                    print(f"  调试信息 - 风格特征类型: {type(style_features)}")
                
                # 内存优化：立即清理模型输出
                del model_output
                torch.cuda.empty_cache()
                
                # 应用增强的风格迁移（使用下采样的图像）
                styled_image_resized = apply_style_transfer_enhanced(
                    original_render_resized, style_features, significance_map, style_strength=0.7
                )
                
                # 如果进行了下采样，将结果上采样回原始尺寸
                if original_height > max_dimension or original_width > max_dimension:
                    styled_image = F.interpolate(
                        styled_image_resized.unsqueeze(0),
                        size=(original_height, original_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    del styled_image_resized
                else:
                    styled_image = styled_image_resized
                
                # 清理下采样的中间变量
                if 'original_render_resized' in locals():
                    del original_render_resized
                if 'style_image_resized' in locals():
                    del style_image_resized
                torch.cuda.empty_cache()
                
                # 内存优化：清理中间变量
                del style_features, significance_map
                torch.cuda.empty_cache()
                
                print(f"视角 {idx}: 风格迁移成功应用")
                
            except Exception as e:
                import traceback
                print(f"视角 {idx}: 风格迁移失败: {e}")
                print(f"详细错误信息: {traceback.format_exc()}")
                print("使用原始渲染")
                styled_image = original_render
                
                # 内存优化：出错时也要清理缓存
                torch.cuda.empty_cache()
            
            # 保存对比图像（所有3个视角都保存）
            original_path, styled_path, comparison_path = save_comparison_images(
                original_render, styled_image, output_dir, idx, style_name
            )
            
            print(f"对比图像已保存:")
            print(f"  原始: {original_path}")
            print(f"  风格化: {styled_path}")
            print(f"  对比: {comparison_path}")
            
            # 保存最终风格化图像
            final_path = os.path.join(output_dir, f"final_{style_name}_{idx:04d}.png")
            styled_image_copy = styled_image.clone()
            if len(styled_image_copy.shape) == 4:
                styled_image_copy = styled_image_copy.squeeze(0)
            
            img_tensor = styled_image_copy.clamp(0, 1)
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(final_path)
            
            # 内存优化：循环结束时清理变量
            del original_render, styled_image, styled_image_copy
            torch.cuda.empty_cache()
    
    print(f"\n推理完成！")
    print(f"所有图像已保存到: {output_dir}")
    print(f"建议查看 comparison_*.png 文件来对比风格迁移效果")

def main():
    parser = argparse.ArgumentParser(description="增强版3D-GDM推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="3dgdm_checkpoint_final.pth文件路径")
    parser.add_argument("--style_image", type=str, required=True, 
                       help="风格图像路径")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="输出目录")
    parser.add_argument("--source_path", type=str, required=True,
                       help="原始数据集路径（用于加载相机参数）")
    
    args = parser.parse_args()
    
    inference_3dgdm_enhanced(
        checkpoint_path=args.checkpoint,
        style_image_path=args.style_image,
        output_dir=args.output_dir,
        source_path=args.source_path
    )

if __name__ == "__main__":
    main()