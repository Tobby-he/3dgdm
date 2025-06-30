#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的3D-GDM推理脚本
用于加载3dgdm_checkpoint_final.pth并进行推理
"""

import torch
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

# 导入必要模块
try:
    from scene import Scene, GaussianModel
    from utils.camera_utils import cameraList_from_camInfos
    from gaussian_renderer import render
    from train_3dgdm import ThreeDGaussianDiffusionModel
    from arguments import ModelParams, PipelineParams, OptimizationParams
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在StyleGaussian项目根目录下运行此脚本")
    exit(1)

def load_style_image(style_path, device):
    """加载并预处理风格图像"""
    style_image = Image.open(style_path).convert('RGB')
    style_image = style_image.resize((256, 256))
    style_tensor = torch.from_numpy(np.array(style_image)).float() / 255.0
    style_tensor = style_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return style_tensor

def inference_3dgdm(checkpoint_path, style_image_path, output_dir, source_path):
    """使用3dgdm_checkpoint_final.pth进行推理"""
    
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
    
    print("正在加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建临时parser用于初始化参数
    temp_parser = argparse.ArgumentParser()
    
    # 初始化参数
    model_params = ModelParams(temp_parser)
    model_params.source_path = source_path
    model_params.model_path = os.path.dirname(checkpoint_path)
    model_params.images = "images"  # 添加 images 属性
    model_params.eval = False  # 添加 eval 属性
    model_params.white_background = False  # 添加 white_background 属性
    model_params.resolution = -1  # 添加 resolution 属性
    
    pipeline_params = PipelineParams(temp_parser)
    
    # 创建优化参数（用于恢复高斯参数）
    opt_params = OptimizationParams(temp_parser)
    
    # 初始化场景和高斯模型
    print("正在初始化场景...")
    gaussians = GaussianModel(model_params.sh_degree)
    scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
    
    # 恢复高斯参数
    print("正在恢复高斯参数...")
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
    model_3dgdm.load_state_dict(checkpoint['model_3dgdm'])
    model_3dgdm.eval()
    
    # 加载风格图像
    print("正在加载风格图像...")
    style_image = load_style_image(style_image_path, device)
    
    # 创建管道参数
    class TempPipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    pipeline_params = TempPipelineParams()
    
    # 获取相机视角
    cameras = scene.getTrainCameras()
    
    print(f"开始推理，共{len(cameras)}个视角...")
    
    with torch.no_grad():
        for idx, camera in enumerate(tqdm(cameras[:10], desc="渲染视角")):
            # 使用标准渲染进行推理
            background = torch.tensor([1.0, 1.0, 1.0], device=device)
            rendered_image = render(camera, gaussians, pipeline_params, background)["render"]
            
            # 应用风格迁移（简化版本）
            # 注意：这里简化了风格迁移过程，实际应用中可能需要更复杂的处理
            try:
                # 尝试使用模型进行风格迁移
                timestep = torch.randint(0, 1000, (1,), device=device)
                styled_output = model_3dgdm(gaussians, style_image, camera, pipeline_params, background, timestep)
                if hasattr(styled_output, 'get') and styled_output.get('rendered_image') is not None:
                    rendered_image = styled_output['rendered_image']
            except Exception as e:
                print(f"风格迁移失败，使用原始渲染: {e}")
            
            # 保存图像
            if rendered_image is not None:
                # 转换为PIL图像
                img_tensor = rendered_image.clamp(0, 1)
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # 保存
                output_path = os.path.join(output_dir, f"render_{idx:04d}.png")
                img_pil.save(output_path)
                
                if idx == 0:
                    print(f"第一张图像已保存到: {output_path}")
    
    print(f"推理完成！所有图像已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="3D-GDM推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="3dgdm_checkpoint_final.pth文件路径")
    parser.add_argument("--style_image", type=str, required=True, 
                       help="风格图像路径")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="输出目录")
    parser.add_argument("--source_path", type=str, required=True,
                       help="原始数据集路径（用于加载相机参数）")
    
    args = parser.parse_args()
    
    inference_3dgdm(
        checkpoint_path=args.checkpoint,
        style_image_path=args.style_image,
        output_dir=args.output_dir,
        source_path=args.source_path
    )

if __name__ == "__main__":
    main()