#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-GDM 推理脚本
用于加载训练好的3D-GDM模型并生成风格化的3D场景
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from PIL import Image
import cv2
from pathlib import Path
import json
from tqdm import tqdm

# 导入必要模块
try:
    from scene import Scene, GaussianModel
    from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
    from gaussian_renderer import render
    from diffusion.train_3dgdm import ThreeDGaussianDiffusionModel
    from diffusion.enhanced_significance_attention import EnhancedSignificanceAttention
    from diffusion.adaptive_style_feature_field import AdaptiveStyleFeatureField
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在StyleGaussian项目根目录下运行此脚本")
    exit(1)

class GDMInference:
    """3D-GDM推理类"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.gaussians = None
        self.scene = None
        self.diffusion_model = None
        self.dataset = None
        
    def load_model(self, iteration=None):
        """加载训练好的模型"""
        print("正在加载3D-GDM模型...")
        
        # 加载数据集配置
        from arguments import ModelParams
        self.dataset = ModelParams()
        self.dataset.model_path = self.model_path
        
        # 尝试从cfg_args文件加载配置
        cfg_args_path = os.path.join(self.model_path, "cfg_args")
        if os.path.exists(cfg_args_path):
            import argparse
            with open(cfg_args_path, 'r') as f:
                content = f.read()
                # 解析Namespace对象
                if 'white_background=False' in content:
                    self.dataset.white_background = False
                elif 'white_background=True' in content:
                    self.dataset.white_background = True
                else:
                    self.dataset.white_background = False  # 默认值
        else:
            self.dataset.white_background = False  # 默认值
            
        print(f"数据集背景设置: {'白色' if self.dataset.white_background else '黑色'}")
        
        # 加载场景
        self.scene = Scene(self.dataset, load_iteration=iteration)
        self.gaussians = self.scene.gaussians
        
        # 加载3D-GDM相关模型
        model_files = {
            'diffusion_model': 'diffusion_model.pth',
            'significance_attention': 'significance_attention.pth',
            'style_feature_field': 'style_feature_field.pth'
        }
        
        checkpoint_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration or 'final'}")
        
        # 初始化模型
        self.diffusion_model = ThreeDGaussianDiffusionModel(
            gaussians=self.gaussians,
            diffusion_steps=1000,
            coarse_ratio=0.7
        ).to(self.device)
        
        # 尝试加载检查点
        for model_name, filename in model_files.items():
            model_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(model_path):
                print(f"加载 {model_name} 从 {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                getattr(self.diffusion_model, model_name).load_state_dict(checkpoint)
            else:
                print(f"警告: 未找到 {model_name} 检查点，使用随机初始化")
        
        print("模型加载完成")
    
    def generate_stylized_scene(self, style_image_path, output_path, 
                              num_inference_steps=50, guidance_scale=7.5):
        """生成风格化场景"""
        print(f"开始生成风格化场景...")
        
        # 加载风格图像
        style_image = self.load_style_image(style_image_path)
        
        # 设置推理参数
        self.diffusion_model.eval()
        
        with torch.no_grad():
            # 从噪声开始逐步去噪
            print("执行扩散去噪过程...")
            
            # 初始化噪声
            noise_gaussians = self.initialize_noise()
            
            # 去噪循环
            timesteps = torch.linspace(1000, 1, num_inference_steps, dtype=torch.long)
            
            for i, t in enumerate(tqdm(timesteps, desc="去噪步骤")):
                # 预测噪声
                with torch.no_grad():
                    # 渲染当前状态
                    rendered_image = self.render_current_state()
                    
                    # 计算显著性
                    significance_map = self.diffusion_model.significance_attention(
                        rendered_image, style_image, self.gaussians.get_xyz
                    )
                    
                    # 预测噪声
                    predicted_noise = self.diffusion_model.unified_diffusion(
                        noise_gaussians, t.unsqueeze(0), 
                        rendered_image, style_image, significance_map
                    )
                    
                    # 去噪步骤
                    noise_gaussians = self.denoise_step(
                        noise_gaussians, predicted_noise, t
                    )
            
            # 更新高斯参数
            self.update_gaussians_from_noise(noise_gaussians)
        
        print("风格化完成，开始渲染结果...")
        
        # 渲染最终结果
        self.render_final_results(output_path)
        
        print(f"结果已保存到: {output_path}")
    
    def load_style_image(self, style_image_path):
        """加载并预处理风格图像"""
        image = Image.open(style_image_path).convert('RGB')
        image = image.resize((512, 512))
        
        # 转换为tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return image_tensor.to(self.device)
    
    def initialize_noise(self):
        """初始化噪声高斯参数"""
        num_gaussians = self.gaussians.get_xyz.shape[0]
        
        noise_gaussians = {
            'xyz': torch.randn_like(self.gaussians.get_xyz),
            'opacity': torch.randn_like(self.gaussians.get_opacity),
            'scaling': torch.randn_like(self.gaussians.get_scaling),
            'rotation': torch.randn_like(self.gaussians.get_rotation)
        }
        
        return noise_gaussians
    
    def render_current_state(self):
        """渲染当前高斯状态"""
        # 使用第一个训练视角进行渲染
        viewpoint = self.scene.getTrainCameras()[0]
        
        # 根据数据集配置设置背景颜色
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        # 渲染图像
        rendering = render(viewpoint, self.gaussians, background=background)
        rendered_image = rendering["render"]
        
        return rendered_image.unsqueeze(0)  # 添加batch维度
    
    def denoise_step(self, noise_gaussians, predicted_noise, timestep):
        """执行单步去噪"""
        # 简化的DDPM去噪步骤
        alpha_t = self.get_alpha_t(timestep)
        alpha_t_prev = self.get_alpha_t(timestep - 1) if timestep > 1 else torch.tensor(1.0)
        
        # 预测原始高斯参数
        pred_gaussians = {}
        for key in noise_gaussians.keys():
            pred_gaussians[key] = (
                noise_gaussians[key] - torch.sqrt(1 - alpha_t) * predicted_noise[key]
            ) / torch.sqrt(alpha_t)
        
        # 添加噪声（除了最后一步）
        if timestep > 1:
            noise = {key: torch.randn_like(val) for key, val in pred_gaussians.items()}
            beta_t = 1 - alpha_t / alpha_t_prev
            
            for key in pred_gaussians.keys():
                pred_gaussians[key] = (
                    torch.sqrt(alpha_t_prev) * pred_gaussians[key] + 
                    torch.sqrt(beta_t) * noise[key]
                )
        
        return pred_gaussians
    
    def get_alpha_t(self, timestep):
        """获取alpha_t值"""
        # 线性噪声调度
        beta_start, beta_end = 0.0001, 0.02
        betas = torch.linspace(beta_start, beta_end, 1000, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return alphas_cumprod[timestep.long()]
    
    def update_gaussians_from_noise(self, denoised_gaussians):
        """从去噪结果更新高斯参数"""
        self.gaussians._xyz.data = denoised_gaussians['xyz']
        self.gaussians._opacity.data = denoised_gaussians['opacity']
        self.gaussians._scaling.data = denoised_gaussians['scaling']
        self.gaussians._rotation.data = denoised_gaussians['rotation']
    
    def render_final_results(self, output_path):
        """渲染最终结果"""
        os.makedirs(output_path, exist_ok=True)
        
        # 渲染训练视角
        train_cameras = self.scene.getTrainCameras()
        
        print(f"渲染 {len(train_cameras)} 个训练视角...")
        
        for idx, viewpoint in enumerate(tqdm(train_cameras, desc="渲染训练视角")):
            rendering = render(viewpoint, self.gaussians, 
                             background=torch.tensor([1, 1, 1], device=self.device))
            
            # 保存图像
            image = rendering["render"]
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            output_file = os.path.join(output_path, f"train_{idx:04d}.png")
            Image.fromarray(image_np).save(output_file)
        
        # 渲染测试视角
        test_cameras = self.scene.getTestCameras()
        
        if test_cameras:
            print(f"渲染 {len(test_cameras)} 个测试视角...")
            
            for idx, viewpoint in enumerate(tqdm(test_cameras, desc="渲染测试视角")):
                rendering = render(viewpoint, self.gaussians,
                                 background=torch.tensor([1, 1, 1], device=self.device))
                
                # 保存图像
                image = rendering["render"]
                image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                output_file = os.path.join(output_path, f"test_{idx:04d}.png")
                Image.fromarray(image_np).save(output_file)
        
        print("渲染完成")
    
    def create_video(self, image_folder, output_video_path, fps=30):
        """从渲染图像创建视频"""
        print("创建视频...")
        
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        
        if not image_files:
            print("未找到图像文件")
            return
        
        # 读取第一张图像获取尺寸
        first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        height, width, _ = first_image.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 写入所有图像
        for image_file in tqdm(image_files, desc="创建视频"):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)
        
        video_writer.release()
        print(f"视频已保存到: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="3D-GDM推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--style_image", type=str, required=True, help="风格图像路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")
    parser.add_argument("--iteration", type=int, default=None, help="模型迭代次数")
    parser.add_argument("--inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导尺度")
    parser.add_argument("--create_video", action="store_true", help="是否创建视频")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    
    args = parser.parse_args()
    
    # 检查输入
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.style_image):
        print(f"错误: 风格图像不存在: {args.style_image}")
        return
    
    # 创建推理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    inference = GDMInference(args.model_path, device)
    
    try:
        # 加载模型
        inference.load_model(args.iteration)
        
        # 生成风格化场景
        inference.generate_stylized_scene(
            args.style_image,
            args.output_path,
            args.inference_steps,
            args.guidance_scale
        )
        
        # 创建视频（如果需要）
        if args.create_video:
            video_path = os.path.join(args.output_path, "stylized_scene.mp4")
            inference.create_video(args.output_path, video_path, args.fps)
        
        print("推理完成！")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()