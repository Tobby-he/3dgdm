#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Gaussian Diffusion Model (3D-GDM) Training Script
统一的扩散化训练，实现从噪声直接生成风格化3D高斯表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import uuid
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from random import randint
import random
import glob
from pathlib import Path

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, cal_adain_style_loss, cal_mse_content_loss
from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion
from diffusion.enhanced_significance_attention import EnhancedSignificanceAttention
from diffusion.adaptive_style_feature_field import AdaptiveStyleFeatureField
from scene.VGG import VGGEncoder, normalize_vgg

import torchvision
from PIL import Image
import torchvision.transforms as T


class StyleDataset:
    """风格图像数据集类"""
    
    def __init__(self, style_paths, transform=None):
        self.style_paths = style_paths
        self.transform = transform
        self.styles_cache = {}  # 缓存加载的风格图像
        
    def __len__(self):
        return len(self.style_paths)
    
    def __getitem__(self, idx):
        if idx in self.styles_cache:
            return self.styles_cache[idx]
            
        style_path = self.style_paths[idx]
        style_image = Image.open(style_path).convert('RGB')
        
        if self.transform:
            style_image = self.transform(style_image)
            
        # 缓存处理后的图像
        self.styles_cache[idx] = style_image
        return style_image
    
    def get_random_style(self):
        """随机获取一个风格图像"""
        idx = random.randint(0, len(self.style_paths) - 1)
        style_image = self[idx]
        style_path = self.style_paths[idx]
        # style_image已经是[C,H,W]格式，添加batch维度变成[1,C,H,W]
        return style_image.cuda().unsqueeze(0)[:, :3, :, :], Path(style_path).stem
    
    def get_mixed_style(self, alpha=None):
        """获取混合风格图像"""
        if len(self.style_paths) < 2:
            return self.get_random_style()
            
        idx1, idx2 = random.sample(range(len(self.style_paths)), 2)
        style1, style2 = self[idx1], self[idx2]
        
        if alpha is None:
            alpha = random.random()
            
        mixed_style = alpha * style1 + (1 - alpha) * style2
        mixed_path = f"mixed_{Path(self.style_paths[idx1]).stem}_{Path(self.style_paths[idx2]).stem}_alpha{alpha:.2f}"
        
        # 添加batch维度并确保只取前3个通道
        return mixed_style.cuda().unsqueeze(0)[:, :3, :, :], mixed_path


class ThreeDGaussianDiffusionModel(nn.Module):
    """3D高斯扩散模型主类"""
    
    def __init__(self, gaussian_dim=3, feature_dim=256, num_timesteps=1000, 
                 enable_style_mixing=True, mixing_probability=0.3, voxel_resolution=32):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.coarse_steps = num_timesteps // 4  # 粗糙扩散步数
        self.fine_steps = num_timesteps * 3 // 4  # 精细扩散步数
        self.enable_style_mixing = enable_style_mixing
        self.mixing_probability = mixing_probability
        self.voxel_resolution = voxel_resolution
        
        # 核心模块
        self.unified_diffusion = UnifiedGaussianDiffusion(
            coarse_steps=self.coarse_steps,
            fine_steps=self.fine_steps,
            voxel_resolution=voxel_resolution
        )
        self.significance_attention = EnhancedSignificanceAttention()
        self.style_feature_field = AdaptiveStyleFeatureField()
        
        # 注意：梯度检查点将在forward方法中手动应用
        self.vgg_encoder = VGGEncoder().cuda()
        
        # 损失权重
        self.content_weight = 1.0
        self.style_weight = 10.0
        self.diffusion_weight = 1.0
        
        # 风格多样性损失权重
        self.diversity_weight = 0.1
        
    def forward(self, gaussians, style_image, viewpoint_cam, pipe, background, timestep):
        """前向传播"""
        # 渲染当前高斯
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_image = render_pkg["render"]
        
        # 计算显著性注意力
        significance_map = self.significance_attention(rendered_image, style_image)
        
        # 获取自适应风格特征
        style_features, global_intensity = self.style_feature_field(style_image, timestep)
        
        return {
            'rendered_image': rendered_image,
            'significance_map': significance_map,
            'style_features': style_features,
            'global_intensity': global_intensity,
            'render_pkg': render_pkg
        }
    
    def compute_loss(self, outputs, gt_image, style_image, timestep):
        """计算综合损失"""
        rendered_image = outputs['rendered_image']
        significance_map = outputs['significance_map']
        style_features = outputs['style_features']
        
        # 内容损失
        content_loss = cal_mse_content_loss(rendered_image, gt_image)
        
        # 风格损失
        # 确保两个图像都有批次维度
        rendered_input = rendered_image.unsqueeze(0) if len(rendered_image.shape) == 3 else rendered_image
        style_input = style_image.unsqueeze(0) if len(style_image.shape) == 3 else style_image
        
        rendered_features = self.vgg_encoder(normalize_vgg(rendered_input))
        style_image_features = self.vgg_encoder(normalize_vgg(style_input))
        style_loss = cal_adain_style_loss(rendered_features.relu3_1, style_image_features.relu3_1)
        
        # 扩散损失（基于显著性加权）
        diffusion_loss = self.unified_diffusion.compute_diffusion_loss(
            rendered_image, style_features, significance_map, timestep
        )
        
        # 总损失
        total_loss = (
            self.content_weight * content_loss +
            self.style_weight * style_loss +
            self.diffusion_weight * diffusion_loss
        )
        
        return {
            'total_loss': total_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
            'diffusion_loss': diffusion_loss
        }


def prepare_output_and_logger(args):
    """准备输出目录和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def load_style_dataset(style_path):
    """加载风格数据集"""
    style_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    
    if os.path.isfile(style_path):
        # 单个风格图像
        style_paths = [style_path]
        print(f"使用单个风格图像: {style_path}")
    elif os.path.isdir(style_path):
        # 风格图像文件夹
        style_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        style_paths = []
        for ext in style_extensions:
            style_paths.extend(glob.glob(os.path.join(style_path, ext)))
            style_paths.extend(glob.glob(os.path.join(style_path, ext.upper())))
        
        if not style_paths:
            raise ValueError(f"在文件夹 {style_path} 中未找到风格图像")
        
        print(f"加载了 {len(style_paths)} 个风格图像从文件夹: {style_path}")
        for i, path in enumerate(style_paths[:5]):  # 显示前5个
            print(f"  {i+1}. {Path(path).name}")
        if len(style_paths) > 5:
            print(f"  ... 还有 {len(style_paths)-5} 个图像")
    else:
        raise ValueError(f"风格路径无效: {style_path}")
    
    return StyleDataset(style_paths, style_transform)


def get_training_style(style_dataset, current_iter, total_iters,
                      enable_multi_style=True, mixing_prob=0.3, curriculum_learning=True):
    """智能风格选择策略"""
    
    # 如果只有一个风格或禁用多风格训练
    if len(style_dataset) == 1 or not enable_multi_style:
        style_image, style_path = style_dataset.get_random_style()
        return style_image, style_path
    
    # 课程学习：训练初期使用简单策略，后期使用复杂策略
    if curriculum_learning:
        progress = current_iter / total_iters
        
        # 前30%迭代：只使用单风格
        if progress < 0.3:
            style_image, style_path = style_dataset.get_random_style()
            return style_image, style_path
        
        # 30%-70%迭代：逐渐引入风格混合
        elif progress < 0.7:
            adjusted_mixing_prob = mixing_prob * (progress - 0.3) / 0.4
            if random.random() < adjusted_mixing_prob:
                mixed_style, mixed_name = style_dataset.get_mixed_style()
                return mixed_style, mixed_name
            else:
                style_image, style_path = style_dataset.get_random_style()
                return style_image, style_path
        
        # 后30%迭代：完全随机策略
        else:
            if random.random() < mixing_prob:
                mixed_style, mixed_name = style_dataset.get_mixed_style()
                return mixed_style, mixed_name
            else:
                style_image, style_path = style_dataset.get_random_style()
                return style_image, style_path
    
    # 非课程学习：直接使用随机策略
    else:
        if random.random() < mixing_prob:
            mixed_style, mixed_name = style_dataset.get_mixed_style()
            return mixed_style, mixed_name
        else:
            style_image, style_path = style_dataset.get_random_style()
            return style_image, style_path


def training_3dgdm(dataset, opt, pipe, testing_iterations, saving_iterations,
                   checkpoint_iterations, checkpoint, debug_from, style_image_path,
                   enable_multi_style=True, style_mixing_prob=0.3, curriculum_learning=True,
                   voxel_resolution=32, enable_memory_monitor=False, gradient_checkpointing=False):
    """3D-GDM统一训练函数"""
    
    # 加载风格数据集
    style_dataset = load_style_dataset(style_image_path)
    print(f"\n=== 风格训练配置 ===")
    print(f"风格图像数量: {len(style_dataset)}")
    print(f"多风格训练: {'启用' if enable_multi_style and len(style_dataset) > 1 else '禁用'}")
    print(f"风格混合概率: {style_mixing_prob if enable_multi_style else 0.0}")
    print(f"课程学习: {'启用' if curriculum_learning else '禁用'}")
    print(f"==================\n")
    
    # 初始化模型
    model_3dgdm = ThreeDGaussianDiffusionModel(
        voxel_resolution=voxel_resolution
    ).cuda()
    
    # 内存监控
    if enable_memory_monitor:
        print(f"GPU内存监控已启用，体素分辨率: {voxel_resolution}")
        torch.cuda.empty_cache()
        print(f"初始GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # 梯度检查点
    if gradient_checkpointing:
        print("梯度检查点已启用")
        torch.utils.checkpoint.checkpoint_sequential = True
    scene = Scene(dataset, GaussianModel(dataset.sh_degree))
    gaussians = scene.gaussians
    
    # 设置高斯模型的训练参数
    gaussians.training_setup_reconstruction(opt)
    
    # 获取高斯模型的参数组
    gaussian_params = []
    for group in gaussians.optimizer.param_groups:
        gaussian_params.extend(group['params'])
    
    # 设置优化器和混合精度scaler
    optimizer = torch.optim.Adam([
        {'params': gaussian_params, 'lr': opt.position_lr_init},
        {'params': model_3dgdm.parameters(), 'lr': opt.position_lr_init * 0.1}
    ])
    
    # 混合精度训练scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 加载检查点
    first_iter = 0
    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        gaussians.restore(checkpoint_data['gaussians'], opt)
        model_3dgdm.load_state_dict(checkpoint_data['model_3dgdm'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        first_iter = checkpoint_data['iteration']
    
    # 准备训练
    tb_writer = prepare_output_and_logger(dataset)
    viewpoint_cameras = scene.getTrainCameras(1.0)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    
    if not viewpoint_cameras:
        raise ValueError("No cameras found in the dataset.")
    
    # 训练循环
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="3D-GDM Training")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    style_stats = {'single': 0, 'mixed': 0, 'total_styles_used': set()}
    
    for iteration in range(first_iter + 1, opt.iterations + 1):
        # 选择随机相机
        if not viewpoint_stack:
            viewpoint_stack = viewpoint_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 随机时间步（用于扩散训练）
        timestep = torch.randint(0, model_3dgdm.num_timesteps, (1,)).cuda()
        
        # 智能风格选择策略
        current_style, style_name = get_training_style(
            style_dataset, iteration, opt.iterations, 
            enable_multi_style, style_mixing_prob, curriculum_learning
        )
        
        # 统计风格使用情况
        if 'mixed' in style_name:
            style_stats['mixed'] += 1
        else:
            style_stats['single'] += 1
            style_stats['total_styles_used'].add(style_name)
        
        # 内存监控
        if enable_memory_monitor and iteration % 100 == 0:
            current_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[Iter {iteration}] GPU内存: {current_memory:.2f}GB / 峰值: {max_memory:.2f}GB")
            
            # 内存清理
            if current_memory > 3.5:  # 接近4GB限制时清理
                torch.cuda.empty_cache()
                print(f"[Iter {iteration}] 执行内存清理")
        
        # 使用混合精度训练和梯度累积来优化内存
        with torch.cuda.amp.autocast():
            # 前向传播
            outputs = model_3dgdm(gaussians, current_style, viewpoint_cam, pipe, background, timestep)
            
            # 计算损失
            gt_image = viewpoint_cam.original_image.cuda()
            losses = model_3dgdm.compute_loss(outputs, gt_image, current_style, timestep)
            
            # 梯度累积（每8步更新一次以节省内存）
            loss = losses['total_loss'] / 8
        
        # 反向传播
        if iteration % 8 == 0:
            optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        
        if (iteration + 1) % 8 == 0:
            scaler.step(optimizer)
            scaler.update()
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        # 更新进度
        with torch.no_grad():
            # 使用原始损失进行记录
            original_total_loss = losses['total_loss'].item()
            ema_loss_for_log = 0.4 * original_total_loss + 0.6 * ema_loss_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Total Loss": f"{ema_loss_for_log:.{7}f}",
                    "Content": f"{losses['content_loss'].item():.{4}f}",
                    "Style": f"{losses['style_loss'].item():.{4}f}",
                    "Diffusion": f"{losses['diffusion_loss'].item():.{4}f}"
                })
                progress_bar.update(10)
            
            # 记录日志
            if iteration % 100 == 0:
                tb_writer.add_scalar('train_loss/total_loss', losses['total_loss'].item(), iteration)
                tb_writer.add_scalar('train_loss/content_loss', losses['content_loss'].item(), iteration)
                tb_writer.add_scalar('train_loss/style_loss', losses['style_loss'].item(), iteration)
                tb_writer.add_scalar('train_loss/diffusion_loss', losses['diffusion_loss'].item(), iteration)
                
                # 记录风格统计
                tb_writer.add_scalar('style_stats/single_style_ratio', 
                                   style_stats['single'] / (style_stats['single'] + style_stats['mixed'] + 1e-8), iteration)
                tb_writer.add_scalar('style_stats/mixed_style_ratio', 
                                   style_stats['mixed'] / (style_stats['single'] + style_stats['mixed'] + 1e-8), iteration)
                tb_writer.add_scalar('style_stats/unique_styles_used', len(style_stats['total_styles_used']), iteration)
            
            # 保存检查点
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving 3D-GDM checkpoint")
                checkpoint_data = {
                    'gaussians': gaussians.capture(),
                    'model_3dgdm': model_3dgdm.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration
                }
                torch.save(checkpoint_data, os.path.join(dataset.model_path, f"3dgdm_checkpoint_{iteration}.pth"))
    
    progress_bar.close()
    
    # 保存最终模型和点云
    print(f"\n[FINAL] 保存最终模型...")
    
    # 保存最终检查点
    final_checkpoint_data = {
        'gaussians': gaussians.capture(),
        'model_3dgdm': model_3dgdm.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': opt.iterations
    }
    torch.save(final_checkpoint_data, os.path.join(dataset.model_path, "3dgdm_checkpoint_final.pth"))
    print(f"最终检查点已保存: {os.path.join(dataset.model_path, '3dgdm_checkpoint_final.pth')}")
    
    # 保存点云
    scene.save(opt.iterations)
    print(f"点云已保存: {os.path.join(dataset.model_path, f'point_cloud/iteration_{opt.iterations}/point_cloud.ply')}")
    
    # 打印训练统计
    total_iterations = style_stats['single'] + style_stats['mixed']
    print(f"\n=== 3D-GDM 训练完成! ===")
    print(f"总迭代次数: {total_iterations}")
    print(f"单风格训练: {style_stats['single']} ({style_stats['single']/total_iterations*100:.1f}%)")
    print(f"混合风格训练: {style_stats['mixed']} ({style_stats['mixed']/total_iterations*100:.1f}%)")
    print(f"使用的独特风格数: {len(style_stats['total_styles_used'])}/{len(style_dataset)}")
    print(f"风格覆盖率: {len(style_stats['total_styles_used'])/len(style_dataset)*100:.1f}%")
    print(f"模型保存路径: {dataset.model_path}")
    print(f"========================\n")


if __name__ == "__main__":
    import argparse
    from arguments import ModelParams, PipelineParams, OptimizationParams
    
    parser = argparse.ArgumentParser(description="3D-GDM Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # 3D-GDM特定参数
    parser.add_argument("--style_image", type=str, required=True, help="Path to style image or style folder")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--coarse_ratio", type=float, default=0.7, help="Ratio of coarse diffusion steps")
    parser.add_argument("--style_weight", type=float, default=10.0, help="Style loss weight")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content loss weight")
    parser.add_argument("--diffusion_weight", type=float, default=1.0, help="Diffusion loss weight")
    parser.add_argument("--significance_weight", type=float, default=0.5, help="Significance attention weight")
    
    # 多风格训练参数
    parser.add_argument("--enable_multi_style", action="store_true", help="Enable multi-style training")
    parser.add_argument("--style_mixing_prob", type=float, default=0.3, help="Probability of style mixing")
    parser.add_argument("--curriculum_learning", action="store_true", help="Enable curriculum learning")
    
    # 内存优化参数
    parser.add_argument("--voxel_resolution", type=int, default=32, help="Voxel resolution for memory optimization (default: 32)")
    parser.add_argument("--enable_memory_monitor", action="store_true", help="Enable GPU memory monitoring")
    parser.add_argument("--memory_threshold", type=float, default=0.8, help="Memory usage threshold for warnings (default: 0.8)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    
    # 训练控制参数
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000], help="Test iterations")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000], help="Save iterations")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000], help="Checkpoint iterations")
    parser.add_argument("--start_checkpoint", type=str, default=None, help="Start from checkpoint")
    parser.add_argument("--debug_from", type=int, default=-1, help="Debug from iteration")
    
    args = parser.parse_args()
    
    print("=== 3D-GDM 统一扩散训练 ===")
    print(f"数据路径: {args.source_path}")
    print(f"模型路径: {args.model_path}")
    print(f"风格路径: {args.style_image}")
    print(f"训练迭代: {args.iterations}")
    print(f"扩散步数: {args.diffusion_steps}")
    print(f"多风格训练: {'启用' if args.enable_multi_style else '禁用'}")
    print(f"课程学习: {'启用' if args.curriculum_learning else '禁用'}")
    print(f"风格混合概率: {args.style_mixing_prob}")
    print(f"体素分辨率: {args.voxel_resolution}")
    print(f"内存监控: {'启用' if args.enable_memory_monitor else '禁用'}")
    print(f"梯度检查点: {'启用' if args.gradient_checkpointing else '禁用'}")
    print("========================\n")
    
    # 初始化参数
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    
    # 开始训练
    training_3dgdm(
        dataset, opt, pipe, 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.style_image,
        args.enable_multi_style, args.style_mixing_prob, args.curriculum_learning,
        args.voxel_resolution, args.enable_memory_monitor, args.gradient_checkpointing
    )