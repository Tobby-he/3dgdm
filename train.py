#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StyleGaussian训练主程序
支持传统分阶段训练和新的3D-GDM统一扩散训练
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_stage(script_name, args):
    """运行训练阶段"""
    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in {script_name}:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description="StyleGaussian Training Pipeline")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save models")
    parser.add_argument("--style_image", type=str, required=True, help="Path to style image")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000], help="Test iterations")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000], help="Save iterations")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000], help="Checkpoint iterations")
    
    # 新增3D-GDM相关参数
    parser.add_argument("--use_3dgdm", action="store_true", help="Use 3D-GDM unified diffusion training")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--coarse_ratio", type=float, default=0.7, help="Ratio of coarse diffusion steps")
    parser.add_argument("--style_weight", type=float, default=1.0, help="Style loss weight")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content loss weight")
    parser.add_argument("--diffusion_weight", type=float, default=1.0, help="Diffusion loss weight")
    parser.add_argument("--significance_weight", type=float, default=0.5, help="Significance attention weight")
    
    # 多风格训练参数
    parser.add_argument("--enable_multi_style", action="store_true", default=True, help="Enable multi-style training")
    parser.add_argument("--style_mixing_prob", type=float, default=0.3, help="Probability of style mixing")
    parser.add_argument("--curriculum_learning", action="store_true", default=True, help="Enable curriculum learning for style training")
    parser.add_argument("--disable_multi_style", action="store_true", help="Disable multi-style training (force single style)")
    parser.add_argument("--disable_curriculum", action="store_true", help="Disable curriculum learning")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    
    # 构建通用参数
    common_args = [
        "--source_path", args.source_path,
        "--model_path", args.model_path,
        "--iterations", str(args.iterations),
        "--test_iterations"] + [str(x) for x in args.test_iterations] + [
        "--save_iterations"] + [str(x) for x in args.save_iterations] + [
        "--checkpoint_iterations"] + [str(x) for x in args.checkpoint_iterations]
    
    print("=== StyleGaussian Training Pipeline ===")
    
    if args.use_3dgdm:
        # 使用新的3D-GDM统一训练
        print("\n--- Using 3D-GDM Unified Diffusion Training ---")
        
        # 处理多风格训练参数
        enable_multi_style = args.enable_multi_style and not args.disable_multi_style
        curriculum_learning = args.curriculum_learning and not args.disable_curriculum
        
        # 构建3D-GDM参数
        gdm_args = common_args + [
            "--style_image", args.style_image,
            "--diffusion_steps", str(args.diffusion_steps),
            "--coarse_ratio", str(args.coarse_ratio),
            "--style_weight", str(args.style_weight),
            "--content_weight", str(args.content_weight),
            "--diffusion_weight", str(args.diffusion_weight),
            "--significance_weight", str(args.significance_weight),
            "--style_mixing_prob", str(args.style_mixing_prob)
        ]
        
        # 添加布尔参数
        if enable_multi_style:
            gdm_args.append("--enable_multi_style")
        if curriculum_learning:
            gdm_args.append("--curriculum_learning")
        
        # 打印训练配置
        print(f"多风格训练: {'启用' if enable_multi_style else '禁用'}")
        print(f"课程学习: {'启用' if curriculum_learning else '禁用'}")
        print(f"风格混合概率: {args.style_mixing_prob}")
        
        if not run_stage("train_3dgdm.py", gdm_args):
            print("3D-GDM training failed!")
            return
    else:
        # 使用传统分阶段训练
        print("\n--- Using Traditional Multi-Stage Training ---")
        
        # 阶段1: 重建训练
        print("\n--- Stage 1: Reconstruction Training ---")
        if not run_stage("train_reconstruction.py", common_args):
            print("Reconstruction training failed!")
            return
        
        # 阶段2: 特征嵌入训练
        print("\n--- Stage 2: Feature Embedding Training ---")
        if not run_stage("train_feature.py", common_args):
            print("Feature training failed!")
            return
        
        # 阶段3: 风格迁移训练
        print("\n--- Stage 3: Artistic Style Transfer Training ---")
        artistic_args = common_args + ["--style_image", args.style_image]
        if not run_stage("train_artistic.py", artistic_args):
            print("Artistic training failed!")
            return
    
    print("\n=== Training Pipeline Completed Successfully! ===")

if __name__ == "__main__":
    main()


