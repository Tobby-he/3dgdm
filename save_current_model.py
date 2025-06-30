#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动保存当前训练状态的脚本
如果训练进程还在运行，可以使用此脚本保存模型
"""

import torch
import os
import sys
from scene import Scene, GaussianModel
from diffusion.unified_gaussian_diffusion import UnifiedGaussianDiffusion
from arguments import ModelParams, PipelineParams, OptimizationParams

def save_current_model(model_path, gaussians, model_3dgdm, optimizer, iteration):
    """
    保存当前模型状态
    """
    print(f"开始保存模型到: {model_path}")
    
    # 确保目录存在
    os.makedirs(model_path, exist_ok=True)
    
    # 保存检查点
    checkpoint_data = {
        'gaussians': gaussians.capture(),
        'model_3dgdm': model_3dgdm.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    
    checkpoint_path = os.path.join(model_path, f"3dgdm_checkpoint_{iteration}.pth")
    torch.save(checkpoint_data, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终检查点
    final_checkpoint_path = os.path.join(model_path, "3dgdm_checkpoint_final.pth")
    torch.save(checkpoint_data, final_checkpoint_path)
    print(f"最终检查点已保存: {final_checkpoint_path}")
    
    return checkpoint_path, final_checkpoint_path

def save_point_cloud(scene, iteration):
    """
    保存点云数据
    """
    print(f"保存点云数据 (迭代 {iteration})...")
    scene.save(iteration)
    point_cloud_path = os.path.join(scene.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
    print(f"点云已保存: {point_cloud_path}")
    return point_cloud_path

if __name__ == "__main__":
    print("=== 手动模型保存脚本 ===")
    print("此脚本用于在训练进程仍在运行时手动保存模型")
    print("请确保训练进程中的变量仍在内存中")
    print("")
    
    # 示例用法说明
    print("使用方法:")
    print("1. 在训练脚本的Python环境中运行:")
    print("   exec(open('save_current_model.py').read())")
    print("")
    print("2. 然后调用保存函数:")
    print("   save_current_model(dataset.model_path, gaussians, model_3dgdm, optimizer, iteration)")
    print("   save_point_cloud(scene, iteration)")
    print("")
    print("3. 或者如果变量名不同，请相应调整")
    print("========================")