#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化的3D-GDM训练脚本
针对4GB GPU内存限制进行优化
"""

import os
import sys
import torch

# 设置PyTorch内存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 检查GPU内存
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"检测到GPU内存: {gpu_memory:.1f}GB")
    
    if gpu_memory < 6.0:
        print("检测到低内存GPU，启用内存优化模式")
        # 设置更保守的内存配置
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 推荐的训练参数
        recommended_args = [
            "--voxel_resolution", "16",  # 降低体素分辨率
            "--enable_memory_monitor",   # 启用内存监控
            "--gradient_checkpointing",  # 启用梯度检查点
            "--iterations", "500",       # 减少迭代次数用于测试
        ]
        
        print("推荐的训练参数:")
        for i in range(0, len(recommended_args), 2):
            if i+1 < len(recommended_args):
                print(f"  {recommended_args[i]} {recommended_args[i+1]}")
            else:
                print(f"  {recommended_args[i]}")
        
        # 添加推荐参数到sys.argv
        if len(sys.argv) == 1:  # 如果没有提供参数
            print("\n使用默认优化参数启动训练...")
            sys.argv.extend([
                "--source_path", "datasets/garden",
                "--model_path", "output/garden/3dgdm_optimized",
                "--style_image", "images",
                "--enable_multi_style",
                "--style_mixing_prob", "0.3",
                "--curriculum_learning",
            ])
            sys.argv.extend(recommended_args)

# 导入并运行原始训练脚本
if __name__ == "__main__":
    print("=== 内存优化3D-GDM训练 ===")
    print("针对4GB GPU内存进行优化")
    print("==============================\n")
    
    # 导入原始训练模块并执行
    import train_3dgdm
    
    # 直接执行原始训练脚本的主函数
    # 这样可以确保所有参数都被正确传递
    with open('train_3dgdm.py', 'r', encoding='utf-8') as f:
        exec(f.read())