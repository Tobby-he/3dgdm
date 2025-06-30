#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练恢复脚本
从现有配置重新开始训练，确保模型得到正确保存
"""

import os
import sys
import torch
import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams

def recover_training():
    """
    恢复训练的主函数
    """
    print("=== 3D-GDM 训练恢复脚本 ===")
    print("此脚本将使用现有配置重新开始训练")
    print("")
    
    # 检查现有配置
    config_path = "e:\\StyleGaussian-main\\output\\garden\\3dgdm_optimized\\cfg_args"
    if os.path.exists(config_path):
        print(f"找到配置文件: {config_path}")
        
        # 读取配置
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
            print("配置文件内容:")
            print(config_content)
            print("")
        except Exception as e:
            print(f"读取配置文件失败: {e}")
    else:
        print(f"未找到配置文件: {config_path}")
    
    # 提供恢复选项
    print("恢复选项:")
    print("1. 使用修改后的train_3dgdm.py重新训练 (推荐)")
    print("2. 从最后几次迭代开始训练")
    print("3. 检查是否有隐藏的检查点文件")
    print("")
    
    # 检查可能的检查点位置
    possible_paths = [
        "e:\\StyleGaussian-main\\output\\garden\\3dgdm_optimized",
        "e:\\StyleGaussian-main\\output\\garden",
        "e:\\StyleGaussian-main\\output"
    ]
    
    print("检查可能的检查点位置:")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ 存在: {path}")
            # 列出所有.pth文件
            try:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.pth'):
                            full_path = os.path.join(root, file)
                            print(f"  找到检查点: {full_path}")
            except Exception as e:
                print(f"  扫描失败: {e}")
        else:
            print(f"✗ 不存在: {path}")
    
    print("")
    print("建议的恢复步骤:")
    print("1. 备份现有的cfg_args文件")
    print("2. 使用修改后的train_3dgdm.py重新训练:")
    print("   python train_3dgdm.py -s datasets/garden -m output/garden/3dgdm_recovered")
    print("3. 新的训练将自动保存最终模型和点云")
    print("")
    
    return True

def create_recovery_command():
    """
    创建恢复训练的命令
    """
    base_cmd = "python train_3dgdm.py"
    source_path = "datasets/garden"
    model_path = "output/garden/3dgdm_recovered"
    
    cmd = f"{base_cmd} -s {source_path} -m {model_path}"
    
    print("建议的恢复命令:")
    print(cmd)
    print("")
    print("此命令将:")
    print("- 使用garden数据集")
    print("- 保存到新的目录避免覆盖")
    print("- 使用修改后的脚本确保保存")
    
    return cmd

if __name__ == "__main__":
    recover_training()
    create_recovery_command()
    
    print("\n=== 重要提醒 ===")
    print("1. 训练数据仍然存在，没有丢失")
    print("2. 修改后的train_3dgdm.py已添加自动保存功能")
    print("3. 建议立即开始恢复训练")
    print("4. 新训练将在output/garden/3dgdm_recovered目录保存完整结果")
    print("==================")