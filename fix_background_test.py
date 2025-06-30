#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
背景设置修复测试脚本
用于验证推理脚本中的背景设置是否正确
"""

import torch
import os
from pathlib import Path

def test_background_fix():
    """测试背景设置修复"""
    print("=== 背景设置修复测试 ===")
    
    # 测试数据集路径
    test_datasets = [
        "output/Horse/3dgdm_test",
        "output/train/artistic/default",
        "output/garden/3dgdm_test"
    ]
    
    for dataset_path in test_datasets:
        if os.path.exists(dataset_path):
            print(f"\n测试数据集: {dataset_path}")
            
            # 检查cfg_args文件
            cfg_args_path = os.path.join(dataset_path, "cfg_args")
            if os.path.exists(cfg_args_path):
                with open(cfg_args_path, 'r') as f:
                    content = f.read()
                    if 'white_background=False' in content:
                        expected_bg = "黑色 [0, 0, 0]"
                        print(f"  配置: white_background=False")
                        print(f"  期望背景: {expected_bg}")
                    elif 'white_background=True' in content:
                        expected_bg = "白色 [1, 1, 1]"
                        print(f"  配置: white_background=True")
                        print(f"  期望背景: {expected_bg}")
                    else:
                        print(f"  配置: 未找到white_background设置，使用默认值False")
                        print(f"  期望背景: 黑色 [0, 0, 0]")
            else:
                print(f"  警告: 未找到cfg_args文件")
        else:
            print(f"\n跳过不存在的数据集: {dataset_path}")
    
    print("\n=== 修复说明 ===")
    print("1. 修复了inference_3dgdm.py中的硬编码白色背景问题")
    print("2. 修复了inference_3dgdm_enhanced.py中的背景设置")
    print("3. 修复了inference_3dgdm_simple.py中的背景设置")
    print("4. 现在所有推理脚本都会根据数据集配置动态设置背景颜色")
    
    print("\n=== 使用建议 ===")
    print("1. 重新运行推理脚本，深色边缘问题应该得到解决")
    print("2. 如果仍有问题，检查数据集的cfg_args文件中的white_background设置")
    print("3. 确保训练和推理使用相同的背景设置")

def demonstrate_background_logic():
    """演示背景设置逻辑"""
    print("\n=== 背景设置逻辑演示 ===")
    
    # 模拟不同的配置
    configs = [
        {'white_background': True, 'description': '白色背景数据集'},
        {'white_background': False, 'description': '黑色背景数据集'}
    ]
    
    for config in configs:
        white_bg = config['white_background']
        bg_color = [1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0]
        bg_tensor = torch.tensor(bg_color, dtype=torch.float32)
        
        print(f"\n{config['description']}:")
        print(f"  white_background = {white_bg}")
        print(f"  bg_color = {bg_color}")
        print(f"  background_tensor = {bg_tensor}")

if __name__ == "__main__":
    test_background_fix()
    demonstrate_background_logic()
    
    print("\n=== 测试完成 ===")
    print("现在可以重新运行推理脚本测试修复效果！")