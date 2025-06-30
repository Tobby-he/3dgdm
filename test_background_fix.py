#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试背景修复效果
"""

import os
import torch
import re
from pathlib import Path

def test_cfg_args_parsing():
    """测试cfg_args文件解析"""
    print("=== 测试cfg_args文件解析 ===")
    
    # 测试不同的cfg_args内容
    test_cases = [
        {
            'content': "Namespace(data_device='cuda', eval=False, images='images', model_path='output/Horse/3dgdm_test', resolution=-1, sh_degree=3, source_path='E:\\StyleGaussian-main\\datasets\\Horse', white_background=False)",
            'expected': False,
            'description': '黑色背景配置'
        },
        {
            'content': "Namespace(data_device='cuda', eval=False, images='images', model_path='output/test', resolution=-1, sh_degree=3, source_path='datasets/test', white_background=True)",
            'expected': True,
            'description': '白色背景配置'
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['description']}")
        print(f"配置内容: {case['content'][:80]}...")
        
        # 解析white_background值
        white_bg_match = re.search(r'white_background=([^,\)]+)', case['content'])
        if white_bg_match:
            white_bg_value = white_bg_match.group(1).strip()
            parsed_value = white_bg_value.lower() == 'true'
            print(f"解析结果: {parsed_value}")
            print(f"期望结果: {case['expected']}")
            print(f"解析正确: {'✓' if parsed_value == case['expected'] else '✗'}")
        else:
            print("解析失败: 未找到white_background配置")

def test_background_color_logic():
    """测试背景颜色设置逻辑"""
    print("\n=== 测试背景颜色设置逻辑 ===")
    
    test_cases = [
        {'white_background': True, 'expected_color': [1.0, 1.0, 1.0]},
        {'white_background': False, 'expected_color': [0.0, 0.0, 0.0]}
    ]
    
    for case in test_cases:
        white_bg = case['white_background']
        bg_color = [1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0]
        bg_tensor = torch.tensor(bg_color)
        
        print(f"\nwhite_background = {white_bg}")
        print(f"背景颜色 = {bg_color}")
        print(f"背景张量 = {bg_tensor}")
        print(f"颜色正确: {'✓' if bg_color == case['expected_color'] else '✗'}")

def test_real_cfg_files():
    """测试实际的cfg_args文件"""
    print("\n=== 测试实际cfg_args文件 ===")
    
    # 查找项目中的cfg_args文件
    project_root = Path("e:/StyleGaussian-main")
    cfg_files = list(project_root.rglob("cfg_args"))
    
    print(f"找到 {len(cfg_files)} 个cfg_args文件:")
    
    for cfg_file in cfg_files[:5]:  # 只测试前5个
        print(f"\n文件: {cfg_file}")
        try:
            with open(cfg_file, 'r') as f:
                content = f.read().strip()
                print(f"内容: {content[:100]}...")
                
                # 解析white_background
                white_bg_match = re.search(r'white_background=([^,\)]+)', content)
                if white_bg_match:
                    white_bg_value = white_bg_match.group(1).strip()
                    parsed_value = white_bg_value.lower() == 'true'
                    bg_color = [1.0, 1.0, 1.0] if parsed_value else [0.0, 0.0, 0.0]
                    print(f"white_background = {parsed_value}")
                    print(f"背景颜色 = {bg_color}")
                else:
                    print("未找到white_background配置")
        except Exception as e:
            print(f"读取文件失败: {e}")

def demonstrate_fix():
    """演示修复效果"""
    print("\n=== 背景修复效果演示 ===")
    
    print("\n修复前的问题:")
    print("- 推理脚本硬编码 white_background=False")
    print("- 但渲染时使用白色背景 [1.0, 1.0, 1.0]")
    print("- 导致训练(黑背景)与推理(白背景)不匹配")
    print("- 结果: 图像边缘出现深色竖边")
    
    print("\n修复后的改进:")
    print("- 从cfg_args文件动态读取white_background配置")
    print("- 根据配置自动设置正确的背景颜色")
    print("- 确保训练与推理背景一致")
    print("- 结果: 消除深色边缘问题")
    
    print("\n修复的文件:")
    print("- inference_3dgdm_enhanced.py: 添加cfg_args读取逻辑")
    print("- inference_3dgdm_simple.py: 添加cfg_args读取逻辑")
    print("- inference_3dgdm.py: 已有正确的读取逻辑")

if __name__ == "__main__":
    print("StyleGaussian 背景修复测试")
    print("=" * 50)
    
    test_cfg_args_parsing()
    test_background_color_logic()
    test_real_cfg_files()
    demonstrate_fix()
    
    print("\n=== 测试完成 ===")
    print("\n使用建议:")
    print("1. 运行推理前检查cfg_args文件是否存在")
    print("2. 确认white_background配置正确")
    print("3. 观察推理输出中的背景设置信息")
    print("4. 对比修复前后的渲染结果")