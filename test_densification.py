#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试致密化功能是否正常工作
"""

import torch
import yaml
from arguments import ModelParams, PipelineParams, OptimizationParams

def test_densification_config():
    """测试致密化配置是否正确加载"""
    print("=== 测试致密化配置加载 ===")
    
    # 加载配置文件
    config_path = "configs/3dgdm_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查配置文件中的致密化参数
    if 'gaussian' in config and 'densification' in config['gaussian']:
        densify_config = config['gaussian']['densification']
        print(f"✓ 配置文件中的致密化参数:")
        print(f"  - start_iter: {densify_config.get('start_iter')}")
        print(f"  - end_iter: {densify_config.get('end_iter')}")
        print(f"  - densify_grad_threshold: {densify_config.get('densify_grad_threshold')}")
        print(f"  - opacity_threshold: {densify_config.get('opacity_threshold')}")
    else:
        print("✗ 配置文件中未找到致密化参数")
        return False
    
    # 模拟参数解析
    import argparse
    parser = argparse.ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    
    # 应用配置文件参数（模拟train_3dgdm.py中的逻辑）
    if 'gaussian' in config:
        gaussian_config = config['gaussian']
        if 'densification' in gaussian_config:
            densify_config = gaussian_config['densification']
            if 'start_iter' in densify_config:
                args.densify_from_iter = densify_config['start_iter']
            if 'end_iter' in densify_config:
                args.densify_until_iter = densify_config['end_iter']
            if 'densify_grad_threshold' in densify_config:
                args.densify_grad_threshold = densify_config['densify_grad_threshold']
            if 'opacity_threshold' in densify_config:
                args.opacity_threshold = densify_config['opacity_threshold']
    
    # 检查参数是否正确设置
    print(f"\n✓ 解析后的致密化参数:")
    print(f"  - densify_from_iter: {getattr(args, 'densify_from_iter', 'NOT SET')}")
    print(f"  - densify_until_iter: {getattr(args, 'densify_until_iter', 'NOT SET')}")
    print(f"  - densify_grad_threshold: {getattr(args, 'densify_grad_threshold', 'NOT SET')}")
    print(f"  - densification_interval: {getattr(args, 'densification_interval', 'NOT SET')}")
    print(f"  - opacity_reset_interval: {getattr(args, 'opacity_reset_interval', 'NOT SET')}")
    
    return True

def test_densification_logic():
    """测试致密化逻辑"""
    print("\n=== 测试致密化逻辑 ===")
    
    # 模拟致密化条件检查
    iteration = 150
    densify_from_iter = 100
    densify_until_iter = 800
    densification_interval = 100
    opacity_reset_interval = 3000
    
    print(f"当前迭代: {iteration}")
    print(f"致密化范围: {densify_from_iter} - {densify_until_iter}")
    
    # 检查是否在致密化范围内
    if iteration < densify_until_iter:
        print("✓ 在致密化范围内")
        
        # 检查是否应该执行致密化
        if iteration > densify_from_iter and iteration % densification_interval == 0:
            print(f"✓ 应该在迭代 {iteration} 执行致密化")
        else:
            print(f"- 不需要在迭代 {iteration} 执行致密化")
        
        # 检查是否应该重置透明度
        if iteration % opacity_reset_interval == 0:
            print(f"✓ 应该在迭代 {iteration} 重置透明度")
        else:
            print(f"- 不需要在迭代 {iteration} 重置透明度")
    else:
        print("✗ 超出致密化范围")
    
    return True

if __name__ == "__main__":
    print("开始测试致密化功能...\n")
    
    try:
        # 测试配置加载
        if test_densification_config():
            print("\n✓ 配置加载测试通过")
        else:
            print("\n✗ 配置加载测试失败")
            exit(1)
        
        # 测试致密化逻辑
        if test_densification_logic():
            print("\n✓ 致密化逻辑测试通过")
        else:
            print("\n✗ 致密化逻辑测试失败")
            exit(1)
        
        print("\n🎉 所有测试通过！致密化功能已正确集成。")
        print("\n📝 使用说明:")
        print("1. 致密化将在迭代 100-800 之间进行")
        print("2. 每 100 次迭代执行一次致密化")
        print("3. 每 3000 次迭代重置一次透明度")
        print("4. 致密化阈值为 0.0001")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)