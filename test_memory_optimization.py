#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化测试脚本
测试修改后的注意力机制是否能有效减少内存使用
"""

import torch
import torch.nn.functional as F
from diffusion.enhanced_significance_attention import EnhancedSignificanceAttention

def test_memory_usage():
    """测试内存使用情况"""
    print("=== 内存优化测试 ===")
    
    # 创建模型
    model = EnhancedSignificanceAttention(feature_dim=512).cuda()
    
    # 测试不同分辨率的输入
    test_cases = [
        (256, 256),   # 中等分辨率
        (512, 512),   # 高分辨率
        (800, 600),   # 超高分辨率
    ]
    
    for h, w in test_cases:
        print(f"\n测试分辨率: {h}x{w}")
        
        try:
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"初始GPU内存: {initial_memory:.2f} GB")
            
            # 创建测试输入
            rendered_image = torch.randn(1, 3, h, w).cuda()
            style_image = torch.randn(1, 3, h, w).cuda()
            
            # 前向传播
            with torch.cuda.amp.autocast():
                significance_map = model(rendered_image, style_image)
            
            # 记录峰值内存
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"峰值GPU内存: {peak_memory:.2f} GB")
            print(f"内存增长: {peak_memory - initial_memory:.2f} GB")
            print(f"输出形状: {significance_map.shape}")
            print("✓ 测试通过")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"✗ 内存不足: {str(e)}")
        except Exception as e:
            print(f"✗ 其他错误: {str(e)}")
        finally:
            # 清理内存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    test_memory_usage()