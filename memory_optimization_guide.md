# StyleGaussian 内存优化指南

## 🚨 CUDA内存不足解决方案

### 1. 立即解决方案

#### 减少批次大小和分辨率
```bash
# 使用更小的体素分辨率
python train_3dgdm.py --source_path datasets/garden --model_path output/garden/3dgdm_test --style_image images --enable_multi_style --style_mixing_prob 0.3 --curriculum_learning --iterations 1000 --voxel_resolution 16
```

#### 设置PyTorch内存管理
```bash
# 在运行前设置环境变量
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_3dgdm.py ...
```

### 2. 代码级优化

#### A. 梯度检查点 (已实现)
- 使用 `torch.utils.checkpoint` 减少前向传播的内存占用
- 以计算时间换取内存空间

#### B. 体素分辨率自适应
```python
# 根据GPU内存动态调整体素分辨率
def get_optimal_voxel_resolution():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory < 6 * 1024**3:  # < 6GB
        return 16
    elif gpu_memory < 12 * 1024**3:  # < 12GB
        return 24
    else:
        return 32
```

#### C. 混合精度训练
```python
# 使用 torch.cuda.amp 减少内存占用
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 训练策略优化

#### A. 渐进式训练
```python
# 从低分辨率开始，逐步增加
resolution_schedule = {
    0: 16,      # 前100步使用16x16x16
    100: 24,    # 100-500步使用24x24x24
    500: 32     # 500步后使用32x32x32
}
```

#### B. 内存监控
```python
def monitor_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
```

### 4. 硬件建议

#### 最低配置
- GPU: 6GB VRAM (GTX 1060 6GB / RTX 2060)
- 体素分辨率: 16x16x16
- 批次大小: 1

#### 推荐配置
- GPU: 8GB+ VRAM (RTX 3070 / RTX 4060 Ti)
- 体素分辨率: 24x24x24
- 批次大小: 1-2

#### 理想配置
- GPU: 12GB+ VRAM (RTX 3080 Ti / RTX 4070 Ti)
- 体素分辨率: 32x32x32
- 批次大小: 2-4

### 5. 紧急内存清理

```python
def emergency_memory_cleanup():
    """紧急内存清理"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()
```

### 6. 监控脚本

```python
# 添加到训练循环中
if iteration % 10 == 0:
    allocated = torch.cuda.memory_allocated() / 1024**3
    if allocated > 3.5:  # 接近4GB限制
        print(f"⚠️  内存使用过高: {allocated:.2f}GB")
        torch.cuda.empty_cache()
```

### 7. 配置文件优化

创建 `low_memory_config.yaml`:
```yaml
voxel_resolution: 16
batch_size: 1
gradient_checkpointing: true
mixed_precision: true
max_iterations: 1000
save_interval: 100
```

### 8. 故障排除

#### 如果仍然内存不足:
1. 减少风格图像数量 (从110个减少到50个)
2. 使用更小的输入图像分辨率
3. 禁用某些非关键功能
4. 考虑使用CPU进行某些计算

#### 内存泄漏检查:
```python
# 检查是否有内存泄漏
import tracemalloc
tracemalloc.start()

# 训练代码...

current, peak = tracemalloc.get_traced_memory()
print(f"当前内存: {current / 1024**2:.1f}MB")
print(f"峰值内存: {peak / 1024**2:.1f}MB")
tracemalloc.stop()
```

## 🎯 推荐的训练命令

### 4GB GPU (紧急模式)
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_3dgdm.py --source_path datasets/garden --model_path output/garden/3dgdm_test --style_image images --enable_multi_style --style_mixing_prob 0.3 --curriculum_learning --iterations 500 --voxel_resolution 12
```

### 6GB GPU (标准模式)
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_3dgdm.py --source_path datasets/garden --model_path output/garden/3dgdm_test --style_image images --enable_multi_style --style_mixing_prob 0.3 --curriculum_learning --iterations 1000 --voxel_resolution 16
```

### 8GB+ GPU (完整模式)
```bash
python train_3dgdm.py --source_path datasets/garden --model_path output/garden/3dgdm_test --style_image images --enable_multi_style --style_mixing_prob 0.3 --curriculum_learning --iterations 1000 --voxel_resolution 24
```