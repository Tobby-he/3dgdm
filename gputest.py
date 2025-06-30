import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.version.cuda)         # 应输出您的 CUDA 版本，例如 '11.8'
print(torch.cuda.get_device_name(0))  # 应输出您的 GPU 名称
