import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 显式指定可见 GPU 索引
print(torch.cuda.device_count())  # 重新检测
import torch

if torch.cuda.is_available():
    # 获取当前设备 ID
    current_device_id = torch.cuda.current_device()
    # 根据设备 ID 获取设备名称
    device_name = torch.cuda.get_device_name(current_device_id)
    print(f"当前 CUDA 设备名称：{device_name}")
else:
    print("CUDA 不可用")