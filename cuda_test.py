import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())  # 应返回True
print("显卡名称:", torch.cuda.get_device_name(0))  # 应显示"NVIDIA GeForce RTX 5090"