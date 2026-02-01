import torch
import os

# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

model = torch.load("./pytorch_tasks/save_models/model_Transformer_3_3_state_dict.pth", map_location=device)

print(type(model))
print(model)