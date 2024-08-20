import torch
import numpy
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"연결 성공! Current GPU: {gpu_name}")
else:
    print("No GPU found")
