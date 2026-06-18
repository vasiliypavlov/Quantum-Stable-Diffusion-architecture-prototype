import torch
print("PyTorch версия:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
print("Имя GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Нет доступа к GPU")
