import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be >0
print(torch.cuda.current_device())  # Should print device ID
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda) # Should be 11.7, but CUDA system is 12.1
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should print GPU name
