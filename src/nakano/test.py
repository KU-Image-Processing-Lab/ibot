import torch

print(torch.cuda.is_available())  # Trueが返ってくるか確認
print(torch.cuda.current_device())  # 現在使用中のGPUデバイスIDを確認
print(
    torch.cuda.get_device_name(torch.cuda.current_device())
)  # 使用中のGPUの名前を確認
