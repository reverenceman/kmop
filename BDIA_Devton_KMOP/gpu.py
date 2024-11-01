import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU 사용
    print("GPU 사용 중")
    print("GPU 디바이스:", torch.cuda.get_device_name(0))  # 첫 번째 GPU의 이름을 가져옵니다. 여러 개의 GPU가 있다면 인덱스를 조정하세요.
else:
    device = torch.device("cpu")          # CPU 사용
    print("CPU 사용 중")

# 현재 사용 중인 디바이스 확인
print("현재 디바이스:", device)
