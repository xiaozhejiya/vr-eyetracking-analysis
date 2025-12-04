import torch
import time
avail = torch.cuda.is_available()
print(avail)
device = torch.device("cuda:0" if avail else "cpu")
print(device)
if avail and torch.cuda.device_count() > 0:
    print(torch.cuda.get_device_name(0))
else:
    print("N/A")
n = 1024
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
c = a @ b
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print({"matmul_time_s": t1 - t0, "device_type": device.type})
