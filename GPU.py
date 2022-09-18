import torch
flag = torch.cuda.is_available()

if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")
ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("驱动为：",device)