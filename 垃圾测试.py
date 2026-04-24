import torch
# 换成你刚才报错的那个 shezhenv3 的 pth 路径
pth_data = torch.load("../dataset/shezhenv3_train_data.pth")

print("整体类型是:", type(pth_data))
if isinstance(pth_data, dict):
    print("拿前两个看看:", list(pth_data.items())[:2])
elif isinstance(pth_data, list):
    print("拿第一个看看:", pth_data[0])