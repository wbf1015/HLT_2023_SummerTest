import torch
import torch.nn as nn

# 假设输入张量为 input_tensor，大小为 [batch_size, src_len, feature_len]
input_tensor = torch.randn(16, 30, 512)

# 创建 LayerNorm 模块
layer_norm = nn.LayerNorm(512)

# 对输入张量进行归一化
output_tensor = layer_norm(input_tensor)

# 打印归一化后的张量
print(input_tensor.shape)
print(output_tensor.shape)
