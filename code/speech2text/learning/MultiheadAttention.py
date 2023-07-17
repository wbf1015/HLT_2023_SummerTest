import torch
import torch.nn as nn

input_tensor = torch.randn(16,30,512)
multihead_attention = nn.MultiheadAttention(512, 8)
output_tensor, _ = multihead_attention(input_tensor, input_tensor, input_tensor)
print(output_tensor.shape)