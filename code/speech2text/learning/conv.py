import torch.nn as nn
import torch

im = torch.randn(16, 512, 128)
# im = torch.unsqueeze(im, dim=1)
c = nn.Conv1d(512, 3, kernel_size=3, stride=1, padding=0)
output = c(im)

print(im.shape)
print(output.shape)
# output = torch.squeeze(output, dim=1)
print(output.shape)
# print(list(c.parameters()))
