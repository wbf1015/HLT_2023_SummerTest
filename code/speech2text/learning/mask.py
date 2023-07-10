import torch
import torch.nn.functional as F
# 词嵌入向量维度
d_model = 512

# 单词表大小
vocab_size = 1000

# dropout比例
dropout = 0.1

# 构造两个序列 序列的长度分别为4 5， 0索引为无效
padding_idx = 0

src_len = [4, 5, 3]
tgt_len = [4, 2, 6]

# 随机生成3个序列 不足最大序列长度的 在序列结尾pad 0
src_seq = x = torch.cat([
    F.pad(torch.randint(1, vocab_size, (1, L)), (0, max(src_len) - L)) for L in src_len
])
# [4, 5, 3]
# tensor([[129, 490, 572, 764,   0],
#         [636, 151, 572, 482, 666],
#         [   0,   0]])
tgt_seq = y = torch.cat([
    F.pad(torch.randint(1, vocab_size, (1, L)), (0, max(tgt_len) - L)) for L in tgt_len
])
# [4, 2, 6]
# tensor([[509, 360, 486,  88,   0,   0],
#         [415, 609,   0,   0,   0,   0],
#         [767, 817,  59, 990, 853, 101]])

# ----------------------------------
# encoder multi-head self-attn mask
# 在计算输入x token之间的attn_score时 需要忽略pad 0
valid_encoder_mhsa_pos = torch.vstack([
    F.pad(torch.ones(L), (0, max(src_len) - L)) for L in src_len
]).unsqueeze(-1)  # 扩展维度 用于批量计算mask矩阵 (B, Ns, 1) x (B, 1, Ns) -> (B, Ns, Ns)

print(f'valid_encoder_mhsa_pos: {valid_encoder_mhsa_pos.shape}')
# print(valid_encoder_mhsa_pos)
encoder_mhsa_mask = 1 - torch.bmm(valid_encoder_mhsa_pos, valid_encoder_mhsa_pos.transpose(-2, -1))
print(f'encoder_mhsa_mask:\n{encoder_mhsa_mask}')
print(f'encoder_mhsa_mask.shape:{encoder_mhsa_mask.shape}')
# ----------------------------------
