import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from config import *

KERNEL_SIZE = 7
DROPOUT = 0.5
DIM_FEEDFORWARD = 2048
CONFORMER_FEATURE_LEN = 512
HEAD_NUM = 8
LAYER = 6

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)  # enc_inputs: [seq_len, d_model]

    def forward(self, inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        inputs += self.pos_table[:inputs.size(1), :]
        return self.dropout(inputs)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ConvolutionModule(nn.Module):
    def __init__(self):
        super(ConvolutionModule, self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=CONFORMER_FEATURE_LEN)
        self.PointWiseConv = nn.Conv1d(in_channels=tgt_max_len, out_channels=2*tgt_max_len, kernel_size=1, stride=1, padding=0)
        self.GLU = nn.GLU()
        self.DepthWiseConv = nn.Conv1d(in_channels=2*tgt_max_len, out_channels=4*tgt_max_len, kernel_size=KERNEL_SIZE, stride=1, padding=0)
        self.BatchNorm = nn.BatchNorm1d(4*tgt_max_len)
        self.Swish = Swish()
        self.Linear = nn.Linear(in_features=get_feature_size(CONFORMER_FEATURE_LEN/2, KERNEL_SIZE), out_features=CONFORMER_FEATURE_LEN)
        self.PointWiseConv2 = nn.Conv1d(in_channels=4*tgt_max_len,out_channels=tgt_max_len, kernel_size=1, stride=1, padding=0)
        self.DropOut = nn.Dropout(DROPOUT)

    def forward(self, x):
        residual = x
        x = self.LayerNorm(x)
        x = self.PointWiseConv(x)
        x = self.GLU(x)
        x = self.DepthWiseConv(x)
        x = self.BatchNorm(x)
        x = self.Swish(x)
        x = self.Linear(x)
        x = self.PointWiseConv2(x)
        x = self.DropOut(x)
        return x+residual

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=CONFORMER_FEATURE_LEN)
        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim=CONFORMER_FEATURE_LEN, num_heads=HEAD_NUM)
        self.DropOut =  nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.LayerNorm(x)
        x, _ = self.MultiHeadAttention(x, x, x)
        x = self.DropOut(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self):
        super(ConformerBlock, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(CONFORMER_FEATURE_LEN, DIM_FEEDFORWARD),
            nn.ReLU(),
            nn.Linear(DIM_FEEDFORWARD, CONFORMER_FEATURE_LEN),
            nn.ReLU()
        )
        self.MultiHeadSelfAttention = AttentionBlock()
        self.Convolution = ConvolutionModule()
        self.feedforward2 = nn.Sequential(
            nn.Linear(CONFORMER_FEATURE_LEN, DIM_FEEDFORWARD),
            nn.ReLU(),
            nn.Linear(DIM_FEEDFORWARD, CONFORMER_FEATURE_LEN),
            nn.ReLU()
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=CONFORMER_FEATURE_LEN)


    def forward(self, x):
        residual = x
        x = self.feedforward(x)
        x = 0.5 * x + residual
        residual = x
        x = self.MultiHeadSelfAttention(x)
        x = residual + x
        residual = x
        x = self.Convolution(x)
        x = residual + x
        residual = x
        x = self.feedforward2(x)
        x = 0.5 * x + residual
        x = self.LayerNorm(x)
        return x

class conformer(nn.Module):
    def __init__(self,tgt_vocab_len):
        super(conformer, self).__init__()
        self.PositionEncoding = PositionalEncoding(d_model=d_model)
        self.Conv1 = nn.Conv1d(in_channels=feature_max_len, out_channels=tgt_max_len, kernel_size=KERNEL_SIZE, stride=1, padding=0)
        self.Linear1 = nn.Linear(get_feature_size(d_model, KERNEL_SIZE), CONFORMER_FEATURE_LEN)
        self.Dropout = nn.Dropout(DROPOUT)
        self.conformer_blocks = nn.ModuleList([ConformerBlock() for _ in range(LAYER)])
        self.projection = nn.Linear(CONFORMER_FEATURE_LEN, tgt_vocab_len, bias=False)

    def forward(self, x):
        x = x + self.PositionEncoding(x)
        x = self.Conv1(x)
        x = self.Linear1(x)
        x = self.Dropout(x)
        for block in self.conformer_blocks:
            x = block(x)
        dec_logits = self.projection(x)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))


def get_feature_size(input_size, kernel_size, padding=0, stride=1):
    temp = math.floor((input_size - kernel_size + 2 * padding) / stride)
    temp += 1
    if temp > 1:
        return temp
    else:
        return 1