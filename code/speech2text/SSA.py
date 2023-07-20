import torch
import torch.nn as nn
import math
import copy
import numpy as np
from config import *

HEAD_NUM = 4
MEAN = 256
STD = 1
DIM_FEEDFORWARD = 2048
ENCODER_LAYER = 6
DECODER_LAYER = 6

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).to(device) # enc_inputs: [seq_len, d_model]

    def forward(self, inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        inputs += self.pos_table[:inputs.size(1), :]
        return self.dropout(inputs)

class MultiHeadSSAN(nn.Module):
    def __init__(self, feature_len):
        super(MultiHeadSSAN, self).__init__()
        self.a = nn.Parameter(torch.randn(feature_len, d_model))
        self.c = nn.Parameter(torch.randn(feature_len, d_model))
        self.b = nn.Parameter(torch.randn(feature_len, d_model))
        self.d = nn.Parameter(torch.randn(feature_len, d_model))
        self.N1 = nn.Parameter(MEAN + STD * torch.randn(1))
        self.N2 = nn.Parameter(MEAN + STD * torch.randn(1))
        # self.a = torch.randn(feature_len, d_model)
        # self.c = torch.randn(feature_len, d_model)
        # self.b = torch.randn(feature_len, d_model)
        # self.d = torch.randn(feature_len, d_model)
        # self.N1 = MEAN + STD * torch.randn(1)
        # self.N2 = MEAN + STD * torch.randn(1)
        self.SAN = nn.MultiheadAttention(embed_dim=d_model, num_heads=HEAD_NUM)

    def generateQ(self, V):
        Q = V.clone()
        if math.floor(self.N1) > 1:
            for i in range(1, math.floor(self.N1)):
                Q[:, i:] += V[:, :-i] * self.a[:-i, :]
            for j in range(1, math.floor(self.N2)):
                Q[:, :-j] += V[:, j:] * self.c[j:, :]
        return Q

    def generateK(self, V):
        K = V.clone()
        if math.floor(self.N1) > 1:
            for i in range(1, math.floor(self.N1)):
                K[:, i:] += V[:, :-i] * self.b[:-i, :]
            for j in range(1, math.floor(self.N2)):
                K[:, :-j] += V[:, j:] * self.d[j:, :]
        return K

    def forward(self, x):
        Q = self.generateQ(x)
        K = self.generateK(x)
        attn, _ = self.SAN(Q, K, x)
        return attn


class SSANBlock(nn.Module):
    def __init__(self, feature_len):
        super(SSANBlock, self).__init__()
        self.Attention = MultiHeadSSAN(feature_len)
        self.LayerNorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        residual = x
        x = self.Attention(x)
        x = self.LayerNorm(x)
        return residual+x

class FeedForwardBlock(nn.Module):
    def __init__(self):
        super(FeedForwardBlock, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, DIM_FEEDFORWARD),
            nn.ReLU(),
            nn.Linear(DIM_FEEDFORWARD, d_model)
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        residual = x
        x = self.feedforward(x)
        x = self.LayerNorm(x)
        return residual + x

class SANBlock(nn.Module):
    def __init__(self):
        super(SANBlock, self).__init__()
        self.Attention = nn.MultiheadAttention(d_model, HEAD_NUM)
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        residual = Q
        outputs, _ = self.Attention(Q, K, V)
        outputs = self.LayerNorm(outputs)
        return outputs+residual


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.Attention = SSANBlock(feature_len=feature_max_len)
        self.FeedForward = FeedForwardBlock()

    def forward(self, x):
        x = self.Attention(x)
        x = self.FeedForward(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.PositionEncoding = PositionalEncoding(d_model=d_model)
        self.EncoderLayers = nn.ModuleList([EncoderLayer() for _ in range(ENCODER_LAYER)])

    def forward(self, enc_inputs):
        enc_outputs = self.PositionEncoding(enc_inputs) + enc_inputs
        for Layer in self.EncoderLayers:
            enc_outputs = Layer(enc_outputs)
        return enc_outputs

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.SSANBlock = SSANBlock(feature_len=tgt_max_len)
        self.Linear = nn.Linear(in_features=feature_max_len, out_features=tgt_max_len, bias=False)
        self.SANBlock = SANBlock()
        self.FeedForward = FeedForwardBlock()

    def forward(self, dec_inputs, enc_outputs):
        enc_outputs = torch.transpose(enc_outputs, 1, 2)
        enc_outputs = self.Linear(enc_outputs)
        enc_outputs = torch.transpose(enc_outputs, 1, 2)

        dec_outputs = self.SSANBlock(dec_inputs)
        dec_outputs = self.SANBlock(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.FeedForward(dec_outputs)
        return dec_outputs



class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model)
        self.PositionEmbedding = PositionalEncoding(d_model)
        self.DecoderLayers = nn.ModuleList([DecoderLayer() for _ in range(DECODER_LAYER)])
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, dec_inputs, enc_outputs):
        dec_inputs = self.embedding(dec_inputs)
        dec_outputs = self.PositionEmbedding(dec_inputs) + dec_inputs
        for layer in self.DecoderLayers:
            dec_outputs = layer(dec_outputs,enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits


class SSANTransformer(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(SSANTransformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder(tgt_vocab_size)
        


    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.Encoder(enc_inputs)
        dec_logits = self.Decoder(dec_inputs, enc_outputs)
        
        return dec_logits.view(-1, dec_logits.size(-1))
