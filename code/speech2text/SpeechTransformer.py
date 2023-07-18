from config import *
import numpy as np
import torch
import torch.nn as nn
import math
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).to(device)  # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


'''
Mask句子中没有实际意义的占位符，这个mask是encoder阶段需要的mask
这东西现在的问题就是我还没搞明白输入的是什么东西
'''


def get_attn_pad_mask(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    seq_q = seq_q[:, :, 0]
    seq_k = seq_k[:, :, 0]
    batch_size, len_q, = seq_q.size()
    batch_size, len_k, = seq_k.size()
    # print(pad_attn_mask.shape)
    # pad_attn_mask = torch.bmm(pad_attn_mask, pad_attn_mask.transpose(1, 2))
    # print(pad_attn_mask.shape)
    # return pad_attn_mask
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # print(pad_attn_mask.shape)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度


'''
当输入的是一个二维变量时调用这个函数
'''


def get_attn_pad_mask2(seq_q, seq_k):
    batch_size, len_q, = seq_q.size()
    batch_size, len_k, = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # print(pad_attn_mask.shape)
    # print(f'seq_q.shape={seq_q.shape}, mask.shape={(pad_attn_mask.expand(batch_size, len_q, len_k)).shape}')
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度


def get_attn_pad_mask3(seq_q, seq_k):
    # print(f'get_attn_pad_mask3中seq_q大小为：{seq_q.shape}')
    # print(f'get_attn_pad_mask3中seq_k大小为：{seq_k.shape}')
    batch_size, len_q, = seq_q.size()
    seq_k = seq_k[:, :, 0]
    batch_size, len_k, = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # print(f'get_attn_pad_mask3中生成的mask大小为：{pad_attn_mask.expand(batch_size, len_q, len_k).shape}')
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_pad_mask4(seq_q):  # [batch,src_len,feature]
    matrix = torch.bmm(seq_q, torch.transpose(seq_q, 1, 2))
    pad_attn_mask = matrix.data.eq(0)
    return pad_attn_mask


'''
Decoder上的mask，需要保证输入seq时的因果性
本质就是构建一个上三角矩阵
'''


def get_attn_subsequence_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


'''
点积注意力机制，就是那个KQV做矩阵运算的过程
'''


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # print(scores.shape)
        # print(attn_mask.shape)
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, onFrequency=False, in_feature=None):
        super(MultiHeadAttention, self).__init__()
        if onFrequency is False and in_feature is None:
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
            self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
            self.onFrequency = False
            self.in_feature = None
        if onFrequency is True and in_feature is None:
            self.W_Q = nn.Linear(feature_max_len, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(feature_max_len, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(feature_max_len, d_v * n_heads, bias=False)
            self.fc = nn.Linear(n_heads * d_v, feature_max_len, bias=False)
            self.onFrequency = True
            self.in_feature = None
        if in_feature is not None:
            self.W_Q = nn.Linear(in_feature, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(in_feature, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(in_feature, d_v * n_heads, bias=False)
            self.fc = nn.Linear(n_heads * d_v, in_feature, bias=False)
            self.onFrequency = None
            self.in_feature = in_feature

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        '''
        按照speech-transformer的说法这里肯定是要改成用卷积生成一下QKV
        '''
        # print(input_Q.shape)
        # print(self.W_Q)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len,
        # seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        if self.onFrequency is False and self.in_feature is None:
            return nn.LayerNorm(d_model).to(device)(output + residual), attn
        if self.onFrequency is True and self.in_feature is None:
            return nn.LayerNorm(feature_max_len).to(device)(output + residual), attn
        if self.in_feature is not None:
            return nn.LayerNorm(self.in_feature).to(device)(output + residual), attn

'''
按照论文的要求使用卷积先提取特征然后再做映射
'''
class MultiHeadConvAttention(nn.Module):
    def __init__(self, onFrequency=False):
        super(MultiHeadConvAttention, self).__init__()
        self.ConvQ = nn.Conv2d(1, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                                        padding=padding)
        self.ConvK = nn.Conv2d(1, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                                        padding=padding)
        self.ConvV = nn.Conv2d(1, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                                        padding=padding)
        if onFrequency is False:
            feature_len = math.floor((n_mfcc + 2 * padding - kernel_size) / stride) + 1
        else:
            feature_len = math.floor((feature_max_len + 2 * padding - kernel_size) / stride) + 1
        self.W_Q = nn.Linear(feature_len, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(feature_len, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(feature_len, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        '''
        按照speech-transformer的说法这里肯定是要改成用卷积生成一下QKV
        '''
        temp_Q = torch.unsqueeze(input_Q, dim=1)
        temp_K = torch.unsqueeze(input_K, dim=1)
        temp_V = torch.unsqueeze(input_V, dim=1)
        input_Q = self.ConvQ(temp_Q)
        input_K = self.ConvK(temp_K)
        input_V = self.ConvV(temp_V)
        input_Q = torch.squeeze(input_Q, dim=1)
        input_K = torch.squeeze(input_K, dim=1)
        input_V = torch.squeeze(input_V, dim=1)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


'''
经过两个全连接成，得到的结果再加上 inputs ，再做LayerNorm归一化。
分不同情况讨论，因为做转置之后输入的维度会不一样
'''


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, onFrequency=False, in_feature=None):
        super(PoswiseFeedForwardNet, self).__init__()
        if onFrequency is False and in_feature is None:
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False))
            self.onFrequency = False
            self.in_feature = None
        if onFrequency is True and in_feature is None:
            self.fc = nn.Sequential(
                nn.Linear(feature_max_len, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, feature_max_len, bias=False))
            self.onFrequency = True
            self.in_feature = None
        if in_feature is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_feature, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, in_feature, bias=False))
            self.onFrequency = None
            self.in_feature = in_feature


    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        if self.onFrequency is False and self.in_feature is None:
            residual = inputs
            output = self.fc(inputs)
            return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]
        if self.onFrequency is True and self.in_feature is None:
            residual = inputs
            output = self.fc(inputs)
            return nn.LayerNorm(feature_max_len).to(device)(output + residual)  # [batch_size, seq_len, d_model]
        if self.in_feature is not None:
            residual = inputs
            output = self.fc(inputs)
            return nn.LayerNorm(self.in_feature).to(device)(output + residual)  # [batch_size, seq_len, d_model]


'''
一个Encoder层
'''


class StartEncoderLayer(nn.Module):
    def __init__(self, onFrequency=False):
        super(StartEncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadConvAttention(onFrequency=onFrequency)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                                # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class EncoderLayer(nn.Module):
    def __init__(self, onFrequency=False):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(onFrequency=onFrequency)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(onFrequency=onFrequency)  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                                # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)  # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.layers[0] = StartEncoderLayer()
        self.layers2 = nn.ModuleList([EncoderLayer(onFrequency=True) for _ in range(n_layers - 1)])

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len]，这里也是，输入进来就是一个二维的东西实际上
        # 对时间做attention
        enc_outputs = self.pos_emb(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,
                                               enc_inputs)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        # 对频率做attention
        enc_outputs2 = torch.transpose(enc_outputs, 1, 2)
        enc_self_attn_mask2 = get_attn_pad_mask(torch.transpose(enc_inputs, 1, 2), torch.transpose(enc_inputs, 1, 2))
        enc_self_attns2 = []
        '''
        这块没啥好说的，就是先做时间上的再做频率上的，因为要共享权重所以这里对频率的attention直接对这时间上的attention的第一个层的特征图
        做转置
        '''
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
            if len(enc_self_attn) == 1:
                enc_outputs2 = torch.transpose(enc_outputs, 1, 2)

        for layer in self.layers2:
            enc_outputs2, enc_self_attn2 = layer(enc_outputs2, enc_self_attn_mask2)
            enc_self_attns2.append(enc_self_attn2)
        return enc_outputs, enc_self_attns, enc_outputs2, enc_self_attns2


class DecoderLayer(nn.Module):
    def __init__(self, self_attn_in_feature=None, enc_attn_in_feature=None, FFN_in_feature=None):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(in_feature=self_attn_in_feature)
        self.dec_enc_attn = MultiHeadAttention(in_feature=enc_attn_in_feature)
        self.pos_ffn = PoswiseFeedForwardNet(in_feature=FFN_in_feature)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):  # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # print(dec_outputs.shape, enc_outputs.shape, enc_outputs.shape)
        # 我感觉应该已经用不到这个了
        # dec_outputs = self.reshape_dec_outputs(dec_outputs, enc_outputs)
        # print(dec_outputs.shape, enc_outputs.shape, enc_outputs.shape)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        # print(f'dec_outputs.shape={dec_outputs.shape}')
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

    def reshape_dec_outputs(self, dec_outputs, enc_outputs):
        if dec_outputs.shape[2] < enc_outputs.shape[2]:
            temp = torch.zeros(enc_outputs.shape[0], enc_outputs.shape[1], enc_outputs.shape[2])
            temp[:, :, :dec_outputs.shape[2]] = dec_outputs
            return temp
        else:
            return dec_outputs


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.layers2 = nn.ModuleList([DecoderLayer(self_attn_in_feature=feature_max_len, enc_attn_in_feature=feature_max_len, FFN_in_feature=feature_max_len) for _ in range(n_layers)])
        # self.layers2[0] = DecoderLayer(self_attn_in_feature=tgt_max_len, enc_attn_in_feature=feature_max_len, FFN_in_feature=feature_max_len)
        self.frequencyLinear = nn.Linear(in_features=tgt_max_len,out_features=feature_max_len,bias=False)
    def forward(self, dec_inputs, enc_inputs, enc_outputs, enc_outputs2):  # dec_inputs: [batch_size, tgt_len]
        # enc_intpus: [batch_size, src_len]
        # enc_outputs: [batsh_size, src_len, d_model]
        # 在时间上做attention
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask2(dec_inputs, dec_inputs).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask3(dec_inputs, enc_inputs).to(device)  # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []

        # 在频率上做embedding
        dec_outputs2 = torch.transpose(dec_outputs, 1, 2)
        '''
        在Decoder阶段我把对时间上做attention的特征图直接拿过来转置之后现在时间是作为模型特征维度出现的
        我需要把这个维度映射到和mfcc一致才行，但是mask因为对应特征维度来说应该都要用到所以全是False
        '''
        # print(f'dec_outputs2.shape={dec_outputs2.shape}')
        dec_outputs2 = self.frequencyLinear(dec_outputs2)
        dec_self_attn_mask2 = get_attn_pad_mask4(dec_outputs2)
        # dec_enc_attn_mask2 = get_attn_pad_mask3(dec_inputs, enc_inputs)
        dec_self_attns2, dec_enc_attns2 = [], []

        for layer in self.layers:  # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        for layer in self.layers2:
            # print('here')
            dec_outputs2, dec_self_attn2, dec_enc_attn2 = layer(dec_outputs2, enc_outputs2, dec_self_attn_mask2,
                                                             dec_self_attn_mask2)
            dec_self_attns2.append(dec_self_attn2)
            dec_enc_attns2.append(dec_enc_attn2)

        return dec_outputs, dec_self_attns, dec_enc_attns, dec_outputs2


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder(tgt_vocab_size=tgt_vocab_size)
        '''
        这个全连接层的作用是把因为转置而造成的扩容再映射回来，其实就是我的tgt_max_len不会有mfcc采样一样大的时间维度长度
        卷积的作用是因为对时间和空间都各有一个特征图[batch_size,2,tgt_max_len,d_model] 把这个2卷积成1
        '''
        self.frequencyLinear = nn.Linear(feature_max_len, tgt_max_len, bias=False)
        self.Conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):  # enc_inputs: [batch_size, src_len]
                                                # dec_inputs: [batch_size, tgt_len]
        # print(f'enc_inputs.shape={enc_inputs.shape},dec_inputs.shape={dec_inputs.shape}')
        enc_outputs, enc_self_attns, enc_outputs2, enc_self_attns2 = self.Encoder(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # print(f'enc_outputs.shape={enc_outputs.shape},enc_outputs2.shape={enc_outputs2.shape}')
        dec_outputs, dec_self_attns, dec_enc_attns, dec_outputs2 = self.Decoder(dec_inputs, enc_inputs, enc_outputs, enc_outputs2)  # dec_outpus    : [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        # print(f'enc_outputs.shape={dec_outputs.shape},enc_outputs2.shape={dec_outputs2.shape}')
        # dec_outputs2 = torch.transpose(dec_outputs2, 1, 2)[:, :dec_outputs.shape[1], :dec_outputs.shape[2]]
        dec_outputs2 = self.frequencyLinear(dec_outputs2)
        dec_outputs2 = torch.transpose(dec_outputs2, 1, 2)
        # dec_outputs += dec_outputs2
        dec_outputs = torch.cat((torch.unsqueeze(dec_outputs, 1), torch.unsqueeze(dec_outputs2, 1)), dim=1)
        dec_outputs = self.Conv(dec_outputs)
        dec_outputs = torch.squeeze(dec_outputs, dim=1)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
