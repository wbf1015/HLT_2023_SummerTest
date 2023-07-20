import torch.nn as nn
import torch.optim as optim
from Dataset import *
from config import *
from tqdm import tqdm
from wordVector import *
from SpeechTransformer import *
from seq2seq import *
from RNN import *
from CNN import *
from CTC import *
from SSA import *
from conformer import *
import numpy as np

def mask_dec_inputs(index, dec_inputs):
    # 使用clone方法复制dec_inputs，生成一个新的张量dec_inputs_masked
    dec_inputs_masked = dec_inputs.clone()

    for batch_idx in range(dec_inputs.shape[0]):
        # 获取当前batch中的一维特征[tgt_len]
        feature = dec_inputs_masked[batch_idx]

        # 将索引大于i的位置的元素置为0
        feature[index + 1:] = 0

    return dec_inputs_masked

def eval_SpeechTransformer(model, enc_inputs, dec_inputs, tgt_vocab_size):
    enc_outputs, _, enc_outputs2, _ = model.Encoder(enc_inputs)
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs, _, _ = model.Decoder(temp_dec_outputs, enc_inputs, enc_outputs, enc_outputs2)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs
    
def eval_SSAN(model, enc_inputs, dec_inputs, tgt_vocab_size):
    enc_outputs = model.Encoder(enc_inputs)
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs = model.Decoder(temp_dec_outputs, enc_outputs)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs

def eval_Conformer(model, enc_inputs, dec_inputs, tgt_vocab_size):
    enc_outputs,= model.Encoder(enc_inputs)
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs= model.Decoder(temp_dec_outputs, enc_inputs, enc_outputs, enc_outputs2)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs

def eval_RNN(model, enc_inputs, dec_inputs, tgt_vocab_size):
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs, _= model(enc_inputs, torch.zeros(1, enc_inputs.shape[0], RNN_hidden).to(device), temp_dec_outputs)
        
        # 在RNN里我没有明着写Decoder，所以他顺下来就是一个二维的，需要我转换一下
        new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
        temp_dec_outputs = temp_dec_outputs.view(new_shape)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs

def eval_seq2seq(model, enc_inputs, dec_inputs, tgt_vocab_size):
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs = model(enc_inputs, temp_dec_outputs)

        # 没必要先Encoder-再Deocder一下，其实这样无非就是浪费了一些计算资源.
        new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
        temp_dec_outputs = temp_dec_outputs.view(new_shape)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs

def eval_CNN(model, enc_inputs, dec_inputs, tgt_vocab_size):
    dec_outputs = torch.zeros(enc_inputs.shape[0], tgt_max_len, tgt_vocab_size)
    for i in range(0,dec_inputs.shape[1]-1):
        temp_dec_outputs = mask_dec_inputs(i,dec_inputs)
        temp_dec_outputs = model(enc_inputs, temp_dec_outputs)

        # 没必要先Encoder-再Deocder一下，其实这样无非就是浪费了一些计算资源.
        new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
        temp_dec_outputs = temp_dec_outputs.view(new_shape)
        dec_outputs[:,i,:] = temp_dec_outputs[:,i,:]
    return dec_outputs


if __name__ == "__main__":
    pass
