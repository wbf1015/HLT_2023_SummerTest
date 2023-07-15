import torch
import numpy as np
import torch.nn as nn
from config import *
import math


class CNN(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(CNN, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.Linear1 = nn.Linear(in_features=self.get_length(), out_features=tgt_max_len)
        self.Linear2 = nn.Linear(in_features=self.get_width(), out_features=d_model)

        self.embedding = self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.Conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)


    def forward(self, enc_inputs, dec_inputs):
        # print(enc_inputs.shape)
        enc_inputs = torch.unsqueeze(enc_inputs, dim=1)
        # print(enc_inputs.shape)
        enc_inputs = self.Conv1(enc_inputs)
        # print(enc_inputs.shape)
        enc_inputs = torch.squeeze(enc_inputs, dim=1)
        # print(enc_inputs.shape)
        # print(self.get_length())
        # print(self.get_width())
        enc_inputs = self.Linear2(enc_inputs)
        enc_inputs = torch.transpose(enc_inputs, 1, 2)
        enc_inputs = self.Linear1(enc_inputs)
        enc_inputs = torch.transpose(enc_inputs, 1, 2)

        dec_inputs = self.embedding(dec_inputs)
        dec_inputs = torch.unsqueeze(dec_inputs, dim=1)
        dec_inputs = self.Conv2(dec_inputs)
        dec_inputs = torch.squeeze(dec_inputs, dim=1)

        dec_outputs = torch.cat((torch.unsqueeze(enc_inputs, 1), torch.unsqueeze(dec_inputs, 1)), dim=1)
        dec_outputs = self.Conv3(dec_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))


    def get_length(self):
        temp = get_feature_size(feature_max_len, 5)
        temp = get_feature_size(temp, 5)
        temp = get_feature_size(temp, 5)
        return temp

    def get_width(self):
        temp = get_feature_size(d_model, 5)
        temp = get_feature_size(temp, 5)
        temp = get_feature_size(temp, 5)
        return temp



def get_feature_size(input_size, kernel_size, padding=0, stride=1):
    temp = math.floor((input_size - kernel_size + 2 * padding) / stride)
    temp += 1
    if temp > 1:
        return temp
    else:
        return 1
