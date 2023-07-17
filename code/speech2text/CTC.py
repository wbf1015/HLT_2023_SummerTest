import torch
import torch.nn as nn
import math
from config import *


class CTC(nn.Module):
    def __init__(self, output_dim, rnn_layers=5, rnn_units=128):
        super(CTC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
        )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.batchnorm2 = nn.BatchNorm2d(1)
        self.Linear1 = nn.Linear(in_features=self.get_length(), out_features=tgt_max_len)
        self.Linear2 = nn.Linear(in_features=self.get_width(), out_features=d_model)

        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.rnns = nn.ModuleList()
        for i in range(rnn_layers):
            input_size = d_model if i == 0 else rnn_units * 2
            rnn = nn.GRU(input_size, rnn_units, bidirectional=True, batch_first=True)
            self.rnns.append(rnn)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(rnn_units * 2, rnn_units * 2)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Linear(rnn_units * 2, output_dim, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)

        x = torch.squeeze(x, dim=1)
        x = self.Linear2(x)
        x = torch.transpose(x, 1, 2)
        x = self.Linear1(x)
        x = torch.transpose(x, 1, 2)
        for i, layer in enumerate(self.rnns):
            x, _ = layer(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        dec_logits = self.projection(x)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))

    def get_length(self):
        temp = get_feature_size(feature_max_len, 7)
        temp = get_feature_size(temp, 7)
        temp = get_feature_size(temp, 7)
        return temp

    def get_width(self):
        temp = get_feature_size(d_model, 7)
        temp = get_feature_size(temp, 7)
        temp = get_feature_size(temp, 7)
        return temp


def get_feature_size(input_size, kernel_size, padding=0, stride=1):
    temp = math.floor((input_size - kernel_size + 2 * padding) / stride)
    temp += 1
    if temp > 1:
        return temp
    else:
        return 1
