import torch
import numpy as np
import torch.nn as nn
from config import *


class RNN(nn.Module):
    def __init__(self, n_class, n_hidden, tgt_word_len):
        super(RNN, self).__init__()
        self.lenReflect = nn.Linear(in_features=tgt_max_len, out_features=feature_max_len, bias=False)
        self.lenReflect2 = nn.Linear(in_features=feature_max_len, out_features=tgt_max_len, bias=False)
        self.embedding = nn.Embedding(num_embeddings=tgt_word_len, embedding_dim=d_model)
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
        self.fc = nn.Linear(n_hidden, tgt_word_len)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input(=input_batch): [batch_size, n_step+1, n_class]
        # dec_input(=output_batch): [batch_size, n_step+1, n_class]
        dec_input = self.embedding(dec_input)
        dec_input = self.lenReflect(torch.transpose(dec_input, 1, 2))
        dec_input = torch.transpose(dec_input, 1, 2)
        enc_input = enc_input.transpose(0, 1)  # enc_input: [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)  # dec_input: [n_step+1, batch_size, n_class]

        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.encoder(enc_input, enc_hidden)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, hidden = self.decoder(dec_input, h_t)
        outputs = torch.transpose(outputs, 0, 1)
        outputs = torch.transpose(outputs, 1, 2)
        outputs = self.lenReflect2(outputs)
        outputs = torch.transpose(outputs, 1, 2)
        dec_logits = self.fc(outputs)  # model : [n_step+1, batch_size, n_class]
        # print(dec_logits.shape)
        # print((dec_logits.view(-1, dec_logits.size(-1))).shape)
        return dec_logits.view(-1, dec_logits.size(-1)), hidden


RNN_hidden = 128
