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
def main():
    pass


if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs, tgt_vocab_size, reflection = makeData()
    test_enc_inputs, test_dec_inputs, test_dec_outputs = make_Testdata(reflection)
    print('enc_inputs.shape=', enc_inputs.shape)
    print('dec_inputs.shape=', dec_inputs.shape)
    print('dec_outputs.shape=', dec_outputs.shape)
    train_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs), batch_size, True)
    model = Transformer(tgt_vocab_size)
    # model = make_seq2seq_model(tgt_vocab_size)
    # model = RNN(d_model, RNN_hidden, tgt_vocab_size)
    # model = CNN(tgt_vocab_size)
    # model = CTC(output_dim=tgt_vocab_size)
    # model = conformer(tgt_vocab_size)
    # model = SSANTransformer(tgt_vocab_size)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    ctc_loss = nn.CTCLoss(blank=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    train_losses = []
    for epoch in range(epoch):
        count = 1
        for enc_inputs, dec_inputs, dec_outputs in tqdm(train_loader):  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)  # Speech Transformer: outputs: [batch_size * tgt_len, tgt_vocab_size]
            # outputs = model(enc_inputs, dec_inputs) # seq2seq只返回最后的outputs
            # outputs = model(enc_inputs, torch.zeros(1, enc_inputs.shape[0], RNN_hidden), dec_outputs) # RNN
            # outputs = model(enc_inputs, dec_inputs) # CNN
            # outputs = model(enc_inputs) # CTC,conformer
            # outputs = model(enc_inputs, dec_inputs) # SSAN Transformer

            # loss = criterion(outputs, dec_outputs.view(-1))

            new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
            outputs = outputs.view(new_shape)
            log_probs = torch.transpose(outputs,0,1).log_softmax(2)
            targets = CTC_targets_generator(dec_outputs)
            input_lengths = torch.full((len(enc_inputs),), tgt_max_len, dtype=torch.long)
            targets_lengths = CTC_targets_len_generator(dec_outputs)
            loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 20 == 0:
                model.eval()
                test_losses = 0.0
                WER_total = 0.0
                test_batch = 0
                with torch.no_grad():
                    for test_enc_inputs, test_dec_inputs, test_dec_outputs in test_loader:
                        test_enc_inputs, test_dec_inputs, test_dec_outputs = test_enc_inputs.to(device), test_dec_inputs.to(device), test_dec_outputs.to(device)
                        test_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(test_enc_inputs, test_dec_inputs)  # Speech Transformer: outputs: [batch_size * tgt_len, tgt_vocab_size]
                        new_shape = (len(test_enc_inputs),tgt_max_len,tgt_vocab_size)
                        test_outputs = test_outputs.view(new_shape)
                        WER_total += calWER(test_outputs, test_dec_outputs, reflection)
                        log_probs = torch.transpose(test_outputs,0,1).log_softmax(2)
                        targets = CTC_targets_generator(test_dec_outputs)
                        input_lengths = torch.full((len(test_enc_inputs),), tgt_max_len, dtype=torch.long)
                        targets_lengths = CTC_targets_len_generator(test_dec_outputs)
                        test_loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)
                        test_batch += 1
                        test_losses += test_loss

                print('Epoch:', '%04d' % (epoch + 1), 'batch=', count, 'train_loss =', '{:.6f}'.format(loss.item()),'test_loss = ','{:.6f}'.format(test_losses/test_batch),'test_WER = ','{:.6f}'.format(WER_total/test_batch))
                model.train()

    torch.save(model.state_dict(), 'model/SpeechTransformermodel.pt')
    save_dic(reflection, 'model/SpeechTransformer_Reflection.txt')
    # print("保存模型")
    np.savetxt('result/SpeechTransformerloss.txt', np.array(train_losses), fmt="%.4f")