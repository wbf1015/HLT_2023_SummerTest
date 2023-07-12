import torch.nn as nn
import torch.optim as optim
from SpeechTransformer import *
from Dataset import *
from config import *
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs, tgt_vocab_size = makeData()
    print('enc_inputs.shape=', enc_inputs.shape)
    print('dec_inputs.shape=', dec_inputs.shape)
    print('dec_outputs.shape=', dec_outputs.shape)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)

    model = Transformer(tgt_vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    losses = []
    for epoch in range(epoch):
        count = 1
        for enc_inputs, dec_inputs, dec_outputs in tqdm(loader):  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            losses.append(loss.item())
            if count % 20 == 0:
                # print('Epoch:', '%04d' % (epoch + 1), 'batch=', count, 'loss =', '{:.6f}'.format(loss))
                pass
            optimizer.zero_grad()
            loss.backward()
            print('Epoch:', '%04d' % (epoch + 1), 'batch=', count, 'loss =', '{:.6f}'.format(loss))
            optimizer.step()
            count += 1
    # torch.save(model, 'model.pth')
    # print("保存模型")
    np.savetxt('loss.txt', np.array(losses), fmt="%.4f")