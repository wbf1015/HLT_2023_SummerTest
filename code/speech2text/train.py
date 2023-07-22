import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import sys
from Dataset import *
from config import *
from tqdm import tqdm
from eval import *
from wordVector import *
from SpeechTransformer import *
from seq2seq import *
from RNN import *
from CNN import *
from CTC import *
from SSA import *
from conformer import *
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description="Training settings")
parser.add_argument('--model', dest='model', type=str, help='model name', default='seq2seq')
parser.add_argument('--loss', dest='loss', type=str, help='model loss', default='ctcloss')
parser.add_argument('--seed', dest='seed', type=int, help='torch seed', default=2244668800)
parser.add_argument('--epoch', dest='epoch', type=int, help='train epoch', default=epoch)
parser.add_argument('--lr', dest='lr', type=float, help='learning rate', default=lr)
parser.add_argument('--batchsize', dest='batchsize', type=int, help='train batch size', default=batch_size)
parser.add_argument('--train_step', dest='train_step', type=int, help='every batch 2 report train loss at cmd', default=train_step)
parser.add_argument('--test_step', dest='test_step', type=int, help='every batch 2 report test loss at cmd', default=test_step)
parser.add_argument('--special_index', dest='special_index', type=str, help='special_index which will add 2 filename', default='')
parser.add_argument('--test_data_mask', dest='test_data_mask', type=str, help='self_regeression or mask the dec_inputs', default='self-regeression')
parser.add_argument('--model_data', dest='model_data', type=int, help='input anything if you want 2 get data of model', default=0)

args = parser.parse_args()

def main(args):
    # 设置随机数种子
    seed_setting(args.seed)

    # 加载数据
    enc_inputs, dec_inputs, dec_outputs, tgt_vocab_size, reflection = makeData()
    if args.test_data_mask == 'mask':
        test_enc_inputs, test_dec_inputs, test_dec_outputs = make_Testdata_with_mask(reflection)
    else:
        test_enc_inputs, test_dec_inputs, test_dec_outputs = make_Testdata(reflection)
    print('enc_inputs.shape=', enc_inputs.shape)
    print('dec_inputs.shape=', dec_inputs.shape)
    print('dec_outputs.shape=', dec_outputs.shape)
    train_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs), batch_size, True)

    # 声明模型
    if args.model == 'SpeechTransformer':
        model = Transformer(tgt_vocab_size)
    elif args.model == 'Conformer':
        model = conformer(tgt_vocab_size)
    elif args.model == 'SSAN':
        model = SSANTransformer(tgt_vocab_size)
    elif args.model == 'RNN':
        model = RNN(d_model, RNN_hidden, tgt_vocab_size)
    elif args.model == 'CTC':
        model = CTC(output_dim=tgt_vocab_size)
    elif args.model == 'seq2seq':
        model = make_seq2seq_model(tgt_vocab_size)
    elif args.model == 'CNN':
        model = CNN(tgt_vocab_size)
    else:
        print('对不起，没有这个模型，请检查后输入')
        sys.exit(-1)
    model = model.to(device)

    if args.model_data != 0 :
        if args.model == 'SpeechTransformer' or args.model == 'CNN' or args.model == 'seq2seq' or args.model == 'Conformer' or args.model == 'SSAN':
            myinput1 = torch.zeros((batch_size, feature_max_len, d_model)).to(device)
            myinput2 = torch.zeros((batch_size, tgt_max_len)).to(device).long()
            flops, params = profile(model, inputs=(myinput1,  myinput2))
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)
        if args.model == 'RNN':
            myinput1 = torch.zeros((batch_size, feature_max_len, d_model)).to(device)
            myinput2 = torch.zeros((1, batch_size, RNN_hidden)).to(device)
            myinput3 = torch.zeros((batch_size, tgt_max_len)).to(device).long()
            flops, params = profile(model, inputs=(myinput1,  myinput2, myinput3))
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)
        if args.model == 'CTC':
            myinput1 = torch.zeros((batch_size, feature_max_len, d_model)).to(device)
            flops, params = profile(model, inputs=(myinput1,))
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)

    # 声明损失函数
    if args.loss == 'ctcloss':
        ctc_loss = nn.CTCLoss(blank=0)
    elif args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    else:
        print('对不起，没有这个损失函数，请检查后输入')
        sys.exit(-1)
    
    # 声明优化器以及数据存储容器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    train_losses = []
    test_losses = []
    test_WER = []

    # 开始训练的回合
    for epoch in range(args.epoch):
        count = 0
        for enc_inputs, dec_inputs, dec_outputs in tqdm(train_loader):  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            
            # 根据不同的模型，对模型训练的输入输出做出相应的调整
            if args.model == 'SpeechTransformer':
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            elif args.model == 'Conformer':
                outputs = model(enc_inputs, dec_inputs)
            elif args.model == 'SSAN':
                outputs = model(enc_inputs, dec_inputs) # SSAN Transformer
            elif args.model == 'RNN':
                outputs, _ = model(enc_inputs, torch.zeros(1, enc_inputs.shape[0], RNN_hidden).to(device), dec_outputs)
            elif args.model == 'CTC':
                outputs = model(enc_inputs)
            elif args.model == 'seq2seq':
                 outputs = model(enc_inputs, dec_inputs)
            elif args.model == 'CNN':
                outputs = model(enc_inputs, dec_inputs)

            # 根据不同的损失函数计算损失
            if args.loss == 'ctcloss':
                new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
                outputs = outputs.view(new_shape)
                log_probs = torch.transpose(outputs,0,1).log_softmax(2)
                targets = CTC_targets_generator(dec_outputs)
                input_lengths = torch.full((len(enc_inputs),), tgt_max_len, dtype=torch.long)
                targets_lengths = CTC_targets_len_generator(dec_outputs)
                loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)
            elif args.loss == 'CrossEntropyLoss':
                loss = criterion(outputs, dec_outputs.view(-1))

            # 记录损失并完成反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            count += 1

            # 在测试集上进行验证
            if (count % args.train_step == 0) and (count % args.test_step != 0):
                print('Epoch:', '%04d' % (epoch + 1), ' batch=', count, ' train_loss =', '{:.6f}'.format(loss.item()))
            if count % args.test_step == 0:
                model.eval()
                total_test_loss = 0.0
                WER_total = 0.0
                test_batch = 0
                with torch.no_grad():
                    for test_enc_inputs, test_dec_inputs, test_dec_outputs in tqdm(test_loader):
                        test_enc_inputs, test_dec_inputs, test_dec_outputs = test_enc_inputs.to(device), test_dec_inputs.to(device), test_dec_outputs.to(device)
                        
                        # 根据不同的测试方法调用不同的函数进行测试
                        if args.test_data_mask == 'mask':
                            if args.model == 'SpeechTransformer':
                                test_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(test_enc_inputs, test_dec_inputs)  # Speech Transformer: outputs: [batch_size * tgt_len, tgt_vocab_size]
                            elif args.model == 'SSAN':
                                test_outputs = model(test_enc_inputs, test_dec_inputs)
                            elif args.model == 'Conformer':
                                test_outputs = model(test_enc_inputs, test_dec_inputs)
                            elif args.model == 'RNN':
                                test_outputs = model(test_enc_inputs, torch.zeros(1, enc_inputs.shape[0], RNN_hidden).to(device), test_dec_inputs)
                            elif args.model == 'CTC':
                                test_outputs = model(test_enc_inputs)
                            elif args.model == 'seq2seq':
                                test_outputs = model(test_enc_inputs, test_dec_inputs)
                            elif args.model == 'CNN':
                                test_outputs = model(test_enc_inputs, test_dec_inputs)
                        else:
                            if args.model == 'SpeechTransformer':
                                test_outputs = eval_SpeechTransformer(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
                            elif args.model == 'SSAN':
                                test_outputs = eval_SSAN(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
                            elif args.model == 'Conformer':
                                test_outputs = eval_Conformer(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
                            elif args.model == 'RNN':
                                test_outputs = eval_RNN(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
                            elif args.model == 'CTC':
                                test_outputs = model(test_enc_inputs)
                            elif args.model == 'seq2seq':
                                test_outputs = eval_seq2seq(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
                            elif args.model == 'CNN':
                                test_outputs = eval_CNN(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)

                        
                        # ctcloss在计算是需要dec_outputs保持[batch,tgt_len,tgt_vocab_len]的大小,这个和WER的计算保持一致
                        if args.loss == 'ctcloss':
                            new_shape = (len(test_enc_inputs),tgt_max_len,tgt_vocab_size)
                            test_outputs = test_outputs.view(new_shape)
                            log_probs = torch.transpose(test_outputs,0,1).log_softmax(2)
                            targets = CTC_targets_generator(test_dec_outputs)
                            input_lengths = torch.full((len(test_enc_inputs),), tgt_max_len, dtype=torch.long)
                            targets_lengths = CTC_targets_len_generator(test_dec_outputs)
                            test_loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)
                            WER_total += calWER(test_outputs, test_dec_outputs, reflection)
                            
                        if args.loss == 'CrossEntropyLoss':
                            new_shape = (len(test_enc_inputs)*tgt_max_len,tgt_vocab_size)
                            test_outputs = test_outputs.view(new_shape)
                            test_loss = criterion(outputs, dec_outputs.view(-1))
                            new_shape = (len(test_enc_inputs),tgt_max_len,tgt_vocab_size)
                            test_outputs = test_outputs.view(new_shape)
                            WER_total += calWER(test_outputs, test_dec_outputs, reflection)
                        
                        test_batch += 1
                        total_test_loss += test_loss.item()

                print()
                print('Epoch:', '%04d' % (epoch + 1), ' batch=', count, ' train_loss =', '{:.6f}'.format(loss.item()),' test_loss = ','{:.6f}'.format(total_test_loss/test_batch),' test_WER = ','{:.6f}'.format(WER_total/test_batch))
                test_losses.append(total_test_loss/test_batch)
                test_WER.append(WER_total/test_batch)
                model.train()
    
    # 保存模型和数据
    model_path = '/data/' + args.model + '/' + 'model' + args.special_index + '.pt'
    dic_path = '/data/' + args.model + '/' + 'reflection' + args.special_index + '.txt'
    train_loss_path = '/data/' + args.model + '/' + 'train_loss' + args.special_index + '.txt'
    test_loss_path = '/data/' + args.model + '/' + 'test_loss' + args.special_index + '.txt'
    test_WER_path = '/data/' + args.model + '/' + 'test_WER' + args.special_index + '.txt'

    torch.save(model.state_dict(), model_path)
    save_dic(reflection, dic_path)
    np.savetxt(train_loss_path, np.array(train_losses), fmt="%.6f")
    np.savetxt(test_loss_path, np.array(test_losses), fmt="%.6f")
    np.savetxt(test_WER_path, np.array(test_WER), fmt="%.6f")

 

def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)


if __name__ == "__main__":
    main(args)
    # seed_setting(264444880)
    # # seed = torch.initial_seed()
    # # print("当前的初始种子值为:", seed)

    # enc_inputs, dec_inputs, dec_outputs, tgt_vocab_size, reflection = makeData()
    # test_enc_inputs, test_dec_inputs, test_dec_outputs = make_Testdata(reflection)
    # print('enc_inputs.shape=', enc_inputs.shape)
    # print('dec_inputs.shape=', dec_inputs.shape)
    # print('dec_outputs.shape=', dec_outputs.shape)
    # train_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    # test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs), batch_size, True)
    # model = Transformer(tgt_vocab_size)
    # # 当你测试不同的模型时请使用不同的模型声明方法
    # # model = make_seq2seq_model(tgt_vocab_size)
    # # model = RNN(d_model, RNN_hidden, tgt_vocab_size)
    # # model = CNN(tgt_vocab_size)
    # # model = CTC(output_dim=tgt_vocab_size)
    # # model = conformer(tgt_vocab_size)
    # # model = SSANTransformer(tgt_vocab_size)
    # model = model.to(device)
    # # 一开始使用的交叉熵损失函数
    # # criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    # ctc_loss = nn.CTCLoss(blank=0)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    # train_losses = []
    # test_losses = []
    # test_WER = []
    # for epoch in range(epoch):
    #     count = 1
    #     for enc_inputs, dec_inputs, dec_outputs in tqdm(train_loader):  # enc_inputs : [batch_size, src_len]
    #                                                         # dec_inputs : [batch_size, tgt_len]
    #                                                         # dec_outputs: [batch_size, tgt_len]

    #         enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
    #         # 根据不同的模型，输入输出也要做相应的调整
    #         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)  # Speech Transformer: outputs: [batch_size * tgt_len, tgt_vocab_size]
    #         # outputs = model(enc_inputs, dec_inputs) # seq2seq只返回最后的outputs
    #         # outputs = model(enc_inputs, torch.zeros(1, enc_inputs.shape[0], RNN_hidden), dec_outputs) # RNN
    #         # outputs = model(enc_inputs, dec_inputs) # CNN
    #         # outputs = model(enc_inputs) # CTC,conformer
    #         # outputs = model(enc_inputs, dec_inputs) # SSAN Transformer

    #         # 这是一开始使用的交叉熵损失函数
    #         # loss = criterion(outputs, dec_outputs.view(-1))
    #         '''
    #         下面这些代码是为了做CTCloss而产生的
    #         '''
    #         new_shape = (len(enc_inputs),tgt_max_len,tgt_vocab_size)
    #         outputs = outputs.view(new_shape)
    #         log_probs = torch.transpose(outputs,0,1).log_softmax(2)
    #         targets = CTC_targets_generator(dec_outputs)
    #         input_lengths = torch.full((len(enc_inputs),), tgt_max_len, dtype=torch.long)
    #         targets_lengths = CTC_targets_len_generator(dec_outputs)
    #         loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)

    #         train_losses.append(loss.item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         count += 1
    #         if (count % train_step == 0) and (count % test_step != 0):
    #             print('Epoch:', '%04d' % (epoch + 1), ' batch=', count, ' train_loss =', '{:.6f}'.format(loss.item()))
    #         if count % test_step == 0:
    #             model.eval()
    #             total_test_loss = 0.0
    #             WER_total = 0.0
    #             test_batch = 0
    #             with torch.no_grad():
    #                 for test_enc_inputs, test_dec_inputs, test_dec_outputs in tqdm(test_loader):
    #                     test_enc_inputs, test_dec_inputs, test_dec_outputs = test_enc_inputs.to(device), test_dec_inputs.to(device), test_dec_outputs.to(device)
    #                     # test_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(test_enc_inputs, test_dec_inputs)  # Speech Transformer: outputs: [batch_size * tgt_len, tgt_vocab_size]
    #                     test_outputs = eval_SpeechTransformer(model, test_enc_inputs, test_dec_inputs, tgt_vocab_size)
    #                     new_shape = (len(test_enc_inputs),tgt_max_len,tgt_vocab_size)
    #                     # test_outputs = test_outputs.view(new_shape)
    #                     # 这个是计算WER的值
    #                     WER_total += calWER(test_outputs, test_dec_outputs, reflection)
    #                     '''
    #                     下面这些也是计算CTCloss
    #                     '''
    #                     log_probs = torch.transpose(test_outputs,0,1).log_softmax(2)
    #                     targets = CTC_targets_generator(test_dec_outputs)
    #                     input_lengths = torch.full((len(test_enc_inputs),), tgt_max_len, dtype=torch.long)
    #                     targets_lengths = CTC_targets_len_generator(test_dec_outputs)
    #                     test_loss = ctc_loss(log_probs, targets, input_lengths, targets_lengths)
    #                     test_batch += 1
    #                     total_test_loss += test_loss.item()
    #             print()
    #             print('Epoch:', '%04d' % (epoch + 1), ' batch=', count, ' train_loss =', '{:.6f}'.format(loss.item()),' test_loss = ','{:.6f}'.format(total_test_loss/test_batch),' test_WER = ','{:.6f}'.format(WER_total/test_batch))
    #             print('==========================================================================================================')
    #             test_losses.append(total_test_loss/test_batch)
    #             test_WER.append(WER_total/test_batch)
    #             model.train()

    # torch.save(model.state_dict(), 'model/SpeechTransformer/model2.pt')
    # save_dic(reflection, 'model/SpeechTransformer/reflection2.txt')
    # # print("正在保存模型及数据")
    # np.savetxt('result/SpeechTransformer/train_loss2.txt', np.array(train_losses), fmt="%.6f")
    # np.savetxt('result/SpeechTransformer/test_loss2.txt', np.array(test_losses), fmt="%.6f")
    # np.savetxt('result/SpeechTransformer/test_WER2.txt', np.array(test_WER), fmt="%.6f")