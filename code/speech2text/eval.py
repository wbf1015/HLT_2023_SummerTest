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

def generate2dic(path):
    word_index_dic = load_dic(path)
    index_word_dic = {value: key for key, value in word_index_dic.items()}
    return word_index_dic,index_word_dic

def translate(dic,predict,groundtruth):
    ret_predict = ''
    ret_groundtruth = ''
    for i in range(0,len(predict)):
        ret_predict.append(dic[predict[i]])
    for i in range(0,len(groundtruth)):
        ret_predict.append(dic[groundtruth[i]])
    return ret_predict,ret_groundtruth


def eval_SpeechTransformer():
    model_path = 'model/SpeechTransformermodel.pt'
    dic_path = 'model/SpeechTransformer_Reflection.txt'
    word_index_dic,index_word_dic = generate2dic(dic_path)
    model = Transformer(len(word_index_dic)).to(device)
    model.load_state_dict(torch.load(model_path))
    # TODO 这个必须弄成三维的才能做
    enc_inputs, dec_inputs, dec_outputs = make_Testdata(word_index_dic)
    for i in range(0,len(enc_inputs)):
        enc_inputs,dec_inputs = enc_inputs.to(device),dec_inputs.to(device)
        outputs = model(enc_inputs[i], dec_inputs[i])
        predict,groundtruth = translate(outputs,dec_outpus[i])
        print(predict,groundtruth)


if __name__ == "__main__":
    eval_SpeechTransformer()
