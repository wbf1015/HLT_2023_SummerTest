import numpy as np
import pandas
import torch
import torch.nn.functional as F
import Levenshtein

def get_max_len(sentences):
    max_len = 0
    for sentence in sentences:
        if len(sentence) > max_len:
            max_len = len(sentence)

    return max_len

def CTC_targets_generator(dec_outputs, padding=0):
    mask = (dec_outputs != padding)  # 创建一个布尔掩码，指示哪些位置的值不是padding
    filtered_dec_outputs = dec_outputs[mask]  # 使用掩码过滤dec_outputs的值
    return filtered_dec_outputs

def CTC_targets_len_generator(dec_outputs, padding=0):
    non_padding_counts = torch.sum(dec_outputs != padding, dim=1)  # 统计每行与padding不同的元素数量
    return non_padding_counts

def reverse_dic(word_index_dic):
    word_index_dic = word_index_dic
    index_word_dic = {value: key for key, value in word_index_dic.items()}
    return word_index_dic,index_word_dic

def sentence_WER(predict_sentence, groundtruth_sentence, index_word_dic):
    predict = ''
    groundtruth = ''
    # 如果你的实验环境没有cuda,可以把.cpu()删掉
    for word in predict_sentence.cpu().numpy():
        if 0<=word<=2:
            pass
        else:
            predict += index_word_dic[word]
    for word in groundtruth_sentence.cpu().numpy():
        if 0<=word<=2:
            pass
        else:
            groundtruth += index_word_dic[word]
    
    reference = groundtruth.split()
    hypothesis = predict.split()

    wer = Levenshtein.distance(" ".join(reference), " ".join(hypothesis)) / len(reference)
    return wer

def calWER(predict,groundtruth,word_index_dic):
    # 对 predict 张量进行 softmax 操作
    predict_softmax = F.softmax(predict, dim=2)

    # 提取概率最大的字的索引
    predict = torch.argmax(predict_softmax, dim=2)

    word_index_dic,index_word_dic = reverse_dic(word_index_dic)

    WER_total = 0.0
    count = 0
    for batch in range(0,len(predict)):
        predict_sentence = predict[batch]
        groundtruth_sentence = groundtruth[batch]
        WER_total += sentence_WER(predict_sentence, groundtruth_sentence, index_word_dic)
        count += 1
    return (WER_total/count)



