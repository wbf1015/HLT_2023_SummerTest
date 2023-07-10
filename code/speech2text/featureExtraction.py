import librosa
import os
import numpy as np
import torch
from config import *

'''
输入：一个路径
输出：一个字典，key是文件名，value是对应文件的mfcc特征信息
'''


def mfcc_process(path):
    all_files = [f for f in os.listdir(path)]
    ret = {}
    print('正在从' + path + '提取音频信息特征...')
    for f in all_files:
        x, sr = librosa.load(path + '\\' + f)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
        # print(type(mfcc))
        # print(str(f)[:-4])
        ret[str(f)[:-4]] = mfcc
    return ret


'''
输入：一个路径
输出：一个列表，返回对应路径下每个文件的mfcc特征信息
'''


def mfcc_process_list(path):
    all_files = [f for f in os.listdir(path)]
    ret = []
    print('正在从' + path + '提取音频信息特征...')
    for f in all_files:
        x, sr = librosa.load(path + '\\' + f)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
        # print(type(mfcc))
        # print(str(f)[:-4])
        ret.append(mfcc)
    return ret


'''
输入：一个列表，列表中的元素时这个路径下每一个音频文件的mfcc特征信息
输出：一个三维tensor，表示[音频数量，n_mfcc,最大音频特征长度]
'''


def format_mfcc_list(l):
    # 遍历每个二维列表进行维度扩充和填充
    extended_data = np.zeros((len(l), n_mfcc, feature_max_len))
    for i in range(len(l)):
        original_size = l[i].shape[1]
        extended_data[i, :, :original_size] = l[i]
    # print(extended_data.shape)
    mfcc_matrix = torch.Tensor(extended_data)
    mfcc_matrix = torch.transpose(mfcc_matrix, 1, 2)
    return mfcc_matrix
