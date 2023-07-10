from featureExtraction import *
from wordVector import *


def featureExtraction_test():
    path = 'G:\\code\\HLT-test\\code\\speech2text\\data\\train'
    mfcc_dic = mfcc_process(path)
    # mfcc_matrix = format_mfcc_list(mfcc_list)
    # print(mfcc_matrix.shape)
    # print(type(mfcc_matrix))
    # print(mfcc_matrix)


def wordVector_test():
    path = 'G:\\code\\HLT-test\\code\\speech2text\\data\\train'
    l = fetchFile(path)
    dic = fetchGroundTruth('G:\\code\\HLT-test\\code\\speech2text\\data\\answer\\aishell_transcript_v0.8.txt', l)
    # for k, v in dic.items():
    #     print(k, v)
    reflection = construct_dic(dic)
    for k, v in reflection.items():
        print(k, v)


def position_encoding_test():
    d_model = 512
    max_len = 10
    pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
    pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
    pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
    # print(pos_table)
    # print(pos_table.shape) 10 512 表示一个句子最多是个字 一个字用一个512维的向量表示


if __name__ == "__main__":
    # featureExtraction_test()
    position_encoding_test()
