import torch
import torch.utils.data as Data
from config import *
from featureExtraction import *
from wordVector import *


def makeData():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    ground_truth_list = []
    for train_file in train_data_file:
        # 输入频谱图，Encoder的输入
        path = train_data_path_base + train_file
        enc_inputs += mfcc_process_list(path)
        # 构建Decoder的输入
        ground_truth_list += fetchGroundTruth_list(ground_truth_path, fetchFile(path))
    reflection, tgt_vocab_size = construct_dic_with_list(ground_truth_list)
    dec_inputs = one_hot_embedding(reflection, ground_truth_list, 'decode_input')
    dec_outputs = one_hot_embedding(reflection, ground_truth_list, 'decode_output')
    dec_inputs = torch.LongTensor(dec_inputs)
    dec_outputs = torch.LongTensor(dec_outputs)
    return format_mfcc_list(enc_inputs), dec_inputs, dec_outputs, tgt_vocab_size,reflection

def check_not_exist(reflection, ground_truth_list):
    indices = []
    for i, string in enumerate(ground_truth_list):
        for char in string:
            if char not in reflection:
                indices.append(i)
                break
    return indices

def delete_list(indices, enc_inputs, ground_truth_list):
    # 将indices从大到小排序，以便正确删除列表中的元素
    indices.sort(reverse=True)

    for i in indices:
        del enc_inputs[i]
        del ground_truth_list[i]

    return enc_inputs, ground_truth_list



def make_Testdata_with_mask(reflection):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    ground_truth_list = []
    for test_file in test_data_file:
        # 输入频谱图，Encoder的输入
        path = test_data_path_base + test_file
        enc_inputs += mfcc_process_list(path)
        # 构建Decoder的输入
        ground_truth_list += fetchGroundTruth_list(ground_truth_path, fetchFile(path))

    # TODO:这里需要筛选一下ground_truth_list里面有字不在reflection的情况
    indices = check_not_exist(reflection,ground_truth_list)
    enc_inputs,ground_truth_list = delete_list(indices,enc_inputs,ground_truth_list)

    dec_inputs = one_hot_embedding(reflection, ground_truth_list, 'decode_input')
    # 我也不是很确定是不是需要这个
    deal_test_dec_inputs(dec_inputs)
    dec_outputs = one_hot_embedding(reflection, ground_truth_list, 'decode_output')
    dec_inputs = torch.LongTensor(dec_inputs)
    dec_outputs = torch.LongTensor(dec_outputs)
    return format_mfcc_list(enc_inputs), dec_inputs, dec_outputs

def make_Testdata(reflection):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    ground_truth_list = []
    for test_file in test_data_file:
        # 输入频谱图，Encoder的输入
        path = test_data_path_base + test_file
        enc_inputs += mfcc_process_list(path)
        # 构建Decoder的输入
        ground_truth_list += fetchGroundTruth_list(ground_truth_path, fetchFile(path))

    # TODO:这里需要筛选一下ground_truth_list里面有字不在reflection的情况
    indices = check_not_exist(reflection,ground_truth_list)
    enc_inputs,ground_truth_list = delete_list(indices,enc_inputs,ground_truth_list)

    dec_inputs = one_hot_embedding(reflection, ground_truth_list, 'decode_input')
    dec_outputs = one_hot_embedding(reflection, ground_truth_list, 'decode_output')
    dec_inputs = torch.LongTensor(dec_inputs)
    dec_outputs = torch.LongTensor(dec_outputs)
    return format_mfcc_list(enc_inputs), dec_inputs, dec_outputs

# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


if __name__ == "__main__":
    enc_input, dec_inputs, dec_outputs = makeData()
    print(enc_input.shape)
    print(dec_inputs.shape)
    print(dec_inputs[:10, :])
    print(dec_outputs.shape)
    print(dec_outputs[:10, :])
