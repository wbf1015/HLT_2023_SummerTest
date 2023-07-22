import torch
'''
处理音频数据时用到的一些参数
'''
feature_max_len = 600  # 音频特征在时间上的参数
n_mfcc = d_model = 128  # 音频特征的采样率 和特征向量的大小
tgt_max_len = 40

'''
构建Transformer时用到的一些参数
注意 n_mfcc必须等于d_model
'''
d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8

output_channel = 1
kernel_size = 3
stride = 1
padding = 1

'''
构建数据集时需要涉及到的参数
'''
# 构建训练集
train_data_path_base = '/codes/data/train/'
train_data_file = []
start = 2
end = 3
for i in range(start, end):
    if i < 10:
        train_data_file.append('S000' + str(i))
    if 10 <= i < 100:
        train_data_file.append('S00' + str(i))
    if 100 <= i < 1000:
        train_data_file.append('S0' + str(i))

# 构建测试集
test_data_path_base = '/codes/data/test/'
test_data_file = []
start = 915
end = 917
for i in range(start, end):
    if i < 10:
        test_data_file.append('S000' + str(i))
    if 10 <= i < 100:
        test_data_file.append('S00' + str(i))
    if 100 <= i < 1000:
        test_data_file.append('S0' + str(i))

# ground——truth的路径
ground_truth_path = '/codes/data/answer/aishell_transcript_v0.8.txt'


'''
训练时可能用到的参数
'''
batch_size = 16
epoch = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_step = 10
test_step = 100
lr = 1e-3