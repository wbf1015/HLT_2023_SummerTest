'''
处理音频数据时用到的一些参数
'''
feature_max_len = 520  # 音频特征在时间上的参数
n_mfcc = d_model = 128  # 音频特征的采样率 和特征向量的大小
tgt_max_len = 30

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
train_data_path_base = 'G:\\code\\HLT-test\\code\\speech2text\\data\\train\\'
train_data_file = []
start = 2
end = 3
for i in range(start, end):
    train_data_file.append('S000'+str(i))
ground_truth_path = 'G:\\code\\HLT-test\\code\\speech2text\\data\\answer\\aishell_transcript_v0.8.txt'

'''
训练时可能用到的参数
'''
batch_size = 16
epoch = 20
