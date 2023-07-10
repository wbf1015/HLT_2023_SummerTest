'''
处理音频数据时用到的一些参数
'''
feature_max_len = 512  # 音频特征在时间上的参数
n_mfcc = 32  # 音频特征的采样率

'''
构建Transformer时用到的一些参数
'''
d_model = 32  # 字 Embedding 的维度
d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8

'''
构建数据集时需要涉及到的参数
'''
train_data_path_base = 'G:\\code\\HLT-test\\code\\speech2text\\data\\train\\'
train_data_file = ['S0002', 'S0003', 'S0004']
ground_truth_path = 'G:\\code\\HLT-test\\code\\speech2text\\data\\answer\\aishell_transcript_v0.8.txt'

'''
训练时可能用到的参数
'''
batch_size = 2
