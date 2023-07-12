import numpy as np

data = np.array([[1, 2], [3, 4]])

np.savetxt('out.txt', data, fmt="%d")  # 保存为整数

np.savetxt('out1.txt', data, fmt="%.2f", delimiter=',')  # 保存为2位小数的浮点数，用逗号分隔

with open('out.txt') as f:
    for line in f:
        print(line, end='')