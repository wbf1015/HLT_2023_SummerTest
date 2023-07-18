from config import *
from utils import *
import os
import random

'''
输入：一个文件路径
输出：一个列表，包含该路径下所有文件的文件名
'''


def fetchFile(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    l = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            # print(str(file))
            l.append(str(file))
    return l


'''
输入：ground truth的路径，这次要寻找的文件名列表
输出：一个字典，key是文件名，value是对应的列表
'''


def fetchGroundTruth(path, filelist):
    f = open(path, encoding='utf-8')
    groundTruth = {}
    for line in f:
        s = line.strip()
        key = s[:16]
        value = s[16:]
        value = value.replace(' ', '')
        groundTruth[key] = value
    ret = {}
    for file in filelist:
        ret[file[:-4]] = groundTruth[file[:-4]]
    return ret


'''
输入：ground truth的路径，这次要寻找的文件名列表
输出：一个字典，key是文件名，value是对应的列表
'''


def fetchGroundTruth_list(path, filelist):
    f = open(path, encoding='utf-8')
    groundTruth = {}
    for line in f:
        s = line.strip()
        key = s[:16]
        value = s[16:]
        value = value.replace(' ', '')
        groundTruth[key] = value
    ret = []
    for file in filelist:
        ret.append(groundTruth[file[:-4]])
    return ret


'''
输入：一个列表，是fetchGroundTruth_list函数的返回值
构造一个字典，这个字典包含句子里所有的字，相当于是一个映射关系：单词-index
'''


def construct_dic_with_list(sentences):
    reflection = {'P': 0, 'S': 1, 'E': 2}  # P是padding的符号，为了后面做mask方便所以这里搞成0了 S是开始符号，E是结束符号
    words = []
    count = 3
    for sentence in sentences:
        for word in sentence:
            if word in words:
                continue
            else:
                words.append(word)
    '''
    这里打乱字典应该是很有用的
    '''
    random.shuffle(words)
    for word in words:
        reflection[word] = count
        count += 1
    return reflection, len(words) + 3


'''
输入：一个字典，是fetchGroundTruth函数的返回值
构造一个字典，这个字典包含句子里所有的字，相当于是一个映射关系：单词-index
'''


def construct_dic(dic):
    reflection = {'P': 0, 'S': 1, 'E': 2}  # P是padding的符号，为了后面做mask方便所以这里搞成0了 S是开始符号，E是结束符号
    words = []
    count = 3
    for k, v in dic.items():
        for word in v:
            if word in words:
                continue
            else:
                words.append(word)
                reflection[word] = count
                count += 1
    return reflection


def one_hot_embedding(reflection, sentences, special=None):
    ret = []
    max_len = tgt_max_len
    for sentence in sentences:
        temp = []
        for word in sentence:
            temp.append(reflection[word])
        for _ in range(len(sentence), max_len-1):
            temp.append(reflection['P'])
        if special == 'decode_input':
            temp = [reflection['S']] + temp
        if special == 'decode_output':
            temp = temp + [reflection['E']]
        if special is None:
            pass
        ret.append(temp)
    return ret

def save_dic(dic, path):
    file = open(path, 'w')
    for k,v in dic.items():
        file.write(str(k)+' '+str(v)+'\n')
    file.close()

def load_dic(path):
    dic = {}
    with open(path, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            items = line.strip().split(' ')
            key = items[0]
            values = int(items[1])
            dic[key] = values
    return dic


if __name__ == "__main__":
    pass
