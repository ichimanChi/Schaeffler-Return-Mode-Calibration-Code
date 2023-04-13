import math
import numpy as np
import csv
import random
import pandas as pd
import copy
import codecs

def insert_vct(idx,vct,mtx):
    '''
    输入:
    idx:需要改变哪一列
    vct:插入该列的数据
    mtx:9x9的初始化矩阵

    输出: 9x9矩阵
    '''
    mtx[:,idx]=vct
    return mtx

def main():
    # 读取原矩阵
    x = []
    idx = 2
    with open("Test/map.csv",encoding="utf-8") as f:
        map_clb = csv.reader(f)
        for row in map_clb:
            x = np.r_[x,row]
    x[0] = 0
    MAP = np.zeros([180,9])
    for i in range(MAP.shape[0]):
        for j in range(MAP.shape[1]):
            MAP[i][j] = x[j+i*9]    #(180,9)

    # 读取20组vector
    v = []
    with open("Test/map_vct.csv",encoding="utf-8") as f:
        map_vct = csv.reader(f)
        for row in map_vct:
            v = np.r_[v,row]    #(160,)

    # 创建(20，9)的矩阵后将矩阵拉成行向量，再将行向量转置成列向量后插入第3列10kph
    map_vector = np.zeros([20,9])
    for i in range(map_vector.shape[0]):
        for j in range(1,map_vector.shape[1]):
            map_vector[i][j] = v[(j-1)+i*8]
    map_vector = np.reshape(map_vector,-1)
    MAP = insert_vct(idx,map_vector,MAP)
    # 保存
    df = pd.DataFrame(MAP)   #将矩阵样本保存为.csv文件
    df.to_csv("Test/map.csv",index= False, header= False)     # ndarray, (180,9)

if __name__ == "__main__":
    main()