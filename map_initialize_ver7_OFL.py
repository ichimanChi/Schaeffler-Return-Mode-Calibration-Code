import math
import numpy as np
import csv
import copy
import pandas as pd
popSize = 20    # 定义种群数目
dnaSize = 8     # 定义基因链长 用4位存储，存储范围2^4-1=15
clb_num = 8     # 要标定的参数的个数
chromosomeSize = dnaSize*clb_num  # 定义基因链长度
top_bottom = 2


# 量子比特初始化
def Init_population(popSize, chromosomeSize, qpv):
    for i in range(popSize):
        for j in range(chromosomeSize):
            theta = np.random.uniform(0, 1) * 90
            theta = math.radians(theta)  # 角度转为弧度
            qpv[i, j, 0] = np.around(math.cos(theta), 3)    # alpha squared
            qpv[i, j, 1] = np.around(math.sin(theta), 3)    # beta squared

# 基因测量，将量子比特转换为染色体值
def Measure(alpha,chromosome,qpv):
    for i in range(popSize):
        for j in range(chromosomeSize):
            if alpha < pow(qpv[i, j, 0], 2):
                chromosome[i, j] = 0
            else:
                chromosome[i, j] = 1

# 生成参数
def map_Parameter(chromosome,map_bound):
    '''
    input: map_bound: list, 8行2列,8个标定参数,2表示上下界范围
    return: 2-D list, 8*20(dnaSize*popSize), 1st:ndarray,10 numbers
    lst[i][j] means the jth number of the ith parameter
    '''
    map_pop = []
    map_trans = []
    for i in range(clb_num):
        map_pop.append(chromosome[:, i::8])
    for i in range(clb_num):
        map_trans.append(map_pop[i].dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_bound[i][1] - map_bound[i][0]) + map_bound[i][0])
    return map_trans

# 【重要】根据甲方提供的csv数据转换成nparray类型的矩阵
def getOrgMap():
    '''
    return: ndarray,(9,9), SFL提供的标定好的参数
    '''
    x = []
    # with open("org_map matrix.csv",encoding="utf-8") as f:
    with open("Results/max_1.csv",encoding="utf-8") as f:
        map_clb = csv.reader(f)
        for row in map_clb:
            x = np.r_[x,row]
    x[0] = 0
    MAP = np.zeros([9,9])
    for i in range(MAP.shape[0]):
        for j in range(MAP.shape[1]):
            MAP[i][j] = x[j+i*9]
    map_initialize_samples = MAP  
    return map_initialize_samples

def getQpv(qpv):
    qpv_map = open('Test/qpv_map.txt', 'w')  #保存为.txt文件
    for scile_2d in qpv:
        np.savetxt(qpv_map, scile_2d, fmt='%f', delimiter=',')
    qpv_map.close()

def main():
    k=0 # map中速度那一列对应的编号
    while(True):
        #vd = int(input("请输入需要标定的速度(kph):"))
        #vd = 30
        vd = 10
        if(vd == 0):
            print("该列数据不需要标定,为0")
            break
        elif(vd == 5):
            k = 1
            break
        elif(vd == 10):
            k = 2
            break
        elif(vd == 20):
            k = 3
            break
        elif(vd == 30):
            k = 4
            break
        elif(vd == 40):
            k = 5
            break
        elif(vd == 50):
            k = 6
            break
        elif(vd == 60):
            k = 7
            break
        elif(vd == 80):
            k = 8
            break
        else:
            print("输入有误，请重新输入")
    print(f"您输入的速度为:{vd}km/h,对应map中的第{k}列")
    f = open("Test/k.txt",'w')
    f.write(str(k))
    f.close()

    map_bound = []
    org_map = getOrgMap()
    original_map_v_list = list(org_map[1:,k])    # 根据用户输入的速度选择需要标定的数据位于第几列
    gap = 40
    for i in range(clb_num):
        map_bound.append([original_map_v_list[i]-gap,original_map_v_list[i]+gap]) # 以提供数据为基准在其上下波动
    # print(map_bound)


    qpv = np.empty([popSize, chromosomeSize, top_bottom])  # qpv: 量子染色体 (or 群体向量, QPV)
    chromosome = np.empty([popSize, chromosomeSize], dtype=int)  # chromosome: 经典染色体  量子染色体通过映射得到标准染色体
    Init_population(popSize, chromosomeSize, qpv)   # ->qpv
    Measure(0.5, chromosome,qpv)                    # ->chromosome
    map_trans = map_Parameter(chromosome,map_bound)           # (8,20)    [8*i+1:8(i+1)] = [:,i]
    
    orgMap = getOrgMap()
    map_initialize_samples = orgMap
    '''    生成19个候选矩阵    '''
    for i in range(1,popSize):
        org_copy = copy.deepcopy(orgMap)
        org_copy[1:,2] = np.array(map_trans)[:,i]
        map_initialize_samples = np.r_[map_initialize_samples,org_copy]
    # print(map_initialize_samples.shape)     # ndarray, (180,9)
    # print(qpv.shape)
    df = pd.DataFrame(map_initialize_samples)   #将矩阵样本保存为.csv文件
    df.to_csv("Test/map.csv",index= False, header= False)     # ndarray, (180,9)
    getQpv(qpv)
    
if __name__ == '__main__':
    main()