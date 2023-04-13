import numpy as np
import math
import csv

'''
基因位测量
'''
def Measure(alpha,chromosome, popSize, chromosomeSize, qpv):
    for i in range(popSize):
        for j in range(chromosomeSize):
            if alpha < pow(qpv[i, j, 0], 2):
                chromosome[i, j] = 0
            else:
                chromosome[i, j] = 1

'''
参数生成，同时重写 map.txt 文档
'''
def map_Parameter(chromosome,dnaSize, map_BOUND):
    map_5_pop = chromosome[:, 0::8]
    map_20_pop = chromosome[:, 1::8]
    map_40_pop = chromosome[:, 2::8]
    map_90_pop = chromosome[:, 3::8]
    map_180_pop = chromosome[:, 4::8]
    map_360_pop = chromosome[:, 5::8]
    map_720_pop = chromosome[:, 6::8]
    map_1000_pop = chromosome[:, 7::8]

    map_5 = map_5_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[0][1] - map_BOUND[0][0]) + map_BOUND[0][0]
    map_20 = map_20_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[1][1] - map_BOUND[1][0]) + map_BOUND[1][0]
    map_40 = map_40_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[2][1] - map_BOUND[2][0]) + map_BOUND[2][0]
    map_90 = map_90_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[3][1] - map_BOUND[3][0]) + map_BOUND[3][0]
    map_180 = map_180_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[4][1] - map_BOUND[4][0]) + map_BOUND[4][0]
    map_360 = map_360_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[5][1] - map_BOUND[5][0]) + map_BOUND[5][0]
    map_720 = map_720_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[6][1] - map_BOUND[6][0]) + map_BOUND[6][0]
    map_1000 = map_1000_pop.dot(2 ** np.arange(dnaSize)[::-1]) / float(2 ** dnaSize - 1) * (map_BOUND[7][1] - map_BOUND[7][0]) + map_BOUND[7][0]

    return list(map_5), list(map_20), list(map_40), list(map_90), list(map_180), list(map_360), list(map_720), list(map_1000)

'''
调用评价模型评分(适应性)，同时记录最优个体
'''
def Fitness_evaluation(fitness, popSize):
    the_best_chrom = 0
    fitness_max = fitness[0]
    for i in range(popSize):
        if fitness[i] >= fitness_max:
            fitness_max = fitness[i]
            the_best_chrom = i
    return the_best_chrom

'''
螺旋进化
'''
def Rotation(chromosome, the_best_chrom,delta_theta,popSize,fitness,chromosomeSize,nqpv,qpv):
    rotate_mat_0 = np.empty([2, 2])       # 逆向螺旋矩阵
    rotate_mat_0[0, 0] = math.cos(delta_theta)
    rotate_mat_0[0, 1] = -math.sin(delta_theta)
    rotate_mat_0[1, 0] = math.sin(delta_theta)
    rotate_mat_0[1, 1] = math.cos(delta_theta)

    rotate_mat_1 = np.empty([2, 2])       # 正向螺旋矩阵
    rotate_mat_1[0, 0] = math.cos(-delta_theta)
    rotate_mat_1[0, 1] = -math.sin(-delta_theta)
    rotate_mat_1[1, 0] = math.sin(-delta_theta)
    rotate_mat_1[1, 1] = math.cos(-delta_theta)

    for i in range(popSize):
        if fitness[i] < fitness[the_best_chrom]:
            for j in range(chromosomeSize):
                if chromosome[i, j] == 0 and chromosome[the_best_chrom, j] == 1:
                    nqpv[i, j, 0] = (rotate_mat_0[0, 0] * qpv[i, j, 0]) + (rotate_mat_0[0, 1] * qpv[i, j, 1])   # a
                    nqpv[i, j, 1] = (rotate_mat_0[1, 0] * qpv[i, j, 0]) + (rotate_mat_0[1, 1] * qpv[i, j, 1])   # b
                    qpv[i, j, 0] = round(nqpv[i, j, 0], 2)  # a
                    qpv[i, j, 1] = round(1 - nqpv[i, j, 0], 2)  # b
                if chromosome[i, j] == 1 and chromosome[the_best_chrom, j] == 0:
                    nqpv[i, j, 0] = (rotate_mat_1[0, 0] * qpv[i, j, 0]) + (rotate_mat_1[0, 1] * qpv[i, j, 1])   # a
                    nqpv[i, j, 1] = (rotate_mat_1[1, 0] * qpv[i, j, 0]) + (rotate_mat_1[1, 1] * qpv[i, j, 1])   # b
                    qpv[i, j, 0] = round(nqpv[i, j, 0], 2)  # a
                    qpv[i, j, 1] = round(1 - nqpv[i, j, 0], 2)  # b

'''
基因变异
'''
def Mutation(pop_mutation_rate, mutation_rate,popSize,chromosomeSize,nqpv,qpv):
    for i in range(1, popSize):
        up = np.random.randint(100)
        up = up / 100
        if up <= pop_mutation_rate:
            for j in range(chromosomeSize):
                um = np.random.randint(100)
                um = um / 100
                if um <= mutation_rate:
                    nqpv[i, j, 0] = qpv[i, j, 1]
                    nqpv[i, j, 1] = qpv[i, j, 0]
                else:
                    nqpv[i, j, 0] = qpv[i, j, 0]
                    nqpv[i, j, 1] = qpv[i, j, 1]
        else:
            for j in range(chromosomeSize):
                nqpv[i, j, 0] = qpv[i, j, 0]
                nqpv[i, j, 1] = qpv[i, j, 1]
    for i in range(1, popSize):
        for j in range(chromosomeSize):
            qpv[i, j, 0] = nqpv[i, j, 0]
            qpv[i, j, 1] = nqpv[i, j, 1]

# 【重要】根据甲方提供的csv数据转换成nparray类型的矩阵
def getOrgMap():
    '''
    return: ndarray,(9,9), SFL提供的标定好的参数
    '''
    x = []
    with open("org_map matrix.csv",encoding="utf-8") as f:
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

def main():
    popSize = 20  # 定义种群数目
    dnaSize = 8  # 定义基因链长 共计三个参数，每个参数6位，两个量子态，交错分布
    chromosomeSize = dnaSize * 8  # 定义基因链长度
    chromosome = np.empty([popSize, chromosomeSize], dtype=int)  # chromosome: 经典染色体  量子染色体通过映射得到标准染色体

    k=int(open("Test/k.txt").read())
    map_bound = []
    gap = 40
    org_map = getOrgMap()
    original_map_v_list = list(org_map[1:,k])    # 根据用户输入的速度选择需要标定的数据位于第几列
    for i in range(8):
        map_bound.append([original_map_v_list[i]-gap,original_map_v_list[i]+gap]) # 以提供数据为基准在其上下波动
    
    top_bottom = 2
    nqpv = np.empty([popSize, chromosomeSize, top_bottom])
    delta_theta = 0.0985398163  # 螺旋系数
    alpha = 0.5    # 测量值


    # 读取 qpv_map.txt 文档生成 qpv
    qpv = np.loadtxt('Test/qpv_map.txt', delimiter=',').reshape((popSize, chromosomeSize, top_bottom))
    # 读取 fitness.txt 文档
    fitness = np.loadtxt('fitness.txt', delimiter=',')

    Measure(alpha, chromosome,popSize,chromosomeSize,qpv)
    the_best_chrom = Fitness_evaluation(fitness, popSize)
    Rotation(chromosome, the_best_chrom,delta_theta,popSize,fitness,chromosomeSize,nqpv,qpv)
    Mutation(0.01,0.01,popSize,chromosomeSize,nqpv,qpv)
    map_5, map_20, map_40, map_90, map_180, map_360, map_720, map_1000 = map_Parameter(chromosome,dnaSize,map_bound)


    # 重写 qpv_map.txt
    with open('Test/qpv_map.txt', 'w') as qpv_map:
        for scile_2d in qpv:
            np.savetxt(qpv_map, scile_2d, fmt='%f', delimiter=',')
    # 重写map_vct.csv
    MAPs = zip(map_5, map_20, map_40, map_90, map_180, map_360, map_720, map_1000)
    with open('Test/map_vct.csv', 'w', newline='') as map:
        writer = csv.writer(map)
        for map in MAPs:
            writer.writerow(map)

if __name__ == '__main__':
    main()