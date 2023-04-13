import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.io import savemat
import GA_funcs
import Assist_funcs
import time
'''-------------------- 该文件为主函数 -------------------------'''
def showResult():
    # 获得初始map
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
    initMap = MAP 
    # 进行优化
    generations = 20    # 迭代次数 200和500的max无差别，123轮达到收敛
    map_params, maxMap = GA_funcs.GA_main(initMap,generations)
    # 展示优化结果
    map_cal, maxFmap = Assist_funcs.assist_main(map_params,maxMap)

    # initMap[:,2] = map_cal
    # np.savetxt("Results/last_500.csv",initMap,delimiter = ',')
    # initMap[:,2] = maxFmap
    # np.savetxt("Results/max_500.csv",initMap,delimiter = ',')

def main():
    start = time.clock()
    showResult()
    end = time.clock()
    print("运行时间：", end-start)

if __name__ == "__main__":
    main()