import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
# 适应度变化曲线图
def plot_Output():
    data = np.loadtxt('output.dat')
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Max fitness of each epoch')
    # plt.ylabel('w for Map')
    plt.show()

# 找到适应度最大的对应的map
def getMax(maxMap):
    f1 = np.loadtxt('output.dat')
    # f2 = np.loadtxt('map result.dat')
    maxFitnessIdx = np.argmax(f1[:,1])
    maxFitnessMap = np.sort(maxMap[maxFitnessIdx])
    print(f"本轮迭代中最大适应度值为{f1[maxFitnessIdx,1]}")
    # maxFitnessMap = f2[maxFitnessIdx]
    return maxFitnessMap

# 最终结果展示散点图
def plotCurve(y,maxMap):
    x = np.array([0,5,20,40,90,180,360,720,1000])
    y_real = [0,52.57091522,115.3861542,136.0529785,167.4309082,175.6696472,181.173645,193.6758118,211.2645264]
    z = getMax(maxMap)
    # n=3
    # map_fit = np.polyfit(x, y, n)
    # yvals = np.polyval(map_fit, x)
    plt.plot(x, y, '*',c="m",label = "optimized",alpha = 1)
    plt.plot(x,y_real,"o",label = "reference",alpha = 0.5)
    plt.plot(x,z,"s",c="green",label="maxFitness",alpha = 0.2)
    # plt.plot(x, yvals, 'r')
    plt.legend(loc=4)
    plt.xlabel('angles')
    plt.ylabel('target w')
    plt.title("angle-w curves(GA)")
    plt.show()
    return z
def plotScatter1(y):
    x = np.array([0,5,20,40,90,180,360,720,1000])
    y_real = [0,52.57091522,115.3861542,136.0529785,167.4309082,175.6696472,181.173645,193.6758118,211.2645264]
    plt.plot(x, y, '*',c="m",label = "optimized",alpha = 1)
    plt.plot(x,y_real,"o",label = "reference",alpha = 0.5)
    plt.legend(loc=4)
    plt.xlabel('angles')
    plt.ylabel('target w')
    plt.title("angle-w curves(GA)")
    plt.show()
def plotScatter2(maxMap):
    x = np.array([0,5,20,40,90,180,360,720,1000])
    y_real = [0,52.57091522,115.3861542,136.0529785,167.4309082,175.6696472,181.173645,193.6758118,211.2645264]
    z = getMax(maxMap)
    plt.plot(x,y_real,"o",c="b",label = "reference",alpha = 0.6)
    plt.plot(x,z,"+",c="r",label="maxFitness",alpha = 1)
    plt.legend(loc=4)
    plt.xlabel('angles')
    plt.ylabel('target w')
    plt.title("angle-w curves(GA)")
    plt.show()
# import plotly.graph_objects as go
# from typing import List
# def draw_table(headers: List[str], cells: List[list]):
#     """
#     画表
#     :param headers: header=dict(values=['A Scores', 'B Scores'])
#     :param cells: cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]])
#     """
#     fig = go.Figure(data=[go.Table(header=dict(values=headers), cells=dict(values=cells))])
#     fig.show()

def assist_main(map_params,maxMap):
    # 对优化后的结果进行排序以符合map参数定义

    #print(map_params[np.argsort(map_params)])
    plot_Output()  # 适应度曲线
    map_cal = map_params[np.argsort(map_params)]

    # draw_table(["angles","map param"],[[0,5,20,40,90,180,360,720,1000],np.round(map_cal.tolist(),4)])
    maxFmap = plotCurve(map_cal,maxMap)
    plotScatter1(map_cal)
    plotScatter2(maxMap)

    return map_cal,maxFmap