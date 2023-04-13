import numpy as np
import torch
from scipy.io import savemat
# (CROSSOVER_RATE, MUTATION_RATE, randomSeed),(0.8,0.15,20),(0.75,0.1,20)
DNA_SIZE = 8    #设map_params为8位精度
POP_SIZE = 20  #种群数量,撒点数
CROSSOVER_RATE = 0.75    #交叉率
MUTATION_RATE = 0.1   #变异率
X_BOUND = [0, 250]       #自变量1范围
np.random.seed(20)
#十进制转二进制
def decimal2binary(a):
    '''
    input: a: scalar
    output: m: np.ndarray (8,)
    '''
    m=np.zeros([8,1])
    i=0
    while a>0:
        m[i] = a%2  #a对2求余，添加到字符串m最后
        a=a//2
        i+=1
    # print(m[::-1])   #反向输出
    m=np.reshape(m,-1)[::-1]
    return m
#计算适应度,此处使用函数值作为适应度值
def get_fitness(pop): 
    pred_sum = np.zeros([POP_SIZE,1])
    map1,map2,map3,map4,map5,map6,map7,map8 = translateDNA(pop)
    # map_params = np.array([0,map1,map2,map3,map4,map5,map6,map7,map8])
    for i in range(POP_SIZE):
        map_params = np.sort(np.array([0,map1[i],map2[i],map3[i],map4[i],map5[i],map6[i],map7[i],map8[i]]))
        map_params_ts = torch.tensor(map_params,dtype=torch.float32)
        mapModel = torch.load('./map_model.pth')
        pred = mapModel(map_params_ts)
        pred_np = pred.detach().numpy()
        pred_sum[i] = pred_np
    # return (pred_sum - np.min(pred_sum)) + 1e-3
    return pred_sum

#把二进制转换为十进制
def translateDNA(pop): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    map1_pop = pop[:,::8]
    map2_pop = pop[:,1::8]
    map3_pop = pop[:,2::8]
    map4_pop = pop[:,3::8]
    map5_pop = pop[:,4::8]
    map6_pop = pop[:,5::8]
    map7_pop = pop[:,6::8]
    map8_pop = pop[:,7::8]
    #此处为交叉存储map params的DNA
    
	#pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    map1 = map1_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    # map1 = map1_pop.dot(2**np.arange(DNA_SIZE)[::-1])
    # map1 = (X_BOUND[1]-X_BOUND[0])*(map1-np.min(map1))/(np.max(map1)-np.min(map1))+X_BOUND[0]
    # 两种方法精度不同
    map2 = map2_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map3 = map3_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map4 = map4_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map5 = map5_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map6 = map6_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map7 = map7_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    map8 = map8_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    return map1,map2,map3,map4,map5,map6,map7,map8

def crossover_and_mutation(pop, CROSSOVER_RATE):
	new_pop = []
	for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
		child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
		if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
			mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
			cross_points = np.random.randint(low=0, high=DNA_SIZE*2)	#随机产生交叉的点
			child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
		mutation(child,MUTATION_RATE)	#每个后代有一定的机率发生变异
		new_pop.append(child)

	return new_pop

def mutation(child, MUTATION_RATE):
	if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
		mutate_point = np.random.randint(0, DNA_SIZE*2)	#随机产生一个实数，代表要变异基因的位置
		child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def select(pop, fitness):    # nature selection wrt pop's fitness
    p = (fitness)/(fitness.sum()).tolist()
    P = [i for list in p for i in list]
    # print(P)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=P)
    return pop[idx]

def GA_main(initMap,N_GENERATIONS):
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*8)) #matrix (POP_SIZE, DNA_SIZE) (20,64)
    for i in range(initMap.shape[0]):
        initMap[i,2] = round(initMap[i,2],0)
    for i in range(DNA_SIZE):
        pop[1,8*i:8*(i+1)] = decimal2binary(initMap[i+1,2])

    fitness1 = 0
    cnt = 0
    f = open('output.dat', 'w')
    f.close()
    for i in range(N_GENERATIONS):#迭代N代
        print((f"现在是第{i+1}代:"))
        fitness = get_fitness(pop)
        pop = select(pop, fitness) #选择生成新的种群
        idx = np.argmax(fitness)
        print(f"该代种群的最大适应度 = {fitness[idx]}")
        fitness2 = fitness[idx]

        f = open("output.dat", "a")
        f.write(str(i) + " " + str(fitness2[0])+"\n")
        f.write("\n")
        f.close()

        map1,map2,map3,map4,map5,map6,map7,map8 = translateDNA(pop)     # ndarray,(20,)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        
        # fitness = get_fitness(pop)
        # pop = select(pop, fitness) #选择生成新的种群
        # idx = np.argmax(fitness)
        # print(f"最大适应度 = {fitness[idx]}")
        # fitness2 = fitness[idx]

        # f = open("output.dat", "a")
        # f.write(str(i) + " " + str(fitness2[0])+"\n")
        # f.write("\n")
        # f.close()

        f = open("map result.dat", "a")
        map_params_str1 = str(np.array([0,map1[idx],map2[idx],
                                map3[idx],map4[idx],map5[idx],
                                map6[idx],map7[idx],map8[idx]]))
        # map_params_str2 = map_params_str1.replace('[','')
        # map_params_str3 = map_params_str2.replace(']','')
        f.write(str(i) + " " + str(map_params_str1)+"\n")
        f.write("\n")
        f.close()

        maxMap = np.array([0,map1[idx],map2[idx],
                                map3[idx],map4[idx],map5[idx],
                                map6[idx],map7[idx],map8[idx]])
        if i==0:
            temp = maxMap
        if i!=0:
            temp = np.row_stack((temp,maxMap))
        # print(temp.shape)
        # print(maxMap.shape)

        # if abs(fitness1 - fitness2) < 1e-10 :
        #     cnt+=1
        #     if cnt == 5:
        #         print(f"迭代至第{i+1}轮时由于适应度值变化不大退出优化过程!")
        #         break
        # else:
        #     fitness1 = fitness2
    max_fitness_index = np.argmax(fitness)
    # print("max_fitness:", fitness[max_fitness_index])
    map1,map2,map3,map4,map5,map6,map7,map8 = translateDNA(pop)
    #print("最优的基因型：", pop[max_fitness_index])
    map_params = np.array([0,map1[max_fitness_index],map2[max_fitness_index],
                                map3[max_fitness_index],map4[max_fitness_index],map5[max_fitness_index],
                                map6[max_fitness_index],map7[max_fitness_index],map8[max_fitness_index]])
    # print(map_params)
    maxMap = temp
    return map_params,maxMap
