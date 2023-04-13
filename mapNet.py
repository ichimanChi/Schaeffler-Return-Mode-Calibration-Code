from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing

import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # 第一个隐含层
        self.hidden1 = nn.Linear(9, 64,bias=True)
        # 第二个隐含层
        self.hidden2 = nn.Linear(64, 32)
        # 第三个隐含层
        self.hidden3 = nn.Linear(32, 16)
        # 回归预测层
        self.predict = nn.Linear(16, 1)

    # 定义网络前向传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        # output = 10* torch.sigmoid(self.predict(x))
        output = self.predict(x)
        # 输出一个一维向量
        # return output[:, 0]
        return output