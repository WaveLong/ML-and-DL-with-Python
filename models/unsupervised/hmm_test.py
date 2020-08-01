# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:54:59 2020

@author: wangxin
"""
#%% 导入包
import numpy as np
from HMM import *
import matplotlib.pyplot as plt
#%% 初始化模型
A = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
    ])
B = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
    ])
pi = np.array([0.2, 0.4, 0.4])
model = Hmm(3, 2, A, B, pi)
#%% 生成指定长度的序列
seq = model.generate(10)
print(seq)
#%% 采用前向和后向算法计算指定序列的概率
seq = np.array([0, 1, 0, 0, 1,1,1,0])
res1, alpha = model.compute_prob(seq)
res2, beta = model.compute_prob(seq, 'backward')
print("res1: ", res1)
print("alpha:\n", alpha)
print("res2: ", res2)
print("beta:\n ", beta)
#%% 学习模型，使用三角波序列
def triangle_data(T):   # 生成三角波形状的序列
    data = []
    for x in range(T):
        x = x % 6
        data.append(x if x <= 3 else 6-x)
    return data

data = np.array(triangle_data(30))
hmm = Hmm(10, 4)
hmm.baum_welch(data)               # 先根据给定数据反推参数
gen_obs = hmm.generate(30)  # 再根据学习的参数生成数据
x = np.arange(30)
plt.scatter(x, gen_obs, marker='*', color='r')
plt.plot(x, data, color='c')
plt.show()
#%% 预测指定观测序列的可能的状态序列
seq = np.array([0, 1, 0], dtype = np.int32)
print(model.verbiti(seq))




