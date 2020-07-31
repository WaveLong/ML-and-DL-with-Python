#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-

import numpy as np
# 裁剪函数
def clip(x, L, H):
    if(x < L):
        x = L
    elif(x > H):
        x = H
    return x
# 随机选择数字
def rand_select(i, m):
    j = i
    while(j == i): j = int(np.random.uniform(0, m))
    return j

class SVM():
    def __init__(self, kernel="linear", C = 1.0, sigma=1.0, **kwargs):
        # 只支持线性核和高斯核
        if (kernel not in ['linear', 'gaussian']):
            raise ValueError("Now only support linear and gaussian kernel")
        elif kernel == "linear":
            kernel_fn = Kernel.linear()
        elif kernel == "gaussian":
            kernel_fn = Kernel.gaussian(sigma)

        self.kernel_method = kernel
        self.kernel = kernel_fn

        self.sigma = sigma
        self.C = C

        self.w , self.b = None, None
        self.n_sv = -1
        self.sv_x, self.sv_y, self.alphas = None, None, None
        self.eps = 1e-6

    # 预测函数， y = \sum_(i in S) alpha_i y_i K(x_i, x) + b
    def predict(self, x):
        wx = self.b
        for i in range(len(self.sv_x)):
            wx += self.alphas[i] * self.sv_y[i] * self.kernel(self.sv_x[i], x)
        return wx

    '''
    svm训练，常见的方法有以下几种:
    1. simple_smo
    2. smo
    3. cvopt
    4. hinge loss
    目前只实现了simple_smo
    '''
    def train(self, X, y, maxIter = 500):
        # 计算核矩阵
        if(self.kernel_method == "linear"):
            K = np.dot(X, X.T)
        elif (self.kernel_method == "gaussian"):
            d2 = np.sum(X ** 2, axis = -1, keepdims= True) - 2 * np.dot(X, X.T) + np.sum(X ** 2, axis = -1,)
            K = np.exp(- 0.5 * d2 / self.sigma)

        # 优化
        alphas, b = self._SMO(K, y, maxIter = maxIter)

        # 计算支持向量
        idx = [i for i in range(len(alphas)) if alphas[i] > self.eps]
        self.n_sv = len(idx)
        self.sv_x = X[idx,:]
        self.alphas = alphas[idx]
        if self.kernel_method == "linear":
            self.w = np.sum((self.alphas * y[idx]).reshape(-1, 1) * X[idx,:], axis = 0)
        self.b = b

    def _SMO(self, K, y, tol = 0.001, maxIter = 40):
        print("begin SMO...")
        m = len(y)
        self.alphas, self.b = np.zeros(m), 0.
        y = y.reshape(1, -1)

        niter = 0
        while (niter < maxIter):
            # select alpha_i, alpha_j
            changed = 0
            for i in range(m):
                # 在外层循环中，选择一个违反了KKT条件的alpha_i
                yi, ai_old = y[0][i], self.alphas[i].copy()
                Ei = np.dot(self.alphas * y , K[i].T) + self.b - yi
                if ((yi * Ei < -tol) and (ai_old < self.C)) or ((yi * Ei > tol) and (ai_old > 0)):
                    # 随机找内层的alpha_j
                    j = rand_select(i, m)
                    yj, aj_old = y[0][j], self.alphas[j].copy()
                    Ej = np.dot(self.alphas * y, K[j].T) + self.b - yj

                    # 更新 alpha_j
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if(eta <= 0): continue;
                    aj_new = aj_old + yj * (Ei - Ej) / eta

                    # 裁剪 alpha_j
                    if yi != yj:
                        L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0, aj_old + ai_old - self.C), min(self.C, aj_old + ai_old)
                    if H - L == 0:
                        continue
                    aj_new = clip(aj_new, L, H)

                    # 更新alpha_i
                    s = yi * yj
                    delta_j = aj_new - aj_old
                    if(abs(delta_j) < self.eps): continue;
                    ai_new = ai_old - s * delta_j
                    delta_i = ai_new - ai_old

                    # 更新b
                    bi = self.b - Ei - yi * delta_i * K[i, i] - yj * delta_j * K[i, j]
                    bj = self.b - Ej - yi * delta_i * K[i, j] - yj * delta_j * K[j, j]

                    if 0 < ai_new < self.C:
                        self.b = bi
                    elif 0 < aj_new < self.C:
                        self.b = bj
                    else:
                        self.b = (bi + bj) / 2
                    self.alphas[i], self.alphas[j] = ai_new, aj_new
                    changed += 1
            if (changed == 0):
                niter +=1;
            else:
                niter = 0

        print("Finish SMO...")
        return self.alphas, self.b

'''
核函数的单独实现，后期可以在里面添加
'''
class Kernel(object):
    # 线性核
    @staticmethod
    def linear():
        return lambda X, y: np.inner(X, y)

    # 高斯核
    @staticmethod
    def gaussian(sigma):
        return lambda X, y: np.exp(-np.sqrt(np.linalg.norm(X - y) ** 2 / (2 * sigma)))