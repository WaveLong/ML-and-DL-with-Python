#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
# 输入变量：x、y、c：常数c、toler：容错率、maxIter：最大循环次数

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i, m):
    j = i
    while(j == i):
        j =  int(np.random.uniform(0, m))
    return j
def clipAlpha(x, H, L):
    if x > H :
        x = H
    elif x < L:
        x = L
    return x

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #dataMatIn, classLabels, C, toler, maxIter=dataArr,lableArr,0.6,0.001,40
    dataMatrix = np.mat(dataMatIn)             # 数据x转换为matrix类型
    labelMat = np.mat(classLabels).transpose() # 标签y转换为matrix类型，转换为一列
    b = 0                                      # 截距b
    m,n = np.shape(dataMatrix)                 # 数据x行数、列数
    alphas = np.mat(np.zeros((m,1)))           # 初始化alpha，有多少行数据就产生多少个alpha
    iter = 0                                   # 遍历计数器
    while (iter < maxIter):
        #print( "iteration number: %d" % iter)
        alphaPairsChanged = 0                  # 记录alpha是否已被优化，每次循环都重置
        for i in range(m):                     # 按行遍历数据，类似随机梯度下降
            # i=0
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b # 预测值y，g(x)函数，《统计学习方法》李航P127，7.104
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions  # 误差，Ei函数，P127，7.105
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 找第一个alphas[i]，找到第一个满足判断条件的，判断负间隔or正间隔，并且保证0<alphas<C
                j = selectJrand(i,m)            # 随机找到第二个alphas[j]
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b # 计算预测值
                Ej = fXj - float(labelMat[j])   # 计算alphas[j]误差
                alphaIold = alphas[i].copy()    # 记录上一次alphas[i]值
                alphaJold = alphas[j].copy()    # 记录上一次alphas[j]值
                if (labelMat[i] != labelMat[j]):# 计算H及L值，《统计学习方法》李航，P126
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    #print( "L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # 《统计学习方法》李航P127，7.107，这里的eta与李航的一致，这里乘了负号
                if eta >= 0:
                    #print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta     # 《统计学习方法》李航P127，7.107，更新alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L)       # alphas[j]调整大于H或小于L的alpha值
                if (abs(alphas[j] - alphaJold) < 0.00001): # 调整后过小，则不更新alphas[i]
                    #print( "j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) #更新alphas[i]，《统计学习方法》李航P127，7.109
                # 更新b值，《统计学习方法》李航P130，7.115，7.116
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): # 判断符合条件的b
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                #print( "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
    return b,alphas