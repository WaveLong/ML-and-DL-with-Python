# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:15:22 2020

@author: wangxin
"""

from knn import *
from time import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
x = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size =0.2) #2/8分

clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree') #实例化  #里面的参数：超参数
clf1 = kNN(5, "scan")
clf2 = kNN(5, "kd-tree")
clf3 = kNN(5, "vector")

clf.fit(Xtrain, Ytrain)
clf1.train(Xtrain, Ytrain)
clf2.train(Xtrain, Ytrain)
clf3.train(Xtrain, Ytrain)

t0 = time()
y_pred = clf.predict(Xtest)
t1 = time()
y_pred1 = clf1.test(Xtest)
t2 = time()
y_pred2 = clf2.test(Xtest)
t3 = time()
y_pred3 = clf3.test(Xtest)
t4 = time()

def acc(y, y_pred):
    return np.sum(y == y_pred) / len(y)

acc0 = acc(Ytest, y_pred)
acc1 = acc(Ytest, y_pred1)
acc2 = acc(Ytest, y_pred2)
acc3 = acc(Ytest, y_pred3)

print("sklearn: {0}, {1}s".format(acc0, t1-t0))
print("scan: {0}, {1}s".format(acc1, t2-t1))
print("kd-tree: {0}, {1}s".format(acc2, t3-t2))
print("vector: {0}, {1}s".format(acc3, t4-t3))

