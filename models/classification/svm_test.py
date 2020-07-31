#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# generate data

anchors = np.array([[-1, -1], [1, 1]]) * 0.5
n1 = np.random.random((30, 2))
n2 = np.random.random((30, 2))
X1 = anchors[0] - n1
X2 = anchors[1] + n2
y1 = np.array([-1]*30)
y2 = np.array([1]*30)

print("trianing...")
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

idx = np.array(range(len(y)))
np.random.shuffle(idx)
X = X[idx,:]
y = y[idx]

from svm import *
model = SVM(C = 0.6, kernel="linear")
model.train(X, y, maxIter = 40)

print(model.n_sv)
print(model.sv_x)
print(model.sv_y)

xx = np.array([-1, 1])
yy = -model.w[0] / model.w[1] * xx - model.b / model.w[1]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(y)):
    if y[i] == -1:
        ax.scatter(X[i][0], X[i][1], marker = "o", color = "red")
    else:
        ax.scatter(X[i][0], X[i][1], marker="x", color="green")

for sv_x in model.sv_x:
    cir1 = Circle(xy = (sv_x[0], sv_x[1]), radius=0.1, alpha=0.5)
    ax.add_patch(cir1)

ax.plot(xx, yy, color = "orange")
plt.show()




