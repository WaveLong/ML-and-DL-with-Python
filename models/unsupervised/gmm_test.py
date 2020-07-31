#%% 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from GMM import *
#%% 生成数据
num1, mu1, var1 = 400, [0.5, 0.5], [1, 3]
X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)

num2, mu2, var2 = 600, [5.5, 2.5], [2, 2]
X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)

num3, mu3, var3 = 1000, [1, 7], [6, 2]
X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)

X = np.vstack((X1, X2, X3))

#%% 可视化数据点
plt.figure(figsize=(10, 8))
plt.axis([-10, 15, -5, 15])
plt.scatter(X1[:, 0], X1[:, 1], s=5)
plt.scatter(X2[:, 0], X2[:, 1], s=5)
plt.scatter(X3[:, 0], X3[:, 1], s=5)
plt.show()
#%%
model = gmm(X.shape[1], 3, X.shape[0])
max_Iter = 60
verbos = 10
model.train(X, max_Iter, verbos)
labels = model.cluster(X)

#%%
plot_clusters(X, model.mu, model.sigma, [mu1, mu2, mu3], [var1, var2, var3])
