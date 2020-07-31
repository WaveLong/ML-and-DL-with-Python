# ML-and-DL-with-Python-

## 环境

Python + numpy + Pytorch

## 算法

### 有监督

1. kNN

在最邻近点扫描部分，实现了线性扫描、向量化计算和kd树三种

2. svm

线性svm、高斯核svm，smo优化部分使用了simple-smo，未对内外层优化变量进行最优的选择

待改进：
+ LRU缓存；在SVM+SMO中的实现中，核矩阵的计算成为了很大的开销。如果预先将核矩阵计算好，空间复杂度为$O(N^2)$，如果边用边计算，又会因为重复计算增加开销。考虑到在实际计算中，用到的样本仅为支持向量附近的一些数据点，因此用两个cache保存核和误差，在对参数更新之后更新缓存即可。

+ 冷热数据分离；在更新参数时，优先更新支持向量（热数据）对应的$\alpha$（即$0<\alpha<C$），在没有这样的点时，再全局（冷数据）寻找进行更新。

### 无监督

1. GMM+EM

以Gaussian + Dirichlet分布，推导GMM混合模型并实现，优化中采用了EM算法

2. VAE

GMM的改进版本，采用Pytorch实现

## 补充

1. 所有的详细文档都在doc文件夹下，在线可能无法查看，可以下载到本地，或者到我的(博客)[https://www.cnblogs.com/vinnson/category/1800670.html]，内容完全一样

2. 本项目不是为了将所有的算法都实现一遍，我的目的是将一些经典的、具有代表性的算法从头到尾实现一遍，弄清楚中间的原理。选取的每个算法都有不同的侧重点，对于像Logistic Regression/CNN/RNN这些广为人知的入门模型，或者像Xgboost这种大型的造不下来的轮子，选择放弃。实现的算法也还存在很多待改进的部分，完全是toy example

3. 后续会继续补充一些，比如HMM、变分推断等，可能会更多侧重概率模型


