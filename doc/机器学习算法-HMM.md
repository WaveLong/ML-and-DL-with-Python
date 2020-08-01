# 机器学习算法-HMM

[TOC]

## 1. 模型定义

​	隐马尔可夫模型（HMM）是一个关于时序的概率模型，是一种特殊的概率图模型。该图模型包含了两个序列：状态序列$\{z_1, z_2, ..., z_T\}$和观测序列$\{x_1, x_2, ..., x_T\}$，取值分别来自于状态集合$Q=\{q_1, q_2, ..., q_N\}$和观测集合$V=\{v_1, v_2, ..., v_M\}$，类比到GMM或者EM算法中，分别对应隐变量和观测变量。HMM包含了两个基本假设：

+ **齐次马尔科夫假设（假设1）**：任何时刻的状态只依赖以前一时刻的状态，与其他因素无关，即$p(z_{t+1}|z_t, \cdot) = p(z_{t+1}|z_t)$

+ **观测独立假设（假设2）：**任何时刻的观测结果只依赖以当前的状态，即$p(x_t|z_t, \cdot)=p(x_t|z_t)$

一个HMM需要用三个参数描述：

+ 状态转移矩阵$A = [a_{ij}]_{N\times N}$：$a_{ij}=p(q_j|q_i)$，即上一个时刻状态为$q_i$时，下一个时刻出现状态$q_j$的概率
+ 观测矩阵$B=[b_{jk}]_{N\times M}$：$b_{jk}=p(v_k|q_j)$，即在状态为$q_j$的情况下，观测到$v_k$的概率
+ 初始向量$\pi = [\pi_i]_N$：$\pi_i = p(z_1 = q_i)$，即在初始条件下，出现状态$q_i$的概率

因此HMM类被定义如下：

```python
class Hmm():
    def __init__(self, state_num, obser_num, 
                 state_transfer_matrix = None, 
                 obser_matrix = None, 
                 init_state_vec = None):
        '''
        state set: Q = {q_1, q_2, ..., q_N}
        observation set : V = {v_1, v_2,..., v_M}
        
        state transfer matrix: a_ij = p(q_j|q_i)
        observation prob. : b_jk = p(v_k|q_j)
        inititial prob.: pi_i = p(i_1 = q_i)
        
        state sequence: I = {i_1, i_2, ..., i_T}
        observation sequence : O = {o_1, o_2, ..., o_T}
        '''
        self.N = state_num
        self.M = obser_num
        
        self.A = state_transfer_matrix
        self.B = obser_matrix
        self.pi = init_state_vec
        
        self.eps = 1e-6
```

## 2. 序列生成

按照模型$\lambda=(A,B,\pi)$生成当前状态、观测结果、下一个时刻的状态即可

```python
	def generate(self, T: int):
        '''
        根据给定的参数生成观测序列
        T: 指定要生成数据的数量
        '''
        z = self._get_data_with_distribute(self.pi)    # 根据初始概率分布生成第一个状态
        x = self._get_data_with_distribute(self.B[z])  # 生成第一个观测数据
        result = [x]
        for _ in range(T-1):        # 依次生成余下的状态和观测数据
            z = self._get_data_with_distribute(self.A[z])
            x = self._get_data_with_distribute(self.B[z])
            result.append(x)
        return np.array(result)
    def _get_data_with_distribute(self, dist): # 根据给定的概率分布随机返回数据（索引）
        return np.random.choice(np.arange(len(dist)), p=dist)
```



## 3. 概率计算

概率计算的目标是，在已知模型的情况下，计算给定的观测序列出现的概率$p(X\mid \lambda)$。一种直接的计算方式$\ref{prob_brute}$是枚举所有的状态序列$Z$，然后求和：
$$
\begin{equation}\label{prob_brute}
p(X\mid \lambda)=\sum_Z p(X,Z\mid \lambda) = \sum_{z_1, z_2, ...,z_T} \pi_{z_1}b_{z_1}(x_1) a_{z_1, z_2}b_{z_2}(x_2)...a_{z_{T-1},z_{T}}b_{z_T}(x_T)
\end{equation}
$$
但是直接计算的复杂度太大，为了提高计算效率，有前向和后向两种计算方式。

```python
	def compute_prob(self, seq, method = 'forward'):  
        ''' 计算概率p(O)，O为长度为T的观测序列
        Parameters
        ----------
        seq : numpy.ndarray
            长度为T的观测序列
        method : string, optional
           概率计算方法，默认为前向计算
           前向：_forward()
           后向：_backward()
        Returns
        -------
        TYPE
            前向：p(O), alpha
        TYPE
            后向：p(O), beta
        '''
        if method == 'forward':
            alpha = self._forward(seq)
            return np.sum(alpha[-1]), alpha
        else:
            beta = self._backward(seq)
            return np.sum(beta[0]*self.pi*self.B[:,seq[0]]), beta
        return None
```

### 3.1 前向计算

定义前向概率$\alpha_t(i)$：
$$
\alpha_t(i) = p(x_1, x_2, ..., x_t, z_t = q_i\mid \lambda)\label{alpha}
$$
根据定义$\ref{alpha}$可知：
$$
\begin{equation}
\alpha_1(i)=p(x_1, z_1 = q_i\mid \lambda) = p(z_1=q_i\mid \lambda)p(x_1\mid z_1=q_i, \lambda)=\pi_i b_i(x_1)\\
\sum_{1\leq i\leq N}\alpha_T(i) = \sum_{1\leq i\leq N} p(x_1, x_2, ..., x_T, z_T = q_i\mid \lambda) = p(X\mid \lambda) \label{prob_forward}
\end{equation}
$$
在已知$\alpha_t(i), \forall i\in{1, 2,.., N}$时，求解$\alpha_{t+1}(i)$：
$$
\begin{align}
\alpha_{t+1}(j) &=\sum_{i}^{N} P\left(x_{1}, \ldots, x_{t+1}, z_{t}= q_{i}, z_{t+1}=q_{j} \mid \lambda\right) \\
&=\sum_{i=1}^{N} \bbox[5px, border:2px solid red]{P\left(x_{t+1} \mid x_{1}, \ldots, x_{t}, z_{t}=q_{i}, z_{t+1}=q_{j}, \lambda\right)} \bbox[5px, border:2px solid blue]{P\left(x_{1}, \ldots, x_{t}, z_{t}=q_{i}, z_{t+1}=q_{j} \mid \lambda\right)}
\end{align}
$$
红色部分根据假设2：
$$
P\left(x_{t+1} \mid x_{1}, \ldots, x_{t}, z_{t}=q_{i}, z_{t+1}=q_{j}, \lambda\right)=P\left(x_{t+1} \mid z_{t+1}=q_{j}\right)=b_{j}\left(x_{t+1}\right)
$$
蓝色部分根据假设1：
$$
\begin{align}
P\left(x_{1}, \ldots, x_{t},z_{t}=q_{i}, z_{t+1} = q_{j} \mid \lambda\right) 
&=P\left(z_{t+1}=q_{j} \mid x_{1}, \ldots, x_{t}, z_{t}=q_{i}, \lambda\right) P\left(x_{1}, \ldots, x_{t}, z_{t}=q_{i} \mid \lambda\right) \\
&=P\left(z_{t+1}=q_{j} \mid z_{t}=q_{i}\right) P\left(x_{1}, \ldots, x_{t}, z_{t}=q_{i} \mid \lambda\right) \\
&=a_{i j} \alpha_{t}(i)
\end{align}
$$
因此：
$$
\alpha_{t+1}(j)=\sum_{i=1}^{N} a_{i j} b_{j}\left(x_{t+1}\right) \alpha_{t}(i)\label{alpha_t+1}
$$
根据$\ref{prob_forward}$和$\ref{alpha_t+1}$得到前向计算的代码：

```python
	def _forward(self, seq):
        T = len(seq)
        alpha = np.zeros((T, self.N))
        alpha[0,:] = self.pi * self.B[:,seq[0]]
        for t in range(T-1):
            alpha[t+1] = np.dot(alpha[t], self.A) * self.B[:, seq[t+1]]
        return alpha
```

### 3.2 后向计算

定义后向概率$\beta_t(i)$：
$$
\beta_{t}(i)=P\left(x_{T}, x_{T-1}, \ldots, x_{t+1} \mid z_{t}=q_{i}, \lambda\right)\label{beta}
$$
根据定义$\ref{beta}$可得：
$$
\begin{equation}
\beta_T(i) = P(x_{T+1}\mid z_T=q_i, \lambda)=1\\
\begin{aligned}
P(X \mid \lambda)&=P\left(x_{1}, x_{2}, \ldots, x_{T} \mid \lambda\right) \\
&=\sum_{i=1}^{N} P\left(x_{1}, x_{2}, \ldots, x_{T}, z_{1}=q_{i} \mid \lambda\right)\\
&=\sum_{i=1}^{N} P\left(x_{1} \mid x_{2}, \ldots, x_{T}, z_{1}=q_{i}, \lambda\right) P\left(x_{2}, \ldots, x_{T}, z_{1}=q_{i} \mid \lambda\right) \\
&=\sum_{i=1}^{N} P\left(x_{1} \mid z_{1}=q_{i}\right) P\left(z_{1}=q_{i} \mid \lambda\right) P\left(x_{T}, x_{T-1}, \ldots, x_{2} \mid z_{1}=q_{i}, \lambda\right) \\
&=\sum_{i=1}^{N} b_{i}\left(x_{1}\right) \pi_{i} \beta_{1}(i)
\end{aligned}
\end{equation}
$$
可以验证，在任意时刻t，存在：
$$
\begin{align}
P(X\mid \lambda) &= \sum_{i}P(X, z_t = q_i\mid \lambda)\\
&= \sum_i P(x_1, x_2, \ldots, x_t, x_{t+1}, \ldots, x_T, z_t = q_i)\\
&= \sum_{i} P(x_1, \ldots, x_t, z_t = q_i)\bbox[border:2px red]{P(x_{t+1}, \ldots, x_T\mid x_{1}, \ldots, x_t, z_t = q_i)}\\
\end{align}
$$
红色部分的计算如下：
$$
\begin{align}
P(x_{t+1}, \ldots, x_T\mid x_{1}, \ldots, x_t, z_t = q_i) &= \frac{P(X\mid z_t = q_i)}{P(x_1, \ldots, x_t\mid z_t = q_i)}\\
&= \frac{P(x_t\mid x_1, \ldots, x_{t-1}, x_{t+1}, \ldots, x_T, z_t = q_i)P(x_1, \ldots, x_{t-1}, x_{t+1}, \ldots, x_T\mid z_t = q_i)}{P(x_t\mid x_1, \dots, x_{t-1}, z_t = q_i)P(x_1, \ldots, x_{t-1}\mid z_t = q_i)}\\
&= \frac{P(x_t\mid z_t = q_i)P(x_1,\ldots, x_{t-1}\mid z_t = q_i)P(x_{t+1}, \ldots, x_T\mid z_t = q_i)}{P(x_t\mid z_t = q_i)P(x_1, \ldots, x_{t-1}\mid z_t = q_i))}\\
&= P(x_{t+1}, \ldots, x_T\mid z_t = q_i)
\end{align}
$$
结合定义$\ref{alpha}$和$\ref{beta}$得到：
$$
P(X\mid \lambda) = \sum_i \alpha_t(i)\beta_t(i), \quad \forall t\in\{1, 2, \ldots, T\}
$$
在已知$\beta_t(i), \forall i\in{1, 2,.., N}$时，求解$\beta_{t+1}(i)$：
$$
\begin{align}

\beta_{t}(i) &=P\left(x_{T}, \ldots, x_{t+1} \mid z_{t}=q_{i}, \lambda\right) \\
&=\sum_{j=1}^{N} P\left(x_{T}, \ldots, x_{t+1}, z_{t+1}=q_{j} \mid z_{t}=q_{i}, \lambda\right) \\
&=\sum_{j=1}^{N} \bbox[border:red 2px solid ]{P\left(x_{T}, \ldots, x_{t+1} \mid z_{t+1}=q_{j}, z_{t}=q_{i}, \lambda\right)}\bbox[border:blue 2px solid] {P\left(z_{t+1}=q_{j} \mid z_{t}=q_{i}, \lambda\right)}
\label{beta_t+1}
\end{align}
$$
蓝色部分即为$a_{ij}$，红色部分为：
$$
\begin{array}{l}
P\left(x_{T}, \ldots, x_{t+1} \mid z_{t+1}=q_{j}, z_{t}=q_{i}, \lambda\right) 
&=P\left(x_{T}, \ldots, x_{t+1} \mid z_{t+1}=q_{j}, \lambda\right) \\
&=P\left(x_{t+1} \mid x_{T}, \ldots, x_{t-2}, z_{t+1}=q_{j}, \lambda\right) P\left(x_{T}, \ldots, x_{t-2} \mid z_{t+1}=q_{j}, \lambda\right) \\
&=P\left(x_{t+1} \mid z_{t+1}=q_{j}\right) P\left(x_{T}, \ldots, x_{t+2} \mid z_{t+1}=q_{j}, \lambda\right) \\
&=b_{j}\left(x_{t+1}\right) \beta_{t+1}(j)\label{beta_red}
\end{array}
$$
第一个等号成立是因为：
$$
\begin{align}
P(x_T, \ldots, x_{t+1}\mid z_{t+1}, z_t) &= \frac{P(x_T, \ldots, x_{t+1}, z_{t+1}, z_t)}{P(z_{t+1}, z_t)}\\
&= \frac{P(x_T, \ldots, x_{t+1}\mid z_{t+1})P(z_{t+1}\mid z_t)P(z_t)}{P(z_{t+1}, z_t)}\\
\end{align}
$$
将$\ref{beta_red}$代入到$\ref{beta_t+1}$得到：
$$
\beta_{t}(i)=\sum_{j=1}^{N} a_{i j} b_{j}\left(x_{t+1}\right) \beta_{t-1}(j)
$$
根据公式得到后向计算代码为：

```python
	def _backward(self, seq):
        T = len(seq)
        beta = np.ones((T, self.N))
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(self.A, self.B[:, seq[t+1]] * beta[t+1])
        return beta
```



## 4. 学习

HMM的学习是根据观测序列去推测模型参数，学习算法采用了EM算法，具体过程为：

E-step：
$$
\begin{align}
\mathcal{Q}(\lambda, \lambda^{\mathrm{old}}) &= \mathbb{E}_{Z\sim P(Z\mid X,\lambda^{\mathrm{old}})}\left[\log P(X, Z\mid \lambda)\right]\\
&= \sum_{Z} \log\big(\bbox[border:red 2px]{P(Z\mid \lambda)} \bbox[border:blue 2px]{P(X\mid Z, \lambda)}\big)\bbox[border:purple 2px]{P(Z\mid X, \lambda^{\mathrm{old}})}\\
\end{align}
$$
红色部分：
$$
P(Z\mid \lambda) = \pi(z_1)\prod_{t=2}^T P(z_t\mid z_{t-1})
$$
蓝色部分：
$$
\begin{align}
P(X\mid Z, \lambda) &= P(x_1, x_2, \ldots, x_T\mid z_1, z_2, \ldots, z_T, \lambda)\\
&= P(x_T\mid x_1, x_2, \ldots, x_{T-1}, z_1, z_2, \ldots, z_T)P(x_1, \ldots, x_{T-1}\mid z_1,\ldots, z_T)\\
&= P(x_T\mid z_T)P(x_1, \ldots, x_{T-1}\mid z_1,\ldots, z_T)\\
&\vdots\\
&= \prod_{t=1}^T P(x_t\mid z_t)
\end{align}
$$
紫色部分：
$$
\begin{align}
P(Z\mid X, \lambda^{\mathrm{old}}) &= \frac{P(X,Z\mid \lambda^{\mathrm{old}})}{P(X\mid \lambda^{\mathrm{old}})}
\end{align}
$$
分母是常量，因此省略。将各个部分代入到Q函数中得到：
$$
\begin{align}
\lambda =& \arg\max_{\lambda} \sum_{Z} P(Z\mid X, \lambda^{\mathrm{old}})\log\big(P(Z\mid \lambda)P(X\mid Z, \lambda)\big) \\
=& \arg\max_{\lambda} \sum_Z P(X,Z\mid \lambda^{\mathrm{old}})\log \big(
\pi(z_1)\prod_{t=2}^T P(z_t\mid z_{t-1})
\prod_{t=1}^T P(x_t\mid z_t)
\big)\\
 =& \arg\max_{\lambda} \bbox[6px, border:red 2px]{\sum_Z P(X,Z\mid \lambda^{\mathrm{old}})\log\pi(z_1)}\\
 &+ \arg\max_{\lambda} \bbox[6px, border:blue 2px]{\sum_Z P(X,Z\mid \lambda^{\mathrm{old}})\log \prod_{t=2}^T P(z_t\mid z_{t-1})} \\
 &+\arg\max_{\lambda} \bbox[6px, border:purple 2px]{\sum_Z P(X,Z\mid \lambda^{\mathrm{old}}) \log \prod_{t=1}^T P(x_t\mid z_t)}\\
\end{align}
$$
红色、蓝色和紫色框分别对应$\pi,B,A$三部分

### 4.1 求解$\pi$

参数$\pi$满足$\sum_{i=1}^N\pi_i = 1$，拉格朗日函数为：
$$
\begin{align}
\mathcal{L}_1 &= \sum_Z P(X, Z\mid \lambda_{\mathrm{old}})\log \pi(z_1) +\mu (\sum_{i=1}^N \pi_i - 1)\\
&= \sum_{1\leq i \leq N} P(z_1 = q_i, X\mid \lambda^{\mathrm{old}})\log \pi_i + \mu (\sum_{i=1}^N \pi_i - 1)\\
\end{align}
$$
求导得到：
$$
\nabla_{\pi_i} \mathcal{L}_1 = \frac{1}{\pi_i} P(z_1 = q_i, X \mid \lambda^{\mathrm{old}}) + \mu \\
\Longrightarrow \pi_i = \frac{P(z_1=q_i, X \mid \lambda^{\mathrm{old}})}{\sum_{1 \leq j \leq N} P(z_1=q_j, X \mid \lambda^{\mathrm{old}})}
$$
定义：$\gamma_t(i) = P(z_t = q_i, X\mid \lambda^{\mathrm{old}})=\alpha_t(i)\beta_t(i)$，得到：
$$
\pi_i = \frac{\gamma_1(i)}{\sum_j \gamma_1(j)}=\frac{\alpha_t(i)\beta_t(i)}{\sum_j \alpha_t(j)\beta_t(j)}
$$

### 4.2 求解$A$

A的行向量的和为1，写出关于$A$的拉格朗日函数：
$$
\begin{align}
\mathcal{L}_2 &= \sum_Z P(X,Z\mid \lambda^{\mathrm{old}}) \log \prod_{t=1}^T P(x_t\mid z_t) + \mu (AE - \mathbb{1})\\
&= \sum_{2\leq t \leq T} \sum_{1\leq i,j \leq N}P(z_{t-1} = q_i, z_t = q_j, X \mid \lambda^{\mathrm{old}})\log P(z_t\mid z_{t-1})  + \mu (AE - \mathbb{1})\\
&= \sum_{2\leq t \leq T} \sum_{1\leq i,j \leq N}P(z_{t-1} = q_i, z_t = q_j, X \mid \lambda^{\mathrm{old}})\log a_{ij} + \mu (AE - \mathbb{1})
\end{align}
$$
其中E的元素全部为1。求导得到：
$$
\begin{align}
\nabla_{a_{ij}}\mathcal{L}_2 = \frac{1}{a_{ij}} \sum_{1\leq t \leq T-1} P(z_t = q_i, z_{t+1}=q_j, X\mid \lambda^{\mathrm{old}}) + \mu_i a_{ij} \\
\Longrightarrow a_{ij} = \frac{\sum_{1\leq t \leq T-1} P(z_t = q_i, z_{t+1}=q_j, X\mid \lambda^{\mathrm{old}}}{\sum_j \sum_{1\leq t \leq T-1} P(z_t = q_i, z_{t+1}=q_j, X\mid \lambda^{\mathrm{old}}}
\end{align}
$$
定义：$\xi_t(i,j) = P(z_t=q_i, z_{t+1}=q_j, X\mid \lambda^{\mathrm{old}})$，结合$\gamma_t(i)$的定义可知$\gamma_t(i) = \sum_j \xi_t(i,j)$，代入上式得到：
$$
a_{ij} = \frac{\sum_{1\leq T \leq T-1} \xi_t(i,j)}{\sum_{1\leq t \leq T-1}\gamma_t(i)}
$$
注意分母上求和到T-1

### 4.3 求解$B$

B的行向量的和为1，写出关于B的拉格朗日函数：
$$
\begin{align}
\mathcal{L}_3 &= \sum_Z P(X,Z\mid \lambda^{\mathrm{old}}) \log \prod_{t=1}^T P(x_t\mid z_t) + \mu(BE-\mathbb{1})\\
&= \sum_{1\leq k \leq M} \sum_{1\leq t \leq T, x_t = v_k} \sum_{1\leq j \leq N} P(z_t = q_j, X \mid \lambda^{\mathrm{old}})\log P(x_t \mid z_t = q_j) +  \mu(BE-\mathbb{1})
\end{align}
$$
求导得到：
$$
\nabla_{b_{jk}} \mathcal{L}_3 = \frac{1}{b_{jk}} \sum_{1\leq t\leq T, x_t = v_k} P(z_t = q_j, X \mid \lambda^{\mathrm{old}}) + \mu_j \\
\Longrightarrow b_jk = \frac{\sum_{1\leq t\leq T, x_t = v_k} P(z_t = q_j, X \mid \lambda^{\mathrm{old}})}{\sum_{1\leq t\leq T} P(z_t = q_j, X \mid \lambda^{\mathrm{old}})} = \frac{\sum_{1\leq t \leq T, x_t = v_k}\gamma_t(j)}{\sum_{1\leq t \leq T} \gamma_t(j)}
$$
HMM模型的学习代码为：

```python
	def baum_welch(self, seq, max_Iter = 500, verbos = 10):
        '''HMM学习，使用Baum_Welch算法
        Parameters
        ----------
        seq : np.ndarray
            观测序列，长度为T
        max_Iter : int, optional
            最大的训练次数，默认为500
        verbos : int, optional
            打印训练信息的间隔，默认为每10步打印一次
        Returns
        -------
        None.

        '''
        # 初始化模型参数
        self._random_set_patameters()
        # 打印训练之前模型信息
        self.print_model_info()
        
        for cnt in range(max_Iter):
            # alpha[t,i] = p(o_1, o_2, ..., o_t, i_t = q_i)
            alpha = self._forward(seq)
            # beta[t,i] = p(o_t+1, o_t+2, ..., o_T|i_t = q_i)
            beta = self._backward(seq)
            
            # gamma[t,i] = p(i_t = q_i | O)
            gamma = self._gamma(alpha, beta)
            # xi[t,i,j] = p(i_t = q_i, i_t+1 = q_j | O)
            xi = self._xi(alpha, beta, seq)
            
            # update model 
            self.pi = gamma[0] / np.sum(gamma[0])
            self.A = np.sum(xi, axis=0)/(np.sum(gamma[:-1], axis=0).reshape(-1,1)+self.eps)
            
            for obs in range(self.M):
                mask = (seq == obs)
                self.B[:, obs] = np.sum(gamma[mask, :], axis=0)
            self.B /= np.sum(gamma, axis=0).reshape(-1,1)
            
            self.A /= np.sum(self.A, axis = -1, keepdims=True)
            self.B /= np.sum(self.B, axis = -1, keepdims=True)
            
            # print training information
            if cnt % verbos == 0:
                logH = np.log(self.compute_prob(seq)[0]+self.eps)
                print("Iteration num: {0} | log likelihood: {1}".format(cnt, logH))
                
    def _gamma(self, alpha, beta):
        G = np.multiply(alpha, beta)
        return  G / (np.sum(G, axis = -1, keepdims = True)+self.eps)
    
    def _xi(self, alpha, beta, seq):
        T = len(seq)
        Xi = np.zeros((T-1, self.N, self.N))
        for t in range(T-1):
            Xi[t] = np.multiply(np.dot(alpha[t].reshape(-1, 1),
                                       (self.B[:, seq[t+1]]*beta[t+1]).reshape(1, -1)),
                                self.A)
            Xi[t] /= (np.sum(Xi[t])+self.eps)
        return Xi
    def _random_set_patameters(self):
        self.A = np.random.rand(self.N, self.N)
        self.A /= np.sum(self.A, axis = -1, keepdims=True)
        
        self.B = np.random.rand(self.N, self.M)
        self.B /= np.sum(self.B, axis = -1, keepdims=True)
        
        self.pi = np.random.rand(self.N)
        self.pi /= np.sum(self.pi)
```

## 5. 预测

HMM的预测任务是指，根据已知的观测序列$X$，推测出最可能的状态序列$Z$。近似算法为：
$$
z_t^\star = \arg\max_i \gamma_t(i)
$$
近似算法的缺点是不能保证求解出来的是全局最优状态序列。

在实际中，常用的是利用动态规划的维特比算法（Viterbi）。其想法是：从节点$x_1$到$x_t$间的最优状态序列一定是全局最优状态序列上的一部分，因此可以通过不断迭代求解出全局最优状态序列。定义：
$$
\delta_t(i) = P(z_t = q_i, z_{t-1}^\star, \ldots, z_1^\star, x_t, x_{t-1},\ldots, x_1)\\
\phi_t(i) = \arg\max_j \delta_{t-1}(j)a_{ji}
$$
根据定义可以得到：
$$
\begin{align}
\delta_{t+1}(i) &= P(z_{t+1} = q_i, z_t^\star, \ldots, z_1^\star, x_{t+1}, x_{t},\ldots, x_1) \\
&= \max_j \delta_t(j)a_{ji}b_i(x_{t+1})\\
\phi_t(i) &= \arg\max_j \delta_{t-1}(j)a_ji
\end{align}
$$
代码如下：

```python
def verbiti(self, obs):
        ''' 在给定观测序列时，计算最可能出现的状态序列

        Parameters
        ----------
        obs : numpy.ndarray
            长度为T的观测序列
        Returns
        -------
        states : numpy.ndarray
            最优的状态序列
        '''
        T = len(obs)
        states = np.zeros(T)
        delta = self.pi * self.B[:, obs[0]]
        states[0] = np.argmax(delta)
        
        for t in range(1, T):
            tmp_delta = np.zeros_like(delta)
            for i in range(self.N):
                tmp_delta[i] = np.max(delta * self.A[:,i] * self.B[i , obs[t]])
            states[t] = np.argmax(delta)
        
        return states
```

