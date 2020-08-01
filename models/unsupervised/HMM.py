# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:26:55 2020

@author: wangxin
"""
import numpy as np

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
    def print_model_info(self):
        print("A: ", self.A)
        print("B: ", self.B)
        print("pi: ", self.pi)

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
        
    def _forward(self, seq):
        T = len(seq)
        alpha = np.zeros((T, self.N))
        alpha[0,:] = self.pi * self.B[:,seq[0]]
        for t in range(T-1):
            alpha[t+1] = np.dot(alpha[t], self.A) * self.B[:, seq[t+1]]
        return alpha
    
    def _backward(self, seq):
        T = len(seq)
        beta = np.ones((T, self.N))
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(self.A, self.B[:, seq[t+1]] * beta[t+1])
        return beta
    
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
    
    def _get_data_with_distribute(self, dist): # 根据给定的概率分布随机返回数据（索引）
        return np.random.choice(np.arange(len(dist)), p=dist)
    
    def _random_set_patameters(self):
        self.A = np.random.rand(self.N, self.N)
        self.A /= np.sum(self.A, axis = -1, keepdims=True)
        
        self.B = np.random.rand(self.N, self.M)
        self.B /= np.sum(self.B, axis = -1, keepdims=True)
        
        self.pi = np.random.rand(self.N)
        self.pi /= np.sum(self.pi)
            