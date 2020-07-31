#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

class gmm():
    def __init__(self, dims, K, N):
        self.dims = dims
        self.K = K
        self.N = N
        self.mu = np.random.randn(K, dims)
        self.sigma = np.ones((K, dims))
        self.alpha = np.ones(K)/self.K

        self.eps = 1e-6
    
    def train(self, X, maxIter, verbos):
        for i in range(maxIter):
            self.step(X)
            if i % verbos == 0:
                logH = self._log_likelihod(X)
                print("In loop: %d, log likelihood : %f" %(i, logH))
        
    def step(self, X):
        
        # E-step
        p = self._prob(X) # N x K matrix
        alpha_p = self.alpha * p
        Z = np.sum(alpha_p, axis = 1, keepdims=True)
        gamma = alpha_p / Z
        N_k = np.sum(gamma, axis = 0)

        # M-step
        self.alpha = N_k / self.N
        tmp_mu = np.zeros_like(self.mu)
        tmp_sigma = np.zeros_like(self.sigma)
        for k in range(self.K):
            tmp_mu[k] = np.average(X, axis = 0, weights = gamma[:,k])
            tmp_sigma[k] = np.average((X - self.mu[k])**2, axis = 0, weights = gamma[:, k])
        self.mu = tmp_mu
        self.sigma = tmp_sigma
    
    def _log_likelihod(self, X):
        n_points, n_clusters = len(X), self.K
        pdfs = (self.alpha*self._prob(X)).sum(axis = 1)
        return np.mean(np.log(pdfs))
                       
    def _prob(self, X):
        n_points, n_clusters = len(X), self.K
        pdfs = np.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = multivariate_normal.pdf(X, self.mu[i], np.diag(self.sigma[i]))
        return pdfs

    def cluster(self, X):
        p = self._prob(X)
        labels = np.argmax(p, axis = -1)
        return labels

def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
    assert X.shape[1] == 2, "this function can't plot 3D figure"

    n_clusters = len(Mu)
    # colors = ['r', 'g', 'b']
    colors = [0]*n_clusters
    for i in range(n_clusters):
        colors[i] = randomcolor()

    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5) # markersize = 5
    ax = plt.gca() # get current axis
    
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 3, 'edgecolor': colors[i], 'ls': '--'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 3, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[np.random.randint(0,14)]
    return "#"+color