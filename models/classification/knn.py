# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:45:33 2020

@author: wangxin
"""
import numpy as np
import operator
from math import sqrt

# kd-tree
class Node():
    def __init__(self, data = -1, label = None, left = None, right = None, split = 0):
        self.left = left
        self.right = right
        self.split = split
        self.data = data
        self.label = label

class kdTree():
    def __init__(self, X, y):
        self.dim = X.shape[1]
        if(y.ndim == 1): y = np.expand_dims(y, 0)
        data = np.hstack((X, y.reshape(-1, 1)))
        self.root = self._create(data, 0)
    
    def search(self, target, K):
        path = []
        nodes = []
        dist = []
        labels = []
        if(K == 0 or self.root == None):
            return nodes, labels, dist 
        
        # step1: 搜索叶子结点
        tmp_node = self.root
        while tmp_node:
            path.append(tmp_node)
            idx = tmp_node.split
            
            if(target[idx] <= tmp_node.data[idx]): 
                tmp_node = tmp_node.left
            else:
                tmp_node = tmp_node.right
        
        # step2: 回溯
        max_index = 0
        first_node = path[-1]
        d = self._compute_dist(first_node.data[:-1], target)
        nodes.append(first_node)
        dist.append(d)
        labels.append(first_node.data[-1])
    
        while len(path)!= 0:
            back_node = path.pop()
            tmp_d = self._compute_dist(back_node.data[:-1], target)
            # 已经到叶子节点
            if(back_node.left == None and back_node.right == None):
                if(back_node != first_node):
                    if(len(nodes) < K or tmp_d < dist[max_index]):
                        if(len(nodes) < K):
                            dist.append(tmp_d)
                            nodes.append(back_node)
                            labels.append(back_node.data[-1])
                            if(len(nodes) == K):
                                max_index = np.argmax(dist)
                        else:
                            dist[max_index] = tmp_d
                            nodes[max_index] = back_node
                            labels[max_index] = back_node.data[-1]
                            max_index = np.argmax(dist)
            # 非叶子节点
            else:
                # 判断是否加入父节点
                if(len(nodes) < K or tmp_d < dist[max_index]):
                    if(len(nodes) < K):
                        dist.append(tmp_d)
                        nodes.append(back_node)
                        labels.append(back_node.data[-1])
                        if(len(nodes) == K):
                            max_index = np.argmax(dist)
                    else:
                        dist[max_index] = tmp_d
                        nodes[max_index] = back_node
                        labels[max_index] = back_node.data[-1]
                        max_index = np.argmax(dist)
                
                #step3: 判断是否需要进入另一个分支
                tmp_idx = back_node.split
                if(len(nodes) < K or
                    abs(back_node.data[tmp_idx]-target[tmp_idx]) <= dist[max_index]):
                    if(target[idx] <= back_node.data[tmp_idx]):
                        child_node = back_node.right
                    else:
                        child_node = back_node.left
                    
                    while(child_node != None):
                        path.append(child_node)
                        if(target[child_node.split] <= child_node.data[child_node.split]):
                            child_node = child_node.left
                        else:
                            child_node = child_node.right
        return nodes, labels, dist
                
    def print_tree(self):
        def print_node(node):
            if(node == None): return
            if(node.father == None): f = -1
            else: f = node.father.data
            print(node.data, node.label, f)
            print_node(node.left)
            print_node(node.right)
        print_node(self.root)
        
    def _create(self, dataset, p):
        if len(dataset) == 0: return None
        
        # numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)
        mid = len(dataset) // 2
        idxs = np.argpartition(dataset[:, p], mid)
        tmp = Node(data = dataset[idxs[mid]])
        lchild = self._create(dataset[idxs[:mid],:], (p+1) % self.dim)
        rchild = self._create(dataset[idxs[mid+1:],:], (p+1) % self.dim)
        
        tmp.left, tmp.right = lchild, rchild
        
        return tmp
    
    def _compute_dist(self, x, y):
        return np.sum((x - y) ** 2)
'''
method:
    scan
    kd-tree
    vector
'''
        
class kNN():
    def __init__(self, K, method = "scan"):
        self.K = K
        self.method = method
        
    def train(self, X, y):
        self.X = X
        self.y = y
        if self.method == "kd-tree":
            self.tree = kdTree(X, y)
    
    def test(self, x):
        if x.ndim == 1: 
            x = np.expand_dims(x, axis = 0)
        labels = []
        y_preds = [0]*x.shape[0]
        
        if self.method == "scan":
            for i in range(x.shape[0]):
                dist = []
                label = []
                # xs = []
                max_index = 0
                for j in range(self.X.shape[0]):
                    tmp_d = np.sum((x[i]-self.X[j]) ** 2)
                    if len(dist) < self.K:
                        dist.append(tmp_d)
                        label.append(self.y[j])
                        # xs.append(self.X[j])
                        if(len(dist) == self.K):
                            max_index = np.argmax(dist)
                    elif(tmp_d < dist[max_index]):
                        dist[max_index] = tmp_d
                        label[max_index] = self.y[j]
                        # xs[max_index] = self.X[j]
                        max_index = np.argmax(dist)
                labels.append(label)
                # print("scan", xs)
        elif self.method == "kd-tree":
            for i in range(x.shape[0]):
                _, label, _ = self.tree.search(x[i], self.K)
                # label = self.tree.search(x[i], self.K)
                labels.append(label)
                # print("kd-tree: ")
                # for node in nodes:
                #     print(node.data)
                # labels.append(label)
            #     y_preds[i] = label
            # return np.array(y_preds)
        else:
            dist = np.sum(np.power(x,2), axis = -1, keepdims = True) - 2 * np.dot(x, self.X.T) + np.sum(self.X ** 2, axis = -1).T
            if dist.ndim == 1: 
                dist = np.expand_dims(dist, axis = 0)
            top_K_index = np.argsort(dist, axis = -1)[:, :self.K]
            # print("vector: ", self.X[top_K_index, :])
            labels = self.y[top_K_index]
        
        labels = np.array(labels)
        
        for i in range(x.shape[0]):
            y_preds[i] = self._vote(labels[i,:])
        
        return np.array(y_preds)
                        
        
    def _vote(self, ys):
        ys_unique = np.unique(ys)
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0]
            