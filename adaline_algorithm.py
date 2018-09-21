"""
Page 20 -
自适应线性神经元，通过梯度下降最小化代价函数
在感知器的代码基础上修改fit()方法，将其权重的更新改为通过梯度下降最小化代价函数来实现Adaline算法
"""

import numpy as np


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X: shape = [n_samples, n_features],分别是样本的序号和特征的序号。
        :param y: shape = [n_samples]，是用于和输出值比较的真值。
        :return: self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)   # X.T是将X对应的矩阵进行转至后返回的矩阵
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0    # 代价函数的输出值
            self.cost_.append(cost)             # 存储代价函数的输出值以检查本轮训练后算法是否收敛
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
