"""
Page 11 -
罗森布拉特感知器学习算法的Python实现
"""

import numpy as np


class Perceptron(object):
    """
    感知分类器
    参数
        eta:float  学习速率
        n_iter：int 迭代次数
    """
    def __init__(self, eta=0.01, n_inter=10):
        self.eta = eta
        self.n_iter = n_inter

    def fit(self, X, y):
        """

        :param X: shape = [n_samples, n_features],分别是样本的序号和特征的序号。
        :param y: shape = [n_samples]，是用于和输出值比较的真值。
        :return: self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
