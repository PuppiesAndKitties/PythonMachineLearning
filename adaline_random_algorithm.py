"""
Page 26 -
使用随机梯度下降实现Adaline学习算法中的权重更新，把fit()方法改为使用单个训练样本更新权重。
此外，增加一个partial_fit方法，对于在线学习，此方法不会充值权重。为了检验算法在训练后是否
收敛，将每次迭代后计算出的代价值作为训练样本的平均消耗。此外还增加了一个shuffle训练数据选
项，每次迭代前重排训练数据避免在优化代价函数阶段陷入循环。通过random_state参数，可以指定
随机数种子以保持多次训练的一致性。
"""

import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """

        :param X: shape = [n_samples, n_features],分别是样本的序号和特征的序号。
        :param y: shape = [n_samples]，是用于和输出值比较的真值。
        :return: self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        # 生成一个包含0~len(y)中间所有整数(包含0，不含len(y))的不重复的随机序列，作为索引
        # 帮助打乱特征矩阵和类标向量
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
