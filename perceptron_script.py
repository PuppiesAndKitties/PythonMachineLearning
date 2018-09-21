import matplotlib.pyplot as plt
import perceptron_algorithm
from common_def import plot_decision_regions, X, y

"""
获取鸢尾数据集，抽取出数据子集，描绘出花瓣长度和萼片长度与鸢尾种类的散点分布关系 
"""
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')            # 描出山鸢尾的散点
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')     # 描出变色鸢尾的散点
plt.xlabel('petal length (cm)')     # x轴为花瓣长度
plt.ylabel('sepal length (cm)')     # y轴为萼片长度
plt.legend(loc='upper left')
plt.show()

"""
利用抽取的鸢尾花数据子集来训练感知器，同时绘制每次迭代的错误分类数量的折线图，
以检验算法是否收敛
"""


ppn = perceptron_algorithm.Perceptron(eta=0.01, n_inter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, 'b-', marker='o')
plt.xlabel('Epochs')    # 训练次数
plt.ylabel('Number of misclassifications')
plt.show()

"""
对二维数据集决策边界的可视化，找到可以分开两种类型鸢尾花的决策边界
"""
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length(cm)')
plt.ylabel('petal length(cm)')
plt.legend(loc='upper left')

plt.show()
