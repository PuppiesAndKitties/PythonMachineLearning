'''
逻辑斯蒂回归、线性支持向量机、核SVM、决策树、随机森林
'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 随机将数据矩阵X与类标向量y按照3:7的比例划分为测试数据集和训练数据集。

# 使用preprocessing模块中的StandardScaler类对特征进行标准化处理
sc = StandardScaler()   # 实例化一个StandardScaler对象
sc.fit(X_train)         # fit方法计算训练数据中每个特征的样本均值和标准差
X_train_std = sc.transform(X_train)     # 使用前面的均值和标准差对训练数据做标准化处理
X_test_std = sc.transform(X_test)
# 要使用相同的缩放参数分别处理训练数据和测试数据，以保证它们的值是彼此相当的

ppn = Perceptron(eta0=0.1, random_state=0, max_iter=40)  # 使用random_state参数在每次迭代后打乱数据集顺序
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('误判样本数: %d' % (y_test != y_pred).sum())
print('精准度： %.2f' % accuracy_score(y_test, y_pred))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 设置标志生成器和色图
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 画出决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # meshgrid()方法  https://www.cnblogs.com/sunshinewang/p/6897966.html
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # 画出所有分类样本
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # 高亮测试样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.4, c='b',
                    linewidth=1, marker='o', s=55, label='test set')


X_combined_std = np.vstack((X_train_std, X_test_std))  # 按照行顺序将数组堆叠
y_combined = np.hstack((y_train, y_test))   # 按照列顺序将数组堆叠
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc='upper left')
plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.01, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()


lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('sepal width[standardized]')
plt.legend(loc='upper left')
plt.show()

np.random.seed(0)
X_xor = np.random.randn(200, 2)
Y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
Y_xor = np.where(Y_xor, 1, -1)
plt.scatter(X_xor[Y_xor == 1, 0], X_xor[Y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[Y_xor == -1, 0], X_xor[Y_xor == -1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10.0)
svm.fit(X_xor, Y_xor)
plot_decision_regions(X_xor, Y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()


svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))


tree = DecisionTreeClassifier(criterion='entropy',  # 选择熵作为不纯度衡量标准
                              max_depth=3,   # 决策树的深度为3
                              random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend(loc='upper left')
plt.show()

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend(loc='upper left')
plt.show()
