from adaline_random_algorithm import AdalineSGD
from common_def import X, y, X_std, plot_decision_regions
import matplotlib.pyplot as plt


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel("sepal length[standardized]")
plt.ylabel("petal length[standardized]")
plt.legend(loc="upper left")
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')
plt.show()
