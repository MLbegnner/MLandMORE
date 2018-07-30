#source & blog https://www.joinquant.com/post/10626?f=study&m=math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
#产生数据集，用于训练和测试
X1,y1 = make_gaussian_quantiles(cov=2,n_samples=200,n_features=2,n_classes=2,random_state=1)
X2,y2 = make_gaussian_quantiles(mean=(3,3),cov=1.5,n_samples=300,n_features=2,n_classes=2,random_state=1)

#对数据进行水平连接
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
# 使用adaboosting模型
# 使用深度为1的决策树为估计器，算法为samme（可以比较samme.R
# https://stackoverflow.com/questions/31981453/why-estimator-weight-in-samme-r-adaboost-algorithm-is-set-to-1）
# 迭代次数为200
# 感觉博客中的算法伪代码有点问题，可以看Stack Overflow中的那个答案
# 注意classfier的权重alpha与数据点权重weight之间的先后顺序呢
# 对于当前错误分类的点，加大权重，在下个classdier中得到更大权重去正确分类

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
bdt.fit(X, y)
plot_colors = "br"
plot_step = 0.02
class_names = "AB"
plt.figure(figsize=(10, 5))

# Plot the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                    np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
            c=c, cmap=plt.cm.Paired,
            s=20, edgecolor='k',
            label="Class %s" % n)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')
plt.show()