#source & blog https://www.joinquant.com/post/10626?f=study&m=math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
#产生数据集，用于训练和测试
X1,y1 = make_gaussian_quantiles(cov=2,n_samples=200,n_features=2,n_classes=2,random_state=1)
X2,y2 = make_gaussian_quantiles(mean=(3,3),cov=1.5,n_samples=300,n_features=2,n_classes=2,random_state=1)
'''
make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=None)
Generate isotropic Gaussian and label samples by quantile

This classification dataset is constructed by taking a multi-dimensional
standard normal distribution and defining classes separated by nested
concentric multi-dimensional spheres such that roughly equal numbers of
samples are in each class (quantiles of the :math:`\chi^2` distribution).

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
mean : array of shape [n_features], optional (default=None)
    The mean of the multi-dimensional normal distribution.
    If None then use the origin (0, 0, ...).

cov : float, optional (default=1.)
    The covariance matrix will be this value times the unit matrix. This
    dataset only produces symmetric normal distributions.
n_samples : int, optional (default=100)

    The total number of points equally divided among classes.

n_features : int, optional (default=2)
    The number of features for each sample.

n_classes : int, optional (default=3)
    The number of classes

shuffle : boolean, optional (default=True)
    Shuffle the samples.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Returns
-------
X : array of shape [n_samples, n_features]
    The generated samples.

y : array of shape [n_samples]
    The integer labels for quantile membership of each sample.

Notes
-----
The dataset is from Zhu et al [1].

References
----------
.. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
'''
#对数据进行水平连接
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
#使用adaboosting模型
#使用深度为1的决策树为估计器，算法为samme（可以比较samme.R
#https://stackoverflow.com/questions/31981453/why-estimator-weight-in-samme-r-adaboost-algorithm-is-set-to-1）
#迭代次数为200
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                        algorithm="SAMME",
                        n_estimators=200)
#AdaBoostClassifier(BaseWeightBoosting, sklearn.base.ClassifierMixin)
bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"
plt.figure(figsize=(10, 5))

# Plot the decision boundaries
#plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                    np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
'''
np.c_[np.array([1,2,3]), np.array([4,5,6])]
array([[1, 4],
    [2, 5],
    [3, 6]])
'''

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