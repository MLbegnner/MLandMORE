# -*- coding: utf-8 -*-
##test
'''
    该LMS目标值为0，然后根据是否大于0来设置输出值
    若用于逻辑递归，则使得X=1/(1+exp(-1*(theta*x)))//该函数是逻辑函数
    使用是batch gradient decent
    后面另一个类是stochastic gradient decent ,如果使用这个类，循环次数应当适当增加
    因为该处直接跟0比较，然后返回相应的数，真正的LMS应该结束于错误连续多次小于摸一个固定的值
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, eta = 0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter;
        pass
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1]);
        self.errors_ = [];
        for _ in range(self.n_iter) :
            errors = 0
            
            for xi, target in zip(X,y):
                
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update;
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass
        pass

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0 , 1, -1)
        pass
    pass

class Perceptron_stochastic(object):
    def __init__(self, eta = 0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter;
        pass
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1]);
        self.errors_ = [];
        for i in range(self.n_iter) :
            errors = 0
            
            sel=i%len(X)
            xi=X[sel,:]
            update = self.eta * (target - self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update;
            errors += int(update != 0.0)
            self.errors_.append(errors)
            pass
        pass

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0 , 1, -1)
        pass
    pass

file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(file,header=None)  #header=None 数据第一行是有用数据，不是表头
#print(df.head(10))  # 显示前十行

#数据可视化 scatter 散点图

#import numpy as np
y = df.loc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
#plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
#plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
#plt.xlabel('花瓣长度') 乱码坐标不显示
#plt.xlabel(u'花瓣长度',fontproperties='SimHei')
#plt.ylabel(u'花径长度',fontproperties='SimHei')
#plt.legend(loc='upper left')
#plt.show()
ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
#plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_,marker='o')
#plt.xlabel(u'Epochs',fontproperties='SimHei')
#plt.ylabel(u'错误分类次数',fontproperties='SimHei')
#plt.show()


def plot_decision_regions(x,y,classifier,resolution=0.02):
    
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max()
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max()
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.8,cmap=cmap)#画等高线，alpha透明度，cmap根据z的数值来确定颜色
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')         
    plt.scatter(X[50:100,0],X[50:100,1],color = 'blue',marker = 'x',label = 'versicolor')
    plt.xlabel(u'花径长度',fontproperties = 'SimHei')
    plt.ylabel(u'花瓣长度',fontproperties = 'SimHei')
    plt.legend(loc = 'upper left')
    plt.show()
plot_decision_regions(X,y,ppn,resolution=0.02)

#import pandas as pd
#file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#df = pd.read_csv(file,header=None)  #header=None 数据第一行是有用数据，不是表头
#print(df.head(10))  # 显示前十行

##数据可视化  scatter 散点图
#import matplotlib.pyplot as plt
#import numpy as np
#y = df.loc[0:100,4].values
#y = np.where(y == 'Iris-setosa',-1,1)
#X = df.iloc[0:100,[0,2]].values
##plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
##plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
##plt.xlabel('花瓣长度') 乱码坐标不显示
##plt.xlabel(u'花瓣长度',fontproperties='SimHei')
##plt.ylabel(u'花径长度',fontproperties='SimHei')
##plt.legend(loc='upper left')
##plt.show()

#ppn = Perceptron(eta=0.1,n_iter=10)
#ppn.fit(X,y)
#plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
#plt.xlabel(u'Epochs',fontproperties='SimHei')
#plt.ylabel(u'错误分类次数',fontproperties='SimHei')
#plt.show()

#from matplotlib.colors import ListedColormap
#def plot_decision_regions(x,y,classifier,resolution=0.02):
    
#    colors = ('red','blue','lightgreen','gray','cyan')
#    cmap = ListedColormap(colors[:len(np.unique(y))])
#    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()
#    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()
#    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
#    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
#    plt.xlim(xx1.min(),xx1.max())
#    plt.ylim(xx2.min(),xx2.max())
#    plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')         
#    plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
#    plt.xlabel(u'花径长度',fontproperties='SimHei')
#    plt.ylabel(u'花瓣长度',fontproperties='SimHei')
#    plt.legend(loc='upper left')
#    plt.show()
#plot_decision_regions(X,y,ppn,resolution=0.02)