
# -*- coding: utf-8 -*-

'''
    高斯模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Gausian(object):
    def __init__(self, fi = 0, mu_0=0,mu_1=0,sigma=0):
        self.fi = fi;
        self.mu_0 = mu_0;
        self.mu_1=mu_1;
        self.sigma=sigma;
        pass
    
    def getData(self):
        file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        df = pd.read_csv(file,header=None)  #header=None 数据第一行是有用数据，不是表头
        self.Y = np.array(df.loc[0:149,4].values)
        self.Y = np.where(self.Y == 'Iris-setosa',0,1)
        self.X = np.array(df.loc[0:149,[0,2]].values)
        self.data_len=len(self.Y);
        pass

    def fit(self):
        #length = len(Y);
        self.fi=np.sum(df)/self.data_len;
        index_1=np.where(self.Y==1)
        self.mu_1=np.sum(self.X[index_1],0)/len(index_1)
        index_0=np.where(self.Y==0)
        self.mu_0=np.sum(self.X[index_0],0)/len(index_0)
        for i in range(self.data_len):
            if i in index_1:
                #k=np.where(index_1==i)
                self.sigma=self.sigma+np.dot((self.X[i,:]-self.mu_1),(self.X[i,:]-self.mu_1).T)
                pass
            else:
                #k=np.where(index_0==i)
                self.sigma=self.sigma+np.dot((self.X[i,:]-self.mu_0),(self.X[i,:]-self.mu_0).T)
                pass
            pass
        pass
    def judge(self,X):
        p_0=np.exp((-1/2)*dot(dot((X-self.mu_0).T,self.sigma.I),(X-self.mu_0)))/(4*np.square(np.pi)*np.sqtr(np.det(self.sigma)))
        p_1=np.exp((-1/2)*dot(dot((X-self.mu_1).T,self.sigma.I),(X-self.mu_1)))/(4*np.square(np.pi)*np.sqtr(np.det(self.sigma)))
        if(p_1>p_0):
            return 1
        else:
            return 0
        pass
    def predict(self,X):
        return judge(self.X)
        pass
    pass



def plot_region(X,Y):
    X_n=X[:,0].T
    Y_n=X[:,1].T
    Z_n=Y.T
    index_0=np.where(Z_n==0)
    index_1=np.where(Z_n==1)
    plt.scatter(X_n[index_0],Y_n[index_0],marker='*',color='red')
    plt.scatter(X_n[index_1],Y_n[index_1],marker='x',color='blue')
    plt.title("origin data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    pass
gas=Gausian()
gas.getData()
gas.fit()

plot_region(gas.X,gas.Y)
result=gas.predict(np.array([5.6,2.4]))

if result==0:
    plt.scatter(5.6,2.4,marker='x',color='yelow')
    pass
else:
    plt.scatter(5.6,2.4,marker='*',color='yellow')
    pass

plt.show()

