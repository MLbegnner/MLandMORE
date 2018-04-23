
# -*- coding: utf-8 -*-

'''
    高斯模型
    二维向量
    经测试线性程度较好
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
        self.fi=np.sum(self.Y)/self.data_len;
        index_1=np.where(self.Y==1)
        self.mu_1=np.sum(self.X[index_1],0)/len(index_1[0])
        index_0=np.where(self.Y==0)
        self.mu_0=np.sum(self.X[index_0],0)/len(index_0[0])
        for i in range(self.data_len):
            if i in index_1[0]:
                #k=np.where(index_1==i)
                self.sigma=self.sigma+np.dot(np.transpose([self.X[i,:]-self.mu_1]),(self.X[i,:]-self.mu_1).reshape(1,2))
                pass
            else:
                #k=np.where(index_0==i)
                self.sigma=self.sigma+np.dot(np.transpose([self.X[i,:]-self.mu_0]),(self.X[i,:]-self.mu_0).reshape(1,2))
                pass
            pass
        pass
    def judge(self,X):
            p_0=np.exp((-1/2)*np.dot(np.dot((X-self.mu_0).T,np.linalg.inv(self.sigma)),(X-self.mu_0)))/(4*np.square(np.pi)*np.sqrt(np.linalg.det(self.sigma)))
            p_1=np.exp((-1/2)*np.dot(np.dot((X-self.mu_1).T,np.linalg.inv(self.sigma)),(X-self.mu_1)))/(4*np.square(np.pi)*np.sqrt(np.linalg.det(self.sigma)))
            if(p_1>p_0):
                return 1
            else:
                return 0
            pass
    def predict(self,X):
        return self.judge(X)

        pass
    pass
    


def plot_region(X,Y):
    X_n=X[:,0].T
    Y_n=X[:,1].T
    Z_n=Y.T
    index_0=np.where(Z_n==0)
    index_1=np.where(Z_n==1)
    flag_for_label0=0
    flag_for_label1=0
    if flag_for_label0==0:
        plt.scatter(X_n[index_0],Y_n[index_0],marker='*',color='red',label='origin data : output is 0')
        flag_for_label0=1
        pass
    else:
        plt.scatter(X_n[index_0],Y_n[index_0],marker='*',color='red')
        pass
    if flag_for_label1==0:
        plt.scatter(X_n[index_1],Y_n[index_1],marker='x',color='blue',label='origin data: output is 1')
        flag_for_label1=1
        pass
    else:
        plt.scatter(X_n[index_1],Y_n[index_1],marker='x',color='blue')
        pass
    plt.title("origin data")
    plt.xlabel("X")
    plt.ylabel("Y")
    pass
gas=Gausian()
gas.getData()
gas.fit()
plot_region(gas.X,gas.Y)
Y_rand=7*np.random.rand(100)
X_rand=4*np.random.rand(100)+4
flag_for_label0=0
flag_for_label1=0
for i in range(100):
    result=gas.predict([X_rand[i],Y_rand[i]])
    if result==0:
        if flag_for_label0==0:
            plt.scatter(X_rand[i],Y_rand[i],marker='x',color='yellow',label='predict data : output is 0')
            flag_for_label0=1;
            pass
        else:
            plt.scatter(X_rand[i],Y_rand[i],marker='x',color='yellow')
        pass
    else: 
        if flag_for_label1==0:
            plt.scatter(X_rand[i],Y_rand[i],marker='*',color='black',label='predict data : output is 1')
            flag_for_label1=1
            pass
        else:
            plt.scatter(X_rand[i],Y_rand[i],marker='*',color='black')
        pass
plt.legend(loc='upper left',frameon=False)
plt.show()

