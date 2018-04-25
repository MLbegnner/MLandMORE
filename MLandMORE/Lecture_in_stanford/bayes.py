
'''
    bayes模型（垃圾邮件识别）
    input data : 经处理后需要为0根1
    output data : 1 and 0
    这边需要说明的是，这边的数据并不是很理想
    因为没有测试数据，或者说这边的数据很不理想，所以不知道是否正确
'''
import numpy as np
import pandas as pd
import math
import matplotlib as plt

def MultiModf(X):
    row,col=X.shape
    for i in range(row):
        for j in range(col):
            X[i,j]=(X[i,j]-int(X[i,j]))*10
            pass
    return X
    pass
def arrayMulti(X):
    result=1
    for i in range(X.size):
        result*=X[i]
        pass
    return result

class Bayes(object):
    def  __init__(self,dim):#接收你的输入维度，初始化各个概率
        self.data_dim=dim
        self.fis_0=np.zeros((1,dim),dtype=np.int16).T
        self.fis_1=np.zeros((1,dim),dtype=np.int16).T
        self.fis_y=0
        #return super().__init__(self)
    def getData(self):
        file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        df = pd.read_csv(file,header=None)  #header=None 数据第一行是有用数据，不是表头
        self.Y = np.array(df.loc[0:149,4].values)
        self.Y = np.where(self.Y == 'Iris-setosa',0,1)
        self.X = np.array(df.loc[0:149,0:3].values)
        self.X = np.where(MultiModf(self.X)>5,0,1)
        self.data_len=len(self.Y);
        pass
    def fit(self):
        for i in range(self.data_dim):
            self.fis_0[i]=np.sum(np.where((self.X[:,i]==1 & self.Y==0))[0])/np.sum(np.where(self.Y==0)[0])
            self.fis_1[i]=np.sum(np.where((self.X[:,i]==1 & self.Y==1))[0])/np.sum(np.where(self.Y==1)[0])
            pass
        self.fis_y=np.sum(np.where(self.Y==1)[0])/self.data_len
        pass
    def predict(self,X):
        fi_n=1-self.fis_1
        index_1=where(X==1)
        index_0=where(X==0)
        fi=[self.fis_1[index_1],fis_n[index_0]]
        p_1=arrayMulti(fi)*self.fis_y
        fi_n=1-self.fis_0
        index_1=where(X==1)
        index_0=where(X==0)
        p_2=fi=[self.fis_0[index_1],fis_n[index_0]]
        p_general=p_1/(p_1+p_2)
        return p_general
    pass
bayes= Bayes(150)
bayes.getData()
bayes.fit()
p=bayes.predict(X=[1,0,1,0])
print("the probability of the spam email is %f" % p)