#source https://github.com/hitcszq/machine_learning_basic_algo/blob/master/pca_final.py
#推荐blog https://blog.csdn.net/lyl771857509/article/details/79435402  或者  https://www.cnblogs.com/sweetyu/p/5085798.html
from numpy import *  
#coding: utf-8      
def pca(dataMat, K=5):  
        meanVals = mean(dataMat,axis=0) 
        meanRemoved = dataMat - meanVals 
        stded = meanRemoved / std(dataMat,axis=0)  
        covMat = cov(stded, rowvar=False) #注意：cov的rowvar参数对于array的用法跟手册一样，但对于mat则相反
        eigVals, eigVects = linalg.eig(mat(covMat)) 
        eigValInd = argsort(eigVals) #得到从小到大的指数矩阵  
        '''
            index[-2:]
            array([4, 2], dtype=int64)
            index[2:]
            array([3, 4, 2], dtype=int64)
            a=[2,3,54,12,34,53,65,233,342]
            index=np.argsort(a)
            index[-2:]
            array([7, 8], dtype=int64)
            index[2:]
            array([3, 4, 5, 2, 6, 7, 8], dtype=int64)
        '''
        eigValInd = eigValInd[-K:] #选择最大的K个index   
        seleEigVects = eigVects[:, eigValInd] #得到特征向量      
        lowDDataMat = stded * seleEigVects    
        return lowDDataMat

randArray = random.random(size=(10,8))
print(randArray.shape)
a=pca(randArray)
print(a)
print(a.shape)
