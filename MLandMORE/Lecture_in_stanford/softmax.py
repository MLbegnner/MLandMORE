#source https://blog.csdn.net/jiede1/article/details/76983938
#这边的数据训练跟预测都是同一个，数据我也不知道去哪里弄，我就不弄了

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from scipy.optimize import minimize

#load_iris ReturnValue features 
#[‘target_names’, ‘data’, ‘target’, ‘DESCR’, ‘feature_names’] 
#target_names : 分类名称 
#[‘setosa’ ‘versicolor’ ‘virginica’]
#target：分类（150个） 
#(150L,)
#[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
#0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
#1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 
#2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
#2 2]
#feature_names: 特征名称 
#(‘feature_names:’, [‘sepal length (cm)’, ‘sepal width (cm)’, ‘petal length (cm)’, ‘petal width (cm)’])
#data : 特征值 
#(150L, 4L)
#data[0]:[ 5.1 3.5 1.4 0.2]

data=load_iris()
dataSet=data['data']
classLabels=data['target']
m,n=dataSet.shape
k=len(np.unique(classLabels))
#打乱数据
listt=shuffle(np.arange(dataSet.shape[0]))
#shuffle([1,2,3,4,5])
#[2, 4, 1, 3, 5]
#把数组内容大乱，这边是行数，避免相同的class都在一个位置
dataSet=dataSet[listt]
classLabels=classLabels[listt]

def sigmoid(X):
    return 1/(1+np.exp(-X))

#theta.shape==(k,n+1)
#lenda是正则化系数/权重衰减项系数，alpha是学习率
def J(X,classLabels,theta,alpha,lenda): 
    bin_classLabels=label_binarize(classLabels,classes=np.unique(classLabels).tolist()).reshape((m,k))  #二值化 (m*k) 
    dataSet=np.concatenate((X,np.ones((m,1))),axis=1).reshape((m,n+1)).T   #转换为（n+1,m）,axis=1水平连接
    theta_data=theta.dot(dataSet)  #(k,m)
    theta_data = theta_data - np.max(theta_data)   #k*m
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)  #(k*m)
    #这边是h(theta)
    #print(bin_classLabels.shape,prob_data.shape
    cost = (-1 / m) * np.sum(np.multiply(bin_classLabels,np.log(prob_data).T)) + (lenda / 2) * np.sum(np.square(theta))  #标量
    #print(dataSet.shape,prob_data.shape)

#    ∇θjJ(θ)=−1m∑i=1m⎡⎣⎢∇θj∑l=1kI{yi=j}logeθjX∑l=1keθlX⎤⎦⎥
#=−1m∑i=1m[I{yi=j}∑l=1keθlXeθjX⋅eθjX⋅X⋅∑l=1keθlX−eθjX⋅eθjX⋅X∑l=1keθlX2]
#=−1m∑i=1mI{yi=j}∑l=1keθlX−eθjX∑l=1keθlX⋅X
#=−1m∑i=1m[(I{yi=j}−P(yi=j||X,θj))⋅X]
#公式位置 https://www.cnblogs.com/PowerTransfer/p/8506440.html
#这个博客里面对应的求梯度式子中的的这个X应该是X（i），感觉写错了，或者拿到中括号外面
#公式还是参考https://www.cnblogs.com/tornadomeet/archive/2013/03/23/2977621.html
#求梯度过程可以参考着上面的那个博客自己推到一下

    grad = (-1 / m) * (dataSet.dot(bin_classLabels - prob_data.T)).T + lenda * theta  #(k*N+1)
   
    return cost,grad

def train(X,classLabels,theta,alpha=0.1,lenda=1e-4,maxiter=1000):
    #options_ = {'maxiter': 400, 'disp': True}
    #result =minimize(J(X,classLabels,theta,alpha,lenda), theta, method='L-BFGS-B', jac=True, options=options_)
    #return result.x
    for i in range(maxiter):
        cost,grad=J(X,classLabels,theta,alpha,lenda)
        theta=theta-alpha*grad
    return theta

def predict(theta,testSet,testClass):  #testSet (m,n+1)
    prod = theta.dot(testSet.T)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)
    accuracy=0.0
    for i in range(len(testClass)):
        if testClass[i]==pred[i]:
            accuracy+=1

    return pred,float(accuracy/len(testClass))

def check_gradient(X,classLabels,theta,alpha,lenda,eplison=1e-4):    # gradient check
    cost= lambda theta:J(X,classLabels,theta,alpha,lenda)
    print("Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n")
    print(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            #print(i,j)
            theta_1=np.array(theta)
            theta_2=np.array(theta)
            theta_1[i,j]=theta[i,j]+eplison
            theta_2[i,j]=theta[i,j]-eplison
            num_cost_1,x=cost(theta_1)
            num_cost_2,x=cost(theta_2)
            num_grad=(num_cost_1-num_cost_2)/(2*eplison)
            x,grad=cost(theta)
            iff = np.linalg.norm(num_grad- grad[i,j]) / np.linalg.norm(num_grad + grad[i,j])
            print("the difference of the grad and num_grad: ",iff)
            #print("Norm of the difference between numerical and analytical num_grad (should be < 1e-7)\n")

theta=np.random.random((k,n+1))   #初始化theta
check_gradient(dataSet,classLabels,theta,alpha=0.1,lenda=1e-4,eplison=1e-4)  #首先进行梯度检验
theta_train=train(dataSet,classLabels,theta,alpha=0.1)    #训练数据,在这里我不太严格，将所有数据都用于训练了
testSet=np.concatenate((dataSet,np.ones((m,1))),axis=1).reshape((m,n+1))    #所有训练数据同时作为预测数据，不建议，我这只是为了方便

pred,accuracy=predict(theta_train,testSet,classLabels)  #预测
print("accuracy: ",accuracy)   #最终准确率在98%左右