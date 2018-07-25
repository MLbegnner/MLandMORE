
#coding:utf-8
#source https://raw.githubusercontent.com/hitcszq/machine_learning_basic_algo/master/em.py
#我的妈呀！！！
#你可以试下他原本的代码，然后迭代1000次，你特么会发现不能收敛，至于为什么，你先思考下，再来问我，我联系方式github主页应该有
#这写源代码的老哥似乎没搞懂这个算法啊。。。
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = False
def ini_data(Sigma1,Sigma2,Mu1,Mu2,k,N):
    #这边生成两个独立的高斯分布数据，一个是N~(1,100)，一个是N~(100,100)
    #建议使用s = np.random.normal(mu, sigma, （size）)这种方式产生数据
    xeq_1=0;
    xeq_0=0;
    global X
    global Mu
    global E
    global Sigma
    global pi
    pi=[]
    pi.append(0.5)
    pi.append(0.5)
    #pi =[0.5 , 0.5]
    X = np.zeros((1,N))
    Mu = []
    Sigma=[]
    Sigma.append(10)
    Sigma.append(10)
    #sigma=[10 10]
    Mu.append(1)
    Mu.append(100)
    #mu=[1 100]
    E = np.random.random((N,k))
    #产生N*k的高斯分布数据，N是数据数量，k的cluster数量，代表的意义是给定x(i)的条件下z(i)=j的概率，即他的先验概率
    for i in range(0,N):
        if np.random.random(1) > 0.5:
          X[0,i] = np.random.normal(Mu1,Sigma1)
          xeq_0+=1
        else:
          X[0,i] = np.random.normal(Mu2,Sigma2)
          xeq_1+=1
    if isdebug:
        print(X)
    print(xeq_0,xeq_1)

#下面是E与M的步骤，andrew的课程没有详细介绍，参照下面的博客
#https://blog.csdn.net/jinping_shi/article/details/59613054
#注意这边cs229讲义上面的1{Z(i)=k}这边代表的是z(i)取得k的概率，而不是跟之前的监督学习部分的那样是z(i)只能取0或者1,
#可能视频或者讲义中提及过，如果提及过请忽略这注释
#还有一点就是这边的sigma是标准差不是方差，也就是方差的根号，跟博客中区别一下
#突然想到可能andrew的方法是对的，也就是z(i)只能取0或者1，可以算一下，如果都是这么用，感觉应该差不多，但感觉似乎算不了
def e_step(k,N):
    global E
    global Mu
    global X
    global Sigma
    global pi
    for i in range(0,N):
        Denom = 0
        for j in range(0,k):
            Denom += pi[j]*math.exp((-1/(2*(float(Sigma[j]**2))))*(float(X[0,i]-Mu[j]))**2)         
        for j in range(0,k):
            Numer = pi[j]*math.exp((-1/(2*(float(Sigma[j]**2))))*(float(X[0,i]-Mu[j]))**2)
            E[i,j] = Numer / Denom
    if isdebug:
        print(E)
def m_step(k,N):
    global E
    global X
    for j in range(0,k):
        Numer = 0
        mosig=0  
        Denom = 0  #Nk 类别是k的数量
        for i in range(0,N):
            Numer += E[i,j]*X[0,i]
            Denom +=E[i,j]
            mosig +=E[i,j]*(((X[0,i])-Mu[j])**2)
        Mu[j] = Numer / Denom
        Sigma[j]=(mosig/Denom)**0.5
        pi[j]=Denom/N
        pass
    #if(abs(pi[0]-pi[1])>0.5):
    #   print("something wrong")
    #   pass
def run(Sigma1,Sigma2,Mu1,Mu2,k,N,iter_num,Epsilon):
    ini_data(Sigma1,Sigma2,Mu1,Mu2,k,N)
    if isdebug:
        print(Mu,Sigma,pi)
    old=objec(k,N)
    for i in range(iter_num):
        e_step(k,N)
        m_step(k,N)
        if isdebug:
            print(i,Mu,Sigma,pi)
        new=objec(k,N)

        if abs(new-old)<2.4e-1000:
                 break
        print("iteration %d : " % i)
        print("Mu is : ",Mu)
        print("Sigma is : " , Sigma)
        print("pi is", pi)


def predict(X_,k):
    prop = [0,0]
    global Mu
    global Sigma
    global pi
    Denom = 0
    for j in range(0,k):
        Denom += pi[j] * math.exp((-1 / (2 * (float(Sigma[j] ** 2)))) * (float(X_ - Mu[j])) ** 2)         
    for j in range(0,k):
        Numer = pi[j] * math.exp((-1 / (2 * (float(Sigma[j] ** 2)))) * (float(X_ - Mu[j])) ** 2)
        prop[j] = Numer / Denom
    print('date is x= %d ,the probability of z=0 is %f \n the probability of z=1 is %f' % (X_ ,prop[0],prop[1]))
    pass

def objec(k,N):
    #感觉这一步不严谨，他是把各个类的pi[j]*E[i,j]全部加起来为temE，然后所有数据的temE相乘，实际上我感觉这两个相乘没有任何实际意义
    #你可以发现他的epsilon根本没有使用，说明他自己也懵逼了，把epsilon自己弄成了2.4e-1000
    #但这一步是为了检查参数是否收敛是可以确定的
    #建议是 检查mu sigma pi 三者是否全部收敛，即三个的new与old值得差值小于你期望的epsilon
    global Sigma
    global Mu
    global E
    global pi
    resul=1.0
    for i in range(0,N):
        num1=0.0
        for j in range(0,k):
            num1=num1+pi[j]*E[i,j]
        resul=resul*num1
    if isdebug:
        print(resul)
    return resul

if __name__ == '__main__':
     run(Sigma1=5,Sigma2=8,Mu1=10,Mu2=100,k=2,N=1000,iter_num=100,Epsilon=0.00001)
     #plt.hist(X[0,:],50)
     #plt.show()
     predict(10,2);
     predict(120,2);