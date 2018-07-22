#source https://raw.githubusercontent.com/wepe/MachineLearning/master/SVM/SVM_by_SMO/SVCSMO.py
#这边的测试数据跟训练数据还是相同的，有条件可以自己再找一组数据来测试
from __future__ import division, print_function
from numpy import linalg
import os
import numpy as np
import random as rnd
filepath = os.path.dirname(os.path.abspath(__file__))
import csv, os, sys
class SVCSMO():
    """
        Simple implementation of a Support Vector Classification using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001, sigma=5.0):
        """
        :param max_iter: maximum iteration
        :param kernel_type: Kernel type to use in training.
                        'linear' use linear kernel function.
                        'quadratic' use quadratic kernel function.
                        'gaussian' use gaussian kernel function
        :param C: Value of regularization parameter C
        :param epsilon: Convergence value.
        :param sigma: parameter for gaussian kernel
        """
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'gaussian' : self.kernel_gaussian
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma
    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                #这边是随机选择i与j
                #建议：
                #每次迭代都要选择最好的αi和αj，
                #为了更快的收敛！那实践中每次迭代到底要怎样选αi和αj呢？
                #采用启发式选择，主要思想是先选择最有可能需要优化（也就是违反KKT条件最严重）的αi，
                #再针对这样的αi选择最有可能取得较大修正步长的αj。
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                #这边对核函数进行了打包
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)
                # E_i is u_i plus y_i
                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                #unc->c
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count
    def predict(self, X):
        #这边的h=sign(wTX+b)
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        #https://img-blog.csdn.net/20170125183034843?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2lsbGJraW1wcw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center
        #上面是公式位置
        #具体可以看博客 https://blog.csdn.net/willbkimps/article/details/54697698 公式6
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        #https://img-blog.csdn.net/20170125183112500?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2lsbGJraW1wcw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center
        #上面是公式
        #具体可以看博客 https://blog.csdn.net/willbkimps/article/details/54697698 公式8
        return np.dot(alpha * y, X)
    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def get_rnd_int(self, a,b,z):
        i = z
        while i == z:
            i = rnd.randint(a,b)
        return i
    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
    def kernel_gaussian(self, x1, x2, sigma=5.0):
        if self.sigma:
            sigma = self.sigma
        return np.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

def calc_mse(y, y_hat):
    return np.nanmean(((y - y_hat) ** 2))

def test_main(filename='data/iris-virginica.txt', C=1.0, kernel_type='linear', epsilon=0.001):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)

    # Split data
    X, y = data[:,0:-1], data[:,-1].astype(int)

    # Initialize model
    model = SVCSMO()

    # Fit model
    support_vectors, iterations = model.fit(X, y)

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)
    mse = calc_mse(y, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("mse:\t%.3f" % (mse))
    print("Converged after %d iterations" % (iterations))

if __name__ == '__main__':
    param = {}
    param['filename'] = './small_data/iris-slwc.txt'
    param['C'] = 0.1
    param['kernel_type'] = 'linear'
    param['epsilon'] = 0.001


    test_main(**param)