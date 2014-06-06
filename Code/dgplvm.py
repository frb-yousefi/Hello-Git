__author__ = 'frb'

import sys
sys.path.append('/home/frb/GPy/GPy')

import GPy
import numpy as np
import pylab as pb


def get_class_label(y):
    # for i in range(len(y)):
    #     if y[i] == 1:
    #         return i
    # return -1
    #
    for idx, v in enumerate(y):
        if v == 1:
            return idx
    return -1

def main():
    data = GPy.util.datasets.oil_100()

    # print type(data)

    X = data['X']
    Y = data['Y']

    cls = {}
    for i in xrange(Y.shape[0]):
        class_label = get_class_label(Y[i])
        if class_label not in cls:
            cls[class_label] = []
        cls[class_label].append(X[i])

    class_means = np.empty((Y.shape[1]))
    for c in cls:
        class_means[c] = np.mean(cls[c])

    #Calculating the mean of the whole training points of all classes.
    M_Total = np.mean(X, axis=0)

    Sw = np.zeros((X.shape[1], X.shape[1]))

    for c in cls:
        v = (class_means[c] - M_Total).reshape(X.shape[1], 1)
        v_tran = v.transpose()#.reshape(1, X.shape[1])
        Sw += float(len(cls[c])) / X.shape[0] * v.dot(v.transpose())

    # print Sw.shape

    Sb = np.zeros_like(Sw)
    for c in cls:
        ni = float(len(cls[c]))
        s = 0
        for xk in cls[c]:
            v = (xk - class_means[c]).reshape(X.shape[1], 1)
            s += v.dot(v.transpose())
        Sb += ni / X.shape[0] * ((1 / ni) * s)

    #def LDA_GDA_Energy_func(Sw,Sb):
    J = np.trace(np.linalg.inv(Sw).dot(Sb))
        #return J
    print J

    Sigma2 = np.power(10,-4)
    #P = np.exp((-1/Sigma2)*np.linalg.inv(J)) 
    #print P

if __name__ == '__main__':
    main()
