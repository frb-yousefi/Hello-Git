# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy.special import gammaln, digamma
from ...util.linalg import pdinv
from domains import _REAL, _POSITIVE
import warnings
import weakref

class Prior(object):
    domain = None

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def plot(self):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import priors_plots
        priors_plots.univariate_plot(self)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

class Gaussian(Prior):
    """
    Implementation of the univariate Gaussian probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []
    def __new__(cls, mu, sigma): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "N(" + str(np.round(self.mu)) + ', ' + str(np.round(self.sigma2)) + ')'

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(x - self.mu) / self.sigma2

    def lnpdf_grad(self, x):
        return -(x - self.mu) / self.sigma2

    def rvs(self, n):
        return np.random.randn(n) * self.sigma + self.mu


class Uniform(Prior):
    domain = _REAL
    _instances = []
    def __new__(cls, lower, upper): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().lower == lower and instance().upper == upper:
                    return instance()
        o = super(Prior, cls).__new__(cls, lower, upper)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    
    def __init__(self, lower, upper):
        self.lower = float(lower)
        self.upper = float(upper)
    
    def __str__(self):
        return "[" + str(np.round(self.lower)) + ', ' + str(np.round(self.upper)) + ']'

    def lnpdf(self, x):
        region = (x>=self.lower) * (x<=self.upper)
        return region

    def lnpdf_grad(self, x):
        return np.zeros(x.shape)

    def rvs(self, n):
        return np.random.uniform(self.lower, self.upper, size=n)
    
class LogGaussian(Prior):
    """
    Implementation of the univariate *log*-Gaussian probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, mu, sigma): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "lnN(" + str(np.round(self.mu)) + ', ' + str(np.round(self.sigma2)) + ')'

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(np.log(x) - self.mu) / self.sigma2 - np.log(x)

    def lnpdf_grad(self, x):
        return -((np.log(x) - self.mu) / self.sigma2 + 1.) / x

    def rvs(self, n):
        return np.exp(np.random.randn(n) * self.sigma + self.mu)


class MultivariateGaussian:
    """
    Implementation of the multivariate Gaussian probability function, coupled with random variables.

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []
    def __new__(cls, mu, var): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu == mu) and np.all(instance().var == var):
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, var)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, var):
        self.mu = np.array(mu).flatten()
        self.var = np.array(var)
        assert len(self.var.shape) == 2
        assert self.var.shape[0] == self.var.shape[1]
        assert self.var.shape[0] == self.mu.size
        self.input_dim = self.mu.size
        self.inv, self.hld = pdinv(self.var)
        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi) - self.hld

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):
        d = x - self.mu
        return self.constant - 0.5 * np.sum(d * np.dot(d, self.inv), 1)

    def lnpdf_grad(self, x):
        d = x - self.mu
        return -np.dot(self.inv, d)

    def rvs(self, n):
        return np.random.multivariate_normal(self.mu, self.var, n)

    def plot(self):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import priors_plots
        priors_plots.multivariate_plot(self)

def gamma_from_EV(E, V):
    warnings.warn("use Gamma.from_EV to create Gamma Prior", FutureWarning)
    return Gamma.from_EV(E, V)


class Gamma(Prior):
    """
    Implementation of the Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, a, b): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        o = super(Prior, cls).__new__(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "Ga(" + str(np.round(self.a)) + ', ' + str(np.round(self.b)) + ')'

    def summary(self):
        ret = {"E[x]": self.a / self.b, \
            "E[ln x]": digamma(self.a) - np.log(self.b), \
            "var[x]": self.a / self.b / self.b, \
            "Entropy": gammaln(self.a) - (self.a - 1.) * digamma(self.a) - np.log(self.b) + self.a}
        if self.a > 1:
            ret['Mode'] = (self.a - 1.) / self.b
        else:
            ret['mode'] = np.nan
        return ret

    def lnpdf(self, x):
        return self.constant + (self.a - 1) * np.log(x) - self.b * x

    def lnpdf_grad(self, x):
        return (self.a - 1.) / x - self.b

    def rvs(self, n):
        return np.random.gamma(scale=1. / self.b, shape=self.a, size=n)
    @staticmethod
    def from_EV(E, V):
        """
        Creates an instance of a Gamma Prior  by specifying the Expected value(s)
        and Variance(s) of the distribution.
    
        :param E: expected value
        :param V: variance
        """
        a = np.square(E) / V
        b = E / V
        return Gamma(a, b)

class inverse_gamma(Prior):
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    def __new__(cls, a, b): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        o = super(Prior, cls).__new__(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "iGa(" + str(np.round(self.a)) + ', ' + str(np.round(self.b)) + ')'

    def lnpdf(self, x):
        return self.constant - (self.a + 1) * np.log(x) - self.b / x

    def lnpdf_grad(self, x):
        return -(self.a + 1.) / x + self.b / x ** 2

    def rvs(self, n):
        return 1. / np.random.gamma(scale=1. / self.b, shape=self.a, size=n)

class LDAprior:
    """
    Implementation of the Discriminative GPLVM

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []
    def __init__(self, sigma2, data, X_shape):
        self.sigma2 = sigma2
        self.data = data
        self.X_shape = X_shape

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):

        X = x.reshape(self.X_shape)
        # Compute J
        #J =
        #data = GPy.util.datasets.oil_100()

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
        return np.linalg.inv(J)/(-self.sigma2)

    def lnpdf_grad(self, x):
        # Compute dL_prior/dX
        return -np.dot(self.inv, d)

    def rvs(self, n):
        raise NotImplementedError
