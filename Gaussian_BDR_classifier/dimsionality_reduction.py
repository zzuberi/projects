import matplotlib.pyplot as plt
import numpy as np


class DimReduce:

    def __init__(self):
        self.transform = None
        self.s = None
        self.max_dim = None
        self.mu = None

    def pca_1(self, data):
        d = data.shape[0]
        n = data.shape[1]
        self.mu = data @ np.ones((n, 1)) / n
        data_c = data - self.mu @ np.ones((1, n))
        _, p, vh = np.linalg.svd(data_c.T)
        self.transform = vh
        self.s = (p ** 2) / n
        self.max_dim = d

    def pca_2(self, data):
        d = data.shape[0]
        n = data.shape[1]
        self.mu = data @ np.ones((n, 1)) / n
        data_c = data - self.mu @ np.ones((1, n))
        cov = data_c @ data_c.T / n
        w, v = np.linalg.eig(cov)
        sorted_inds = (-w).argsort()
        self.transform = v[:, sorted_inds].T
        self.s = w[sorted_inds]
        self.max_dim = d

    def apply(self, data, dim):
        if dim > self.max_dim:
            raise
        data_c = (data.T - self.mu.T).T
        return self.transform[0:dim, :] @ data_c

    def var_ratio_graph(self):
        denom = np.sum(self.s ** 2)
        x = np.zeros(self.max_dim)
        for i in range(self.max_dim):
            x[i] = (self.s[0:i] ** 2).sum()
        x = x / denom
        plt.plot(x, range(self.max_dim))
        plt.show()
