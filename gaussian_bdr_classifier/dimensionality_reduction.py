import matplotlib.pyplot as plt
import numpy as np


class DimReduce:

    def __init__(self):
        self.transform = None

    def train(self, data):
        raise NotImplementedError

    def apply(self, data):
        raise NotImplementedError


class PCA(DimReduce):

    def __init__(self):
        super().__init__()
        self.s = None
        self.max_dim = None
        self.mu = None

    def train(self, data, pca=1):
        if pca == 1:
            self.pca_1(data)
        elif pca == 2:
            self.pca_2(data)

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


class LDA(DimReduce):

    def __init__(self):
        super().__init__()

    def train(self, data, labels, gamma=0):
        classes = np.unique(labels)
        classes.sort()
        self.transform = []
        for i, c in enumerate(classes):
            cls1 = np.where(labels == c)[0]
            for j in range(i + 1, classes.shape[0]):
                cls2 = np.where(labels == classes[j])[0]
                data1 = data[:, cls1]
                data2 = data[:, cls2]
                cov1 = np.cov(data1)
                cov2 = np.cov(data2)
                mu1 = np.mean(data1, axis=1)
                mu2 = np.mean(data2, axis=1)
                w = np.linalg.inv(cov1 + cov2 + gamma * np.eye(cov1.shape[0])) @ (mu1 - mu2)
                self.transform.append(w)
        self.transform = np.array(self.transform).squeeze()

    def apply(self, data):
        return self.transform @ data
