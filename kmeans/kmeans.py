import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata


class KMeans:

    def __init__(self):
        self.means = None

    def train(self, x, k, thresh=.001, max_iters=300):
        d, _ = x.shape
        data = x / np.linalg.norm(x, axis=0)
        self.means = np.random.rand(d, k)
        self.means = self.means / np.linalg.norm(self.means, axis=0)

        means_old = np.full(self.means.shape, 1)
        means_old = means_old / np.linalg.norm(means_old, axis=0)

        diff = np.sum(abs(self.means - means_old))
        diff_old = diff + thresh + 1

        iters = 0
        while abs(diff - diff_old) > thresh and iters < max_iters:
            print(abs(diff - diff_old))
            print(iters, end='\r')
            means_old = self.means
            pred = np.argmax(self.means.T @ data, axis=0)

            for i in range(k):
                indicies = np.where(pred == i)[0]
                self.means[:, i] = np.average(data[:, indicies], axis=1)
            self.means = self.means / np.linalg.norm(self.means, axis=0)

            diff_old = diff
            diff = np.sum(abs(self.means - means_old))
            iters += 1


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', transpose_data=False)
    kmeans = KMeans()
    kmeans.train(mnist.data, 10)
    for i in range(10):
        plt.imshow(kmeans.means[:, i].reshape((28, 28)))
        plt.show()
