import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.utils import shuffle


def clamp_sample(x):
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x


class NBC:
    def fit(self, X, Y):
        # assume classes are numbered 0...K-1
        self.K = len(set(Y))

        self.gaussians = []
        self.p_y = np.zeros(self.K)
        for k in range(self.K):
            Xk = X[Y == k]
            self.p_y[k] = len(Xk)
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            g = {'m': mean, 'c': cov}
            self.gaussians.append(g)
        # normalize p(y)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        gaussian = self.gaussians[y]
        sample = mvn.rvs(mean=gaussian['m'], cov=gaussian['c'])
        return clamp_sample(sample)

    def sample(self):
        y = np.random.choice(self.K, p=self.p_y)
        gaussian = self.gaussians[y]
        sample = mvn.rvs(mean=gaussian['m'], cov=gaussian['c'])
        return clamp_sample(sample)


if __name__ == '__main__':
    df = pd.read_csv('machine_learning_examples/large_files/train.csv')
    data = df.values
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]
    X, Y = shuffle(X, Y)
    clf = NBC()
    clf.fit(X, Y)
    for k in range(clf.K):
        # show one sample for each class
        # also show the mean image learned

        sample = clf.sample_given_y(k).reshape(28, 28)
        mean = clf.gaussians[k]['m'].reshape(28, 28)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    sample = clf.sample().reshape(28, 28)
    plt.figure()
    plt.imshow(sample, cmap='gray')
    plt.title("Sample")
    plt.show()


