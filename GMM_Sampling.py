import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.utils import shuffle
from sklearn.mixture import BayesianGaussianMixture as BGM

def clamp_sample(x):
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x


class BC:
    def fit(self, X, Y):
        # assume classes are numbered 0...K-1
        self.K = len(set(Y))

        self.gaussians = []
        self.p_y = np.zeros(self.K)
        for k in range(self.K):
            print("Fitting gmm", k)
            Xk = X[Y == k]
            self.p_y[k] = len(Xk)
            gmm = BGM(10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)
        # normalize p(y)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        gaussian = self.gaussians[y]
        sample, pred_class = gaussian.sample()
        mean = gaussian.means_[pred_class]
        return clamp_sample(sample).reshape(28, 28), mean.reshape(28, 28)

    def sample(self):
        y = np.random.choice(self.K, p=self.p_y)
        gaussian = self.gaussians[y]
        sample, pred_class = gaussian.sample()
        mean = gaussian.means_[pred_class]
        return clamp_sample(sample).reshape(28, 28), mean.reshape(28, 28)



if __name__ == '__main__':
    df = pd.read_csv('machine_learning_examples/large_files/train.csv')
    data = df.values
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]
    X, Y = shuffle(X, Y)
    clf = BC()
    clf.fit(X, Y)
    for k in range(clf.K):
        # show one sample for each class
        # also show the mean image learned

        sample, mean = clf.sample_given_y(k)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    sample, mean = clf.sample()
    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap='gray')
    plt.title("Corresponding Cluster Mean")
    plt.show()
