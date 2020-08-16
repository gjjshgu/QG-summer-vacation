import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SPCA

class PCA:
    def __init__(self, n_components):
        self._w = None
        self.n_components = n_components
        self.components_ = None
    def fit(self, X, eta=0.03, n_iters=1e4):
        """
        pca训练
        :param X: 特征集
        :param eta: 学习率
        :param n_iters: 最大迭代次数
        :return:
        """
        assert self.n_components <= X.shape[1]
        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w): #转化为单位向量（除以模）
            return w / np.linalg.norm(w)

        def gradient_ascent(X, initial_w, eta, epsilon=1e-7):
            w = direction(initial_w)
            i = 0
            for _ in range(int(n_iters)):
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    i=1
                    break
            if i == 0:
                raise ValueError()
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = gradient_ascent(X_pca, initial_w, eta)
            self.components_[i,:] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w
        return self

    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

if __name__ == '__main__':
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    pca = PCA(10)
    pca.fit(X)
    x1 = pca.transform(X)
    a = np.std(x1, axis=0)
    print(a)
    # print(pca.components_)

    plt.plot(np.arange(len(a)), a)
    pca1 = SPCA(n_components=10)
    pca1.fit(X)
    b = np.std(pca1.transform(X), axis=0)
    plt.plot(np.arange(len(b)), b, color='r')
    plt.show()