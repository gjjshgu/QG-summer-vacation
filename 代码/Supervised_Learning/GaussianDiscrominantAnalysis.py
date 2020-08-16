import numpy as np
import matplotlib.pyplot as plt


class GDAClassifier:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):
        """
        训练模型
        :param X_train: 特征集
        :param y_train: 结果集
        :return:
        """
        def count_mean(X):
            return np.mean(X, axis=0)
        X1 = X_train[y_train == 1]
        X0 = X_train[y_train == 0]
        u_1 = count_mean(X1)
        u_0 = count_mean(X0)
        Sw = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False)
        # Sb = np.matmul((u_0-u_1).reshape(-1, 1), (u_0-u_1).reshape(1, -1))
        self.w = np.dot(np.linalg.inv(Sw), u_0-u_1)
        self.b = -0.5 * (u_0+u_1).dot(self.w)

    def predict(self, X_test):
        return (X_test).dot(self.w.reshape(-1, 1)) + self.b

if __name__ == '__main__':
    x_0 = np.array(np.random.normal(size=(100, 2)) * 5 + 20)
    x_1 = np.array(np.random.normal(size=(100, 2)) * 5 + 2)
    X = np.vstack((x_0, x_1))
    y = np.array(np.hstack((np.zeros(100), np.ones(100))))
    lda = GDAClassifier()
    lda.fit(X, y)
    y_pr = np.squeeze(lda.predict(X))
    x_0 = X[y_pr > 0]
    x_1 = X[y_pr < 0]
    plt.scatter(x_0[:, 0], x_0[:, 1], color='r')
    plt.scatter(x_1[:, 0], x_1[:, 1])
    plt.show()