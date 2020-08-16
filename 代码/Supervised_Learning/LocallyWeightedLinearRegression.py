import numpy as np
import matplotlib.pyplot as plt

def load_data():
    m = 100
    np.random.seed(42)
    x_values = 100 * np.random.rand(m, 1)
    x_values = np.sort(x_values, axis=0)
    X = x_values.reshape(-1, 1)
    # X = np.hstack((np.ones((100, 1)), X))
    y = 7 * np.sin(0.12 * x_values) + x_values + 2 * np.random.randn(m, 1)
    return X, y
class LWLR:
    def __init__(self):
        self.m = None
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]
        #
        # def calculate_theta(X, k):
        #     w = np.eye(self.m, self.m)
        #     for i in range(self.m):
        #         w[i, i] = np.exp(-np.sum(np.square(X_train[i] - X)) / (2 * k ** 2))
        #     theta = np.linalg.inv(X_train.T.dot(w).dot(X_train)).dot(w).dot(y_train)
        #     return theta
        self.m = X_train.shape[0]
        self.X = np.hstack((np.ones((self.m, 1)), X_train))
        self.y = y_train.reshape(-1, 1)
    def predict(self, X_test, k):
        def calculate_theta(x_test, k):
            # 构造矩阵 W
            W = np.eye(self.m, self.m)
            for i in range(self.m):
                W[i, i] = np.exp(- np.sum(np.square(self.X[i] - x_test)) / (2 * k ** 2))
            # 应用局部加权线性回归，求解 theta
            theta = np.linalg.inv(self.X.T.dot(W).dot(self.X)).dot(self.X.T).dot(W).dot(self.y)
            return theta

        def predict(x_test, k):
            theta = calculate_theta(x_test, k)
            y_pred = theta[0] + x_test * theta[1]
            return y_pred

        y_predict = np.empty((X_test.shape[0], ))
        for i in range(len(X_test)):
            y_predict[i] = predict(X_test[i], 0.3)
        return y_predict

if __name__ == '__main__':
    k = 100
    X,y = load_data()
    X = X.reshape(-1, 1)
    lwlr = LWLR()
    lwlr.fit(X, y)
    y_pr = lwlr.predict(X, k)

    plt.scatter(X, y)
    plt.plot(X, y_pr, color='r')
    plt.show()



