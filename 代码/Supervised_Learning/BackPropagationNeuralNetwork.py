import numpy as np
# from sklearn.datasets import load_iris
# import pandas as pd
# from TV_examine.data_preprocessing import preprocessing
# from TV_examine.model_assessment import divide_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetworkClassifier:
    def __init__(self, layers, eta=0.5, n_iters=1e5):
        """
        :param layers: 单元层数，传入一个列表，如[2,2,1]表示输入单元2个，中间单元1个，输出单元1个
        :param eta: 学习率，用于梯度下降
        :param n_iters: 最大迭代次数
        """
        self._weights = None
        self._b = None
        self.score = None
        self.layers = layers
        self.eta = eta
        self.n_iters = n_iters

    def fit(self, X_train, y_train):
        """
        :param X_train: 训练特征集
        :param y_train: 训练结果集
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0]
        # y_train = y_train.reshape((-1, 1))
        weights = []
        b = []  # 行向量的集合
        for i in range(len(self.layers) - 1):
            weights.append(np.random.random((self.layers[i + 1], self.layers[i])))  # 初始化权重
            b.append(np.random.random((1, self.layers[i + 1])))  # 初始化偏置单元

        def sigmoid_derivative(x):  # 输入的是a不是z
            return x * (1 - x)

        def J(y_predict, y_train):
            # return np.sum((y_predict - y_train) ** 2)
            return np.sum(-(y_train * np.log(y_predict) + (1-y_train) * np.log(1-y_predict)))

        for _ in range(int(self.n_iters)):
            i = np.random.randint(X_train.shape[0])
            a = self._positive_way(weights, b, X_train[i])
            error = a[-1] - y_train[i]
            delta = [error * sigmoid_derivative(a[-1])]  # 元素全都是行向量
            error0 = J(a[-1], y_train[i])
            # print("前：",error0)

            for l in range(len(a) - 2, 0, -1):  # 反向传播
                x = delta[-1].reshape(-1, 1)
                res = weights[l].T.dot(x) * sigmoid_derivative(a[l].reshape(-1, 1))
                delta.append(res.reshape((-1,)))
            delta.append(0)  # 输入层没有偏差
            delta.reverse()

            for l in range(len(weights)):  # 更新值
                res_w = delta[l + 1].reshape(-1, 1).dot(a[l].reshape(1, -1))
                weights[l] = weights[l] - self.eta * res_w
                res_b = delta[l + 1]
                b[l] = b[l] - self.eta * res_b

            a1 = self._positive_way(weights, b, X_train[i])
            error1 = J(a1[-1], y_train[i])
            # print("后：", error1)

        self._weights = weights
        self._b = b

    def _positive_way(self, weights, b, first_element):  # 正向传播
        a = [first_element]
        for l in range(len(weights)):
            x = a[l].reshape((-1, 1))
            res = sigmoid((weights[l].dot(x)) + b[l].reshape(-1, 1))
            a.append(res.reshape((-1,)))
        return a

    def predict(self, x_test):
        # assert self._weights, self._b is not None

        if self._weights is None and self._b is None:
            print("failed")
            return
        a = []
        for x in x_test:
            result = self._positive_way(self._weights, self._b, x)
            # if result[-1] > 0.5:
            #     a.append(1)
            # else:
            #     a.append(0)
            a.append(result[-1])
        return np.argmax(a, axis=1)

    def get_score(self, x_test):
        # assert self._weights, self._b is not None
        if self._weights and self._b is None:
            print("failed")
            return
        a = []
        for x in x_test:
            result = self._positive_way(self._weights, self._b, x)
            if result[-1]>0.5: a.append(1)
            else: a.append(0)
            a.append(result[-1][0])
        return a

    def accurate_rate(self, y_predict, y_test):
        return np.sum(y_predict == y_test) / len(y_test)

# if __name__ == '__main__':
#     iris = load_iris()
#     X = iris.data
#     y = iris.target
#     y = pd.get_dummies(y).values
#     sb = preprocessing.Standardization()
#     X = sb.fit_transform(X)
#     X_train, X_test, y_train, y_test = divide_data.train_test_split(X, y, 0.2)
#     nn = NeuralNetwork()
#
#     nn.fit(X_train, y_train, [4,5,3])
#     y_pr = nn.predict(X_test)
#     print(y_pr)
#     y2 = np.argmax(y_test, axis=1)
#     print(y2)
#     print(nn.accurate_rate(y_pr, y2))
