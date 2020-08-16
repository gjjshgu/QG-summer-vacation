import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def __init__(self, lamb=1):
        self.lamb = lamb  # 贝叶斯估计的参数
        self.prior = dict()  # 存储先验概率
        self.conditional = dict()  # 存储条件概率

    def fit(self, X_train, y_train):
        """
        根据朴素贝叶斯算法原理,使用 贝叶斯估计 计算先验概率和条件概率
        特征集集为离散型数据,预测类别为多元.  数据集格式为np.array
        :param X_train: 特征集m*n,m为样本数,n为特征数
        :param y_train: 标签集m*1
        :return: 不返回任何值,更新成员变量
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        m, n = X_train.shape
        labels = Counter(y_train.flatten().tolist())  # 计算各类别的样本个数
        k = len(labels.keys())  # 类别数
        for label, amount in labels.items():
            self.prior[label] = (amount + self.lamb) / (m + k * self.lamb)  # 计算平滑处理后的先验概率
        for feature in range(n):  # 遍历每个特征
            self.conditional[feature] = {}
            values = np.unique(X_train[:, feature])
            for value in values:  # 遍历每个特征值
                self.conditional[feature][value] = {}
                for label, amount in labels.items():  # 遍历每种类别
                    feature_label = X_train[y_train[:, 0] == label, :]  # 截取该类别的数据集
                    c_label = Counter(feature_label[:, feature].flatten().tolist())  # 计算该类别下各特征值出现的次数
                    self.conditional[feature][value][label] = (c_label.get(value, 0) + self.lamb) / \
                                                              (amount + len(values) * self.lamb)  # 计算平滑处理后的条件概率

    def _predict(self, X_test):
        """预测单个样本"""
        best_poster, best_label = -np.inf, -1
        for label in self.prior:
            poster = np.log(self.prior[label])  # 初始化后验概率为先验概率,同时把连乘换成取对数相加，防止下溢（即太多小于1的数相乘，结果会变成0）
            for feature in range(X_test.shape[0]):
                try:
                    poster += np.log(self.conditional[feature][X_test[feature]][label])
                except :
                    poster += np.log(self.lamb / len(X_test[feature]))
            if poster > best_poster:  # 获取后验概率最大的类别
                best_poster = poster
                best_label = label
        return best_label

    def predict(self, X_test):
        return np.array([self._predict(i) for i in X_test])


class GaussianNBClassifier:
    def __init__(self):
        self.mean = None
        self.var = None
        self.prior = None
        self.n_classes = None

    def fit(self, X_train, y_train):
        labels = Counter(y_train)
        # 结果的先验概率
        self.prior = np.array([labels[i] / len(y_train) for i in range(len(labels))])
        # 多少类，设为k
        self.n_classes = len(self.prior)
        # 均值与方差，设特征有n个，得到的矩阵为n*k
        self.mean = np.array([X_train[y_train == i].mean(axis=0) for i in range(self.n_classes)])
        self.var = np.array([X_train[y_train == i].var(axis=0) for i in range(self.n_classes)])

    def _predict(self, X_test):
        # 高斯函数
        _get_likelihood = lambda row: (1 / np.sqrt(2 * np.pi * self.var) * np.exp(-(row - self.mean) ** 2 / \
                                                                                  (2 * self.var))).prod(axis=1)
        # 似然的连乘
        likelihood = np.apply_along_axis(_get_likelihood, axis=1, arr=X_test)
        # 再乘上先验概率
        probs = self.prior * likelihood
        # return probs
        # 获得对应的条件概率
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, X_test):
        return self._predict(X_test).argmax(axis=1)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # nb = NaiveBayes()
    nb = GaussianNBClassifier()
    nb.fit(X_train, y_train)
    res = nb.predict(X_test)
    print(np.sum(nb.predict(X_test) == y_test) / len(y_test))