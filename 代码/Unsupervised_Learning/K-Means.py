import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self):
        self.m = None
        self.n = None
        self.datacent = None
        self.data = None

    def fit(self, X_train, k):
        """
        训练模型
        :param X_train: 训练的数据集
        :param k: k个簇
        :return:
        """
        def randomCent(X, k):
            """
            生成随机点
            :param X: 在X数据集中搜索
            :param k: k个簇心
            :return:
            """
            while True: #防止重复
                a = np.random.randint(self.m, size=k)
                if len(set(a)) == k:
                    break
            data_cent = [X[i] for i in a]
            data_cent = np.array(data_cent)
            return data_cent

        def distance(one_sample, data_cent):
            """
            计算所有点与簇心得距离
            :param one_sample: 一个点与k个簇心的距离
            :param data_cent: 簇心集
            :return: 返回最小距离和簇心索引
            """
            one_distance = np.sum(np.power((one_sample - data_cent), 2), axis=1)
            min_distance = one_distance.min()
            min_index = np.where(one_distance == min_distance)
            a = min_index[0][0]
            return min_distance, a

        self.m, self.n = X_train.shape
        n = self.n
        data_cent = randomCent(X_train, k)
        add = np.empty((self.m, 3))
        add[:, 0] = np.inf
        add[:, 1:3] = -1
        data_with_add = np.hstack((X_train, add))

        while True:
            count = [[] for i in range(k)] #记录每个簇的元素在data中的索引
            for i in range(self.m):
                min_values, min_index = distance(data_with_add[i, :n], data_cent)
                data_with_add[i, n] = min_values
                data_with_add[i, n+1] = min_index
                count[min_index].append(data_with_add[i, :n])
            if (data_with_add[:, -1] == data_with_add[:, -2]).all() : break
            for i in range(k):
                count[i] = np.array(count[i])
                data_cent[i] = np.mean(count[i], axis=0)
            data_with_add[:, -1] = data_with_add[:, -2]

        self.datacent = data_cent
        self.data = data_with_add
        return self

class BiKmeans:
    def __init__(self):
        pass

    def fit(self, X_train, k):
        """
        训练模型
        :param X_train: 训练集
        :param k: k个簇
        :return:
        """
        def one_to_two_divide(X):
            """
            二分训练
            :param X: 数据集
            :return: 字典，键为SSE，值为簇
            """
            km = Kmeans()
            km.fit(X, 2)
            data_0 = km.data[km.data[:, -1] == 0][:, :n]
            data_1 = km.data[km.data[:, -1] == 1][:, :n]
            SSE_0 = np.linalg.norm(data_0-km.datacent[0])
            SSE_1 = np.linalg.norm(data_1-km.datacent[1])
            return SSE_0, SSE_1, data_0, data_1

        def divide(X):
            """
            递归，依次找出最大SSE的簇再传入二分函数
            :param X: 数据集
            :return:
            """
            nonlocal dict
            sse0, sse1, data0, data1 = one_to_two_divide(X)
            dict[sse0] = data0
            dict[sse1] = data1
            if len(dict) == k: return
            a = np.max(list(dict.keys()))
            b = dict.pop(a)
            divide(b)

        m, n = X_train.shape
        dict = {}
        divide(X_train)
        return dict

if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
    bkm = BiKmeans()
    res = bkm.fit(X, 4)
    for i in res.values():
        plt.scatter(i[:, 0], i[:, 1],color='blue')
    plt.show()
