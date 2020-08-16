import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
from collections import Counter


class KNNRegressor:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_text):
        n_samples = self.X_train.shape[0]
        res = np.empty((X_text.shape[0],))
        for i in range(X_text.shape[0]):
            distance = np.empty((n_samples,))
            for j in range(n_samples):
                distance[j] = np.linalg.norm(X_text[i] - self.X_train[j], ord=2)
            idx = np.argsort(distance)[:self.k]
            res[i] = np.mean([self.y_train[k] for k in idx])
        return res


class KNNClassifier:
    def __init__(self, k):
        assert k>=1
        self.k=k
        self._x_train = None
        self._y_train = None
    def fit(self,x_train,y_train):
        """
        训练数据
        :param x_train: 特征集
        :param y_train: 结果集
        :return:
        """
        assert x_train.shape[0]==y_train.shape[0]
        assert self.k<=y_train.shape[0]
        self._x_train=x_train
        self._y_train=y_train
        return self
    def _predict(self,x_predict): #单数据
        assert self._x_train is not None
        assert self._y_train is not None
        assert x_predict.shape[0] == self._x_train.shape[1]
        distances=[]
        for x in self._x_train:
            distances.append(np.sqrt(np.sum((x - x_predict)**2)))
        sort = np.argsort(distances)
        top = [self._y_train[i] for i in sort[:self.k]]
        votes = Counter(top)
        y_predict = votes.most_common(1)[0][0]
        return y_predict

    def predict(self, x): #多数据
        result = []
        for x1 in x:
            result.append(self._predict(x1))
        return result

    def __repr__(self):
        return "KNN(k=%d)" % self.k

class KDNode:
    def __init__(self, data=None, label=None, left=None, right=None, axis=None, parent=None):
        """
        KD树节点定义
        :param data: 特征数据
        :param label: 结果数据
        :param left: 左孩子
        :param right: 右孩子
        :param axis: 分割线
        :param parent: 父节点
        """
        self.data = data
        self.left = left
        self.right = right
        self.label = label
        self.axis = axis
        self.parent = parent


class KNNclassifier_KDTree:
    def __init__(self):
        self.root = None
        # self.y_exist = False if y is None else True
        # self.create(X, y)

    def fit(self, X, y=None):
        def _create(X, axis, parent=None):
            """
            递归创建KD树
            :param X: 特征集
            :param axis: 分割线
            :param parent: 父节点
            :return:
            """
            nums = X.shape[0]
            if nums == 0: return None
            mid = (nums >> 1)
            index = np.argsort(X[:, axis])
            small = np.array([X[i] for i in index[:mid]])
            big = np.array([X[i] for i in index[mid+1:]])
            if self.y_exist:
                kd_node = KDNode(X[index[mid]][:-1], X[mid][-1], axis=axis, parent=parent)
            else:
                kd_node = KDNode(X[index[mid]], axis=axis, parent=parent)
            next_axis = (axis + 1) % k_dimensions
            kd_node.left = _create(small, next_axis, kd_node)
            kd_node.right = _create(big, next_axis, kd_node)
            return kd_node

        self.y_exist = False if y is None else True
        k_dimensions = X.shape[1]
        print("正在创建KD树......")
        if y is not None:
            X = np.hstack((X, y.reshape(-1, 1)))
        self.root = _create(X, 0)
        print("创建完毕")

    def search_k(self, k, point, ord=2):
        """
        寻找k个最邻近点
        :param k: k个点
        :param point: 目标点
        :param ord: 范数，默认是L2范数（欧氏距离）
        :return:
        """
        def dist(x):
            """
            计算x与point的距离
            :param x: 计算点
            :return:
            """
            return np.linalg.norm(x-point, ord=ord)

        def axis_distance(node):
            """
            计算当前点与分割线的距离
            :param node:
            :return:
            """
            return abs(node.data[node.axis] - point[node.axis])

        def update(node, distance):
            """
            更新最邻近点
            :param node: 节点
            :param distance: 距离
            :return:
            """
            nonlocal count, nodes
            if len(count) < k:
                count.append(distance)
                nodes.append(node)
            elif len(count) == k:
                max_index = np.argmax(count)
                if distance < count[max_index]:
                    count[max_index] = distance
                    nodes[max_index] = node

        def _search(node):
            """
            传入节点，递归搜索
            :param node: 搜索节点
            :return:
            """
            nonlocal count, nodes
            if node is None: return
            distance = dist(node.data)
            update(node, distance)
            if point[node.axis] < node.data[node.axis]:
                _search(node.left)
            else:
                _search(node.right)
            rang = axis_distance(node)
            if rang > np.max(count):
                return
            if point[node.axis] < node.data[node.axis]:
                _search(node.right)
            else:
                _search(node.left)
        count = []
        nodes = []
        _search(self.root)
        x = np.array([i.data for i in nodes])
        y = np.array([int(i.label) for i in nodes]) if self.y_exist is True else None
        return x, y

    def predict(self, X_test, k):
        """
        预测
        :param X_test: 预测数据集
        :param k: k个邻近点
        :return:
        """
        assert self.y_exist is True
        m = X_test.shape[0]
        y_pr = np.empty(m)
        for i in range(m):
            _x, y = self.search_k(k, X_test[i])
            y_pr[i] = y[np.argmax(np.bincount(y))]
        return y_pr


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from Utils.DataPreprocessing import Standardization
    from sklearn.model_selection import train_test_split
    from Utils.Assessiment import RegressionAssessment
    std = Standardization()
    boston = load_boston()
    X = std.fit_transform(boston.data)
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    reg = KNNRegressor(6)
    reg.fit(X_train, y_train)
    y_pr = reg.predict(X_test)
    ass = RegressionAssessment(y_predict=y_pr, y_test=y_test)
    ass.assessment()


# if __name__ == '__main__':
#     X = load_iris().data
#     y = load_iris().target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     kdt = KDTree(X_train, y_train)
#     y_predict = kdt.predict(X_test, 13)
#     print(np.sum(y_predict == y_test)/len(y_test))