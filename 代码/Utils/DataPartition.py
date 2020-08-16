import numpy as np


def train_test_split(X, y, test_ratio, seed=None): # 留出法
    """
    :param X: 特征集
    :param y: 结果集
    :param test_ratio: 测试集占比
    :param seed: 随机种子
    :return:
    """
    if seed: np.random.seed(seed)
    a = len(X)
    shuffled_indexes = np.random.permutation(a)
    shuffled_train = shuffled_indexes[:int(a*(1-test_ratio))]
    shuffled_test = shuffled_indexes[int(a*(1-test_ratio)):a]
    X_train = np.array([X[i] for i in shuffled_train])
    X_test = np.array([X[i] for i in shuffled_test])
    y_train = np.array([y[i] for i in shuffled_train])
    y_test = np.array([y[i] for i in shuffled_test])
    return X_train, X_test, y_train, y_test

def cross_validation(algorihm, X, y, n): #交叉验证法
    """
    :param algorihm: 传入算法的类
    :param X: 特征集
    :param y: 结果集
    :param n: 划分个数
    :return:
    """
    def accuracy_score(y_true, y_predict):
        return np.sum(y_true == y_predict) / len(y_true)
    all = np.random.permutation(len(X))
    size = len(X) // n
    s = []
    for i in range(0, len(X), size):
        a = all[i:i+size]
        s.append(a)
    result = np.empty((n,))
    for i in range(len(s)):
        b = s.copy()
        verify = b.pop(i)
        train = np.hstack(b)
        verify_real_X = np.array([X[i] for i in verify])
        verify_real_y = np.array([y[i] for i in verify])
        train_real_X = np.array([X[i] for i in train])
        train_real_y = np.array([y[i] for i in train])
        algorihm.fit(train_real_X, train_real_y)
        verify_predict = algorihm.predict(verify_real_X)
        result[i] = accuracy_score(verify_real_y, verify_predict)
    print(np.mean(result))

def bootstraping(X, y): #自助法
    """
    :param X: 特征集
    :param y: 结果集
    :return:
    """
    a = len(X)
    test_index = np.empty((a,), dtype=int)
    for i in range(a):
        test_index[i] = np.random.randint(a)
    X_train = np.array([X[i] for i in test_index])
    y_train = np.array([y[i] for i in test_index])
    X_test = []
    y_test = []
    for i in range(a):
        if i not in test_index:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train, X_test, y_train, y_test