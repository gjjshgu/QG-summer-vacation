import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def gini(y):
    length = y.shape[0]
    if length == 0:
        return 1
    unique = np.unique(y)
    res = 0.0
    for i in unique:
        res += (np.sum(y == i) / length) ** 2
    return 1 - res


class TreeNode:
    def __init__(self, left_child=None, right_child=None,
                 col=-1, value=None, data=None, result=None):
        self.left_child = left_child
        self.right_child = right_child
        self.col = col
        self.value = value
        self.data = data
        self.result = result


class ClassificationAndRegressionTree:
    def __init__(self, max_depth=np.inf, min_impurity=1e-7, min_sample=2, loss=None):
        self.root = None
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.min_sample = min_sample
        self.count_info = None
        self.majority_vote = None
        self.loss = loss

    def fit(self, X_train, y_train):
        def split_data(data, value, col):
            if isinstance(value, int) or isinstance(value, float):
                func = lambda sample: sample[col] >= value
            else:
                func = lambda sample: sample[col] == value
            X1 = np.array([sample for sample in data if func(sample)])
            X2 = np.array([sample for sample in data if not func(sample)])
            return X1, X2

        def create_tree(data, current_depth):
            m, n = data.shape
            best_gain = 0.0
            best_feature = None
            best_data = None
            if m >= self.min_sample and current_depth <= self.max_depth:
                for col in range(n-1):
                    data_col = data[:, col]
                    unique_col = np.unique(data_col)
                    for value in unique_col:
                        left, right = split_data(data, value, col)
                        if len(left) > 0 and len(right) > 0:
                            info_gain = self.count_info(left, right, data)
                            if info_gain > best_gain:
                                best_gain = info_gain
                                best_feature = (col, value)
                                best_data = (left, right)

            if best_gain > self.min_impurity:
                true_branch = create_tree(best_data[0], current_depth+1)
                false_branch = create_tree(best_data[1], current_depth+1)
                return TreeNode(true_branch, false_branch, col=best_feature[0], value=best_feature[1], data=data)

            else:
                return TreeNode(result=self.majority_vote(data))

        dataset = np.hstack((X_train, y_train.reshape(-1, 1)))
        self.root = create_tree(dataset, 0)
        return self.root

    def _predict(self, X_test):
        def find_label(tree):
            if tree.result is not None:
                return tree.result
            v = X_test[tree.col]
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.left_child
                else:
                    branch = tree.right_child
            else:
                if v == tree.value:
                    branch = tree.left_child
                else:
                    branch = tree.right_child
            return find_label(branch)
        return find_label(self.root)

    def predict(self, X_test):
        m = X_test.shape[0]
        res = np.empty(m)
        for i in range(m):
            res[i] = self._predict(X_test[i])
        return res


class DecisionTreeRegressor(ClassificationAndRegressionTree):
    def count_info_fun(self, left, right, data):
        var_tot = np.var(data[:, -1])
        var_1 = np.var(left[:, -1])
        var_2 = np.var(right[:, -1])
        frac_1 = len(left) / len(data)
        frac_2 = len(right) / len(data)
        # 计算均方差
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return variance_reduction

    def calcu_majority_vote(self, data):
        value = np.mean(data[:, -1:], axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X_train, y_train):
        self.count_info = self.count_info_fun
        self.majority_vote = self.calcu_majority_vote
        super(DecisionTreeRegressor, self).fit(X_train, y_train)


class DecisionTreeClassifier(ClassificationAndRegressionTree):
    def count_info_fun(self, left, right, data):
        base_gain = gini(data[:, -1])
        prop = left.shape[0] / data.shape[0]
        info_gain = base_gain - prop * gini(left[:, -1]) - (1 - prop) * gini(right[:, -1])
        return info_gain

    def calcu_majority_vote(self, data):
        a = np.unique(data[:, -1])
        max_feature = np.empty(2)
        for i in a:
            b = np.sum(data[:, -1] == i)
            if b > max_feature[0]:
                max_feature[0] = b
                max_feature[1] = i
        return max_feature[1]

    def fit(self, X_train, y_train):
        self.count_info = self.count_info_fun
        self.majority_vote = self.calcu_majority_vote
        super(DecisionTreeClassifier, self).fit(X_train, y_train)

# def load_data(data, bin=None):
#     le = LabelEncoder()
#     dt = []
#     for i in bin:
#         data2 = data[:, i]
#         res = le.fit_transform(data2).reshape(-1, 1)
#         dt.append(res)
#     dt = np.hstack(dt)
#     return dt
#
#
# def cut_data(data, bin=None):
#     # 离散值连续化
#     le = LabelEncoder()
#     res = []
#     a = range(data.shape[1]) if bin is None else bin
#     for i in a:
#         data1 = np.unique(np.sort(data[:, i]))
#         divide = [data1[0]-1]
#         for j in range(data1.shape[0]-1):
#             divide.append((data1[j]+data1[j+1])/2)
#         divide.append(data1[-1]+1)
#         data2 = np.array(pd.cut(data[:, i], divide))
#         data3 = le.fit_transform(data2).reshape(-1, 1)
#         res.append(data3)
#     return np.hstack(res)


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from Utils.Assessiment import RegressionAssessment
    from sklearn.metrics import r2_score
    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # reg = ClassificationAndRegressionTree(tree_type='reg')
    reg = DecisionTreeRegressor()
    # reg = DecisionTreeClassifier()
    reg.fit(X_train, y_train)
    y_pr = reg.predict(X_test)
    # print(np.sum(y_test == y_pr) / len(y_pr))
    ass = RegressionAssessment(y_pr, y_test)
    ass.assessment()
    print(r2_score(y_test, y_pr))
