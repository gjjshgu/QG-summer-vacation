import numpy as np
from CART import DecisionTreeRegressor
from Utils.LossFunction import SquareLoss, CrossEntropy
from Utils.DataPreprocessing import to_categorical


class GradientBoostingTree:
    def __init__(self, n_estimators, learning_rate, min_samples, min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.loss = SquareLoss()
        if not self.regression:
            self.loss = CrossEntropy()
        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(min_sample=self.min_samples,
                                         min_impurity=self.min_impurity,
                                         max_depth=self.max_depth)
            self.trees.append(tree)
            
    def fit(self, X, y):
        y_predict = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_predict)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            y_predict -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_predict = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_predict = -update if not y_predict.any() else y_predict - update
        
        if not self.regression:
            y_predict = np.exp(y_predict) / np.expand_dims(np.sum(np.exp(y_predict), axis=1), axis=1)
            y_predict = np.argmax(y_predict, axis=1)

        return y_predict


class GradientBoostingRegressor(GradientBoostingTree):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_sample=2, min_var=1e-7, max_depth=4):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples=min_sample,
                                                        min_impurity=min_var,
                                                        max_depth=max_depth,
                                                        regression=True)


class GradientBoostingClassifier(GradientBoostingTree):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_sample=2, min_info_gain=1e-7, max_depth=4):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples=min_sample,
                                                        min_impurity=min_info_gain,
                                                        max_depth=max_depth,
                                                        regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    iris = load_boston()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(y_predict)
