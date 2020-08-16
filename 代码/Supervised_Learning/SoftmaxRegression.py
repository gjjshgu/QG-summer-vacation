import numpy as np


class SoftmaxRegressionClassifier:
    def __init__(self):
        self.weight = None

    def fit(self, X_train, y_train, n_iters=100, alpha=0.1):
        def softmax(X):
            return np.exp(X) / np.sum(np.exp(X))

        target_nums = len(set(y_train))
        self.weight = np.random.random((target_nums, X_train.shape[1]))
        for _ in range(n_iters):
            for i in range(len(X_train)):
                x = X_train[i].reshape(-1, 1)
                y = np.zeros((target_nums, 1))
                y[y_train[i]] = 1
                _y = softmax(np.dot(self.weight, x))
                a = (np.dot((_y - y), x.T))
                self.weight -= alpha * a

    def predict(self, X_test):
        return np.argmax(self.weight.dot(X_test.T), axis=0)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    X = load_iris().data
    y = load_iris().target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    sf = SoftmaxRegressionClassifier()
    sf.fit(X_train, y_train)
    print(classification_report(y_test, sf.predict(X_test)))
