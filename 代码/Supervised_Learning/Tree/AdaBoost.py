import numpy as np


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []
        
    def fit(self, X, y):
        n_sample, n_feature = np.shape(X)
        w = np.full(n_sample, (1 / n_sample))
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = np.inf
            for feature_i in range(n_feature):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            clf.alpha = 0.5 * np.log((1.0 - min_error / (min_error + 1e-10)))
            predictions = np.ones(np.shape(y))
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_predict = np.zeros((n_samples, 1))
        for clf in self.clfs:
            predictions = np.ones(np.shape(y_predict))
            negative_index = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_index] = -1
            y_predict += clf.alpha * predictions
        y_predict = np.sign(y_predict).flatten()
        return y_predict

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data.data
    y = data.target
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = AdaBoost(n_clf=3)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(1 - np.sum(y_predict == y_test) / len(y_predict))

