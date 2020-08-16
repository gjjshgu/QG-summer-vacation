import numpy as np
import math


class GaussianMixtureModel:
    def __init__(self, k, max_iterations=2000, tolerance=1e-8):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.parameters = []
        self.responsibilities = []
        self.sample_assignments = None
        self.responsibility = None
        self.priors = None

    def predict(self, X_train):
        def calculate_covariance_matrix(X_train):
            n_samples = np.shape(X_train)[0]
            convariance_matrix = (1 / (n_samples-1)) * \
                                 (X_train - X_train.mean(axis=0)).T.dot(X_train - X_train.mean(axis=0))
            return np.array(convariance_matrix, dtype=float)

        def init_random_gaussians():
            n_samples = X_train.shape[0]
            self.priors = (1 / self.k) * np.ones(self.k)
            for i in range(self.k):
                parms = {}
                parms["mean"] = X_train[np.random.choice(range(n_samples))]
                parms["cov"] = calculate_covariance_matrix(X_train)
                self.parameters.append(parms)

        def multivariate_gaussian(parms):
            d = np.shape(X_train)[1]
            mean = parms["mean"]
            covar = parms["cov"]
            determinant = np.linalg.det(covar)
            likelihoods = np.zeros((np.shape(X_train)[0]))
            for i, sample in enumerate(X_train):
                coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2) * math.sqrt(determinant)))
                exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
                likelihoods[i] = coeff * exponent
            return likelihoods

        def get_likelihoods():
            n_samples = np.shape(X_train)[0]
            likelihoods = np.zeros((n_samples, self.k))
            for i in range(self.k):
                likelihoods[:, i] = multivariate_gaussian(self.parameters[i])
            return likelihoods

        def expectation():
            weighted_likelihood = get_likelihoods() * self.priors
            sum_likelihoods = np.expand_dims(np.sum(weighted_likelihood, axis=1), axis=1)
            self.responsibility = weighted_likelihood / sum_likelihoods
            self.sample_assignments = self.responsibility.argmax(axis=1)
            self.responsibilities.append(np.max(self.responsibility, axis=1))

        def maximization():
            for i in range(self.k):
                resp = np.expand_dims(self.responsibility[:, i], axis=1)
                mean = (resp * X_train).sum(axis=0) / resp.sum()
                covariance = (X_train - mean).T.dot((X_train - mean) * resp) / resp.sum()
                self.parameters[i]["mean"], self.parameters[i]["cov"] = mean, covariance

            self.priors = self.responsibility.sum(axis=0) / np.shape(X_train)[0]

        def covergence():
            if len(self.responsibilities) < 2:
                return False
            diff = np.linalg.norm(self.responsibilities[-1] - self.responsibilities[-2])
            return diff <= self.tolerance

        if __name__ == '__main__':
            init_random_gaussians()
            for _ in range(self.max_iterations):
                expectation()
                maximization()
                if covergence():
                    break
            expectation()
            return self.sample_assignments

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    X, y = make_blobs()
    cxk = GaussianMixtureModel(k=3)
    y_predict = cxk.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_predict)
    plt.show()
