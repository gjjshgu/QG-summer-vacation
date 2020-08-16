import numpy as np


class SquareLoss:
    def loss(self, y_true, y_predict):
        return 0.5 * np.power((y_true - y_predict), 2)

    def gradient(self, y_true, y_predict):
        return -(y_true - y_predict)


class CrossEntropy:
    def loss(self, y_true, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y_true * np.log(p) - (1 - y_true) * np.log(1 - p)

    def gradient(self, y_true, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y_true / p) + (1 - y_true) / (1 - p)