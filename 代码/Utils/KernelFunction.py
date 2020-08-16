import numpy as np


def linear_kernel(**kwargs):  # 线性核函数
    def f(x1, x2):
        return np.inner(x1, x2)
    return f


def rbf_kernel(gamma, **kwargs):  # 高斯核函数
    def f(x1, x2):
        dist = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * dist)
    return f


def polynomial_kernel(power, coef, **kwargs):  # 多项式核函数
    def f(x1, x2):
        return (np.inner(x1, x2) + coef) ** power
    return f