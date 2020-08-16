import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
class LogisticRegressionClassifier:
    def __init__(self, eta=0.01, n_iters=1e4, epslion=1e-7):
        self._theta = None
        self.m = None
        self.n = None
        self.eta = eta
        self.n_iters = n_iters
        self.epslion = epslion

    def fit(self,X_train, y_train):
        """
        :param X_train: 训练特征集
        :param y_train: 训练结果集
        :param eta: 学习率
        :param n_iters: 最大循环次数
        :param epslion: 误差
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0]
        self.m=X_train.shape[0]
        self.n=X_train.shape[1]
        theta = np.zeros(X_train.shape[1]).reshape(-1, 1)
        y_train=y_train.reshape(-1,1)

        def h(X,theta):
            res=X.dot(theta)
            return 1 / (1+np.exp(-res))

        def J(X,y,theta):
            res=y*np.log10(h(X,theta))+(1-y)*np.log10(1-h(X,theta))
            return -(np.sum(res) / self.m)

        def dJ(X,theta):
            beta=h(X,theta)-(y_train)
            res=(X.T).dot(beta)
            return res / self.m

        for _ in range(int(self.n_iters)):
            gredient=dJ(X_train,theta)
            last_theta=theta
            theta=theta-self.eta*gredient
            if (abs(J(X_train,y_train,theta)-J(X_train,y_train,last_theta))<self.epslion):
                break

        self._theta=theta
        return self

    def predict(self,X_test):
        assert self._theta is not None
        res=X_test.dot(self._theta)
        res=1 / (1+np.exp(-res))
        for i in range(len(X_test)):
            if res[i]>0.5: res[i]=1
            else: res[i]=0
        return res

    def accuracy(self,y_predict,y_test):
        return np.sum(y_predict==y_test) / len(y_test)

# if __name__ == '__main__':
#     clf=LogisticRegression()
#
#     iris=load_iris()
#     X=iris.data
#     y=iris.target
#     X=X[y<2]
#     y=y[y<2]
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#
#     clf.fit(X_train,y_train)
#     y_predict=clf.predict(X_test).reshape((-1,))
#     print("准确率：",clf.accuracy(y_predict,y_test))