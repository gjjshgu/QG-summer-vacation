import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

class LinearRegression_LS:
    def __init__(self):
        self._theta = None

    def fit(self, X_train, y_train):
        """
        :param X_train: 训练特征集
        :param y_train: 训练结果集
        :return:
        """
        assert X_train.shape[0]==y_train.shape[0]
        X=np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)
        return self

    def predict(self,X_predict):
        assert self._theta is not None
        X=np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X.dot(self._theta)

class LinearRegression_GD:
    def __init__(self, eta=0.01, n_iters=1e4, epsilon=1e-7):
        self.theta = None
        self.m = None
        self.eta = eta
        self.n_iters = n_iters
        self.epsilon = epsilon

    def fit(self,X_train, y_train):
        """
        :param X_train: 训练特征集
        :param y_train: 训练结果集
        :return:
        """

        assert X_train.shape[0] == y_train.shape[0]
        X_train = np.hstack([np.ones((len(X_train),1)),X_train])
        theta = np.zeros(X_train.shape[1]).reshape(-1,1)
        self.m = len(y_train)

        def J(theta,X,y): #theta应该是列向量
            return np.sum((y-X.dot(theta))**2) / self.m

        def dJ(theta,X,y):
            res = np.empty(len(theta)).reshape(-1,1)
            hx = X.dot(theta)-y
            res[0] = np.sum(hx)
            for i in range(1,len(theta)):
                a = X[:, i]
                res[i] = a.dot(hx)
            return res*2/self.m

        for _ in range(int(self.n_iters)):
            gradient = dJ(theta,X_train,y_train)
            last_theta = theta
            theta = theta-self.eta*gradient
            if (abs(J(theta,X_train,y_train)-J(last_theta,X_train,y_train))<self.epsilon):
                break
        self.theta = theta
        return self

    def predict(self,X_predict):
        assert self.theta is not None
        X = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X.dot(self.theta)

# if __name__ == '__main__':
#
#     boston=load_boston()
#     X=boston.data
#     y=boston.target
#     X=X[y<50.0]
#     y=y[y<50.0]
#     y=y.reshape(-1,1)
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#     data_x=StandardScaler()
#     X_train=data_x.fit_transform(X_train)
#     X_test=data_x.fit_transform(X_test)
#     data_y=StandardScaler()
#     y_train=data_y.fit_transform(y_train.reshape(-1,1))
#     y_test=data_y.fit_transform(y_test.reshape(-1,1))
#
#     gd=gradient_descent()
#     gd.fit(X_train,y_train)
#     y_predict=gd.predict(X_test)
#
#     gd_R_squared=r2_score(y_test,y_predict)
#     mse=mean_squared_error(data_y.inverse_transform(y_test),data_y.inverse_transform(y_predict))
#     mae=mean_absolute_error(data_y.inverse_transform(y_test),data_y.inverse_transform(y_predict))
#     print('R方值',gd_R_squared)
#     print('均方误差',mse)
#     print('平均绝对误差',mae)