import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class RegressionAssessment:
    def __init__(self, y_predict, y_test):
        """
        回归评估
        :param y_predict: 预测值
        :param y_test: 真实值
        """
        self.y_predict = y_predict
        self.y_test = y_test
        self.MAE = None
        self.MSE = None
        self.R_square = None
        self.RMSE = None
        self.MAPE = None

    def get_MSE(self):
        if self.MSE is not None: return self.MSE
        self.MSE = np.sum((self.y_predict - self.y_test)**2) / len(self.y_predict)
        return self.MSE

    def get_MAE(self):
        if self.MAE is not None: return self.MAE
        self.MAE = np.sum(np.abs(self.y_predict - self.y_test)) / len(self.y_predict)
        return self.MAE

    def get_RMSE(self):
        if self.RMSE is not None: return self.RMSE
        if self.MSE is None: self.get_MSE()
        self.RMSE = np.sqrt(self.MSE)
        return self.RMSE

    def get_R2(self):
        # if self.R_square is not None: return self.R_square
        # if self.MSE is None: self.get_MSE()
        # self.R_square = 1 - (self.MSE / np.sum((self.y_test - np.mean(self.y_test))**2))
        self.R_square = r2_score(self.y_test, self.y_predict)
        return self.R_square

    def get_MAPE(self):
        if self.MAPE is not None: return self.MAPE
        self.MAPE = np.sum(np.abs((self.y_predict - self.y_test) / self.y_test)) / len(self.y_predict)
        return self.MAPE

    def assessment(self):
        print("均方误差（MSE）：", self.get_MSE())
        print("平均绝对误差（MAE）：", self.get_MAE())
        print("均方根误差（RMSE）：", self.get_RMSE())
        print("平均绝对百分比误差（MAPE）：", self.get_MAPE())
        print("R平方：", self.get_R2())


class ClassificationAssessment:
    def __init__(self,y_predict_rate, y_test):
        """
        分类评估
        :param y_predict_rate: 模型得出的概率
        :param y_test: 真实数据
        """
        self.TP, self.FP, self.TN, self.FN = 0,0,0,0
        self.y_predict = None
        self.y_test = y_test
        self.y_predict_rate = y_predict_rate
        self.predict()
        self.confuse_matrix()

    def predict(self, threshold=0.5):
        a = np.empty(shape=np.array(self.y_predict_rate).shape)
        for i in range(len(self.y_predict_rate)):
            if self.y_predict_rate[i] >= threshold: a[i] = 1
            else: a[i] = 0
        self.y_predict = a

    def confuse_matrix(self):
        TP, FP, TN, FN = 0,0,0,0
        for i in range(len(self.y_predict)):
            if self.y_predict[i] == self.y_test[i]:
                if self.y_predict[i] == 1: TP += 1
                else: TN += 1
            elif self.y_predict[i] == 1 and self.y_test[i] == 0:
                FP += 1
            elif self.y_predict[i] == 0 and self.y_test[i] == 1:
                FN += 1
        self.TP, self.FP, self.TN, self.FN = TP, FP, TN, FN

    def print_matrix(self):
        print("           预测结果    ")
        print("真实     正例      反例")
        print("正例     %3d      %3d" % (self.TP, self.FN))
        print("反例     %3d      %3d" % (self.FP, self.TN))

    def TPR(self):
        try:
           return self.TP / (self.TP + self.FN)
        except :
            return 0.

    def FPR(self):
        try:
            return self.FP / (self.FP + self.TN)
        except :
            return 0.

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN)

    def precision(self):
        try:
            return self.TP / (self.TP + self.FP)
        except:
            return 0.

    def recall(self):
        try:
            return self.TP / (self.TP + self.FN)
        except:
            return 0.

    def F1_score(self):
        p = self.precision()
        r = self.recall()
        try:
            return (2 * p * r) / (p + r)
        except:
            return 0.

    def paint_PR(self):
        thresholds = np.arange(np.min(self.y_predict_rate), np.max(self.y_predict_rate), 0.1)
        precisions = []
        recalls = []
        for i in thresholds:
            self.predict(i)
            self.confuse_matrix()
            precisions.append(self.precision())
            recalls.append(self.recall())
        plt.plot(recalls, precisions)
        plt.show()

    def paint_ROC(self):
        thresholds = np.arange(np.min(self.y_predict_rate), np.max(self.y_predict_rate), 0.1)
        fprs = []
        tprs = []
        for i in thresholds:
            self.predict(i)
            self.confuse_matrix()
            fprs.append(self.FPR())
            tprs.append(self.TPR())
        plt.plot(fprs, tprs)
        plt.show()

    def count_AUC(self):
        N = 0
        P = 0
        neg_prob = []
        pos_prob = []
        for i in range(len(self.y_test)):
            if (self.y_test[i] == 1):
                P += 1
                pos_prob.append(self.y_predict_rate[i])
            else:
                N += 1
                neg_prob.append(self.y_predict_rate[i])
            number = 0
            for pos in pos_prob:
                for neg in neg_prob:
                    if pos > neg:
                        number += 1
                    elif pos == neg:
                        number += 0.5
        print("AUC值为：", number/(N*P))
