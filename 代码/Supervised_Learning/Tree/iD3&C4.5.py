import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, X, y, type='ID3'):
        def count_ent(dataset):
            """
            计算信息熵
            :param dataset: 数据集，类型为dataframe
            :return: 计算后的信息熵
            """
            labels = dataset[dataset.columns.tolist()[-1]]
            a = labels.value_counts()
            ent = 0.0
            for key in a.index:
                prob = a[key] / dataset.shape[0]
                ent -= prob * np.log2(prob)
            return ent

        def split_data(dataset, axis, value):
            """
            划分数据集
            :param dataset: 数据集，类型为DataFrame
            :param axis: 划分的特征
            :param value: 特征值
            :return: 划分后的数据
            """
            label_data = dataset[axis]
            # 更新数据集
            new_data = dataset.drop(axis, axis=1)
            index = label_data[label_data != value].index.tolist()
            new_data = new_data.drop(index)
            return new_data.reset_index(drop=True)

        def choose_feature(dataset):
            """
            选择信息增益最大的特征
            :param dataset: DataFrame数据集
            :return: 特征索引和信息增益
            """
            feature_nums = dataset.shape[1] - 1
            ent = count_ent(dataset)
            best_gain = 0.0
            best_feature = -1
            cols = dataset.columns.tolist()
            for i in range(feature_nums):
                new_ent = 0.0
                split_info = 0.0
                for value in set(dataset[cols[i]].tolist()):
                    data1 = split_data(dataset, cols[i], value)
                    prob = data1.shape[0] / dataset.shape[0]
                    new_ent += prob * count_ent(data1)
                    if type == 'C4.5':
                        split_info += -prob * np.log2(prob)
                # 计算信息增益或信息增益比
                gain = (ent - new_ent)/split_info if type == 'C4.5' else (ent - new_ent)
                print(cols[i], gain)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = cols[i]
            return best_feature, best_gain

        def create_tree(dataset):
            """
            建立决策树
            :param dataset: DataFrame数据集
            :return: 字典树
            """
            label = dataset.columns.tolist()[-1]
            class_list = dataset[label].tolist()

            # 若数据集中所有实例属于同一类Ck，则为单节点树，并将Ck作为该节点的类标记
            if class_list.count(class_list[0]) == len(class_list):
                return class_list[0]

            # 若特征集为空集，则为单节点树，并将数据集中实例数最大的类Ck作为该节点的类标记
            if dataset.shape[1] == 1:
                return dataset[label].value_counts().argmax()
            print('特征集和类别:', dataset.columns.tolist())
            best_feature, best_gain = choose_feature(dataset)
            tree = {best_feature: {}}

            # 得到列表包括节点所有的属性值
            print("选择特征：", best_feature)
            feat_values = dataset[best_feature]
            unique_values = set(feat_values)
            for value in unique_values:
                tree[best_feature][value] = create_tree(split_data(dataset, best_feature, value))
            return tree
        data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))))
        self.root = create_tree(data)
        return self.root

    def _predict(self, X_test):
        def find_label(tree, label):
            b = tree[X_test[label]]
            if type(b) != dict:
                return b
            c = list(b.keys())[0]
            return find_label(b[c], c)
        a = list(self.root.keys())[0]
        return find_label(self.root[a], a)

    def predict(self, X_test):
        X_test = pd.DataFrame(X_test)
        return [self._predict(X_test.loc[i, :]) for i in range(X_test.shape[0])]

# if __name__ == '__main__':
#     data = pd.read_table("/Users/huangzhuoxi/Documents/工作室考核/qg/中期考核/数据集/lense/lenses.txt", header=None, delim_whitespace=True)
#     data = data.drop(0, axis=1)
#     print(data)
#     id3 = DecisionTree()
#     # id3.fit(data, type='C4.5')
#     id3.fit(data.values[:, :-1], data.values[:, -1])
#     a = id3.predict(data)
#     # print()
#     # print(data[5].tolist())
#     pass

