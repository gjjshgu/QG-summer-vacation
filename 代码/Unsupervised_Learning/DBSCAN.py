import numpy as np
# from sklearn.datasets import load_iris

class DBSCAN:
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps  # 最小半径
        self.min_samples = min_samples  # 点的最小数量
        self.cluster = []  # 二维列表，每个元素为当前簇
        self.visited = []  # 记录已经搜索过的点的索引
        self.neighbours = {}  # 记录搜索点附近点的索引

    def fit(self, data):
        def get_neighbours(x):
            """
            获得附近min_samples个点的索引
            :param x: 单个点的坐标
            :return: 索引的数组
            """
            neighbours = []  # 存储索引
            index = np.arange(m)
            for i, sample in enumerate(data[index != x]):  # 不把自己算在内
                distance = np.linalg.norm(data[x]-sample)  # 计算欧氏距离
                if distance < self.eps:  # 判断是否小于最小半径
                    neighbours.append(i)
            return np.array(neighbours)

        def find_cluster(x, neighbours):
            """
            寻找簇，并将同类的簇合并
            :param x: 单个点的坐标
            :param neighbours: 点x附近的点的索引
            :return: 归为一个簇的索引列表
            """
            cluster = [x]  # 先加入自身
            for i in neighbours:  # 遍历
                if i not in self.visited:  # 判断是否搜索过
                    self.visited.append(i)  # 记录以及搜索
                    self.neighbours[i] = get_neighbours(i)  # 获得当前点附近的点
                    if len(self.neighbours[i]) >= self.min_samples:
                        cluster += find_cluster(i, self.neighbours[i])  # 不断递归，将相邻点归为一个列表，然后再合并
                    else:
                        cluster.append(i)  # 不要忘了把自己加上去
            return cluster
        """遍历每个点搜索"""
        m = len(data)
        for i in range(m):
            if i in self.visited: continue  # 判断是否搜索过
            self.neighbours[i] = get_neighbours(i)  # 获得当前点附近的点
            if len(self.neighbours[i]) >= self.min_samples:
                self.visited.append(i)
                self.cluster.append(find_cluster(i, self.neighbours[i]))  # 找簇
        """对每个簇分类"""
        labels = np.full(shape=m, fill_value=len(self.cluster))
        for cluster_i in range(len(self.cluster)):
            for sample_i in self.cluster[cluster_i]:
                labels[sample_i] = cluster_i
        return labels

# if __name__ == '__main__':
#     data = load_iris().data
#     db = DBSCAN(0.46, 9)
#     res = db.fit(data)
#     print(res)
