import pandas as pd

class Apriori:
    def __init__(self, min_support=0.5, min_conf=0.7):
        self.min_support = min_support
        self.min_conf = min_conf
        self.support_data = None
        self.frequent_element = None
        self.rule_list = []

    def _combine(self, sets, k):
        """
        求并集
        :param sets: 未合并的集合
        :param k: 集合前k个元素，用于对比
        :return: 并集的列表
        """
        ret_list = []
        n = len(sets)
        for i in range(n):
            for j in range(i + 1, n):
                l1 = list(sets[i])[:k]
                l2 = list(sets[j])[:k]
                l1.sort()
                l2.sort()
                if l1 == l2:
                    # 取并集
                    ret_list.append(sets[i] | sets[j])
        return ret_list

    def get_frequent_element(self, data):
        def unique_set(dataset):
            """
            找出单独的元素，并转为集合
            :param dataset: 数据集
            :return: 储存集合的列表
            """
            unique = []
            for transaction in dataset:
                for item in transaction:
                    if not [item] in unique:
                        unique.append([item])
            unique.sort()
            return list(map(frozenset, unique))

        def scan_frequent(dataset, sets):
            """
            找出频繁项，计算支持度
            :param dataset: 数据集
            :param sets: 集合的列表
            :return: 频繁项集合的列表和对应支持度的字典
            """
            # 统计出现次数
            count = {}
            for tid in dataset:
                for can in sets:
                    if can.issubset(tid):
                        if can not in count:
                            count[can] = 1
                        else:
                            count[can] += 1
            ret_list = []
            support_data = {}
            for key in count:
                support = count[key] / len(dataset)
                if support >= self.min_support:
                    ret_list.insert(0, key)
                support_data[key] = support
            return ret_list, support_data

        unique = unique_set(data)
        data = list(map(set, data))
        self.frequent_element, self.support_data = scan_frequent(data, unique)
        self.frequent_element = [self.frequent_element]
        k = 0
        while len(self.frequent_element[k]) > 0:
            combined_element = self._combine(self.frequent_element[k], k)
            ret_lists, supports = scan_frequent(data, combined_element)
            self.support_data.update(supports)
            self.frequent_element.append(ret_lists)
            k += 1
        return self.frequent_element, self.support_data

    def generate(self):
        def count_conf(sets, H):
            prunedH = []
            for conseq in H:
                conf = self.support_data[sets] / self.support_data[sets - conseq]
                if conf >= self.min_conf:
                    # print(tuple(sets-conseq), '--->', tuple(conseq), 'conf:', conf)
                    self.rule_list.append((tuple(sets - conseq), tuple(conseq), conf))
                    prunedH.append(conseq)
            return prunedH

        def divide_sets(sets, H):
            m = len(H[0])
            if len(sets) > (m + 1):
                hmp1 = self._combine(H, m-1)
                hmp1 = count_conf(sets, hmp1)
                if len(hmp1) > 1:
                    divide_sets(sets, hmp1)
        # 变量每个元素，忽略只有一个元素的集合
        for i in range(1, len(self.frequent_element)):
            # 继续遍历
            for sets in self.frequent_element[i]:
                # 获得每一个元素
                h = [frozenset([item]) for item in sets]
                if i > 1:
                    # 如果集合元素有3个或以上，则要做进一步的合并
                    divide_sets(sets, h)
                else:
                    # 计算置信度
                    count_conf(sets, h)
        return self.rule_list

def load_data():
    df = pd.read_csv('/Users/huangzhuoxi/Documents/学术部/学术部作业/寒假作业/资料/mushrooms.csv')
    col = df.columns.tolist()
    data = df.values
    for x in range(len(data)):
        for y in range(len(col)):
            data[x][y] = col[y] + '=' + data[x][y]
    return data

if __name__ == '__main__':
    # data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    data = load_data()
    ap = Apriori(min_support=0.46)
    L, sp = ap.get_frequent_element(data)
    # print(L)
    # print(sp)
    # print('*' * 30)
    res = ap.generate()
    # print(res)
    for i in res:
        for j in range(2):
            if 'class=p' in i[j]:
                print(i[0], "--->", i[1], "conf", i[2])

    print("success")
