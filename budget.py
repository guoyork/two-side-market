import numpy as np
import BA

m = 5
n = 10
N = 100000

np.random.seed(99999999)

edge_list = BA.BA_graph(5, m, n - 5)
match1 = []
match2 = []
'''
match1 = [1, 1, 1, 2, 3, 4, 8, 9]
match2 = [2, 3, 4, 3, 6, 7, 5, 2]
m = len(match1)
'''
#'''
for edge in edge_list:
    k = np.random.randint(0, 2)
    match1.append(edge[k])
    match2.append(edge[1 - k])
m = len(edge_list)
print("edge number:", m)
#'''
'''
match1 = np.random.randint(0, n, size=m)
match2 = np.random.randint(0, n, size=m)
'''
w1 = abs(np.random.normal(size=(m, 1)))
w2 = 2 * abs(np.random.normal(size=(m, 1)))
w0 = np.concatenate((w1, w2), axis=1)
budget = np.random.randint(0, int(np.sqrt(m)), size=n)


def feed(match):
    maps = {}
    for i in range(m):
        if not match[i] in maps:
            maps[match[i]] = []
        maps[match[i]].append(i)
    return maps


def cal_outcome(w, budget, match):
    res = 0.0
    used_budget = budget.copy()
    for i in range(m):
        used_budget, temp = settle(match[i], used_budget, w[i])
        res += temp
    return res


def settle(ad_id, budget, revenue):
    if budget[ad_id] > 0:
        budget[ad_id] -= 1
        return budget, revenue
    else:
        return budget, 0


def estimator1(w, budget, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                temp = int(np.random.random() > p)
                for k in groups[i][j]:
                    real_match[k] = temp
    res1 = 0.0
    res2 = 0.0
    used_budget = budget.copy()
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    for j in range(n):
        if sorted(map3.get(j, [])) == sorted(map1.get(j, [])):
            factor = 1.0
            group1 = set([match2[i] for i in map1.get(j, []) if match2[i] != j])
            factor *= p**len(group1)
            group2 = set([match1[i] for i in map2.get(j, []) if match1[i] != j])
            factor *= (1 - p)**len(group2)
            for i in sorted(map3.get(j, [])):
                used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                res1 += temp / factor
        if sorted(map3.get(j, [])) == sorted(map2.get(j, [])):
            factor = 1.0
            group2 = set([match1[i] for i in map2.get(j, []) if match1[i] != j])
            factor *= (1 - p)**len(group2)
            if j in map1:
                group1 = set([match2[i] for i in map1.get(j, []) if match2[i] != j])
                factor *= p**len(group1)
            for i in sorted(map3.get(j, [])):
                used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                res2 += temp / factor
    return res1 / m, res2 / m


def estimator2(w, budget, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = 1
                else:
                    for k in groups[i][j]:
                        real_match[k] = 0
    res1 = 0.0
    res2 = 0.0
    weight1 = 0.0
    weight2 = 0.0
    used_budget = budget.copy()
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                factor = 1.0
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                factor *= p**len(group1)
                if j in map2:
                    group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                    factor *= (1 - p)**len(group2)
                for i in sorted(map3[j]):
                    used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                    res1 += temp / factor
                    weight1 += 1.0 / factor
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                factor = 1.0
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                factor *= (1 - p)**len(group2)
                if j in map1:
                    group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                    factor *= p**len(group1)
                for i in sorted(map3[j]):
                    used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                    res2 += temp / factor
                    weight2 += 1.0 / factor

    if weight1:
        res1 /= weight1
    if weight2:
        res2 /= weight2
    return res1, res2, weight1 / m, weight2 / m


def estimator3(w, budget, match1, match2, p=0.5):
    real_match = np.array([1 if np.random.random() > p else 0 for i in range(m)], dtype=int)
    res1 = 0.0
    res2 = 0.0
    used_budget = budget.copy()
    for i in range(m):
        j = match2[i] if real_match[i] else match1[i]
        used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
        if real_match[i]:
            res2 += temp
        else:
            res1 += temp
    if m - np.sum(real_match):
        res1 /= m - np.sum(real_match)
    if np.sum(real_match):
        res2 /= np.sum(real_match)
    return res1, res2


def estimator4(w, budget, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m, dtype=int)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = 1
                else:
                    for k in groups[i][j]:
                        real_match[k] = 0
    res1 = 0.0
    res2 = 0.0
    used_budget = budget.copy()
    for i in range(m):
        j = match2[i] if real_match[i] else match1[i]
        temp = w[i][real_match[i]]
        used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
        if real_match[i]:
            res2 += temp
        else:
            res1 += temp
    if m - np.sum(real_match):
        res1 /= m - np.sum(real_match)
    if np.sum(real_match):
        res2 /= np.sum(real_match)
    return res1, res2


def estimator5(w, budget, match1, match2, p=0.5, threshold=4):
    res1 = np.zeros(m)
    count1 = np.zeros(m)
    res2 = np.zeros(m)
    count2 = np.zeros(m)
    groups = {}
    groups2 = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
        if not match2[i] in groups2:
            groups2[match2[i]] = {}
        if not match1[i] in groups2[match2[i]]:
            groups2[match2[i]][match1[i]] = []
        groups2[match2[i]][match1[i]].append(i)
    real_match = np.zeros(m, dtype=int) - 1
    for i in range(n):
        if i in groups:
            if len(groups[i]) < threshold:
                for j in range(n):
                    if j in groups[i]:
                        if np.random.random() > p:
                            for k in groups[i][j]:
                                if real_match[k] < 0:
                                    real_match[k] = 1
                        else:
                            for k in groups[i][j]:
                                if real_match[k] < 0:
                                    real_match[k] = 0
            else:
                if np.random.random() > p:
                    for j in groups[i]:
                        for k in groups[i][j]:
                            if real_match[k] < 0:
                                real_match[k] = 1
                else:
                    for j in groups[i]:
                        for k in groups[i][j]:
                            if real_match[k] < 0:
                                real_match[k] = 0
        if i in groups2:
            if len(groups2[i]) < threshold:
                for j in range(n):
                    if j in groups2[i]:
                        if np.random.random() > p:
                            for k in groups2[i][j]:
                                if real_match[k] < 0:
                                    real_match[k] = 1
                        else:
                            for k in groups2[i][j]:
                                if real_match[k] < 0:
                                    real_match[k] = 0
            else:
                if np.random.random() > p:
                    for j in groups2[i]:
                        for k in groups2[i][j]:
                            if real_match[k] < 0:
                                real_match[k] = 1
                else:
                    for j in groups2[i]:
                        for k in groups2[i][j]:
                            if real_match[k] < 0:
                                real_match[k] = 0
    used_budget = budget.copy()
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed([match2[i] if real_match[i] else match1[i] for i in range(m)])
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                if j in map2:
                    group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                for i in sorted(map3[j]):
                    used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                    res1[i] += temp
                    count1[i] += 1
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                if j in map1:
                    group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                for i in sorted(map3[j]):
                    used_budget, temp = settle(j, used_budget, w[i][real_match[i]])
                    res2[i] += temp
                    count2[i] += 1
    return res1 / m, res2 / m, count1, count2


if __name__ == "__main__":
    outcome1 = cal_outcome(w1, budget, match1)
    outcome2 = cal_outcome(w2, budget, match2)
    np.random.seed()
    '''
    test1 = [1, 1, 2, 1, 3]
    test2 = [1, 1, 1, 2, 2]
    test_w = abs(np.random.normal(size=(5, 5)))
    estimator2(test_w, test1, test2)
    '''
    print("real average outcome:", outcome1 / m, outcome2 / m)
    print("---------------------------------------------------------")
    #'''
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator1(w0, budget, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 1 expectation:", np.mean(res1), np.mean(res2))
    print("method 1 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1))
    print("---------------------------------------------------------")
    #'''
    '''
    res1 = []
    res2 = []
    weight1 = []
    weight2 = []
    for k in range(N):
        temp1, temp2, temp3, temp4 = estimator2(w0, budget, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
        weight1.append(temp3)
        weight2.append(temp4)
    print("method 2 expectation:", np.mean(res1), np.mean(res2),
          np.mean(weight1), np.mean(weight2))
    print("method 2 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1),
          np.std(weight1, ddof=1), np.std(weight2, ddof=1))
    print("---------------------------------------------------------")
    '''
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator3(w0, budget, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 3 expectation:", np.mean(res1), np.mean(res2))
    print("method 3 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))
    print("---------------------------------------------------------")
    #'''
    '''
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator4(w0, budget, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 4 expectation:", np.mean(res1), np.mean(res2))
    print("method 4 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))
    print("---------------------------------------------------------")
    '''
    res1 = []
    res2 = []
    count1 = np.zeros(m)
    count2 = np.zeros(m)
    for k in range(N):
        temp1, temp2, temp3, temp4 = estimator5(w0, budget, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
        count1 += temp3
        count2 += temp4
    count1 = np.maximum(1, count1) / N
    count2 = np.maximum(1, count2) / N
    print(count1)
    for k in range(N):
        res1[k] = np.sum(res1[k] / count1)
        res2[k] = np.sum(res2[k] / count2)
    print("method 5 expectation:", np.mean(res1), np.mean(res2))
    print("method 5 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))
    print("---------------------------------------------------------")