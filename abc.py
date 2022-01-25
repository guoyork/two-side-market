import numpy as np

m = 50
n = 5
N = 100000

np.random.seed(10086)

match1 = np.random.randint(0, n, size=m)
match2 = np.random.randint(0, n, size=m)

w1 = abs(np.random.normal(size=(m, m)))
w2 = abs(np.random.normal(size=(m, m)))


def feed(match):
    maps = {}
    for i in range(m):
        if not match[i] in maps:
            maps[match[i]] = []
        maps[match[i]].append(i)
    return maps


def cal_outcome(w, match):
    maps = feed(match)
    res = 0.0
    for i in range(m):
        j = match[i]
        res += w[i][i]+np.mean(list(map(lambda x: w[i][x], maps[j])))**2
    return res


def estimator1(w1, w2, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = match1[k]
                else:
                    for k in groups[i][j]:
                        real_match[k] = match2[k]
    res1 = 0.0
    res2 = 0.0
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed(real_match)
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                factor = 1.0
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                factor *= p**len(group1)
                if j in map2:
                    group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                    factor *= (1-p)**len(group2)
                for i in map3[j]:
                    temp = w1[i][i] + np.mean(list(map(lambda x: w1[i][x], map3[j])))**2
                    res1 += temp/factor
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                factor = 1.0
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                factor *= (1-p)**len(group2)
                if j in map1:
                    group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                    factor *= p**len(group1)
                for i in map3[j]:
                    temp = w2[i][i] + np.mean(list(map(lambda x: w2[i][x], map3[j])))**2
                    res2 += temp/factor
    return res1/m, res2/m


def estimator2(w1, w2, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_match = np.zeros(m)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_match[k] = match1[k]
                else:
                    for k in groups[i][j]:
                        real_match[k] = match2[k]
    res1 = 0.0
    res2 = 0.0
    weight1 = 0.0
    weight2 = 0.0
    map1 = feed(match1)
    map2 = feed(match2)
    map3 = feed(real_match)
    for j in range(n):
        if j in map3 and j in map1:
            if sorted(map3[j]) == sorted(map1[j]):
                factor = 1.0
                group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                factor *= p**len(group1)
                if j in map2:
                    group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                    factor *= (1-p)**len(group2)
                for i in map3[j]:
                    temp = w1[i][i] + np.mean(list(map(lambda x: w1[i][x], map3[j])))**2
                    res1 += temp/factor
                    weight1 += 1.0/factor
        if j in map3 and j in map2:
            if sorted(map3[j]) == sorted(map2[j]):
                factor = 1.0
                group2 = set([match1[i] for i in map2[j] if match1[i] != j])
                factor *= (1-p)**len(group2)
                if j in map1:
                    group1 = set([match2[i] for i in map1[j] if match2[i] != j])
                    factor *= p**len(group1)
                for i in map3[j]:
                    temp = w2[i][i] + np.mean(list(map(lambda x: w2[i][x], map3[j])))**2
                    res2 += temp/factor
                    weight2 += 1.0/factor
    if weight1:
        res1 /= weight1
    if weight2:
        res2 /= weight2
    return res1, res2


def estimator3(w1, w2, match1, match2, p=0.5):
    real_assignment = np.array([1 if np.random.random() > p else 0 for i in range(m)])
    real_match = np.array([match1[i] if real_assignment[i] else match2[i] for i in range(m)])
    maps = feed(real_match)
    res1 = 0.0
    res2 = 0.0
    for i in range(m):
        j = real_match[i]
        if real_assignment[i]:
            res1 += w1[i][i]+np.mean(list(map(lambda x: w1[i][x], maps[j])))**2
        else:
            res2 += w2[i][i]+np.mean(list(map(lambda x: w2[i][x], maps[j])))**2
    if np.mean(real_assignment):
        res1 /= np.mean(real_assignment)
    if m-np.mean(real_assignment):
        res2 /= m-np.mean(real_assignment)
    return res1, res2


def estimator4(w1, w2, match1, match2, p=0.5):
    groups = {}
    for i in range(m):
        if not match1[i] in groups:
            groups[match1[i]] = {}
        if not match2[i] in groups[match1[i]]:
            groups[match1[i]][match2[i]] = []
        groups[match1[i]][match2[i]].append(i)
    real_assignment = np.zeros(m)
    real_match = np.zeros(m)
    for i in range(n):
        for j in range(n):
            if i in groups and j in groups[i]:
                if np.random.random() > p:
                    for k in groups[i][j]:
                        real_assignment[k] = 1
                        real_match[k] = match1[k]
                else:
                    for k in groups[i][j]:
                        real_assignment[k] = 0
                        real_match[k] = match2[k]
    maps = feed(real_match)
    res1 = 0.0
    res2 = 0.0
    for i in range(m):
        j = real_match[i]
        if real_assignment[i]:
            res1 += w1[i][i]+np.mean(list(map(lambda x: w1[i][x], maps[j])))**2
        else:
            res2 += w2[i][i]+np.mean(list(map(lambda x: w2[i][x], maps[j])))**2
    if np.mean(real_assignment):
        res1 /= np.mean(real_assignment)
    if m-np.mean(real_assignment):
        res2 /= m-np.mean(real_assignment)
    return res1, res2


outcome1 = cal_outcome(w1, match1)
outcome2 = cal_outcome(w2, match2)
np.random.seed()
'''
test1 = [1, 1, 2, 1, 3]
test2 = [1, 1, 1, 2, 2]
test_w = abs(np.random.normal(size=(5, 5)))
estimator2(test_w, test1, test2)
'''

if __name__ == "__main__":
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator1(w1, w2, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 1 expectation:", np.mean(res1), np.mean(res2))
    print("method 1 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1))
    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator2(w1, w2, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 2 expectation:", np.mean(res1),  np.mean(res2))
    print("method 2 variance:", np.std(res1, ddof=1), np.std(res2, ddof=1))

    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator3(w1, w2, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 3 expectation:", np.mean(res1),  np.mean(res2))
    print("method 3 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))

    res1 = []
    res2 = []
    for k in range(N):
        temp1, temp2 = estimator4(w1, w2, match1, match2)
        res1.append(temp1)
        res2.append(temp2)
    print("method 4 expectation:", np.mean(res1),  np.mean(res2))
    print("method 4 variance:", np.var(res1, ddof=1), np.var(res2, ddof=1))

    print(outcome1/m, outcome2/m)
