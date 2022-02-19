import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def BA_graph(node_org, edge_num, node_time):
    pair = np.random.randint(0, node_org, size=(node_org, 2)).tolist()
    for i in range(node_time):
        temp = sum(pair, [])
        prob = [temp.count(j) for j in range(i + node_org)]
        prob /= np.sum(prob)
        node_id = np.random.choice(i + node_org,
                                   size=edge_num,
                                   replace=True,
                                   p=prob)
        pair += [[j, i + node_org] for j in node_id]
    return pair


'''
edge_list = BA_graph(10, 5, 100)
temp = sum(edge_list, [])
k = [temp.count(j) for j in range(110)]
print([k.count(j) for j in range(110)])

G = nx.MultiGraph()
for edge in edge_list:
    G.add_edge(edge[0], edge[1])

nx.draw(G)
plt.show()
'''
