import numpy as np
import sys
import os
import pickle as pkl
import networkx
import scipy.sparse as sp

def load_data(dataset):
    files = ['x','tx', 'allx', 'graph']
    obj = []
    for i in files:
        with open(f"data/ind.{dataset}.{i}", 'rb') as f:
            obj.append(pkl.load(f))
    
    x, tx, allx, graph = tuple(obj)
    index = []
    for line in open(f"data/ind.{dataset}.test.index"):
        index.append(int(line.strip()))
    test_idx = np.sort(index)

    features = sp.vstack((allx, x)).tolil()
    features[test_idx, :] = features[index, :]
    adj_mat = nx.adj_mat(nx.from_dict_of_lists(graph))

    return adj, features