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
            if sys.version_info > (3, 0):
                obj.append(pkl.load(f, encoding='latin1'))
            else:
                obj.append(pkl.load(f))
    
    x, tx, allx, graph = tuple(obj)
    index = []
    with open(f"data/ind.{dataset}.test.index") as ff:
        for line in ff:
            index.append(int(line.strip()))
    test_idx = np.sort(index)

    features = sp.vstack((allx, tx)).tolil()
    features[index, :] = features[test_idx, :]
    adj_mat = networkx.adjacency_matrix(networkx.from_dict_of_lists(graph))

    return adj_mat, features