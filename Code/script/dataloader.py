import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_adj(dataset_path):
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    num_nodes = adj.shape[0]

    return adj, num_nodes


def convert_adj_to_edge_index(adj):
    if not isinstance(adj, torch.Tensor):
        if hasattr(adj, 'toarray'):
            adj_dense = adj.toarray()
        else:
            adj_dense = adj
        adj_tensor = torch.tensor(adj_dense, dtype=torch.float)
    else:
        adj_tensor = adj

    edge_index = adj_tensor.nonzero().t().contiguous()

    return edge_index


def load_adj_with_weights(data_dir, dataset_name):
    dataset_path = str(os.path.join(data_dir, dataset_name))
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    num_nodes = adj.shape[0]

    coo = adj.tocoo()
    edge_index = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)

    return edge_index, edge_weight, num_nodes


def load_data(data_dir, dataset_name, len_train, len_val):
    dataset_path = str(os.path.join(data_dir, dataset_name))
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:len_train + len_val + len_val]
    #print(train.shape, val.shape, test.shape)
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail : tail + n_pred]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
