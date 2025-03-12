import os
import torch
import numpy as np
from dotmap import DotMap
from torch.utils.data import Dataset

class COODataset(Dataset):
    def __init__(self, idxs, vals):
        self.idxs = idxs
        self.vals = vals

    def __len__(self):
        return self.vals.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.vals[idx]

def load_data(train_file, test_file, device, delimiter='::'):
    print(f"start loading data...")
    max_i_id = 0
    max_j_id = 0
    max_k_id = 0

    train_idxs_list = []
    train_vals_list = []
    with open(train_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split(delimiter)
            i_id = int(float(parts[0]))
            j_id = int(float(parts[1]))
            k_id = int(float(parts[2]))
            value = float(parts[3])
            max_i_id = i_id if i_id > max_i_id else max_i_id
            max_j_id = j_id if j_id > max_j_id else max_j_id
            max_k_id = k_id if k_id > max_k_id else max_k_id
            train_idxs_list.append([i_id, j_id, k_id])
            train_vals_list.append(value)
    train_idxs = torch.LongTensor(train_idxs_list).to(device)
    train_vals = torch.FloatTensor(train_vals_list).to(device)

    train_data = COODataset(train_idxs, train_vals)

    test_idxs_list = []
    test_vals_list = []
    with open(test_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split(delimiter)
            i_id = int(float(parts[0]))
            j_id = int(float(parts[1]))
            k_id = int(float(parts[2]))
            value = float(parts[3])
            max_i_id = i_id if i_id > max_i_id else max_i_id
            max_j_id = j_id if j_id > max_j_id else max_j_id
            max_k_id = k_id if k_id > max_k_id else max_k_id
            test_idxs_list.append([i_id, j_id, k_id])
            test_vals_list.append(value)
    test_idxs = torch.LongTensor(test_idxs_list).to(device)
    test_vals = torch.FloatTensor(test_vals_list).to(device)

    max_i_id += 1
    max_j_id += 1
    max_k_id += 1
    size = [max_i_id, max_j_id, max_k_id]
    print(f"tensor size: {size}")

    return train_data, test_idxs, test_vals, size
