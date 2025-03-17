import os
import scipy.sparse as sp
import numpy as np
from torch._C import Value
from torch.utils.data import Dataset, DataLoader
import torch
import networkx as nx
import pickle
graph_pkl = ["primary1","Brain","high2012","primary2","workplace","high2011","enron","500","1000","5000","10000"]
label_num_dic = {"primary1": 13,"Brain":11,"primary2":13,"high2012":5,"high2011":4, "workplace":5,"enron":13,"500":10,"1000":10,"5000":10,"10000":10}
COMPLETE_GRAPH = False
def load_graphs(file_name,network_type):
    if file_name in graph_pkl:
        return load_graphs_pkl('Data/'+file_name,network_type)
    else:
        raise NameError

def load_graphs_pkl(file_name,network_type, complete_graph=COMPLETE_GRAPH):
    with open(file_name + '.pkl', 'rb') as handle:
        try:
            graph_snapshots = pickle.load(handle, encoding='bytes', fix_imports=True)
        except ValueError:
            handle.seek(0)
            graph_snapshots = pickle.load(handle, encoding='bytes', fix_imports=True, protocol=2)
    with open(file_name + '_label.pkl', 'rb') as handle:
        try:
            labels = pickle.load(handle, encoding='bytes', fix_imports=True)
        except ValueError:
            handle.seek(0)
            labels = pickle.load(handle, encoding='bytes', fix_imports=True, protocol=2)
    
    print("Lengths of snapshots:", len(graph_snapshots))
    print("Types of labels:", label_num_dic[file_name.split('/')[-1]])
    if file_name == "DBLP":
        graph_snapshots = graph_snapshots[:8] # take first 8 snapshots in DBLP for GPU memory limit
    if complete_graph:
        graph_snapshots = get_complete_graphs(graph_snapshots)
    return NetworkSnapshots(graph_snapshots,labels,network_type,file_name), label_num_dic[file_name.split('/')[-1]]

def NetworkSnapshots(graph_snapshots,labels,network_type,file_name):
    # prepare networkx Graph into Torch Tensor adj A and features X
    snapshots = []
    if file_name != "Data/DBLPdyn":
        label_set = set(labels.values())
        label_map = {l: i for i, l in enumerate(label_set)}
    else:
        label_map = {str(i): i-1 for i in range(1,15)}
        assert len(label_map) == 14

    for i, g in enumerate(graph_snapshots):
        adj = sp.coo_matrix(nx.adjacency_matrix(g))
        features = sp.coo_matrix(np.eye(adj.shape[0]), dtype=np.int64)  # use one hot as features
        features = sparse_to_tuple(features.tocoo())
        features = torch.cuda.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).cuda()
        if file_name == "Data/DBLPdyn":
            label_snap = [label_map[labels[i][n]] for idx, n in enumerate(g.nodes())]
        else:
            label_snap = [label_map[labels[n]] for idx, n in enumerate(g.nodes())]
        if network_type in ["DAEGC","SDCN"]:
            adj = torch.from_numpy(adj.toarray()).float().cuda()
        snapshots.append([adj,features,label_snap])
    return snapshots

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def graph_normalization(adj):
    if isinstance(adj,sp.coo_matrix):
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return sparse_to_tuple(adj_normalized)
    elif isinstance(adj, torch.Tensor):
        device = torch.device("cuda" if adj.is_cuda else "cpu")
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        mx = torch.mm(mx, r_mat_inv)
        return mx
    else:
        raise TypeError
    
def get_complete_graphs(dynamic_graph):
    # Get all the unique node ids from the dynamic graph
    all_node_ids = set()
    for graph in dynamic_graph:
        all_node_ids.update(graph.nodes)

    complete_graphs = []

    # Create a complete graph for each graph in the dynamic graph
    for graph in dynamic_graph:
        complete_graph = nx.Graph()

        # Add all the nodes from the dynamic graph
        complete_graph.add_nodes_from(all_node_ids)

        # Add the edges from the original graph
        complete_graph.add_edges_from(graph.edges)

        complete_graphs.append(complete_graph)

    return complete_graphs
