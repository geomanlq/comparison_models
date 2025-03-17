import sys,os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Code.train import base_train, retrain_with_topo
from Code.dataloader import load_graphs
from Models.GraphFiltrationLayer import WrcfLayer,build_community_graph
import torch
import os
from Models import *
from Code.train import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import scipy.sparse as sp
import pickle

class Args(dict):
    def __init__(self,n_cluster,file_name,network_type) -> None:
        self.encoded_space_dim = 50
        self.n_cluster = n_cluster  # clusters
        self.num_epoch = 701 #1000
        self.learning_rate = 0.001 # for topo
        self.LAMBDA = 1
        self.card = 20 # num of ph considered
        self.file_name = file_name
        self.network_type = network_type
        self.start_mf = 500

class InitModel():
    def __init__(self, device):
        self.device = device
    def __call__(self, network_type, adj, feature_dim, args):
        if network_type == "DAEGC":
            return DAEGC(num_features=feature_dim, 
                        hidden_size=256,
                        embedding_size=args.encoded_space_dim, 
                        alpha=0.2, 
                        num_clusters=args.n_cluster).to(self.device)
        elif network_type == "GEC":
            adj_norm = graph_normalization(adj) # norm
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2])).to(self.device)
            return GAE(adj_norm, feature_dim, args).to(self.device)
        elif network_type == "MFC":
            adj_norm = graph_normalization(adj) # norm
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2])).to(self.device)
            return GAEMF(adj_norm, feature_dim, args).to(self.device)
        elif network_type == "SDCN":
            return SDCN(500, 500, 2000, 2000, 500, 500,
                        n_input=feature_dim,
                        n_z=args.encoded_space_dim,
                        n_clusters=args.n_cluster,
                        v=1.0).to(self.device)
        else:
            raise ValueError(f'Unknown network type: {network_type}')

def main(file_name, network_type):
    model_init = InitModel(device = "cuda")
    snapshot_list, n_cluster = load_graphs(file_name=file_name, network_type=network_type)
    print(len(snapshot_list))
    args = Args(n_cluster, file_name, network_type) # fix 20 cluster or assume known n_cluster
    model_list = []
    dgm_list = []
    wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
    wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)

    results_raw = [] 
    results_topo = []
    # base deep clustering training
    for idx, (adj,features,labels) in enumerate(snapshot_list):
        model = model_init(network_type, adj, features.size(1), args)
        model_list.append(model)
        base_train(network_type,
                   model,
                   features,
                   adj,
                   args,
                   str(idx))
        with torch.no_grad():
            if network_type == "SDCN":
                _, Q, _, Z = model(features,adj)
            else:
                _, Z, Q = model(features,adj)
            results_raw.append([
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj,labels
            ])
            # record dgm at each time step
            community_graph = build_community_graph(Q,adj)
            dgm0 = wrcf_layer_dim0(community_graph)
            dgm1 = wrcf_layer_dim1(community_graph)
            dgm_list.append([dgm0,dgm1])
    nmisum=0
    puritysum = 0
    F1sum=0
    # topological regulaized training
    for t in range(len(snapshot_list)):
        m = model_list[t]
        adj,features,labels = snapshot_list[t]
        if t == 0:
            gt_dgm = [None, dgm_list[t+1]]
        elif t == len(snapshot_list)-1: 
            gt_dgm = [dgm_list[t-1], None]
        else:
            gt_dgm = [dgm_list[t-1],dgm_list[t+1]]
        retrain_with_topo(
            network_type,
            m,
            gt_dgm,
            adj,
            features,
            args,
            str(t)
        )
        with torch.no_grad():
            if network_type == "SDCN":
                _, Q, _, Z = m(features,adj)
            else:
                _, Z, Q = m(features,adj)
            results_topo.append([
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj,labels
            ])
            print(Q.shape)
            # print(len(labels))
            # nmi=calculate_nmi(Q,labels)
            # nmisum = nmisum+nmi
            # purity = calculate_purity(Q,labels)
            # puritysum = puritysum+ purity
            # F1 = calculate_f1_score(Q,labels)
            # F1sum = F1sum+F1
            # print("NMI:", nmi)
            # print("Purity:", purity)
            # print("F1-score:", F1)
            # print(Q)
            # update dgm at time 
            community_graph = build_community_graph(Q,adj)
            dgm0_new = wrcf_layer_dim0(community_graph)
            dgm1_new = wrcf_layer_dim1(community_graph)
            dgm_list[t] = [dgm0_new,dgm1_new]
    # print("平均nmi=",nmisum/len(snapshot_list))
    # print("平均purity=",puritysum/len(snapshot_list))
    # print("平均F1-score=",F1sum/len(snapshot_list))
    endtime=time.time()
    print("用时:", endtime - starttime)

    # with open("Output/"+file_name+'/results_raw.pkl', 'wb') as handle:
    #     pickle.dump(results_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("Output/"+file_name+'/results_topo.pkl', 'wb') as handle:
    #     pickle.dump(results_topo, handle, protocol=pickle.HIGHEST_PROTOCOL)
import os

# 打印当前工作目录
print("当前工作目录:", os.getcwd())
if __name__ == "__main__":
    starttime=time.time()
    torch.manual_seed(42)
    network = "MFC" # GEC/DAEGC/MFC/SDCN
    graph_pkl = ["high2011"]
    for g in graph_pkl:
        print(g)
        main(g, network_type=network)
    endtime=time.time()
    print("用时:",endtime-starttime)

        



    
