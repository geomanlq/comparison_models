import pickle
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx


class DataHandler:
    def __init__(self):
        self.device = "cuda:"+str(args.gpu)
        if args.data == 'yelp':
            predir = '../Data/sparse_yelp/'
        elif args.data == 'ifashion':
            predir = '../Data/ifashion/'
        elif args.data == 'lastfm':
            predir = '../Data/lastfm/'
        elif args.data == 'spcora':
            predir = '../Data/spcora/'
        elif args.data == 'spbook':
            predir = '../Data/spbook/'
        elif args.data == 'spaminer':
            predir = '../Data/spaminer/'
        elif args.data == 'spama':
            predir = '../Data/spama/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'
        self.valfile = predir + 'valMat.pkl'

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff=None)
        return dists_dict

    def get_random_anchorset(self):
        n = self.num_nodes
        annchorset_id = np.random.choice(n, size=args.anchor_set_num, replace=False)
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(args.user + args.item))

        rows = self.allOneAdj._indices()[0, :]
        cols = self.allOneAdj._indices()[1, :]

        rows = np.array(rows.cpu())
        cols = np.array(cols.cpu())

        edge_pair = list(zip(rows, cols))
        graph.add_edges_from(edge_pair)
        dists_array = np.zeros((len(annchorset_id), self.num_nodes))

        dicts_dict = self.single_source_shortest_path_length_range(graph, annchorset_id, None)
        for i, node_i in enumerate(annchorset_id):
            shortest_dist = dicts_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[i, j] = 1 / (dist + 1)
        self.dists_array = dists_array
        self.anchorset_id = annchorset_id #

    def preSelect_anchor_set(self):
        self.num_nodes = args.user + args.item
        self.get_random_anchorset()

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = pickle.load(fs)#ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, trainMat):

        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, trainMat]), sp.hstack([trainMat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = t.ones_like(torchAdj._values())
        shape = torchAdj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        valMat = self.loadOneFile(self.valfile)

        args.user, args.item = trnMat.shape

        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
        valData = TstData(valMat, trnMat)
        self.valLoader = dataloader.DataLoader(valData, batch_size=args.tstBat, shuffle=False, num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.data = coomat.data  # 假设这是真实值

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # return self.rows[idx], self.cols[idx], self.negs[idx],self.data[idx]


        return self.rows[idx - 1], self.cols[idx - 1], self.negs[idx - 1], self.data[idx - 1]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs
        # 构建 truth 矩阵
        self.truth = np.zeros((coomat.shape[0], coomat.shape[1]), dtype=np.float32)
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            self.truth[row, col] = coomat.data[i]

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

    def gettruth_for_users(self, users):
        users = users.cpu().numpy()  # 确保 users 是一个 numpy array
        return t.tensor(self.truth[users], dtype=t.float32).cuda()
