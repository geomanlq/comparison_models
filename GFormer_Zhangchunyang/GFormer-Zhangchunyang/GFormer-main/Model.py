import torch
import torch as t
from torch import nn
from Params import args
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
    def __init__(self, gtLayer):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer)])
        self.gcnLayer = GCNLayer()
        self.gtLayers = gtLayer
        self.pnnLayers = nn.Sequential(*[PNNLayer() for i in range(args.pnn_layer)])

    def getEgoEmbeds(self):
        return t.cat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, handler, is_test, sub, cmp, encoderAdj, decoderAdj=None):
        embeds = t.cat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        emb, _ = self.gtLayers(cmp, embeds)
        cList = [embeds, args.gtw*emb]
        emb, _ = self.gtLayers(sub, embeds)
        subList = [embeds, args.gtw*emb]

        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoderAdj, embedsLst[-1])
            embeds2 = gcn(sub, embedsLst[-1])
            embeds3 = gcn(cmp, embedsLst[-1])
            subList.append(embeds2)
            embedsLst.append(embeds)
            cList.append(embeds3)
        if is_test is False:
            for i, pnn in enumerate(self.pnnLayers):
                embeds = pnn(handler, embedsLst[-1])
                embedsLst.append(embeds)
        if decoderAdj is not None:
            embeds, _ = self.gtLayers(decoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        cList = sum(cList)
        subList = sum(subList)

        return embeds[:args.user], embeds[args.user:], cList, subList


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class PNNLayer(nn.Module):
    def __init__(self):
        super(PNNLayer, self).__init__()
        self.linear_out_position = nn.Linear(args.latdim, 1)
        self.linear_out = nn.Linear(args.latdim, args.latdim)
        self.linear_hidden = nn.Linear(2 * args.latdim, args.latdim)
        self.act = nn.ReLU()

    def forward(self, handler, embeds):
        t.cuda.empty_cache()
        anchor_set_id = handler.anchorset_id
        dists_array = t.tensor(handler.dists_array, dtype=t.float32).to("cuda:0")
        set_ids_emb = embeds[anchor_set_id]
        set_ids_reshape = set_ids_emb.repeat(dists_array.shape[1], 1).reshape(-1, len(set_ids_emb),
                                                                              args.latdim)  # 69534.256.32
        dists_array_emb = dists_array.T.unsqueeze(2)  #
        messages = set_ids_reshape * dists_array_emb  # 69000*256*32

        self_feature = embeds.repeat(args.anchor_set_num, 1).reshape(-1, args.anchor_set_num, args.latdim)
        messages = torch.cat((messages, self_feature), dim=-1)
        messages = self.linear_hidden(messages).squeeze()

        outposition1 = t.mean(messages, dim=1)

        return outposition1


class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + 0.01*noise

    def forward(self, adj, embeds, flag=False):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], args.head]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = t.zeros([adj.shape[0], args.latdim]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds, att


class LocalGraph(nn.Module):

    def __init__(self, gtLayer):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer
        self.sft = t.nn.Softmax(0)
        self.device = "cuda:0"
        self.num_users = args.user
        self.num_items = args.item
        self.pnn = PNNLayer().cuda()

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        return dists_dict

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)
        num_workers = 1  # windows
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)

        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def precompute_dist_data(self, edge_index, num_nodes, approximate=0):
        '''
            Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
            :return:
            '''
        graph = nx.Graph()
        graph.add_edges_from(edge_index)

        n = num_nodes
        dists_dict = self.all_pairs_shortest_path_length_parallel(graph,
                                                                  cutoff=approximate if approximate > 0 else None)
        dists_array = np.zeros((n, n), dtype=np.int8)

        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array

    def forward(self, adj, embeds, handler):

        embeds = self.pnn(handler, embeds)
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        tmp_rows = np.random.choice(rows.cpu(), size=[int(len(rows) * args.addRate)])
        tmp_cols = np.random.choice(cols.cpu(), size=[int(len(cols) * args.addRate)])

        add_cols = t.tensor(tmp_cols).to(self.device)
        add_rows = t.tensor(tmp_rows).to(self.device)

        newRows = t.cat([add_rows, add_cols, t.arange(args.user + args.item).cuda(), rows])
        newCols = t.cat([add_cols, add_rows, t.arange(args.user + args.item).cuda(), cols])

        ratings_keep = np.array(t.ones_like(t.tensor(newRows.cpu())))
        adj_mat = sp.csr_matrix((ratings_keep, (newRows.cpu(), newCols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        embeds_l2, atten = self.gt_layer(add_adj, embeds)
        att_edge = t.sum(atten, dim=-1)

        return att_edge, add_adj


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.device = "cuda:0"
        self.sft = t.nn.Softmax(1)

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def create_sub_adj(self, adj, att_edge, flag):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        if flag:
            att_edge = (np.array(att_edge.detach().cpu() + 0.001))
        else:
            att_f = att_edge
            att_f[att_f > 3] = 3
            att_edge = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))  # 基于mlp可以去除
        att_f = att_edge / att_edge.sum()
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.sub),
                                      replace=False, p=att_f)

        keep_index.sort()

        drop_edges = []
        i = 0
        j = 0
        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(args.user + args.item).cuda(), rows])
        cols = t.cat([t.arange(args.user + args.item).cuda(), cols])

        ratings_keep = np.array(t.ones_like(t.tensor(rows.cpu())))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = att_edge
        att_f[att_f > 3] = 3
        att_f = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f1 = att_f / att_f.sum()

        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.keepRate),
                                          replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(args.user + args.item).cuda(), rows])
        cols = t.cat([t.arange(args.user + args.item).cuda(), cols])
        drop_edges = []
        i, j = 0, 0

        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        ratings_keep = np.array(t.ones_like(t.tensor(rows.cpu())))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)


        drop_row_ids = users_up[drop_edges]
        drop_col_ids = items_up[drop_edges]

        ext_rows = np.random.choice(rows.cpu(), size=[int(len(drop_row_ids) * args.ext)])
        ext_cols = np.random.choice(cols.cpu(), size=[int(len(drop_col_ids) * args.ext)])

        ext_cols = t.tensor(ext_cols).to(self.device)
        ext_rows = t.tensor(ext_rows).to(self.device)
        #
        tmp_rows = t.cat([ext_rows, drop_row_ids])
        tmp_cols = t.cat([ext_cols, drop_col_ids])

        new_rows = np.random.choice(tmp_rows.cpu(), size=[int(adj._values().shape[0] * args.reRate)])
        new_cols = np.random.choice(tmp_cols.cpu(), size=[int(adj._values().shape[0] * args.reRate)])

        new_rows = t.tensor(new_rows).to(self.device)
        new_cols = t.tensor(new_cols).to(self.device)

        newRows = t.cat([new_rows, new_cols, t.arange(args.user + args.item).cuda(), rows])
        newCols = t.cat([new_cols, new_rows, t.arange(args.user + args.item).cuda(), cols])

        hashVal = newRows * (args.user + args.item) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (args.user + args.item)
        newRows = ((hashVal - newCols) / (args.user + args.item)).long()

        decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
                                          adj.shape)

        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp



