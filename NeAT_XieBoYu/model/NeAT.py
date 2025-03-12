
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.func import stack_module_state


class MLP(nn.Module):
    def __init__(self, dims, act='ReLU'):
        super(MLP, self).__init__()
        layers = []
        # First and hidden layers
        for i in range(0, len(dims)-2):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if ''!=act:
                layers.append(getattr(nn, 'ReLU')())

        # Last layers
        in_dim, out_dim = dims[-2], dims[-1]
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class NeAT(nn.Module):
    def __init__(self, cfg, sizes):
        super(NeAT, self).__init__()

        self.cfg = cfg
        self.nn = cfg.nn
        self.sizes = sizes
        self.rank = cfg.rank
        self.layer_dims = cfg.layer_dims

        # Embedding (factor matrices)
        self.embeds = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                     for i in range(len(self.sizes))])

        # MLPs for each rank-1 tensor  
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.dropout2 = nn.Dropout(p=cfg.dropout2)
        # Simple one
        # self.mlps = nn.ModuleList([MLP(self.layer_dims, act=cfg.act)
                                # for _ in range(self.rank)])
        self.make_mlps()
        self._initialize()

    def _initialize(self):
        for i in range(len(self.embeds)):
            nn.init.uniform_(self.embeds[i].weight.data)

    def make_mlps(self):
        '''
        Bath operation with mlp
        Speed up operation on neural networks to avoid loops in forward propagation
        '''
        mlps = nn.ModuleList([MLP(self.layer_dims, act=self.cfg.act).to(self.cfg.device)
                                    for _ in range(self.rank)])
        params, _ = stack_module_state(mlps)
        self.weight = nn.ParameterList([params[k] for k in params.keys() if k.endswith('weight')])
        self.bias = nn.ParameterList([params[k] for k in params.keys() if k.endswith('bias')])

    def _normalize(self):
        '''
        Normalize each rank-1 factors.
        '''
        for i in range(len(self.embeds)):
            self.embeds[i].weight.data = F.normalize(self.embeds[i].weight.data)
            
    def calc(self, x):
        '''
        Rank-wise matmul.
        '''
        for d in range(self.cfg.depth-1):
            x = x @ self.weight[d].permute(0, 2, 1) # transpose
            x = x + self.bias[d].unsqueeze(1)
            if d != self.cfg.depth-2: #Final layer does not need activation function
                x = torch.relu(x)
                x = self.dropout2(x)
        return x
    
    def forward(self, idxs):
        '''
        idxs: COO type indices (batch x nmode)
        '''

        # size of embeds[i] = rank x batch x 1
        # batch x mode should be the input to each MLP.
        embeds = [self.embeds[i](idxs[:, i]).permute(1, 0).unsqueeze(-1)
                for i in range(len(self.sizes))]
        
        # rank x batch x mode 
        x = torch.cat(embeds, dim=-1)
        x = F.normalize(x, dim=-1) # each concat vector is normalized

        # batch x 1
        # x = [self.mlps[r](x[r]) for r in range(self.rank)] # Easy but slow
        x = self.calc(x)
        x = self.dropout(x)
        x = x.sum(0).view(-1)
        
        return x
