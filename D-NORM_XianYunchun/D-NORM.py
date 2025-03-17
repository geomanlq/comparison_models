import torch.nn as nn
import torch.nn.functional as F
import argparse

from utils import *

data_name = "BTC"
model_name = "D-NORM"
eta = 0.005
lamda = 0.00001
layer = [15]  # R=5
cnn_nc = [5, 5, 5, 5]  # cn=5
stride = [1, 1, 1]
batch = 256
lear = 'adam'

error_gap = 1E-5
threshold = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run DNORM.")
    parser.add_argument('--path', nargs='?', default='D:/Dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=f'{data_name}/{data_name}',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=batch,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default=layer,
                        help="Size of each layer.")
    parser.add_argument('--net_channel', nargs='?', default=cnn_nc,
                        help='net_channel, should be 5 layers here')
    parser.add_argument('--learning_rate', type=float, default=eta,
                        help='Learning rate.')
    parser.add_argument('--regular_c', type=float, default=lamda,
                        help='Regularisation coefficient.')
    parser.add_argument('--learner', nargs='?', default=lear,
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    return parser.parse_args()


class DNORM(nn.Module):
    def __init__(self, num_users, num_items, num_times, layers, cnn_nc):
        super(DNORM, self).__init__()
        self.embedding_dim = int(layers[0] / 3)
        embedding_dim = self.embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.time_embedding = nn.Embedding(num_times, embedding_dim)

        nn.init.uniform_(self.user_embedding.weight, a=1e-7, b=0.4)
        nn.init.uniform_(self.item_embedding.weight, a=1e-7, b=0.4)
        nn.init.uniform_(self.time_embedding.weight, a=1e-7, b=0.4)

        iszs = [1] + cnn_nc[:-1]
        oszs = cnn_nc
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()

        for isz, osz in zip(iszs, oszs):
            weight = nn.Parameter(torch.randn(osz, isz, 2, 2, 2) * 0.1)
            bias = nn.Parameter(torch.zeros(osz))
            self.conv_weights.append(weight)
            self.conv_biases.append(bias)

        fc_input_dim = cnn_nc[-1]
        self.fc_weight = nn.Parameter(torch.randn(fc_input_dim, 1) * 0.1)
        self.fc_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_input, item_input, time_input, return_y1_only=False):
        embedding_dim = self.embedding_dim
        user_emb = self.user_embedding(user_input.long()).view(user_input.size(0), -1)
        item_emb = self.item_embedding(item_input.long()).view(item_input.size(0), -1)
        time_emb = self.time_embedding(time_input.long()).view(time_input.size(0), -1)

        user_emb = user_emb.unsqueeze(2)
        item_emb = item_emb.unsqueeze(2)

        user_item_outer = torch.bmm(user_emb, item_emb.transpose(1, 2))

        user_item_outer = user_item_outer.unsqueeze(-1)

        time_emb_expanded = time_emb.unsqueeze(1).unsqueeze(2)

        user_item_time_outer = user_item_outer * time_emb_expanded

        cnn_input = user_item_time_outer.unsqueeze(1)

        for weight, bias in zip(self.conv_weights, self.conv_biases):
            cnn_input = F.conv3d(cnn_input, weight, bias=bias, stride=stride, padding=0)
            cnn_input = F.relu(cnn_input)

        flatten_output = cnn_input.view(cnn_input.size(0), -1)

        prediction = torch.matmul(flatten_output, self.fc_weight) + self.fc_bias
        prediction = torch.sigmoid(prediction)

        return prediction

