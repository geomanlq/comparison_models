import torch
from keras.utils.np_utils import to_categorical
from sklearn import metrics
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,f1_score
from scipy.stats import mode


def calculate_nmi(adj_rec, adj_label):
    # 将预测结果转换为二进制标签
    # labels_all = adj_label.long()
    preds_all = torch.argmax(adj_rec,dim=1)
    # 计算 NMI
    labels_all=np.array(adj_label)
    # print(labels_all,preds_all)
    nmi_score = normalized_mutual_info_score(labels_all, preds_all.cpu().numpy())
    return nmi_score


def calculate_purity(adj_rec, adj_label):
    # 将预测结果转换为社区索引
    preds_all = torch.argmax(adj_rec, dim=1)

    # 转换真实标签为 NumPy 数组
    labels_all = np.array(adj_label)

    # 唯一的聚类标签
    clusters = np.unique(preds_all.cpu().numpy())

    # 计算每个聚类中最常见的真实标签
    correct = 0
    for cluster in clusters:
        indices = np.where(preds_all.cpu().numpy() == cluster)
        most_common_label = mode(labels_all[indices])[0][0]
        correct += mode(labels_all[indices])[1][0]

    # 计算 purity
    purity = correct / float(preds_all.size(0))
    return purity


def calculate_f1_score(adj_rec, adj_label):
    # 将预测结果转换为社区索引
    preds_all = torch.argmax(adj_rec, dim=1)

    # 转换真实标签为 NumPy 数组
    labels_all = np.array(adj_label)

    # 计算 F1-score, 使用 micro 平均
    f1 = f1_score(labels_all, preds_all.cpu().numpy(), average='micro')
    return f1


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.long()
    preds_all = torch.gt(adj_rec, 0.5).long()
    return torch.eq(labels_all,preds_all).float().mean()


def modularity(adjacency_matrix:sp.coo_matrix, label_list): 
    S = to_categorical(label_list, num_classes=None)
    if isinstance(adjacency_matrix, torch.Tensor):
        A = adjacency_matrix.cpu().numpy()
    else:
        A = adjacency_matrix.toarray()
    m = np.sum(A) / 2
    k = np.sum(A, axis=1)
    B = A - np.outer(k, k) / (2 * m)
    Q = 1 / (2 * m) * np.trace(np.dot(np.dot(S.T, B), S))
    return Q 

def acc(y_pred, y_true):
    """
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)