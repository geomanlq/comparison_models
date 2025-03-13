import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import logging
import numpy as np

from utils import logger_info
from utils import compute_rmse, compute_mape

from utils_lrtc import unfolding, folding, shrinkage, svd_

save_dir = ""


def svt_scad(mat, tau, gamma=20, lamb=10):
    u, s, v = svd_(mat)
    ss = shrinkage(s, [tau, gamma, lamb], mode="scad")
    idx = torch.where(ss > 0)[0]
    return u[:, idx] @ torch.diag(ss[idx]) @ v[idx, :]


def read_and_build_tensor(file_paths):
    """
    读取多个数据文件并构建张量。

    :param file_paths: 数据文件路径列表
    :return: 构建好的张量
    """
    # 初始化一个空列表来存储所有文件的数据
    all_data = []

    # 遍历文件路径列表，读取每个文件的数据
    for file_path in file_paths:
        data = np.loadtxt(file_path)
        all_data.append(data)

    # 将所有数据拼贴在一起
    data = np.concatenate(all_data, axis=0)

    # 提取所有索引并找到最大值以确定张量的维度
    indices = data[:, :3].astype(int)
    max_dim1, max_dim2, max_dim3 = np.max(indices, axis=0)

    # 构建张量
    tensor = np.zeros((max_dim1 + 1, max_dim2 + 1, max_dim3 + 1))

    # 提取值
    values = data[:, 3]

    # 索引赋值
    tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = values

    return tensor


# 定义两个输入文件的路径
file1 = 'E:/resourch1/a_b_diverse/交通数据集/数据集/3-杭州地铁/train_out.txt'
file2 = 'E:/resourch1/a_b_diverse/交通数据集/数据集/3-杭州地铁/test_out.txt'


# 读取训练集、测试集和验证集
file_paths = [file1, file2]
train_tensor = read_and_build_tensor(file_paths)
file_paths = [file1]
test_tensor = read_and_build_tensor(file_paths)
print(train_tensor.shape)
print(test_tensor.shape)


def read_test_positions(file_path):
    # 读取数据文件
    data = np.loadtxt(file_path)

    # 提取位置索引（前三列）
    indices = data[:, :3].astype(int)

    # 分别提取三个维度的位置索引
    dim1_indices = torch.tensor(indices[:, 0])
    dim2_indices = torch.tensor(indices[:, 1])
    dim3_indices = torch.tensor(indices[:, 2])
    pos = (dim1_indices, dim2_indices, dim3_indices)
    return pos


pos_test = read_test_positions(file1)


def recover_data(sparse_tensor, dense_tensor, pos_test, rho, gamma, lamb, logger=None, factor=1.05, tol=1e-4,
                 max_iter=200, checkpoint=1000,errorgap=1e-5):
    #确保sparse_tensor是PyTorch张量
    if isinstance(sparse_tensor, np.ndarray):
        sparse_tensor = torch.from_numpy(sparse_tensor).float()
    elif not isinstance(sparse_tensor, torch.Tensor):
        raise TypeError("sparse_tensor must be a numpy.ndarray or torch.Tensor")

    # 确保 dense_tensor 是 PyTorch 张量
    if isinstance(dense_tensor, np.ndarray):
        dense_tensor = torch.from_numpy(dense_tensor).float()
    elif not isinstance(dense_tensor, torch.Tensor):
        raise TypeError("dense_tensor must be a numpy.ndarray or torch.Tensor")

    # initialization
    alpha = torch.ones(3) / 3
    dim = len(sparse_tensor.shape)
    dim_k = {k: [sparse_tensor.shape[d] for d in range(dim) if d != k] for k in range(dim)}

    M = sparse_tensor.clone()
    #pos_missing = torch.where(sparse_tensor == 0)
    mean_value = torch.mean(sparse_tensor[sparse_tensor != 0])
    M[pos_test] = mean_value
    #M[pos_test] = torch.mean(sparse_tensor[pos_train])

    # initialize Z3, T3
    Z3 = torch.cat([torch.zeros(1, *sparse_tensor.shape) for _ in range(dim)], dim=0)  # shape: 3 * dim1 * dim2 * dim3
    T3 = Z3.clone()  # shape: same as Z3

    max_value = max(sparse_tensor.max(), 1e4)
    min_value = min(sparse_tensor.min(), 0)
    print(max_value, min_value)

    # result recorder
    RMSE = torch.zeros(max_iter + 1)
    MAPE = torch.zeros(max_iter + 1)
    best_rmse = 200
    best_mae = 200
    minrmse_round = 0
    minmae_round = 0
    threshold = 2
    flag_rmse = True
    flag_mae = True
    tr = 0

    used_time = 0
    for it in range(max_iter):

        # compute the MAPE, RMSE
        mape = compute_mape(dense_tensor[pos_test], M[pos_test])
        rmse = compute_rmse(dense_tensor[pos_test], M[pos_test])
        MAPE[it] = mape
        RMSE[it] = rmse

        #if it % checkpoint == 0:
        logger.info(f"Iter: {it}, MAPE: {mape:.6f}, RMSE: {rmse:.6f}")

        start_time = time.time()

        # update rho
        M_latest = M.clone()
        rho = np.clip(rho * factor, 1e-10, 1e5)

        for k in range(dim):
            # update Zk
            M_k = unfolding(M - T3[k] / rho, k)
            Z_k = svt_scad(M_k, alpha[k] / rho, gamma=gamma, lamb=lamb)
            Z3[k] = folding(Z_k, k, dim_k[k])
            del M_k, Z_k

        # update M
        M[pos_test] = torch.mean(Z3 + T3 / rho, dim=0)[pos_test]
        M.clamp_(min_value, max_value)

        # update dual variable T
        T3 += rho * (Z3 - torch.cat([M.clone().unsqueeze(0) for _ in range(dim)], dim=0))

        tole = torch.linalg.norm(M - M_latest) / torch.linalg.norm(M_latest)

        used_time += time.time() - start_time
        # compute the tolerance
        if tole < tol:
            break
        if MAPE[it-1] - MAPE[it] > errorgap:
            if best_mae > mape:
                best_mae = mape
            minmae_round = it
            flag_mae = False
            tr = 0
        if RMSE[it-1] - RMSE[it] > errorgap:
            if best_rmse > rmse:
                best_rmse = rmse
            minrmse_round = it
            flag_rmse = False
            tr = 0
        if flag_rmse and flag_mae:
            tr += 1
            if tr == threshold:
                minrmse_round=it
                break
        flag_rmse = True
        flag_mae = True


    # compute the MAPE, RMSE
    #mape = compute_mape(dense_tensor[pos_test], M[pos_test])
    #rmse = compute_rmse(dense_tensor[pos_test], M[pos_test])
    #MAPE[it] = mape
    #RMSE[it] = rmse

    logger.info(f"Total iteration: {it + 1}, Running time: {used_time:.5f}, Tolerance: {tole * 1e5:.2f}e-5, minrmse_round: {minrmse_round}, minmae_round: {minmae_round}")
    logger.info(f"Imputation MAPE / RMSE: {best_rmse:.2f} / {best_mae:.2f}.")

    return M, RMSE[:it + 1], MAPE[:it + 1], used_time


initial_rho = 1e-5
gamma = 1000.
lamb = 3.

logger_dir = f"scad_contrast_model"
logging_save_dir = os.path.join(save_dir, logger_dir)
logger_info(logger_dir, log_path=os.path.join(logging_save_dir, f"completion_results.log"))
result_logger = logging.getLogger(logger_dir)
result_logger.info(
    "--------------------------------------------------------------------------------------------------------")
result_logger.info(f"initial rho: {initial_rho} \t gamma: {gamma} \t lamb: {lamb}.")

M, RMSE, MAPE, used_time = recover_data(
    test_tensor, train_tensor,pos_test,  # data and test mask
    initial_rho, gamma, lamb,  # algorithm parameters
    logger=result_logger,  # logger
)
