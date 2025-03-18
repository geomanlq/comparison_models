import numpy as np
from numpy.linalg import inv as inv

from rttc.metrics import compute_mape, compute_rmse


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)

def svt_tnn(mat, alpha, rho, theta):
    tau = alpha / rho
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices = 0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta:idx] = s[theta:idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]

# def svt_tnn(mat, alpha, rho, theta):
#     """This is a Numpy dependent singular value thresholding (SVT) process."""
#     u, s, v = np.linalg.svd(mat, full_matrices = False)
#     vec = s.copy()
#     vec[theta :] = s[theta :] - alpha / rho
#     vec[vec < 0] = 0
#     return np.matmul(np.matmul(u, np.diag(vec)), v)


def lrtc_tnn(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, max_iter, 
             verbose=0, pos_missing=None, pos_test=None, progress_verbose=True):
    """Low-Rank Tenor Completion with Truncated Nuclear Norm, LRTC-TNN."""
    
    dim = np.array(sparse_tensor.shape)
    
    X = np.zeros(np.insert(dim, 0, len(dim))) # \boldsymbol{\mathcal{X}}
    T = np.zeros(np.insert(dim, 0, len(dim))) # \boldsymbol{\mathcal{T}}
    Z = sparse_tensor.copy()
    last_tensor = sparse_tensor.copy()
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            X[k] = mat2ten(svt_tnn(ten2mat(Z - T[k] / rho, k), alpha[k], 
                                   rho, int(np.ceil(theta * dim[k]))), dim, k)
        Z[pos_missing] = np.mean(X + T / rho, axis = 0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
        tensor_hat = np.einsum('k, kmnt -> mnt', alpha, X)
        tol = (np.linalg.norm(tensor_hat - last_tensor) / np.linalg.norm(last_tensor)).item()
        last_tensor = tensor_hat.copy()
        it += 1
        if verbose > 0:
            if (it + 1) % verbose == 0:
                print('Iter: {}'.format(it + 1), end = '\t')
                print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
            elif progress_verbose:
                print('Iter: {}'.format(it + 1), end = '\r')
        if (tol < epsilon) or (it >= max_iter):
            break

    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
    
    return tensor_hat
