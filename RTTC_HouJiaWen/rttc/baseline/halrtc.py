import numpy as np
from rttc.metrics import compute_mape, compute_rmse


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, dim, mode):
    index = list()
    index.append(mode)
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order='F'), 0, mode)


def svt(mat, tau):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    vec = s - tau
    vec[vec < 0] = 0
    return np.matmul(np.matmul(u, np.diag(vec)), v)


def halrtc(dense_tensor, sparse_tensor,
           alpha: list, rho: float,
           epsilon: float, maxiter: int,
           vmean=0, vstd=1,
           pos_missing=None, pos_test=None,
           progress_verbose=True):
    dim = np.array(sparse_tensor.shape)
    
    dense_tensor = (dense_tensor - vmean) / vstd
    sparse_tensor = (sparse_tensor - vmean) / vstd

    dense_test = dense_tensor[pos_test]
    del dense_tensor
    tensor_hat = sparse_tensor.copy()
    B = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
    Y = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
    last_ten = sparse_tensor.copy()

    for it in range(maxiter):
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            B[k] = mat2ten(svt(ten2mat(tensor_hat + Y[k] / rho, k), alpha[k] / rho), dim, k)
        tensor_hat[pos_missing] = ((sum(B) - sum(Y) / rho) / 3)[pos_missing]
        for k in range(len(dim)):
            Y[k] = Y[k] - rho * (B[k] - tensor_hat)
        tol = (np.linalg.norm(tensor_hat - last_ten) / np.linalg.norm(last_ten)).item()
        last_ten = tensor_hat.copy()
        if it % 50 == 0:
            rmse = compute_rmse(dense_test, tensor_hat[pos_test])
            print(f'Iter: {it} - Tol: {tol:.6} - RMSE: {rmse}')
        elif progress_verbose:
            print(f'Iter: {it}', end='\r')
        if tol < epsilon:
            break

    print('RMSE: {:.6}'.format(compute_rmse(dense_test, tensor_hat[pos_test])))
    
    tensor_hat = tensor_hat * vstd + vmean
    return tensor_hat
