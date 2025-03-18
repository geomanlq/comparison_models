import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from rttc.metrics import compute_mape, compute_rmse, compute_mae


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


def svt_tnn(mat, tau, theta):
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=False)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[: theta] = 1
        mid[theta: idx] = (s[theta: idx] - tau) / s[theta: idx]
        return (u[:, : idx] @ np.diag(mid)) @ (u[:, : idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    idx = np.sum(s > tau)
    vec = s[: idx].copy()
    vec[theta: idx] = s[theta: idx] - tau
    return u[:, : idx] @ np.diag(vec) @ v[: idx, :]


def generate_Psi(dim_time, time_lags):
    Psis = []
    max_lag = np.max(time_lags)
    for i in range(len(time_lags) + 1):
        row = np.arange(0, dim_time - max_lag)
        if i == 0:
            col = np.arange(0, dim_time - max_lag) + max_lag
        else:
            col = np.arange(0, dim_time - max_lag) + max_lag - time_lags[i - 1]
        data = np.ones(dim_time - max_lag)
        Psi = sparse.coo_matrix((data, (row, col)), shape=(dim_time - max_lag, dim_time))
        Psis.append(Psi)
    return Psis


def latc(dense_tensor, sparse_tensor, time_lags,
         alpha, rho0, lambda0, theta,
         epsilon=1e-4, maxiter=100, K=3,
         pos_missing=None, pos_test=None,
         verbose=0):
    """Low-Rank Autoregressive Tensor Completion (LATC)"""

    dim = np.array(sparse_tensor.shape)
    dim_time = int(np.prod(dim) / dim[0])
    d = len(time_lags)
    max_lag = np.max(time_lags)
    sparse_mat = ten2mat(sparse_tensor, 0)
    dense_test = dense_tensor[pos_test]
    del dense_tensor

    T = np.zeros(dim)
    Z_tensor = sparse_tensor.copy()
    Z = sparse_mat.copy()
    A = 0.001 * np.random.rand(dim[0], d)
    Psis = generate_Psi(dim_time, time_lags)
    iden = sparse.coo_matrix((np.ones(dim_time), (np.arange(0, dim_time), np.arange(0, dim_time))),
                             shape=(dim_time, dim_time))
    ind = np.zeros((d, dim_time - max_lag), dtype=np.int_)
    for i in range(d):
        ind[i, :] = np.arange(max_lag - time_lags[i], dim_time - time_lags[i])
    last_mat = sparse_mat.copy()
    rho = rho0
    for it in range(maxiter):
        temp = []
        for m in range(dim[0]):
            Psis0 = Psis.copy()
            for i in range(d):
                Psis0[i + 1] = A[m, i] * Psis[i + 1]
            B = Psis0[0] - sum(Psis0[1:])
            temp.append(B.T @ B)
        for k in range(K):
            rho = min(rho * 1.05, 1e5)
            tensor_hat = np.zeros(dim)
            for p in range(len(dim)):
                tensor_hat += alpha[p] * mat2ten(svt_tnn(ten2mat(Z_tensor - T / rho, p),
                                                         alpha[p] / rho, theta), dim, p)
            temp0 = rho / lambda0 * ten2mat(tensor_hat + T / rho, 0)
            mat = np.zeros((dim[0], dim_time))
            for m in range(dim[0]):
                mat[m, :] = spsolve(temp[m] + rho * iden / lambda0, temp0[m, :])
            Z_tensor = mat2ten(Z, dim, 0)
            mat_tensor = mat2ten(mat, dim, 0)
            Z_tensor[pos_missing] = mat_tensor[pos_missing]
            T = T + rho * (tensor_hat - Z_tensor)
        for m in range(dim[0]):
            A[m, :] = np.linalg.lstsq(Z[m, ind].T, Z[m, max_lag:], rcond=None)[0]
        mat_hat = ten2mat(tensor_hat, 0)
        tol = (np.linalg.norm(mat_hat - last_mat) / np.linalg.norm(last_mat)).item()
        # tol = np.mean((mat_hat - last_mat)**2)**0.5
        last_mat = mat_hat.copy()
        
        if verbose > 0:
            if (it + 1) % verbose == 0:
                rmse = compute_rmse(dense_test, tensor_hat[pos_test])
                mae = compute_mae(dense_test, tensor_hat[pos_test])
                print(f"Iter {it + 1:>3}, {rmse = :.4f}, {mae = :.4f}"
                        f", {tol = :.6f}, {rho = :.3f}")
            else:
                print(f"Iter {it + 1}", end="\r")

        if tol < epsilon:
            print(f"Converged at iter {it + 1}.")
            break

    return tensor_hat
