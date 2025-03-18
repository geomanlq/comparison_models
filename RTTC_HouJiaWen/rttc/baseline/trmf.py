import numpy as np
from numpy.linalg import inv

from rttc.metrics import compute_rmse


def trmf(
    dense_tensor, sparse_tensor,
    init_para, init_hyper,
    time_lags, maxiter,
    dim, vmean, vstd,
    pos_obs, pos_test,
    verbose=50,
    progress_verbose=True
):
    """Temporal Regularized Matrix Factorization, TRMF."""
    dim_mat = tuple((dim[0], dim[1] * dim[2]))
    dense_mat = (dense_tensor.reshape(dim_mat) - vmean) / vstd
    sparse_mat = (sparse_tensor.reshape(dim_mat) - vmean) / vstd
    
    ## Initialize parameters
    W = init_para["W"]
    X = init_para["X"]
    theta = init_para["theta"]
    
    ## Set hyperparameters
    lambda_w = init_hyper["lambda_w"]
    lambda_x = init_hyper["lambda_x"]
    lambda_theta = init_hyper["lambda_theta"]
    eta = init_hyper["eta"]
    
    dim1, dim2 = sparse_mat.shape
    # pos_obs = np.where(sparse_mat != 0)
    # pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))
    binary_tensor = sparse_tensor.copy()
    binary_tensor[pos_test] = 1
    binary_mat = binary_tensor.reshape(dim_mat)
    pos_test_mat = np.where(binary_mat == 1)

    binary_tensor = sparse_tensor.copy()
    binary_tensor[pos_obs] = 1
    binary_mat = binary_tensor.reshape(dim_mat)
    del binary_tensor
    
    d, rank = theta.shape
    
    def inv_scale_rmse(y, yhat, idx):
        y_ = y[idx] * vstd + vmean
        yhat_ = yhat[idx] * vstd + vmean
        return compute_rmse(y_, yhat_)
    
    for it in range(maxiter):
        ## Update spatial matrix W
        for i in range(dim1):
            # pos0 = np.where(sparse_mat[i, :] != 0)
            pos0 = np.where(binary_mat[i, :] == 1)
            Xt = X[pos0[0], :]
            vec0 = Xt.T @ sparse_mat[i, pos0[0]]
            mat0 = inv(Xt.T @ Xt + lambda_w * np.eye(rank))
            W[i, :] = mat0 @ vec0
        ## Update temporal matrix X
        for t in range(dim2):
            # pos0 = np.where(sparse_mat[:, t] != 0)
            pos0 = np.where(binary_mat[:, t] == 1)
            Wt = W[pos0[0], :]
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    Ak = theta[k, :]
                    Mt += np.diag(Ak ** 2)
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Nt += np.multiply(Ak, X[t + time_lags[k], :]
                                      - np.einsum('ij, ij -> j', theta0, X[t + time_lags[k] - time_lags, :]))
            vec0 = Wt.T @ sparse_mat[pos0[0], t] + lambda_x * Nt + lambda_x * Qt
            mat0 = inv(Wt.T @ Wt + lambda_x * Mt + lambda_x * Pt + lambda_x * eta * np.eye(rank))
            X[t, :] = mat0 @ vec0
            
        ## Update AR coefficients theta
        for k in range(d):
            theta0 = theta.copy()
            theta0[k, :] = 0
            mat0 = np.zeros((dim2 - np.max(time_lags), rank))
            for L in range(d):
                mat0 += X[np.max(time_lags) - time_lags[L] : dim2 - time_lags[L] , :] @ np.diag(theta0[L, :])
            VarPi = X[np.max(time_lags) : dim2, :] - mat0
            var1 = np.zeros((rank, rank))
            var2 = np.zeros(rank)
            for t in range(np.max(time_lags), dim2):
                B = X[t - time_lags[k], :]
                var1 += np.diag(np.multiply(B, B))
                var2 += np.diag(B) @ VarPi[t - np.max(time_lags), :]
            theta[k, :] = inv(var1 + lambda_theta * np.eye(rank) / lambda_x) @ var2

        
        if (it + 1) % verbose == 0:
            mat_hat = W @ X.T
            rmse = inv_scale_rmse(dense_mat, mat_hat, pos_test_mat)
            # rmse = compute_rmse(dense_mat[pos_test_mat], mat_hat[pos_test_mat])
            print(f'Iter: {it} - Imputation RMSE: {rmse:.6}')
        elif progress_verbose:
            print(f'Iter: {it}', end='\r')
        
    mat_hat = W @ X.T
    yhat = mat_hat.reshape(dim) * vstd + vmean
    return yhat


# def trmf_torch(
#     dense_tensor, sparse_tensor,
#     init_para, init_hyper,
#     time_lags, maxiter,
#     dim, pos_obs, pos_test,
#     progress_verbose=True,
#     device='cpu'
# ):
#     import torch
#     import tensorly as tl
#     from torch.linalg import inv
#     tl.set_backend('pytorch')
    
#     def mk_tsr(arr):
#         return tl.tensor(arr).long()
    
#     """Temporal Regularized Matrix Factorization, TRMF."""
#     dim_mat = tuple((dim[0], dim[1] * dim[2]))
#     dense_mat = tl.tensor(dense_tensor.reshape(dim_mat)).to(device)
#     sparse_mat = tl.tensor(sparse_tensor.reshape(dim_mat)).to(device)
    
#     ## Initialize parameters
#     W = tl.tensor(init_para["W"]).to(device)
#     X = tl.tensor(init_para["X"]).to(device)
#     theta = tl.tensor(init_para["theta"]).to(device)
#     time_lags = tl.tensor(time_lags).to(device)
#     max_lags = int(torch.max(time_lags).item())
    
#     ## Set hyperparameters
#     lambda_w = init_hyper["lambda_w"]
#     lambda_x = init_hyper["lambda_x"]
#     lambda_theta = init_hyper["lambda_theta"]
#     eta = init_hyper["eta"]
    
#     dim1, dim2 = sparse_mat.shape
#     binary_tensor = tl.tensor(sparse_tensor).to(device)
#     binary_tensor[pos_test] = 1
#     binary_mat = binary_tensor.reshape(dim_mat)
#     pos_test_mat = torch.where(binary_mat == 1)

#     binary_tensor = tl.tensor(sparse_tensor).to(device)
#     binary_tensor[pos_obs] = 1
#     binary_mat = binary_tensor.reshape(dim_mat)
#     del binary_tensor
    
#     d, rank = theta.shape
#     eye_rank = torch.eye(rank).to(device)
    
#     for it in range(maxiter):
#         ## Update spatial matrix W
#         for i in range(dim1):
#             pos0 = torch.where(binary_mat[i, :] == 1)
#             Xt = X[pos0[0], :]
#             vec0 = Xt.T @ sparse_mat[i, pos0[0]]
#             mat0 = inv(Xt.T @ Xt + lambda_w * eye_rank)
#             W[i, :] = mat0 @ vec0
#         ## Update temporal matrix X
#         for t in range(dim2):
#             # pos0 = torch.where(sparse_mat[:, t] != 0)
#             pos0 = torch.where(binary_mat[:, t] == 1)
#             Wt = W[pos0[0], :]
#             Mt = torch.zeros((rank, rank)).to(device)
#             Nt = torch.zeros(rank).to(device)
#             if t < max_lags:
#                 Pt = torch.zeros((rank, rank)).to(device)
#                 Qt = torch.zeros(rank).to(device)
#             else:
#                 Pt = eye_rank
#                 Qt = torch.einsum('ij, ij -> j', theta, X[mk_tsr(t - time_lags), :])
#             if t < dim2 - torch.min(time_lags):
#                 if t >= max_lags and t < dim2 - max_lags:
#                     index = list(range(0, d))
#                 else:
#                     index = list(torch.where((t + time_lags >= max_lags) & (t + time_lags < dim2)))[0]
#                 for k in index:
#                     Ak = theta[k, :]
#                     Mt += torch.diag(Ak ** 2).to(device)
#                     theta0 = theta.clone()
#                     theta0[k, :] = 0
#                     Nt += torch.multiply(Ak, X[mk_tsr(t + time_lags[k]), :]
#                                       - torch.einsum('ij, ij -> j', 
#                                                      theta0, X[mk_tsr(t + time_lags[k] - time_lags), :]))
#             vec0 = Wt.T @ sparse_mat[pos0[0], t] + lambda_x * Nt + lambda_x * Qt
#             mat0 = inv(Wt.T @ Wt + lambda_x * Mt + lambda_x * Pt + lambda_x * eta * eye_rank)
#             X[t, :] = mat0 @ vec0
            
#         ## Update AR coefficients theta
#         for k in range(d):
#             theta0 = theta.clone()
#             theta0[k, :] = 0
#             mat0 = torch.zeros((dim2 - max_lags, rank)).to(device)
#             for L in range(d):
#                 mat0 += X[mk_tsr(max_lags - time_lags[L]) : mk_tsr(dim2 - time_lags[L]) , :] @ torch.diag(theta0[L, :]).to(device)
#             VarPi = X[max_lags : dim2, :] - mat0
#             var1 = torch.zeros((rank, rank)).to(device)
#             var2 = torch.zeros(rank).to(device)
#             for t in range(max_lags, dim2):
#                 B = X[mk_tsr(t - time_lags[k]), :]
#                 var1 += torch.diag(torch.multiply(B, B)).to(device)
#                 var2 += torch.diag(B).to(device) @ VarPi[t - max_lags, :]
#             theta[k, :] = inv(var1 + lambda_theta * eye_rank / lambda_x) @ var2

        
#         if (it + 1) % 100 == 0:
#             mat_hat = W @ X.T
#             rmse = compute_rmse(dense_mat.to('cpu').numpy()[pos_test_mat], 
#                                 mat_hat.to('cpu').numpy()[pos_test_mat])
#             print(f'Iter: {it} - Imputation RMSE: {rmse:.6}')
#         elif progress_verbose:
#             print(f'Iter: {it}', end='\r')
        
#     mat_hat = W @ X.T
#     yhat = mat_hat.reshape(dim)
#     return yhat
