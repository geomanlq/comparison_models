import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut # Solve the equation a x = b for x, assuming a is a triangular matrix.

from rttc.miscs import seed_everything
from rttc.metrics import compute_mape, compute_rmse


def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def sample_factor(tau_sparse_tensor, tau_ind, factor, k, beta0 = 1):
    dim, rank = factor[k].shape
    dim = factor[k].shape[0]
    factor_bar = np.mean(factor[k], axis = 0)
    temp = dim / (dim + beta0)
    var_mu_hyper = temp * factor_bar
    var_W_hyper = inv(np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
    var_Lambda_hyper = wishart.rvs(df = dim + rank, scale = var_W_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)
    
    idx = list(filter(lambda x: x != k, range(len(factor))))
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ ten2mat(tau_sparse_tensor, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for i in range(dim):
        factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]

def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)


def bgcp(dense_tensor, sparse_tensor, factor, burn_iter, gibbs_iter, seed, 
         pos_missing=None, pos_test=None, progress_verbose=True):
    """Bayesian Gaussian CP (BGCP) decomposition."""
    seed_everything(seed)

    dim = np.array(sparse_tensor.shape)
    rank = factor[0].shape[1]
    ind = np.ones(dim)
    ind[pos_missing] = 0

    show_iter = 200
    tau = 1
    factor_plus = []
    for k in range(len(dim)):
        factor_plus.append(np.zeros((dim[k], rank)))
    temp_hat = np.zeros(dim)
    tensor_hat_plus = np.zeros(dim)
    for it in range(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor
        for k in range(len(dim)):
            factor[k] = sample_factor(tau_sparse_tensor, tau_ind, factor, k)
        tensor_hat = cp_combine(factor)
        temp_hat += tensor_hat
        tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
        if it + 1 > burn_iter:
            factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
            tensor_hat_plus += tensor_hat
            print(f'Burn iter: {it + 1}', end = '\r')
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print(f'  check temp hat: {np.isnan(temp_hat).any()}')
            print('Iter: {}'.format(it + 1))
            # print('MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], temp_hat[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], temp_hat[pos_test])))
            temp_hat = np.zeros(sparse_tensor.shape)
        elif progress_verbose:
            print(f'Iter: {it + 1}', end = '\r')
    factor = [i / gibbs_iter for i in factor_plus]
    tensor_hat = tensor_hat_plus / gibbs_iter
    # print('Imputation MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
    
    return tensor_hat, factor
