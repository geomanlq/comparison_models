import numpy as np


def _multi_ravel(*arr):
    return tuple(a.ravel() for a in arr)


def compute_rmse(y, y_hat):
    y, y_hat = _multi_ravel(y, y_hat)
    return np.sqrt(np.mean((y - y_hat) ** 2))


def compute_mape(y, y_hat):
    y, y_hat = _multi_ravel(y, y_hat)
    return np.mean(np.abs(y - y_hat) / y)


def compute_mae(y, y_hat):
    y, y_hat = _multi_ravel(y, y_hat)
    return np.mean(np.abs(y - y_hat))


def compute_nmae(y, y_hat):
    y, y_hat = _multi_ravel(y, y_hat)
    return np.mean(np.abs(y - y_hat) / np.ptp(y))


def compute_nrmse(y, y_hat):
    y, y_hat = _multi_ravel(y, y_hat)
    return np.sqrt(np.mean((y - y_hat) ** 2)) / np.ptp(y)


def compute_mase(y, y_hat):
    """Mean absolute scaled error."""
    y, y_hat = _multi_ravel(y, y_hat)
    return np.mean(np.abs(y - y_hat)) / np.mean(np.abs(np.diff(y)))


def compute_smape(y, y_hat):
    """Symmetric mean absolute percentage error.
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    
    SMAPE = 100% / n * ∑((|y - y_hat|) / (|y| + |y_hat|))
    """
    y, y_hat = _multi_ravel(y, y_hat)
    total_ape = np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat))
    total_ape = np.nan_to_num(total_ape)
    return np.mean(total_ape) * 100


def compute_logerror(y, y_hat):
    return np.log(y_hat / y)


# def compute_dtw(y, yhat, radius=3):
#     from joblib import Parallel, delayed
#     from tqdm import tqdm
#     from src.miscs import tqdm_joblib
#     dim = y.shape if isinstance(y, np.ndarray) else y.dim
#     def cc(a, b):
#         return fastdtw(a, b, radius=radius)

#     with tqdm_joblib(tqdm(desc="Computing DTW", total=dim[0])):
#         dtws = Parallel(n_jobs=16)(
#             delayed(cc)(a, b)
#             for a, b in zip(y.reshape(dim[0], -1), yhat.reshape(dim[0], -1))
#         )
#     dtw_tucker = [di[0] for di in dtws]
#     return dtw_tucker
