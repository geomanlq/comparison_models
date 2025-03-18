import numpy as np

def sensor_mean(dense_tensor, sparse_tensor, pos_missing=None, pos_obs=None):
    sparse_tensor = sparse_tensor.copy()
    sparse_tensor[pos_missing] = np.nan
    dim = dense_tensor.shape
    tensor_hat = np.nanmean(sparse_tensor, axis=(1, 2), keepdims=True)
    tensor_hat = np.array(np.broadcast_to(tensor_hat, dim))
    tensor_hat[pos_obs] = sparse_tensor[pos_obs]

    return tensor_hat


def daily_mean(dense_tensor, sparse_tensor, pos_missing=None, pos_obs=None):
    sensor_mean_hat = sensor_mean(dense_tensor, sparse_tensor, pos_missing, pos_obs)
    
    sparse_tensor = sparse_tensor.copy()
    sparse_tensor[pos_missing] = np.nan
    dim = dense_tensor.shape
    
    tensor_hat = np.nanmean(sparse_tensor, axis=1, keepdims=True)
    tensor_hat = np.array(np.broadcast_to(tensor_hat, dim))
    tensor_hat[pos_obs] = sparse_tensor[pos_obs]
    tensor_hat[np.isnan(tensor_hat)] = sensor_mean_hat[np.isnan(tensor_hat)]

    return tensor_hat
