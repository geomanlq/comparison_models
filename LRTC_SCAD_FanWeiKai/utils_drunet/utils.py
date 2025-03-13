import os
import re
import glob
import torch
import random
import logging
import datetime
import numpy as np
from scipy import io

def initial_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        torch.cuda.manual_seed(seed)
    random.seed(seed)


'''
# --------------------------------------------
# Data Loading
# --------------------------------------------
'''
def load_data(dataset_name, dataroot):
    '''
    Load the dataset.
    Input:
        dataset_name: the name of dataset
        dataroot: the path of dataset
    Output:
        tensor: the original data with the shape of `(locations, days, time_intervals)`
    '''
    dataset_name += "-data-set"
    if dataset_name == "Hangzhou-data-set":
        file = os.path.join(dataroot, dataset_name, "tensor.mat")
        tensor = io.loadmat(file)['tensor']
    
    elif dataset_name == "PeMS-data-set":
        file = os.path.join(dataroot, dataset_name, "pems.npy")
        tensor = np.load(file).reshape(228, -1, 288)
    
    elif dataset_name == "Portland-data-set":
        file = os.path.join(dataroot, dataset_name, "volume.npy") # occupancy, speed and volume
        tensor = np.load(file).reshape(1156, -1, 96)
        
    elif dataset_name == "Seattle-data-set":
        file = os.path.join(dataroot, dataset_name, "tensor.npz")
        tensor = np.load(file)["arr_0"]

    try:
        tensor = torch.Tensor(tensor)
    except:
        tensor = torch.Tensor(tensor.astype(np.int16))
        
    return tensor


'''
# --------------------------------------------
# Missing Pattern Generation
# --------------------------------------------
'''
def missing_pattern(dense_tensor, ms, kind="random", block_window=12, seed=1000):
    initial_seed(seed)

    if kind == "random":
        binary_tensor = torch.round(torch.Tensor(np.random.rand(*dense_tensor.shape)) + 0.5 - ms)

    elif kind == "non-random":
        dim1, dim2, _ = dense_tensor.shape
        binary_tensor = torch.round(torch.Tensor(np.random.rand(dim1, dim2)) + 0.5 - ms)[:, :, None]

    elif kind == "blackout":
        dense_mat = dense_tensor.reshape(dense_tensor.shape[0], -1)
        T = dense_mat.shape[1]
        binary_blocks = np.round(np.random.rand(T // block_window) + 0.5 - ms)
        binary_mat = np.array([binary_blocks] * block_window).reshape(T, order="F")[None, :]
        binary_tensor = torch.Tensor(binary_mat.reshape(dense_tensor.shape[1], -1))[None, :, :]

    else:
        raise ValueError("Only 'random', 'non-random', and 'blackout' 3 kinds of missing patterns.")
    
    if kind == "blackout":
        # binary blocks used for showing the missing pattern
        return binary_tensor, binary_blocks
    else:
        return binary_tensor
    

'''
# --------------------------------------------
# Metrics
# --------------------------------------------
'''
def compute_rmse(var, var_hat):
    return torch.sqrt(torch.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return torch.sum(torch.abs(var - var_hat)) / var.shape[0]

    
'''
# --------------------------------------------
# logger
# --------------------------------------------
'''
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

def logger_close(logger):
    # close the logger
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

def find_last_checkpoint(params_dir, pretrained_path=None):
    """
    Args: 
        params_dir: model folder
        pretrained_path: pretrained model path. If params_dir does not have any model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(params_dir, '*_G.pth'))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_G.pth", file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(params_dir, '{}_G.pth'.format(init_iter))
    else:
        init_iter = 0
        init_path = pretrained_path
    return init_iter, init_path

