import os
import random
import numpy as np
import contextlib
import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class no_tqdm:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is not None:
        if min_seed_value <= seed <= max_seed_value:
            seed = int(seed)
        else:
            print(f"Seed must be between {min_seed_value} and {max_seed_value}.")
            seed = random.randint(min_seed_value, max_seed_value)
    else:
        seed = random.randint(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
    return seed


# Empty Placeholder for Neptune Object
from neptune.metadata_containers import Run

class NoneNeptuneRun(Run):
    def __init__(self):
        self.items = EmptyNeptuneItem()

    def __getitem__(self, index):
        return self.items

    def __setitem__(self, index, value):
        pass


class EmptyNeptuneItem:
    def __init__(self):
        pass
    
    def append(self, *args, **kwargs):
        pass
    
    def add(self, *args, **kwargs):
        pass

    def __getitem__(self, index):
        return self

    def __setitem__(self, index, value):
        pass


def check_folder(filename):
    parent_folder = os.path.dirname(filename)
    os.makedirs(parent_folder, exist_ok=True)
