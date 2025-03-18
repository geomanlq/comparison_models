import pickle
import numpy as np

from pathlib import Path
from einops import rearrange

from rttc.baseline import daily_mean
from rttc.dataset import Dataset
from rttc.missing import MissingMask


def load_all_data(
    args,
    city,
    nan_value,
    mr1, mr2, mr3, mr4
):
    """Load data util for run file."""

    # Load dataset
    data_root = Path(args.data_path)
    dense_tensor = Dataset(city, root=data_root).tensor
    dense_tensor = np.nan_to_num(dense_tensor, nan=nan_value)
    dim = dense_tensor.shape

    # Generate missing patterns
    missing_pat_name = "composite"
    missing_masker = MissingMask(dim=dim, seed=args.missseed)
    mask_1 = missing_masker.generate_mask(mr1, pattern="blackout")
    mask_2 = missing_masker.generate_mask(mr2, pattern="random")
    mask_3 = missing_masker.generate_mask(mr3, pattern="day")
    mask_4 = missing_masker.generate_mask(mr4, pattern="time")
    mask = mask_1 * mask_2 * mask_3 * mask_4
    sparse_tensor = dense_tensor.copy()
    sparse_tensor[mask == 0] = nan_value

    # Load precomputed changepoints
    cpt_segs_filename = Path(args.cpt_path).glob(
                             f'segs_{city}_{missing_pat_name}_'
                             f'{mr1}_{mr2}_{mr3}_{mr4}*.pkl')
    cpt_segs_filename = list(cpt_segs_filename)[0]
    cpt_segs = pickle.load(open(cpt_segs_filename, 'rb'))

    pos_obs = np.where(sparse_tensor != nan_value)
    pos_missing = np.where(sparse_tensor == nan_value)
    pos_test = np.where((dense_tensor != nan_value) & (sparse_tensor == nan_value))
    
    # Exclude anomalies during evaluation
    ravel_pos_test = np.ravel_multi_index(pos_test, dim)
    ravel_pos_anomaly = np.load(data_root / f'ravel_pos_anomaly/{city}.npy')
    ravel_pos_test = np.setdiff1d(ravel_pos_test, ravel_pos_anomaly)
    pos_test = np.unravel_index(ravel_pos_test, dim)
    
    binary_tensor = np.zeros_like(dense_tensor)
    binary_tensor[pos_test] = 1
    
    n_test = len(pos_test[0])
    n_week = np.ceil(dim[1] / 7).astype(int)

    def tensor3to4(tensor, pad_value=0):
        new_tensor = np.pad(
            tensor,
            pad_width=((0, 0), (0, n_week * 7 - dim[1]), (0, 0)),
            constant_values=pad_value,
        )
        new_tensor = rearrange(new_tensor, "s (w d) t -> s w d t", d=7)
        return new_tensor
    
    sparse_tensor_4d = tensor3to4(sparse_tensor, pad_value=nan_value)
    dense_tensor_4d = tensor3to4(dense_tensor, pad_value=nan_value)
    binary_tensor_4d = tensor3to4(binary_tensor, pad_value=0)
    
    pos_obs_4d = np.where(sparse_tensor_4d != nan_value)
    pos_test_4d = np.where(binary_tensor_4d == 1)
    pos_missing_4d = np.where(sparse_tensor_4d == nan_value)

    sparse_tensor[sparse_tensor == nan_value] = 0
    sparse_tensor_4d[sparse_tensor_4d == nan_value] = 0

    del binary_tensor, binary_tensor_4d

    # Initialize tensor
    if args.init == "daymean":
        sparse_init = sparse_tensor.copy()
        sparse_init[pos_missing] = np.nan
        init_tensor = daily_mean(
            dense_tensor, sparse_init, pos_missing=pos_missing, pos_obs=pos_obs
        )
        init_tensor_4d = np.pad(
            init_tensor,
            pad_width=((0, 0), (0, n_week * 7 - dim[1]), (0, 0)),
            constant_values=nan_value,
        )
        init_tensor_4d = rearrange(init_tensor_4d, "s (w d) t -> s w d t", d=7)
    elif args.init == "precomputed":
        init_tensor = np.load(args.init_file)
        init_tensor_4d = np.pad(
            init_tensor,
            pad_width=((0, 0), (0, n_week * 7 - dim[1]), (0, 0)),
            constant_values=nan_value,
        )
        init_tensor_4d = rearrange(init_tensor_4d, "s (w d) t -> s w d t", d=7)
    else:
        init_tensor = init_tensor_4d = args.init

    if args.normalize:
        vmax = np.percentile(dense_tensor[pos_obs], 99)
        vmean = np.median(dense_tensor[pos_obs])
    else:
        vmax = vmean = None

    return (
        dense_tensor, sparse_tensor,
        dense_tensor_4d, sparse_tensor_4d,
        pos_obs, pos_test, pos_missing,
        pos_obs_4d, pos_test_4d, pos_missing_4d,
        init_tensor, init_tensor_4d,
        cpt_segs, dim,
        vmax, vmean,
        n_test, n_week
    )
