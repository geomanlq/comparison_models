"""Changepoint detection for traffic data using kernel PELT algorithm."""

import argparse
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from joblib import delayed, Parallel
import ruptures as rpt

from rttc.dataset import Dataset
from rttc.missing import MissingMask
from rttc.miscs import tqdm_joblib


def get_args():
    parser = argparse.ArgumentParser(description='Script to perform changepoint detection on traffic data.')

    parser.add_argument("--city", default="guangzhou", type=str, 
                        help="Name of the dataset. Options: guangzhou, t4c-london, t4c-madrid, t4c-melbourne.")
    parser.add_argument("--data_path", default="data/", type=str, 
                        help="Path where the dataset is stored.")
    parser.add_argument("--cpt_path", default="data/changepoints/", type=str, 
                        help="Path where the changepoints are stored.")
    
    parser.add_argument("--mr1", default=0.1, type=float, 
                        help="Missing rate for blackout missing.")
    parser.add_argument("--mr2", default=0.1, type=float, 
                        help="Missing rate for random missing.")
    parser.add_argument("--mr3", default=0.1, type=float, 
                        help="Missing rate for day missing.")
    parser.add_argument("--mr4", default=0.1, type=float, 
                        help="Missing rate for time missing.")
    
    parser.add_argument("--minsize", default=5, type=int, 
                        help="Minimal day gap to introduce a changepoint.")
    parser.add_argument("--pen", default=30, type=int, 
                        help="Regularization parameter in the PELT algorithm.")
    parser.add_argument("--missseed", default=33, type=int, 
                        help="Random state used to generate the missing pattern.")
    parser.add_argument("--njobs", default=16, type=int, 
                        help="Number of jobs to run in parallel for changepoint detection.")
    
    parser.add_argument("--dryrun", action='store_true', 
                        help="If enabled, run a test of the pipeline without exporting results.")
    parser.add_argument("--export_cpt", action='store_true', 
                        help="If enabled, export the changepoints to a file.")

    args = parser.parse_args()
    return args


def export_segments():
    args = get_args()
    
    # Load data
    city = args.city
    if city.startswith("t4c"):
        nan_value = -1
    else:
        nan_value = 0

    dense_tensor = Dataset(city, root=Path(args.data_path)).tensor
    dense_tensor = np.nan_to_num(dense_tensor, nan=nan_value)
    dim = dense_tensor.shape
    print(f"> Tensor shape: {dim}")
    n_time = dim[1] * dim[2]
    dense_matrix = dense_tensor.reshape(dense_tensor.shape[0], -1)
    
    # Generate missing patterns
    missing_pat_name = "composite"
    mr1, mr2, mr3, mr4 = args.mr1, args.mr2, args.mr3, args.mr4
    missing_masker = MissingMask(dim=dim, seed=args.missseed)
    mask_1 = missing_masker.generate_mask(mr1, pattern="blackout")
    mask_2 = missing_masker.generate_mask(mr2, pattern="random")
    mask_3 = missing_masker.generate_mask(mr3, pattern="day")
    mask_4 = missing_masker.generate_mask(mr4, pattern="time")
    mask = mask_1 * mask_2 * mask_3 * mask_4
    sparse_tensor = dense_tensor.copy()
    sparse_tensor[mask == 0] = np.nan
    sparse_tensor[sparse_tensor == nan_value] = np.nan
    missing_pattern = "".join([str(int(mr * 10)) for mr in (mr1, mr2, mr3, mr4)])
    print(missing_pattern)

    kernel = "rbf"
    min_size = dim[2] * args.minsize
    pen = args.pen
    print(f"min_size: {min_size} - args.minsize: {args.minsize} - pen: {pen}")

    def cpt_idx(idx):
        # Detect changepoints for a segment
        sig = rearrange(sparse_tensor, "s d t -> s (d t)")[idx]
        sig = sig[~np.isnan(sig)]
        std_sig = np.std(sig)
        if std_sig == 0:
            std_sig = 1
        norm_sig = (sig - np.mean(sig)) / std_sig

        result = rpt.KernelCPD(kernel=kernel, min_size=min_size).fit_predict(
            norm_sig, pen=pen
        )
        return result

    with tqdm_joblib(
        tqdm(desc="Changepoint detection", total=sparse_tensor.shape[0])
    ) as pbar:
        all_cpts = Parallel(n_jobs=args.njobs)(
            delayed(cpt_idx)(i) for i in range(sparse_tensor.shape[0])
        )

    # Export changepoints
    cpt_dir = Path(args.cpt_path)
    cpt_dir.mkdir(parents=True, exist_ok=True)
    if args.export_cpt:
        pickle.dump(
            all_cpts,
            open(
                cpt_dir / f"cpts_{city}_{missing_pat_name}_"
                f"{mr1}_{mr2}_{mr3}_{mr4}"
                f"__{args.minsize}_{pen}.pkl",
                "wb",
            ),
        )

    ################
    # Get indices of changepoints
    
    sensor_matrix = rearrange(sparse_tensor, "s d t -> s (d t)")
    all_cpts = [np.array(c) for c in all_cpts]

    # Indices of non-nan values in the original sequence
    nonan_idx = [
        np.argwhere(~np.isnan(sensor_ts)).ravel() for sensor_ts in sensor_matrix
    ]

    # Indices of changepoints in the original sequence
    shift_window = dim[2] // 2
    cpts_idx_prev = [
        nonan_ts[cpts[:-1] - shift_window] if len(cpts) > 1 else None
        for nonan_ts, cpts in zip(nonan_idx, all_cpts)
    ]
    cpts_idx_next = [
        nonan_ts[cpts[:-1] + shift_window] if len(cpts) > 1 else None
        for nonan_ts, cpts in zip(nonan_idx, all_cpts)
    ]
    cpts_idx = [
        ((_prev + _next) / 2).astype(int) if _prev is not None else None
        for _prev, _next in zip(cpts_idx_prev, cpts_idx_next)
    ]

    # Organize time series segments according to changepoints (index, start, end)
    ts_segments = []
    for idx, cpts in enumerate(cpts_idx):
        if cpts is None:
            ts_segments.append((idx, 0, sensor_matrix.shape[1]))
        else:
            ts_segments.append((idx, 0, cpts[0]))
            for i in range(len(cpts) - 1):
                ts_segments.append((idx, cpts[i], cpts[i + 1]))
            ts_segments.append((idx, cpts[-1], sensor_matrix.shape[1]))

    if not args.dryrun:
        cpt_dir = Path(args.cpt_path)
        # Export segments
        pickle.dump(
            ts_segments,
            open(
                cpt_dir / f"segs_{city}_{missing_pat_name}_"
                f"{mr1}_{mr2}_{mr3}_{mr4}"
                f"__{args.minsize}_{pen}.pkl",
                "wb",
            ),
        )


if __name__ == "__main__":
    export_segments()
