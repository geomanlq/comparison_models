"""Run single traffic imputation scenario."""

import os

os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"
] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
from pathlib import Path
import torch
import wandb

from rttc.model import *
from rttc.metrics import *
from rttc.miscs import seed_everything
from rttc.quickload import load_all_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="guangzhou", type=str)

    parser.add_argument("--data_path", default="data/", type=str)
    parser.add_argument("--cpt_path", default="data/changepoints/", type=str)
    parser.add_argument("--name_suffix", default="", type=str)
    parser.add_argument("--mr1", default=0.1, type=float)
    parser.add_argument("--mr2", default=0.1, type=float)
    parser.add_argument("--mr3", default=0.1, type=float)
    parser.add_argument("--mr4", default=0.1, type=float)

    parser.add_argument("--rank1", default=30, type=int)
    parser.add_argument("--rank2", default=9, type=int)
    parser.add_argument("--rank3", default=5, type=int)
    parser.add_argument("--rank4", default=20, type=int)

    parser.add_argument("--mu", default=0.1, type=float)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--xi", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.01, type=float)
    parser.add_argument("--rho", default=None, type=float)
    parser.add_argument("--init", default="daymean", type=str)
    parser.add_argument("--maxiter", default=250, type=int)
    parser.add_argument("--warmiter", default=20, type=int)
    parser.add_argument("--epsilon", default=1e-4, type=float)
    parser.add_argument("--incre", default=0.05, type=float)
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--init_file", default="", type=str)
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--verbose", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--missseed", default=33, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--output_path", default="out/", type=str)
    parser.add_argument("--save_tensor", action="store_true")
    parser.add_argument("--wandb_proj", default="rttc", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load data
    city = args.city
    # Specify nan value
    if city.startswith("t4c"):
        nan_value = -1
    elif city == "guangzhou":
        nan_value = 0
    else:
        nan_value = 0

    # Load arguments
    cpt_type = "RBF"
    max_iter = args.maxiter
    warm_iter = args.warmiter
    epsilon = args.epsilon
    incre = args.incre
    tau = args.tau
    rho = args.rho

    verbose = args.verbose
    init = args.init
    seed = args.seed
    missseed = args.missseed
    device = args.device
    wandb_proj = args.wandb_proj

    if args.dryrun:
        os.environ['WANDB_MODE'] = "dryrun"

    #######################################################
    # Load data
    seed_everything(args.seed)
    mr1, mr2, mr3, mr4 = args.mr1, args.mr2, args.mr3, args.mr4
    scenario = load_all_data(args, city, nan_value, mr1, mr2, mr3, mr4)

    #######################################################
    # Run models

    def meta_objective(rank, mu, lam, xi, gamma, rho):
        if rho is None:
            rho = 1e-5 if city != "guangzhou" else 1e-2
        rho_ = rho

        r1, r2, r3, r4 = [int(r_) for r_ in rank]

        print("--", mu, lam, xi, gamma, r1, r2, r3, r4, rho_)

        short_param = f"{mu}-{lam}-{xi}-{gamma}-{r1}-{r2}-{r3}-{r4}-{rho}"

        # load scenario
        (
            dense_tensor,
            sparse_tensor,
            dense_tensor_4d,
            sparse_tensor_4d,
            pos_obs,
            pos_test,
            pos_missing,
            pos_obs_4d,
            pos_test_4d,
            pos_missing_4d,
            init_tensor,
            init_tensor_4d,
            cpt_segs,
            dim,
            vmax,
            vmean,
            n_test,
            n_week,
        ) = scenario

        fail_flag = True
        for try_times in range(3):
            try:
                # Initialize wandb logger
                wandb.init(
                    name=f"{city}-{cpt_type}{args.name_suffix}",
                    project=wandb_proj,
                    config={
                        "city": city,
                        "mr1": mr1,
                        "mr2": mr2,
                        "mr3": mr3,
                        "mr4": mr4,
                        "model": "RTTC",
                        "cpt_type": cpt_type,
                        "mu": mu,
                        "lam": lam,
                        "xi": xi,
                        "gamma": gamma,
                        "r1": r1,
                        "r2": r2,
                        "r3": r3,
                        "r4": r4,
                        "rho": rho_,
                        "tau": tau,
                        "init": init,
                        "maxiter": max_iter,
                        "warmiter": warm_iter,
                        "epsilon": epsilon,
                        "incre": incre,
                        "seed": seed,
                        "missseed": missseed,
                        "device": device,
                    },
                )

                imputer = TuckerADMM(
                    dense_tensor=dense_tensor_4d,
                    cpt_segs=cpt_segs,
                    rank=rank,
                    mu=mu,
                    lam=lam,
                    xi=xi,
                    rho=rho_,
                    gamma=gamma,
                    tau=tau,
                    vmax=vmax,
                    vmean=vmean,
                    max_iter=max_iter,
                    warm_iter=warm_iter,
                    epsilon=epsilon,
                    incre=incre,
                    verbose=verbose,
                    random_state=seed,
                    device=device,
                    logger=wandb,
                )
                yhat = imputer.fit(
                    sparse_tensor_4d,
                    pos_missing=pos_missing_4d,
                    pos_test=pos_test_4d,
                    init_tensor=init_tensor_4d,
                )
                yhat = yhat[:, : dim[1], :]
                # yhat[yhat < 0] = 0
                fail_flag = False
                break
            except (
                ValueError,
                np.linalg.LinAlgError,
                torch.linalg.LinAlgError,
            ) as e:
                rho_ *= 10
                print(try_times, " - ", e)
                wandb.finish()
        if fail_flag:
            print(f"Failed: {short_param}")

        mae = compute_mae(dense_tensor[pos_test], yhat[pos_test])
        rmse = compute_rmse(dense_tensor[pos_test], yhat[pos_test])
        nmae = compute_nmae(dense_tensor[pos_test], yhat[pos_test])
        nrmse = compute_nrmse(dense_tensor[pos_test], yhat[pos_test])
        smape = compute_smape(dense_tensor[pos_test], yhat[pos_test])

        if args.save_tensor:
            output_dir = Path(args.output_path)
            save_file = f"yhat_{city}-{r1}-{r2}-{r3}-{r4}.npy"
            np.save(output_dir / save_file, yhat)

        try:
            wandb.log({
                "metrics/mae": mae,
                "metrics/rmse": rmse,
                "metrics/nmae": nmae,
                "metrics/nrmse": nrmse,
                "metrics/smape": smape,
            })
        except:
            pass

        wandb.finish()

    #######################################################

    meta_objective(
        [args.rank1, args.rank2, args.rank3, args.rank4],
        args.mu,
        args.lam,
        args.xi,
        args.gamma,
        rho,
    )

    print("Finished!!!")
