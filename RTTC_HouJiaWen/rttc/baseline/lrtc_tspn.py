#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for tensor operations

@author: nietong
"""

import numpy as np
from collections import deque

from rttc.metrics import compute_rmse, compute_mae


def shiftdim(array, n=None):
    if n is not None:
        if n >= 0:
            axes = tuple(range(len(array.shape)))
            new_axes = deque(axes)
            new_axes.rotate(n)
            return np.moveaxis(array, axes, tuple(new_axes))
        return np.expand_dims(array, axis=tuple(range(-n)))
    else:
        idx = 0
        for dim in array.shape:
            if dim == 1:
                idx += 1
            else:
                break
        axes = tuple(range(idx))
        # Note that this returns a tuple of 2 results


def Fold(X, dim, i):
    # Fold a matrix into a tensor in mode i, dim is a tuple of the targeted tensor.
    dim = np.roll(dim, -i)
    X = shiftdim(np.reshape(X, dim, order='F'), len(dim)-i)
    return X


def Unfold(X, dim, i):
    # Unfold a tensor into a matrix in mode i.
    X_unfold = np.reshape(shiftdim(X,i), (dim[i],-1), order='F')
    return X_unfold

# Optimize Truncated Schatten p-norm via ADMM

def truncation(unfoldj, theta):
    # calculate the truncation of each unfolding
    dim = np.array(unfoldj.shape)
    wj = np.zeros(min(dim),)
    r = int(np.ceil(theta * min(dim)))
    wj[r:] = 1
    return wj


def GST(sigma, w, p, J=5):
    # J is the inner iterations of GST, J=5 is supposed to be enough
    # Generalized soft-thresholding algorithm
    if w == 0:
        Sp = sigma
    else:
        dt = np.zeros(J+1)
        tau = (2*w*(1-p))**(1/(2-p)) + w*p*(2*w*(1-p))**((p-1)/(2-p))
        if np.abs(sigma) <= tau:
            Sp = 0
        else:
            dt[0] = np.abs(sigma)
            for k in range(J):
                dt[k+1] = np.abs(sigma) - w*p*(dt[k])**(p-1)
            Sp = np.sign(sigma)*dt[k].item()

    return Sp


def update_Mi(mat, alphai, beta, p, theta):
    #update M variable
    delta = []
    try:
        u, d, v = np.linalg.svd(mat, full_matrices=False)
    except:
        import scipy.linalg as sl
        u, d, v = sl.svd(mat, full_matrices=False, lapack_driver='gesvd')
    wi = truncation(mat, theta)
    for j in range(len(d)):
        deltaj = GST(d[j], (alphai / beta) * wi[j], p)
        delta.append(deltaj)
    delta = np.diag(delta)
    Mi = u @ delta @ v
    return Mi


def TSpN_ADMM(X_true, X_missing, Omega,
              p, theta, alpha, rho, incre,
              vmean=0, vstd=1, maxiter=200,
              epsilon=1e-4, verbose=0,
              pos_missing=None, pos_test=None, **kwargs):
    X = X_missing.copy()
    X[Omega == False] = np.mean(X_missing[Omega]) # Initialize with mean values
    dim = X_missing.shape
    M = np.zeros(np.insert(dim, 0, len(dim))) # M is a 4-th order tensor
    Q = np.zeros(np.insert(dim, 0, len(dim))) # Q is a 4-th order tensor - Lagrange multiplier
    
    for idx in range(maxiter):
        rho = rho * (1 + incre) #Increase beta with given step
        
        # Update M variable
        for i in range(np.ndim(X_missing)):
            # M is a 4D array
            M[i] = Fold(
                update_Mi(
                    Unfold(X + (1 / rho) * Q[i], dim, i),
                    alpha[i], rho, p, theta
                ), dim, i
            )
        
        Xlast = X.copy()
        # Updata X variable
        X = np.sum(rho * M - Q, axis=0) / (rho * (X_missing.ndim))
        # Observed data
        X[Omega] = X_missing[Omega]
        
        # Update Q variable
        Q = Q + rho * (np.broadcast_to(X, np.insert(dim, 0, len(dim))) - M)
        
        tol = (np.linalg.norm(X - Xlast) / np.linalg.norm(Xlast)).item()
        # theta = theta * np.exp(-missing_rate)
        
        if verbose > 0:
            if (idx + 1) % verbose == 0:
                restore_y = X_true[pos_test] * vstd + vmean
                restore_yhat = X[pos_test] * vstd + vmean
                rmse = compute_rmse(restore_y, restore_yhat)
                mae = compute_mae(restore_y, restore_yhat)
                print(f"Iter {idx + 1:>3}, {rmse = :.4f}, {mae = :.4f}"
                        f", {tol = :.6f}, {rho = :.3f}")
            else:
                print(f"Iter {idx + 1}", end="\r")

        if tol < epsilon:
            print(f"Converged at iter {idx + 1}.")
            break

    return X


def lrtc_tspn(dense_tensor, sparse_tensor, 
              p=0.5, theta=0.1,
              alpha=np.ones(3),
              rho=1e-5, incre=0.05,
              maxiter=200,
              vmean=0, vstd=1,
              epsilon=1e-5, verbose=0, 
              pos_missing=None, pos_test=None, **kwargs):
    
    dim = np.array(sparse_tensor.shape)
    Omega = np.ones(dim)
    Omega[pos_missing] = 0
    Omega = Omega.astype(bool)

    X_true = (dense_tensor - vmean) / vstd
    X_missing = (sparse_tensor - vmean) / vstd
    alpha = alpha.reshape(-1, 1)
    alpha = alpha / np.sum(alpha)
    tensor_hat = TSpN_ADMM(
        X_true, X_missing, Omega,
        p=p, theta=theta,
        alpha=alpha, rho=rho, incre=incre,
        vmean=vmean, vstd=vstd,
        maxiter=maxiter, epsilon=epsilon,
        verbose=verbose,
        pos_missing=pos_missing,
        pos_test=pos_test,
        **kwargs
    )

    tensor_hat = tensor_hat * vstd + vmean
    return tensor_hat
