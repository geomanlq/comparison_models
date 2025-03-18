import torch
from einops import rearrange, repeat


def avg_pooling(mat, n):
    if mat.ndim == 2:
        tensor = rearrange(mat, "s (d n) -> s d n", n=n)
        return tensor.mean(dim=2)
    elif mat.ndim == 3:
        tensor = rearrange(mat, "k s (d n) -> k s d n", n=n)
        return tensor.mean(dim=3)


def repeating(mat, n):
    longmat = repeat(mat, "s d -> s (d n)", n=n)
    return longmat


def solve_sylvester(A, B, C):
    """
    Solve the Sylvester equation AX + XB = C
    """
    m = B.shape[-1]
    n = A.shape[-1]
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = U[..., :n, :n] @ Y[..., :n, :m] @ torch.linalg.inv(V)[..., :m, :m]
    return X.real
