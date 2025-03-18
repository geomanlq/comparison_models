"""
Updates:
- torch script: compute trend sum
- update u: precompute G @ P^T
"""
import torch
import tensorly as tl
from math import prod
from numpy import ndarray
from tensorly.decomposition import Tucker
from time import time
from scipy.linalg import solve_sylvester


tl.set_backend("pytorch")


class TuckerADMM:
    """Tucker decomposition with ADMM.

    Parameters
    ----------
    dense_tensor : ndarray
        The dense tensor to be decomposed.
    cpt_segs : list of tuples
        The list of tuples of the form (sensor, start, end) that
        indicate the start and end of the change point.
    rank : list of int
        The list of ranks of the core tensor.
    mu : float
        The coefficient of the core tensor.
    lam : float
        The coefficient of the factor rank.
    xi : float
        The coefficient of the total variation.
    rho : float
        The penalty parameter of the augmented Lagrangian multiplier.
    gamma : float, optional
        The parameter of the rank surrogate function, by default 1.
    eta : float, optional
        The parameter of the Bregman distance, by default 1e-4.
    tau : float, optional
        The parameter of the Lagrangian multiplier update, by default 1.
    vmax : float, optional
        The maximum value of the dense tensor, by default None.
    vmean : float, optional
        The mean value of the dense tensor, by default None.
    max_iter : int, optional
        The maximum number of iterations, by default 200.
    warm_iter : int, optional
        The number of warm-up iterations, by default 20.
    epsilon : float, optional
        The tolerance of the stopping criterion, by default 1e-4.
    incre : float, optional
        The increment of the penalty parameter, by default 0.1.
    verbose : int, optional
        The verbosity level, by default 0.
    random_state : int, optional
        The random seed, by default 42.
    device : str, optional
        The device to run the model, by default "cpu".

    """

    def __init__(
        self,
        dense_tensor,
        cpt_segs,
        rank,
        mu,
        lam,
        xi,
        rho,
        gamma=1,
        eta=0.3,
        tau=1,
        vmax=None,
        vmean=None,
        max_iter=200,
        warm_iter=20,
        epsilon=1e-4,
        incre=0.1,
        verbose=0,
        random_state=42,
        device="cpu",
        trunc=0,
        logger=None,
    ):
        self.cpt_segs = cpt_segs
        self.mu = mu  # coefficient of core tensor
        self.lam = lam  # coefficient of factor rank
        self.xi = xi  # coefficient of total variation
        self.rho = rho  # penalty parameter of ALM
        self.gamma = gamma  # parameter of rank surrogate function
        self.eta = eta  # parameter of Bregman distance
        self.tau = tau  # parameter of Lagrangian multiplier update
        self.vmax = vmax
        self.vmean = vmean
        self.max_iter = max_iter
        self.warm_iter = warm_iter
        self.epsilon = epsilon
        self.incre = incre
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.trunc = trunc
        self.logger = logger

        self.dense_tensor = tl.tensor(dense_tensor).to(device)
        self.dim = dense_tensor.shape
        self.k = len(self.dim)
        self.k_dim = [self.k, *self.dim]
        self.n_sensor = self.dim[0]
        self.n_time = prod(self.dim[1:])
        self.n_day = prod(self.dim[1:-1])
        self.n_timeofday = self.dim[-1]
        assert len(rank) >= self.k
        self.rank = rank[: self.k]

        if self.vmax is None:
            self.vmax = 1.0
        if self.vmean is None:
            self.vmean = 0.0
        self.dense_tensor /= self.vmax

    def fit(
        self,
        sparse_tensor,
        pos_missing,
        pos_test,
        init_tensor=None,
        return_components=False,
    ):
        start_time = time()
        sparse_tensor = tl.tensor(sparse_tensor).to(self.device)
        sparse_tensor /= self.vmax
        yhat = self._admm(sparse_tensor, pos_missing, pos_test, init_tensor)
        yhat *= self.vmax
        yhat = yhat.reshape(self.n_sensor, self.n_day, self.n_timeofday)
        end_time = time()
        self.logger.log({"overall_time": end_time - start_time})
        
        if return_components:
            return yhat, self.components
        else:
            return yhat

    def _compute_rmse(self, yhat, pos_test):
        a = self.dense_tensor[pos_test]
        ahat = yhat[pos_test]
        rmse = tl.mean(((a - ahat) * self.vmax) ** 2) ** 0.5
        return rmse.item()

    def _compute_mae(self, yhat, pos_test):
        a = self.dense_tensor[pos_test]
        ahat = yhat[pos_test]
        mae = tl.mean(tl.abs(a - ahat) * self.vmax)
        return mae.item()

    @staticmethod
    @torch.jit.script
    def jit_compute_trend_sum(trend_coef, cpt_mask_mat):
        trend_sum = torch.einsum("ijk,i->jk", cpt_mask_mat, trend_coef)
        return trend_sum

    def _init_x(self, init_tensor, sparse_tensor, pos_missing):
        if isinstance(init_tensor, ndarray):
            init_tensor = tl.tensor(init_tensor).to(self.device)
            init_tensor = tl.reshape(init_tensor, self.dim)
            init_tensor /= self.vmax
            X = sparse_tensor.clone()
            X[pos_missing] = init_tensor[pos_missing]
        elif init_tensor == "random":
            X = sparse_tensor.clone()
            X[pos_missing] = torch.randn_like(X)[pos_missing]
        elif init_tensor == "hosvd":
            X = sparse_tensor.clone()
            X_mask = torch.ones(self.dim).to(self.device)
            X_mask[pos_missing] = 0
            tucker = Tucker(
                rank=self.rank, n_iter_max=100, random_state=42, mask=X_mask, init="svd"
            )
            core_, factors_ = tucker.fit_transform(X)
            X[pos_missing] = tl.tucker_to_tensor((core_, factors_))[pos_missing]
        elif init_tensor == "zero":
            X = sparse_tensor.clone()
            X[pos_missing] = 0.0
        else:
            raise ValueError("init_tensor must be 'random', 'zero' or an array.")
        return X

    def _init_tucker(self, X):
        tucker = Tucker(rank=self.rank, init="random", random_state=self.random_state)
        core, factors = tucker.fit_transform(X)
        return core, factors

    def _init_trend(self):
        Psi = tl.zeros(len(self.cpt_segs), self.n_sensor, self.n_time).to(self.device)
        for i, seg in enumerate(self.cpt_segs):
            Psi[i, seg[0], seg[1] : seg[2]] = 1.0
        c = tl.ones(len(self.cpt_segs)).to(self.device) * self.vmean / self.vmax
        return c, Psi

    def _get_d(self):
        D = dict()
        for i in range(1, self.k):
            Di = torch.eye(self.dim[i]).to(self.device)
            Di -= torch.roll(Di, shifts=1, dims=0)
            Di[0] = 0.0
            D[i] = Di
        return D

    def _get_i(self, use_rank=False):
        n = self.rank if use_rank else self.dim
        I = {i: torch.eye(n[i]).to(self.device) for i in range(self.k)}
        return I

    def _update_u(self, X, T, G, U_, E, M, V_, Gamma_, Irank_, Idim_, D_, rho, i):
        eta = self.eta
        xi = self.xi
        Z = X - T - E + M
        P = tl.tenalg.kronecker([ui.contiguous() for ui in U_], skip_matrix=i)
        Q = V_[i] - Gamma_[i]
        Z_unfold = tl.unfold(Z, i)
        G_unfold = tl.unfold(G, i)

        GP_T = G_unfold @ P.T
        if i == 0:
            left = rho * Z_unfold @ GP_T.T - rho * Q - eta * U_[i]
            right = rho * GP_T @ GP_T.T - (rho + eta) * Irank_[i]
            UU = left @ torch.linalg.pinv(right)
            return UU
        else:
            A = rho * Idim_[i] + xi * D_[i].T @ D_[i]
            B = -(eta * Irank_[i] + rho * GP_T @ GP_T.T)
            R = rho * Q + eta * U_[i] - rho * Z_unfold @ GP_T.T
            UU = solve_sylvester(
                A.detach().cpu().numpy(),
                B.detach().cpu().numpy(),
                R.detach().cpu().numpy(),
            )
            UU = torch.tensor(UU).float().to(self.device)
            return UU

    def _update_v(self, U_, V_, Gamma_, rho, i):
        eta = self.eta
        P = (rho * (U_[i] + Gamma_[i]) + eta * V_[i]) / (rho + eta)
        A, s, Bh = torch.linalg.svd(P, full_matrices=False)
        w = (1.0 + self.gamma) * self.gamma / (self.gamma + s) ** 2
        s_new = tl.tenalg.proximal.soft_thresholding(s, self.lam / (rho + eta) * w)
        V_new = s_new * A @ Bh
        return V_new

    def _update_g(self, G, U_, X, T, E, M, rho, i):
        mu, eta = self.mu, self.eta
        Z = X - T - E + M
        P = tl.tenalg.kronecker(U_, skip_matrix=i)
        Z_unfold = tl.unfold(Z, i)
        G_unfold = tl.unfold(G, i)

        UT_U = U_[i].T @ U_[i]
        PT_P = P.T @ P

        U_i_spec_norm = max(torch.linalg.svdvals(UT_U))
        P_spec_norm = max(torch.linalg.svdvals(PT_P))

        one_sigma = rho / mu * U_i_spec_norm * P_spec_norm + eta / mu
        
        def grad_phi(x):
            left = UT_U @ x @ PT_P - U_[i].T @ Z_unfold @ P
            return rho * left / mu

        def psi(x):
            return x - grad_phi(x) / one_sigma

        H = psi(G_unfold)
        G_new = tl.tenalg.proximal.soft_thresholding(H, 1 / one_sigma)
        return tl.fold(G_new, i, self.rank)

    def _update_e(self, E, X, T, S, M, rho):
        eta = self.eta
        B = (rho * (X - T - S + M) + eta * E) / (rho + eta)
        E_new = tl.tenalg.proximal.soft_thresholding(B, 1 / (rho + eta))
        return E_new

    def _update_c(self, c, Psi, X, S, E, M, rho):
        eta = self.eta
        Z = X - S - E + M
        Z_unfold = tl.unfold(Z, 0)
        Psi_bool = Psi.bool()
        Z_mean = torch.stack([Z_unfold[pi].mean() for pi in Psi_bool])
        Z_mean = torch.nan_to_num(Z_mean)
        cc = [(rho * zi + eta * ci) / (rho + eta) for zi, ci in zip(Z_mean, c)]
        return tl.tensor(cc).to(self.device)

    def _update_x(self, T, S, E, M, X, rho):
        Z = T + S + E - M
        eta = self.eta
        return (rho * Z + eta * X) / (rho + eta)

    def _admm(self, sparse_tensor, pos_missing, pos_test, init_tensor=None):
        # initialize penalty parameters
        rho0 = self.rho
        k = self.k
        # initialize tucker decomposition
        X = self._init_x(init_tensor, sparse_tensor, pos_missing)
        del init_tensor, sparse_tensor
        torch.cuda.empty_cache()
        # initialize trend
        c, Psi = self._init_trend()
        T = self.jit_compute_trend_sum(c, Psi).reshape(self.dim)
        # initialize tucker
        G, U_ = self._init_tucker(X - T)
        V_ = [ui.clone() for ui in U_]
        # initialize seasonality
        S = tl.tucker_to_tensor((G, U_))
        # initialize error
        E = X - T - S
        E[pos_missing] = 0.0
        # initialize lagrange multipliers
        M = tl.zeros(self.dim).to(self.device)
        # M = X - T - S - E
        Gamma_ = [torch.zeros_like(vi).to(self.device) for vi in V_]
        # initialize constants
        D_ = self._get_d()
        Idim_ = self._get_i(use_rank=False)
        Irank_ = self._get_i(use_rank=True)

        print("Initialization finished.")

        rmse = self._compute_rmse(X, pos_test)
        mae = self._compute_mae(X, pos_test)
        obj = self._compute_obj(E, G, V_, U_, D_, X, T, S, M, Gamma_, rho0)
        print(f"Init, {rmse = :.4f}, {mae = :.4f}, {obj = :.4f}")

        last_X = X.clone()
        for idx in range(self.max_iter):
            # update penalty parameters
            rho = min(rho0 * (1 + self.incre) ** idx, 1e7)
            self.eta = rho
            # update factor matrices
            for i in range(k):
                U_[i] = self._update_u(
                    X, T, G, U_, E, M, V_, Gamma_, Irank_, Idim_, D_, rho, i
                )
            for i in range(k):
                V_[i] = self._update_v(U_, V_, Gamma_, rho, i)
            # update core tensor
            G = self._update_g(G, U_, X, T, E, M, rho, i=0)
            # reconstruct Tucker tensor
            S = tl.tucker_to_tensor((G, U_))
            # update error
            E = self._update_e(E, X, T, S, M, rho)
            if idx % 20 == 0:
                # lazy update trend
                c = self._update_c(c, Psi, X, S, E, M, rho)
                T = self.jit_compute_trend_sum(c, Psi).reshape(self.dim)
            # combine TSE components
            X[pos_missing] = self._update_x(T, S, E, M, X, rho)[pos_missing]
            # update Lagrangian multipliers
            M = M + self.tau * (X - T - S - E)
            Gamma_ = [Gamma_[i] + self.tau * (U_[i] - V_[i]) for i in range(k)]

            # check convergence
            tol = (tl.norm(X - last_X) / tl.norm(last_X)).item()

            self.logger.log({'train/tolerance': tol})

            rmse = self._compute_rmse(X, pos_test)
            mae = self._compute_mae(X, pos_test)
            self.logger.log({
                'train/rmse': rmse,
                'train/mae': mae,
            })
            if mae > 1e3:
                raise ValueError("Exploded.")
                
            
            if self.verbose > 0:
                if (idx + 1) % self.verbose == 0:
                    obj = self._compute_obj(E, G, V_, U_, D_, X, T, S, M, Gamma_, rho)
                    print(
                        f"Iter {idx + 1:>3}, {rmse = :.4f}"
                        f", {mae = :.4f}, {obj = :.4f}"
                        f", {tol = :.6f}, {rho = :.3f}"
                    )
                    self.logger.log({
                        'train/obj': obj
                    })

                else:
                    print(f"Iter {idx + 1}", end="\r")

            if idx >= self.warm_iter and tol < self.epsilon:
                print(f"Converged at iter {idx + 1}.")
                break
            last_X = X.clone()

        self.logger.log({'total_iter': idx})
        
        self.components = {
            "T": self.jit_compute_trend_sum(c, Psi).detach().cpu().reshape(self.dim).numpy()
            * self.vmax,
            "S": S.detach().cpu().numpy() * self.vmax,
            "E": E.detach().cpu().numpy() * self.vmax,
            "G": G.detach().cpu().numpy() * self.vmax,
            "U0": U_[0].detach().cpu().numpy(),
            "U1": U_[1].detach().cpu().numpy(),
            "U2": U_[2].detach().cpu().numpy(),
            "U3": U_[3].detach().cpu().numpy(),
        }
        yhat = X.detach().cpu().numpy()
        return yhat

    def _compute_obj(self, E, G, V_, U_, D_, X, T, S, M, Gamma_, rho, aug_lag=False):
        obj = 0.0
        obj += tl.norm(E, 1)
        obj += self.mu * tl.norm(G, 1)
        for i in range(self.k):
            sig = torch.linalg.svdvals(V_[i])
            sig = (1 + self.gamma) * sig / (self.gamma + sig)
            obj += self.lam * tl.sum(sig)
        for i in range(2, self.k):
            du = D_[i] @ U_[i]
            obj += self.xi / 2 * tl.norm(du, 2) ** 2
        if aug_lag:
            obj += rho / 2 * tl.norm(X - T - S - E + M, 2) ** 2
            for i in range(self.k):
                obj += rho / 2 * tl.norm(U_[i] - V_[i] + Gamma_[i], 2) ** 2
        return obj.item()
