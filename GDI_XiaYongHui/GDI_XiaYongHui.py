import math, time, argparse, random
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================

DEF_TXT_PATH    = ' '
OBS_RATIO       = 0.5        # 每个设备在其 split 的窗口中观测的比例（论文的测量率 p）
SPLIT_TRAIN     = 0.6
SPLIT_VAL       = 0.2
SPLIT_TEST      = 0.2
MASK_NONZERO    = False      # 仅在非零处评估

SEED            = 0
GPU             = 0          # -1=CPU
DET_BENCH       = False      # cudnn benchmark

LATDIM          = 10         # 嵌入维度 d
GNN_LAYERS      = 2          # 二部图消息传递层数
DROPOUT         = 0.0
SIGMOID_OUT     = True       # 归一化到[0,1]建议 True
ATTN_HEADS      = 2          # 时间精炼多头注意力

EPOCHS          = 1000
BATCH           = 2048       # 以三元组 (i,j,k) 为 batch
LR              = 5e-3
WD              = 1e-4
CLIP_NORM       = 5.0

PATIENCE        = 10
MIN_DELTA       = 1e-5
LOG_EVERY       = 10

# =====================

def set_seed(seed: int, deterministic_benchmark: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = deterministic_benchmark


def load_quads(txt_path: str) -> np.ndarray:

    data = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',') if ',' in line else line.split()
            assert len(parts) == 4, f'每行应有4列，得到{len(parts)}: {line}'
            i, j, k = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
            v = float(parts[3])
            data.append((i, j, k, v))
    return np.array(data, dtype=np.float64)


def get_sizes(quads: np.ndarray) -> Tuple[int, int, int]:
    I = int(np.max(quads[:, 0])) + 1
    J = int(np.max(quads[:, 1])) + 1
    K = int(np.max(quads[:, 2])) + 1
    return I, J, K


def minmax_fit(values: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmax, vmin):
        vmax = vmin + 1.0
    return vmin, vmax


def minmax_apply(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (values - vmin) / (vmax - vmin)


def minmax_inv(values01: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return values01 * (vmax - vmin) + vmin


def time_contiguous_splits(K: int, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1.0) < 1e-8
    n_train = int(K * ratios[0]); n_val = int(K * ratios[1])
    train_k = np.arange(0, n_train, dtype=np.int64)
    val_k   = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_k  = np.arange(n_train + n_val, K, dtype=np.int64)
    return train_k, val_k, test_k


def sample_windows_per_device(I: int, K_idx: np.ndarray, p: float, seed: int) -> List[set]:

    rng = np.random.default_rng(seed)
    obs = [None] * I
    for i in range(I):
        kk = K_idx
        take = max(1, int(len(kk) * p))
        obs_k = rng.choice(kk, size=take, replace=False)
        obs[i] = set(int(x) for x in obs_k)
    return obs


def build_edge_features_from_quads(quads: np.ndarray, I: int, J: int, K: int,
                                   observed_windows: List[set], k_allow: np.ndarray):

    X_edge: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    mask_allow = np.isin(quads[:, 2].astype(np.int64), k_allow)
    q = quads[mask_allow]

    for i_f, j_f, k_f, v_f in q:
        i = int(i_f); j = int(j_f); k = int(k_f); v = float(v_f)
        if k not in observed_windows[i]:
            continue
        key = (i, k)
        if key not in X_edge:
            X_edge[key] = {
                'x': np.zeros(J, dtype=np.float32),
                'mask': np.zeros(J, dtype=bool)
            }
        X_edge[key]['x'][j] = v
        X_edge[key]['mask'][j] = True

    edges_imap = [[] for _ in range(I)]
    edges_kmap = [[] for _ in range(K)]
    for (i, k) in X_edge.keys():
        edges_imap[i].append(k)
        edges_kmap[k].append(i)
    return edges_imap, edges_kmap, X_edge


def build_unobserved_mask(arr: np.ndarray, observed_windows: List[set]) -> np.ndarray:

    mask = np.zeros(arr.shape[0], dtype=bool)
    for t in range(arr.shape[0]):
        i = int(arr[t, 0]); k = int(arr[t, 2])
        mask[t] = (k not in observed_windows[i])
    return mask


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))

# =====================

class TimeRefine(nn.Module):

    def __init__(self, K: int, dim: int, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, W: torch.Tensor) -> torch.Tensor:  # W: [K,d]
        X = W + self.pos
        attn, _ = self.mha(X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0))
        H1 = self.ln1(X + attn.squeeze(0))
        H2 = self.ffn(H1)
        return self.ln2(H1 + H2)


class GTCInductive(nn.Module):

    def __init__(self, I: int, J: int, K: int, dim: int, layers: int, dropout: float,
                 sigmoid_out: bool, attn_heads: int = 2):
        super().__init__()
        self.I, self.J, self.K, self.dim = I, J, K, dim
        self.sigmoid_out = sigmoid_out


        self.u0 = nn.Embedding(I, dim)
        nn.init.xavier_uniform_(self.u0.weight)
        self.w0 = nn.Parameter(torch.randn(K, dim) * 0.01)


        self.edge_proj = nn.Linear(J, dim)


        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'u_up': nn.Linear(dim * 2, dim),
                'w_up': nn.Linear(dim * 2, dim),
                'drop': nn.Dropout(dropout)
            }) for _ in range(layers)
        ])


        self.refine = TimeRefine(K, dim, heads=attn_heads, dropout=dropout)


        self.V = nn.Embedding(J, dim)
        nn.init.xavier_uniform_(self.V.weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def message_passing(self, edges_imap: List[List[int]], edges_kmap: List[List[int]],
                        X_edge: Dict[Tuple[int, int], Dict[str, np.ndarray]], device: torch.device):

        U = self.u0.weight.to(device)
        W = self.w0.to(device)

        for layer in self.layers:
            U_msg = torch.zeros_like(U)
            W_msg = torch.zeros_like(W)
            cnt_u = torch.zeros((self.I, 1), device=device)
            cnt_w = torch.zeros((self.K, 1), device=device)

            for (i, k), feat in X_edge.items():
                x = torch.from_numpy(feat['x']).to(device)
                e = self.edge_proj(x)
                U_msg[i] += (W[k] + e)
                W_msg[k] += (U[i] + e)
                cnt_u[i] += 1.0; cnt_w[k] += 1.0

            cnt_u = cnt_u.clamp_min(1.0); cnt_w = cnt_w.clamp_min(1.0)
            U_agg = U_msg / cnt_u
            W_agg = W_msg / cnt_w

            U = F.relu(layer['u_up'](torch.cat([U, U_agg], dim=1)))
            W = F.relu(layer['w_up'](torch.cat([W, W_agg], dim=1)))
            U = layer['drop'](U); W = layer['drop'](W)


        W = self.refine(W)
        return U, W

    def decode_triples(self, U: torch.Tensor, W: torch.Tensor, ijk_idx: torch.Tensor) -> torch.Tensor:
        i = ijk_idx[:, 0].long(); j = ijk_idx[:, 1].long(); k = ijk_idx[:, 2].long()
        s = U[i] * W[k]                       # [B,d]
        Vj = self.V(j)                        # [B,d]
        out = (s * Vj).sum(dim=1) + self.bias # [B]
        return torch.sigmoid(out) if self.sigmoid_out else out

    def forward(self, edges_imap, edges_kmap, X_edge, idx_ijk: torch.Tensor, device: torch.device):
        U, W = self.message_passing(edges_imap, edges_kmap, X_edge, device)
        return self.decode_triples(U, W, idx_ijk.to(device))

# =====================

def to_tensors(arr: np.ndarray):
    idx = torch.from_numpy(arr[:, :3].astype(np.int64))
    y = torch.from_numpy(arr[:, 3].astype(np.float32))
    return idx, y


def iter_minibatch(idx_tensor: torch.Tensor, y_tensor: torch.Tensor, batch_size: int):
    n = idx_tensor.shape[0]
    order = torch.randperm(n)
    for s in range(0, n, batch_size):
        sel = order[s:s + batch_size]
        yield idx_tensor[sel], y_tensor[sel]


def eval_on_split(model: GTCInductive, edges_imap, edges_kmap, X_edge,
                  arr_split: np.ndarray, observed_windows: List[set], name: str,
                  device: torch.device, vmin: float, vmax: float,
                  mask_nonzero_only: bool = False):

    mask_unobs = build_unobserved_mask(arr_split, observed_windows)
    if not np.any(mask_unobs):
        print(f'[{name}] No unobserved entries to evaluate.')
        return (float('nan'),) * 4

    idx_eval, y_eval = to_tensors(arr_split[mask_unobs])
    model.eval()
    with torch.no_grad():
        pred = model(edges_imap, edges_kmap, X_edge, idx_eval, device).cpu().numpy()
        y = y_eval.cpu().numpy()

        if mask_nonzero_only:
            m = (y != 0.0)
            if not np.any(m):
                print(f'[{name}] No non-zero entries under mask.')
                return (float('nan'),) * 4
            pred, y = pred[m], y[m]

        rmse_n = rmse(pred, y)
        mae_n = mae(pred, y)
        pred_o = minmax_inv(pred, vmin, vmax)
        y_o = minmax_inv(y, vmin, vmax)
        rmse_o = rmse(pred_o, y_o)
        mae_o = mae(pred_o, y_o)

    print(f'[{name}] RMSE(norm)={rmse_n:.6f} MAE(norm)={mae_n:.6f} | RMSE(orig)={rmse_o:.6f} MAE(orig)={mae_o:.6f}')
    return rmse_n, mae_n, rmse_o, mae_o

# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, default=DEF_TXT_PATH, help='txt 路径：i,j,k,value')
    parser.add_argument('--obs_ratio', type=float, default=OBS_RATIO, help='每设备窗口观测比例 p (0~1)')
    parser.add_argument('--splits', type=float, nargs=3, default=[SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST], help='时间连续切分比例')
    parser.add_argument('--mask_nonzero', action='store_true', default=MASK_NONZERO, help='只在非零处评估')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--latdim', type=int, default=LATDIM)
    parser.add_argument('--layers', type=int, default=GNN_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--heads', type=int, default=ATTN_HEADS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--wd', type=float, default=WD)
    parser.add_argument('--clip', type=float, default=CLIP_NORM)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--mindelta', type=float, default=MIN_DELTA)
    parser.add_argument('--log_every', type=int, default=LOG_EVERY)
    parser.add_argument('--no_sigmoid', action='store_true', help='关闭 Sigmoid 输出（默认开）')
    args = parser.parse_args()

    set_seed(args.seed, deterministic_benchmark=DET_BENCH)

    use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print(f'[Info] Device = {device}')


    t0 = time.time()
    quads = load_quads(args.txt)
    I, J, K = get_sizes(quads)
    print(f'[Data] Loaded {quads.shape[0]} samples; sizes I={I}, J={J}, K={K}')

    train_k, val_k, test_k = time_contiguous_splits(K, ratios=tuple(args.splits))

    mask_train = np.isin(quads[:, 2].astype(np.int64), train_k)
    mask_val   = np.isin(quads[:, 2].astype(np.int64), val_k)
    mask_test  = np.isin(quads[:, 2].astype(np.int64), test_k)

    train = quads[mask_train].copy()
    val   = quads[mask_val].copy()
    test  = quads[mask_test].copy()

    vmin, vmax = minmax_fit(train[:, 3])
    train[:, 3] = minmax_apply(train[:, 3], vmin, vmax)
    val[:, 3]   = minmax_apply(val[:, 3],   vmin, vmax)
    test[:, 3]  = minmax_apply(test[:, 3],  vmin, vmax)
    print(f'[Norm] Min={vmin:.6g}, Max={vmax:.6g} (fitted on Train)')

    obs_train = sample_windows_per_device(I, train_k, p=args.obs_ratio, seed=args.seed)
    obs_val   = sample_windows_per_device(I, val_k,   p=args.obs_ratio, seed=args.seed + 1)
    obs_test  = sample_windows_per_device(I, test_k,  p=args.obs_ratio, seed=args.seed + 2)

    edges_i_train, edges_k_train, X_edge_train = build_edge_features_from_quads(train, I, J, K, obs_train, train_k)
    edges_i_val,   edges_k_val,   X_edge_val   = build_edge_features_from_quads(val,   I, J, K, obs_val,   val_k)
    edges_i_test,  edges_k_test,  X_edge_test  = build_edge_features_from_quads(test,  I, J, K, obs_test,  test_k)

    print(f'[Graph] Train edges={len(X_edge_train)} | Val edges={len(X_edge_val)} | Test edges={len(X_edge_test)}')

    mask_obs_train = np.array([ (int(train[t,2]) in obs_train[int(train[t,0])]) for t in range(train.shape[0]) ], dtype=bool)
    train_idx_t, train_y_t = to_tensors(train[mask_obs_train])

    model = GTCInductive(I, J, K, dim=args.latdim, layers=args.layers, dropout=args.dropout,
                         sigmoid_out=(not args.no_sigmoid), attn_heads=args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf'); best_state = None; patience_left = args.patience
    print('[Train] Start training...')
    for ep in range(1, args.epochs + 1):
        model.train(); ep_loss = 0.0; nb = 0
        for xb, yb in iter_minibatch(train_idx_t, train_y_t, args.batch):
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(edges_i_train, edges_k_train, X_edge_train, xb, device)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            if args.clip and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ep_loss += float(loss.detach().cpu()); nb += 1
        ep_loss /= max(1, nb)

        rmse_v, *_ = eval_on_split(model, edges_i_val, edges_k_val, X_edge_val, val, obs_val, 'Val', device, vmin, vmax, mask_nonzero_only=args.mask_nonzero)

        if (ep % args.log_every == 0) or ep == 1:
            print(f'Epoch {ep:04d} | train_mse={ep_loss:.6f} | val_RMSE(norm)={rmse_v:.6f}')

        improved = (best_val - rmse_v) > args.mindelta
        if improved:
            best_val = rmse_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[EarlyStop] No improvement for {args.patience} checks. Stop at epoch {ep}.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print('[Eval] Using best checkpoint on Val')
    eval_on_split(model, edges_i_val,  edges_k_val,  X_edge_val,  val,  obs_val,  'Val',  device, vmin, vmax, mask_nonzero_only=args.mask_nonzero)
    eval_on_split(model, edges_i_test, edges_k_test, X_edge_test, test, obs_test, 'Test', device, vmin, vmax, mask_nonzero_only=args.mask_nonzero)


if __name__ == '__main__':
    main()
