import torch
from torch.nn import functional as F


'''
# --------------------------------------------
# Tensor Fold and Unfold Operations
# --------------------------------------------
'''
def unfolding(tensor, mode):
    dimk = tensor.shape[mode]
    return torch.reshape(torch.moveaxis(tensor, mode, 0), (dimk, -1))

def folding(unfolded_matrix, mode, dim_k):
    return torch.moveaxis(torch.reshape(unfolded_matrix, (-1, *dim_k)), 0, mode)


'''
# --------------------------------------------
# Faster SVD on CPU
# --------------------------------------------
'''
def svd_values(mat):
    _, s, _ = svd_(mat)
    return s

def svd_(mat):
    # faster SVD
    [m, n] = mat.shape
    try:
        if 2 * m < n:
            u, s, _ = torch.linalg.svd(mat @ mat.T, full_matrices=False)
            s = torch.sqrt(s)
            tol = n * torch.finfo(float).eps
            idx = torch.sum(s > tol)
            return u[:, :idx], s[:idx],  torch.diag(1/s[:idx]) @ u[:, :idx].T @ mat
        elif m > 2 * n:
            v, s, u = svd_(mat.T)
            return u, s, v
    except: 
        pass
    u, s, v = torch.linalg.svd(mat, full_matrices=False)
    return u, s, v
    

'''
# --------------------------------------------
# Shrinkage Operations
# --------------------------------------------
'''
def shrinkage(vec, params, mode="firm"):
    if mode == "hard":
        # the shrinkage of the rank norm 
        return hard_shrinkage(vec, params)
    
    elif mode == "soft":
        # the shrinkage of the nuclear norm (HaLRTC or LRTC-TNN) 
        return soft_shrinkage(vec, params)
    
    elif mode == "firm":
        # the shrinkage of the MCP (LRTC-TMCP)
        return firm_shrinkage(vec, params[0], params[1])
    
    elif mode == "scad":
        # the shrinkage of the SCAD (LRTC-SCAD)
        return scad_shrinkage(vec, params[0], params[1], params[2])
    
    else:
        try:
            # the shrinkage of the Schatten p-norm (0 < p < 1) (LRTC-TSpN)
            return sp_shrinkage(vec, params, mode)
        except:
            raise ValueError("Only 'hard', 'soft', 'firm', 'scad' or 'GSP' (0 < p < 1) 4 kinds of shrinkage functions.")

def hard_shrinkage(vec, lam):
    ss = F.relu(vec - lam)
    try:
        ss[ss > 0] += lam[ss > 0]
    except:
        ss[ss > 0] += lam
    return ss

def soft_shrinkage(vec, lam):
    return F.relu(vec - lam)

def firm_shrinkage(vec, lam, gamma):
    # vec >= 0
    if gamma <= 1:
        return hard_shrinkage(vec, lam)
    
    else:
        ss = gamma / (gamma - 1) * F.relu(vec - lam)
        ss[vec > gamma * lam] = vec[vec > gamma * lam]
        return ss
    
def sp_shrinkage(x, w, p, iter=5):
    # generalized soft-thresholding algorithm
    # inner iteration is supposed to be 5
    if torch.sum(w) == 0:
        return x
    else:
        tau = (2 * w * (1 - p)) ** (1 / (2 - p)) + w * p * (2 * w * (1 - p)) ** ((p - 1) / (2 - p))
        ans = F.relu(x - tau)

        ins = torch.where(ans > 0)
        try:
            ans[ins] += tau[ins]
            weight = w[ins]
        except:
            ans[ins] += tau
            weight = w

        x, y = [ans[ins].clone() for _ in range(2)]
        
        for _ in range(iter):
            ans[ins] = y - weight * p * x ** (p - 1)
            x = ans[ins].clone()
        
        return ans
    
def _scad1(x, tau, gamma, lamb):
    '''
    SCAD proximal operator: for gamma > (1 + tau)
    '''
    s = F.relu(x - lamb * tau)
    ind = torch.where((x <= (gamma * lamb)) & (x > (1 + tau) * lamb))
    s[x > (gamma * lamb)] = x[x > (gamma * lamb)]
    try:
        s[ind] = (x[ind] * (gamma - 1) - lamb * gamma * tau) / (gamma - tau - 1)
    except:
        s[ind] = (x[ind] * (gamma - 1) - lamb * gamma * tau[ind]) / (gamma - tau - 1)

    return s

def _scad2(x, tau, gamma, lamb):
    '''
    SCAD proximal operator: for gamma <= (1 + tau)
    '''
    s = F.relu(x - lamb * tau)
    ind = torch.where(x > 0.5 * (tau + 1 + gamma) * lamb)
    s[ind] = x[ind]

    return s

def scad_shrinkage(x, tau, gamma, lamb):
    """
    SCAD shrinkage operator: \tau F(y, gamma, lamb) + \frac{1}{2} \| y - x \|_2^2
    x : input vector, by default larger than 0.
    tau : threshold
    lamb : tuning parameter
    gamma : tuning parameter
    """
    if gamma > (1 + tau):
        return _scad1(x, tau, gamma, lamb)
    else:
        return _scad2(x, tau, gamma, lamb)