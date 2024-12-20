import os
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as spr
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.stats import gamma, norm, ks_2samp
from scipy.special import digamma
from scipy.optimize import root_scalar

os.environ['SCIPY_ARRAY_API'] = '1'
from inverse_covariance import QuicGraphLasso
from joblib import Parallel, delayed
from tqdm import tqdm

def dmix(x, pars):
    return pars[0] * gamma.pdf(x, a=pars[1], scale=1/pars[2]) + (1 - pars[0]) * norm.pdf(x, loc=pars[3], scale=pars[4])

def calculate_weight(x, paramt):
    if paramt[0] == 0:
        return np.column_stack((np.zeros(len(x)), np.ones(len(x))))
    
    pz1 = paramt[0] * gamma.pdf(x, a=paramt[1], scale=1/paramt[2])
    pz2 = (1 - paramt[0]) * norm.pdf(x, loc=paramt[3], scale=paramt[4])
    pz = pz1 / (pz1 + pz2)
    pz[pz1 == 0] = 0
    return np.column_stack((pz, 1 - pz))

def fn(alpha, target):
    return np.log(alpha) - digamma(alpha) - target

def update_gmm_pars(x, wt):
    tp_s = np.sum(wt)
    tp_t = np.sum(wt * x)
    tp_u = np.sum(wt * np.log(x))
    tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)

    if tp_v <= 0:
        alpha = 20
    else:
        alpha0 = (3 - tp_v + np.sqrt((tp_v - 3)**2 + 24 * tp_v)) / (12 * tp_v)
        if alpha0 >= 20:
            alpha = 20
        else:
            res = root_scalar(fn, args=(tp_v,), bracket=[0.9 * alpha0, 1.1 * alpha0], method='bisect')
            alpha = res.root if res.converged else 20

    beta = tp_s / tp_t * alpha
    return np.array([alpha, beta])

def get_mix(xdata, point):
    inits = np.zeros(5)
    inits[0] = np.sum(xdata == point) / len(xdata)
    if inits[0] == 0:
        inits[0] = 0.01
    inits[1:3] = [0.5, 1]

    xdata_rm = xdata[xdata > point]
    inits[3:5] = [np.mean(xdata_rm), np.std(xdata_rm)]
    if np.isnan(inits[4]):
        inits[4] = 0

    paramt = inits
    eps = 10
    iter_count = 0
    loglik_old = 0

    while eps > 0.5:
        wt = calculate_weight(xdata, paramt)
        paramt[0] = np.sum(wt[:, 0]) / len(wt)
        paramt[3] = np.sum(wt[:, 1] * xdata) / np.sum(wt[:, 1])
        paramt[4] = np.sqrt(np.sum(wt[:, 1] * (xdata - paramt[3])**2) / np.sum(wt[:, 1]))
        paramt[1:3] = update_gmm_pars(x=xdata, wt=wt[:, 0])

        loglik = np.sum(np.log10(dmix(xdata, paramt)))
        eps = (loglik - loglik_old)**2
        loglik_old = loglik
        iter_count += 1

        if iter_count > 100:
            break

    return paramt

def process_row(ii, count, null_genes, point):
    if ii in null_genes:
        return np.full(5, np.nan)

    xdata = count[ii, :]
    try:
        paramt = get_mix(xdata, point)
    except Exception:
        return np.full(5, np.nan)

    l1a = gamma.pdf(count[ii, :], a=paramt[1], scale=1/paramt[2])
    l1b = norm.pdf(count[ii, :], loc=paramt[3], scale=paramt[4])
    l1 = np.sum(np.log(paramt[0] * l1a + (1 - paramt[0]) * l1b))
    mu = np.mean(count[ii, :])
    sd = np.std(count[ii, :])
    l2 = np.sum(np.log(norm.pdf(count[ii, :], loc=mu, scale=sd)))
    pval = 1 - np.exp(-2 * (l2 - l1))  # Approximation for LRT

    if pval >= 0.05 / count.shape[0]:
        paramt = [0, 1, 1, mu, sd]

    return paramt

def get_mix_parameters(count, point=np.log10(1.01), ncores=8):
    count = np.array(count)
    null_genes = np.where(np.abs(np.sum(count, axis=1) - point * count.shape[1]) < 1e-10)[0]

    # Use multiprocessing to parallelize the processing of rows
    # with Pool(processes=ncores) as pool:
    #     parslist = pool.starmap(process_row, [(i, count, null_genes, point) for i in range(count.shape[0])])
    
    parslist = Parallel(n_jobs=-1 if ncores is None else ncores)(delayed(process_row)(i, count, null_genes, point) for i in range(count.shape[0]))

    parslist = np.vstack(parslist)
    return parslist

def calc_droprate(i, I, mat, pa):
    if np.isnan(pa[i, 0]):
        return np.zeros(I)  # Match the correct axis for output
    wt = calculate_weight(mat[:, i], pa[i, :])
    return wt[:, 0]

def if_dropout_scimpute(mat, ncores, dthre=0.9):
    mat[mat <= np.log10(1.01)] = np.log10(1.01)
    pa = get_mix_parameters(mat.T, ncores=ncores)  # pa is calculated for the transposed matrix
    I, J = mat.shape

    # with Pool(processes=ncores) as pool:
    #     droprate = pool.starmap(calc_droprate, [(j, I, mat, pa) for j in range(J)])
    
    droprate = Parallel(n_jobs=-1 if ncores is None else ncores)(delayed(calc_droprate)(j, I, mat, pa) for j in range(J))

    droprate = np.array(droprate).T  # Transpose to align with the original matrix shape
    dropind = (droprate > dthre).astype(int)
    return dropind

def est_pearson_scimpute(count, cor_p, thre_no=20, dthre=0.9, ncores=8):
    count_dr = count.copy()
    x = np.log10(10**count - 1 + 1.01)
    dropind = if_dropout_scimpute(x, ncores=ncores, dthre=dthre)
    count_dr[dropind == 1] = np.nan

    cor_complete = np.corrcoef(count_dr, rowvar=False)
    cor_complete = np.nan_to_num(cor_complete, nan=cor_p)

    count_dr = (~np.isnan(count_dr)).astype(int)
    nobs = np.dot(count_dr.T, count_dr)
    cor_complete[nobs < thre_no] = cor_p[nobs < thre_no]
    return cor_complete

def easy_psd(sigma, method="perturb"):
    """
    Converts a covariance matrix to a positive semi-definite (PSD) matrix.

    Parameters:
    - sigma: The input covariance matrix (must be square and symmetric).
    - method: The method to use for ensuring PSD. Options are:
        - "perturb": Add a constant to the diagonal based on the smallest eigenvalue.
        - "npd": Reconstruct the matrix using only non-negative eigenvalues.

    Returns:
    - sigma_psd: The adjusted positive semi-definite matrix.
    """
    if method == "perturb":
        p = sigma.shape[0]
        eigvals = np.linalg.eigvalsh(sigma)  # Compute eigenvalues for symmetric matrix
        const = max(0, -np.min(eigvals))    # Calculate constant to make eigenvalues non-negative
        sigma_psd = sigma + np.eye(p) * const

    elif method == "npd":
        eigvals, eigvecs = eigh(sigma)  # Compute eigenvalues and eigenvectors
        d = np.maximum(eigvals, 0)               # Replace negative eigenvalues with 0
        sigma_psd = eigvecs @ np.diag(d) @ eigvecs.T  # Reconstruct matrix

    else:
        raise ValueError("Invalid method. Choose 'perturb' or 'npd'.")

    return sigma_psd

def QUIC(cov, rho, weights):
    return {"X": QuicGraphLasso(rho*weights, max_iter=1500).fit(cov).precision_}

def process_lambda(lda, covobs, weights):
        print(f"Processing lambda = {lda}\n")
        res = QUIC(covobs, rho=lda, weights=weights)
        X = res["X"]
        np.fill_diagonal(X, np.diag(X))  # Ensure diagonal consistency
        return X

def get_res_wGL_quic(covobs, weights, genes, nobs, l1, ncores):
    
    # Calculate the weighted covariance matrix
    theta_list = Parallel(n_jobs=-1 if ncores is None else ncores)(delayed(process_lambda)(lda, covobs) for lda in l1)

    # Create results for each lambda
    results = []
    for i, Sigma in enumerate(theta_list):
        adj = (Sigma != 0).astype(int)
        
        p = Sigma.shape[0]
        loglik = -p * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) - np.sum(np.diag(covobs @ Sigma))
        loglik *= nobs
        nedge = (np.sum(adj) - p) / 2
        bic = np.log(nobs) * nedge - loglik

        results.append({
            "adj": adj,
            "Sigma": Sigma,
            "nedge": nedge,
            "bic": bic,
            "lambda": l1[i],
        })

    return results

def find_ks_d(x, y):
    n_x = len(x)
    n_y = len(y)
    w = np.concatenate((x, y))
    
    # Order and cumsum logic
    order_indices = np.argsort(w)
    z = np.cumsum(np.where(order_indices < n_x, 1 / n_x, -1 / n_y))
    
    # Sorting and finding maximum
    w_sorted = np.sort(w)
    w_max = np.max(w_sorted[z == np.max(z)])
    
    # Filtering and finding minimum
    z_sub = z[w_sorted > w_max]
    w_min_candidates = w_sorted[w_sorted > w_max]
    w_min = np.min(w_min_candidates[z_sub == np.min(z_sub)])
    
    return {'w_min': w_min, 'w_max': w_max}