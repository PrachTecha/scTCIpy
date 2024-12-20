from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as spr
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp
from tqdm import tqdm
from ._utils import easy_psd, est_pearson_scimpute, get_res_wGL_quic, find_ks_d

def sclink_cor(expr, ncores, nthre=20, dthre=0.9):
    """Estimate Pearson correlation matrix of Gene-Pair Pearson Correlation Coefficients (GPPCCs) using scImpute.

    Args:
        expr (Array): Expression data.
        ncores (int): Number of cores to use for parallel processing.
        nthre (int, optional): Number of thresholds for scImpute. Defaults to 20.
        dthre (float, optional): Threshold for scImpute. Defaults to 0.9.

    Returns:
        Array: Estimated Pearson correlation matrix.
    """
    cor_pearson = np.corrcoef(expr, rowvar=False)
    x_q = np.std(expr, axis=0)
    cor_pearson_c = est_pearson_scimpute(expr, cor_p=cor_pearson, thre_no=nthre, dthre=dthre, ncores=ncores)
    return cor_pearson_c

def gene_pearson(adata, highly_variable_gene=None, n_neighbor=300, n_gene=50, ncores=6):
    """Calculate Gene-Pair Pearson Correlation Coefficients (GPPCCs) for each cell in the dataset.

    Args:
        adata (AnnData): AnnData object containing expression data.
        highly_variable_gene (Union[list, Array], optional): List of highly variable genes to use for the analysis. If None, highly variable genes are identified using `sc.pp.highly_variable_genes`. Defaults to None.
        n_neighbor (int, optional): Number of neighbors to use for constructing the neighborhood graph. Defaults to 300.
        n_gene (int, optional): Number of highly variable genes to use for the analysis. Defaults to 50.
        ncores (int, optional): Number of cores to use for parallel processing. If None, all available cores are used. Defaults to 6.

    Returns:
        DataFrame: DataFrame containing the GPPCCs for each cell.
    """
    X = adata.X.toarray() if isinstance(adata.X, csr_matrix) else adata.X
    graph = adata.obsp['connectivities']
    c_names = adata.obs_names.values
    var_names = adata.var_names.values
    
    # Identify highly variable genes if not provided
    if highly_variable_gene is None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_gene, flavor='cell_ranger')
        highly_variable_gene = adata.var_names[adata.var.highly_variable].values
        highly_variable_indices = np.isin(var_names, highly_variable_gene)
    else:
        highly_variable_indices = np.isin(var_names, highly_variable_gene)
        
    # Extract highly variable genes' expression data
    hvg_exprs = X[:, highly_variable_indices]

    # Precompute column names efficiently
    name_tmp = np.array([f'{i}_{j}' for i in var_names[highly_variable_indices] for j in var_names[highly_variable_indices]])

    # Allocate results in chunks
    res = np.empty((len(c_names), len(var_names[highly_variable_indices]) ** 2), dtype=np.float32)

    for c_idx, c_n in tqdm(enumerate(c_names), total=len(c_names)):
        # Get neighbors
        graph_1d = graph[c_idx].indices

        # Extract relevant rows, ensure proper type
        val = hvg_exprs[graph_1d]

        # Compute correlation matrix for this subset
        corr = sclink_cor(val, ncores=-1 if ncores is None else ncores)

        # Set diagonal to NaN and flatten
        np.fill_diagonal(corr, np.nan)
        res[c_idx, :] = corr.flatten()

    # Return as a DataFrame
    return pd.DataFrame(res, columns=name_tmp, index=c_names)


def transition_index(data, highly_variable_gene:Union[list, np.ndarray]=None, group='leiden', n_neighbors=300, n_gene=50, return_pearson=False, layer=None, ncores=6):
    """Calculate transition index for each cell in the dataset to identify cells with high transition probabilities.

    Args:
        data (AnnData): AnnData object containing expression data.
        highly_variable_gene (Union[list, Array], optional): List of highly variable genes to use for the analysis. If None, highly variable genes are identified using `sc.pp.highly_variable_genes`. Defaults to None.
        group (str, optional): Column name in `adata.obs` containing group information. This can be clusters of timepoints. Defaults to 'leiden'.
        n_neighbors (int, optional): Number of neighbors to use for constructing the neighborhood graph. Defaults to 300.
        n_gene (int, optional): Number of highly variable genes to use for the analysis. Defaults to 50.
        return_pearson (bool, optional): Whether to return the correlation matrix. Defaults to False.
        layer (_type_, optional): Layer of AnnData object to use for the analysis. If None, `.X` is used. Defaults to None.
        ncores (int, optional): Number of cores to use for parallel processing. If None, all available cores are used. Defaults to 6.

    Returns:
        dict: Dictionary containing the correlation matrix and transition index if return_pearson is True.
    """
    
    results = {}
    
    for cluster in data.obs[group].unique().sort_values():
        print(f'Processing: {cluster}')
        
        # Subset data by group
        data_sub = data[data.obs[group] == cluster].copy()
        
        if data_sub.n_obs <= n_neighbors:
            continue
        
        # Preprocess: find variable features, scale data, and PCA
        sc.pp.highly_variable_genes(data_sub, flavor='cell_ranger', n_top_genes=2000)
        
        if layer is not None:
            X = data_sub[:,data_sub.var.highly_variable].layers[layer].copy()
        else:
            X = data_sub[:,data_sub.var.highly_variable].X.copy()
        
        # Calculate PCA
        data_sub.obsm['X_pca'] = sc.pp.pca(sc.pp.scale(X), n_comps=min(20, data_sub.n_obs))
        
        # Find neighbors using cosine distance
        sc.pp.neighbors(data_sub, n_neighbors=n_neighbors, metric='cosine')
        
        # Calculate gene-pair Pearson correlation coefficients
        pearson = gene_pearson(data_sub, highly_variable_gene=highly_variable_gene, n_neighbor=n_neighbors, n_gene=n_gene, ncores=ncores)
        
        # Fine-tuning: absolute values
        pearson = np.abs(pearson)
        res_1 = np.apply_along_axis(lambda x: ks_2samp(x, pearson.values[0, :], nan_policy='omit', alternative='greater').statistic, axis=1, arr=pearson)
        res_2 = np.apply_along_axis(lambda x: ks_2samp(x, pearson.values[0, :], nan_policy='omit', alternative='less').statistic, axis=1, arr=pearson)
        
        res_tmp = find_ks_d(pearson.values[np.argmax(res_1), :], pearson.values[np.argmax(res_2), :])
        min_res, max_res = min(res_tmp.values()), max(res_tmp.values())
        
        tmp = np.apply_along_axis(
            lambda x: np.sum((np.abs(x) > min_res) & (np.abs(x) < max_res)) / np.sum(np.abs(x) >= 0),
            axis=1, arr=pearson
        )
        
        data_sub.obs['pearson'] = tmp
        
        if np.min(tmp) == np.max(tmp):
            results[str(cluster)] = pearson
            continue
        
        # Filter cells with high pearson values
        high_pearson_cells = data_sub.obs[data_sub.obs['pearson'] > np.quantile(data_sub.obs['pearson'], 0.8)].index
        
        # Get top variable genes
        hvg = data_sub[high_pearson_cells].to_df().var(axis=0).nlargest(n_gene).index.values
    
        # Recalculate pearson with new variable genes
        pearson = gene_pearson(data_sub, highly_variable_gene=hvg, n_neighbor=n_neighbors, ncores=ncores)
        results[str(cluster)] = pearson
    
    # Combine results
    res = pd.concat([v for v in results.values()])
    pearson = res
    res = np.abs(res)
    
    res_1 = np.apply_along_axis(lambda x: ks_2samp(x, res.values[0, :], nan_policy='omit', alternative='greater').statistic, axis=1, arr=res)
    res_2 = np.apply_along_axis(lambda x: ks_2samp(x, res.values[0, :], nan_policy='omit', alternative='less').statistic, axis=1, arr=res)
    
    res_tmp = find_ks_d(res.values[np.argmax(res_1), :], res.values[np.argmax(res_2), :])
    min_res, max_res = min(res_tmp.values()), max(res_tmp.values())
    
    tmp = pd.DataFrame(
        np.apply_along_axis(
            lambda x: np.sum((np.abs(x) > min_res) & (np.abs(x) < max_res)) / np.sum(np.abs(x) >= 0),
            axis=1, arr=res),
        index=res.index, columns=['transition_index'])
    
    data.obs['transition_index'] = tmp
    
    # Return results
    if not return_pearson:
        return None
    else:
        return {'GPPCCs': pearson, 'transition_index': tmp}


def sclink_net(data, ncores=None, lda:list=None, nthre=20, dthre=0.9, return_summary=False):
    """scLink network construction for single-cell data.

    Args:
        data (AnnData): AnnData object containing expression data.
        ncores (int, optional): Number of cores to use for parallel processing. If None, all available cores are used. Defaults to None.
        lda (list, optional): List of lambda values to use for QUIC optimization. Defaults to None.
        nthre (int, optional): Number of thresholds for scImpute. Defaults to 20.
        dthre (float, optional): Threshold for scImpute. Defaults to 0.9.
        return_summary (bool, optional): Whether to return the summary results. Defaults to False.

    Returns:
        dict: Dictionary containing the summary results and correlation matrix if return_summary is True. 
              Otherwise, `.uns` and `.var` in the AnnData object are updated with 'scLink_summary' and 'scLink' keys respectively.
    """
    
    expr = data.X.toarray() if isinstance(data.X, csr_matrix) else data.X
    
    if lda is None:
        lda = np.linspace(1., 0.1, 19, dtype=np.float64).round(2).tolist()  # Equivalent to seq(1, 0.1, -0.05)

    if expr.shape[1] > 1000:
        warnings.warn(
            "For large expression matrices, computation may take a long time. "
            "Consider first using the sclink_cor function to check correlation structures."
        )
    
    # Step 1: Compute Pearson correlation matrix
    cor_pearson = np.corrcoef(expr, rowvar=False)
    
    # Step 2: Compute standard deviation of each gene (column)
    x_q = np.std(expr, axis=0)
    
    # Step 3: Estimate scImpute Pearson correlation
    cor_pearson_c = est_pearson_scimpute(
        expr, cor_p=cor_pearson, thre_no=nthre, dthre=dthre, ncores=ncores
    )
    
    # Step 4: Compute weights based on the estimated correlation
    weights_p = 1 - np.abs(cor_pearson_c)
    
    # Step 5: Convert correlation to covariance
    cov_pearson_c = np.diag(x_q) @ cor_pearson_c @ np.diag(x_q)
    
    # Step 6: Ensure positive semi-definiteness (PSD) of the covariance matrix
    cov_pearson_c = easy_psd(cov_pearson_c, method="perturb")
    
    # Step 7: Gene names and observations count
    nobs = expr.shape[0]
    
    # Step 8: Construct gene networks
    print("Constructing gene networks...")
    res_seq = get_res_wGL_quic(
        covobs=cov_pearson_c, weights=weights_p, nobs=nobs, l1=lda, ncores=ncores
    )
    
    # Step 9: Return results
    data.varp['scLink'] = cor_pearson_c
    data.uns['scLink_summary'] = res_seq
    
    return {"summary": res_seq, "cor": cor_pearson_c} if return_summary else None