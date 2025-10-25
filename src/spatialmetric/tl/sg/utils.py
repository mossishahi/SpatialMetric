from typing import Optional, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp

def compute_group_stats(
    adata,
    cell_type_column: str,
    *,
    min_cells: int = 10,
    use_raw: bool = True,
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sparse-efficient, group-wise statistics needed for marker and exclusivity analysis.

    This function performs a **single vectorized pass** over the AnnData expression matrix to
    extract the quantitative signals that multiple downstream exclusivity-finder functions rely on.

    Given a categorical label column (e.g., `"cell_type"`), it returns:

    - `group_mean[g, j]`: mean expression of gene `j` within group `g`
      (on whatever numerical scale is stored in `.raw.X` or `.layers[layer]`)
    - `expr_frac[g, j]`: fraction of cells in group `g` expressing gene `j` (>0)
    - `rest_frac[g, j]`: fraction of cells *outside* group `g` expressing gene `j`
      (weighted by varying group sizes)
    - `group_sizes[g]`: number of cells in each group
    - `keep_cell_mask`: boolean mask for cells in sufficiently large groups
      (aligned with rows of X and labels)

    Performance is optimized by:
    - Using **sparse matrix multiplications** for all group-wise summaries
    - Avoiding Python loops over genes or groups
    - Returning only groups with ≥ `min_cells` observations

    Parameters
    ----------
    adata : AnnData
        AnnData object with expression in `.X`, `.raw.X`, or `.layers[layer]`.
    cell_type_column : str
        Name of `adata.obs` column containing group annotations.
    min_cells : int, optional
        Minimum required group size. Smaller groups are excluded entirely.
    use_raw : bool, optional
        If True, uses `adata.raw.X` when present.
    layer : str or None, optional
        Use expression from the specified `.layers` entry instead of `.X` or `.raw.X`.

    Returns
    -------
    groups : (G,) ndarray[str]
        Names of groups retained after filtering.
    genes : (P,) ndarray[str]
        Gene names matching returned matrices.
    group_sizes : (G,) ndarray[float32]
        Number of cells per retained group.
    group_mean : (G,P) ndarray[float32]
        Group mean expression for each gene.
    expr_frac : (G,P) ndarray[float32]
        Detection fraction (`#cells with x>0 / group_size`).
    rest_frac : (G,P) ndarray[float32]
        Detection fraction outside each group.
    keep_cell_mask : (N,) ndarray[bool]
        Mask to subset `.obs`, `.X`, or `.layers` consistently downstream.

    Notes
    -----
    • All returned arrays are aligned and ready for statistical exclusivity scoring.
    • Recommend calling **after log-normalization** (e.g., `sc.pp.log1p`) for robust effect sizes.
    """
    if layer is not None:
        X = adata.layers[layer]
        use_raw = False
    elif use_raw and (adata.raw is not None):
        X = adata.raw.X
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = sp.csr_matrix(np.asarray(X))

    genes = np.asarray(adata.var_names, dtype=str)

    ct = pd.Categorical(adata.obs[cell_type_column])
    groups_all = np.asarray(ct.categories, dtype=str)
    codes_all = np.asarray(ct.codes)

    # filter groups with too few cells
    counts_all = np.bincount(codes_all[codes_all >= 0], minlength=len(groups_all))
    keep_group_mask = counts_all >= min_cells
    keep_cell_mask = (codes_all >= 0) & keep_group_mask[codes_all]

    if keep_cell_mask.sum() == 0:
        empty = np.array([], dtype=float)
        return (
            np.array([], dtype=str),
            genes,
            np.array([], dtype=float),
            np.zeros((0, genes.size), dtype=np.float32),
            np.zeros((0, genes.size), dtype=np.float32),
            np.zeros((0, genes.size), dtype=np.float32),
            np.zeros(adata.n_obs, dtype=bool),
        )

    X = X[keep_cell_mask, :]
    codes = codes_all[keep_cell_mask]
    groups = groups_all[keep_group_mask]

    old_to_new = -np.ones(len(keep_group_mask), dtype=int)
    old_to_new[np.where(keep_group_mask)[0]] = np.arange(keep_group_mask.sum())
    codes = old_to_new[codes]

    n_cells, n_genes = X.shape
    n_groups = len(groups)

    data = np.ones(n_cells, dtype=np.float32)
    rows = np.arange(n_cells, dtype=np.int32)
    cols = codes.astype(np.int32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_groups))

    group_sizes = np.asarray(G.sum(axis=0)).ravel().astype(np.float32)

    group_sum = (G.T @ X).astype(np.float32)
    inv_sizes = sp.diags(1.0 / np.maximum(group_sizes, 1.0), format="csr")
    group_mean = (inv_sizes @ group_sum).toarray().astype(np.float32)

    X_bin = X.copy()
    X_bin.data[:] = 1.0
    group_sum_bin = (G.T @ X_bin).astype(np.float32)
    expr_frac = (inv_sizes @ group_sum_bin).toarray().astype(np.float32)

    total_sum_bin = np.asarray(group_sum_bin.sum(axis=0)).ravel()
    total_cells = float(group_sizes.sum())
    rest_sum = total_sum_bin[None, :] - group_sum_bin.toarray()
    rest_sizes = (total_cells - group_sizes)[:, None]
    rest_sizes = np.maximum(rest_sizes, 1.0)
    rest_frac = (rest_sum / rest_sizes).astype(np.float32)

    return groups, genes, group_sizes, group_mean, expr_frac, rest_frac, keep_cell_mask


def _roc_auc_from_sorted(scores_sorted, y_sorted, n_pos, n_neg):
    """
    Compute ROC AUC from an already-sorted vector of non-zero scores and labels,
    accounting for implicit zeros.

    This helper assumes scores are non-negative and that only entries with
    scores > 0 are provided in `scores_sorted`/`y_sorted`. It adds the correct
    Mann–Whitney U-statistic contributions from the omitted zero-valued
    observations so the result matches the ROC AUC computed on the full dense
    vector.

    Parameters
    ----------
    scores_sorted : (m,) ndarray[float]
        Non-zero scores for a single feature, sorted in ascending order.
    y_sorted : (m,) ndarray[{0,1} or bool]
        Binary labels aligned with `scores_sorted` (1 = positive, 0 = negative).
    n_pos : int
        Total number of positives in the full dataset (including zeros).
    n_neg : int
        Total number of negatives in the full dataset (including zeros).

    Returns
    -------
    float
        ROC AUC in [0, 1]. Returns 0.5 for degenerate cases (no positives,
        no negatives, or no non-zero scores).

    Notes
    -----
    - Equal-score ties are handled via average ranks.
    - Zero-valued scores are treated as ties among themselves and lower than any
      strictly positive score.
    - Time complexity: O(m) after sorting, where m is the number of non-zero
      scores provided.
    """
    m = scores_sorted.size
    if m == 0:
        return 0.5

    b = np.empty(m + 1, dtype=np.int64)
    b[0] = 0
    k = 1
    for i in range(1, m):
        if scores_sorted[i] != scores_sorted[i - 1]:
            b[k] = i
            k += 1
    b[k] = m
    b = b[:k + 1]

    sum_pos_ranks = 0.0
    pos_nz = 0
    for g in range(k):
        s = b[g]
        e = b[g + 1]
        avg_rank = (s + 1 + e) * 0.5
        pos_in_block = int(y_sorted[s:e].sum())
        sum_pos_ranks += pos_in_block * avg_rank
        pos_nz += pos_in_block
    neg_nz = m - pos_nz

    U_nz = sum_pos_ranks - pos_nz * (pos_nz + 1) * 0.5

    pos_zero = n_pos - pos_nz
    neg_zero = n_neg - neg_nz

    U_zero_nz = pos_nz * neg_zero
    U_zero_zero = 0.5 * pos_zero * neg_zero

    U_total = U_nz + U_zero_nz + U_zero_zero
    return U_total / (n_pos * n_neg) if (n_pos > 0 and n_neg > 0) else 0.5


def roc_auc_by_column_csc(y_binary, X_csc, cols):
    """
    Compute per-column ROC AUC for a non-negative sparse matrix in CSC format.

    The algorithm sorts only the non-zero values of each requested column and
    analytically adds the contribution of the implicit zeros. This yields the
    same AUC as if the full dense column (with many zeros) had been sorted,
    but in O(nnz_j log nnz_j) time per column j.

    Parameters
    ----------
    y_binary : (n_samples,) ndarray[{0,1} or bool]
        Binary ground-truth labels (1 = positive, 0 = negative).
    X_csc : scipy.sparse.csc_matrix, shape (n_samples, n_features)
        Non-negative feature matrix in CSC format. Zeros represent absence and
        are not stored.
    cols : (k,) ndarray[int]
        Indices of columns for which to compute AUC.

    Returns
    -------
    aucs : (k,) ndarray[float64]
        ROC AUC for each requested column. Returns 0.5 for columns with no
        non-zero entries or when y contains only one class.

    Notes
    -----
    - Assumes non-negative scores; zero is treated as a common baseline value.
    - Ties among equal non-zero scores are handled by average ranks (stable
      merge sort is used).
    - For best performance, ensure the matrix is already CSC (`X.tocsc()`).
    """
    n_pos = int(y_binary.sum())
    n_neg = y_binary.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(cols.size, 0.5, dtype=np.float64)

    aucs = np.empty(cols.size, dtype=np.float64)
    indptr = X_csc.indptr
    indices = X_csc.indices
    data = X_csc.data

    for t, j in enumerate(cols):
        start, end = indptr[j], indptr[j + 1]
        rows = indices[start:end]
        if rows.size == 0:
            aucs[t] = 0.5
            continue
        scores = data[start:end]
        order = np.argsort(scores, kind="mergesort")
        scores_sorted = scores[order]
        y_sorted = y_binary[rows][order]
        aucs[t] = _roc_auc_from_sorted(scores_sorted, y_sorted, n_pos, n_neg)
    return aucs


def roc_auc_by_column_dense(y_binary, X_block):
    """
    Compute per-column ROC AUC for a dense, non-negative feature block.

    Only strictly positive entries in each column are explicitly sorted; zeros
    are handled analytically to obtain the exact dense AUC. This mirrors the
    sparse implementation but operates on dense arrays.

    Parameters
    ----------
    y_binary : (n_samples,) ndarray[{0,1} or bool]
        Binary ground-truth labels (1 = positive, 0 = negative).
    X_block : ndarray[float], shape (n_samples, n_features)
        Dense, non-negative feature matrix.

    Returns
    -------
    aucs : (n_features,) ndarray[float64]
        ROC AUC for each column. 0.5 for columns with no positive-valued
        entries or when y contains only one class.

    Notes
    -----
    - Values are assumed non-negative; entries equal to zero are treated as the
      baseline value and are not explicitly sorted.
    - Time complexity per column is O(nz_j log nz_j), where nz_j is the count
      of strictly positive entries in that column.
    """
    n_pos = int(y_binary.sum())
    n_neg = y_binary.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(X_block.shape[1], 0.5, dtype=np.float64)

    aucs = np.empty(X_block.shape[1], dtype=np.float64)
    for t in range(X_block.shape[1]):
        x = X_block[:, t]
        nz = x > 0
        if not np.any(nz):
            aucs[t] = 0.5
            continue
        scores = x[nz]
        y_nz = y_binary[nz]
        order = np.argsort(scores, kind="mergesort")
        scores_sorted = scores[order]
        y_sorted = y_nz[order]
        aucs[t] = _roc_auc_from_sorted(scores_sorted, y_sorted, n_pos, n_neg)
    return aucs

