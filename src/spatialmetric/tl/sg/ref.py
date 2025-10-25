from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from .utils import (
    compute_group_stats,
    roc_auc_by_column_csc,
    roc_auc_by_column_dense,
)

def meecs(
    adata,
    cell_type_column: str,
    *,
    in_thresh: float = 0.20,
    out_thresh: float = 0.05,
    fdr: float = 0.05,
    min_or: float = 4.0,
    min_auc: float = 0.85,
    min_lfc: float = 0.25,
    min_cells: int = 10,
    use_raw: bool = True,
    layer: Optional[str] = None,
    max_auc_genes_per_group: Optional[int] = 5000,
    global_fdr: bool = False,
) -> pd.DataFrame:
    """
    MEeCS — **Mutual Exclusivity with Evidence-Combined Score**.

    This method detects **statistically exclusive cell-type marker genes** by combining
    multiple independent evidence streams in a **two-stage** design:

    Stage 1 — Fast Screening (vectorized)
        Keep genes that are:
        - Expressed in ≥ `in_thresh` fraction **inside** the group
        - Expressed in ≤ `out_thresh` fraction **outside** the group

    Stage 2 — Statistical Evidence Integration
        For screened genes, compute:
        • **Fisher’s exact test** on detection (enrichment)
           → p-values + Odds Ratios (OR)
        • **AUROC** (one-vs-rest) on continuous expression scale
           → distributional separation
        • **Log fold-change** (group means vs rest means)
           → magnitude of effect

        Genes must pass:
        - FDR-corrected p-value ≤ `fdr`
        - OR ≥ `min_or`
        - AUROC ≥ `min_auc`
        - LFC ≥ `min_lfc`

        Final score:
            score = log10(1/p_adj) * scaled(AUC) * scaled(LFC)

        Genes are returned **rank-ordered** by this composite exclusivity score.

    Parameters
    ----------
    adata : AnnData
        Expression matrix and cell metadata.
    cell_type_column : str
        Categorical column defining groups (e.g., `"cell_type"`).
    in_thresh : float, optional
        Inside-group minimum detection fraction.
    out_thresh : float, optional
        Outside-group maximum detection fraction.
    fdr : float, optional
        Benjamini–Hochberg false discovery rate cutoff.
    min_or : float, optional
        Minimum odds ratio for Fisher enrichment.
    min_auc : float, optional
        Minimum AUROC for distributional exclusivity.
    min_lfc : float, optional
        Minimum log fold-change.
    min_cells : int, optional
        Exclude groups smaller than this.
    use_raw : bool, optional
        Use `.raw.X` if present.
    layer : str or None, optional
        Expression from `.layers[layer]` instead of `.X` or `.raw.X`.
    max_auc_genes_per_group : int or None, optional
        Cap number of AUC computations per group (for speed).
    global_fdr : bool, optional
        If True, apply BH correction across all groups jointly; else within each group.

    Returns
    -------
    results : dict[str, pandas.DataFrame]
        An integrated dataframe of all cell types scores.
    """
    groups, genes, group_sizes, group_mean, expr_frac, rest_frac, keep_cell_mask = compute_group_stats(
        adata, cell_type_column, min_cells=min_cells, use_raw=use_raw, layer=layer
    )
    if groups.size == 0:
        return pd.DataFrame(columns=[
            "gene","expr_in","expr_out","OR","p_adj","AUC","LFC","score","cell_type"
        ])
    total_cells = group_sizes.sum()
    w = (group_sizes / max(total_cells, 1.0))[:, None]                     
    total_mean = (w * group_mean).sum(axis=0, keepdims=True)
    rest_mean = (total_mean - w * group_mean) / np.maximum(1.0 - w, 1e-8)
    lfc = (group_mean - rest_mean).astype(np.float32)
    screen = (expr_frac >= in_thresh) & (rest_frac <= out_thresh)
    if layer is not None:
        X = adata.layers[layer]
        use_raw = False
    elif use_raw and (adata.raw is not None):
        X = adata.raw.X
    else:
        X = adata.X
    X_sparse = sp.issparse(X)
    X = X.tocsr() if X_sparse else np.asarray(X)
    X = X[keep_cell_mask, :]
    X_csc = X.tocsc() if X_sparse else None
    labels = adata.obs[cell_type_column].to_numpy()
    labels = labels[keep_cell_mask]
    all_pvals = []
    all_idx_positions = []
    all_idx_arrays = []
    fisher_per_group = []
    for gi, gname in enumerate(groups):
        idx = np.where(screen[gi])[0]
        all_idx_arrays.append(idx)
        if idx.size == 0:
            fisher_per_group.append({
                "idx": idx,
                "keep_fisher_mask": np.zeros(0, dtype=bool),
                "ORs": np.zeros(0, dtype=float),
                "pvals": np.ones(0, dtype=float),
            })
            continue
        n_in = int(group_sizes[gi])
        n_out = int(total_cells - n_in)
        a = np.round(expr_frac[gi, idx]  * n_in).astype(int)
        b = n_in  - a
        c = np.round(rest_frac[gi, idx] * n_out).astype(int)
        d = n_out - c
        pvals = np.empty(idx.size, float)
        ORs   = np.empty(idx.size, float)
        for k in range(idx.size):
            table = np.array([[a[k], b[k]], [c[k], d[k]]], dtype=int)
            or_hat, p = fisher_exact(table, alternative="greater")
            ORs[k] = np.inf if (table[0,1] == 0 and table[1,0] > 0) else (max(or_hat, 0.0))
            pvals[k] = p
        fisher_per_group.append({
            "idx": idx,
            "ORs": ORs,
            "pvals": pvals,
        })
        if global_fdr:
            all_pvals.extend(pvals.tolist())
            all_idx_positions.extend([(gi, k) for k in range(idx.size)])
    if global_fdr and len(all_pvals) > 0:
        _, all_padj, _, _ = multipletests(np.asarray(all_pvals), method="fdr_bh")
        cursor = 0
        for gi, rec in enumerate(fisher_per_group):
            padj = np.ones(rec["pvals"].size, dtype=float)
            for k in range(rec["pvals"].size):
                padj[k] = all_padj[cursor]
                cursor += 1
            rec["padj"] = padj
    else:
        for rec in fisher_per_group:
            if rec["pvals"].size:
                _, padj, _, _ = multipletests(rec["pvals"], method="fdr_bh")
                rec["padj"] = padj
            else:
                rec["padj"] = rec["pvals"]
    results_list: List[pd.DataFrame] = []
    for gi, gname in enumerate(groups):
        rec = fisher_per_group[gi]
        idx = rec["idx"]
        if idx.size == 0:
            continue
        ORs = rec["ORs"]
        padj = rec["padj"]
        keep_fisher = (padj <= fdr) & (ORs >= min_or)
        idx2 = idx[keep_fisher].astype(int, copy=False)
        if idx2.size == 0:
            continue
        if (max_auc_genes_per_group is not None) and (idx2.size > max_auc_genes_per_group):
            sel_positions = np.random.choice(np.arange(idx2.size), size=max_auc_genes_per_group, replace=False)
            idx2_sampled = idx2[sel_positions]
            ORs2 = ORs[keep_fisher][sel_positions]
            padj2 = padj[keep_fisher][sel_positions]
        else:
            idx2_sampled = idx2
            ORs2 = ORs[keep_fisher]
            padj2 = padj[keep_fisher]
        y = (labels == gname).astype(np.int8)
        if X_sparse:
            aucs = roc_auc_by_column_csc(y, X_csc, idx2_sampled)
        else:
            X_block = X[:, idx2_sampled]
            aucs = roc_auc_by_column_dense(y, X_block)
        lfc2     = lfc[gi, idx2_sampled]
        expr_in  = expr_frac[gi, idx2_sampled]
        expr_out = rest_frac[gi, idx2_sampled]
        pass_auc = aucs >= min_auc
        pass_lfc = lfc2 >= min_lfc
        keep_final = pass_auc & pass_lfc
        if not np.any(keep_final):
            continue
        idx3 = idx2_sampled[keep_final]
        eps = 1e-12
        score = (
            np.log10(1.0 / np.maximum(padj2[keep_final], eps)) *
            (aucs[keep_final] - 0.5) * 2.0 *           
            np.clip(lfc2[keep_final], 0, 5) / 5.0           
        )
        df = pd.DataFrame({
            "gene": genes[idx3],
            "expr_in": expr_in[keep_final],
            "expr_out": expr_out[keep_final],
            "OR": ORs2[keep_final],
            "p_adj": padj2[keep_final],
            "AUC": aucs[keep_final],
            "LFC": lfc2[keep_final],
            "score": score,
        })
        df["cell_type"] = str(gname)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        results_list.append(df)

    if len(results_list) == 0:
        return pd.DataFrame(columns=[
            "gene","expr_in","expr_out","OR","p_adj","AUC","LFC","score","cell_type"
        ])
    out_df = pd.concat(results_list, ignore_index=True)
    out_df = out_df.sort_values("score", ascending=False).reset_index(drop=True)
    return out_df