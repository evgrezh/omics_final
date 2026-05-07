from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from .models import QCConfig


def filter_adata(adata: ad.AnnData, cfg: QCConfig = QCConfig()) -> ad.AnnData:
    """
    Run QC filtering workflow.

    Steps:
        1) Compute QC metrics and mark mitochondrial genes.
        2) Filter by genes-per-cell, cells-per-gene, and mitochondrial fraction.
        3) Optionally run Scrublet to remove predicted doublets.
           If 'predicted_doublet' already exists in obs (added by atlas authors),
           skips Scrublet and uses existing labels.

    Args:
        adata: Input AnnData before filtering.
        cfg: QC filtering parameters (thresholds and Scrublet toggle).

    Returns:
        New AnnData object after QC filtering.
    """
    a = adata.copy()
    n_start = a.n_obs

    a.var["mt"] = a.var_names.str.startswith(cfg.mt_prefix)
    sc.pp.calculate_qc_metrics(
        a,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    sc.pp.filter_cells(a, min_genes=cfg.min_genes)
    sc.pp.filter_cells(a, max_genes=cfg.max_genes)
    sc.pp.filter_genes(a, min_cells=cfg.min_cells_per_gene)
    a = a[a.obs["pct_counts_mt"] < cfg.max_pct_mt, :].copy()
    n_after_filters = a.n_obs

    if cfg.run_scrublet:
        if "predicted_doublet" in a.obs.columns:
            print("[QC] Scrublet already run by atlas authors — using existing 'predicted_doublet'")
        else:
            sc.pp.scrublet(a)
        a = a[~a.obs["predicted_doublet"], :].copy()

    n_final = a.n_obs
    print(
        f"[QC] {n_start} → {n_after_filters} (after filters) → {n_final} (after doublets) "
        f"| removed {n_start - n_final} cells ({(n_start - n_final) / n_start * 100:.1f}%)"
    )
    return a


def stratified_subsample(
    adata: ad.AnnData,
    n_cells: int = 125_000,
    stratify_by: list[str] = ["cell_type", "batch"],
    random_state: int = 42,
) -> ad.AnnData:
    """
    Stratified subsample jointly by cell_type and batch.
    Preserves rare populations proportionally.

    Args:
        adata: Input AnnData after QC.
        n_cells: Target number of cells after subsampling.
        stratify_by: Columns to stratify by (default: cell_type + batch).
        random_state: Random seed for reproducibility.

    Returns:
        Subsampled AnnData object.
    """
    for col in stratify_by:
        if col not in adata.obs.columns:
            raise KeyError(f"Column `{col}` is missing in adata.obs.")

    if n_cells >= adata.n_obs:
        print(f"[Subsample] Requested {n_cells} >= n_obs {adata.n_obs}, skipping subsample.")
        return adata.copy()

    groups = adata.obs[stratify_by].astype(str).apply(
        lambda x: "__".join(x), axis=1
    )
    counts = groups.value_counts()
    fracs = (counts / counts.sum() * n_cells).astype(int).clip(lower=1)

    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for group, count in fracs.items():
        idx = np.where(groups == group)[0]
        pick = rng.choice(idx, size=min(count, len(idx)), replace=False)
        selected.extend(pick.tolist())

    result = adata[selected].copy()
    print(
        f"[Subsample] {adata.n_obs} → {result.n_obs} cells "
        f"| stratified by {stratify_by}"
    )
    return result
