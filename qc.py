from __future__ import annotations

import anndata as ad
import scanpy as sc

from .models import QCConfig


def filter_adata(adata: ad.AnnData, cfg: QCConfig = QCConfig()) -> ad.AnnData:
    """
    Run QC filtering workflow.

    Steps:
    1) Compute QC metrics and mark mitochondrial genes.
    2) Filter by genes-per-cell, cells-per-gene, and mitochondrial fraction.
    3) Optionally run Scrublet to remove predicted doublets.

    Args:
        adata: Input AnnData before filtering.
        cfg: QC filtering parameters (thresholds and Scrublet toggle).

    Returns:
        New AnnData object after QC filtering.
    """
    a = adata.copy()
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

    if cfg.run_scrublet:
        sc.pp.scrublet(a)
        a = a[~a.obs["predicted_doublet"], :].copy()

    return a
