from __future__ import annotations

import anndata as ad
import scanpy as sc

from .models import NormalizationConfig


def _safe_n_pcs(adata: ad.AnnData, requested_n_pcs: int) -> int:
    """
    Return a PCA component count valid for current data shape.

    Args:
        adata: Dataset for which PCA will be computed.
        requested_n_pcs: Requested number of PCA components.

    Returns:
        Safe component count that does not exceed data limits.
    """
    max_allowed = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_allowed < 1:
        raise ValueError(
            f"Not enough data for PCA: n_obs={adata.n_obs}, n_vars={adata.n_vars}."
        )
    return int(min(requested_n_pcs, max_allowed))


def normalize_batches_separately(
    adata: ad.AnnData,
    batch_key: str,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Normalize each batch independently.

    Pipeline per batch:
    normalize_total -> log1p -> highly_variable_genes -> scale.

    Args:
        adata: Input AnnData.
        batch_key: Batch column in ``adata.obs``.
        target_sum: ``normalize_total`` target sum.

    Returns:
        AnnData after per-batch normalization and concatenation.
    """
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{batch_key}` is missing in adata.obs.")

    normalized_batches: list[ad.AnnData] = []
    batch_series = adata.obs[batch_key].astype(str)
    for batch_name in batch_series.unique():
        batch_adata = adata[batch_series == batch_name].copy()
        sc.pp.normalize_total(batch_adata, target_sum=target_sum)
        sc.pp.log1p(batch_adata)
        sc.pp.highly_variable_genes(batch_adata)
        sc.pp.scale(batch_adata)
        normalized_batches.append(batch_adata)

    out = ad.concat(normalized_batches, join="inner", label=batch_key)
    out.obs_names_make_unique()
    return out


def preprocess_for_integration(
    adata: ad.AnnData,
    cfg: NormalizationConfig,
    use_hvg_by_batch: bool = False,
    batch_key_override: str | None = None,
    normalize_per_batch: bool = False,
) -> ad.AnnData:
    """
    Generic preprocessing block:
    normalize_total -> log1p -> HVG -> optional subset -> scale -> PCA.

    Args:
        adata: Input AnnData.
        cfg: Normalization configuration.
        use_hvg_by_batch: Compute HVGs using ``batch_key`` awareness.
        batch_key_override: Override ``cfg.batch_key`` for this call.
        normalize_per_batch: Run normalize/log/HVG/scale per batch.

    Returns:
        Preprocessed AnnData with PCA embedding.
    """
    if normalize_per_batch:
        eff_batch_key = batch_key_override if batch_key_override is not None else cfg.batch_key
        if eff_batch_key is None:
            raise ValueError(
                "normalize_per_batch=True requires a batch_key (cfg.batch_key or batch_key_override)."
            )
        a = normalize_batches_separately(
            adata=adata,
            batch_key=eff_batch_key,
            target_sum=cfg.target_sum,
        )
    else:
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=cfg.target_sum)
        sc.pp.log1p(a)
        hvg_batch_key = batch_key_override if batch_key_override is not None else cfg.batch_key
        if not use_hvg_by_batch:
            hvg_batch_key = None
        sc.pp.highly_variable_genes(a, batch_key=hvg_batch_key)
        sc.pp.scale(a)

    if cfg.subset_hvg and "highly_variable" in a.var.columns:
        a = a[:, a.var["highly_variable"]].copy()

    if not normalize_per_batch:
        sc.pp.scale(a)
    n_pcs = _safe_n_pcs(a, cfg.n_pcs)
    sc.tl.pca(a, n_comps=n_pcs)
    return a


def run_basic_scanpy_embedding(
    adata: ad.AnnData,
    cfg: NormalizationConfig = NormalizationConfig(),
    use_hvg_by_batch: bool = False,
    neighbor_rep: str = "X_pca",
    normalize_per_batch: bool = False,
) -> ad.AnnData:
    """
    End-to-end baseline block: preprocessing + neighbors + Leiden + UMAP.

    Args:
        adata: Input AnnData.
        cfg: Normalization/clustering configuration.
        use_hvg_by_batch: Enable batch-aware HVG selection.
        neighbor_rep: Representation key for neighbors (for example ``X_pca``).
        normalize_per_batch: Normalize cells per batch.

    Returns:
        AnnData with neighbors graph, Leiden labels, and UMAP.
    """
    a = preprocess_for_integration(
        adata=adata,
        cfg=cfg,
        use_hvg_by_batch=use_hvg_by_batch,
        normalize_per_batch=normalize_per_batch,
    )
    n_pcs = min(cfg.n_pcs, a.obsm["X_pca"].shape[1])
    sc.pp.neighbors(a, n_pcs=n_pcs, use_rep=neighbor_rep)
    sc.tl.leiden(a, resolution=cfg.leiden_resolution)
    sc.tl.umap(a)
    return a
