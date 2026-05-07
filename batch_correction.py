from __future__ import annotations

import importlib
from collections.abc import Sequence

import anndata as ad
import numpy as np
import scanpy as sc

from .models import NormalizationConfig
from .normalization import preprocess_for_integration, run_basic_scanpy_embedding


def _load_optional_module(module_name: str):
    """
    Lazily import an optional dependency.

    Args:
        module_name: Name of the Python module to import.

    Returns:
        Imported module object.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Install `{module_name}` to use this method.") from exc


def run_no_correction_workflow(
    adata: ad.AnnData,
    cfg: NormalizationConfig,
) -> ad.AnnData:
    """
    Baseline workflow without batch correction.

    Args:
        adata: Input AnnData.
        cfg: Preprocessing/embedding configuration.

    Returns:
        AnnData after normalization, PCA, neighbors, Leiden, and UMAP.
    """
    return run_basic_scanpy_embedding(
        adata=adata,
        cfg=cfg,
        use_hvg_by_batch=False,
        neighbor_rep="X_pca",
        normalize_per_batch=True,
    )


def run_bbknn_workflow(
    adata: ad.AnnData,
    cfg: NormalizationConfig,
    batch_key: str = "batch",
) -> ad.AnnData:
    """
    Batch correction workflow using BBKNN.

    Args:
        adata: Input AnnData.
        cfg: Preprocessing/embedding configuration.
        batch_key: Batch column in ``adata.obs``.

    Returns:
        AnnData with BBKNN graph, Leiden labels, and UMAP.
    """
    bbknn = _load_optional_module("bbknn")

    a = preprocess_for_integration(
        adata=adata,
        cfg=cfg,
        use_hvg_by_batch=True,
        batch_key_override=batch_key,
        normalize_per_batch=True,
    )
    n_pcs = min(cfg.n_pcs, a.obsm["X_pca"].shape[1])
    bbknn.bbknn(a, batch_key=batch_key, n_pcs=n_pcs)
    sc.tl.leiden(a, resolution=cfg.leiden_resolution)
    sc.tl.umap(a)
    return a


def run_scanorama_workflow(
    adatas_by_batch: Sequence[ad.AnnData],
    cfg: NormalizationConfig,
    batch_labels: Sequence[str] | None = None,
    batch_key: str = "batch",
) -> ad.AnnData:
    """
    Batch correction workflow using Scanorama.

    Args:
        adatas_by_batch: List of AnnData objects, one per batch.
        cfg: Preprocessing/embedding configuration.
        batch_labels: Optional batch labels for merged AnnData.
        batch_key: Batch column name in merged AnnData.

    Returns:
        Integrated AnnData with ``X_scanorama``, neighbors, Leiden, and UMAP.
    """
    scanorama = _load_optional_module("scanorama")

    if len(adatas_by_batch) < 2:
        raise ValueError("Scanorama requires at least 2 batches.")

    prepared = [x.copy() for x in adatas_by_batch]
    for i, a in enumerate(prepared):
        prepared[i] = preprocess_for_integration(
            adata=a,
            cfg=cfg,
            use_hvg_by_batch=False,
        )

    scanorama.integrate_scanpy(prepared)

    keys = (
        {batch_labels[i]: prepared[i] for i in range(len(prepared))}
        if batch_labels is not None
        else {str(i): prepared[i] for i in range(len(prepared))}
    )
    a = ad.concat(keys, label=batch_key, join="inner")
    a.obs_names_make_unique()
    n_pcs = min(cfg.n_pcs, a.obsm["X_scanorama"].shape[1])
    sc.pp.neighbors(a, n_pcs=n_pcs, use_rep="X_scanorama")
    sc.tl.leiden(a, resolution=cfg.leiden_resolution)
    sc.tl.umap(a)
    return a


def run_harmony_workflow(
    adata: ad.AnnData,
    cfg: NormalizationConfig,
    batch_key: str = "batch",
    max_iter_harmony: int = 20,
) -> ad.AnnData:
    """
    Batch correction workflow using Harmony.

    Args:
        adata: Input AnnData.
        cfg: Preprocessing/embedding configuration.
        batch_key: Batch column in ``adata.obs``.
        max_iter_harmony: Maximum number of Harmony iterations.

    Returns:
        AnnData with ``X_pca_harmony``, neighbors, Leiden, and UMAP.
    """
    hm = _load_optional_module("harmonypy")

    a = preprocess_for_integration(
        adata=adata,
        cfg=cfg,
        use_hvg_by_batch=True,
        batch_key_override=batch_key,
        normalize_per_batch=True,
    )
    # harmonypy -> torch does not accept numpy views with negative strides.
    # Ensure a contiguous array before passing PCA matrix to Harmony.
    pca_data = np.ascontiguousarray(a.obsm["X_pca"])
    meta_data = a.obs[[batch_key]]
    ho = hm.run_harmony(
        pca_data,
        meta_data,
        [batch_key],
        max_iter_harmony=max_iter_harmony,
    )

    if ho.Z_corr.shape[0] == pca_data.shape[1]:
        a.obsm["X_pca_harmony"] = ho.Z_corr.T
    else:
        a.obsm["X_pca_harmony"] = ho.Z_corr

    n_pcs = min(cfg.n_pcs, a.obsm["X_pca_harmony"].shape[1])
    sc.pp.neighbors(a, n_pcs=n_pcs, use_rep="X_pca_harmony")
    sc.tl.leiden(a, resolution=cfg.leiden_resolution)
    sc.tl.umap(a)
    return a
