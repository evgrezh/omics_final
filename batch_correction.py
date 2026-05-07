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


def run_combat_workflow(
    adata: ad.AnnData,
    cfg: NormalizationConfig,
    batch_key: str = "batch",
) -> ad.AnnData:
    from combat.pycombat import pycombat
    import pandas as pd

    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=cfg.target_sum)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, batch_key=batch_key)
    a = a[:, a.var["highly_variable"]].copy()

    expr_df = pd.DataFrame(
        a.X.toarray().T, index=a.var_names, columns=a.obs_names
    )
    corrected_df = pycombat(expr_df, a.obs[batch_key].tolist())
    a.X = corrected_df.T.values

    sc.pp.scale(a)
    n_pcs = _safe_n_pcs(a, cfg.n_pcs)   # _safe_n_pcs импортировать из normalization
    sc.tl.pca(a, n_comps=n_pcs)
    sc.pp.neighbors(a, use_rep="X_pca")
    sc.tl.leiden(a)
    sc.tl.umap(a)
    return a


 
def run_scvi_workflow(
    adata: ad.AnnData,
    batch_key: str = "batch",
    n_latent: int = 30,
    n_layers: int = 2,
    n_epochs: int = 400,
    batch_size: int = 1024,
) -> ad.AnnData:
    import scvi

    a = adata.copy()
    # raw counts should be in layers["counts"]
    scvi.model.SCVI.setup_anndata(a, batch_key=batch_key, layer="counts")
    model = scvi.model.SCVI(
        a, n_latent=n_latent, n_layers=n_layers, gene_likelihood="nb"
    )
    model.train(max_epochs=n_epochs, early_stopping=True, batch_size=batch_size)

    a.obsm["X_scVI"] = model.get_latent_representation()
    a.uns["scvi_model"] = model     
    sc.pp.neighbors(a, use_rep="X_scVI")
    sc.tl.leiden(a)
    sc.tl.umap(a)
    return a


 
def run_scanvi_workflow(
    adata: ad.AnnData,
    scvi_model,
    label_key: str = "cell_type",
    n_epochs: int = 20,
) -> ad.AnnData:
    import scvi

    a = adata.copy()
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="Unknown",
        labels_key=label_key,
    )
    scanvi_model.train(max_epochs=n_epochs, n_samples_per_label=100)

    a.obsm["X_scANVI"] = scanvi_model.get_latent_representation()
    sc.pp.neighbors(a, use_rep="X_scANVI")
    sc.tl.leiden(a)
    sc.tl.umap(a)
    return a


 
def run_scgpt_workflow(
    adata: ad.AnnData,
    model_dir: str = "models/scgpt",
    batch_size: int = 64,
) -> ad.AnnData:
    import scgpt

    a = adata.copy()
    embed = scgpt.tasks.embed_data(
        a, model_dir=model_dir, batch_size=batch_size,
        use_fast_transformer=True,
    )
    a.obsm["X_scGPT"] = embed
    sc.pp.neighbors(a, use_rep="X_scGPT")
    sc.tl.leiden(a)
    sc.tl.umap(a)
    return a



def run_geneformer_workflow(
    adata: ad.AnnData,
    model_dir: str = "models/geneformer",
    forward_batch_size: int = 200,
    nproc: int = 4,
) -> ad.AnnData:
    import numpy as np
    # upload saved embedding .npy
    embed_path = f"{model_dir}/geneformer_zeroshot_embeddings.npy"
    a = adata.copy()
    a.obsm["X_Geneformer"] = np.load(embed_path)
    sc.pp.neighbors(a, use_rep="X_Geneformer")
    sc.tl.leiden(a)
    sc.tl.umap(a)
    return a
