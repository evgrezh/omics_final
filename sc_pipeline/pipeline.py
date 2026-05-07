from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import anndata as ad

from .batch_correction import (
    run_combat_workflow,
    run_geneformer_workflow,
    run_no_correction_workflow,
    run_scanvi_workflow,
    run_scgpt_workflow,
    run_scvi_workflow,
)
from .metrics import (
    evaluate_methods,
    metric_ari,
    metric_asw_batch,
    metric_asw_label,
    metric_bras,
    metric_clisi,
    metric_graph_connectivity,
    metric_ilisi,
    metric_isolated_labels_f1,
    metric_kbet,
    metric_kbet_per_label,
    metric_nmi,
)
from .models import NormalizationConfig, PipelineConfig, PipelineResult, QCConfig
from .qc import filter_adata


def _run_qc(
    adata: ad.AnnData,
    batch_key: str,
    qc_cfg: QCConfig,
    per_batch: bool,
) -> ad.AnnData:
    """
    Run QC either globally or per batch.

    Args:
        adata: Input AnnData.
        batch_key: Batch column in obs.
        qc_cfg: QC configuration.
        per_batch: If True, run QC independently within each batch.

    Returns:
        AnnData after QC filtering.
    """
    if not per_batch:
        return filter_adata(adata, qc_cfg)

    cleaned: list[ad.AnnData] = []
    for batch_name in adata.obs[batch_key].astype(str).unique():
        batch_adata = adata[adata.obs[batch_key].astype(str) == batch_name].copy()
        cleaned.append(filter_adata(batch_adata, qc_cfg))

    out = ad.concat(cleaned, join="inner")
    out.obs_names_make_unique()
    return out


def _pick_rep(a: ad.AnnData) -> str:
    """
    Auto-select the appropriate embedding representation.
    Checks for method-specific embeddings in priority order.

    Args:
        a: AnnData after batch correction.

    Returns:
        Key in obsm to use as embedding.
    """
    for rep in ("X_scANVI", "X_scVI", "X_scGPT", "X_Geneformer", "X_pca_harmony", "X_scanorama"):
        if rep in a.obsm:
            return rep
    return "X_pca"


def _build_method_handlers(
    adata_qc: ad.AnnData,
    cfg: PipelineConfig,
    norm_cfg: NormalizationConfig,
    corrected: dict[str, ad.AnnData],
) -> Mapping[str, Callable[[], ad.AnnData]]:
    """
    Build method registry for batch-correction workflows.
    Note: 'scanvi' must run after 'scvi' — it reads the trained model
    from corrected['scvi'].uns['scvi_model'].

    Args:
        adata_qc: AnnData after QC.
        cfg: Pipeline configuration.
        norm_cfg: Normalization configuration.
        corrected: Dict accumulating results (needed for scanvi → scvi dependency).

    Returns:
        Mapping method_name -> zero-argument runner.
    """
    return {
        "no_correction": lambda: run_no_correction_workflow(
            adata_qc,
            norm_cfg,
        ),
        "combat": lambda: run_combat_workflow(
            adata_qc,
            norm_cfg,
            batch_key=cfg.batch_key,
        ),
        "scvi": lambda: run_scvi_workflow(
            adata_qc,
            batch_key=cfg.batch_key,
            n_latent=cfg.scvi_cfg.n_latent,
            n_layers=cfg.scvi_cfg.n_layers,
            n_epochs=cfg.scvi_cfg.n_epochs,
            batch_size=cfg.scvi_cfg.batch_size,
        ),
        "scanvi": lambda: run_scanvi_workflow(
            adata_qc,
            scvi_model=corrected["scvi"].uns["scvi_model"],  # scvi should be in cfg.methods earlier
            label_key=cfg.label_key,
        ),
        "scgpt": lambda: run_scgpt_workflow(
            adata_qc,
            model_dir=cfg.scgpt_cfg.model_dir,
            batch_size=cfg.scgpt_cfg.batch_size,
        ),
        "geneformer": lambda: run_geneformer_workflow(
            adata_qc,
            model_dir=cfg.geneformer_cfg.model_dir,
        ),
    }


def _build_default_metrics(
    metric_names: Sequence[str],
    batch_key: str,
    label_key: str,
) -> Mapping[str, Callable[[ad.AnnData], float]]:
    """
    Build mapping of builtin metric names to callables.
    Note: pcr_comparison and hvg_overlap require adata_raw — compute
    separately in notebook 05 and pass via custom_metrics.

    Args:
        metric_names: List of builtin metric names.
        batch_key: Batch column for batch-related metrics.
        label_key: Cell type annotation column (e.g. 'cell_type').

    Returns:
        Mapping metric_name -> callable(AnnData) -> float.
    """
    metric_builders: dict[str, Callable] = {
        # ── Batch removal ─────────────────────────────────────────────
        "ilisi": lambda b: (
            lambda a: metric_ilisi(a, batch_key=b, use_rep=_pick_rep(a))
        ),
        "kbet": lambda b: (
            lambda a: metric_kbet(a, batch_key=b, use_rep=_pick_rep(a))
        ),
        "kbet_pg": lambda b: (
            lambda a: metric_kbet_per_label(a, batch_key=b, label_key=label_key, use_rep=_pick_rep(a))
        ),
        "bras": lambda b: (
            lambda a: metric_bras(a, batch_key=b, label_key=label_key, use_rep=_pick_rep(a))
        ),
        "asw_batch": lambda b: (
            lambda a: metric_asw_batch(a, batch_key=b, label_key=label_key, use_rep=_pick_rep(a))
        ),
        # ── Biological conservation ────────────────────────────────────
        "clisi": lambda b: (
            lambda a: metric_clisi(a, label_key=label_key, use_rep=_pick_rep(a))
        ),
        "isolated_labels_f1": lambda b: (
            lambda a: metric_isolated_labels_f1(a, label_key=label_key, use_rep=_pick_rep(a))
        ),
        "ari": lambda b: (
            lambda a: metric_ari(a, label_key=label_key)
        ),
        "nmi": lambda b: (
            lambda a: metric_nmi(a, label_key=label_key)
        ),
        "asw_label": lambda b: (
            lambda a: metric_asw_label(a, label_key=label_key, use_rep=_pick_rep(a))
        ),
        "graph_connectivity": lambda b: (
            lambda a: metric_graph_connectivity(a, label_key=label_key, use_rep=_pick_rep(a))
        ),
        # ── pcr_comparison, hvg_overlap require adata_raw
    }

    metrics: dict[str, Callable[[ad.AnnData], float]] = {}
    for name in metric_names:
        if name not in metric_builders:
            raise ValueError(
                f"Unknown builtin metric: `{name}`. "
                f"Available: {sorted(metric_builders.keys())}. "
                f"For pcr_comparison / hvg_overlap use custom_metrics."
            )
        metrics[name] = metric_builders[name](batch_key)
    return metrics


def run_pipeline(
    adata: ad.AnnData,
    cfg: PipelineConfig = PipelineConfig(),
    qc_cfg: QCConfig = QCConfig(),
    norm_cfg: NormalizationConfig = NormalizationConfig(),
    custom_metrics: Mapping[str, Callable[[ad.AnnData], float]] | None = None,
) -> PipelineResult:
    """
    Unified pipeline entrypoint:
    QC -> batch correction -> metric evaluation.

    For 'scanvi', 'scvi' must appear earlier in cfg.methods — the pipeline
    runs methods in order and scanvi reads the trained model from the scvi result.

    Args:
        adata: Input AnnData (raw counts expected in layers['counts']).
        cfg: Pipeline config (batch key, methods, metrics, model configs).
        qc_cfg: QC filtering configuration.
        norm_cfg: Normalization/embedding configuration for ComBat and baseline.
        custom_metrics: Additional metrics merged with builtin metrics.
                        Use this for pcr_comparison and hvg_overlap.

    Returns:
        PipelineResult with adata_qc, corrected dict, and metrics_table.
    """
    if cfg.batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{cfg.batch_key}` is missing in adata.obs.")
    if cfg.label_key not in adata.obs.columns:
        raise KeyError(f"Column `{cfg.label_key}` is missing in adata.obs.")

    # scANVI зависит от scVI — проверить порядок
    if "scanvi" in cfg.methods and "scvi" in cfg.methods:
        methods_list = list(cfg.methods)
        if methods_list.index("scvi") > methods_list.index("scanvi"):
            raise ValueError("'scvi' must appear before 'scanvi' in cfg.methods.")

    if "scanvi" in cfg.methods and "scvi" not in cfg.methods:
        raise ValueError("'scanvi' requires 'scvi' to also be in cfg.methods.")

    # QC
    adata_qc = _run_qc(adata, cfg.batch_key, qc_cfg, per_batch=cfg.run_qc_per_batch)

    # Batch correction 
    corrected: dict[str, ad.AnnData] = {}
    method_handlers = _build_method_handlers(adata_qc, cfg, norm_cfg, corrected)

    for method in cfg.methods:
        if method not in method_handlers:
            raise ValueError(
                f"Unknown batch-correction method: `{method}`. "
                f"Available: {sorted(method_handlers.keys())}."
            )
        print(f"[Pipeline] Running method: {method} ...")
        corrected[method] = method_handlers[method]()
        print(f"[Pipeline] Done: {method}")

    metrics = _build_default_metrics(cfg.metrics, batch_key=cfg.batch_key, label_key=cfg.label_key)
    if custom_metrics:
        metrics = {**metrics, **custom_metrics}

    metrics_table = evaluate_methods(corrected, metrics)
    return PipelineResult(adata_qc=adata_qc, corrected=corrected, metrics_table=metrics_table)
