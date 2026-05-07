from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import anndata as ad

from .batch_correction import (
    run_bbknn_workflow,
    run_harmony_workflow,
    run_no_correction_workflow,
    run_scanorama_workflow,
)
from .metrics import evaluate_methods, metric_ilisi
from .metrics import (
    metric_asw_batch,
    metric_asw_label,
    metric_clisi,
    metric_graph_connectivity,
    metric_kbet,
    metric_kbet_per_label,
)
from .models import NormalizationConfig, PipelineConfig, PipelineResult, QCConfig
from .qc import filter_adata


def _run_qc(adata: ad.AnnData, batch_key: str, qc_cfg: QCConfig, per_batch: bool) -> ad.AnnData:
    """
    Run QC either globally or per batch.

    Args:
        adata: Input AnnData.
        batch_key: Batch column in ``obs``.
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
    out = ad.concat(cleaned, join="inner", label=batch_key)
    out.obs_names_make_unique()
    return out


def _build_default_metrics(
    metric_names: Sequence[str],
    batch_key: str,
    label_key: str,
) -> Mapping[str, Callable[[ad.AnnData], float]]:
    """
    Build mapping of builtin metric names to callables.

    Args:
        metric_names: List of builtin metric names.
        batch_key: Batch column for batch-related metrics.
        label_key: Biological label column (for example ``leiden`` or ``cell_type``).

    Returns:
        Mapping ``metric_name -> callable(AnnData) -> float``.
    """
    def _pick_rep(a: ad.AnnData) -> str:
        if "X_pca_harmony" in a.obsm:
            return "X_pca_harmony"
        if "X_scanorama" in a.obsm:
            return "X_scanorama"
        return "X_pca"

    metric_builders: Mapping[str, Callable[[str], Callable[[ad.AnnData], float]]] = {
        "ilisi": lambda b: (
            lambda a: metric_ilisi(
                a,
                batch_key=b,
                use_rep=_pick_rep(a),
            )
        ),
        "asw_batch": lambda b: (
            lambda a: metric_asw_batch(
                a,
                batch_key=b,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "asw_label": lambda b: (
            lambda a: metric_asw_label(
                a,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "kbet": lambda b: (
            lambda a: metric_kbet(
                a,
                batch_key=b,
                use_rep=_pick_rep(a),
            )
        ),
        "kbet_pg": lambda b: (
            lambda a: metric_kbet_per_label(
                a,
                batch_key=b,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "kbet_pg_label": lambda b: (
            lambda a: metric_kbet_per_label(
                a,
                batch_key=b,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "graph_connectivity": lambda b: (
            lambda a: metric_graph_connectivity(
                a,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "clisi": lambda b: (
            lambda a: metric_clisi(
                a,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
        "cilisi": lambda b: (
            lambda a: metric_clisi(
                a,
                label_key=label_key,
                use_rep=_pick_rep(a),
            )
        ),
    }

    metrics: dict[str, Callable[[ad.AnnData], float]] = {}
    for name in metric_names:
        if name not in metric_builders:
            raise ValueError(f"Unknown builtin metric: `{name}`")
        metrics[name] = metric_builders[name](batch_key)
    return metrics


def _run_scanorama_for_batch_key(
    adata_qc: ad.AnnData,
    cfg: PipelineConfig,
    norm_cfg: NormalizationConfig,
) -> ad.AnnData:
    """
    Prepare per-batch AnnData objects and run Scanorama.

    Args:
        adata_qc: AnnData after QC.
        cfg: Pipeline configuration.
        norm_cfg: Normalization configuration.

    Returns:
        AnnData after Scanorama integration.
    """
    batches = list(adata_qc.obs[cfg.batch_key].astype(str).unique())
    if len(batches) < 2:
        raise ValueError("Scanorama requires at least 2 batches.")
    adatas_by_batch = [
        adata_qc[adata_qc.obs[cfg.batch_key].astype(str) == b].copy() for b in batches
    ]
    return run_scanorama_workflow(
        adatas_by_batch=adatas_by_batch,
        cfg=norm_cfg,
        batch_labels=batches,
        batch_key=cfg.batch_key,
    )


def _build_method_handlers(
    adata_qc: ad.AnnData,
    cfg: PipelineConfig,
    norm_cfg: NormalizationConfig,
) -> Mapping[str, Callable[[], ad.AnnData]]:
    """
    Build method registry for batch-correction workflows.

    Args:
        adata_qc: AnnData after QC.
        cfg: Pipeline configuration.
        norm_cfg: Normalization configuration.

    Returns:
        Mapping ``method_name -> zero-argument runner``.
    """
    return {
        "no_correction": lambda: run_no_correction_workflow(adata_qc, norm_cfg),
        "bbknn": lambda: run_bbknn_workflow(adata_qc, norm_cfg, batch_key=cfg.batch_key),
        "harmony": lambda: run_harmony_workflow(adata_qc, norm_cfg, batch_key=cfg.batch_key),
        "scanorama": lambda: _run_scanorama_for_batch_key(adata_qc, cfg, norm_cfg),
    }


def run_pipeline(
    adata: ad.AnnData,
    cfg: PipelineConfig = PipelineConfig(),
    qc_cfg: QCConfig = QCConfig(),
    norm_cfg: NormalizationConfig = NormalizationConfig(),
    custom_metrics: Mapping[str, Callable[[ad.AnnData], float]] | None = None,
) -> PipelineResult:
    """
    Unified pipeline entrypoint:
    QC -> normalization -> batch correction -> metric evaluation.

    Args:
        adata: Input AnnData.
        cfg: Pipeline config (batch key, methods, metrics, and flags).
        qc_cfg: QC filtering configuration.
        norm_cfg: Normalization/embedding configuration.
        custom_metrics: Additional metrics merged with builtin metrics.

    Returns:
        PipelineResult containing QC output, corrected objects, and metric table.
    """
    if cfg.batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{cfg.batch_key}` is missing in adata.obs.")

    adata_qc = _run_qc(adata, cfg.batch_key, qc_cfg, per_batch=cfg.run_qc_per_batch)
    method_handlers = _build_method_handlers(adata_qc, cfg, norm_cfg)

    corrected: dict[str, ad.AnnData] = {}
    for method in cfg.methods:
        if method not in method_handlers:
            raise ValueError(f"Unknown batch-correction method: `{method}`")
        corrected[method] = method_handlers[method]()

    metrics = _build_default_metrics(
        cfg.metrics,
        batch_key=cfg.batch_key,
        label_key=cfg.label_key,
    )
    if custom_metrics:
        metrics = {**metrics, **custom_metrics}
    metrics_table = evaluate_methods(corrected, metrics)
    return PipelineResult(adata_qc=adata_qc, corrected=corrected, metrics_table=metrics_table)
