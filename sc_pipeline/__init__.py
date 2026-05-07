"""Modular single-cell pipeline for batch-effect correction benchmarking."""

from .batch_correction import (
    run_combat_workflow,
    run_geneformer_workflow,
    run_no_correction_workflow,
    run_scanvi_workflow,
    run_scgpt_workflow,
    run_scvi_workflow,
)
from .metrics import (
    aggregate_pseudobulk,
    evaluate_methods,
    ilisi_summary_from_embedding,
    metric_ari,
    metric_asw_batch,
    metric_asw_label,
    metric_bras,
    metric_clisi,
    metric_graph_connectivity,
    metric_hvg_overlap,
    metric_ilisi,
    metric_isolated_labels_f1,
    metric_kbet,
    metric_kbet_per_label,
    metric_nmi,
    metric_pcr_comparison,
    plot_batch_fraction,
    plot_ilisi_boxplot,
    plot_metrics_barplot,
    plot_metrics_heatmap,
)
from .models import (
    GeneformerConfig,
    NormalizationConfig,
    PipelineConfig,
    PipelineResult,
    QCConfig,
    ScGPTConfig,
    ScVIConfig,
    SubsampleConfig,
)
from .normalization import (
    normalize_batches_separately,
    preprocess_for_integration,
    run_basic_scanpy_embedding,
)
from .pipeline import run_pipeline
from .qc import filter_adata, stratified_subsample

__all__ = [
    # ── Models ────────────────────────────────────────────────────────
    "GeneformerConfig",
    "NormalizationConfig",
    "PipelineConfig",
    "PipelineResult",
    "QCConfig",
    "ScGPTConfig",
    "ScVIConfig",
    "SubsampleConfig",
    # ── QC ────────────────────────────────────────────────────────────
    "filter_adata",
    "stratified_subsample",
    # ── Normalization ─────────────────────────────────────────────────
    "normalize_batches_separately",
    "preprocess_for_integration",
    "run_basic_scanpy_embedding",
    # ── Batch correction ──────────────────────────────────────────────
    "run_no_correction_workflow",
    "run_combat_workflow",
    "run_scvi_workflow",
    "run_scanvi_workflow",
    "run_scgpt_workflow",
    "run_geneformer_workflow",
    # ── Metrics ───────────────────────────────────────────────────────
    "evaluate_methods",
    "ilisi_summary_from_embedding",
    "metric_ilisi",
    "metric_kbet",
    "metric_kbet_per_label",
    "metric_bras",
    "metric_pcr_comparison",
    "metric_clisi",
    "metric_isolated_labels_f1",
    "metric_hvg_overlap",
    "metric_ari",
    "metric_nmi",
    "metric_asw_batch",
    "metric_asw_label",
    "metric_graph_connectivity",
    "aggregate_pseudobulk",
    # ── Visualization ─────────────────────────────────────────────────
    "plot_batch_fraction",
    "plot_ilisi_boxplot",
    "plot_metrics_heatmap",
    "plot_metrics_barplot",
    # ── Pipeline ──────────────────────────────────────────────────────
    "run_pipeline",
]
