from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import pandas as pd


@dataclass(frozen=True)
class QCConfig:
    min_genes: int = 500
    max_genes: int = 6000
    min_cells_per_gene: int = 3
    max_pct_mt: float = 20.0
    mt_prefix: str = "mt-"
    run_scrublet: bool = True


@dataclass(frozen=True)
class NormalizationConfig:
    target_sum: float = 1e4
    n_pcs: int = 40
    leiden_resolution: float = 1.0
    batch_key: str | None = None
    subset_hvg: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    batch_key: str = "batch"
    label_key: str = "leiden"
    run_qc_per_batch: bool = True
    methods: tuple[str, ...] = ("no_correction", "bbknn", "harmony", "scanorama")
    metrics: tuple[str, ...] = ("ilisi", "asw_batch")
    save_intermediate: bool = True


@dataclass
class PipelineResult:
    adata_qc: ad.AnnData
    corrected: dict[str, ad.AnnData]
    metrics_table: pd.DataFrame
