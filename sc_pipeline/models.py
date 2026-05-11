from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import pandas as pd


@dataclass(frozen=True)
class QCConfig:
    min_genes: int = 200            
    max_genes: int = 8000           
    min_cells_per_gene: int = 3
    max_pct_mt: float = 20.0
    mt_prefix: str = "MT-"          
    run_scrublet: bool = True

@dataclass(frozen=True)
class SubsampleConfig:              
    n_cells: int = 120000
    stratify_by: tuple[str, ...] = ("cell_type", "batch")
    random_state: int = 42

@dataclass(frozen=True)
class NormalizationConfig:          
    target_sum: float = 1e4
    n_pcs: int = 40
    leiden_resolution: float = 1.0
    batch_key: str | None = None
    subset_hvg: bool = True

@dataclass(frozen=True)
class ScVIConfig:                   
    n_latent: int = 30
    n_layers: int = 2
    dropout_rate: float = 0.1
    n_epochs: int = 400
    batch_size: int = 1024
    early_stopping: bool = True

@dataclass(frozen=True)
class ScGPTConfig:                  
    model_dir: str = "models/scgpt"
    n_hvgs: int = 2000
    batch_size: int = 64

@dataclass(frozen=True)
class GeneformerConfig:             
    model_dir: str = "models/geneformer"
    n_hvgs: int = 2000
    forward_batch_size: int = 200
    nproc: int = 4

@dataclass(frozen=True)
class PipelineConfig:
    batch_key: str = "batch"
    label_key: str = "cell_type"    
    run_qc_per_batch: bool = True
    methods: tuple[str, ...] = ("no_correction", "combat", "scvi", "scanvi", "scgpt", "geneformer")
    metrics: tuple[str, ...] = ("ilisi", "kbet", "bras", "pcr_comparison", "clisi", "isolated_labels_f1", "hvg_overlap", "ari", "nmi")
    save_intermediate: bool = True
    scgpt_cfg: ScGPTConfig = ScGPTConfig()         
    geneformer_cfg: GeneformerConfig = GeneformerConfig()  
    scvi_cfg: ScVIConfig = ScVIConfig()             

@dataclass
class PipelineResult:
    adata_qc: ad.AnnData
    corrected: dict[str, ad.AnnData]
    metrics_table: pd.DataFrame
