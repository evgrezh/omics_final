#!/usr/bin/env python3
from __future__ import annotations
"""CLI entry point for running batch-correction benchmark on a .h5ad file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

from sc_pipeline.metrics import plot_batch_fraction, plot_metrics_heatmap
from sc_pipeline.models import (
    GeneformerConfig,
    NormalizationConfig,
    PipelineConfig,
    QCConfig,
    ScGPTConfig,
    ScVIConfig,
    SubsampleConfig,
)
from sc_pipeline.pipeline import run_pipeline
from sc_pipeline.qc import stratified_subsample


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    p = argparse.ArgumentParser(
        description="Modular single-cell batch-correction benchmark pipeline."
    )
    # ── Input / output ────────────────────────────────────────────────
    p.add_argument("--input-h5ad",  type=Path, required=True,  help="Path to input .h5ad")
    p.add_argument("--output-dir",  type=Path, required=True,  help="Directory for output files")

    # ── Data keys ─────────────────────────────────────────────────────
    p.add_argument("--batch-key",   type=str, default="batch",     help="Batch column in adata.obs")
    p.add_argument("--label-key",   type=str, default="cell_type", help="Cell type column in adata.obs")

    # ── Methods & metrics ─────────────────────────────────────────────
    p.add_argument(
        "--methods",
        type=str,
        default="no_correction,combat,scvi,scanvi,scgpt,geneformer",
        help="Comma-separated batch-correction methods.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="ilisi,kbet,bras,clisi,isolated_labels_f1,ari,nmi",
        help="Comma-separated builtin metrics.",
    )

    # ── QC & subsampling ──────────────────────────────────────────────
    p.add_argument(
        "--qc-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run QC independently per batch.",
    )
    p.add_argument(
        "--n-subsample",
        type=int,
        default=125_000,
        help="Target number of cells after stratified subsampling (0 = skip).",
    )

    # ── Foundation model paths ────────────────────────────────────────
    p.add_argument("--scgpt-dir",       type=str, default="models/scgpt",       help="Path to scGPT model weights.")
    p.add_argument("--geneformer-dir",  type=str, default="models/geneformer",  help="Path to Geneformer model weights.")

    # ── scVI hyperparameters ──────────────────────────────────────────
    p.add_argument("--scvi-n-latent",  type=int, default=30,  help="scVI latent dimensions.")
    p.add_argument("--scvi-n-epochs",  type=int, default=400, help="scVI max training epochs.")

    # ── Output control ────────────────────────────────────────────────
    p.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Do not save per-method .h5ad files.",
    )

    return p.parse_args()


def main() -> None:
    """Run pipeline from CLI and save all requested artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"[CLI] Reading {args.input_h5ad} ...")
    adata = sc.read_h5ad(args.input_h5ad)

    # ── Stratified subsample (опционально) ────────────────────────────
    if args.n_subsample > 0 and adata.n_obs > args.n_subsample:
        sub_cfg = SubsampleConfig(n_cells=args.n_subsample)
        adata = stratified_subsample(
            adata,
            n_cells=sub_cfg.n_cells,
            stratify_by=list(sub_cfg.stratify_by),
            random_state=sub_cfg.random_state,
        )

    # ── Configs ───────────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig(
        batch_key=args.batch_key,
        label_key=args.label_key,
        run_qc_per_batch=args.qc_per_batch,
        methods=tuple(x.strip() for x in args.methods.split(",") if x.strip()),
        metrics=tuple(x.strip() for x in args.metrics.split(",") if x.strip()),
        save_intermediate=not args.no_save_intermediate,
        scvi_cfg=ScVIConfig(
            n_latent=args.scvi_n_latent,
            n_epochs=args.scvi_n_epochs,
        ),
        scgpt_cfg=ScGPTConfig(
            model_dir=args.scgpt_dir,
        ),
        geneformer_cfg=GeneformerConfig(
            model_dir=args.geneformer_dir,
        ),
    )
    norm_cfg = NormalizationConfig(batch_key=args.batch_key)
    qc_cfg   = QCConfig()

    # ── Run pipeline ──────────────────────────────────────────────────
    result = run_pipeline(
        adata=adata,
        cfg=pipeline_cfg,
        qc_cfg=qc_cfg,
        norm_cfg=norm_cfg,
    )

    # ── Save outputs ──────────────────────────────────────────────────
    result.adata_qc.write_h5ad(args.output_dir / "adata_qc.h5ad")
    print(f"[CLI] Saved adata_qc.h5ad")

    if pipeline_cfg.save_intermediate:
        for method, corrected in result.corrected.items():
            corrected.write_h5ad(args.output_dir / f"adata_{method}.h5ad")
            plot_batch_fraction(
                corrected,
                batch_key=args.batch_key,
                cluster_key="leiden",
            )
            plt.savefig(args.output_dir / f"batch_fraction_{method}.png", dpi=150)
            plt.close()
            print(f"[CLI] Saved adata_{method}.h5ad + batch_fraction_{method}.png")

    # metrics_summary.csv
    result.metrics_table.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    print(f"[CLI] Saved metrics_summary.csv")

    # metrics heatmap
    plot_metrics_heatmap(
        result.metrics_table,
        title=f"Batch correction benchmark — {args.input_h5ad.stem}",
    )
    plt.savefig(args.output_dir / "metrics_heatmap.png", dpi=150)
    plt.close()
    print(f"[CLI] Saved metrics_heatmap.png")

    print(result.metrics_table.to_string(index=False))
    print(f"\n[CLI] Done. All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
