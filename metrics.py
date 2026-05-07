from __future__ import annotations

from collections.abc import Callable, Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults
from scib.metrics.lisi import lisi_graph_py
from sklearn.neighbors import NearestNeighbors


def plot_batch_fraction(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    batch_key: str = "batch",
    figsize: tuple[int, int] = (6, 3),
) -> None:
    """
    Plot stacked batch fractions per cluster.

    Args:
        adata: AnnData with cluster and batch columns in ``obs``.
        cluster_key: Cluster column (typically ``leiden``).
        batch_key: Batch column.
        figsize: Figure size.
    """
    batch_sum = (
        adata.obs[[cluster_key, batch_key]]
        .groupby([cluster_key, batch_key], as_index=False)
        .value_counts()
        .pivot(index=cluster_key, columns=batch_key, values="count")
        .fillna(0)
    )
    batch_frac = batch_sum.div(batch_sum.sum(axis=1), axis=0)
    batch_frac.plot.bar(stacked=True, figsize=figsize)
    plt.ylabel("Batch distribution")
    plt.xlabel("Cluster")
    plt.tight_layout()


def ilisi_summary_from_embedding(
    adata: ad.AnnData,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
    perplexity: int = 30,
) -> dict[str, np.ndarray | float]:
    """
    Compute raw per-cell iLISI from an embedding (distance-based kNN).

    Args:
        adata: AnnData with embedding in ``obsm``.
        batch_key: Batch column in ``obs``.
        use_rep: Embedding key in ``obsm``.
        n_neighbors: Number of kNN neighbors.
        perplexity: LISI perplexity parameter.

    Returns:
        Dict with per-cell scores and summary statistics.
    """
    if scib_metrics is None or NeighborsResults is None:
        raise ImportError("Install `scib-metrics` to compute iLISI.")

    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding `{use_rep}` is missing in adata.obsm.")

    x = np.asarray(adata.obsm[use_rep])
    batches = adata.obs[batch_key].to_numpy()

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    neighbors = NeighborsResults(indices=indices, distances=distances)
    scores = scib_metrics.ilisi_knn(
        X=neighbors,
        batches=batches,
        perplexity=perplexity,
        scale=False,
    )
    scores = np.asarray(scores).ravel()

    return {
        "scores": scores,
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "q05": float(np.percentile(scores, 5)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "q95": float(np.percentile(scores, 95)),
    }


def build_neighbors_from_embedding(
    adata: ad.AnnData,
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
) -> NeighborsResults:
    """
    Build ``NeighborsResults`` from an embedding for scib-metrics.

    Args:
        adata: AnnData with embedding in ``obsm``.
        use_rep: Embedding key in ``obsm``.
        n_neighbors: Number of neighbors.

    Returns:
        ``NeighborsResults`` object (indices + distances).
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding `{use_rep}` is missing in adata.obsm.")
    x = np.asarray(adata.obsm[use_rep])
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    return NeighborsResults(indices=indices, distances=distances)


def build_neighbors_from_graph(
    adata: ad.AnnData,
    n_neighbors: int = 90,
) -> NeighborsResults:
    """
    Build ``NeighborsResults`` from precomputed ``adata.obsp['distances']``.
    """
    if "distances" not in adata.obsp:
        raise KeyError("`distances` is missing in adata.obsp.")
    d = adata.obsp["distances"].tocsr()
    n_cells = d.shape[0]
    k = min(n_neighbors, max(1, d.shape[1] - 1))
    indices = np.empty((n_cells, k), dtype=int)
    distances = np.empty((n_cells, k), dtype=float)

    for i in range(n_cells):
        row_start, row_end = d.indptr[i], d.indptr[i + 1]
        row_idx = d.indices[row_start:row_end]
        row_val = d.data[row_start:row_end]
        mask = row_idx != i
        row_idx = row_idx[mask]
        row_val = row_val[mask]
        if row_idx.size == 0:
            # fallback for isolated cell
            row_idx = np.array([i], dtype=int)
            row_val = np.array([0.0], dtype=float)
        order = np.argsort(row_val)[:k]
        pick_idx = row_idx[order]
        pick_val = row_val[order]
        if pick_idx.size < k:
            pad_n = k - pick_idx.size
            pick_idx = np.concatenate([pick_idx, np.repeat(pick_idx[-1], pad_n)])
            pick_val = np.concatenate([pick_val, np.repeat(pick_val[-1], pad_n)])
        indices[i, :] = pick_idx
        distances[i, :] = pick_val

    return NeighborsResults(indices=indices, distances=distances)


def metric_asw_batch(
    adata: ad.AnnData,
    batch_key: str = "batch",
    label_key: str = "leiden",
    use_rep: str = "X_pca",
) -> float:
    """
    Compute ASW batch score (OpenProblems-style ``asw_batch``).

    Args:
        adata: AnnData with embedding.
        batch_key: Batch column in ``obs``.
        label_key: Biological label/cluster column in ``obs``.
        use_rep: Embedding key in ``obsm``.

    Returns:
        ASW batch value.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding `{use_rep}` is missing in adata.obsm.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{batch_key}` is missing in adata.obs.")
    if label_key not in adata.obs.columns:
        raise KeyError(f"Column `{label_key}` is missing in adata.obs.")
    x = np.asarray(adata.obsm[use_rep])
    batches = adata.obs[batch_key].astype(str).to_numpy()
    labels = adata.obs[label_key].astype(str).to_numpy()
    return float(scib_metrics.silhouette_batch(X=x, labels=labels, batch=batches))


def metric_asw_label(
    adata: ad.AnnData,
    label_key: str = "leiden",
    use_rep: str = "X_pca",
) -> float:
    """
    Compute ASW label score (OpenProblems-style ``asw_label``).

    Args:
        adata: AnnData with embedding.
        label_key: Biological label/cluster column in ``obs``.
        use_rep: Embedding key in ``obsm``.

    Returns:
        ASW label value.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"Embedding `{use_rep}` is missing in adata.obsm.")
    if label_key not in adata.obs.columns:
        raise KeyError(f"Column `{label_key}` is missing in adata.obs.")
    x = np.asarray(adata.obsm[use_rep])
    labels = adata.obs[label_key].astype(str).to_numpy()
    return float(scib_metrics.silhouette_label(X=x, labels=labels))


def metric_kbet(
    adata: ad.AnnData,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
    alpha: float = 0.05,
) -> float:
    """
    Compute kBET batch-mixing metric (OpenProblems ``kbet``).
    """
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{batch_key}` is missing in adata.obs.")
    neighbors = build_neighbors_from_embedding(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    batches = adata.obs[batch_key].astype(str).to_numpy()
    score = scib_metrics.kbet(X=neighbors, batches=batches, alpha=alpha)
    # Some versions return tuple(score, details).
    if isinstance(score, tuple):
        score = score[0]
    return float(score)


def metric_kbet_per_label(
    adata: ad.AnnData,
    batch_key: str = "batch",
    label_key: str = "leiden",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
    alpha: float = 0.05,
) -> float:
    """
    Compute kBET-per-label metric (``kbet_pg`` / ``kbet_pg_label``).
    """
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{batch_key}` is missing in adata.obs.")
    if label_key not in adata.obs.columns:
        raise KeyError(f"Column `{label_key}` is missing in adata.obs.")
    neighbors = build_neighbors_from_embedding(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    batches = adata.obs[batch_key].astype(str).to_numpy()
    labels = adata.obs[label_key].astype(str).to_numpy()
    score = scib_metrics.kbet_per_label(X=neighbors, batches=batches, labels=labels, alpha=alpha)
    if isinstance(score, tuple):
        score = score[0]
    return float(score)


def metric_graph_connectivity(
    adata: ad.AnnData,
    label_key: str = "leiden",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
) -> float:
    """
    Compute graph connectivity metric (OpenProblems ``graph_connectivity``).
    """
    if label_key not in adata.obs.columns:
        raise KeyError(f"Column `{label_key}` is missing in adata.obs.")
    neighbors = build_neighbors_from_embedding(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    labels = adata.obs[label_key].astype(str).to_numpy()
    return float(scib_metrics.graph_connectivity(X=neighbors, labels=labels))


def metric_clisi(
    adata: ad.AnnData,
    label_key: str = "leiden",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
    perplexity: int | None = None,
) -> float:
    """
    Compute cLISI (OpenProblems-style graph-based score scaled to [0, 1]).
    """
    if label_key not in adata.obs.columns:
        raise KeyError(f"Column `{label_key}` is missing in adata.obs.")

    try:
        clisi_scores = lisi_graph_py(
            adata=adata,
            obs_key=label_key,
            n_neighbors=n_neighbors,
            perplexity=perplexity,
            subsample=None,
            n_cores=1,
            verbose=False,
        )
        clisi_raw = float(np.nanmedian(clisi_scores))
    except Exception:
        # Fallback when scib's external knn_graph binary is unavailable
        # (e.g. Exec format error on some environments).
        if "distances" in adata.obsp:
            neighbors = build_neighbors_from_graph(adata, n_neighbors=n_neighbors)
        else:
            neighbors = build_neighbors_from_embedding(
                adata,
                use_rep=use_rep,
                n_neighbors=n_neighbors,
            )
        labels = adata.obs[label_key].astype(str).to_numpy()
        clisi_raw = float(
            scib_metrics.clisi_knn(
                X=neighbors,
                labels=labels,
                perplexity=perplexity,
                scale=False,
            )
        )
    n_labels = int(adata.obs[label_key].astype(str).nunique())
    if n_labels <= 1:
        return 0.0
    return float((n_labels - clisi_raw) / (n_labels - 1))


def metric_ilisi(
    adata: ad.AnnData,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    n_neighbors: int = 90,
    perplexity: int | None = None,
) -> float:
    """
    Compute iLISI (OpenProblems-style graph-based score scaled to [0, 1]).

    Args:
        adata: AnnData with neighbors graph in ``uns``/``obsp``.
        batch_key: Batch column.
        use_rep: Unused, kept for backward-compatible API.
        n_neighbors: Number of neighbors for iLISI.
        perplexity: iLISI perplexity parameter.

    Returns:
        Scaled iLISI score.
    """
    _ = use_rep
    # OpenProblems uses lisi_graph_py; fallback to graph neighbors if compiled binary fails.
    try:
        ilisi_scores = lisi_graph_py(
            adata=adata,
            obs_key=batch_key,
            n_neighbors=n_neighbors,
            perplexity=perplexity,
            subsample=None,
            n_cores=1,
            verbose=False,
        )
    except Exception:
        neighbors = build_neighbors_from_graph(adata, n_neighbors=n_neighbors)
        batches = adata.obs[batch_key].astype(str).to_numpy()
        ilisi_scores = scib_metrics.ilisi_knn(
            X=neighbors,
            batches=batches,
            perplexity=perplexity,
            scale=False,
        )
    ilisi_raw = float(np.nanmedian(ilisi_scores))
    n_batches = int(adata.obs[batch_key].astype(str).nunique())
    if n_batches <= 1:
        return 0.0
    return float((ilisi_raw - 1.0) / (n_batches - 1.0))


def metric_bras(
    adata: ad.AnnData,
    batch_key: str = "batch",
    label_key: str = "cell_type",
    use_rep: str = "X_pca",
) -> float:
    return float(scib_metrics.silhouette_batch(
        X=np.asarray(adata.obsm[use_rep]),
        labels=adata.obs[label_key].astype(str).to_numpy(),
        batch=adata.obs[batch_key].astype(str).to_numpy(),
        rescale=True,
    ))


def metric_pcr_comparison(
    adata_pre: ad.AnnData,
    adata_post: ad.AnnData,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
) -> float:
    return float(scib_metrics.pcr_comparison(
        adata_pre=adata_pre,
        adata_post=adata_post,
        covariate=batch_key,
        embed=use_rep,
    ))


def metric_isolated_labels_f1(
    adata: ad.AnnData,
    label_key: str = "cell_type",
    batch_key: str = "batch",
    use_rep: str = "X_pca",
) -> float:
    return float(scib_metrics.isolated_labels(
        adata, label_key=label_key, batch_key=batch_key,
        embed=use_rep, cluster=True,
    ))


def metric_hvg_overlap(
    adata_pre: ad.AnnData,
    adata_post: ad.AnnData,
    batch_key: str = "batch",
    n_hvgs: int = 2000,
) -> float:
    return float(scib_metrics.hvg_overlap(
        adata_pre, adata_post, batch=batch_key, n_hvg=n_hvgs
    ))


def metric_ari(
    adata: ad.AnnData,
    label_key: str = "cell_type",
    cluster_key: str = "leiden",
) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(
        adata.obs[label_key].astype(str),
        adata.obs[cluster_key].astype(str),
    ))


def metric_nmi(
    adata: ad.AnnData,
    label_key: str = "cell_type",
    cluster_key: str = "leiden",
) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(
        adata.obs[label_key].astype(str),
        adata.obs[cluster_key].astype(str),
    ))

def evaluate_methods(
    method_to_adata: Mapping[str, ad.AnnData],
    metrics: Mapping[str, Callable[[ad.AnnData], float]],
) -> pd.DataFrame:
    """
    Evaluate arbitrary methods on an arbitrary metric set.

    Args:
        method_to_adata: Mapping method name -> AnnData result.
        metrics: Mapping metric name -> callable ``AnnData -> float``.

    Returns:
        DataFrame with one row per method and one column per metric.
    """
    rows: list[dict[str, float | str]] = []
    for method_name, method_adata in method_to_adata.items():
        row: dict[str, float | str] = {"method": method_name}
        for metric_name, metric_fn in metrics.items():
            row[metric_name] = float(metric_fn(method_adata))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ilisi_boxplot(
    ilisi_scores: Mapping[str, np.ndarray],
    figsize: tuple[int, int] = (7, 4),
) -> None:
    """
    Boxplot per-cell iLISI distributions across methods.
    Args:
        ilisi_scores: Mapping method_name -> per-cell iLISI array.
        figsize: Figure size.
    """
    plot_df = pd.DataFrame({
        "method": np.concatenate([[k] * len(v) for k, v in ilisi_scores.items()]),
        "iLISI":  np.concatenate(list(ilisi_scores.values())),
    })
    plt.figure(figsize=figsize)
    plot_df.boxplot(column="iLISI", by="method", grid=False)
    plt.suptitle("")
    plt.title("iLISI per-cell distribution")
    plt.ylabel("iLISI")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.tight_layout()


def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    method_col: str = "method",
    figsize: tuple[int, int] = (12, 5),
    title: str = "Batch correction benchmark",
) -> None:
    import seaborn as sns

    df = metrics_df.set_index(method_col)
    plt.figure(figsize=figsize)
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()


def plot_metrics_barplot(
    metrics_df: pd.DataFrame,
    metric_col: str,
    method_col: str = "method",
    figsize: tuple[int, int] = (8, 4),
    title: str | None = None,
) -> None:
    df = metrics_df.sort_values(metric_col, ascending=False)
    plt.figure(figsize=figsize)
    plt.bar(df[method_col], df[metric_col], color="steelblue", edgecolor="white")
    plt.ylabel(metric_col)
    plt.xlabel("")
    plt.title(title or metric_col)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()


def plot_batch_fraction(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    batch_key: str = "batch",
    figsize: tuple[int, int] = (6, 3),
) -> None:
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Column `{cluster_key}` is missing in adata.obs.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column `{batch_key}` is missing in adata.obs.")
        
    batch_counts = (
        adata.obs
        .groupby([cluster_key, batch_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    batch_frac = batch_counts.div(batch_counts.sum(axis=1), axis=0)

    batch_frac.plot.bar(stacked=True, figsize=figsize, colormap="tab20")
    plt.ylabel("Batch fraction")
    plt.xlabel(cluster_key)
    plt.title(f"Batch distribution per {cluster_key}")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title=batch_key)
    plt.tight_layout()


def aggregate_pseudobulk(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    donor_key: str = "donor_id",
    output_dir: str = "results/deseq2",
) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    for ct in adata.obs[cell_type_key].unique():
        sub = adata[adata.obs[cell_type_key] == ct]
        pseudo = (
            sub.obs[[donor_key]]
            .assign(idx=range(len(sub)))
            .groupby(donor_key)["idx"]
            .apply(lambda idxs: np.array(
                sub[idxs.values].X.sum(axis=0)
            ).ravel())
        )
        pd.DataFrame(
            pseudo.tolist(),
            index=pseudo.index,
            columns=sub.var_names
        ).T.to_csv(f"{output_dir}/pseudobulk_{ct.replace(' ', '_')}.csv")
