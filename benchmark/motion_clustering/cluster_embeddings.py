"""
Phase 2: Hierarchical clustering with automatic optimal k selection.

Uses parallel computation for metric evaluation across k values.
Adapted for 1052-dimensional motion statistical features.

Metrics used:
- Silhouette Score: Measures cluster cohesion and separation (higher is better)
- Davies-Bouldin Index: Measures cluster similarity (lower is better)

Note: Calinski-Harabasz Index was excluded because it monotonically decreases 
with k, always favoring fewer clusters regardless of actual separation quality.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt
import pandas as pd
import json

from config import EMBEDDINGS_DIR, RESULTS_DIR, K_RANGE, N_JOBS, LINKAGE_METHOD


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed motion statistical features."""
    embeddings = np.load(EMBEDDINGS_DIR / "motion_embeddings.npy")
    motion_ids = np.load(EMBEDDINGS_DIR / "motion_ids.npy", allow_pickle=True)
    print(f"Loaded motion embeddings: {embeddings.shape}")
    print(f"Loaded {len(motion_ids)} motion IDs")
    return embeddings, motion_ids


def compute_linkage_matrix(embeddings: np.ndarray, method: str = LINKAGE_METHOD) -> np.ndarray:
    """
    Compute hierarchical clustering linkage matrix.
    
    Args:
        embeddings: (N, D) feature matrix
        method: Linkage method ('ward', 'complete', 'average', 'single')
        
    Returns:
        Linkage matrix Z
    """
    print(f"Computing linkage matrix ({method} method)...")
    
    # For large datasets, use fastcluster if available
    try:
        import fastcluster
        Z = fastcluster.linkage(embeddings, method=method)
        print("Used fastcluster for faster computation")
    except ImportError:
        Z = linkage(embeddings, method=method)
        print("Used scipy.cluster.hierarchy.linkage")
    
    return Z


def compute_metrics_for_k(embeddings: np.ndarray, Z: np.ndarray, k: int) -> dict:
    """
    Compute clustering metrics for a specific k value.
    
    Args:
        embeddings: (N, D) feature matrix
        Z: Linkage matrix
        k: Number of clusters
        
    Returns:
        Dict with metric values
    """
    # Get cluster labels by cutting the dendrogram
    labels = fcluster(Z, k, criterion='maxclust')
    
    # Compute Silhouette Score
    try:
        # Use sample for silhouette if too many points (faster)
        if len(embeddings) > 10000:
            np.random.seed(42)  # For reproducibility
            sample_idx = np.random.choice(len(embeddings), 10000, replace=False)
            sil_score = silhouette_score(embeddings[sample_idx], labels[sample_idx])
        else:
            sil_score = silhouette_score(embeddings, labels)
    except Exception:
        sil_score = np.nan
    
    # Compute Davies-Bouldin Index
    try:
        db_score = davies_bouldin_score(embeddings, labels)
    except Exception:
        db_score = np.nan
    
    # Count samples per cluster
    unique, counts = np.unique(labels, return_counts=True)
    min_cluster_size = counts.min()
    max_cluster_size = counts.max()
    
    return {
        'k': k,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'min_cluster_size': int(min_cluster_size),
        'max_cluster_size': int(max_cluster_size),
        'n_clusters_actual': len(unique),
    }


def parallel_metric_sweep(
    embeddings: np.ndarray, 
    Z: np.ndarray, 
    k_range: tuple[int, int],
    n_jobs: int = N_JOBS
) -> pd.DataFrame:
    """
    Compute clustering metrics for all k values in parallel.
    
    Args:
        embeddings: (N, D) feature matrix
        Z: Linkage matrix
        k_range: (k_min, k_max) range
        n_jobs: Number of parallel workers
        
    Returns:
        DataFrame with metrics for each k
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))
    
    print(f"Computing metrics for k={k_min} to k={k_max} using {n_jobs} cores...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_metrics_for_k)(embeddings, Z, k) 
        for k in k_values
    )
    
    df = pd.DataFrame(results)
    return df


def find_optimal_k(metrics_df: pd.DataFrame) -> dict:
    """
    Find optimal k using Silhouette Score and Davies-Bouldin Index.
    
    Returns dict with:
    - optimal_k: Combined score optimal
    - optimal_by_metric: Dict of metric -> optimal k for that metric
    """
    df = metrics_df.copy()
    
    # Normalize metrics to 0-1 range
    # For Silhouette: higher is better (normalize to 0-1)
    df['sil_norm'] = (df['silhouette'] - df['silhouette'].min()) / \
                     (df['silhouette'].max() - df['silhouette'].min() + 1e-8)
    
    # For Davies-Bouldin: lower is better (invert then normalize)
    df['db_norm'] = 1 - (df['davies_bouldin'] - df['davies_bouldin'].min()) / \
                        (df['davies_bouldin'].max() - df['davies_bouldin'].min() + 1e-8)
    
    # Combined score: average of normalized Silhouette and Davies-Bouldin
    df['combined_score'] = (df['sil_norm'] + df['db_norm']) / 2
    
    # Find optimal k for each metric
    best_sil_k = int(df.loc[df['silhouette'].idxmax(), 'k'])
    best_db_k = int(df.loc[df['davies_bouldin'].idxmin(), 'k'])
    best_combined_k = int(df.loc[df['combined_score'].idxmax(), 'k'])
    
    print(f"\nOptimal k analysis:")
    print(f"  Best Silhouette: k={best_sil_k} (score={df.loc[df['k']==best_sil_k, 'silhouette'].values[0]:.4f})")
    print(f"  Best Davies-Bouldin: k={best_db_k} (score={df.loc[df['k']==best_db_k, 'davies_bouldin'].values[0]:.4f})")
    print(f"  Combined optimal: k={best_combined_k}")
    
    # Also find local minima in Davies-Bouldin (often indicates natural clusters)
    db_values = df['davies_bouldin'].values
    k_values = df['k'].values
    
    local_minima = []
    for i in range(1, len(db_values) - 1):
        if db_values[i] < db_values[i-1] and db_values[i] < db_values[i+1]:
            local_minima.append((int(k_values[i]), float(db_values[i])))
    
    if local_minima:
        print(f"\n  Davies-Bouldin local minima:")
        for k, db in sorted(local_minima, key=lambda x: x[1])[:5]:
            print(f"    k={k}: DB={db:.4f}")
    
    return {
        'optimal_k': best_combined_k,
        'optimal_by_metric': {
            'silhouette': best_sil_k,
            'davies_bouldin': best_db_k,
            'combined': best_combined_k,
        },
        'db_local_minima': local_minima,
    }


def plot_dendrogram(Z: np.ndarray, output_path: Path, max_d: float = None):
    """Generate and save dendrogram visualization."""
    print("Generating dendrogram...")
    
    plt.figure(figsize=(20, 10))
    
    # Plot truncated dendrogram for readability
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=50,  # Show last 50 merges
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
    )
    
    if max_d:
        plt.axhline(y=max_d, c='r', lw=2, linestyle='--', label=f'Cut at {max_d:.2f}')
        plt.legend()
    
    plt.title('Hierarchical Clustering Dendrogram (Motion Features, Ward Linkage)')
    plt.xlabel('Sample Index (or cluster size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved dendrogram to {output_path}")


def plot_metrics(metrics_df: pd.DataFrame, optimal_k: int, output_path: Path):
    """Plot clustering metrics across k values."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Silhouette Score
    ax = axes[0]
    ax.plot(metrics_df['k'], metrics_df['silhouette'], 'g-', lw=2)
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (higher is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin Index
    ax = axes[1]
    ax.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'm-', lw=2)
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_title('Davies-Bouldin Index (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cluster Size Range
    ax = axes[2]
    ax.fill_between(metrics_df['k'], metrics_df['min_cluster_size'], 
                    metrics_df['max_cluster_size'], alpha=0.3, label='Size range')
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Cluster Size')
    ax.set_title('Cluster Size Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Motion-Based Clustering Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved metrics plot to {output_path}")


def save_cluster_assignments(
    motion_ids: np.ndarray,
    Z: np.ndarray,
    optimal_k: int,
    output_dir: Path
) -> np.ndarray:
    """
    Save cluster assignments for the optimal k.
    
    Returns:
        labels: (N,) array of cluster labels
    """
    labels = fcluster(Z, optimal_k, criterion='maxclust')
    
    # Save as JSON (motion_id -> cluster_id mapping)
    cluster_assignments = {
        str(motion_id): int(label) 
        for motion_id, label in zip(motion_ids, labels)
    }
    
    output_path = output_dir / f"cluster_assignments_k{optimal_k}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'optimal_k': optimal_k,
            'assignments': cluster_assignments
        }, f, indent=2)
    print(f"Saved cluster assignments to {output_path}")
    
    # Also save as numpy array
    np.save(output_dir / f"cluster_labels_k{optimal_k}.npy", labels)
    
    return labels


def main():
    """Main entry point for clustering."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    embeddings, motion_ids = load_embeddings()
    
    # Compute linkage matrix
    Z = compute_linkage_matrix(embeddings, method=LINKAGE_METHOD)
    
    # Save linkage matrix
    linkage_path = EMBEDDINGS_DIR / "linkage_matrix.npy"
    np.save(linkage_path, Z)
    print(f"Saved linkage matrix to {linkage_path}")
    
    # Parallel metric sweep
    metrics_df = parallel_metric_sweep(embeddings, Z, K_RANGE, N_JOBS)
    
    # Save metrics
    metrics_path = RESULTS_DIR / "cluster_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Find optimal k
    optimal_info = find_optimal_k(metrics_df)
    optimal_k = optimal_info['optimal_k']
    
    # Save optimal k info
    with open(RESULTS_DIR / "optimal_k_analysis.json", 'w') as f:
        json.dump(optimal_info, f, indent=2)
    
    # Get and save final cluster assignments
    labels = save_cluster_assignments(motion_ids, Z, optimal_k, RESULTS_DIR)
    
    # Generate visualizations
    plot_dendrogram(Z, RESULTS_DIR / "dendrogram.png")
    plot_metrics(metrics_df, optimal_k, RESULTS_DIR / "cluster_metrics.png")
    
    # Print cluster distribution
    print(f"\n{'='*60}")
    print(f"FINAL CLUSTERING RESULTS (k={optimal_k})")
    print("="*60)
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort by cluster size (descending)
    sorted_indices = np.argsort(-counts)
    for idx in sorted_indices:
        cluster_id = unique[idx]
        count = counts[idx]
        pct = 100 * count / len(labels)
        print(f"  Cluster {cluster_id:2d}: {count:5d} motions ({pct:5.1f}%)")
    
    print(f"\n  Total: {len(labels)} motions in {len(unique)} clusters")
    print(f"  Min cluster size: {counts.min()}")
    print(f"  Max cluster size: {counts.max()}")
    print(f"  Mean cluster size: {counts.mean():.1f}")


if __name__ == "__main__":
    main()
