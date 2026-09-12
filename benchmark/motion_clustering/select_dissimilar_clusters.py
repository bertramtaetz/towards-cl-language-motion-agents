"""
Phase 3: Select maximally dissimilar clusters for continual learning experiments.

Uses greedy farthest-point algorithm to select clusters with maximum separation
in the motion feature space.
"""
import numpy as np
from pathlib import Path
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.manifold import TSNE
import seaborn as sns

from config import (
    EMBEDDINGS_DIR, RESULTS_DIR, JOINT_VECS_DIR,
    NUM_TASKS, MIN_CLUSTER_SIZE, 
    TRAIN_SAMPLES_PER_TASK, VAL_SAMPLES_PER_TASK, TEST_SAMPLES_PER_TASK,
    MIN_MOTION_LENGTH, MAX_MOTION_LENGTH
)


def load_data() -> tuple:
    """Load all required data for cluster selection."""
    # Motion embeddings
    embeddings = np.load(EMBEDDINGS_DIR / "motion_embeddings.npy")
    motion_ids = np.load(EMBEDDINGS_DIR / "motion_ids.npy", allow_pickle=True)
    
    # Find the most recent cluster assignments file
    assignment_files = list(RESULTS_DIR.glob("cluster_assignments_k*.json"))
    if not assignment_files:
        raise FileNotFoundError("No cluster assignment files found in results directory")
    
    # Get the latest one (or use optimal k from analysis)
    optimal_k_path = RESULTS_DIR / "optimal_k_analysis.json"
    if optimal_k_path.exists():
        with open(optimal_k_path, 'r') as f:
            optimal_info = json.load(f)
        optimal_k = optimal_info['optimal_k']
        assignment_path = RESULTS_DIR / f"cluster_assignments_k{optimal_k}.json"
    else:
        assignment_path = sorted(assignment_files)[-1]
    
    print(f"Loading cluster assignments from {assignment_path}")
    with open(assignment_path, 'r') as f:
        cluster_data = json.load(f)
    
    optimal_k = cluster_data['optimal_k']
    assignments = cluster_data['assignments']
    
    # Descriptions for labeling
    desc_path = EMBEDDINGS_DIR / "motion_descriptions.json"
    if desc_path.exists():
        with open(desc_path, 'r') as f:
            descriptions = json.load(f)
    else:
        descriptions = {}
    
    return embeddings, motion_ids, assignments, optimal_k, descriptions


def get_motion_length(motion_id: str) -> int | None:
    """Get the length (number of frames) of a motion sequence."""
    motion_path = JOINT_VECS_DIR / f"{motion_id}.npy"
    if motion_path.exists():
        motion = np.load(motion_path)
        return motion.shape[0]
    return None


def filter_motions_by_length(
    motion_ids: list[str],
    min_length: int = MIN_MOTION_LENGTH,
    max_length: int = MAX_MOTION_LENGTH
) -> set[str]:
    """
    Filter motions by sequence length.
    
    Returns set of motion IDs that pass the length filter.
    """
    valid_ids = set()
    
    for motion_id in motion_ids:
        length = get_motion_length(motion_id)
        if length is not None and min_length <= length <= max_length:
            valid_ids.add(motion_id)
    
    return valid_ids


def compute_cluster_centroids(
    embeddings: np.ndarray,
    motion_ids: np.ndarray,
    assignments: dict,
) -> tuple[dict, dict]:
    """
    Compute centroid embeddings for each cluster.
    
    Returns:
        centroids: dict of cluster_id -> centroid embedding
        cluster_motion_ids: dict of cluster_id -> list of motion IDs
    """
    # Create mapping from motion_id to embedding index
    id_to_idx = {str(mid): i for i, mid in enumerate(motion_ids)}
    
    # Group embeddings by cluster
    cluster_embeddings = {}
    cluster_motion_ids = {}
    
    for motion_id, cluster_id in assignments.items():
        if motion_id in id_to_idx:
            idx = id_to_idx[motion_id]
            cluster_id = int(cluster_id)
            
            if cluster_id not in cluster_embeddings:
                cluster_embeddings[cluster_id] = []
                cluster_motion_ids[cluster_id] = []
            
            cluster_embeddings[cluster_id].append(embeddings[idx])
            cluster_motion_ids[cluster_id].append(motion_id)
    
    # Compute centroids
    centroids = {}
    for cluster_id, embs in cluster_embeddings.items():
        centroids[cluster_id] = np.mean(embs, axis=0)
    
    return centroids, cluster_motion_ids


def greedy_farthest_point_selection(
    centroids: dict,
    cluster_motion_ids: dict,
    valid_motion_ids: set[str],
    n_select: int = NUM_TASKS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    distance_metric: str = 'euclidean'
) -> list[int]:
    """
    Select n_select clusters that are maximally dissimilar.
    
    Uses greedy farthest-point algorithm:
    1. Start with the cluster farthest from mean of all clusters
    2. Iteratively add the cluster that maximizes minimum distance to selected clusters
    
    Args:
        centroids: dict of cluster_id -> centroid embedding
        cluster_motion_ids: dict of cluster_id -> list of motion IDs
        valid_motion_ids: Set of motion IDs that pass length filter
        n_select: number of clusters to select
        min_cluster_size: minimum valid motions required in cluster
        distance_metric: 'euclidean' or 'cosine'
    
    Returns:
        List of selected cluster IDs
    """
    # Filter clusters by size (considering only valid motions)
    eligible_clusters = []
    cluster_valid_counts = {}
    
    for cluster_id, motion_ids in cluster_motion_ids.items():
        valid_count = sum(1 for mid in motion_ids if mid in valid_motion_ids)
        cluster_valid_counts[cluster_id] = valid_count
        
        if valid_count >= min_cluster_size:
            eligible_clusters.append(cluster_id)
    
    print(f"\nCluster eligibility (min {min_cluster_size} valid motions):")
    print(f"  Total clusters: {len(centroids)}")
    print(f"  Eligible clusters: {len(eligible_clusters)}")
    
    if len(eligible_clusters) < n_select:
        print(f"\nWarning: Only {len(eligible_clusters)} eligible clusters, need {n_select}")
        print("Top clusters by valid motion count:")
        sorted_clusters = sorted(cluster_valid_counts.items(), key=lambda x: -x[1])
        for cid, count in sorted_clusters[:10]:
            print(f"  Cluster {cid}: {count} valid motions")
        
        # Use top clusters even if below threshold
        eligible_clusters = [cid for cid, _ in sorted_clusters[:max(n_select, len(sorted_clusters))]]
    
    # Build centroid matrix for eligible clusters only
    cluster_ids = eligible_clusters
    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])
    
    # Compute pairwise distances
    if distance_metric == 'cosine':
        distances = cosine_distances(centroid_matrix)
    else:
        distances = euclidean_distances(centroid_matrix)
    
    # Greedy selection
    selected_indices = []
    
    # Start with the cluster that has maximum average distance to all others
    avg_distances = distances.mean(axis=1)
    first_idx = np.argmax(avg_distances)
    selected_indices.append(first_idx)
    
    # Iteratively select farthest point
    for _ in range(n_select - 1):
        # For each unselected point, compute min distance to selected points
        min_distances = np.full(len(cluster_ids), -np.inf)
        
        for i in range(len(cluster_ids)):
            if i not in selected_indices:
                min_dist = min(distances[i, j] for j in selected_indices)
                min_distances[i] = min_dist
        
        # Select the point with maximum min-distance (farthest from selected set)
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
    
    selected_cluster_ids = [cluster_ids[i] for i in selected_indices]
    
    # Print selection info
    print(f"\nSelected {n_select} clusters using farthest-point algorithm:")
    for i, cid in enumerate(selected_cluster_ids):
        valid_count = cluster_valid_counts.get(cid, 0)
        print(f"  {i+1}. Cluster {cid} ({valid_count} valid motions)")
    
    # Print pairwise distances between selected clusters
    print(f"\nPairwise {distance_metric} distances between selected clusters:")
    selected_matrix = distances[np.ix_(selected_indices, selected_indices)]
    total_dist = 0
    count = 0
    for i in range(n_select):
        for j in range(i+1, n_select):
            dist = selected_matrix[i, j]
            total_dist += dist
            count += 1
            print(f"  Cluster {selected_cluster_ids[i]} <-> {selected_cluster_ids[j]}: {dist:.4f}")
    
    print(f"\nAverage pairwise distance: {total_dist/count:.4f}")
    
    return selected_cluster_ids


def get_cluster_keywords(
    descriptions: dict,
    assignments: dict,
    cluster_id: int,
    top_k: int = 10
) -> list[str]:
    """
    Extract most common keywords/actions for a cluster.
    """
    from collections import Counter
    
    # Common words to ignore
    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'person', 'man', 'woman',
        'someone', 'something', 'their', 'his', 'her', 'its', 'they', 'them',
        'he', 'she', 'it', 'who', 'which', 'that', 'this', 'these', 'those'
    }
    
    words = Counter()
    
    for motion_id, cid in assignments.items():
        if int(cid) == cluster_id and motion_id in descriptions:
            for desc in descriptions[motion_id]:
                # Simple tokenization
                tokens = desc.lower().replace('.', '').replace(',', '').split()
                for token in tokens:
                    if token not in stopwords and len(token) > 2:
                        words[token] += 1
    
    return [word for word, count in words.most_common(top_k)]


def visualize_clusters(
    embeddings: np.ndarray,
    motion_ids: np.ndarray,
    assignments: dict,
    selected_clusters: list[int],
    output_path: Path
):
    """
    Create t-SNE visualization of clusters highlighting selected ones.
    """
    print("\nGenerating t-SNE visualization...")
    
    # Sample for visualization (t-SNE is slow for large datasets)
    n_samples = min(5000, len(embeddings))
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), n_samples, replace=False)
    
    sample_embeddings = embeddings[sample_idx]
    sample_ids = motion_ids[sample_idx]
    
    # Get cluster labels for samples
    labels = []
    for mid in sample_ids:
        mid_str = str(mid)
        if mid_str in assignments:
            cid = int(assignments[mid_str])
            if cid in selected_clusters:
                labels.append(f"Selected-{cid}")
            else:
                labels.append("Other")
        else:
            labels.append("Unknown")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    coords = tsne.fit_transform(sample_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Plot "Other" clusters first (gray, smaller)
    other_mask = np.array([l == "Other" for l in labels])
    plt.scatter(coords[other_mask, 0], coords[other_mask, 1], 
                c='lightgray', s=10, alpha=0.3, label='Other clusters')
    
    # Plot selected clusters with distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_clusters)))
    
    for i, cid in enumerate(selected_clusters):
        mask = np.array([l == f"Selected-{cid}" for l in labels])
        if mask.sum() > 0:
            plt.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[i]], s=30, alpha=0.7, 
                       label=f'Cluster {cid}', edgecolors='black', linewidth=0.5)
    
    plt.title('t-SNE Visualization of Motion-Based Clusters\n(Selected clusters highlighted)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def create_task_splits(
    assignments: dict,
    selected_clusters: list[int],
    cluster_keywords: dict,
    valid_motion_ids: set[str],
    output_path: Path,
    seed: int = 42
):
    """
    Create the final task split file for continual learning experiments.
    
    Creates balanced splits: 360 train + 100 val + 100 test per task.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_per_task = TRAIN_SAMPLES_PER_TASK + VAL_SAMPLES_PER_TASK + TEST_SAMPLES_PER_TASK
    
    # Map clusters to task names based on keywords
    task_names = []
    for cid in selected_clusters:
        keywords = cluster_keywords.get(cid, [])
        task_name = f"task_{cid}_" + "_".join(keywords[:3]) if keywords else f"task_{cid}"
        task_names.append(task_name)
    
    # Create splits
    splits = {
        'n_tasks': len(selected_clusters),
        'cluster_ids': selected_clusters,
        'task_names': task_names,
        'samples_per_task': {
            'train': TRAIN_SAMPLES_PER_TASK,
            'val': VAL_SAMPLES_PER_TASK,
            'test': TEST_SAMPLES_PER_TASK,
        },
        'tasks': {}
    }
    
    for i, cid in enumerate(selected_clusters):
        # Get all valid motions for this cluster
        task_motions = [
            mid for mid, assigned_cid in assignments.items()
            if int(assigned_cid) == cid and mid in valid_motion_ids
        ]
        
        # Shuffle and sample
        random.shuffle(task_motions)
        
        if len(task_motions) < total_per_task:
            print(f"Warning: Cluster {cid} has only {len(task_motions)} valid motions, need {total_per_task}")
            # Use all available, adjusting split ratios
            n_train = int(0.64 * len(task_motions))  # ~360/560
            n_val = int(0.18 * len(task_motions))    # ~100/560
            n_test = len(task_motions) - n_train - n_val
        else:
            n_train = TRAIN_SAMPLES_PER_TASK
            n_val = VAL_SAMPLES_PER_TASK
            n_test = TEST_SAMPLES_PER_TASK
        
        train_ids = task_motions[:n_train]
        val_ids = task_motions[n_train:n_train + n_val]
        test_ids = task_motions[n_train + n_val:n_train + n_val + n_test]
        
        splits['tasks'][f'task_{i+1}'] = {
            'cluster_id': cid,
            'name': task_names[i],
            'keywords': cluster_keywords.get(cid, []),
            'n_total': len(task_motions),
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'n_train': len(train_ids),
            'n_val': len(val_ids),
            'n_test': len(test_ids),
        }
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSaved task splits to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL TASK SPLITS FOR CONTINUAL LEARNING")
    print("="*60)
    for i, cid in enumerate(selected_clusters):
        task_info = splits['tasks'][f'task_{i+1}']
        print(f"\nTask {i+1}: Cluster {cid}")
        print(f"  Keywords: {', '.join(task_info['keywords'][:5])}")
        print(f"  Train/Val/Test: {task_info['n_train']}/{task_info['n_val']}/{task_info['n_test']}")
    
    return splits


def main():
    """Main entry point for dissimilar cluster selection."""
    # Load data
    embeddings, motion_ids, assignments, optimal_k, descriptions = load_data()
    
    print(f"Loaded {len(embeddings)} embeddings with k={optimal_k} clusters")
    
    # Filter motions by length
    print(f"\nFiltering motions by length ({MIN_MOTION_LENGTH}-{MAX_MOTION_LENGTH} frames)...")
    all_motion_ids = list(assignments.keys())
    valid_motion_ids = filter_motions_by_length(all_motion_ids)
    print(f"  Valid motions: {len(valid_motion_ids)} / {len(all_motion_ids)} ({100*len(valid_motion_ids)/len(all_motion_ids):.1f}%)")
    
    # Compute cluster centroids
    centroids, cluster_motion_ids = compute_cluster_centroids(
        embeddings, motion_ids, assignments
    )
    
    print(f"Computed centroids for {len(centroids)} clusters")
    
    # Select maximally dissimilar clusters
    selected_clusters = greedy_farthest_point_selection(
        centroids, 
        cluster_motion_ids,
        valid_motion_ids,
        n_select=NUM_TASKS, 
        min_cluster_size=MIN_CLUSTER_SIZE,
        distance_metric='euclidean'  # Use Euclidean for motion features
    )
    
    # Get keywords for each selected cluster
    cluster_keywords = {}
    print("\n" + "="*60)
    print("SELECTED CLUSTER ANALYSIS")
    print("="*60)
    
    for cid in selected_clusters:
        keywords = get_cluster_keywords(descriptions, assignments, cid, top_k=15)
        cluster_keywords[cid] = keywords
        
        n_total = len(cluster_motion_ids.get(cid, []))
        n_valid = sum(1 for mid in cluster_motion_ids.get(cid, []) if mid in valid_motion_ids)
        
        print(f"\nCluster {cid}:")
        print(f"  Total motions: {n_total}")
        print(f"  Valid motions (40-200 frames): {n_valid}")
        print(f"  Top keywords: {', '.join(keywords[:10])}")
    
    # Visualize
    visualize_clusters(
        embeddings, motion_ids, assignments, selected_clusters,
        RESULTS_DIR / f"selected_clusters_tsne_k{optimal_k}.png"
    )
    
    # Create task splits
    splits = create_task_splits(
        assignments, selected_clusters, cluster_keywords, valid_motion_ids,
        RESULTS_DIR / f"task_splits_k{optimal_k}.json"
    )
    
    # Save selected cluster info
    selection_info = {
        'optimal_k': optimal_k,
        'selected_cluster_ids': selected_clusters,
        'cluster_keywords': {str(k): v for k, v in cluster_keywords.items()},
        'selection_method': 'greedy_farthest_point',
        'distance_metric': 'euclidean',
        'min_cluster_size': MIN_CLUSTER_SIZE,
        'motion_length_filter': f'{MIN_MOTION_LENGTH}-{MAX_MOTION_LENGTH}',
    }
    
    with open(RESULTS_DIR / "cluster_selection_info.json", 'w') as f:
        json.dump(selection_info, f, indent=2)


if __name__ == "__main__":
    main()

