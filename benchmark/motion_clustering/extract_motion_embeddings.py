"""
Phase 1: Extract 1052-dimensional motion statistical features.

For each motion sequence, computes 4 statistics (mean, std, min, max) 
for each of the 263 motion dimensions, producing a fixed-length 
representation regardless of sequence length.

This approach captures:
- Central tendency (mean): typical pose/velocity during the motion
- Variability (std): how dynamic/varied the motion is
- Range (min, max): extreme poses/velocities reached
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import json

from config import (
    JOINT_VECS_DIR, EMBEDDINGS_DIR, TEXTS_DIR,
    MOTION_DIM, EMBEDDING_DIM, N_JOBS
)


def extract_motion_statistics(motion_path: Path) -> tuple[str, np.ndarray] | None:
    """
    Extract statistical features from a single motion file.
    
    For a motion of shape (T, 263), computes:
    - mean across time: (263,)
    - std across time: (263,)
    - min across time: (263,)
    - max across time: (263,)
    
    Concatenated to produce a (1052,) feature vector.
    
    Args:
        motion_path: Path to the .npy motion file
        
    Returns:
        Tuple of (motion_id, features) or None if extraction fails
    """
    try:
        motion = np.load(motion_path)  # Shape: (T, 263)
        
        if motion.shape[1] != MOTION_DIM:
            print(f"Warning: {motion_path.stem} has shape {motion.shape}, expected (T, {MOTION_DIM})")
            return None
        
        # Compute statistics across time axis
        mean_features = np.mean(motion, axis=0)  # (263,)
        std_features = np.std(motion, axis=0)    # (263,)
        min_features = np.min(motion, axis=0)    # (263,)
        max_features = np.max(motion, axis=0)    # (263,)
        
        # Concatenate to form 1052-dim feature vector
        # Order: [mean_0, mean_1, ..., mean_262, std_0, ..., std_262, min_0, ..., min_262, max_0, ..., max_262]
        features = np.concatenate([mean_features, std_features, min_features, max_features])
        
        assert features.shape[0] == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} features, got {features.shape[0]}"
        
        return motion_path.stem, features.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {motion_path}: {e}")
        return None


def parallel_feature_extraction(motion_paths: list[Path], n_jobs: int = N_JOBS) -> tuple[list[str], np.ndarray]:
    """
    Extract motion features from all files in parallel.
    
    Args:
        motion_paths: List of paths to motion .npy files
        n_jobs: Number of parallel workers
        
    Returns:
        motion_ids: List of motion IDs
        features: (N, 1052) feature matrix
    """
    print(f"Extracting motion features from {len(motion_paths)} files using {n_jobs} workers...")
    
    results = Parallel(n_jobs=n_jobs, prefer='threads', verbose=10)(
        delayed(extract_motion_statistics)(path) for path in motion_paths
    )
    
    # Filter out failed extractions and separate IDs from features
    valid_results = [r for r in results if r is not None]
    print(f"Successfully extracted features for {len(valid_results)} / {len(motion_paths)} motions")
    
    motion_ids = [r[0] for r in valid_results]
    features = np.stack([r[1] for r in valid_results], axis=0)
    
    return motion_ids, features


def normalize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features for clustering.
    
    Args:
        features: (N, D) feature matrix
        
    Returns:
        normalized: (N, D) normalized features
        mean: (D,) feature means
        std: (D,) feature stds
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def load_motion_descriptions() -> dict[str, list[str]]:
    """
    Load text descriptions for each motion (for cluster analysis).
    
    Returns:
        Dict mapping motion_id -> list of descriptions
    """
    print("Loading motion descriptions...")
    descriptions = {}
    
    text_files = sorted(TEXTS_DIR.glob("*.txt"))
    
    for txt_file in tqdm(text_files, desc="Loading texts"):
        motion_id = txt_file.stem
        motion_descriptions = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: description#tokenized#start#end
                    parts = line.split('#')
                    if parts:
                        desc = parts[0].strip()
                        if desc:
                            motion_descriptions.append(desc)
        
        if motion_descriptions:
            descriptions[motion_id] = motion_descriptions
    
    print(f"Loaded descriptions for {len(descriptions)} motions")
    return descriptions


def main():
    """Main entry point for motion feature extraction."""
    # Create output directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of motion files
    motion_paths = sorted(JOINT_VECS_DIR.glob("*.npy"))
    print(f"Found {len(motion_paths)} motion files in {JOINT_VECS_DIR}")
    
    # Parallel feature extraction
    motion_ids, features = parallel_feature_extraction(motion_paths, N_JOBS)
    
    print(f"\nRaw features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Check for NaN/Inf values
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"Warning: Found {n_nan} NaN and {n_inf} Inf values in features")
        # Replace with column means
        col_means = np.nanmean(features, axis=0)
        for col in range(features.shape[1]):
            mask = np.isnan(features[:, col]) | np.isinf(features[:, col])
            features[mask, col] = col_means[col]
    
    # Normalize features
    normalized_features, mean, std = normalize_features(features)
    
    print(f"\nNormalized features range: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
    
    # Save raw features (before normalization, for analysis)
    raw_path = EMBEDDINGS_DIR / "motion_embeddings_raw.npy"
    np.save(raw_path, features)
    print(f"Saved raw features to {raw_path}")
    
    # Save normalized features (for clustering)
    output_path = EMBEDDINGS_DIR / "motion_embeddings.npy"
    np.save(output_path, normalized_features)
    print(f"Saved normalized features to {output_path}")
    print(f"Shape: {normalized_features.shape}")
    
    # Save motion IDs
    ids_path = EMBEDDINGS_DIR / "motion_ids.npy"
    np.save(ids_path, np.array(motion_ids))
    print(f"Saved motion IDs to {ids_path}")
    
    # Save normalization parameters (for reproducibility)
    norm_params_path = EMBEDDINGS_DIR / "normalization_params.npz"
    np.savez(norm_params_path, mean=mean, std=std)
    print(f"Saved normalization parameters to {norm_params_path}")
    
    # Load and save descriptions for later cluster analysis
    descriptions = load_motion_descriptions()
    desc_path = EMBEDDINGS_DIR / "motion_descriptions.json"
    with open(desc_path, 'w') as f:
        json.dump(descriptions, f)
    print(f"Saved descriptions to {desc_path}")
    
    # Print feature statistics summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total motions: {len(motion_ids)}")
    print(f"Feature dimensions: {EMBEDDING_DIM}")
    print(f"  - Mean features: dims 0-262")
    print(f"  - Std features: dims 263-525")
    print(f"  - Min features: dims 526-788")
    print(f"  - Max features: dims 789-1051")
    
    # Print some statistics about the features
    print("\nFeature group statistics (before normalization):")
    for i, (name, start, end) in enumerate([
        ("Mean", 0, 263),
        ("Std", 263, 526),
        ("Min", 526, 789),
        ("Max", 789, 1052)
    ]):
        group = features[:, start:end]
        print(f"  {name}: range=[{group.min():.4f}, {group.max():.4f}], mean={group.mean():.4f}")


if __name__ == "__main__":
    main()

