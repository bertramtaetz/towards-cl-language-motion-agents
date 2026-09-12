"""Configuration for motion-based clustering pipeline.

This experiment uses 1052-dimensional motion statistical features
(mean, std, min, max for each of 263 motion dimensions) instead of
text embeddings for clustering.
"""
from pathlib import Path
import os

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2] / "outputs/motion_clustering"
MSAI_ROOT = PROJECT_ROOT.parent.parent
HUMANML3D_ROOT = Path(os.environ.get("MOTION_DATA_ROOT", Path(__file__).resolve().parents[2] / "datasets/HumanML3D"))

# Original text-based clustering experiment (for comparison)
TEXT_CLUSTERING_ROOT = MSAI_ROOT / "experiments" / "humanml3d_clustering"

# Data paths
TEXTS_DIR = HUMANML3D_ROOT / "texts"
JOINT_VECS_DIR = HUMANML3D_ROOT / "new_joint_vecs"
JOINTS_DIR = HUMANML3D_ROOT / "new_joints"

# Output paths
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Motion Feature Settings
# =============================================================================
# HumanML3D motion representation dimensions
MOTION_DIM = 263

# Feature extraction: 4 statistics per dimension = 1052 total features
STATS_PER_DIM = 4  # mean, std, min, max
EMBEDDING_DIM = MOTION_DIM * STATS_PER_DIM  # 1052

# Dimension slices for the 263-dim motion representation
ROOT_VEL_DIM = slice(0, 2)       # x, z velocity (2D)
ROOT_ANG_VEL_DIM = 2             # angular velocity (1D)
ROOT_HEIGHT_DIM = 3              # y position (1D)
LOCAL_POS_DIM = slice(4, 67)     # 21 joints × 3 (63D)
LOCAL_VEL_DIM = slice(67, 130)   # 21 joints × 3 (63D)
JOINT_ROT_DIM = slice(130, 256)  # 21 joints × 6 (126D, 6D rotation)
FOOT_CONTACT_DIM = slice(260, 263)  # 3-4 foot contacts

# =============================================================================
# Clustering Settings
# =============================================================================
K_RANGE = (5, 100)  # Range of k values to test
LINKAGE_METHOD = "ward"  # Hierarchical clustering linkage method

# =============================================================================
# Task Selection Settings
# =============================================================================
NUM_TASKS = 5  # Number of dissimilar clusters to select
MIN_CLUSTER_SIZE = 560  # Minimum samples per cluster for selection
TRAIN_SAMPLES_PER_TASK = 360
VAL_SAMPLES_PER_TASK = 100
TEST_SAMPLES_PER_TASK = 100

# Motion length filtering (for VQ-VAE compatibility)
MIN_MOTION_LENGTH = 40   # frames
MAX_MOTION_LENGTH = 200  # frames

# =============================================================================
# Hardware Settings
# =============================================================================
N_JOBS = 48  # Number of CPU cores for parallel processing

# =============================================================================
# Comparison Settings
# =============================================================================
# Path to text-based clustering results for comparison
TEXT_EMBEDDINGS_PATH = TEXT_CLUSTERING_ROOT / "embeddings" / "text_embeddings.npy"
TEXT_MOTION_IDS_PATH = TEXT_CLUSTERING_ROOT / "embeddings" / "motion_ids.npy"
TEXT_CLUSTER_ASSIGNMENTS_PATH = TEXT_CLUSTERING_ROOT / "results" / "cluster_assignments_k35.json"
TEXT_DESCRIPTIONS_PATH = TEXT_CLUSTERING_ROOT / "embeddings" / "motion_descriptions.json"

