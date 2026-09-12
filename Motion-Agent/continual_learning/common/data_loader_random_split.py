"""
Data Loader with Random 80/20 Train-Test Split for D-MoLE

This module provides data loading that:
1. Combines ALL available data for each task (train + val + test)
2. Creates a random 80/20 split with a fixed seed for reproducibility

This ensures we use ALL available data and have a consistent split.
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Motion-Agent/
import utils.paramUtil as paramUtil


# Task order for 5-task experiment (motion-based clustering, max forgetting order)
TASK_ORDER_5 = task_order()

TASK_ORDER = TASK_ORDER_5
TASK_NAMES = [t[0] for t in TASK_ORDER]
TASK_NAMES_5 = [t[0] for t in TASK_ORDER_5]


def set_task_mode(mode: str):
    """Set the task mode. Only '5_tasks' is supported."""
    global TASK_ORDER, TASK_NAMES
    if mode == '5_tasks':
        TASK_ORDER = TASK_ORDER_5
        TASK_NAMES = TASK_NAMES_5
    else:
        raise ValueError(f"Unknown mode: {mode}. Only '5_tasks' is supported.")


def collate_fn(batch):
    """Sort batch by sentence length (descending) for efficient RNN packing."""
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class RandomSplitMotionDataset(data.Dataset):
    """
    Dataset with random 80/20 train-test split.
    
    Combines ALL data for a task (train + val + test from original splits)
    and creates a new random split with:
    - 80% for training
    - 20% for testing
    
    Uses a fixed seed for reproducibility.
    """
    
    def __init__(
        self, 
        task_name: str,
        split: str,  # 'train' or 'test' (80/20 random split)
        w_vectorizer, 
        task_splits_path: str = None,
        max_text_len: int = 20, 
        unit_length: int = 4,
        data_root: str = None,
        train_ratio: float = 0.8,
        random_seed: int = 42,
    ):
        self.task_name = task_name
        self.split = split
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # Set default paths based on task mode
        if task_splits_path is None:
            base_path = Path(__file__).resolve().parent.parent.parent / "baselines" / "transfer_learning"
            # Use all15 splits if task name matches 15-task format
            # if task_name in TASK_NAMES_15:
            #     task_splits_path = base_path / "task_splits_all15.json"
            # else:
            task_splits_path = TASK_SPLITS
        if data_root is None:
            # Data is at msai-thesis/datasets/HumanML3D/HumanML3D
            data_root = DATA_ROOT
        
        self.data_root = Path(data_root).resolve()
        self.motion_dir = self.data_root / "new_joint_vecs"
        self.text_dir = self.data_root / "texts"
        
        # HumanML3D specific settings
        self.joints_num = 22
        self.max_motion_length = 196
        self.dim_pose = 263
        self.fps = 20
        self.min_motion_len = 40
        
        # Load normalization stats
        self.meta_dir = PRETRAINED / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"
        self.mean = np.load(self.meta_dir / "mean.npy")
        self.std = np.load(self.meta_dir / "std.npy")
        
        # Load task splits
        with open(task_splits_path, 'r') as f:
            self.task_splits = json.load(f)
        
        # Map task name to task key (e.g., "task_1_jumping" -> "task_1")
        self.task_key = self._find_task_key(task_name)
        if self.task_key is None:
            available = self.task_splits.get('task_names', list(self.task_splits['tasks'].keys()))
            raise ValueError(f"Task '{task_name}' not found. Available: {available}")
        
        # Build dataset from ALL data for this task
        self._load_all_data_and_split()
    
    def _find_task_key(self, task_name: str) -> Optional[str]:
        """Find the task key that matches the task name.
        
        Fix: Uses exact matching and proper prefix matching to avoid 
        substring collision (e.g., 'task_1' matching 'task_10').
        """
        # First: Check for exact name match (most reliable)
        for key in self.task_splits['tasks']:
            if self.task_splits['tasks'][key].get('name', '') == task_name:
                return key
        
        # Second: Check if task_name starts with key followed by underscore
        # This prevents 'task_1' from matching 'task_10_...'
        # Sort keys by length descending to match longest first (task_10 before task_1)
        sorted_keys = sorted(self.task_splits['tasks'].keys(), key=len, reverse=True)
        for key in sorted_keys:
            # key must be followed by underscore or end of string
            if task_name.startswith(key + '_') or task_name == key:
                return key
        
        return None
    
    def _load_all_data_and_split(self):
        """Load ALL data for the task and create 80/20 random split."""
        self.data_dict = {}
        self.name_list = []
        
        task_info = self.task_splits['tasks'][self.task_key]
        
        # Combine ALL motion IDs (train + val + test)
        all_motion_ids = []
        for split_key in ['train', 'val', 'test']:
            all_motion_ids.extend(task_info.get(split_key, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_motion_ids = []
        for mid in all_motion_ids:
            if mid not in seen:
                seen.add(mid)
                unique_motion_ids.append(mid)
        
        print(f"Loading {self.task_name} - ALL data ({len(unique_motion_ids)} motion IDs)...")
        
        # Load all valid motions first
        valid_motions = []
        for motion_id in tqdm(unique_motion_ids, desc=self.task_name, leave=False):
            try:
                motion = np.load(self.motion_dir / f"{motion_id}.npy")
                
                # Length filtering
                if len(motion) < self.min_motion_len or len(motion) >= 200:
                    continue
                
                # Load text descriptions
                text_data = []
                with cs.open(self.text_dir / f"{motion_id}.txt") as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            text_data.append({
                                'caption': caption,
                                'tokens': tokens
                            })
                
                if not text_data:
                    continue
                
                valid_motions.append({
                    'motion_id': motion_id,
                    'motion': motion,
                    'text_data': text_data
                })
                
            except Exception as e:
                continue
        
        # Now create the random 80/20 split
        rng = np.random.RandomState(self.random_seed)
        indices = np.arange(len(valid_motions))
        rng.shuffle(indices)
        
        n_train = int(len(valid_motions) * self.train_ratio)
        
        if self.split == 'train':
            selected_indices = indices[:n_train]
        else:  # test
            selected_indices = indices[n_train:]
        
        # Build final dataset
        for idx in selected_indices:
            item = valid_motions[idx]
            motion_id = item['motion_id']
            
            self.data_dict[motion_id] = {
                'motion': item['motion'],
                'length': len(item['motion']),
                'text': item['text_data']
            }
            self.name_list.append(motion_id)
        
        print(f"  {self.task_name} [{self.split}]: {len(self.name_list)} samples "
              f"(from {len(valid_motions)} total, {self.train_ratio*100:.0f}/{(1-self.train_ratio)*100:.0f} split)")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        motion_id = self.name_list[idx]
        data = self.data_dict[motion_id]
        
        motion = data['motion']
        m_length = data['length']
        text_data = data['text']
        
        # Random text selection
        text = random.choice(text_data)
        caption = text['caption']
        tokens = text['tokens']
        
        # Motion processing
        if len(motion) >= self.max_motion_length:
            idx_start = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[idx_start:idx_start + self.max_motion_length]
            m_length = self.max_motion_length
        
        # Normalize motion
        motion = (motion - self.mean) / self.std
        
        # Pad motion
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        
        # Text vectorization
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = ['sos/OTHER'] + tokens[:self.max_text_len] + ['eos/OTHER']
            sent_len = len(tokens)
        
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            '_'.join(tokens),
            motion_id  # Include motion_id as name
        )


def get_random_split_loader(
    task_name: str,
    split: str,  # 'train' or 'test'
    batch_size: int,
    w_vectorizer,
    unit_length: int = 4,
    num_workers: int = 4,
    task_splits_path: str = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    Create a DataLoader with random 80/20 train-test split.
    
    Args:
        task_name: Name of the task
        split: 'train' (80%) or 'test' (20%)
        batch_size: Batch size
        w_vectorizer: Word vectorizer
        unit_length: VQ-VAE temporal factor
        num_workers: Data loading workers
        task_splits_path: Path to task_splits.json
        train_ratio: Fraction of data for training (default 0.8)
        random_seed: Seed for reproducible splits
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Prefetch factor for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = RandomSplitMotionDataset(
        task_name=task_name,
        split=split,
        w_vectorizer=w_vectorizer,
        task_splits_path=task_splits_path,
        unit_length=unit_length,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return loader


# Test the implementation
if __name__ == '__main__':
    print("Testing RandomSplitMotionDataset...")
    
    # Load word vectorizer
    from os.path import join as pjoin
    motion_agent_root = Path(__file__).parent.parent.parent
    glove_path = PRETRAINED / 'glove'
    
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer(str(glove_path / "our_vab"), str(glove_path / "our_vab"))
    
    # Test each task
    for task_name in TASK_NAMES:
        print(f"\n--- Testing {task_name} ---")
        
        # Test train split
        train_loader = get_random_split_loader(
            task_name=task_name,
            split='train',
            batch_size=8,
            w_vectorizer=w_vectorizer,
            random_seed=42,
        )
        
        # Test test split
        test_loader = get_random_split_loader(
            task_name=task_name,
            split='test',
            batch_size=8,
            w_vectorizer=w_vectorizer,
            random_seed=42,
        )
        
        print(f"  Train batches: {len(train_loader)}, samples: {len(train_loader.dataset)}")
        print(f"  Test batches: {len(test_loader)}, samples: {len(test_loader.dataset)}")
        
        # Verify no overlap
        train_ids = set(train_loader.dataset.name_list)
        test_ids = set(test_loader.dataset.name_list)
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Found overlap: {overlap}"
        
        total = len(train_ids) + len(test_ids)
        train_pct = len(train_ids) / total * 100
        test_pct = len(test_ids) / total * 100
        print(f"  Split: {train_pct:.1f}% train, {test_pct:.1f}% test")
        print(f"  ✅ No overlap between train and test")
    
    print("\n✅ All tests passed!")
