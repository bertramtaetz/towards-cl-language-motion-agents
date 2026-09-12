"""
Data Loader for Transfer Learning Experiment 2 (Motion-Based Clustering)

Loads balanced samples from the 5 motion-based dissimilar clusters.
Task splits are based on kinematic similarity (1052-dim motion features).

Task Order (optimized for maximum catastrophic forgetting):
    Task 1: Jumping (Cluster 8)
    Task 2: Arms/Hands (Cluster 4)
    Task 3: Walking (Cluster 2)
    Task 4: Gestures (Cluster 5)
    Task 5: Sit/Stand (Cluster 14)
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

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Motion-Agent/
import utils.paramUtil as paramUtil


# Task order for this experiment (motion-based clustering, max forgetting order)
TASK_ORDER = task_order()

TASK_NAMES = [t[0] for t in TASK_ORDER]


def collate_fn(batch):
    """Sort batch by sentence length (descending) for efficient RNN packing."""
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class SingleTaskMotionDataset(data.Dataset):
    """
    Dataset for single-task training (Transfer Learning).
    
    Loads motions from ONE specific task based on motion clustering.
    """
    
    def __init__(
        self, 
        task_name: str,
        split: str,  # 'train', 'val', or 'test'
        w_vectorizer, 
        task_splits_path: str = None,
        max_text_len: int = 20, 
        unit_length: int = 4,
        data_root: str = None
    ):
        self.task_name = task_name
        self.split = split
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        
        # Set default paths
        if task_splits_path is None:
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
        
        # Build dataset from this task only
        self.data_dict = {}
        self.name_list = []
        
        task_info = self.task_splits['tasks'][self.task_key]
        
        # Support 'trainval' split that combines train and val
        if split == 'trainval':
            motion_ids_for_split = task_info.get('train', []) + task_info.get('val', [])
        else:
            motion_ids_for_split = task_info.get(split, [])
        
        print(f"Loading {task_name} [{split}] data...")
        
        for motion_id in tqdm(motion_ids_for_split, desc=task_name, leave=False):
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
                
                # Store sample
                self.data_dict[motion_id] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data,
                    'motion_id': motion_id
                }
                self.name_list.append(motion_id)
                
            except Exception as e:
                continue
        
        print(f"  {task_name} [{split}]: {len(self.name_list)} samples loaded")
    
    def _find_task_key(self, task_name: str) -> str:
        """Find the task key (e.g., 'task_1') from the task name."""
        # Direct match in task_names list
        if 'task_names' in self.task_splits:
            for i, name in enumerate(self.task_splits['task_names']):
                if name == task_name:
                    return f"task_{i+1}"
        
        # Try matching by task key directly
        for key in self.task_splits['tasks']:
            task_info = self.task_splits['tasks'][key]
            if task_info.get('name') == task_name:
                return key
        
        # Try partial match
        for key in self.task_splits['tasks']:
            if task_name in key or key in task_name:
                return key
        
        return None
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        
        # Tokenize text
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        
        # Get word embeddings
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # Motion length adjustment for VQ-VAE
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std
        
        # Pad to max length
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        
        return (
            word_embeddings, 
            pos_one_hots, 
            caption, 
            sent_len, 
            motion, 
            m_length, 
            '_'.join(tokens), 
            name
        )


def get_single_task_loader(
    task_name: str,
    split: str,
    batch_size: int,
    w_vectorizer,
    unit_length: int = 4,
    num_workers: int = 4,
    task_splits_path: str = None,
    distributed: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int = 2
):
    """
    Create a DataLoader for single-task training/evaluation.
    
    Args:
        task_name: Name of the task (e.g., 'task_1_jumping')
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        w_vectorizer: Word vectorizer for text encoding
        unit_length: VQ-VAE temporal downsampling factor
        num_workers: Number of data loading workers
        task_splits_path: Path to task_splits.json
        distributed: If True, prepared for DDP
        pin_memory: If True, pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        DataLoader instance
    """
    dataset = SingleTaskMotionDataset(
        task_name=task_name,
        split=split,
        w_vectorizer=w_vectorizer,
        task_splits_path=task_splits_path,
        unit_length=unit_length
    )
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split in ['train', 'trainval']),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=(split in ['train', 'trainval']),
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0  # Keep workers alive between batches
    )
    
    return loader


class MultiTaskMotionDataset(data.Dataset):
    """
    Dataset for multi-task learning (all 5 tasks).
    
    Loads motions from all 5 motion-based clusters.
    """
    
    def __init__(
        self, 
        split: str,
        w_vectorizer, 
        task_splits_path: str = None,
        max_text_len: int = 20, 
        unit_length: int = 4,
        data_root: str = None
    ):
        self.split = split
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        
        # Set default paths
        if task_splits_path is None:
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
        
        # Build dataset from all tasks
        self.data_dict = {}
        self.name_list = []
        self.task_labels = []
        self.task_to_indices = {}
        
        self.task_names = self.task_splits.get('task_names', [])
        
        if split == 'train':
            print(f"Loading multi-task {split} data...")
        
        for task_idx, task_key in enumerate(self.task_splits['tasks']):
            task_info = self.task_splits['tasks'][task_key]
            task_name = task_info.get('name', task_key)
            self.task_to_indices[task_idx] = []
            task_count = 0
            
            motion_ids_for_split = task_info.get(split, [])
            
            for motion_id in tqdm(motion_ids_for_split, desc=task_name, leave=False):
                try:
                    motion = np.load(self.motion_dir / f"{motion_id}.npy")
                    
                    if len(motion) < self.min_motion_len or len(motion) >= 200:
                        continue
                    
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
                    
                    sample_name = f"task{task_idx}_{motion_id}"
                    self.data_dict[sample_name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data,
                        'task_id': task_idx,
                        'motion_id': motion_id
                    }
                    
                    idx = len(self.name_list)
                    self.name_list.append(sample_name)
                    self.task_labels.append(task_idx)
                    self.task_to_indices[task_idx].append(idx)
                    task_count += 1
                    
                except Exception as e:
                    continue
            
            if split == 'train':
                print(f"  {task_name}: {task_count} samples loaded")
        
        if split == 'train':
            print(f"Total {split} samples: {len(self.name_list)}")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        task_id = data['task_id']
        
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        motion = (motion - self.mean) / self.std
        
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        
        return (
            word_embeddings, 
            pos_one_hots, 
            caption, 
            sent_len, 
            motion, 
            m_length, 
            '_'.join(tokens), 
            name
        )


def get_multitask_loader(
    split: str,
    batch_size: int,
    w_vectorizer,
    unit_length: int = 4,
    num_workers: int = 8,
    task_splits_path: str = None,
    distributed: bool = False
):
    """
    Create a DataLoader for multi-task training/evaluation.
    """
    dataset = MultiTaskMotionDataset(
        split=split,
        w_vectorizer=w_vectorizer,
        task_splits_path=task_splits_path,
        unit_length=unit_length
    )
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return loader


if __name__ == "__main__":
    # Test the data loader
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "Motion-Agent"))
    from utils.word_vectorizer import WordVectorizer
    
    print("Testing SingleTaskMotionDataset for Motion-Based Clustering...")
    
    w_vectorizer = WordVectorizer(
        str(Path(__file__).resolve().parent.parent.parent / "Motion-Agent" / str(PRETRAINED / 'glove')),
        'our_vab'
    )
    
    # Test each task
    for task_name, cluster_id in TASK_ORDER:
        print(f"\n--- Testing {task_name} (Cluster {cluster_id}) ---")
        loader = get_single_task_loader(
            task_name=task_name,
            split='train',
            batch_size=8,
            w_vectorizer=w_vectorizer
        )
        print(f"  Batches: {len(loader)}")
        
        # Test one batch
        for batch in loader:
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch
            print(f"  Batch shape: motion={motion.shape}")
            print(f"  Sample caption: {caption[0][:80]}...")
            break
    
    print("\nTest complete!")

