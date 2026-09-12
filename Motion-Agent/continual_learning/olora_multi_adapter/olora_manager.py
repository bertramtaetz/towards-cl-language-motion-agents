"""
O-LoRA Multi-Adapter Implementation: Exact Algorithm from Paper

This implements the exact O-LoRA algorithm from:
"Orthogonal Subspace Learning for Language Model Continual Learning"
Wang et al., EMNLP 2023 Findings
https://github.com/cmnfriend/O-LoRA

Key mechanism (from paper Section 3.2):
- Create a NEW LoRA adapter {A_t, B_t} for each task t
- Freeze previous LoRA parameters {A_i, B_i | i < t} during task t training
- Add orthogonality loss: L_orth(A_i, A_t) = sum_{j,k} ||O_{i,t}[j,k]||^2
  where O_{i,t} = A_i^T @ A_t
- Subspace U_t is spanned by column vectors of A_t

Training objective (Eq. 7):
    sum_{x,y} log p(y|x) + λ_1 * sum_{i<t} L_orth(A_i, A_t)

After training, can merge all LoRA into base weights:
    W_init := W_init + sum_{i=1}^{t} A_i @ B_i
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional
from peft import LoraConfig, TaskType
import gc


class MultiAdapterOLoRAManager:
    """
    Manages O-LoRA continual learning with per-task LoRA adapters.
    
    Implements the exact algorithm from the paper:
    1. Create new LoRA adapter for each task
    2. Freeze previous task adapters
    3. Compute orthogonality loss between current and previous A matrices
    """
    
    def __init__(
        self, 
        model: nn.Module,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        """
        Args:
            model: The base model (will be wrapped with PEFT if not already)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            target_modules: Modules to apply LoRA to (default: q_proj, v_proj per paper)
        """
        self.model = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Paper applies LoRA to query and value projections.
        # We keep that as the default, but allow experiments with different
        # target modules (e.g., the full Motion-Agent LoRA module set).
        self.target_modules = self._resolve_target_modules(target_modules)
        
        # Track completed tasks and their adapter names
        self.completed_tasks: List[int] = []
        self.task_adapter_names: Dict[int, str] = {}
        
        # Store frozen A matrices for orthogonality loss computation
        # Structure: {layer_name: {task_id: A_matrix}}
        self.frozen_A_matrices: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        
        # Current task being trained
        self.current_task_id: Optional[int] = None
        self.current_adapter_name: Optional[str] = None

    @staticmethod
    def _resolve_target_modules(target_modules: Optional[List[str]] | str) -> List[str]:
        """Resolve target modules configuration.

        Supports:
          - None              -> default q_proj/v_proj
          - "qv"             -> q_proj/v_proj
          - "full"           -> 7-module Motion-Agent LoRA set
          - "a,b,c"          -> explicit comma-separated list
          - ["a", "b"]      -> explicit list
        """
        if target_modules is None:
            return ["q_proj", "v_proj"]
        if isinstance(target_modules, list):
            return target_modules
        mode = str(target_modules).strip().lower()
        if mode in {"qv", "q_proj,v_proj", "q_proj_v_proj"}:
            return ["q_proj", "v_proj"]
        if mode in {"full", "default"}:
            return [
                "o_proj", "q_proj", "up_proj", "v_proj",
                "k_proj", "down_proj", "gate_proj",
            ]
        if "," in mode:
            return [m.strip() for m in mode.split(",") if m.strip()]
        raise ValueError(
            f"Unknown target_modules spec: {target_modules}. Use None, 'qv', 'full', a comma-separated string, or a list."
        )
        
    def get_adapter_name(self, task_id: int) -> str:
        """Generate adapter name for a task."""
        return f"task_{task_id}"
    
    def prepare_task(self, task_id: int, peft_model) -> str:
        """
        Prepare model for training on a new task.
        
        This:
        1. Freezes all previous task adapters
        2. Creates a new LoRA adapter for the current task
        3. Sets the new adapter as active
        
        Args:
            task_id: ID of the task to prepare
            peft_model: The PEFT model to add adapter to
            
        Returns:
            Name of the new adapter
        """
        adapter_name = self.get_adapter_name(task_id)
        
        print(f"[O-LoRA Multi] Preparing Task {task_id} with adapter '{adapter_name}'")
        
        # Freeze all previous task adapters
        for prev_task_id, prev_adapter_name in self.task_adapter_names.items():
            self._freeze_adapter(peft_model, prev_adapter_name)
            print(f"[O-LoRA Multi] Froze adapter '{prev_adapter_name}' for Task {prev_task_id}")
        
        # CRITICAL: Freeze original t2m/m2t adapters (they should never be trained)
        # These are the base motion adapters from the pretrained model
        for base_adapter in ['t2m', 'm2t']:
            if base_adapter in peft_model.peft_config:
                self._freeze_adapter(peft_model, base_adapter)
                print(f"[O-LoRA Multi] Froze base adapter '{base_adapter}'")
        
        # Check if adapter already exists (resuming training)
        existing_adapters = list(peft_model.peft_config.keys())
        
        if adapter_name not in existing_adapters:
            # Create new LoRA adapter for this task
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            peft_model.add_adapter(adapter_name, lora_config)
            print(f"[O-LoRA Multi] Created new adapter '{adapter_name}'")
        else:
            print(f"[O-LoRA Multi] Using existing adapter '{adapter_name}'")
        
        # Set the new adapter as active
        peft_model.set_adapter(adapter_name)
        
        # Ensure new adapter is trainable
        self._unfreeze_adapter(peft_model, adapter_name)
        
        self.current_task_id = task_id
        self.current_adapter_name = adapter_name
        self.task_adapter_names[task_id] = adapter_name
        
        return adapter_name
    
    def _freeze_adapter(self, peft_model, adapter_name: str):
        """Freeze all parameters of a specific adapter."""
        for name, param in peft_model.named_parameters():
            if adapter_name in name:
                param.requires_grad = False
    
    def _unfreeze_adapter(self, peft_model, adapter_name: str):
        """Unfreeze all parameters of a specific adapter."""
        for name, param in peft_model.named_parameters():
            if adapter_name in name:
                param.requires_grad = True
    
    def complete_task(self, task_id: int, peft_model):
        """
        Complete training on a task.
        
        This:
        1. Stores the A matrices for orthogonality loss computation
        2. Marks the task as completed
        
        Args:
            task_id: ID of the completed task
            peft_model: The PEFT model
        """
        adapter_name = self.get_adapter_name(task_id)
        
        print(f"[O-LoRA Multi] Completing Task {task_id}...")
        
        # Store A matrices (frozen for future orthogonality computation)
        num_stored = 0
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A') and isinstance(module.lora_A, nn.ModuleDict):
                if adapter_name in module.lora_A:
                    # Get A matrix and store a copy
                    A = module.lora_A[adapter_name].weight.data.clone().cpu()
                    layer_name = name
                    self.frozen_A_matrices[layer_name][task_id] = A
                    num_stored += 1
        
        self.completed_tasks.append(task_id)
        
        # Freeze this adapter's parameters
        self._freeze_adapter(peft_model, adapter_name)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[O-LoRA Multi] Stored {num_stored} A matrices for Task {task_id}")
        print(f"[O-LoRA Multi] Total completed tasks: {len(self.completed_tasks)}")
    
    def compute_orthogonality_loss(self, peft_model) -> torch.Tensor:
        """
        Compute orthogonality loss between current task's A matrices 
        and all previous tasks' A matrices.
        
        From paper Eq. 6 and 8:
            O_{i,t} = A_i^T @ A_t
            L_orth(A_i, A_t) = sum_{j,k} ||O_{i,t}[j,k]||^2
        
        Args:
            peft_model: The PEFT model
            
        Returns:
            Orthogonality loss tensor (scalar)
        """
        if not self.completed_tasks or self.current_adapter_name is None:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=next(peft_model.parameters()).device)
        num_terms = 0
        
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A') and isinstance(module.lora_A, nn.ModuleDict):
                if self.current_adapter_name in module.lora_A:
                    # Get current task's A matrix
                    A_current = module.lora_A[self.current_adapter_name].weight
                    layer_name = name
                    
                    # Compute orthogonality loss with each previous task
                    if layer_name in self.frozen_A_matrices:
                        for prev_task_id, A_prev in self.frozen_A_matrices[layer_name].items():
                            # Move previous A to same device
                            A_prev = A_prev.to(A_current.device, dtype=A_current.dtype)
                            
                            # O_{i,t} = A_i^T @ A_t (Eq. 6)
                            # A shape: (r, k) where r is rank, k is input dim
                            # O shape: (r_prev, r_current)
                            O = A_prev @ A_current.T
                            
                            # L_orth = sum_{j,k} ||O[j,k]||^2 (Eq. 8)
                            # This is just the squared Frobenius norm
                            loss = torch.sum(O ** 2)
                            
                            total_loss = total_loss + loss
                            num_terms += 1
        
        # Average over all terms for stability
        if num_terms > 0:
            total_loss = total_loss / num_terms
        
        return total_loss
    
    def compute_orthogonality_metric(self, peft_model) -> float:
        """
        Compute orthogonality metric for monitoring (non-differentiable).
        
        Returns:
            Average orthogonality measure (lower = more orthogonal)
        """
        with torch.no_grad():
            loss = self.compute_orthogonality_loss(peft_model)
            return loss.item()
    
    def get_num_completed_tasks(self) -> int:
        """Return the number of completed tasks."""
        return len(self.completed_tasks)
    
    def get_active_adapters(self) -> List[str]:
        """Return list of all adapter names."""
        return list(self.task_adapter_names.values())
    
    def merge_adapters(self, peft_model):
        """
        Merge all task adapters into base model weights.
        
        From paper: W_init := W_init + sum_{i=1}^{t} A_i @ B_i
        
        Args:
            peft_model: The PEFT model to merge
        """
        print("[O-LoRA Multi] Merging all adapters into base model...")
        
        # Enable all adapters for merging
        for adapter_name in self.task_adapter_names.values():
            peft_model.set_adapter(adapter_name)
        
        # Merge and unload
        peft_model.merge_and_unload()
        
        print("[O-LoRA Multi] All adapters merged successfully")
    
    def get_trainable_params_info(self, peft_model) -> Dict:
        """Get information about trainable parameters."""
        total_params = 0
        trainable_params = 0
        
        for name, param in peft_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
            "num_adapters": len(self.task_adapter_names),
            "completed_tasks": len(self.completed_tasks),
        }
    
    def clear(self):
        """Clear all stored state."""
        self.frozen_A_matrices.clear()
        self.completed_tasks.clear()
        self.task_adapter_names.clear()
        self.current_task_id = None
        self.current_adapter_name = None
        gc.collect()
        torch.cuda.empty_cache()


# Backward compatibility alias
OLoRAManager = MultiAdapterOLoRAManager
