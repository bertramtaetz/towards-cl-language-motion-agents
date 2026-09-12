"""
Standardized Wandb Logging for Continual Learning Experiments

Provides a unified logging interface across all CL methods (D-MoLE, O-LoRA, OGD, etc.)
to enable easy cross-method comparison in wandb dashboards.

All metrics are logged at effective batch granularity (after gradient accumulation
completes and gradients are synchronized across all GPUs).

Usage:
    from continual_learning.utils import CLWandbLogger
    
    logger = CLWandbLogger(is_main_process=accelerator.is_main_process)
    
    # During training (after each effective batch / optimizer step)
    logger.log_effective_batch(
        task_id=task_id,
        loss=loss_value,
        accuracy=acc_value,
        orth_loss=orth_loss_value,  # O-LoRA specific
    )
    
    # After task evaluation
    logger.log_task_eval(task_id=task_id, metrics_dict={
        'accuracy': acc,
        'fid': fid,
        'diversity': div,
        ...
    })
"""

from typing import Optional, Dict, Any

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class CLWandbLogger:
    """Standardized wandb logging for continual learning experiments.
    
    Logs metrics at effective batch granularity with consistent naming:
    - train/loss, train/accuracy: For cross-method comparison
    - task_{id}/loss, task_{id}/accuracy: For task-specific panels
    - eval/task_{id}/{metric}: For evaluation metrics
    - cl_metrics/{metric}: For final continual learning metrics
    """
    
    def __init__(self, is_main_process: bool = True, disabled: bool = False):
        """
        Initialize the logger.
        
        Args:
            is_main_process: Only log on main process (for distributed training)
            disabled: If True, all logging is skipped (e.g., --no-wandb flag)
        """
        self.is_main_process = is_main_process
        self.disabled = disabled or not HAS_WANDB
        self.global_step = 0
    
    def log_effective_batch(
        self,
        task_id: int,
        loss: float,
        accuracy: float,
        orth_loss: Optional[float] = None,
        task_loss: Optional[float] = None,
        recon_loss: Optional[float] = None,
        contrastive_loss: Optional[float] = None,
        projection_ratio: Optional[float] = None,
        **extra_metrics
    ):
        """
        Log metrics at effective batch level (after GPU sync).
        
        Called after each optimizer step when accelerator.sync_gradients is True.
        
        Args:
            task_id: Current task being trained (0-indexed)
            loss: Primary loss value (total loss for the step)
            accuracy: Token prediction accuracy
            orth_loss: O-LoRA orthogonality loss term
            task_loss: Task-specific loss (before regularization)
            recon_loss: D-MoLE router autoencoder reconstruction loss
            contrastive_loss: D-MoLE router contrastive loss
            projection_ratio: OGD gradient projection ratio
            **extra_metrics: Any additional method-specific metrics
        """
        if not self.is_main_process or self.disabled:
            return
        
        self.global_step += 1
        
        # Common metrics (for cross-method comparison)
        metrics = {
            "train/loss": loss,
            "train/accuracy": accuracy,
            "train/task_id": task_id,
            "train/global_step": self.global_step,
            # Task-grouped (for per-task panels in wandb)
            f"task_{task_id}/loss": loss,
            f"task_{task_id}/accuracy": accuracy,
        }
        
        # Method-specific metrics (O-LoRA)
        if orth_loss is not None:
            metrics["train/orth_loss"] = orth_loss
            metrics[f"task_{task_id}/orth_loss"] = orth_loss
        
        if task_loss is not None:
            metrics["train/task_loss"] = task_loss
            metrics[f"task_{task_id}/task_loss"] = task_loss
        
        # Method-specific metrics (D-MoLE)
        if recon_loss is not None:
            metrics["train/recon_loss"] = recon_loss
            metrics[f"task_{task_id}/recon_loss"] = recon_loss
        
        if contrastive_loss is not None:
            metrics["train/contrastive_loss"] = contrastive_loss
            metrics[f"task_{task_id}/contrastive_loss"] = contrastive_loss
        
        # Method-specific metrics (OGD)
        if projection_ratio is not None:
            metrics["train/projection_ratio"] = projection_ratio
            metrics[f"task_{task_id}/projection_ratio"] = projection_ratio
        
        # Any additional metrics
        for key, value in extra_metrics.items():
            if value is not None:
                metrics[f"train/{key}"] = value
                metrics[f"task_{task_id}/{key}"] = value
        
        wandb.log(metrics)
    
    def log_task_eval(
        self,
        task_id: int,
        metrics_dict: Dict[str, Any],
        stage: Optional[str] = None,
        completed_tasks: Optional[int] = None
    ):
        """
        Log evaluation metrics after task completion.
        
        Args:
            task_id: Task being evaluated (0-indexed)
            metrics_dict: Dictionary of metric name -> value
                Expected keys: accuracy, fid, diversity, r_precision_top1, 
                r_precision_top2, r_precision_top3, mm_dist, loss
            stage: Optional stage identifier (e.g., "after_task_2")
            completed_tasks: Number of tasks completed so far
        """
        if not self.is_main_process or self.disabled:
            return
        
        eval_metrics = {}
        for key, value in metrics_dict.items():
            if value is not None:
                eval_metrics[f"eval/task_{task_id}/{key}"] = value
        
        # Add meta information
        if stage is not None:
            eval_metrics["eval/stage"] = stage
        if completed_tasks is not None:
            eval_metrics["eval/completed_tasks"] = completed_tasks
        
        wandb.log(eval_metrics)
    
    def log_all_tasks_eval(
        self,
        results: Dict[int, Dict[str, Any]],
        stage: str,
        completed_tasks: int
    ):
        """
        Log evaluation results for all tasks at once.
        
        Args:
            results: Dict mapping task_id -> metrics_dict
            stage: Stage identifier (e.g., "after_task_2")
            completed_tasks: Number of tasks completed
        """
        if not self.is_main_process or self.disabled:
            return
        
        eval_metrics = {}
        for task_id, metrics_dict in results.items():
            for key, value in metrics_dict.items():
                if value is not None:
                    eval_metrics[f"eval/task_{task_id}/{key}"] = value
        
        eval_metrics["eval/stage"] = stage
        eval_metrics["eval/completed_tasks"] = completed_tasks
        
        wandb.log(eval_metrics)
    
    def log_cl_metrics(self, metrics: Dict[str, Any]):
        """
        Log final continual learning metrics.
        
        Args:
            metrics: Dictionary of CL metrics
                Expected keys: bwt, fwt, acc, max_forgetting, avg_forgetting,
                and per-metric variants like bwt_fid, final_avg_accuracy, etc.
        """
        if not self.is_main_process or self.disabled:
            return
        
        cl_metrics = {}
        for key, value in metrics.items():
            if value is not None:
                cl_metrics[f"cl_metrics/{key}"] = value
        
        wandb.log(cl_metrics)
    
    def log_config(self, config: Dict[str, Any], algorithm: str):
        """
        Log configuration/hyperparameters.
        
        Args:
            config: Configuration dictionary
            algorithm: Algorithm name ("D-MoLE", "O-LoRA", "OGD", "Transfer", "MultiTask")
        """
        if not self.is_main_process or self.disabled:
            return
        
        # Update wandb config
        wandb.config.update({"algorithm": algorithm, **config})
    
    def increment_step(self):
        """Manually increment global step (useful for custom logging)."""
        self.global_step += 1
    
    def get_global_step(self) -> int:
        """Get current global step."""
        return self.global_step
    
    def set_global_step(self, step: int):
        """Set global step (useful when resuming)."""
        self.global_step = step
