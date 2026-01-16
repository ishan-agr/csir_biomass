"""
Utilities for CSIRO Pasture Biomass Prediction.

Contains:
- Weighted R² metric (competition metric)
- Training utilities (seed, logging, checkpointing)
- Learning rate schedulers
- EarlyStopping
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json


# Competition target weights
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5
}

TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> float:
    """
    Compute weighted R² score as per competition rules.

    The weighted R² is computed globally over all (image, target) pairs,
    with per-row weights based on target type.

    Formula:
        R² = 1 - SS_res / SS_tot

    where:
        SS_res = sum(w_i * (y_i - y_hat_i)²)
        SS_tot = sum(w_i * (y_i - y_bar_weighted)²)
        y_bar_weighted = sum(w_i * y_i) / sum(w_i)

    Args:
        y_true: (n_samples, 5) or (n_samples * 5,) ground truth values
        y_pred: (n_samples, 5) or (n_samples * 5,) predicted values
        target_names: List of target names in order (if flattened)

    Returns:
        Weighted R² score
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle different input shapes
    if y_true.ndim == 2:
        # Shape: (n_samples, 5) - wide format
        n_samples, n_targets = y_true.shape
        assert n_targets == 5, f"Expected 5 targets, got {n_targets}"

        # Create weight array
        weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])

        # Expand weights to match shape
        weights = np.tile(weights, n_samples)

        # Flatten arrays
        y_true_flat = y_true.flatten(order='C')  # [sample0_t0, sample0_t1, ..., sample1_t0, ...]

        # Need to reorder to match: flatten should give [s0_t0, s0_t1, ..., s0_t4, s1_t0, ...]
        # Then weights should be [w0, w1, w2, w3, w4, w0, w1, ...]
        y_pred_flat = y_pred.flatten(order='C')

    else:
        # Shape: (n_samples * 5,) - long format
        # Assume target_names is provided or default order
        if target_names is None:
            # Assume repeating pattern
            n_samples = len(y_true) // 5
            weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
            weights = np.tile(weights, n_samples)
        else:
            weights = np.array([TARGET_WEIGHTS[t] for t in target_names])

        y_true_flat = y_true
        y_pred_flat = y_pred

    # Compute weighted mean of y_true
    y_bar_weighted = np.sum(weights * y_true_flat) / np.sum(weights)

    # Compute SS_res (weighted residual sum of squares)
    ss_res = np.sum(weights * (y_true_flat - y_pred_flat) ** 2)

    # Compute SS_tot (weighted total sum of squares)
    ss_tot = np.sum(weights * (y_true_flat - y_bar_weighted) ** 2)

    # Compute R²
    if ss_tot == 0:
        return 0.0 if ss_res != 0 else 1.0

    r2 = 1 - (ss_res / ss_tot)

    return float(r2)


def per_target_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute R² for each target separately.

    Args:
        y_true: (n_samples, 5) ground truth
        y_pred: (n_samples, 5) predictions

    Returns:
        Dictionary with R² for each target
    """
    from sklearn.metrics import r2_score

    r2_scores = {}
    for i, target in enumerate(TARGET_ORDER):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Handle edge cases
        if np.std(y_t) == 0:
            r2_scores[target] = 0.0
        else:
            r2_scores[target] = r2_score(y_t, y_p)

    return r2_scores


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # For full reproducibility (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # For max performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_gpu(gpu_config=None):
    """
    Setup GPU for maximum utilization.

    This function configures CUDA settings for optimal performance.

    Args:
        gpu_config: GPUConfig object or None for defaults

    Returns:
        torch.device: The configured device
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')

    # Print GPU info
    gpu_count = torch.cuda.device_count()
    print(f"\n{'='*50}")
    print("GPU Configuration")
    print(f"{'='*50}")
    print(f"CUDA Available: True")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")

    # Apply GPU config settings
    if gpu_config is not None:
        # cuDNN settings
        torch.backends.cudnn.benchmark = gpu_config.cudnn_benchmark
        torch.backends.cudnn.deterministic = gpu_config.cudnn_deterministic

        # TF32 settings (Ampere+ GPUs for faster matmul)
        if gpu_config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"\nTF32 Enabled: True (faster on Ampere+ GPUs)")
    else:
        # Default: optimize for speed
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\ncuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"{'='*50}\n")

    return torch.device('cuda')


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3

        info[f'gpu_{i}'] = {
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'total_gb': round(total, 2),
            'free_gb': round(total - reserved, 2),
            'utilization_pct': round(allocated / total * 100, 1)
        }

    return info


def print_gpu_memory():
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if not info:
        print("No GPU available")
        return

    for gpu, mem in info.items():
        print(f"{gpu}: {mem['allocated_gb']:.2f}GB / {mem['total_gb']:.2f}GB "
              f"({mem['utilization_pct']:.1f}% utilized)")


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def to_device(data, device, non_blocking: bool = True):
    """
    Move data to device efficiently.

    Args:
        data: Tensor, dict, list, or tuple of tensors
        device: Target device
        non_blocking: Use non-blocking transfer for pinned memory

    Returns:
        Data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    else:
        return data


def optimize_model_for_inference(model: nn.Module, use_channels_last: bool = True):
    """
    Optimize model for inference.

    Args:
        model: PyTorch model
        use_channels_last: Use NHWC memory format (faster for CNNs)

    Returns:
        Optimized model
    """
    model.eval()

    if use_channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    return model


def compile_model(model: nn.Module, mode: str = "reduce-overhead"):
    """
    Compile model with torch.compile for faster execution (PyTorch 2.0+).

    NOTE: torch.compile with inductor backend requires Triton, which is
    only available on Linux. On Windows, this will gracefully skip compilation.

    Args:
        model: PyTorch model
        mode: Compilation mode
            - "default": Good balance of compile time and performance
            - "reduce-overhead": Minimize CPU overhead (best for training)
            - "max-autotune": Maximum performance (longer compile time)

    Returns:
        Compiled model (or original model if compilation not supported)
    """
    import platform

    # Check if on Windows - Triton (required by inductor) is Linux-only
    if platform.system() == "Windows":
        print("WARNING: torch.compile is not supported on Windows (requires Triton/Linux)")
        print("Continuing without compilation - GPU optimizations still active")
        return model

    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model

    try:
        print(f"Compiling model with mode='{mode}'...")
        model = torch.compile(model, mode=mode)
        print("Model compiled successfully")
    except Exception as e:
        print(f"WARNING: torch.compile failed: {e}")
        print("Continuing without compilation - GPU optimizations still active")

    return model


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and optional restarts.

    Reference: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.max_lr)

    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [
                (self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
                for _ in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                self.min_lr + (self.max_lr - self.min_lr) *
                (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) /
                            (self.cur_cycle_steps - self.warmup_steps))) / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1

            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult
                    ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * (self.cycle_mult ** n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_lrs[0] * (self.gamma ** self.cycle)
        self.last_epoch = max(0, epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_improvement = lambda score, best: score < best - min_delta
        else:
            self.is_improvement = lambda score, best: score > best + min_delta

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_improvement(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(log_dir: Path, name: str = "train") -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    score: float,
    path: Path,
    config: dict = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'score': score
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None
) -> Tuple[nn.Module, int, float]:
    """Load training checkpoint."""
    # weights_only=False needed for PyTorch 2.6+ with numpy scalars in checkpoint
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    score = checkpoint.get('score', 0.0)

    return model, epoch, score


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    # Test weighted R² calculation
    np.random.seed(42)

    # Simulate predictions
    n_samples = 100
    y_true = np.random.rand(n_samples, 5) * 50  # Random ground truth
    y_pred = y_true + np.random.randn(n_samples, 5) * 5  # Predictions with noise

    # Test weighted R²
    r2 = weighted_r2_score(y_true, y_pred)
    print(f"Weighted R²: {r2:.4f}")

    # Test per-target R²
    per_target = per_target_r2(y_true, y_pred)
    print(f"\nPer-target R²:")
    for target, score in per_target.items():
        print(f"  {target}: {score:.4f}")

    # Verify with perfect predictions
    r2_perfect = weighted_r2_score(y_true, y_true)
    print(f"\nPerfect predictions R²: {r2_perfect:.4f}")

    # Test with constant predictions (should give low score)
    y_pred_const = np.ones_like(y_true) * y_true.mean()
    r2_const = weighted_r2_score(y_true, y_pred_const)
    print(f"Constant predictions R²: {r2_const:.4f}")
