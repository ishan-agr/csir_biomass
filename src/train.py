"""
Training Pipeline for CSIRO Pasture Biomass Prediction.

Features:
- Multi-task learning with weighted loss
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling with warmup
- Early stopping
- Comprehensive logging and checkpointing
- Cross-validation support
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from config import Config, get_config
from dataset import get_dataloaders, MixupCutmix
from model import BiomassModel, MultiTaskLoss, create_model
from utils import (
    set_seed,
    setup_gpu,
    clear_gpu_cache,
    print_gpu_memory,
    to_device,
    compile_model,
    weighted_r2_score,
    per_target_r2,
    EarlyStopping,
    AverageMeter,
    CosineAnnealingWarmupRestarts,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    format_time,
    TARGET_ORDER
)

warnings.filterwarnings('ignore')


def move_batch_to_device(batch: Dict, device: torch.device, non_blocking: bool = True) -> Dict:
    """Move batch tensors to device with non-blocking transfers."""
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class Trainer:
    """
    Trainer class for biomass prediction model.
    """

    def __init__(self, config: Config, fold: int = 0, compile_model_flag: bool = False):
        self.config = config
        self.fold = fold

        # Setup GPU for maximum utilization
        self.device = setup_gpu(config.gpu)

        # Set seed (non-deterministic for speed)
        set_seed(config.training.seed, deterministic=config.gpu.cudnn_deterministic)

        # Setup logging
        self.logger = setup_logging(
            config.training.save_dir / "logs",
            name=f"fold_{fold}"
        )
        self.logger.info(f"Config: {config}")

        # Print GPU info
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")

        # Create dataloaders with optimized settings
        self.logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config, fold)

        # Create model
        self.logger.info(f"Creating model with backbone: {config.model.backbone}")
        self.model = create_model(config)

        # Move to device and apply memory format optimization
        self.model = self.model.to(self.device)

        # Use channels_last memory format for faster CNN operations
        if config.gpu.channels_last and torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
            self.logger.info("Using channels_last memory format (NHWC)")

        # Optional: Compile model for faster execution (PyTorch 2.0+)
        if compile_model_flag:
            self.model = compile_model(self.model, mode="reduce-overhead")
            self.logger.info("Model compiled with torch.compile")

        # Loss function
        self.criterion = MultiTaskLoss(config)

        # Optimizer
        param_groups = self.model.get_param_groups(config)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Scheduler
        steps_per_epoch = len(self.train_loader) // config.training.accumulation_steps
        total_steps = steps_per_epoch * config.training.epochs
        warmup_steps = steps_per_epoch * config.training.warmup_epochs

        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=total_steps,
            max_lr=config.training.learning_rate,
            min_lr=config.training.min_lr,
            warmup_steps=warmup_steps
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max',  # Higher R² is better
            verbose=True
        )

        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None

        # Mixup/Cutmix
        self.mixup = MixupCutmix(
            mixup_alpha=config.augmentation.mixup_alpha,
            cutmix_alpha=config.augmentation.cutmix_alpha
        )

        # Tracking
        self.best_score = -float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        losses = AverageMeter()
        loss_green = AverageMeter()
        loss_dead = AverageMeter()
        loss_clover = AverageMeter()

        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device with non-blocking transfers (requires pinned memory)
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            # Apply channels_last format to input if enabled
            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Apply mixup/cutmix
            if self.config.augmentation.mixup_alpha > 0:
                images, targets_a, targets_b, lam = self.mixup(images, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(images, metadata)
                predictions = output['base_preds']

                # Compute loss for both mixup components
                loss_a, loss_dict_a = self.criterion(predictions, targets_a)
                loss_b, loss_dict_b = self.criterion(predictions, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b

                # Scale for gradient accumulation
                loss = loss / self.config.training.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.training.accumulation_steps == 0:
                if self.scaler is not None:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item() * self.config.training.accumulation_steps, batch_size)
            loss_green.update(loss_dict_a['loss_green'].item(), batch_size)
            loss_dead.update(loss_dict_a['loss_dead'].item(), batch_size)
            loss_clover.update(loss_dict_a['loss_clover'].item(), batch_size)

            # Periodically clear GPU cache to prevent memory fragmentation
            if self.config.gpu.empty_cache_freq > 0 and (step + 1) % self.config.gpu.empty_cache_freq == 0:
                clear_gpu_cache()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        return {
            'loss': losses.avg,
            'loss_green': loss_green.avg,
            'loss_dead': loss_dead.avg,
            'loss_clover': loss_clover.avg
        }

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        """Validate on validation set."""
        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Non-blocking transfers
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)

            # Apply channels_last format
            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Get predictions for all 5 targets
            with autocast(enabled=self.config.training.use_amp):
                predictions = self.model.predict_all_targets(
                    images, metadata,
                    use_log_transform=self.config.training.use_log_transform
                )

            all_predictions.append(predictions.cpu().numpy())

            # Get raw targets (not log-transformed)
            if 'raw_targets' in batch:
                all_targets.append(batch['raw_targets'].numpy())

        # Concatenate
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute weighted R²
        weighted_r2 = weighted_r2_score(all_targets, all_predictions)

        # Per-target R²
        per_target = per_target_r2(all_targets, all_predictions)

        metrics = {
            'weighted_r2': weighted_r2,
            **{f'r2_{k}': v for k, v in per_target.items()}
        }

        return metrics, weighted_r2

    def train(self) -> Dict[str, float]:
        """
        Full training loop.

        Returns:
            Dictionary with best metrics
        """
        self.logger.info("=" * 50)
        self.logger.info(f"Starting training for fold {self.fold}")
        self.logger.info(f"Epochs: {self.config.training.epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Accumulation steps: {self.config.training.accumulation_steps}")
        self.logger.info("=" * 50)

        start_time = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics, val_score = self.validate()

            epoch_time = time.time() - epoch_start

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{self.config.training.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val R²: {val_score:.4f} | "
                f"Time: {format_time(epoch_time)}"
            )

            # Log per-target R²
            self.logger.info(
                f"  Per-target R²: " +
                " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('r2_')])
            )

            # Save best model
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch

                checkpoint_path = self.config.training.save_dir / f"best_fold{self.fold}.pt"
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    score=val_score,
                    path=checkpoint_path
                )
                self.logger.info(f"  New best model saved! R²: {val_score:.4f}")

            # Early stopping check
            if self.early_stopping(val_score):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best R²: {self.best_score:.4f} at epoch {self.best_epoch}")
        self.logger.info("=" * 50)

        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_time': total_time
        }


def train_fold(config: Config, fold: int, compile_model_flag: bool = False) -> Dict[str, float]:
    """Train a single fold."""
    trainer = Trainer(config, fold, compile_model_flag=compile_model_flag)
    return trainer.train()


def train_cv(config: Config, compile_model_flag: bool = False) -> Dict[str, float]:
    """
    Train with cross-validation.

    Returns:
        Dictionary with CV results
    """
    results = []

    for fold in range(config.training.n_folds):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{config.training.n_folds}")
        print(f"{'='*50}\n")

        # Clear GPU cache between folds
        clear_gpu_cache()

        fold_result = train_fold(config, fold, compile_model_flag=compile_model_flag)
        results.append(fold_result)

    # Aggregate results
    cv_scores = [r['best_score'] for r in results]
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print(f"{'='*50}")
    for fold, score in enumerate(cv_scores):
        print(f"  Fold {fold}: R² = {score:.4f}")
    print(f"  Mean R²: {mean_score:.4f} ± {std_score:.4f}")
    print(f"{'='*50}\n")

    return {
        'mean_r2': mean_score,
        'std_r2': std_score,
        'fold_scores': cv_scores
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base',
                        choices=['convnext_base', 'convnext_large', 'efficientnetv2_m', 'swin_base_patch4_window12_384'])
    parser.add_argument('--fold', type=int, default=None,
                        help='Train single fold (0-4). If None, train all folds.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Create config
    config = get_config(args.backbone)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed

    # Train
    if args.fold is not None:
        # Single fold
        result = train_fold(config, args.fold)
        print(f"Fold {args.fold} Best R²: {result['best_score']:.4f}")
    else:
        # Cross-validation
        result = train_cv(config)
        print(f"CV Mean R²: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
