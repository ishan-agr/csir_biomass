"""
Advanced Training Pipeline with MGDA, GradNorm, PCGrad, and MLflow.

This module extends the base training with:
1. Advanced gradient balancing (MGDA, GradNorm, PCGrad, CAGrad, DWA)
2. MLflow experiment tracking with comprehensive logging
3. GPU profiling and system metrics
4. Experiment comparison and analysis

Based on winning strategies from:
- BioMassters Competition (1st place: U-Net + TTA)
- Multi-Task Learning Survey (Sener & Koltun, NeurIPS 2018)
- GradNorm (Chen et al., ICML 2018)
- PCGrad (Yu et al., NeurIPS 2020)

References:
- MGDA: https://arxiv.org/abs/1810.04650
- GradNorm: https://arxiv.org/abs/1711.02257
- PCGrad: https://arxiv.org/abs/2001.06782
- BioMassters: https://github.com/drivendataorg/the-biomassters
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
from dataclasses import asdict

from config import Config, get_config
from dataset import get_dataloaders, MixupCutmix
from model import BiomassModel, MultiTaskLoss, create_model
from gradient_balancing import (
    GradientBalancer,
    MGDA, GradNorm, PCGrad, CAGrad, DynamicWeightAveraging,
    get_gradient_balancer
)
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

# MLflow imports with fallback
try:
    from mlflow_tracking import MLflowTracker, create_tracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow tracking not available")

warnings.filterwarnings('ignore')


class AdvancedTrainer:
    """
    Advanced trainer with gradient balancing and MLflow tracking.

    Features:
    - MGDA, GradNorm, PCGrad, CAGrad, DWA gradient balancing
    - MLflow experiment tracking
    - Comprehensive metrics logging
    - GPU profiling
    """

    def __init__(
        self,
        config: Config,
        fold: int = 0,
        compile_model_flag: bool = False,
        run_name: Optional[str] = None
    ):
        self.config = config
        self.fold = fold

        # Setup GPU
        self.device = setup_gpu(config.gpu)

        # Set seed
        set_seed(config.training.seed, deterministic=config.gpu.cudnn_deterministic)

        # Setup logging
        self.logger = setup_logging(
            config.training.save_dir / "logs",
            name=f"fold_{fold}"
        )
        self.logger.info(f"Config: {config}")

        # Setup MLflow tracking
        self.tracker = None
        if config.mlflow.enabled and MLFLOW_AVAILABLE:
            self.tracker = create_tracker(
                experiment_name=config.mlflow.experiment_name,
                tracking_uri=config.mlflow.tracking_uri,
                run_name=run_name or f"fold{fold}_{config.model.backbone}_{config.gradient_balancing.method}",
                tags={
                    "fold": str(fold),
                    "backbone": config.model.backbone,
                    "gradient_method": config.gradient_balancing.method,
                    **config.mlflow.tags
                },
                log_system_metrics=config.mlflow.log_system_metrics,
                log_gpu_metrics=config.mlflow.log_gpu_metrics
            )

        # Print GPU info
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")

        # Create dataloaders
        self.logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config, fold)

        # Create model
        self.logger.info(f"Creating model with backbone: {config.model.backbone}")
        self.model = create_model(config)
        self.model = self.model.to(self.device)

        # Use channels_last memory format
        if config.gpu.channels_last and torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
            self.logger.info("Using channels_last memory format")

        # Optional model compilation
        if compile_model_flag:
            self.model = compile_model(self.model, mode="reduce-overhead")

        # Base loss function (for non-gradient-balancing methods)
        self.criterion = MultiTaskLoss(config)

        # Setup gradient balancer
        self._setup_gradient_balancer()

        # Optimizer
        param_groups = self.model.get_param_groups(config)

        # Add GradNorm weights to optimizer if using GradNorm
        if isinstance(self.gradient_balancer, GradNorm):
            param_groups.append({
                'params': [self.gradient_balancer.log_weights],
                'lr': config.gradient_balancing.gradnorm_weight_lr
            })

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
            mode='max',
            verbose=True
        )

        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None

        # Mixup
        self.mixup = MixupCutmix(
            mixup_alpha=config.augmentation.mixup_alpha,
            cutmix_alpha=config.augmentation.cutmix_alpha
        )

        # Tracking
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.global_step = 0

    def _setup_gradient_balancer(self):
        """Setup gradient balancing method."""
        gb_config = self.config.gradient_balancing
        method = gb_config.method.lower()

        self.use_advanced_balancing = method in ['mgda', 'gradnorm', 'pcgrad', 'cagrad', 'dwa']

        if self.use_advanced_balancing:
            self.logger.info(f"Using advanced gradient balancing: {method}")

            if method == 'mgda':
                self.gradient_balancer = MGDA(
                    normalize_grads=gb_config.mgda_normalize,
                    use_rep_grad=gb_config.mgda_use_rep_grad
                )
            elif method == 'gradnorm':
                self.gradient_balancer = GradNorm(
                    n_tasks=3,  # 3 base targets
                    alpha=gb_config.gradnorm_alpha,
                    weight_lr=gb_config.gradnorm_weight_lr
                )
            elif method == 'pcgrad':
                self.gradient_balancer = PCGrad(
                    reduction=gb_config.pcgrad_reduction
                )
            elif method == 'cagrad':
                self.gradient_balancer = CAGrad(
                    c=gb_config.cagrad_c,
                    rescale=gb_config.cagrad_rescale
                )
            elif method == 'dwa':
                self.gradient_balancer = DynamicWeightAveraging(
                    n_tasks=3,
                    temperature=gb_config.dwa_temperature
                )
        else:
            self.gradient_balancer = None
            self.logger.info(f"Using basic loss weighting: {method}")

    def _compute_task_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute individual task losses."""
        # predictions: (batch, 3), targets: (batch, 3)
        losses = []

        # Get base loss function
        if self.config.training.loss_type == "mse":
            loss_fn = nn.MSELoss(reduction='mean')
        elif self.config.training.loss_type == "huber":
            loss_fn = nn.HuberLoss(delta=self.config.training.huber_delta, reduction='mean')
        else:
            loss_fn = nn.SmoothL1Loss(reduction='mean')

        for i in range(3):
            task_loss = loss_fn(predictions[:, i], targets[:, i])
            losses.append(task_loss)

        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with advanced gradient balancing."""
        self.model.train()

        losses = AverageMeter()
        loss_green = AverageMeter()
        loss_dead = AverageMeter()
        loss_clover = AverageMeter()

        # Track gradient balancing info
        grad_info_accum = {}

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Apply mixup
            if self.config.augmentation.mixup_alpha > 0:
                images, targets_a, targets_b, lam = self.mixup(images, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            # Forward pass with AMP
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(images, metadata, return_features=True)
                predictions = output['base_preds']

                if self.use_advanced_balancing:
                    # Compute individual task losses
                    task_losses_a = self._compute_task_losses(predictions, targets_a)
                    task_losses_b = self._compute_task_losses(predictions, targets_b)

                    # Apply gradient balancing
                    # Get shared parameters (backbone + fusion)
                    shared_params = list(self.model.backbone.parameters()) + \
                                    list(self.model.fusion.parameters())

                    # For MGDA, we can use representation gradients
                    representations = output.get('fused_features', None)

                    # Get last shared layer for GradNorm
                    last_shared_layer = self.model.fusion

                    # Compute balanced loss
                    if isinstance(self.gradient_balancer, GradNorm):
                        loss_a, grad_info_a = self.gradient_balancer.balance(
                            task_losses_a, shared_params,
                            last_shared_layer=last_shared_layer
                        )
                        loss_b, grad_info_b = self.gradient_balancer.balance(
                            task_losses_b, shared_params,
                            last_shared_layer=last_shared_layer
                        )
                        grad_info = grad_info_a
                    elif isinstance(self.gradient_balancer, MGDA):
                        loss_a, grad_info_a = self.gradient_balancer.balance(
                            task_losses_a, shared_params,
                            representations=representations
                        )
                        loss_b, grad_info_b = self.gradient_balancer.balance(
                            task_losses_b, shared_params,
                            representations=representations
                        )
                        grad_info = grad_info_a
                    elif isinstance(self.gradient_balancer, (PCGrad, CAGrad)):
                        # PCGrad/CAGrad modify gradients directly
                        loss_a, grad_info_a = self.gradient_balancer.balance(
                            task_losses_a, shared_params
                        )
                        loss_b, grad_info_b = self.gradient_balancer.balance(
                            task_losses_b, shared_params
                        )
                        grad_info = grad_info_a
                    else:
                        loss_a, grad_info_a = self.gradient_balancer.balance(
                            task_losses_a, shared_params
                        )
                        loss_b, grad_info_b = self.gradient_balancer.balance(
                            task_losses_b, shared_params
                        )
                        grad_info = grad_info_a

                    loss = lam * loss_a + (1 - lam) * loss_b

                    # Accumulate gradient info
                    for k, v in grad_info.items():
                        if k not in grad_info_accum:
                            grad_info_accum[k] = []
                        grad_info_accum[k].append(v)
                else:
                    # Use standard loss
                    loss_a, loss_dict_a = self.criterion(predictions, targets_a)
                    loss_b, loss_dict_b = self.criterion(predictions, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b

                loss = loss / self.config.training.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.training.accumulation_steps == 0:
                if self.scaler is not None:
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
                self.global_step += 1

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item() * self.config.training.accumulation_steps, batch_size)

            # Compute individual task losses for logging
            with torch.no_grad():
                task_losses = self._compute_task_losses(predictions, targets_a)
                loss_green.update(task_losses[0].item(), batch_size)
                loss_dead.update(task_losses[1].item(), batch_size)
                loss_clover.update(task_losses[2].item(), batch_size)

            # Clear GPU cache periodically
            if self.config.gpu.empty_cache_freq > 0 and (step + 1) % self.config.gpu.empty_cache_freq == 0:
                clear_gpu_cache()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # Average gradient info
        avg_grad_info = {k: np.mean(v) for k, v in grad_info_accum.items()}

        return {
            'loss': losses.avg,
            'loss_green': loss_green.avg,
            'loss_dead': loss_dead.avg,
            'loss_clover': loss_clover.avg,
            **avg_grad_info
        }

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], float]:
        """Validate on validation set."""
        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)

            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            with autocast(enabled=self.config.training.use_amp):
                predictions = self.model.predict_all_targets(
                    images, metadata,
                    use_log_transform=self.config.training.use_log_transform
                )

            all_predictions.append(predictions.cpu().numpy())

            if 'raw_targets' in batch:
                all_targets.append(batch['raw_targets'].numpy())

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
        """Full training loop with MLflow tracking."""
        # Start MLflow run
        if self.tracker is not None:
            self.tracker.start_run()
            self.tracker.log_config(self.config)

        self.logger.info("=" * 50)
        self.logger.info(f"Starting training for fold {self.fold}")
        self.logger.info(f"Gradient balancing: {self.config.gradient_balancing.method}")
        self.logger.info(f"Epochs: {self.config.training.epochs}")
        self.logger.info("=" * 50)

        start_time = time.time()

        try:
            for epoch in range(1, self.config.training.epochs + 1):
                epoch_start = time.time()

                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate
                val_metrics, val_score = self.validate()

                epoch_time = time.time() - epoch_start

                # Log to MLflow
                if self.tracker is not None:
                    self.tracker.log_epoch_metrics(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        lr=self.optimizer.param_groups[0]['lr']
                    )

                    # Log gradient balancing info
                    grad_keys = [k for k in train_metrics.keys()
                                 if k.startswith('weight_') or k.startswith('min_norm')
                                 or k.startswith('n_conflicts')]
                    if grad_keys:
                        grad_info = {k: train_metrics[k] for k in grad_keys}
                        self.tracker.log_gradients_info(grad_info, epoch)

                # Log to console
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

                # Log gradient balancing weights
                if self.use_advanced_balancing:
                    weight_info = " | ".join([
                        f"{k}: {v:.3f}" for k, v in train_metrics.items()
                        if k.startswith('weight_task')
                    ])
                    if weight_info:
                        self.logger.info(f"  Task weights: {weight_info}")

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

                    # Log checkpoint to MLflow
                    if self.tracker is not None and self.config.mlflow.log_checkpoints:
                        self.tracker.log_checkpoint(checkpoint_path)

                # Early stopping
                if self.early_stopping(val_score):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            total_time = time.time() - start_time

            # Log final results
            self.logger.info("=" * 50)
            self.logger.info(f"Training completed in {format_time(total_time)}")
            self.logger.info(f"Best R²: {self.best_score:.4f} at epoch {self.best_epoch}")

            # Log model to MLflow
            if self.tracker is not None:
                self.tracker.log_metric("best_r2", self.best_score)
                self.tracker.log_metric("best_epoch", self.best_epoch)
                self.tracker.log_metric("total_time_seconds", total_time)

                if self.config.mlflow.log_model:
                    # Load best model and log
                    best_path = self.config.training.save_dir / f"best_fold{self.fold}.pt"
                    if best_path.exists():
                        checkpoint = torch.load(best_path, map_location='cpu')
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.tracker.log_model(self.model, artifact_path=f"model_fold{self.fold}")

        finally:
            # End MLflow run
            if self.tracker is not None:
                self.tracker.end_run()

        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_time': total_time
        }


def train_fold_advanced(
    config: Config,
    fold: int,
    compile_model_flag: bool = False,
    run_name: Optional[str] = None
) -> Dict[str, float]:
    """Train a single fold with advanced features."""
    trainer = AdvancedTrainer(
        config, fold,
        compile_model_flag=compile_model_flag,
        run_name=run_name
    )
    return trainer.train()


def train_cv_advanced(
    config: Config,
    compile_model_flag: bool = False
) -> Dict[str, float]:
    """Train with cross-validation and advanced features."""
    results = []

    for fold in range(config.training.n_folds):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{config.training.n_folds}")
        print(f"{'='*50}\n")

        clear_gpu_cache()

        fold_result = train_fold_advanced(
            config, fold,
            compile_model_flag=compile_model_flag,
            run_name=f"cv_fold{fold}_{config.model.backbone}_{config.gradient_balancing.method}"
        )
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

    parser = argparse.ArgumentParser(description="Advanced Training with Gradient Balancing")
    parser.add_argument('--backbone', type=str, default='convnext_base',
                        choices=['convnext_base', 'convnext_large', 'efficientnetv2_m',
                                 'swin_base_patch4_window12_384'])
    parser.add_argument('--gradient_method', type=str, default='mgda',
                        choices=['equal', 'competition', 'uncertainty',
                                 'mgda', 'gradnorm', 'pcgrad', 'cagrad', 'dwa'])
    parser.add_argument('--fold', type=int, default=None,
                        help='Train single fold (0-4)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_mlflow', action='store_true', help='Disable MLflow tracking')
    parser.add_argument('--experiment_name', type=str, default='csiro_biomass')
    args = parser.parse_args()

    # Create config
    config = get_config(args.backbone)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed
    config.gradient_balancing.method = args.gradient_method
    config.mlflow.enabled = not args.no_mlflow
    config.mlflow.experiment_name = args.experiment_name

    # Train
    if args.fold is not None:
        result = train_fold_advanced(config, args.fold)
        print(f"Fold {args.fold} Best R²: {result['best_score']:.4f}")
    else:
        result = train_cv_advanced(config)
        print(f"CV Mean R²: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
