"""
Advanced Training Pipeline with Proper MGDA, GradNorm, PCGrad, and MLflow.

This module implements the ACTUAL gradient balancing algorithms as described in:
- MTL Survey: https://hav4ik.github.io/articles/mtl-a-practical-survey
- MGDA Paper: https://arxiv.org/abs/1810.04650

Key Algorithm (MGDA):
1. For each task t: compute ∇_{θ^sh} L^t (T backward passes)
2. Solve: min ||Σ λ^t ∇L^t||² s.t. Σλ^t=1, λ^t≥0 (Frank-Wolfe)
3. Apply: θ^sh ← θ^sh - η * Σ λ^t ∇L^t

References:
- MGDA: https://arxiv.org/abs/1810.04650
- GradNorm: https://arxiv.org/abs/1711.02257
- PCGrad: https://arxiv.org/abs/2001.06782
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
    MGDAOptimizer, GradNormOptimizer, PCGradOptimizer, DWAOptimizer,
    UncertaintyWeighting, create_gradient_optimizer
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
    Advanced trainer implementing proper MGDA/GradNorm/PCGrad.

    MGDA Algorithm (from survey):
    - Step 1: Compute per-task gradients via T backward passes
    - Step 2: Solve min-norm problem via Frank-Wolfe
    - Step 3: Apply weighted gradient to shared parameters
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

        # Optional model compilation
        if compile_model_flag:
            self.model = compile_model(self.model, mode="reduce-overhead")

        # Loss function
        if config.training.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif config.training.loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=config.training.huber_delta, reduction='mean')
        else:
            self.loss_fn = nn.SmoothL1Loss(reduction='mean')

        # Setup gradient optimizer based on method
        self._setup_gradient_optimizer()

        # Main optimizer for all parameters
        param_groups = self.model.get_param_groups(config)

        # Add UncertaintyWeighting parameters to optimizer if using uncertainty method
        if isinstance(self.grad_optimizer, UncertaintyWeighting):
            # Move uncertainty module to device
            self.grad_optimizer = self.grad_optimizer.to(self.device)
            # Add its parameters with a separate learning rate
            param_groups.append({
                'params': list(self.grad_optimizer.parameters()),
                'lr': config.training.learning_rate,  # Same LR as main model
                'weight_decay': 0.0  # No weight decay on log_vars
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

        # Mixed precision - NOTE: MGDA/PCGrad don't work well with AMP scaler
        # because they compute gradients manually
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp and not self.uses_manual_grads else None

        # Mixup (disabled for proper gradient computation)
        # Mixup complicates gradient balancing - disable for MGDA/PCGrad
        self.use_mixup = config.augmentation.mixup_alpha > 0 and not self.uses_manual_grads
        if self.use_mixup:
            self.mixup = MixupCutmix(
                mixup_alpha=config.augmentation.mixup_alpha,
                cutmix_alpha=config.augmentation.cutmix_alpha
            )

        # Tracking
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.global_step = 0

    def _setup_gradient_optimizer(self):
        """Setup the gradient balancing optimizer."""
        method = self.config.gradient_balancing.method.lower()
        self.gradient_method = method

        # Methods that compute gradients manually (no loss.backward())
        self.uses_manual_grads = method in ['mgda', 'pcgrad']

        # Get shared parameters (backbone + fusion layer)
        self.shared_params = list(self.model.backbone.parameters()) + \
                             list(self.model.fusion.parameters())

        self.logger.info(f"Gradient balancing method: {method}")
        self.logger.info(f"Shared parameters: {sum(p.numel() for p in self.shared_params):,}")

        if method == 'mgda':
            self.grad_optimizer = MGDAOptimizer(
                shared_params=self.shared_params,
                normalize_grads=self.config.gradient_balancing.mgda_normalize,
                rescale_grads=self.config.gradient_balancing.mgda_rescale
            )
            self.logger.info("Using MGDA with Frank-Wolfe solver (proper implementation)")
            self.logger.info(f"  normalize_grads={self.config.gradient_balancing.mgda_normalize}, rescale_grads={self.config.gradient_balancing.mgda_rescale}")

        elif method == 'pcgrad':
            self.grad_optimizer = PCGradOptimizer(shared_params=self.shared_params)
            self.logger.info("Using PCGrad with gradient projection")

        elif method == 'gradnorm':
            self.grad_optimizer = GradNormOptimizer(
                n_tasks=3,
                shared_layer=self.model.fusion,
                alpha=self.config.gradient_balancing.gradnorm_alpha,
                lr=self.config.gradient_balancing.gradnorm_weight_lr
            )
            self.logger.info(f"Using GradNorm with alpha={self.config.gradient_balancing.gradnorm_alpha}")

        elif method == 'dwa':
            self.grad_optimizer = DWAOptimizer(
                n_tasks=3,
                temperature=self.config.gradient_balancing.dwa_temperature
            )
            self.logger.info("Using Dynamic Weight Averaging")

        elif method == 'uncertainty':
            self.grad_optimizer = UncertaintyWeighting(
                n_tasks=3,
                init_log_var=0.0  # Start with equal weights (σ=1)
            )
            self.logger.info("Using Uncertainty Weighting (Kendall et al.)")
            self.logger.info("  L = Σ (1/2σ²) * L_i + log(σ_i)")

        else:
            # Fallback to standard weighted loss
            self.grad_optimizer = None
            self.uses_manual_grads = False
            self.logger.info("Using standard loss weighting")

    def _soft_clamp(
        self,
        x: torch.Tensor,
        min_val: float = -2.0,
        max_val: float = 8.0
    ) -> torch.Tensor:
        """
        Soft clamp that preserves gradients everywhere.

        CRITICAL: torch.clamp has ZERO gradient outside its range!
        When model outputs are extreme (e.g., -18 to +18) but get clamped,
        the model can't learn because gradients are zero.

        This soft clamp uses a smooth tanh-based transformation:
        - Maps x to (min_val, max_val) smoothly
        - Always has non-zero gradient
        - Steeper near the boundaries
        """
        range_size = max_val - min_val
        center = (max_val + min_val) / 2

        # Normalize to [-1, 1] range, apply tanh, denormalize
        normalized = (x - center) / (range_size / 2)
        # Use a scaled tanh that is nearly linear in valid range
        # but smoothly saturates outside
        soft = torch.tanh(normalized * 0.5) * 1.2  # Slight stretch for linearity in center
        return center + soft * (range_size / 2)

    def _compute_task_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute individual task losses."""
        task_losses = []
        # Use soft clamp to preserve gradients (critical fix!)
        # Hard clamp (torch.clamp) has zero gradient outside range, blocking learning
        # Use same range as inference: [-1, 5.5] (log space, matches expm1 clamp)
        predictions = self._soft_clamp(predictions, min_val=-1.0, max_val=5.5)
        for i in range(3):
            task_loss = self.loss_fn(predictions[:, i], targets[:, i])
            task_losses.append(task_loss)
        return task_losses

    def train_epoch_mgda(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch using MGDA/PCGrad (manual gradient computation).

        Algorithm:
        1. Forward pass to get predictions
        2. Compute per-task losses
        3. MGDA: compute per-task gradients → solve min-norm → apply weighted grad
        4. Optimizer step
        """
        self.model.train()

        losses = AverageMeter()
        loss_green = AverageMeter()
        loss_dead = AverageMeter()
        loss_clover = AverageMeter()
        grad_info_accum = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
             # ADD THIS DEBUG LINE (only first batch):
            if step == 0:
                print(f"TRAIN targets (log space): min={targets.min().item():.2f}, max={targets.max().item():.2f}")
            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (with autocast for speed, gradients computed outside)
            with autocast(enabled=self.use_amp):
                output = self.model(images, metadata)
                predictions = output['base_preds']

            # Compute per-task losses (outside autocast for gradient computation)
            # Convert to float32 for stable gradient computation
            predictions_fp32 = predictions.float()
            targets_fp32 = targets.float()
            task_losses = self._compute_task_losses(predictions_fp32, targets_fp32)

            # Apply gradient balancing (computes and applies gradients to shared params)
            info = self.grad_optimizer.step(task_losses, retain_graph=True)

            # Compute gradients for task-specific parameters
            # (heads get standard gradients from sum of losses)
            total_loss = sum(task_losses)
            head_params = list(self.model.head_green.parameters()) + \
                          list(self.model.head_dead.parameters()) + \
                          list(self.model.head_clover.parameters())

            head_grads = torch.autograd.grad(
                total_loss, head_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

            # Apply head gradients
            for param, grad in zip(head_params, head_grads):
                if grad is not None:
                    param.grad = grad

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Update metrics
            batch_size = images.size(0)
            losses.update(total_loss.item(), batch_size)
            loss_green.update(task_losses[0].item(), batch_size)
            loss_dead.update(task_losses[1].item(), batch_size)
            loss_clover.update(task_losses[2].item(), batch_size)

            # Accumulate gradient info
            for k, v in info.items():
                if k not in grad_info_accum:
                    grad_info_accum[k] = []
                grad_info_accum[k].append(v)

            # Clear cache periodically
            if self.config.gpu.empty_cache_freq > 0 and (step + 1) % self.config.gpu.empty_cache_freq == 0:
                clear_gpu_cache()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # Average gradient info
        avg_info = {k: np.mean(v) for k, v in grad_info_accum.items()}

        return {
            'loss': losses.avg,
            'loss_green': loss_green.avg,
            'loss_dead': loss_dead.avg,
            'loss_clover': loss_clover.avg,
            **avg_info
        }

    def train_epoch_weighted(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch using weighted loss (GradNorm, DWA, or standard).

        Uses normal loss.backward() with dynamically computed weights.
        """
        self.model.train()

        losses = AverageMeter()
        loss_green = AverageMeter()
        loss_dead = AverageMeter()
        loss_clover = AverageMeter()
        weight_accum = {f'weight_task{i}': [] for i in range(3)}

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            metadata = batch['metadata'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            if self.config.gpu.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # Apply mixup if enabled
            if self.use_mixup:
                images, targets_a, targets_b, lam = self.mixup(images, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                output = self.model(images, metadata)
                predictions = output['base_preds']

                # Compute task losses
                task_losses_a = self._compute_task_losses(predictions, targets_a)
                task_losses_b = self._compute_task_losses(predictions, targets_b)

                # Get weights and compute weighted loss
                if isinstance(self.grad_optimizer, GradNormOptimizer):
                    weighted_loss_a, info_a = self.grad_optimizer.step(task_losses_a)
                    weighted_loss_b, info_b = self.grad_optimizer.step(task_losses_b)
                    loss = lam * weighted_loss_a + (1 - lam) * weighted_loss_b
                    info = info_a

                elif isinstance(self.grad_optimizer, DWAOptimizer):
                    weights, info = self.grad_optimizer.get_weights(task_losses_a)
                    loss_a = sum(w * l for w, l in zip(weights, task_losses_a))
                    loss_b = sum(w * l for w, l in zip(weights, task_losses_b))
                    loss = lam * loss_a + (1 - lam) * loss_b

                elif isinstance(self.grad_optimizer, UncertaintyWeighting):
                    # Uncertainty weighting: learns task-specific uncertainty σ
                    # L = Σ (1/2σ²) * L_i + log(σ_i)
                    loss_a, info_a = self.grad_optimizer(task_losses_a)
                    loss_b, info_b = self.grad_optimizer(task_losses_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    info = info_a

                else:
                    # Standard equal weighting
                    loss_a = sum(task_losses_a) / 3
                    loss_b = sum(task_losses_b) / 3
                    loss = lam * loss_a + (1 - lam) * loss_b
                    info = {f'weight_task{i}': 1/3 for i in range(3)}

                loss = loss / self.config.training.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
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
            loss_green.update(task_losses_a[0].item(), batch_size)
            loss_dead.update(task_losses_a[1].item(), batch_size)
            loss_clover.update(task_losses_a[2].item(), batch_size)

            # Track weights
            for i in range(3):
                weight_accum[f'weight_task{i}'].append(info.get(f'weight_task{i}', 1/3))

            # Clear cache periodically
            if self.config.gpu.empty_cache_freq > 0 and (step + 1) % self.config.gpu.empty_cache_freq == 0:
                clear_gpu_cache()

            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_weights = {k: np.mean(v) for k, v in weight_accum.items()}

        return {
            'loss': losses.avg,
            'loss_green': loss_green.avg,
            'loss_dead': loss_dead.avg,
            'loss_clover': loss_clover.avg,
            **avg_weights
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Route to appropriate training method."""
        if self.uses_manual_grads:
            return self.train_epoch_mgda(epoch)
        else:
            return self.train_epoch_weighted(epoch)

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

            with autocast(enabled=self.use_amp):
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
        # DEBUG: Print sample predictions vs targets
        print("\n=== DEBUG R² ===")
        print(f"Predictions shape: {all_predictions.shape}")
        print(f"Targets shape: {all_targets.shape}")
        print(f"Predictions sample [0:3]:\n{all_predictions[:3]}")
        print(f"Targets sample [0:3]:\n{all_targets[:3]}")
        print(f"Predictions min/max: {all_predictions.min():.2f} / {all_predictions.max():.2f}")
        print(f"Targets min/max: {all_targets.min():.2f} / {all_targets.max():.2f}")
        print(f"Predictions mean per col: {all_predictions.mean(axis=0)}")
        print(f"Targets mean per col: {all_targets.mean(axis=0)}")
        print("=================\n")

        return metrics, weighted_r2

    def train(self) -> Dict[str, float]:
        """Full training loop."""
        # Start MLflow run
        if self.tracker is not None:
            self.tracker.start_run()
            self.tracker.log_config(self.config)

        self.logger.info("=" * 50)
        self.logger.info(f"Starting training for fold {self.fold}")
        self.logger.info(f"Gradient method: {self.gradient_method}")
        self.logger.info(f"Uses manual gradients: {self.uses_manual_grads}")
        self.logger.info("=" * 50)

        start_time = time.time()
        total_time = 0

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

                # Log task weights / MGDA info
                if 'weight_task0' in train_metrics:
                    weight_info = " | ".join([
                        f"w{i}: {train_metrics.get(f'weight_task{i}', 0.33):.3f}"
                        for i in range(3)
                    ])
                    self.logger.info(f"  Task weights: {weight_info}")

                if 'min_norm' in train_metrics:
                    self.logger.info(f"  MGDA min_norm: {train_metrics['min_norm']:.4f}")

                if 'avg_grad_norm' in train_metrics:
                    self.logger.info(f"  Avg gradient norm: {train_metrics['avg_grad_norm']:.4f}")

                if 'n_conflicts' in train_metrics:
                    self.logger.info(f"  PCGrad conflicts: {train_metrics['n_conflicts']:.1f}")

                # Log uncertainty weighting info (σ values)
                if 'sigma_task0' in train_metrics:
                    sigma_info = " | ".join([
                        f"σ{i}: {train_metrics.get(f'sigma_task{i}', 1.0):.3f}"
                        for i in range(3)
                    ])
                    self.logger.info(f"  Uncertainty σ: {sigma_info}")

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

                # Early stopping
                if self.early_stopping(val_score):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            total_time = time.time() - start_time

            self.logger.info("=" * 50)
            self.logger.info(f"Training completed in {format_time(total_time)}")
            self.logger.info(f"Best R²: {self.best_score:.4f} at epoch {self.best_epoch}")

            # Log final metrics to MLflow
            if self.tracker is not None:
                self.tracker.log_metric("best_r2", self.best_score)
                self.tracker.log_metric("best_epoch", self.best_epoch)
                self.tracker.log_metric("total_time_seconds", total_time)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
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
    """Train with cross-validation."""
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

    parser = argparse.ArgumentParser(description="Advanced Training with Proper MGDA")
    parser.add_argument('--backbone', type=str, default='convnext_base')
    parser.add_argument('--gradient_method', type=str, default='mgda',
                        choices=['equal', 'competition', 'mgda', 'gradnorm', 'pcgrad', 'dwa', 'uncertainty'])
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_mlflow', action='store_true')
    args = parser.parse_args()

    config = get_config(args.backbone)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed
    config.gradient_balancing.method = args.gradient_method
    config.mlflow.enabled = not args.no_mlflow

    if args.fold is not None:
        result = train_fold_advanced(config, args.fold)
        print(f"Fold {args.fold} Best R²: {result['best_score']:.4f}")
    else:
        result = train_cv_advanced(config)
        print(f"CV Mean R²: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
