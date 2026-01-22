"""
Multi-Task Learning Model for Pasture Biomass Prediction.

Architecture:
- Shared backbone (ConvNeXt/EfficientNetV2/Swin Transformer)
- Metadata fusion via learned embeddings
- Task-specific regression heads for 3 base targets
- Derived targets (GDM_g, Dry_Total_g) computed from base predictions

Key Design Decisions:
1. Hard parameter sharing for efficiency
2. Predict 3 independent base targets, derive 2 composite targets
3. Late fusion of image features and metadata
4. Separate heads allow task-specific learned representations

References:
- Multi-Task Learning Survey: https://hav4ik.github.io/articles/mtl-a-practical-survey
- ConvNeXt: https://arxiv.org/abs/2201.03545
- timm library: https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

from config import Config, ModelConfig


class MetadataEncoder(nn.Module):
    """
    Encode metadata features (NDVI, Height, State, Species) into a dense vector.

    Numerical features are normalized externally and passed through MLP.
    Categorical features use learned embeddings.
    """

    def __init__(self, config: Config):
        super().__init__()
        model_cfg = config.model
        data_cfg = config.data

        # Embeddings for categorical features
        self.state_embed = nn.Embedding(
            num_embeddings=len(data_cfg.states),
            embedding_dim=model_cfg.state_embed_dim
        )
        self.species_embed = nn.Embedding(
            num_embeddings=data_cfg.n_species + 1,  # +1 for unknown
            embedding_dim=model_cfg.species_embed_dim
        )

        # MLP for numerical features
        self.numerical_mlp = nn.Sequential(
            nn.Linear(2, 32),  # NDVI + Height
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )

        # Combined output dimension
        self.output_dim = 32 + model_cfg.state_embed_dim + model_cfg.species_embed_dim

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: (batch, 4) tensor [ndvi, height, state_idx, species_idx]

        Returns:
            (batch, output_dim) encoded metadata
        """
        # Split metadata
        numerical = metadata[:, :2]  # NDVI, Height (already normalized)
        state_idx = metadata[:, 2].long()
        species_idx = metadata[:, 3].long()

        # Clamp indices to valid range to prevent CUDA indexing errors
        state_idx = torch.clamp(state_idx, 0, self.state_embed.num_embeddings - 1)
        species_idx = torch.clamp(species_idx, 0, self.species_embed.num_embeddings - 1)

        # Encode
        num_features = self.numerical_mlp(numerical)
        state_features = self.state_embed(state_idx)
        species_features = self.species_embed(species_idx)

        # Concatenate
        return torch.cat([num_features, state_features, species_features], dim=1)


class RegressionHead(nn.Module):
    """
    Task-specific regression head.

    Architecture: MLP with residual connections and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer bias to reasonable starting point
        # This helps the model start in the correct output range
        # 2.5 ≈ log1p(11), which is near the median of typical biomass values
        self._init_output_bias()

    def _init_output_bias(self, bias_value: float = 2.5):
        """Initialize the final layer bias to start predictions in valid range."""
        # Find the last Linear layer
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.bias.fill_(bias_value)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BiomassModel(nn.Module):
    """
    Multi-Task Learning model for biomass prediction.

    Predicts 3 base targets (Dry_Green_g, Dry_Dead_g, Dry_Clover_g)
    and derives 2 composite targets (GDM_g, Dry_Total_g).

    Architecture:
    1. Backbone: Pretrained CNN/Transformer
    2. Metadata Encoder: Embeddings + MLP
    3. Fusion: Concatenation + MLP
    4. Heads: 3 separate regression heads (shared hidden layers option available)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        # Backbone (pretrained)
        self.backbone = self._create_backbone(model_cfg)

        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(config)

        # Fusion layer
        fusion_input_dim = model_cfg.backbone_dim + self.metadata_encoder.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, model_cfg.fusion_hidden_dim),
            nn.BatchNorm1d(model_cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.fusion_dropout),
            nn.Linear(model_cfg.fusion_hidden_dim, model_cfg.fusion_hidden_dim),
            nn.BatchNorm1d(model_cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Task-specific heads (3 base targets)
        self.head_green = RegressionHead(
            input_dim=model_cfg.fusion_hidden_dim,
            hidden_dims=model_cfg.head_hidden_dims,
            output_dim=1,
            dropout=model_cfg.head_dropout
        )
        self.head_dead = RegressionHead(
            input_dim=model_cfg.fusion_hidden_dim,
            hidden_dims=model_cfg.head_hidden_dims,
            output_dim=1,
            dropout=model_cfg.head_dropout
        )
        self.head_clover = RegressionHead(
            input_dim=model_cfg.fusion_hidden_dim,
            hidden_dims=model_cfg.head_hidden_dims,
            output_dim=1,
            dropout=model_cfg.head_dropout
        )

        # Initialize heads
        self._init_weights()

    def _create_backbone(self, model_cfg: ModelConfig) -> nn.Module:
        """Create pretrained backbone using timm."""
        backbone = timm.create_model(
            model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        return backbone

    def _init_weights(self):
        """Initialize head weights."""
        for module in [self.head_green, self.head_dead, self.head_clover, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (batch, 3, H, W) input images
            metadata: (batch, 4) metadata [ndvi, height, state_idx, species_idx]
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with:
            - 'base_preds': (batch, 3) predictions for base targets [Green, Dead, Clover]
            - 'all_preds': (batch, 5) all predictions [Green, Dead, Clover, GDM, Total]
            - 'features': (optional) intermediate features
        """
        # Extract image features
        image_features = self.backbone(images)  # (batch, backbone_dim)

        # Encode metadata
        metadata_features = self.metadata_encoder(metadata)  # (batch, metadata_dim)

        # Fuse features
        combined = torch.cat([image_features, metadata_features], dim=1)
        fused = self.fusion(combined)  # (batch, fusion_hidden_dim)

        # Task-specific predictions
        pred_green = self.head_green(fused).squeeze(-1)  # (batch,)
        pred_dead = self.head_dead(fused).squeeze(-1)
        pred_clover = self.head_clover(fused).squeeze(-1)

        # Stack base predictions
        base_preds = torch.stack([pred_green, pred_dead, pred_clover], dim=1)

        # Derive composite targets (in log space if using log transform)
        # GDM = Green + Clover, Total = Green + Dead + Clover
        # In log space: log(A+B) != log(A) + log(B), so we need to transform
        # We'll compute derived targets in original space during inference

        # For training loss, we only need base predictions
        # For inference, we compute derived targets

        output = {
            'base_preds': base_preds,  # (batch, 3)
        }

        if return_features:
            output['image_features'] = image_features
            output['fused_features'] = fused

        return output

    def predict_all_targets(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        use_log_transform: bool = True
    ) -> torch.Tensor:
        """
        Predict all 5 targets for inference.

        This method:
        1. Gets base predictions (in log space if log transform used)
        2. Converts to original space
        3. Computes derived targets
        4. Returns all 5 predictions

        Args:
            images: (batch, 3, H, W)
            metadata: (batch, 4)
            use_log_transform: Whether model was trained with log transform

        Returns:
            (batch, 5) predictions [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
        """
        output = self.forward(images, metadata)
        base_preds = output['base_preds']  # (batch, 3) in log space
        if not self.training:
            print(f"DEBUG base_preds (log space): min={base_preds.min().item():.2f}, max={base_preds.max().item():.2f}, mean={base_preds.mean().item():.2f}")
        print(f"RAW base_preds: min={base_preds.min().item():.2f}, max={base_preds.max().item():.2f}")
        if use_log_transform:
            # Clamp predictions BEFORE expm1 to prevent explosion
            # log1p(1000) ≈ 6.9, so clamp to reasonable range
            base_preds = torch.clamp(base_preds, min=-1, max=5.5)
            base_preds_orig = torch.expm1(base_preds)
            # Convert from log space to original space
          # expm1(x) = exp(x) - 1
        else:
            base_preds_orig = base_preds

        # Ensure non-negative predictions and reasonable max
        base_preds_orig = torch.clamp(base_preds_orig, min=0, max=500)

        # Extract individual predictions
        pred_green = base_preds_orig[:, 0]
        pred_dead = base_preds_orig[:, 1]
        pred_clover = base_preds_orig[:, 2]

        # Compute derived targets
        pred_gdm = pred_green + pred_clover
        pred_total = pred_green + pred_dead + pred_clover

        # Stack all predictions in correct order
        all_preds = torch.stack([
            pred_green, pred_dead, pred_clover, pred_gdm, pred_total
        ], dim=1)

        return all_preds

    def get_param_groups(self, config: Config) -> List[Dict]:
        """
        Get parameter groups with different learning rates.

        Backbone gets lower LR, heads get full LR.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.metadata_encoder.parameters()) + \
                      list(self.fusion.parameters()) + \
                      list(self.head_green.parameters()) + \
                      list(self.head_dead.parameters()) + \
                      list(self.head_clover.parameters())

        return [
            {
                'params': backbone_params,
                'lr': config.training.learning_rate * config.training.backbone_lr_mult
            },
            {
                'params': head_params,
                'lr': config.training.learning_rate
            }
        ]


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with configurable weighting.

    Supports:
    - Equal weighting
    - Competition weighting (based on target importance)
    - Uncertainty weighting (learnable, Kendall et al.)

    Reference: https://arxiv.org/abs/1705.07115 (Uncertainty Weighting)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        training_cfg = config.training

        # Base loss function
        if training_cfg.loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif training_cfg.loss_type == "huber":
            self.base_loss = nn.HuberLoss(delta=training_cfg.huber_delta, reduction='none')
        else:  # smooth_l1
            self.base_loss = nn.SmoothL1Loss(reduction='none')

        # Loss weighting strategy
        self.weighting = training_cfg.loss_weighting

        # Competition weights for base targets (sum = 0.3)
        # We weight the 3 base targets based on their contribution to final score
        # Dry_Green contributes to Dry_Total (0.5) and GDM (0.2) -> importance = 0.7
        # Dry_Dead contributes to Dry_Total (0.5) -> importance = 0.5
        # Dry_Clover contributes to Dry_Total (0.5) and GDM (0.2) -> importance = 0.7
        # Normalize so they're roughly balanced but reflect importance
        self.base_weights = torch.tensor([0.35, 0.30, 0.35], dtype=torch.float32)

        if self.weighting == "uncertainty":
            # Learnable log variances for uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.

        Args:
            predictions: (batch, 3) base target predictions
            targets: (batch, 3) base target ground truth

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual losses
        """
        # Compute per-task losses
        losses = self.base_loss(predictions, targets)  # (batch, 3)
        task_losses = losses.mean(dim=0)  # (3,)

        # Apply weighting
        if self.weighting == "equal":
            weights = torch.ones(3, device=predictions.device) / 3
        elif self.weighting == "competition":
            weights = self.base_weights.to(predictions.device)
        elif self.weighting == "uncertainty":
            # Uncertainty weighting: L_total = sum_i (L_i / (2 * sigma_i^2) + log(sigma_i))
            precisions = torch.exp(-self.log_vars)
            weighted_losses = task_losses * precisions + self.log_vars
            total_loss = weighted_losses.sum()

            return total_loss, {
                'loss_green': task_losses[0],
                'loss_dead': task_losses[1],
                'loss_clover': task_losses[2],
                'log_var_green': self.log_vars[0],
                'log_var_dead': self.log_vars[1],
                'log_var_clover': self.log_vars[2]
            }
        else:
            weights = torch.ones(3, device=predictions.device) / 3

        # Weighted sum
        total_loss = (task_losses * weights).sum()

        return total_loss, {
            'loss_green': task_losses[0],
            'loss_dead': task_losses[1],
            'loss_clover': task_losses[2]
        }


def create_model(config: Config) -> BiomassModel:
    """Factory function to create model."""
    return BiomassModel(config)


def load_model(config: Config, checkpoint_path: str) -> BiomassModel:
    """Load model from checkpoint."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    # Test model
    from config import get_config

    config = get_config("convnext_base")

    # Create model
    model = create_model(config)
    print(f"Model created: {config.model.backbone}")
    print(f"Backbone dim: {config.model.backbone_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 384, 384)
    metadata = torch.tensor([
        [0.5, 0.3, 0, 0],
        [0.6, -0.2, 1, 1],
        [0.4, 0.1, 2, 2],
        [0.7, 0.5, 3, 3]
    ], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(images, metadata)
        print(f"\nBase predictions shape: {output['base_preds'].shape}")

        all_preds = model.predict_all_targets(images, metadata)
        print(f"All predictions shape: {all_preds.shape}")
        print(f"Sample predictions: {all_preds[0].numpy()}")

    # Test loss
    loss_fn = MultiTaskLoss(config)
    targets = torch.randn(batch_size, 3)
    loss, loss_dict = loss_fn(output['base_preds'], targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
