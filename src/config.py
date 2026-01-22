"""
Configuration for CSIRO Pasture Biomass Prediction Model.

Architecture: Multi-Task Learning with Hard Parameter Sharing
- Shared CNN/Transformer backbone (pretrained)
- Task-specific regression heads for 3 base targets
- Derived targets via compositional relationships

References:
- Multi-Task Learning: https://hav4ik.github.io/articles/mtl-a-practical-survey
- EfficientNetV2: https://arxiv.org/abs/2104.00298
- ConvNeXt: https://arxiv.org/abs/2201.03545
- Swin Transformer V2: https://arxiv.org/abs/2111.09883
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: Path = Path(r"/workspace/data/csiro-biomass/csiro-biomass/")
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"

    # Image settings
    img_size: Tuple[int, int] = (384, 384)  # (H, W) - efficient for pretrained models
    crop_size: Tuple[int, int] = (352, 352)  # For random crop augmentation

    # Target columns (order matters for indexing)
    base_targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"
    ])
    derived_targets: List[str] = field(default_factory=lambda: [
        "GDM_g", "Dry_Total_g"
    ])
    all_targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"
    ])

    # Competition weights for weighted R²
    target_weights: Dict[str, float] = field(default_factory=lambda: {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5
    })

    # Metadata features
    categorical_features: List[str] = field(default_factory=lambda: ["State", "Species"])
    numerical_features: List[str] = field(default_factory=lambda: ["Pre_GSHH_NDVI", "Height_Ave_cm"])

    # States and Species (from EDA)
    states: List[str] = field(default_factory=lambda: ["Tas", "NSW", "WA", "Vic"])
    n_species: int = 15  # Will be computed from data

    # Normalization stats (from EDA - can be recomputed)
    ndvi_mean: float = 0.55
    ndvi_std: float = 0.15
    height_mean: float = 10.0
    height_std: float = 12.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone selection
    backbone: str = "convnext_large"  # Options: convnext_base, efficientnetv2_m, swin_base_patch4_window12_384
    pretrained: bool = True

    # Feature dimensions (will be set based on backbone)
    backbone_dim: int = 1024  # ConvNeXt-Base output dim

    # Metadata embedding
    state_embed_dim: int = 16
    species_embed_dim: int = 32

    # Fusion MLP
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3

    # Task-specific heads
    head_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    head_dropout: float = 0.2

    # Number of outputs (3 base targets)
    n_base_outputs: int = 3

    # Whether to use auxiliary heads for derived targets
    use_auxiliary_heads: bool = False  # We compute derived targets from base


@dataclass
class GPUConfig:
    """GPU optimization configuration for maximum utilization."""
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_ids: List[int] = field(default_factory=lambda: [0])  # GPU IDs for DataParallel

    # CUDA optimizations
    cudnn_benchmark: bool = True  # Enable cuDNN auto-tuner for consistent input sizes
    cudnn_deterministic: bool = False  # Set True for reproducibility (slower)

    # Memory optimizations
    empty_cache_freq: int = 10  # Clear cache every N batches (0 to disable)
    gradient_checkpointing: bool = False  # Trade compute for memory

    # Data loading optimizations (only used when num_workers > 0)
    prefetch_factor: int = 2  # Number of batches to prefetch per worker
    persistent_workers: bool = True  # Disabled for Windows compatibility

    # TF32 precision (Ampere+ GPUs)
    allow_tf32: bool = True  # Allow TF32 on Ampere GPUs for faster matmul

    # Memory format
    channels_last: bool = True  # Use NHWC format for faster CNN ops


@dataclass
class GradientBalancingConfig:
    """Gradient balancing configuration for multi-task learning."""
    # Method: equal, competition, uncertainty, mgda, gradnorm, pcgrad, cagrad, dwa
    method: str = "mgda"

    # MGDA settings
    mgda_normalize: bool = True  # Normalize gradients for solver (numerical stability)
    mgda_rescale: bool = True  # Rescale weighted gradient to match original magnitude (important!)
    mgda_use_rep_grad: bool = True  # Use representation gradients (more efficient)

    # GradNorm settings
    gradnorm_alpha: float = 1.5  # Asymmetry parameter (higher = focus on lagging tasks)
    gradnorm_weight_lr: float = 0.025  # Learning rate for weight updates

    # PCGrad settings
    pcgrad_reduction: str = "mean"  # 'mean' or 'sum'

    # CAGrad settings
    cagrad_c: float = 0.5  # Trade-off parameter (0=average, 1=conflict-averse)
    cagrad_rescale: bool = True

    # DWA settings
    dwa_temperature: float = 2.0  # Higher = more uniform weights


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""
    enabled: bool = True
    experiment_name: str = "csiro_biomass"
    tracking_uri: str = "file:./mlruns"  # Local tracking, or use remote URI
    run_name: Optional[str] = None  # Auto-generated if None
    log_system_metrics: bool = True
    log_gpu_metrics: bool = True
    log_model: bool = True  # Log final model to MLflow
    log_checkpoints: bool = False  # Log all checkpoints (can be large)
    artifact_location: Optional[str] = None

    # Tags to add to all runs
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Windows has issues with too many workers due to shared memory limitations
    # Use 0 for Windows (loads data in main process) or 2-4 for Linux
    num_workers: int = 4  # Set to 0 for Windows compatibility
    pin_memory: bool = True

    # Training schedule
    epochs: int = 30
    batch_size: int = 32  # Increased for better GPU utilization
    accumulation_steps: int = 1  # Reduced since batch size increased

    # Optimizer (AdamW)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    backbone_lr_mult: float = 0.1  # Lower LR for pretrained backbone

    # Scheduler (Cosine with warmup)
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Loss function
    loss_type: str = "huber"  # Options: mse, huber, smooth_l1
    huber_delta: float = 1.0

    # Target transformation
    use_log_transform: bool = True  # log1p(target) for training

    # Loss weighting strategy (legacy - use gradient_balancing.method instead)
    loss_weighting: str = "competition"  # Options: equal, competition, uncertainty

    # Cross-validation
    n_folds: int = 5
    val_fold: int = 0  # Which fold to use for validation (0 to n_folds-1)
    stratify_by: str = "State"  # Stratify splits by State

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    save_dir: Path = Path(r"/workspace/data/csiro-biomass/csiro-biomass/checkpoints")
    save_best_only: bool = True

    # Mixed precision
    use_amp: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Geometric transforms
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate_p: float = 0.3
    rotate_limit: int = 45

    # Color transforms (conservative for vegetation)
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    saturation_limit: float = 0.1
    hue_limit: float = 0.05  # Very small - hue is important for green/dead

    # Spatial transforms
    random_crop_p: float = 0.5
    random_resized_crop_scale: Tuple[float, float] = (0.8, 1.0)

    # Regularization
    cutout_p: float = 0.3
    cutout_max_holes: int = 4
    cutout_max_size: int = 40

    # Advanced
    mixup_alpha: float = 0.2  # 0 to disable
    cutmix_alpha: float = 0.0  # 0 to disable (less suitable for regression)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Test-time augmentation
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: [
        "original", "hflip", "vflip", "hflip_vflip"
    ])

    # Ensemble
    ensemble_weights: Optional[List[float]] = None  # None = equal weights

    # Output
    output_dir: Path = Path(r"/workspace/data/csiro-biomass/csiro-biomass/submissions")
    submission_name: str = "submission.csv"


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    gradient_balancing: GradientBalancingConfig = field(default_factory=GradientBalancingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    def __post_init__(self):
        """Create directories and validate config."""
        self.training.save_dir.mkdir(parents=True, exist_ok=True)
        self.inference.output_dir.mkdir(parents=True, exist_ok=True)

        # Set backbone dimension based on selection
        backbone_dims = {
            "convnext_base": 1024,
            "convnext_large": 1536,
            "convnext_tiny": 768,
            "efficientnetv2_m": 1280,
            "efficientnetv2_l": 1280,
            "efficientnetv2_s": 1280,
            "swin_base_patch4_window12_384": 1024,
            "swin_large_patch4_window12_384": 1536,
        }
        if self.model.backbone in backbone_dims:
            self.model.backbone_dim = backbone_dims[self.model.backbone]


# Pretrained model URLs and recommendations
PRETRAINED_MODELS = {
    "convnext_base": {
        "source": "timm",
        "weights": "convnext_base.fb_in22k_ft_in1k_384",
        "input_size": 384,
        "notes": "Best balance of speed and accuracy for dense prediction tasks"
    },
    "convnext_large": {
        "source": "timm",
        "weights": "convnext_large.fb_in22k_ft_in1k_384",
        "input_size": 384,
        "notes": "Higher capacity, slower inference"
    },
    "efficientnetv2_m": {
        "source": "timm",
        "weights": "tf_efficientnetv2_m.in21k_ft_in1k",
        "input_size": 384,
        "notes": "Efficient architecture, good for limited compute"
    },
    "swin_base_patch4_window12_384": {
        "source": "timm",
        "weights": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
        "input_size": 384,
        "notes": "Transformer-based, captures global context well"
    }
}


def get_config(backbone: str = "convnext_base") -> Config:
    """Get configuration with specified backbone."""
    cfg = Config()
    cfg.model.backbone = backbone
    cfg.__post_init__()
    return cfg
