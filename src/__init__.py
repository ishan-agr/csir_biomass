"""
CSIRO Pasture Biomass Prediction - Multi-Task Learning Solution

Modules:
    config: Configuration dataclasses
    dataset: Data loading and augmentation
    model: Multi-task neural network architecture
    train: Training pipeline
    inference: Prediction and submission generation
    utils: Utilities and metrics
"""

# For direct script execution, imports are handled in individual files
# This __init__.py is for package usage

try:
    from .config import Config, get_config
    from .model import BiomassModel, create_model, MultiTaskLoss
    from .train import Trainer, train_fold, train_cv
    from .inference import Inferencer, generate_submission, ensemble_from_folds
    from .utils import weighted_r2_score, per_target_r2, set_seed
except ImportError:
    # Running as script, not as package
    pass

__version__ = "1.0.0"
__author__ = "CSIRO Biomass Team"

__all__ = [
    "Config",
    "get_config",
    "BiomassModel",
    "create_model",
    "MultiTaskLoss",
    "Trainer",
    "train_fold",
    "train_cv",
    "Inferencer",
    "generate_submission",
    "ensemble_from_folds",
    "weighted_r2_score",
    "per_target_r2",
    "set_seed",
]
