"""
Dataset and DataLoader for CSIRO Pasture Biomass Prediction.

Features:
- Efficient image loading with caching
- Metadata integration (NDVI, Height, State, Species)
- Target scaling (divide by target_scale, NOT log1p)
- Comprehensive augmentation pipeline
- Stratified K-Fold splitting

NOTE: Based on research, log1p transform should only be applied during
R² evaluation, not during training. The working solution (0.72 R²) uses
simple scaling instead. Reference: CSIRO paper states log transform is
for "computing R² values", not for training targets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import Config, DataConfig, AugmentationConfig


class BiomassDataset(Dataset):
    """
    Dataset for pasture biomass prediction.

    Each sample contains:
    - Image tensor (C, H, W)
    - Metadata tensor (numerical + categorical embeddings)
    - Target tensor (3 base targets or 5 all targets)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        config: Config,
        mode: str = "train",
        transform: Optional[A.Compose] = None,
        species_list: Optional[List[str]] = None
    ):
        """
        Args:
            df: DataFrame with image paths and targets (wide format, one row per image)
            data_dir: Root directory containing image folders
            config: Configuration object
            mode: 'train', 'val', or 'test'
            transform: Albumentations transform pipeline
            species_list: Complete list of all species (shared across folds)
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        self.transform = transform
        self.species_list = species_list

        # Extract unique image paths
        self.image_paths = self.df['image_path'].tolist()

        # Prepare encoders for categorical features
        self._prepare_encoders()

        # Precompute targets if in train/val mode
        if mode != "test":
            self._prepare_targets()

    def _prepare_encoders(self):
        """Prepare label encoders for categorical features."""
        self.state_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()

        # Fit on all known categories
        self.state_encoder.fit(self.config.data.states)

        # Use shared species list if provided, otherwise use current data
        if self.species_list is not None:
            self.species_encoder.fit(self.species_list)
        elif 'Species' in self.df.columns:
            all_species = self.df['Species'].unique().tolist()
            self.species_encoder.fit(all_species)

    def _prepare_targets(self):
        """Prepare target arrays."""
        # Ensure targets are in correct order
        self.targets = self.df[self.config.data.base_targets].values.astype(np.float32)

        # Also store all targets for evaluation
        if all(t in self.df.columns for t in self.config.data.all_targets):
            self.all_targets = self.df[self.config.data.all_targets].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        img_path = self.data_dir / row['image_path']
        image = np.array(Image.open(img_path).convert('RGB'))

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare metadata
        metadata = self._prepare_metadata(row)

        # Prepare output dict
        output = {
            'image': image,
            'metadata': metadata,
            'image_id': row.get('image_id', Path(row['image_path']).stem)
        }

        # Add targets if not test mode
        if self.mode != "test":
            targets = self.targets[idx]

            # Apply target transformation
            if self.config.training.use_log_transform:
                # Legacy: log1p transform (NOT RECOMMENDED)
                targets = np.log1p(targets)
            else:
                # Recommended: Simple scaling (divide by target_scale)
                # This normalizes targets to ~0-1 range for stable training
                # Based on working solution achieving 0.72 R²
                target_scale = getattr(self.config.training, 'target_scale', 100.0)
                targets = targets / target_scale

            output['targets'] = torch.tensor(targets, dtype=torch.float32)

            # Also include raw targets for evaluation (always in original scale)
            if hasattr(self, 'all_targets'):
                output['raw_targets'] = torch.tensor(
                    self.all_targets[idx], dtype=torch.float32
                )

        return output

    def _prepare_metadata(self, row: pd.Series) -> torch.Tensor:
        """Prepare metadata tensor from row."""
        # Numerical features (normalized)
        ndvi = (row.get('Pre_GSHH_NDVI', 0.55) - self.config.data.ndvi_mean) / self.config.data.ndvi_std
        height = (row.get('Height_Ave_cm', 10.0) - self.config.data.height_mean) / self.config.data.height_std

        # Handle missing values
        ndvi = 0.0 if pd.isna(ndvi) else ndvi
        height = 0.0 if pd.isna(height) else height

        # Categorical features (encoded as integers)
        state = row.get('State', 'Tas')
        species = row.get('Species', 'Ryegrass')

        try:
            state_idx = self.state_encoder.transform([state])[0]
            # Clamp to valid range
            state_idx = min(state_idx, len(self.config.data.states) - 1)
        except ValueError:
            state_idx = 0  # Default for unseen

        try:
            species_idx = self.species_encoder.transform([species])[0]
            # Clamp to valid range (n_species is the embedding size)
            max_species_idx = self.config.data.n_species - 1
            species_idx = min(species_idx, max_species_idx)
        except ValueError:
            species_idx = 0  # Default for unseen

        # Combine: [ndvi, height, state_idx, species_idx]
        metadata = torch.tensor([ndvi, height, state_idx, species_idx], dtype=torch.float32)

        return metadata


def get_train_transforms(config: Config) -> A.Compose:
    """Get training augmentation pipeline."""
    aug_cfg = config.augmentation
    data_cfg = config.data

    transforms = [
        # Resize to slightly larger for random crop
        A.Resize(height=data_cfg.img_size[0], width=data_cfg.img_size[1]),

        # Geometric transforms
        A.HorizontalFlip(p=aug_cfg.horizontal_flip_p),
        A.VerticalFlip(p=aug_cfg.vertical_flip_p),
        A.Rotate(limit=aug_cfg.rotate_limit, p=aug_cfg.rotate_p, border_mode=0),

        # Random crop (with resize back)
        A.RandomResizedCrop(
            size=(data_cfg.crop_size[0], data_cfg.crop_size[1]),
            scale=aug_cfg.random_resized_crop_scale,
            ratio=(0.9, 1.1),
            p=aug_cfg.random_crop_p
        ),

        # Resize back to target size
        A.Resize(height=data_cfg.img_size[0], width=data_cfg.img_size[1]),

        # Color transforms (conservative for vegetation)
        A.ColorJitter(
            brightness=aug_cfg.brightness_limit,
            contrast=aug_cfg.contrast_limit,
            saturation=aug_cfg.saturation_limit,
            hue=aug_cfg.hue_limit,
            p=0.5
        ),

        # Regularization
        A.CoarseDropout(
            max_holes=aug_cfg.cutout_max_holes,
            max_height=aug_cfg.cutout_max_size,
            max_width=aug_cfg.cutout_max_size,
            fill_value=0,
            p=aug_cfg.cutout_p
        ),

        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]

    return A.Compose(transforms)


def get_val_transforms(config: Config) -> A.Compose:
    """Get validation/test augmentation pipeline (no augmentation)."""
    data_cfg = config.data

    return A.Compose([
        A.Resize(height=data_cfg.img_size[0], width=data_cfg.img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(config: Config, tta_type: str) -> A.Compose:
    """Get test-time augmentation transforms."""
    data_cfg = config.data

    base_transforms = [
        A.Resize(height=data_cfg.img_size[0], width=data_cfg.img_size[1]),
    ]

    if tta_type == "hflip":
        base_transforms.append(A.HorizontalFlip(p=1.0))
    elif tta_type == "vflip":
        base_transforms.append(A.VerticalFlip(p=1.0))
    elif tta_type == "hflip_vflip":
        base_transforms.extend([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ])

    base_transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return A.Compose(base_transforms)


def prepare_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare train/test dataframes.

    Returns:
        train_df: Wide format DataFrame (one row per image)
        test_df: Wide format DataFrame for test set
    """
    data_cfg = config.data

    # Load train data
    train_df = pd.read_csv(data_cfg.data_dir / data_cfg.train_csv)

    # Pivot to wide format (one row per image)
    train_wide = train_df.pivot_table(
        index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Add image_id column
    train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

    # Load test data
    test_df = pd.read_csv(data_cfg.data_dir / data_cfg.test_csv)

    # Test data doesn't have targets, just get unique images
    test_wide = test_df.drop_duplicates('image_path')[['image_path', 'target_name']].copy()
    test_wide = test_df.drop_duplicates('image_path')[['image_path']].copy()
    test_wide['image_id'] = test_wide['image_path'].apply(lambda x: Path(x).stem)

    return train_wide, test_wide


def create_folds(
    df: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    """
    Create stratified K-Fold splits.

    Args:
        df: Wide format DataFrame
        config: Configuration object

    Returns:
        DataFrame with 'fold' column added
    """
    n_folds = config.training.n_folds
    stratify_col = config.training.stratify_by

    df = df.copy()
    df['fold'] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.training.seed)

    for fold, (_, val_idx) in enumerate(skf.split(df, df[stratify_col])):
        df.loc[val_idx, 'fold'] = fold

    return df


def get_dataloaders(
    config: Config,
    fold: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration object
        fold: Which fold to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Prepare data
    train_df, test_df = prepare_data(config)
    train_df = create_folds(train_df, config)

    # Extract COMPLETE species list from ALL training data (before fold split)
    # This ensures all folds use the same species encoding
    all_species = sorted(train_df['Species'].unique().tolist())
    config.data.n_species = len(all_species)
    print(f"Total unique species: {len(all_species)}")

    # Split by fold
    train_fold_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_fold_df = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Fold {fold}: Train={len(train_fold_df)}, Val={len(val_fold_df)}, Test={len(test_df)}")

    # Create datasets with shared species list
    train_dataset = BiomassDataset(
        df=train_fold_df,
        data_dir=config.data.data_dir,
        config=config,
        mode="train",
        transform=get_train_transforms(config),
        species_list=all_species
    )

    val_dataset = BiomassDataset(
        df=val_fold_df,
        data_dir=config.data.data_dir,
        config=config,
        mode="val",
        transform=get_val_transforms(config),
        species_list=all_species
    )

    test_dataset = BiomassDataset(
        df=test_df,
        data_dir=config.data.data_dir,
        config=config,
        mode="test",
        transform=get_val_transforms(config),
        species_list=all_species
    )

    # GPU-optimized dataloader settings
    gpu_cfg = config.gpu
    num_workers = config.training.num_workers

    # Common kwargs for all dataloaders
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': config.training.pin_memory,
        'prefetch_factor': gpu_cfg.prefetch_factor if num_workers > 0 else None,
        'persistent_workers': gpu_cfg.persistent_workers if num_workers > 0 else False,
    }

    # Create dataloaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,  # Can use larger batch for val
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader


class MixupCutmix:
    """
    Mixup and Cutmix augmentation for regression.
    Applied at batch level during training.

    Reference: https://arxiv.org/abs/1710.09412 (Mixup)
    """

    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 0.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup/cutmix to batch.

        Returns:
            mixed_images, targets_a, targets_b, lam
        """
        batch_size = images.size(0)

        # Decide which augmentation to use
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = np.random.random() < 0.5
        else:
            use_cutmix = self.cutmix_alpha > 0

        if use_cutmix and self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        elif self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            return images, targets, targets, 1.0

        # Random permutation
        indices = torch.randperm(batch_size)

        if use_cutmix:
            # Cutmix: cut and paste regions
            _, _, H, W = images.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

            # Adjust lambda based on actual cut area
            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        else:
            # Mixup: linear interpolation
            mixed_images = lam * images + (1 - lam) * images[indices]

        return mixed_images, targets, targets[indices], lam


if __name__ == "__main__":
    # Test dataset
    from config import get_config

    config = get_config()
    train_loader, val_loader, test_loader = get_dataloaders(config, fold=0)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Metadata shape: {batch['metadata'].shape}")
    print(f"  Targets shape: {batch['targets'].shape}")
