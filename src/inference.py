"""
Inference and Submission Generation for CSIRO Pasture Biomass Prediction.

Features:
- Test-Time Augmentation (TTA)
- Model ensemble
- Submission file generation in competition format
"""

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config, get_config, InferenceConfig
from dataset import BiomassDataset, get_val_transforms, get_tta_transforms, prepare_data
from model import BiomassModel, load_model, create_model
from utils import TARGET_ORDER


class Inferencer:
    """
    Inference class for generating predictions and submissions.
    """

    def __init__(
        self,
        config: Config,
        checkpoint_paths: List[Path],
        device: str = None
    ):
        """
        Args:
            config: Configuration object
            checkpoint_paths: List of paths to model checkpoints (for ensemble)
            device: Device to use
        """
        self.config = config
        self.device = torch.device(device or config.training.device)
        self.checkpoint_paths = checkpoint_paths

        # Load models
        self.models = self._load_models()

    def _load_models(self) -> List[BiomassModel]:
        """Load all models from checkpoints."""
        models = []

        for path in self.checkpoint_paths:
            print(f"Loading model from {path}")
            model = create_model(self.config)
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            models.append(model)

        print(f"Loaded {len(models)} models for ensemble")
        return models

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        use_tta: bool = True
    ) -> np.ndarray:
        """
        Predict for a batch with optional TTA and ensemble.

        Args:
            images: (batch, 3, H, W) input images
            metadata: (batch, 4) metadata
            use_tta: Whether to use test-time augmentation

        Returns:
            (batch, 5) predictions for all targets
        """
        images = images.to(self.device)
        metadata = metadata.to(self.device)

        all_predictions = []

        for model in self.models:
            if use_tta:
                # Get TTA predictions
                tta_predictions = []

                for tta_type in self.config.inference.tta_transforms:
                    # Apply TTA transform
                    if tta_type == "original":
                        aug_images = images
                    elif tta_type == "hflip":
                        aug_images = torch.flip(images, dims=[3])
                    elif tta_type == "vflip":
                        aug_images = torch.flip(images, dims=[2])
                    elif tta_type == "hflip_vflip":
                        aug_images = torch.flip(images, dims=[2, 3])
                    else:
                        aug_images = images

                    with autocast(enabled=self.config.training.use_amp):
                        preds = model.predict_all_targets(
                            aug_images, metadata,
                            use_log_transform=self.config.training.use_log_transform
                        )
                    tta_predictions.append(preds.cpu().numpy())

                # Average TTA predictions
                model_pred = np.mean(tta_predictions, axis=0)
            else:
                # Single prediction
                with autocast(enabled=self.config.training.use_amp):
                    model_pred = model.predict_all_targets(
                        images, metadata,
                        use_log_transform=self.config.training.use_log_transform
                    ).cpu().numpy()

            all_predictions.append(model_pred)

        # Average ensemble predictions
        if self.config.inference.ensemble_weights is not None:
            weights = np.array(self.config.inference.ensemble_weights)
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(all_predictions, axis=0)

        return ensemble_pred

    def predict_dataset(
        self,
        dataloader: DataLoader,
        use_tta: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate predictions for entire dataset.

        Args:
            dataloader: DataLoader for test data
            use_tta: Whether to use TTA

        Returns:
            predictions: (n_samples, 5) array
            image_ids: List of image IDs
        """
        all_predictions = []
        all_image_ids = []

        for batch in tqdm(dataloader, desc="Generating predictions"):
            images = batch['image']
            metadata = batch['metadata']
            image_ids = batch['image_id']

            predictions = self.predict_batch(images, metadata, use_tta=use_tta)

            all_predictions.append(predictions)
            all_image_ids.extend(image_ids)

        predictions = np.concatenate(all_predictions, axis=0)

        return predictions, all_image_ids

    def create_submission(
        self,
        predictions: np.ndarray,
        image_ids: List[str],
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create submission DataFrame in competition format.

        Format:
            sample_id,target
            ID1001187975__Dry_Green_g,0.0
            ID1001187975__Dry_Dead_g,0.0
            ...

        Args:
            predictions: (n_samples, 5) predictions
            image_ids: List of image IDs
            output_path: Path to save submission CSV

        Returns:
            Submission DataFrame
        """
        rows = []

        for i, image_id in enumerate(image_ids):
            for j, target_name in enumerate(TARGET_ORDER):
                sample_id = f"{image_id}__{target_name}"
                target_value = predictions[i, j]

                # Ensure non-negative
                target_value = max(0.0, target_value)

                rows.append({
                    'sample_id': sample_id,
                    'target': target_value
                })

        submission_df = pd.DataFrame(rows)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission_df.to_csv(output_path, index=False)
            print(f"Submission saved to {output_path}")
            print(f"  Rows: {len(submission_df)}")
            print(f"  Images: {len(image_ids)}")

        return submission_df


def generate_submission(
    config: Config,
    checkpoint_paths: List[Path],
    output_name: str = "submission.csv",
    use_tta: bool = True
) -> pd.DataFrame:
    """
    Convenience function to generate a submission.

    Args:
        config: Configuration object
        checkpoint_paths: List of checkpoint paths for ensemble
        output_name: Name of output CSV file
        use_tta: Whether to use TTA

    Returns:
        Submission DataFrame
    """
    # Prepare test data
    _, test_df = prepare_data(config)

    # Create test dataset
    test_dataset = BiomassDataset(
        df=test_df,
        data_dir=config.data.data_dir,
        config=config,
        mode="test",
        transform=get_val_transforms(config)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    # Create inferencer
    inferencer = Inferencer(config, checkpoint_paths)

    # Generate predictions
    predictions, image_ids = inferencer.predict_dataset(test_loader, use_tta=use_tta)

    # Create submission
    output_path = config.inference.output_dir / output_name
    submission_df = inferencer.create_submission(predictions, image_ids, output_path)

    return submission_df


def ensemble_from_folds(
    config: Config,
    n_folds: int = 5,
    output_name: str = "submission_ensemble.csv",
    use_tta: bool = True
) -> pd.DataFrame:
    """
    Generate submission using ensemble of fold models.

    Args:
        config: Configuration object
        n_folds: Number of folds
        output_name: Name of output CSV
        use_tta: Whether to use TTA

    Returns:
        Submission DataFrame
    """
    # Collect checkpoint paths
    checkpoint_paths = []
    for fold in range(n_folds):
        path = config.training.save_dir / f"best_fold{fold}.pt"
        if path.exists():
            checkpoint_paths.append(path)
        else:
            print(f"Warning: Checkpoint not found: {path}")

    if not checkpoint_paths:
        raise ValueError("No checkpoints found!")

    print(f"Using {len(checkpoint_paths)} fold models for ensemble")

    return generate_submission(config, checkpoint_paths, output_name, use_tta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_base')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint path. If None, uses all fold checkpoints.')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--no_tta', action='store_true', help='Disable TTA')
    args = parser.parse_args()

    config = get_config(args.backbone)

    if args.checkpoint:
        # Single model
        submission = generate_submission(
            config,
            checkpoint_paths=[Path(args.checkpoint)],
            output_name=args.output,
            use_tta=not args.no_tta
        )
    else:
        # Fold ensemble
        submission = ensemble_from_folds(
            config,
            n_folds=config.training.n_folds,
            output_name=args.output,
            use_tta=not args.no_tta
        )

    print(f"\nSubmission preview:")
    print(submission.head(10))
