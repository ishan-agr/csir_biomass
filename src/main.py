"""
Main Entry Point for CSIRO Pasture Biomass Prediction.

Usage:
    # Train single fold
    python main.py train --fold 0

    # Train all folds (cross-validation)
    python main.py train --cv

    # Generate submission
    python main.py predict --output submission.csv

    # Full pipeline (train + predict)
    python main.py full --cv --output submission.csv

Architecture & Approach:
-----------------------
This solution uses Multi-Task Learning (MTL) with Hard Parameter Sharing:

1. BACKBONE: Pretrained ConvNeXt/EfficientNetV2/Swin Transformer
   - Extracts rich visual features from pasture images
   - Pretrained on ImageNet-21K for better transfer learning

2. METADATA FUSION:
   - NDVI and Height: Normalized numerical features
   - State and Species: Learned embeddings
   - Late fusion with image features

3. MULTI-TASK HEADS:
   - 3 independent regression heads for base targets:
     * Dry_Green_g, Dry_Dead_g, Dry_Clover_g
   - Derived targets computed from base predictions:
     * GDM_g = Dry_Green_g + Dry_Clover_g
     * Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g

4. TRAINING STRATEGY:
   - Log1p transform for right-skewed targets
   - Competition-aware loss weighting
   - Cosine annealing with warmup
   - Mixed precision training
   - Gradient accumulation

5. INFERENCE:
   - Test-Time Augmentation (TTA)
   - Fold ensemble

Key References:
- Multi-Task Learning: https://hav4ik.github.io/articles/mtl-a-practical-survey
- ConvNeXt: https://arxiv.org/abs/2201.03545
- timm library: https://github.com/huggingface/pytorch-image-models
- Uncertainty Weighting: https://arxiv.org/abs/1705.07115
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, PRETRAINED_MODELS
from train import train_fold, train_cv
from inference import generate_submission, ensemble_from_folds


def main():
    parser = argparse.ArgumentParser(
        description="CSIRO Pasture Biomass Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train single fold with ConvNeXt backbone
    python main.py train --backbone convnext_base --fold 0

    # Train all folds with EfficientNetV2
    python main.py train --backbone efficientnetv2_m --cv

    # Generate submission from trained models
    python main.py predict --backbone convnext_base --output submission.csv

    # Full pipeline
    python main.py full --backbone convnext_base --cv --output submission.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model(s)')
    train_parser.add_argument('--backbone', type=str, default='convnext_base',
                              choices=list(PRETRAINED_MODELS.keys()),
                              help='Backbone architecture')
    train_parser.add_argument('--fold', type=int, default=None,
                              help='Train single fold (0-4)')
    train_parser.add_argument('--cv', action='store_true',
                              help='Train all folds (cross-validation)')
    train_parser.add_argument('--epochs', type=int, default=50,
                              help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=16,
                              help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                              help='Learning rate')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    train_parser.add_argument('--compile', action='store_true',
                              help='Use torch.compile for faster training (Linux only, requires Triton)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--backbone', type=str, default='convnext_base',
                                choices=list(PRETRAINED_MODELS.keys()))
    predict_parser.add_argument('--checkpoint', type=str, default=None,
                                help='Single checkpoint path')
    predict_parser.add_argument('--output', type=str, default='submission.csv',
                                help='Output filename')
    predict_parser.add_argument('--no_tta', action='store_true',
                                help='Disable test-time augmentation')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Train and predict')
    full_parser.add_argument('--backbone', type=str, default='convnext_base',
                             choices=list(PRETRAINED_MODELS.keys()))
    full_parser.add_argument('--cv', action='store_true',
                             help='Use cross-validation')
    full_parser.add_argument('--fold', type=int, default=0,
                             help='Fold to train (if not --cv)')
    full_parser.add_argument('--epochs', type=int, default=50)
    full_parser.add_argument('--batch_size', type=int, default=16)
    full_parser.add_argument('--lr', type=float, default=1e-4)
    full_parser.add_argument('--output', type=str, default='submission.csv')
    full_parser.add_argument('--seed', type=int, default=42)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show model info')
    info_parser.add_argument('--backbone', type=str, default='convnext_base')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'train':
        config = get_config(args.backbone)
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.training.seed = args.seed

        compile_flag = getattr(args, 'compile', False)

        if args.cv:
            result = train_cv(config, compile_model_flag=compile_flag)
            print(f"\nCV Results: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
        elif args.fold is not None:
            result = train_fold(config, args.fold, compile_model_flag=compile_flag)
            print(f"\nFold {args.fold} Best R²: {result['best_score']:.4f}")
        else:
            print("Please specify --fold or --cv")

    elif args.command == 'predict':
        config = get_config(args.backbone)

        if args.checkpoint:
            submission = generate_submission(
                config,
                checkpoint_paths=[Path(args.checkpoint)],
                output_name=args.output,
                use_tta=not args.no_tta
            )
        else:
            submission = ensemble_from_folds(
                config,
                n_folds=config.training.n_folds,
                output_name=args.output,
                use_tta=not args.no_tta
            )

        print(f"\nSubmission shape: {submission.shape}")

    elif args.command == 'full':
        config = get_config(args.backbone)
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.training.seed = args.seed

        # Train
        print("=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)

        if args.cv:
            result = train_cv(config)
            print(f"\nCV Results: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
        else:
            result = train_fold(config, args.fold)
            print(f"\nFold {args.fold} Best R²: {result['best_score']:.4f}")

        # Predict
        print("\n" + "=" * 60)
        print("INFERENCE PHASE")
        print("=" * 60)

        if args.cv:
            submission = ensemble_from_folds(
                config,
                n_folds=config.training.n_folds,
                output_name=args.output,
                use_tta=True
            )
        else:
            checkpoint_path = config.training.save_dir / f"best_fold{args.fold}.pt"
            submission = generate_submission(
                config,
                checkpoint_paths=[checkpoint_path],
                output_name=args.output,
                use_tta=True
            )

        print(f"\nSubmission saved: {args.output}")
        print(f"Shape: {submission.shape}")

    elif args.command == 'info':
        from model import create_model
        from utils import count_parameters

        config = get_config(args.backbone)
        model = create_model(config)

        print(f"\n{'='*50}")
        print(f"Model: {args.backbone}")
        print(f"{'='*50}")
        print(f"Backbone dimension: {config.model.backbone_dim}")
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
        print(f"\nPretrained weights: {PRETRAINED_MODELS[args.backbone]['weights']}")
        print(f"Notes: {PRETRAINED_MODELS[args.backbone]['notes']}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
