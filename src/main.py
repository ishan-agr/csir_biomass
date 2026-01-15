"""
Main Entry Point for CSIRO Pasture Biomass Prediction.

Usage:
    # Train single fold with basic loss weighting
    python main.py train --fold 0

    # Train with advanced gradient balancing (MGDA)
    python main.py train --fold 0 --gradient_method mgda

    # Train all folds (cross-validation) with GradNorm
    python main.py train --cv --gradient_method gradnorm

    # Generate submission
    python main.py predict --output submission.csv

    # Full pipeline (train + predict)
    python main.py full --cv --output submission.csv

    # View MLflow experiments
    mlflow ui --port 5000

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

4. GRADIENT BALANCING METHODS:
   - MGDA: Multi-Objective Gradient Descent (Pareto optimization)
   - GradNorm: Adaptive gradient normalization
   - PCGrad: Projecting Conflicting Gradients
   - CAGrad: Conflict-Averse Gradient Descent
   - DWA: Dynamic Weight Averaging

5. TRAINING STRATEGY:
   - Log1p transform for right-skewed targets
   - Cosine annealing with warmup
   - Mixed precision training
   - MLflow experiment tracking

6. INFERENCE:
   - Test-Time Augmentation (TTA)
   - Fold ensemble

Key References:
- Multi-Task Learning: https://hav4ik.github.io/articles/mtl-a-practical-survey
- MGDA: https://arxiv.org/abs/1810.04650
- GradNorm: https://arxiv.org/abs/1711.02257
- PCGrad: https://arxiv.org/abs/2001.06782
- BioMassters 1st Place: https://github.com/drivendataorg/the-biomassters
- ConvNeXt: https://arxiv.org/abs/2201.03545
- timm library: https://github.com/huggingface/pytorch-image-models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, PRETRAINED_MODELS
from train import train_fold, train_cv
from inference import generate_submission, ensemble_from_folds

# Try to import advanced training (with gradient balancing and MLflow)
try:
    from train_advanced import train_fold_advanced, train_cv_advanced
    ADVANCED_TRAINING_AVAILABLE = True
except ImportError:
    ADVANCED_TRAINING_AVAILABLE = False

GRADIENT_METHODS = ['equal', 'competition', 'uncertainty', 'mgda', 'gradnorm', 'pcgrad', 'cagrad', 'dwa']


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
    # Advanced gradient balancing options
    train_parser.add_argument('--gradient_method', type=str, default='competition',
                              choices=GRADIENT_METHODS,
                              help='Gradient balancing method (mgda, gradnorm, pcgrad, cagrad, dwa)')
    train_parser.add_argument('--gradnorm_alpha', type=float, default=1.5,
                              help='GradNorm alpha parameter (asymmetry)')
    # MLflow options
    train_parser.add_argument('--no_mlflow', action='store_true',
                              help='Disable MLflow tracking')
    train_parser.add_argument('--experiment_name', type=str, default='csiro_biomass',
                              help='MLflow experiment name')

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

    # MLflow experiment commands
    mlflow_parser = subparsers.add_parser('mlflow', help='MLflow experiment management')
    mlflow_subparsers = mlflow_parser.add_subparsers(dest='mlflow_command')

    # mlflow compare
    compare_parser = mlflow_subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--experiment', type=str, default='csiro_biomass',
                                help='Experiment name')
    compare_parser.add_argument('--top_n', type=int, default=10,
                                help='Show top N runs')
    compare_parser.add_argument('--metric', type=str, default='val/weighted_r2',
                                help='Metric to sort by')

    # mlflow export
    export_parser = mlflow_subparsers.add_parser('export', help='Export experiment results')
    export_parser.add_argument('--experiment', type=str, default='csiro_biomass')
    export_parser.add_argument('--output', type=str, default='experiment_results.json')

    # mlflow ui
    ui_parser = mlflow_subparsers.add_parser('ui', help='Launch MLflow UI')
    ui_parser.add_argument('--port', type=int, default=5000)

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

        # Configure gradient balancing
        gradient_method = getattr(args, 'gradient_method', 'competition')
        config.gradient_balancing.method = gradient_method
        if hasattr(args, 'gradnorm_alpha'):
            config.gradient_balancing.gradnorm_alpha = args.gradnorm_alpha

        # Configure MLflow
        config.mlflow.enabled = not getattr(args, 'no_mlflow', False)
        if hasattr(args, 'experiment_name'):
            config.mlflow.experiment_name = args.experiment_name

        # Use advanced training for gradient balancing methods
        use_advanced = gradient_method in ['mgda', 'gradnorm', 'pcgrad', 'cagrad', 'dwa']

        if use_advanced and ADVANCED_TRAINING_AVAILABLE:
            print(f"Using advanced training with {gradient_method} gradient balancing")
            if args.cv:
                result = train_cv_advanced(config, compile_model_flag=compile_flag)
                print(f"\nCV Results: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
            elif args.fold is not None:
                result = train_fold_advanced(config, args.fold, compile_model_flag=compile_flag)
                print(f"\nFold {args.fold} Best R²: {result['best_score']:.4f}")
            else:
                print("Please specify --fold or --cv")
        elif use_advanced and not ADVANCED_TRAINING_AVAILABLE:
            print(f"Warning: Advanced training not available, using basic training")
            if args.cv:
                result = train_cv(config, compile_model_flag=compile_flag)
                print(f"\nCV Results: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
            elif args.fold is not None:
                result = train_fold(config, args.fold, compile_model_flag=compile_flag)
                print(f"\nFold {args.fold} Best R²: {result['best_score']:.4f}")
            else:
                print("Please specify --fold or --cv")
        else:
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
        print(f"\nGradient balancing methods available:")
        for method in GRADIENT_METHODS:
            print(f"  - {method}")
        print(f"{'='*50}")

    elif args.command == 'mlflow':
        try:
            from mlflow_tracking import ExperimentManager
            import subprocess
        except ImportError:
            print("MLflow not installed. Install with: pip install mlflow")
            return

        if args.mlflow_command == 'compare':
            manager = ExperimentManager()
            runs = manager.get_experiment_runs(
                args.experiment,
                order_by=[f"metrics.{args.metric} DESC"],
                max_results=args.top_n
            )

            if not runs:
                print(f"No runs found in experiment: {args.experiment}")
                return

            print(f"\n{'='*80}")
            print(f"Top {len(runs)} runs in '{args.experiment}' (sorted by {args.metric})")
            print(f"{'='*80}")
            print(f"{'Run Name':<40} {'R²':>10} {'Backbone':>20} {'Method':>15}")
            print("-" * 80)

            for run in runs:
                run_name = run.info.run_name or run.info.run_id[:8]
                r2 = run.data.metrics.get(args.metric, 0)
                backbone = run.data.tags.get('backbone', 'N/A')
                method = run.data.tags.get('gradient_method', 'N/A')
                print(f"{run_name:<40} {r2:>10.4f} {backbone:>20} {method:>15}")

            print(f"{'='*80}\n")

        elif args.mlflow_command == 'export':
            manager = ExperimentManager()
            manager.export_run_summary(args.experiment, args.output)
            print(f"Exported to {args.output}")

        elif args.mlflow_command == 'ui':
            print(f"Starting MLflow UI on port {args.port}...")
            print(f"Open http://localhost:{args.port} in your browser")
            subprocess.run(['mlflow', 'ui', '--port', str(args.port)])

        else:
            print("Please specify a mlflow subcommand: compare, export, or ui")


if __name__ == "__main__":
    main()
