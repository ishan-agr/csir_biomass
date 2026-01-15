"""
MLflow Integration for Experiment Tracking.

Features:
- Automatic logging of training metrics, parameters, and artifacts
- Model versioning and registry
- GPU profiling and system metrics
- Experiment comparison and visualization
- Checkpoint management

Reference: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/
"""

import os
import sys
import time
import json
import platform
import psutil
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLflowTracker:
    """
    MLflow experiment tracker for training pipeline.

    Handles:
    - Experiment creation and run management
    - Parameter, metric, and artifact logging
    - Model checkpointing and registry
    - System profiling (GPU, CPU, memory)
    """

    def __init__(
        self,
        experiment_name: str = "csiro_biomass",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_system_metrics: bool = True,
        log_gpu_metrics: bool = True
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Custom artifact storage location
            run_name: Name for this run (default: auto-generated)
            tags: Additional tags for the run
            log_system_metrics: Whether to log CPU/memory metrics
            log_gpu_metrics: Whether to log GPU metrics
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.artifact_location = artifact_location
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}
        self.log_system_metrics = log_system_metrics
        self.log_gpu_metrics = log_gpu_metrics and TORCH_AVAILABLE and torch.cuda.is_available()

        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            self.experiment_id = self.experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        # Initialize client for advanced operations
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Run state
        self.run = None
        self.run_id = None
        self.step = 0

        # Profiling
        self._start_time = None
        self._gpu_baseline = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        description: Optional[str] = None
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            nested: Whether this is a nested run
            description: Run description

        Returns:
            Run ID
        """
        run_name = run_name or self.run_name

        # Add default tags
        tags = {
            **self.tags,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }

        if TORCH_AVAILABLE:
            tags["pytorch_version"] = torch.__version__
            if torch.cuda.is_available():
                tags["cuda_version"] = torch.version.cuda
                tags["gpu_name"] = torch.cuda.get_device_name(0)
                tags["gpu_count"] = str(torch.cuda.device_count())

        self.run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            nested=nested,
            tags=tags,
            description=description
        )
        self.run_id = self.run.info.run_id
        self._start_time = time.time()

        # Log initial GPU state
        if self.log_gpu_metrics:
            self._gpu_baseline = self._get_gpu_memory()

        print(f"MLflow run started: {run_name} (ID: {self.run_id})")
        print(f"Tracking URI: {self.tracking_uri}")

        return self.run_id

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self.run is not None:
            # Log final metrics
            if self._start_time:
                duration = time.time() - self._start_time
                mlflow.log_metric("total_duration_seconds", duration)

            mlflow.end_run(status=status)
            print(f"MLflow run ended: {self.run_id} ({status})")
            self.run = None
            self.run_id = None

    def log_config(self, config: Any, prefix: str = ""):
        """
        Log configuration as parameters.

        Args:
            config: Configuration object (dataclass or dict)
            prefix: Prefix for parameter names
        """
        if hasattr(config, '__dataclass_fields__'):
            # Dataclass
            params = self._flatten_dict(asdict(config), prefix)
        elif isinstance(config, dict):
            params = self._flatten_dict(config, prefix)
        else:
            params = {prefix: str(config)}

        # MLflow has 500 param limit, truncate long values
        truncated_params = {}
        for k, v in params.items():
            str_v = str(v)
            if len(str_v) > 250:
                str_v = str_v[:247] + "..."
            truncated_params[k] = str_v

        mlflow.log_params(truncated_params)

    def _flatten_dict(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key))
            elif isinstance(v, (list, tuple)):
                items[key] = str(v)
            elif isinstance(v, Path):
                items[key] = str(v)
            else:
                items[key] = v
        return items

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (default: auto-increment)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step

        prefixed_metrics = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            # Handle numpy/torch types
            if hasattr(v, 'item'):
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = float(v)
            prefixed_metrics[key] = v

        mlflow.log_metrics(prefixed_metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if step is None:
            step = self.step
        if hasattr(value, 'item'):
            value = value.item()
        mlflow.log_metric(key, value, step=step)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: Optional[float] = None
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            lr: Current learning rate
        """
        self.step = epoch

        # Log train metrics
        self.log_metrics(train_metrics, step=epoch, prefix="train")

        # Log val metrics
        self.log_metrics(val_metrics, step=epoch, prefix="val")

        # Log learning rate
        if lr is not None:
            mlflow.log_metric("learning_rate", lr, step=epoch)

        # Log system metrics
        if self.log_system_metrics:
            self._log_system_metrics(epoch)

        if self.log_gpu_metrics:
            self._log_gpu_metrics(epoch)

    def _log_system_metrics(self, step: int):
        """Log CPU and memory metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            mlflow.log_metrics({
                "system/cpu_percent": cpu_percent,
                "system/memory_used_gb": memory.used / (1024 ** 3),
                "system/memory_percent": memory.percent
            }, step=step)
        except Exception as e:
            pass  # Silently ignore system metric errors

    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.log_gpu_metrics:
            return {}

        try:
            return {
                'allocated': torch.cuda.memory_allocated(0) / (1024 ** 3),
                'reserved': torch.cuda.memory_reserved(0) / (1024 ** 3),
                'max_allocated': torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            }
        except Exception:
            return {}

    def _log_gpu_metrics(self, step: int):
        """Log GPU metrics."""
        if not self.log_gpu_metrics:
            return

        try:
            gpu_mem = self._get_gpu_memory()

            mlflow.log_metrics({
                "gpu/memory_allocated_gb": gpu_mem.get('allocated', 0),
                "gpu/memory_reserved_gb": gpu_mem.get('reserved', 0),
                "gpu/memory_max_allocated_gb": gpu_mem.get('max_allocated', 0),
                "gpu/utilization_percent": gpu_mem.get('allocated', 0) /
                    (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) * 100
            }, step=step)
        except Exception:
            pass

    def log_model(
        self,
        model: "torch.nn.Module",
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Path within artifacts to store model
            registered_model_name: Name to register in model registry
            signature: Model signature
            input_example: Example input for signature inference
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )

    def log_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        artifact_path: str = "checkpoints"
    ):
        """Log a checkpoint file as artifact."""
        mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_path)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """Log a file or directory as artifact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_text(self, text: str, artifact_file: str):
        """Log text as artifact."""
        mlflow.log_text(text, artifact_file)

    def log_gradients_info(self, gradient_info: Dict[str, float], step: int):
        """
        Log gradient balancing information.

        Args:
            gradient_info: Info from gradient balancer
            step: Training step
        """
        prefixed_info = {f"gradients/{k}": v for k, v in gradient_info.items()}
        self.log_metrics(prefixed_info, step=step)

    def set_tag(self, key: str, value: str):
        """Set a tag for the current run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags."""
        mlflow.set_tags(tags)

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FINISHED" if exc_type is None else "FAILED"
        self.end_run(status=status)
        return False


class ExperimentManager:
    """
    Manager for MLflow experiments.

    Provides utilities for:
    - Comparing runs
    - Finding best runs
    - Loading models from registry
    - Experiment analysis
    """

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """
        Initialize experiment manager.

        Args:
            tracking_uri: MLflow tracking URI
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required")

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def get_experiment_runs(
        self,
        experiment_name: str,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[Any]:
        """
        Get runs from an experiment.

        Args:
            experiment_name: Experiment name
            filter_string: MLflow filter string
            order_by: Ordering columns
            max_results: Maximum runs to return

        Returns:
            List of Run objects
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["metrics.val/weighted_r2 DESC"],
            max_results=max_results
        )

        return runs

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "val/weighted_r2",
        mode: str = "max"
    ) -> Optional[Any]:
        """
        Get the best run from an experiment.

        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Best Run object
        """
        order = f"metrics.{metric} DESC" if mode == "max" else f"metrics.{metric} ASC"
        runs = self.get_experiment_runs(
            experiment_name,
            order_by=[order],
            max_results=1
        )

        return runs[0] if runs else None

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs
            metrics: Metrics to compare

        Returns:
            Dictionary of run_id -> metric -> value
        """
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'run_name': run.info.run_name,
                **{m: run.data.metrics.get(m) for m in metrics}
            }

        return comparison

    def load_model(self, run_id: str, artifact_path: str = "model") -> "torch.nn.Module":
        """
        Load a model from a run.

        Args:
            run_id: Run ID
            artifact_path: Path to model artifact

        Returns:
            Loaded PyTorch model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.pytorch.load_model(model_uri)

    def get_run_artifacts(self, run_id: str) -> List[str]:
        """Get list of artifacts for a run."""
        artifacts = self.client.list_artifacts(run_id)
        return [a.path for a in artifacts]

    def delete_run(self, run_id: str):
        """Delete a run."""
        self.client.delete_run(run_id)

    def get_run_params(self, run_id: str) -> Dict[str, str]:
        """Get parameters for a run."""
        run = self.client.get_run(run_id)
        return run.data.params

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get final metrics for a run."""
        run = self.client.get_run(run_id)
        return run.data.metrics

    def export_run_summary(
        self,
        experiment_name: str,
        output_path: Union[str, Path],
        max_runs: int = 100
    ):
        """
        Export run summary to JSON.

        Args:
            experiment_name: Experiment name
            output_path: Output file path
            max_runs: Maximum runs to export
        """
        runs = self.get_experiment_runs(experiment_name, max_results=max_runs)

        summary = []
        for run in runs:
            summary.append({
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'params': dict(run.data.params),
                'metrics': dict(run.data.metrics),
                'tags': dict(run.data.tags)
            })

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Exported {len(summary)} runs to {output_path}")


def create_tracker(
    experiment_name: str = "csiro_biomass",
    tracking_uri: Optional[str] = None,
    **kwargs
) -> MLflowTracker:
    """
    Factory function to create MLflow tracker.

    Args:
        experiment_name: Experiment name
        tracking_uri: Tracking URI
        **kwargs: Additional tracker arguments

    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        **kwargs
    )


if __name__ == "__main__":
    # Test MLflow tracker
    print("Testing MLflow Tracker")
    print("=" * 50)

    if not MLFLOW_AVAILABLE:
        print("MLflow not available, skipping tests")
        sys.exit(0)

    # Create tracker
    tracker = create_tracker(
        experiment_name="test_experiment",
        run_name="test_run"
    )

    # Start run
    tracker.start_run()

    # Log config
    test_config = {
        "model": {
            "backbone": "convnext_base",
            "learning_rate": 1e-4
        },
        "training": {
            "epochs": 50,
            "batch_size": 16
        }
    }
    tracker.log_config(test_config)

    # Log metrics
    for epoch in range(5):
        tracker.log_epoch_metrics(
            epoch=epoch,
            train_metrics={"loss": 1.0 - epoch * 0.1, "r2": 0.5 + epoch * 0.05},
            val_metrics={"loss": 1.1 - epoch * 0.1, "r2": 0.45 + epoch * 0.05},
            lr=1e-4 * (0.9 ** epoch)
        )

    # Log gradient info
    tracker.log_gradients_info({
        "weight_task0": 0.35,
        "weight_task1": 0.30,
        "weight_task2": 0.35,
        "min_norm": 0.5
    }, step=5)

    # End run
    tracker.end_run()

    print("\nMLflow tracking test completed!")
    print(f"View results at: {tracker.tracking_uri}")
    print("Run: mlflow ui --port 5000")
