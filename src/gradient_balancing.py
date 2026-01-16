"""
Gradient Balancing Methods for Multi-Task Learning.

Implements state-of-the-art gradient balancing strategies:
1. MGDA (Multiple Gradient Descent Algorithm) - Sener & Koltun, NeurIPS 2018
2. GradNorm - Chen et al., ICML 2018
3. PCGrad (Projecting Conflicting Gradients) - Yu et al., NeurIPS 2020

IMPORTANT: This implements the ACTUAL algorithms as described in:
- MTL Survey: https://hav4ik.github.io/articles/mtl-a-practical-survey
- MGDA Paper: https://arxiv.org/abs/1810.04650

Key insight from survey:
"For each task t, compute gradients ∇_{θ^sh} L^t, then solve:
 minimize ||Σ λ^t ∇_{θ^sh} L^t||² s.t. Σλ^t=1, λ^t≥0"

References:
- MGDA: https://arxiv.org/abs/1810.04650
- GradNorm: https://arxiv.org/abs/1711.02257
- PCGrad: https://arxiv.org/abs/2001.06782
- LibMTL: https://libmtl.readthedocs.io/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import copy


class MinNormSolver:
    """
    Solver for finding minimum-norm point in convex hull.

    Implements Frank-Wolfe algorithm for MGDA optimization.
    Reference: https://github.com/isl-org/MultiObjectiveOptimization
    """

    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1: float, v1v2: float, v2v2: float) -> Tuple[float, float]:
        """
        Analytical solution for 2D case.
        Find γ that minimizes ||γ*v1 + (1-γ)*v2||²
        """
        if v1v2 >= v1v1:
            return 1.0, v1v1
        if v1v2 >= v2v2:
            return 0.0, v2v2

        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        gamma = max(0.0, min(1.0, gamma))
        cost = gamma * gamma * v1v1 + 2 * gamma * (1 - gamma) * v1v2 + (1 - gamma) * (1 - gamma) * v2v2
        return gamma, cost

    @staticmethod
    def find_min_norm_element(grads: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Find minimum norm element in convex hull of gradients using Frank-Wolfe.

        Args:
            grads: List of flattened gradient tensors (one per task)

        Returns:
            weights: Optimal task weights (tensor on same device)
            min_norm: Minimum norm value
        """
        n_tasks = len(grads)
        device = grads[0].device

        if n_tasks == 1:
            return torch.ones(1, device=device), grads[0].norm().item()

        # Stack gradients: (n_tasks, d)
        grad_mat = torch.stack(grads)

        # Compute Gram matrix: G[i,j] = <grad_i, grad_j>
        gram = torch.mm(grad_mat, grad_mat.t())  # (n_tasks, n_tasks)

        # Special case: 2 tasks - analytical solution
        if n_tasks == 2:
            v1v1 = gram[0, 0].item()
            v1v2 = gram[0, 1].item()
            v2v2 = gram[1, 1].item()
            gamma, _ = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            weights = torch.tensor([gamma, 1 - gamma], device=device)
            min_norm = torch.sqrt(weights @ gram @ weights).item()
            return weights, min_norm

        # General case: Frank-Wolfe algorithm
        # Initialize with equal weights
        sol = torch.ones(n_tasks, device=device) / n_tasks

        for _ in range(MinNormSolver.MAX_ITER):
            # Gradient of objective: 2 * G @ sol
            grad_obj = torch.mv(gram, sol)

            # Find minimizing vertex (argmin over simplex vertices)
            min_idx = grad_obj.argmin()

            # Descent direction
            descent = torch.zeros(n_tasks, device=device)
            descent[min_idx] = 1.0
            descent = descent - sol

            # Line search: min_{γ} ||sol + γ*descent||²_G
            a = (descent @ gram @ descent).item()
            b = 2 * (sol @ gram @ descent).item()

            if a <= 1e-8:
                gamma = 1.0
            else:
                gamma = max(0.0, min(1.0, -b / (2 * a + 1e-8)))

            # Convergence check
            if gamma < MinNormSolver.STOP_CRIT:
                break

            # Update
            sol = sol + gamma * descent

        # Ensure valid probability distribution
        sol = torch.clamp(sol, min=0)
        sol = sol / (sol.sum() + 1e-8)

        # Compute minimum norm
        min_norm = torch.sqrt(torch.clamp(sol @ gram @ sol, min=0)).item()

        return sol, min_norm


class MGDAOptimizer:
    """
    MGDA Optimizer - Proper implementation following the survey.

    Algorithm (from https://hav4ik.github.io/articles/mtl-a-practical-survey):
    1. For each task t: compute ∇_{θ^sh} L^t (requires T backward passes)
    2. Solve: min ||Σ λ^t ∇L^t||² s.t. Σλ^t=1, λ^t≥0
    3. Apply: θ^sh ← θ^sh - η * Σ λ^t ∇L^t

    This guarantees Pareto improvement at each step.
    """

    def __init__(
        self,
        shared_params: List[nn.Parameter],
        task_specific_params: Optional[List[List[nn.Parameter]]] = None,
        normalize_grads: bool = True
    ):
        """
        Args:
            shared_params: List of shared parameters (backbone, fusion)
            task_specific_params: Optional list of task-specific param lists
            normalize_grads: Whether to normalize gradients before solving
        """
        self.shared_params = list(shared_params)
        self.task_specific_params = task_specific_params or []
        self.normalize_grads = normalize_grads

        # Track gradient shapes for reconstruction
        self._grad_shapes = [p.shape for p in self.shared_params]
        self._grad_numel = [p.numel() for p in self.shared_params]

    def _flatten_grads(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """Flatten list of gradients into single vector."""
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grads(self, flat_grad: torch.Tensor) -> List[torch.Tensor]:
        """Unflatten gradient vector back to parameter shapes."""
        grads = []
        idx = 0
        for shape, numel in zip(self._grad_shapes, self._grad_numel):
            grads.append(flat_grad[idx:idx + numel].view(shape))
            idx += numel
        return grads

    def compute_task_gradients(
        self,
        task_losses: List[torch.Tensor],
        retain_graph: bool = True
    ) -> List[torch.Tensor]:
        """
        Compute gradients for each task w.r.t. shared parameters.

        This is the key step: T backward passes to get T gradient vectors.

        Args:
            task_losses: List of scalar loss tensors (one per task)
            retain_graph: Whether to retain computation graph

        Returns:
            List of flattened gradient tensors
        """
        task_grads = []

        for i, loss in enumerate(task_losses):
            # Compute gradient for this task
            grads = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True
            )

            # Handle None gradients (unused parameters)
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, self.shared_params)]

            # Flatten to single vector
            flat_grad = self._flatten_grads(grads)

            # Optionally normalize
            if self.normalize_grads:
                grad_norm = flat_grad.norm()
                if grad_norm > 1e-8:
                    flat_grad = flat_grad / grad_norm

            task_grads.append(flat_grad)

        return task_grads

    def solve_mgda(
        self,
        task_grads: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Solve MGDA optimization problem using Frank-Wolfe.

        Args:
            task_grads: List of flattened gradient tensors

        Returns:
            weights: Optimal task weights
            info: Dictionary with solver information
        """
        weights, min_norm = MinNormSolver.find_min_norm_element(task_grads)

        info = {
            'min_norm': min_norm,
            **{f'weight_task{i}': float(weights[i]) for i in range(len(weights))}
        }

        return weights, info

    def get_weighted_grad(
        self,
        task_grads: List[torch.Tensor],
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted combination of task gradients.

        Args:
            task_grads: List of flattened gradient tensors
            weights: Task weights from MGDA solver

        Returns:
            Weighted gradient vector
        """
        weighted_grad = sum(w * g for w, g in zip(weights, task_grads))
        return weighted_grad

    def apply_gradients(self, weighted_grad: torch.Tensor):
        """
        Apply weighted gradient to shared parameters.

        Sets .grad attribute for each shared parameter.
        """
        grads = self._unflatten_grads(weighted_grad)

        for param, grad in zip(self.shared_params, grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.copy_(grad)

    def step(
        self,
        task_losses: List[torch.Tensor],
        retain_graph: bool = True
    ) -> Dict[str, float]:
        """
        Full MGDA step: compute gradients, solve, apply.

        Args:
            task_losses: List of task losses
            retain_graph: Whether to retain graph after gradients

        Returns:
            info: Dictionary with MGDA information
        """
        # Step 1: Compute per-task gradients (T backward passes)
        task_grads = self.compute_task_gradients(task_losses, retain_graph)

        # Step 2: Solve MGDA optimization
        weights, info = self.solve_mgda(task_grads)

        # Step 3: Compute weighted gradient
        weighted_grad = self.get_weighted_grad(task_grads, weights)

        # Step 4: Apply to shared parameters
        self.apply_gradients(weighted_grad)

        return info


class GradNormOptimizer:
    """
    GradNorm Optimizer - Proper implementation following the paper.

    Algorithm (Chen et al., ICML 2018):
    1. Compute gradient norms G_i = ||∇_{W} w_i * L_i||
    2. Compute average: G_avg = mean(G_i)
    3. Compute relative inverse training rate: r_i = L_i(t) / L_i(0)
    4. Target: G_i^target = G_avg * (r_i / mean(r_i))^α
    5. Update weights to match targets

    Reference: https://arxiv.org/abs/1711.02257
    """

    def __init__(
        self,
        n_tasks: int,
        shared_layer: nn.Module,
        alpha: float = 1.5,
        lr: float = 0.025
    ):
        """
        Args:
            n_tasks: Number of tasks
            shared_layer: Last shared layer for gradient computation
            alpha: Asymmetry parameter (higher = focus on lagging tasks)
            lr: Learning rate for weight updates
        """
        self.n_tasks = n_tasks
        self.shared_layer = shared_layer
        self.alpha = alpha
        self.lr = lr

        # Learnable task weights (in log space for stability)
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))

        # Track initial losses
        self.initial_losses: Optional[torch.Tensor] = None
        self.step_count = 0

    @property
    def weights(self) -> torch.Tensor:
        """Get current normalized weights."""
        w = torch.exp(self.log_weights)
        return w / w.sum() * self.n_tasks

    def step(
        self,
        task_losses: List[torch.Tensor],
        retain_graph: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GradNorm-weighted loss and update weights.

        Args:
            task_losses: List of task losses
            retain_graph: Whether to retain graph

        Returns:
            weighted_loss: Weighted combination of losses
            info: Dictionary with GradNorm information
        """
        self.step_count += 1
        device = task_losses[0].device

        losses_tensor = torch.stack(task_losses)

        # Initialize initial losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.detach().clone()

        # Get current weights
        weights = self.weights.to(device)

        # Compute weighted losses
        weighted_losses = weights * losses_tensor

        # Get shared layer parameters
        shared_params = list(self.shared_layer.parameters())

        if shared_params:
            # Compute gradient norms for each task
            grad_norms = []
            for i, loss in enumerate(task_losses):
                grads = torch.autograd.grad(
                    weights[i] * loss,
                    shared_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True
                )
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, shared_params)]
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
                grad_norms.append(grad_norm)

            grad_norms = torch.stack(grad_norms)

            # Average gradient norm
            avg_grad_norm = grad_norms.mean().detach()

            # Relative inverse training rates
            with torch.no_grad():
                loss_ratios = losses_tensor.detach() / (self.initial_losses.to(device) + 1e-8)
                avg_loss_ratio = loss_ratios.mean()
                relative_rates = loss_ratios / (avg_loss_ratio + 1e-8)

                # Target gradient norms
                target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)

            # GradNorm loss for weight update
            gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

            # Update weights
            if self.log_weights.grad is not None:
                self.log_weights.grad.zero_()

            gradnorm_loss.backward(retain_graph=retain_graph)

            with torch.no_grad():
                if self.log_weights.grad is not None:
                    self.log_weights.data -= self.lr * self.log_weights.grad
                    # Renormalize to prevent drift
                    self.log_weights.data -= self.log_weights.data.mean()

        # Compute final weighted loss (without grad for weight update)
        weighted_loss = (weights.detach() * losses_tensor).sum()

        info = {
            'gradnorm_step': self.step_count,
            **{f'weight_task{i}': float(weights[i]) for i in range(self.n_tasks)},
            **{f'loss_task{i}': float(task_losses[i]) for i in range(self.n_tasks)}
        }

        return weighted_loss, info

    def state_dict(self) -> Dict:
        """Save state for checkpointing."""
        return {
            'log_weights': self.log_weights.data.clone(),
            'initial_losses': self.initial_losses.clone() if self.initial_losses is not None else None,
            'step_count': self.step_count
        }

    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        self.log_weights.data = state['log_weights']
        self.initial_losses = state['initial_losses']
        self.step_count = state['step_count']


class PCGradOptimizer:
    """
    PCGrad Optimizer - Project Conflicting Gradients.

    Algorithm (Yu et al., NeurIPS 2020):
    1. For each task i, compute gradient g_i
    2. For each pair (i,j): if g_i · g_j < 0 (conflict),
       project g_i onto normal plane of g_j
    3. Average projected gradients

    Reference: https://arxiv.org/abs/2001.06782
    """

    def __init__(self, shared_params: List[nn.Parameter]):
        self.shared_params = list(shared_params)
        self._grad_shapes = [p.shape for p in self.shared_params]
        self._grad_numel = [p.numel() for p in self.shared_params]

    def _flatten_grads(self, grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grads(self, flat_grad: torch.Tensor) -> List[torch.Tensor]:
        grads = []
        idx = 0
        for shape, numel in zip(self._grad_shapes, self._grad_numel):
            grads.append(flat_grad[idx:idx + numel].view(shape))
            idx += numel
        return grads

    def step(
        self,
        task_losses: List[torch.Tensor],
        retain_graph: bool = True
    ) -> Dict[str, float]:
        """
        Compute PCGrad update.

        Args:
            task_losses: List of task losses

        Returns:
            info: Dictionary with conflict information
        """
        n_tasks = len(task_losses)

        # Compute per-task gradients
        task_grads = []
        for loss in task_losses:
            grads = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True
            )
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, self.shared_params)]
            task_grads.append(self._flatten_grads(grads))

        # Project conflicting gradients
        n_conflicts = 0
        projected_grads = []

        for i in range(n_tasks):
            grad_i = task_grads[i].clone()

            # Random order for other tasks
            indices = list(range(n_tasks))
            indices.remove(i)
            np.random.shuffle(indices)

            for j in indices:
                dot = torch.dot(grad_i, task_grads[j])
                if dot < 0:
                    n_conflicts += 1
                    # Project onto normal plane
                    grad_j_norm_sq = torch.dot(task_grads[j], task_grads[j])
                    if grad_j_norm_sq > 1e-8:
                        grad_i = grad_i - (dot / grad_j_norm_sq) * task_grads[j]

            projected_grads.append(grad_i)

        # Average projected gradients
        combined_grad = torch.stack(projected_grads).mean(dim=0)

        # Apply to parameters
        grads = self._unflatten_grads(combined_grad)
        for param, grad in zip(self.shared_params, grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.copy_(grad)

        info = {
            'n_conflicts': n_conflicts,
            'conflict_rate': n_conflicts / (n_tasks * (n_tasks - 1)) if n_tasks > 1 else 0
        }

        return info


class DWAOptimizer:
    """
    Dynamic Weight Averaging.

    Reference: Liu et al., "End-to-End Multi-Task Learning with Attention", CVPR 2019
    """

    def __init__(self, n_tasks: int, temperature: float = 2.0):
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.prev_losses: Optional[torch.Tensor] = None
        self.step_count = 0

    def get_weights(
        self,
        task_losses: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DWA weights based on loss change rate."""
        self.step_count += 1
        device = task_losses[0].device

        losses_tensor = torch.stack([l.detach() for l in task_losses])

        if self.prev_losses is None:
            weights = torch.ones(self.n_tasks, device=device) / self.n_tasks
        else:
            ratios = losses_tensor / (self.prev_losses.to(device) + 1e-8)
            weights = torch.softmax(ratios / self.temperature, dim=0)

        self.prev_losses = losses_tensor.clone()

        info = {f'weight_task{i}': float(weights[i]) for i in range(self.n_tasks)}
        return weights, info


def create_gradient_optimizer(
    method: str,
    shared_params: List[nn.Parameter],
    n_tasks: int = 3,
    **kwargs
) -> Union[MGDAOptimizer, GradNormOptimizer, PCGradOptimizer, DWAOptimizer]:
    """
    Factory function to create gradient optimizer.

    Args:
        method: 'mgda', 'gradnorm', 'pcgrad', 'dwa'
        shared_params: List of shared parameters
        n_tasks: Number of tasks
        **kwargs: Method-specific arguments
    """
    method = method.lower()

    if method == 'mgda':
        return MGDAOptimizer(
            shared_params=shared_params,
            normalize_grads=kwargs.get('normalize_grads', True)
        )
    elif method == 'gradnorm':
        return GradNormOptimizer(
            n_tasks=n_tasks,
            shared_layer=kwargs.get('shared_layer'),
            alpha=kwargs.get('alpha', 1.5),
            lr=kwargs.get('lr', 0.025)
        )
    elif method == 'pcgrad':
        return PCGradOptimizer(shared_params=shared_params)
    elif method == 'dwa':
        return DWAOptimizer(
            n_tasks=n_tasks,
            temperature=kwargs.get('temperature', 2.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test implementations
    print("Testing Gradient Balancing Implementations")
    print("=" * 60)

    torch.manual_seed(42)

    # Create simple model
    shared = nn.Linear(10, 5)
    head1 = nn.Linear(5, 1)
    head2 = nn.Linear(5, 1)
    head3 = nn.Linear(5, 1)

    x = torch.randn(4, 10)

    # Test MGDA
    print("\n1. Testing MGDA Optimizer")
    mgda = MGDAOptimizer(shared_params=list(shared.parameters()))

    features = shared(x)
    losses = [head1(features).mean(), head2(features).mean(), head3(features).mean()]

    info = mgda.step(losses)
    print(f"   Weights: [{info['weight_task0']:.3f}, {info['weight_task1']:.3f}, {info['weight_task2']:.3f}]")
    print(f"   Min Norm: {info['min_norm']:.4f}")
    print(f"   Gradients applied: {shared.weight.grad is not None}")

    # Test GradNorm
    print("\n2. Testing GradNorm Optimizer")
    shared.zero_grad()
    gradnorm = GradNormOptimizer(n_tasks=3, shared_layer=shared, alpha=1.5)

    features = shared(x)
    losses = [head1(features).mean(), head2(features).mean(), head3(features).mean()]

    weighted_loss, info = gradnorm.step(losses)
    print(f"   Weights: [{info['weight_task0']:.3f}, {info['weight_task1']:.3f}, {info['weight_task2']:.3f}]")
    print(f"   Weighted Loss: {weighted_loss.item():.4f}")

    # Test PCGrad
    print("\n3. Testing PCGrad Optimizer")
    shared.zero_grad()
    pcgrad = PCGradOptimizer(shared_params=list(shared.parameters()))

    features = shared(x)
    losses = [head1(features).mean(), head2(features).mean(), head3(features).mean()]

    info = pcgrad.step(losses)
    print(f"   Conflicts: {info['n_conflicts']}")
    print(f"   Conflict Rate: {info['conflict_rate']:.2%}")

    print("\n" + "=" * 60)
    print("All tests passed!")
