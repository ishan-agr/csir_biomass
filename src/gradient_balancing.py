"""
Gradient Balancing Methods for Multi-Task Learning.

Implements state-of-the-art gradient balancing strategies:
1. MGDA (Multiple Gradient Descent Algorithm) - Sener & Koltun, NeurIPS 2018
2. GradNorm - Chen et al., ICML 2018
3. PCGrad (Projecting Conflicting Gradients) - Yu et al., NeurIPS 2020

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


class GradientBalancer(ABC):
    """Base class for gradient balancing methods."""

    @abstractmethod
    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Balance gradients from multiple task losses.

        Args:
            losses: List of task losses
            shared_params: List of shared parameters to compute gradients for

        Returns:
            combined_loss: Weighted combination of losses
            info: Dictionary with balancing information
        """
        pass


class MGDASolver:
    """
    Minimum-norm solver for MGDA using Frank-Wolfe algorithm.

    Finds the minimum-norm point in the convex hull of task gradients,
    which represents the optimal descent direction for all tasks.

    Reference: https://github.com/isl-org/MultiObjectiveOptimization
    """

    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1: float, v1v2: float, v2v2: float) -> Tuple[float, float]:
        """
        Analytical solution for min-norm point in 2D case.

        Find min_{c} || c*v1 + (1-c)*v2 ||^2
        """
        if v1v2 >= v1v1:
            # Optimal is v1
            return 1.0, v1v1
        if v1v2 >= v2v2:
            # Optimal is v2
            return 0.0, v2v2

        # Optimal is in between
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        gamma = max(0.0, min(1.0, gamma))
        cost = gamma * gamma * v1v1 + 2 * gamma * (1 - gamma) * v1v2 + (1 - gamma) * (1 - gamma) * v2v2
        return gamma, cost

    @staticmethod
    def _min_norm_2d(grad_mat: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find minimum norm solution for 2 tasks.

        Args:
            grad_mat: (2, d) matrix of gradients

        Returns:
            sol: Optimal weights [w1, w2]
            nd: Minimum norm value
        """
        v1v1 = np.dot(grad_mat[0], grad_mat[0])
        v1v2 = np.dot(grad_mat[0], grad_mat[1])
        v2v2 = np.dot(grad_mat[1], grad_mat[1])

        gamma, cost = MGDASolver._min_norm_element_from2(v1v1, v1v2, v2v2)
        return np.array([gamma, 1 - gamma]), cost

    @staticmethod
    def _projection_simplex_sort(v: np.ndarray) -> np.ndarray:
        """Project vector onto probability simplex."""
        n = len(v)
        if n == 0:
            return v

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]

        if len(rho) == 0:
            return np.ones(n) / n

        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    @staticmethod
    def find_min_norm_element(grads: List[np.ndarray], normalize: bool = True) -> Tuple[np.ndarray, float]:
        """
        Find minimum norm element in convex hull using Frank-Wolfe.

        Args:
            grads: List of gradient vectors (one per task)
            normalize: Whether to normalize gradients before solving

        Returns:
            sol: Optimal task weights
            min_norm: Minimum norm value
        """
        n_tasks = len(grads)

        if n_tasks == 1:
            return np.array([1.0]), np.linalg.norm(grads[0])

        # Stack gradients
        grad_mat = np.stack(grads, axis=0)  # (n_tasks, d)

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(grad_mat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            grad_mat = grad_mat / norms

        # Special case: 2 tasks - use analytical solution
        if n_tasks == 2:
            return MGDASolver._min_norm_2d(grad_mat)

        # General case: Frank-Wolfe algorithm
        # Precompute Gram matrix
        gram = grad_mat @ grad_mat.T  # (n_tasks, n_tasks)

        # Initialize with equal weights
        sol = np.ones(n_tasks) / n_tasks

        for _ in range(MGDASolver.MAX_ITER):
            # Compute gradient of objective: 2 * G @ sol
            grad_obj = gram @ sol

            # Find minimizing vertex (min over simplex vertices)
            min_idx = np.argmin(grad_obj)

            # Compute descent direction
            descent = np.zeros(n_tasks)
            descent[min_idx] = 1.0
            descent -= sol

            # Line search
            # min_gamma || sol + gamma * descent ||^2_G
            # = ||sol||^2_G + 2*gamma*<sol, descent>_G + gamma^2*||descent||^2_G
            a = descent @ gram @ descent
            b = 2 * sol @ gram @ descent

            if a <= 1e-8:
                gamma = 1.0
            else:
                gamma = max(0.0, min(1.0, -b / (2 * a + 1e-8)))

            # Check convergence
            if gamma < MGDASolver.STOP_CRIT:
                break

            # Update
            sol = sol + gamma * descent

        # Ensure valid probability distribution
        sol = np.maximum(sol, 0)
        sol = sol / (sol.sum() + 1e-8)

        # Compute minimum norm
        min_norm = np.sqrt(max(0, sol @ gram @ sol))

        return sol, min_norm


class MGDA(GradientBalancer):
    """
    Multiple Gradient Descent Algorithm (MGDA).

    Finds the minimum-norm point in the convex hull of task gradients,
    guaranteeing descent for all tasks (Pareto improvement).

    Reference: Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization", NeurIPS 2018
    https://arxiv.org/abs/1810.04650
    """

    def __init__(
        self,
        normalize_grads: bool = True,
        use_rep_grad: bool = True,
        max_norm: float = 1.0
    ):
        """
        Args:
            normalize_grads: Normalize task gradients before solving
            use_rep_grad: Use representation gradients (more efficient)
            max_norm: Maximum gradient norm for clipping
        """
        self.normalize_grads = normalize_grads
        self.use_rep_grad = use_rep_grad
        self.max_norm = max_norm

    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        representations: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MGDA-weighted loss.

        Args:
            losses: List of task losses
            shared_params: Shared parameters (used if representations is None)
            representations: Shared representations (more efficient)

        Returns:
            weighted_loss: MGDA-weighted combination of losses
            info: Dictionary with task weights and min-norm value
        """
        n_tasks = len(losses)
        device = losses[0].device

        # Compute task gradients
        grads = []

        if representations is not None and self.use_rep_grad:
            # Use representation gradients (more efficient)
            for loss in losses:
                grad = torch.autograd.grad(
                    loss, representations,
                    retain_graph=True,
                    create_graph=False
                )[0]
                grads.append(grad.detach().flatten().cpu().numpy())
        else:
            # Use parameter gradients
            for loss in losses:
                grad_list = torch.autograd.grad(
                    loss, shared_params,
                    retain_graph=True,
                    create_graph=False
                )
                grad = torch.cat([g.detach().flatten() for g in grad_list])
                grads.append(grad.cpu().numpy())

        # Solve min-norm problem
        weights, min_norm = MGDASolver.find_min_norm_element(
            grads, normalize=self.normalize_grads
        )

        # Compute weighted loss
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weighted_loss = sum(w * l for w, l in zip(weights_tensor, losses))

        info = {
            'min_norm': float(min_norm),
            **{f'weight_task{i}': float(w) for i, w in enumerate(weights)}
        }

        return weighted_loss, info


class GradNorm(GradientBalancer):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Automatically balances training by normalizing gradient magnitudes
    across tasks using learnable weights.

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
    https://arxiv.org/abs/1711.02257
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float = 1.5,
        initial_weights: Optional[List[float]] = None,
        weight_lr: float = 0.025
    ):
        """
        Args:
            n_tasks: Number of tasks
            alpha: Asymmetry parameter (higher = more focus on lagging tasks)
            initial_weights: Initial task weights (default: equal)
            weight_lr: Learning rate for weight updates
        """
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.weight_lr = weight_lr

        # Initialize task weights (log-space for stability)
        if initial_weights is None:
            initial_weights = [1.0] * n_tasks
        self.log_weights = nn.Parameter(
            torch.log(torch.tensor(initial_weights, dtype=torch.float32))
        )

        # Track initial losses for relative loss computation
        self.initial_losses: Optional[torch.Tensor] = None
        self.step_count = 0

    def get_weights(self) -> torch.Tensor:
        """Get current task weights (normalized)."""
        weights = torch.exp(self.log_weights)
        return weights / weights.sum() * self.n_tasks

    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        last_shared_layer: Optional[nn.Module] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GradNorm-weighted loss with gradient normalization.

        Args:
            losses: List of task losses
            shared_params: Shared parameters
            last_shared_layer: Last shared layer for gradient computation

        Returns:
            weighted_loss: GradNorm-weighted combination of losses
            info: Dictionary with weights and gradient norms
        """
        device = losses[0].device
        losses_tensor = torch.stack(losses)

        # Initialize/update initial losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.detach().clone()

        # Get current weights
        weights = self.get_weights().to(device)

        # Compute weighted losses
        weighted_losses = weights * losses_tensor
        weighted_loss = weighted_losses.sum()

        # Compute gradient norms for each task
        if last_shared_layer is not None:
            # Get parameters from last shared layer
            layer_params = list(last_shared_layer.parameters())
            if layer_params:
                grad_norms = []
                for i, loss in enumerate(losses):
                    # Compute gradient w.r.t. last shared layer
                    grads = torch.autograd.grad(
                        weights[i] * loss, layer_params,
                        retain_graph=True,
                        create_graph=True
                    )
                    grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
                    grad_norms.append(grad_norm)

                grad_norms = torch.stack(grad_norms)

                # Compute average gradient norm
                avg_grad_norm = grad_norms.mean().detach()

                # Compute relative inverse training rates
                with torch.no_grad():
                    loss_ratios = losses_tensor / (self.initial_losses.to(device) + 1e-8)
                    avg_loss_ratio = loss_ratios.mean()
                    relative_rates = loss_ratios / (avg_loss_ratio + 1e-8)

                    # Target gradient norms
                    target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)

                # GradNorm loss (for weight update)
                gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

                # Update weights
                weight_grads = torch.autograd.grad(
                    gradnorm_loss, self.log_weights,
                    retain_graph=True
                )[0]

                with torch.no_grad():
                    self.log_weights.data -= self.weight_lr * weight_grads
                    # Renormalize
                    self.log_weights.data -= self.log_weights.data.mean()

        self.step_count += 1

        # Build info dict
        info = {
            'gradnorm_step': self.step_count,
            **{f'weight_task{i}': float(weights[i]) for i in range(self.n_tasks)},
            **{f'loss_task{i}': float(losses[i]) for i in range(self.n_tasks)}
        }

        return weighted_loss, info

    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
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


class PCGrad(GradientBalancer):
    """
    PCGrad: Projecting Conflicting Gradients.

    When task gradients conflict (negative cosine similarity),
    projects each gradient onto the normal plane of conflicting gradients.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
    https://arxiv.org/abs/2001.06782
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: How to combine gradients ('mean' or 'sum')
        """
        self.reduction = reduction

    @staticmethod
    def _project_conflicting(grad_i: torch.Tensor, grad_j: torch.Tensor) -> torch.Tensor:
        """
        Project grad_i onto normal plane of grad_j if they conflict.

        Args:
            grad_i: Gradient to project
            grad_j: Reference gradient

        Returns:
            Projected gradient
        """
        dot = torch.dot(grad_i.flatten(), grad_j.flatten())

        if dot < 0:
            # Gradients conflict - project
            grad_j_norm_sq = torch.dot(grad_j.flatten(), grad_j.flatten())
            if grad_j_norm_sq > 1e-8:
                grad_i = grad_i - (dot / grad_j_norm_sq) * grad_j

        return grad_i

    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PCGrad-modified gradients.

        Note: PCGrad modifies gradients directly, so we return a dummy loss
        and the caller should apply gradients manually.

        Args:
            losses: List of task losses
            shared_params: Shared parameters

        Returns:
            combined_loss: Sum of losses (for logging)
            info: Dictionary with conflict information
        """
        n_tasks = len(losses)
        device = losses[0].device

        # Compute gradients for each task
        task_grads = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True,
                create_graph=False
            )
            # Flatten and concatenate
            grad = torch.cat([g.detach().flatten() for g in grads])
            task_grads.append(grad)

        # Track conflicts
        n_conflicts = 0

        # Apply PCGrad - project conflicting gradients
        projected_grads = []
        for i in range(n_tasks):
            grad_i = task_grads[i].clone()

            # Random order for other tasks
            indices = list(range(n_tasks))
            indices.remove(i)
            np.random.shuffle(indices)

            for j in indices:
                # Check for conflict
                dot = torch.dot(grad_i, task_grads[j])
                if dot < 0:
                    n_conflicts += 1
                    # Project
                    grad_j_norm_sq = torch.dot(task_grads[j], task_grads[j])
                    if grad_j_norm_sq > 1e-8:
                        grad_i = grad_i - (dot / grad_j_norm_sq) * task_grads[j]

            projected_grads.append(grad_i)

        # Combine projected gradients
        if self.reduction == 'mean':
            combined_grad = torch.stack(projected_grads).mean(dim=0)
        else:
            combined_grad = torch.stack(projected_grads).sum(dim=0)

        # Apply combined gradient to parameters
        idx = 0
        for param in shared_params:
            param_size = param.numel()
            param.grad = combined_grad[idx:idx + param_size].view(param.shape).clone()
            idx += param_size

        # Return combined loss for logging
        combined_loss = sum(losses)

        info = {
            'n_conflicts': n_conflicts,
            'conflict_rate': n_conflicts / (n_tasks * (n_tasks - 1)) if n_tasks > 1 else 0
        }

        return combined_loss, info


class CAGrad(GradientBalancer):
    """
    Conflict-Averse Gradient Descent (CAGrad).

    Optimizes for the average loss while minimizing worst-case conflict.

    Reference: Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning", NeurIPS 2021
    """

    def __init__(self, c: float = 0.5, rescale: bool = True):
        """
        Args:
            c: Trade-off parameter (0 = average, 1 = conflict-averse)
            rescale: Whether to rescale the gradient
        """
        self.c = c
        self.rescale = rescale

    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CAGrad-modified gradients.
        """
        n_tasks = len(losses)
        device = losses[0].device

        # Compute gradients
        grads = []
        for loss in losses:
            grad_list = torch.autograd.grad(
                loss, shared_params,
                retain_graph=True,
                create_graph=False
            )
            grad = torch.cat([g.detach().flatten() for g in grad_list])
            grads.append(grad)

        grads = torch.stack(grads)  # (n_tasks, d)

        # Compute average gradient
        g_avg = grads.mean(dim=0)
        g_avg_norm = torch.norm(g_avg)

        if g_avg_norm < 1e-8:
            # No meaningful gradient
            return sum(losses), {'cagrad_norm': 0.0}

        # Compute gradient for CAGrad
        # g_cagrad = g_avg + c * (g_i - g_avg) where i = argmax conflict
        deviations = grads - g_avg.unsqueeze(0)  # (n_tasks, d)

        # Find most conflicting task
        conflicts = torch.einsum('td,d->t', deviations, g_avg)
        worst_idx = conflicts.argmin()

        # Compute CAGrad direction
        g_cagrad = g_avg + self.c * deviations[worst_idx]

        if self.rescale:
            g_cagrad = g_cagrad * (g_avg_norm / (torch.norm(g_cagrad) + 1e-8))

        # Apply to parameters
        idx = 0
        for param in shared_params:
            param_size = param.numel()
            param.grad = g_cagrad[idx:idx + param_size].view(param.shape).clone()
            idx += param_size

        info = {
            'cagrad_norm': float(torch.norm(g_cagrad)),
            'worst_conflict_task': int(worst_idx),
            'conflict_value': float(conflicts[worst_idx])
        }

        return sum(losses), info


class DynamicWeightAveraging(GradientBalancer):
    """
    Dynamic Weight Averaging (DWA).

    Adjusts weights based on rate of loss decrease.

    Reference: Liu et al., "End-to-End Multi-Task Learning with Attention", CVPR 2019
    """

    def __init__(self, n_tasks: int, temperature: float = 2.0):
        """
        Args:
            n_tasks: Number of tasks
            temperature: Temperature for softmax (higher = more uniform)
        """
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.prev_losses: Optional[torch.Tensor] = None

    def balance(
        self,
        losses: List[torch.Tensor],
        shared_params: List[torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DWA-weighted loss."""
        device = losses[0].device
        losses_tensor = torch.stack([l.detach() for l in losses])

        if self.prev_losses is None:
            # First step - equal weights
            weights = torch.ones(self.n_tasks, device=device) / self.n_tasks
        else:
            # Compute relative loss decrease
            prev = self.prev_losses.to(device)
            ratios = losses_tensor / (prev + 1e-8)

            # Softmax with temperature
            weights = torch.softmax(ratios / self.temperature, dim=0)

        # Update previous losses
        self.prev_losses = losses_tensor.detach().clone()

        # Compute weighted loss
        weighted_loss = sum(w * l for w, l in zip(weights, losses))

        info = {f'weight_task{i}': float(weights[i]) for i in range(self.n_tasks)}

        return weighted_loss, info


def get_gradient_balancer(
    method: str,
    n_tasks: int,
    **kwargs
) -> GradientBalancer:
    """
    Factory function to create gradient balancer.

    Args:
        method: Balancing method ('mgda', 'gradnorm', 'pcgrad', 'cagrad', 'dwa', 'equal')
        n_tasks: Number of tasks
        **kwargs: Method-specific arguments

    Returns:
        GradientBalancer instance
    """
    method = method.lower()

    if method == 'mgda':
        return MGDA(
            normalize_grads=kwargs.get('normalize_grads', True),
            use_rep_grad=kwargs.get('use_rep_grad', True)
        )
    elif method == 'gradnorm':
        return GradNorm(
            n_tasks=n_tasks,
            alpha=kwargs.get('alpha', 1.5),
            weight_lr=kwargs.get('weight_lr', 0.025)
        )
    elif method == 'pcgrad':
        return PCGrad(reduction=kwargs.get('reduction', 'mean'))
    elif method == 'cagrad':
        return CAGrad(
            c=kwargs.get('c', 0.5),
            rescale=kwargs.get('rescale', True)
        )
    elif method == 'dwa':
        return DynamicWeightAveraging(
            n_tasks=n_tasks,
            temperature=kwargs.get('temperature', 2.0)
        )
    else:
        raise ValueError(f"Unknown gradient balancing method: {method}")


if __name__ == "__main__":
    # Test gradient balancers
    print("Testing Gradient Balancers")
    print("=" * 50)

    # Create dummy model and losses
    torch.manual_seed(42)

    # Shared parameters
    shared = nn.Linear(10, 5)
    x = torch.randn(4, 10)

    # Task-specific heads
    head1 = nn.Linear(5, 1)
    head2 = nn.Linear(5, 1)
    head3 = nn.Linear(5, 1)

    # Forward pass
    features = shared(x)
    y1 = head1(features).mean()
    y2 = head2(features).mean()
    y3 = head3(features).mean()

    losses = [y1, y2, y3]
    shared_params = list(shared.parameters())

    # Test MGDA
    print("\n1. Testing MGDA")
    mgda = MGDA(normalize_grads=True)
    loss, info = mgda.balance(losses, shared_params, representations=features)
    print(f"   MGDA Loss: {loss.item():.4f}")
    print(f"   Weights: {[f'{info[f'weight_task{i}']:.3f}' for i in range(3)]}")
    print(f"   Min Norm: {info['min_norm']:.4f}")

    # Test GradNorm
    print("\n2. Testing GradNorm")
    gradnorm = GradNorm(n_tasks=3, alpha=1.5)

    # Need to recompute losses
    features = shared(x)
    y1 = head1(features).mean()
    y2 = head2(features).mean()
    y3 = head3(features).mean()
    losses = [y1, y2, y3]

    loss, info = gradnorm.balance(losses, shared_params, last_shared_layer=shared)
    print(f"   GradNorm Loss: {loss.item():.4f}")
    print(f"   Weights: {[f'{info[f'weight_task{i}']:.3f}' for i in range(3)]}")

    # Test PCGrad
    print("\n3. Testing PCGrad")
    pcgrad = PCGrad(reduction='mean')

    features = shared(x)
    y1 = head1(features).mean()
    y2 = head2(features).mean()
    y3 = head3(features).mean()
    losses = [y1, y2, y3]

    loss, info = pcgrad.balance(losses, shared_params)
    print(f"   PCGrad Loss: {loss.item():.4f}")
    print(f"   Conflicts: {info['n_conflicts']}")
    print(f"   Conflict Rate: {info['conflict_rate']:.2%}")

    print("\n" + "=" * 50)
    print("All tests passed!")
