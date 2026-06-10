from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class SVDBasis:
    """Public spectral basis for one frozen linear layer.

    U and V are frozen, public, and reproducible from the released backbone weights.
    Only the small core matrix C in the adapter is trainable/uploadable.
    """

    layer_name: str
    U: torch.Tensor
    V: torch.Tensor
    S: torch.Tensor
    rank: int
    retained_energy: float
    total_energy: float


@torch.no_grad()
def retained_energy(singular_values: torch.Tensor, rank: int) -> float:
    """Return sum_{i<=rank} s_i^2 / sum_i s_i^2 with numerical guards."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    values = singular_values.detach().float().flatten()
    if values.numel() == 0 or rank == 0:
        return 0.0
    total = torch.sum(values**2).clamp_min(1e-12)
    used = torch.sum(values[: min(rank, values.numel())] ** 2)
    return float((used / total).item())


@torch.no_grad()
def build_truncated_svd(weight: torch.Tensor, rank: int, *, layer_name: str = "") -> SVDBasis:
    """Build a deterministic truncated SVD basis from a frozen public weight matrix.

    For W_l in R^{d_out x d_in}, the adapter basis is U_{l,p}, V_{l,p} from
    W_l = U_l Sigma_l V_l^T. The returned tensors are CPU float32 to make cache
    artifacts portable and independent of the training device.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight matrix, got shape={tuple(weight.shape)}")
    if rank < 1:
        raise ValueError("rank must be >= 1")

    matrix = weight.detach().float().cpu()
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    selected_rank = min(rank, S.numel())
    total_energy = float(torch.sum(S**2).item())
    return SVDBasis(
        layer_name=layer_name,
        U=U[:, :selected_rank].contiguous(),
        V=Vh[:selected_rank, :].T.contiguous(),
        S=S[:selected_rank].contiguous(),
        rank=selected_rank,
        retained_energy=retained_energy(S, selected_rank),
        total_energy=total_energy,
    )


def allocate_layer_ranks(
    layer_singular_values: Mapping[str, torch.Tensor | Sequence[float]],
    candidate_ranks: Sequence[int] = (4, 8, 16, 32),
    *,
    param_budget: int,
    layer_weights: Mapping[str, float] | None = None,
    objective: str = "energy_per_parameter",
) -> dict[str, int]:
    """Greedily allocate public-spectrum ranks under a core-parameter budget.

    The core parameter cost of layer l is p_l^2. The allocator starts from the
    smallest candidate rank for each layer, then repeatedly applies the upgrade
    with the largest public marginal utility until the parameter budget is full.

    This is intentionally private-data-free: all utilities are functions only of
    the released backbone spectrum and optional public layer weights.
    """
    if not layer_singular_values:
        return {}
    if param_budget <= 0:
        raise ValueError("param_budget must be positive")

    ranks = sorted({int(rank) for rank in candidate_ranks if int(rank) > 0})
    if not ranks:
        raise ValueError("candidate_ranks must contain at least one positive rank")
    if objective not in {"energy", "energy_per_parameter"}:
        raise ValueError("objective must be 'energy' or 'energy_per_parameter'")

    layer_names = sorted(layer_singular_values)
    min_rank = ranks[0]
    base_cost = len(layer_names) * (min_rank**2)
    if base_cost > param_budget:
        raise ValueError(
            f"param_budget={param_budget} is too small for {len(layer_names)} layers at minimum rank {min_rank}"
        )

    tensors: dict[str, torch.Tensor] = {
        name: torch.as_tensor(values, dtype=torch.float32).flatten()
        for name, values in layer_singular_values.items()
    }
    weights = {name: float((layer_weights or {}).get(name, 1.0)) for name in layer_names}
    allocation = {name: min_rank for name in layer_names}
    used_budget = base_cost

    def utility(name: str, rank: int) -> float:
        return weights[name] * retained_energy(tensors[name], rank)

    while True:
        best: tuple[float, str, int, int] | None = None
        for name in layer_names:
            current = allocation[name]
            next_candidates = [rank for rank in ranks if rank > current]
            if not next_candidates:
                continue
            new_rank = next_candidates[0]
            extra_cost = new_rank**2 - current**2
            if used_budget + extra_cost > param_budget:
                continue
            gain = utility(name, new_rank) - utility(name, current)
            score = gain / max(extra_cost, 1) if objective == "energy_per_parameter" else gain
            candidate = (score, name, new_rank, extra_cost)
            if best is None or candidate > best:
                best = candidate

        if best is None or best[0] <= 0:
            break
        _, layer_name, new_rank, extra_cost = best
        allocation[layer_name] = new_rank
        used_budget += extra_cost

    return allocation
