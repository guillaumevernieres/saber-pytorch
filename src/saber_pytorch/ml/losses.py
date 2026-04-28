"""Optional auxiliary loss terms for ML balance training."""

from typing import Any, Dict, List

import torch
import torch.nn as nn


class VerticalSmoothnessLoss(nn.Module):
    """Penalize vertical roughness in physical-space profile predictions."""

    def __init__(self, weight: float, order: int = 2) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError("vertical smoothness weight must be non-negative")
        if order not in (1, 2):
            raise ValueError("vertical smoothness order must be 1 or 2")
        self.weight = float(weight)
        self.order = int(order)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        del targets
        predictions_phys = model._denormalize_output(predictions)
        if predictions_phys.dim() < 2 or predictions_phys.shape[-1] <= self.order:
            return predictions_phys.sum() * 0.0

        if self.order == 1:
            diff = predictions_phys[:, 1:] - predictions_phys[:, :-1]
        else:
            diff = (
                predictions_phys[:, 2:]
                - 2.0 * predictions_phys[:, 1:-1]
                + predictions_phys[:, :-2]
            )
        return self.weight * torch.mean(diff ** 2)


def build_loss_terms(config: Dict[str, Any]) -> List[nn.Module]:
    """Build configured auxiliary loss terms.

    The factory keeps config-specific branching out of the training step.
    """
    terms: List[nn.Module] = []
    reg = config.get("training", {}).get("regularization", {})

    smooth = reg.get("vertical_smoothness", {})
    if smooth.get("enabled", False):
        terms.append(
            VerticalSmoothnessLoss(
                weight=float(smooth.get("weight", 1.0e-4)),
                order=int(smooth.get("order", 2)),
            )
        )

    return terms
