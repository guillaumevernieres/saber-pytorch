"""Tests for optional training loss terms."""

import torch

from saber_pytorch.ml.ffnn import FFNN
from saber_pytorch.ml.losses import VerticalSmoothnessLoss, build_loss_terms


def _model_with_output_norm(output_mean: torch.Tensor, output_std: torch.Tensor) -> FFNN:
    model = FFNN(input_size=4, output_size=4, hidden_size=8)
    model.init_norm(
        input_mean=torch.zeros(4),
        input_std=torch.ones(4),
        output_mean=output_mean,
        output_std=output_std,
    )
    return model


def test_vertical_smoothness_zero_for_linear_profile():
    model = _model_with_output_norm(torch.zeros(4), torch.ones(4))
    loss = VerticalSmoothnessLoss(weight=2.0, order=2)
    predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    targets = torch.zeros_like(predictions)

    assert torch.allclose(loss(predictions, targets, model), torch.tensor(0.0))


def test_vertical_smoothness_uses_physical_output_space():
    model = _model_with_output_norm(
        output_mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        output_std=torch.tensor([1.0, 2.0, 1.0, 1.0]),
    )
    loss = VerticalSmoothnessLoss(weight=1.0, order=2)
    predictions = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    targets = torch.zeros_like(predictions)

    # Physical profile is [1, 2, 1, 1], so second differences are [-2, 1].
    expected = torch.tensor((4.0 + 1.0) / 2.0)
    assert torch.allclose(loss(predictions, targets, model), expected)


def test_build_loss_terms_from_config():
    terms = build_loss_terms(
        {
            "training": {
                "regularization": {
                    "vertical_smoothness": {
                        "enabled": True,
                        "weight": 0.25,
                        "order": 1,
                    }
                }
            }
        }
    )

    assert len(terms) == 1
    assert isinstance(terms[0], VerticalSmoothnessLoss)
    assert terms[0].weight == 0.25
    assert terms[0].order == 1
