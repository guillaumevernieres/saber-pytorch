"""Feed-forward neural network with input/output normalization.

Core ML model brought in from aibalance/ufsemulator, stripped of training
infrastructure.  Intended to be wrapped by TorchBalance-contract classes
(see ml_balance.py); not used directly by SABER C++.

Design notes
------------
- forward() operates in *normalized* space (used during training).
- predict() / _jac_physical() operate in *physical* space (used at inference).
- _jac_physical() is a private helper; wrappers expose it under the
  TorchBalance-required signature.
- output_size is stored as an int attribute so TorchScript does not need to
  index into nn.Sequential to recover it.
"""

from typing import List, Optional

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    table = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "silu": nn.SiLU(),
    }
    key = name.lower()
    if key not in table:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(table)}"
        )
    return table[key]


class FFNN(nn.Module):
    """MLP with per-feature input/output normalization.

    Attributes
    ----------
    input_size, output_size : int
        Stored explicitly so TorchScript methods can use them without
        inspecting nn.Sequential children.

    Buffers (serialized with TorchScript)
    --------------------------------------
    input_mean, input_std   : [input_size]
    output_mean, output_std : [output_size]
        Normalization statistics set via init_norm().  Default: identity
        (mean=0, std=1).
    """

    input_size: int
    output_size: int

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers: int = 2,
        activation: str = "gelu",
        use_conv1d: bool = False,
        conv_channels: int = 32,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.use_conv1d = use_conv1d
        if use_conv1d:
            padding = conv_kernel_size // 2
            self.conv1d = nn.Conv1d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=conv_kernel_size,
                padding=padding,
            )
            self.conv_activation = _make_activation(activation)
            first_layer_input_size = conv_channels * input_size
        else:
            self.conv1d = None
            self.conv_activation = None
            first_layer_input_size = input_size

        layers: List[nn.Module] = [
            nn.Linear(first_layer_input_size, hidden_size),
            _make_activation(activation),
        ]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(_make_activation(activation))
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_std", torch.ones(input_size))
        self.register_buffer("output_mean", torch.zeros(output_size))
        self.register_buffer("output_std", torch.ones(output_size))

        self.input_mean: torch.Tensor
        self.input_std: torch.Tensor
        self.output_mean: torch.Tensor
        self.output_std: torch.Tensor

    # ------------------------------------------------------------------
    # Normalization helpers (not exported; called from within the module)
    # ------------------------------------------------------------------

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def _denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.output_std + self.output_mean

    # ------------------------------------------------------------------
    # Forward pass — normalized space
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate network in normalized space.

        Args:
            x: [N, input_size], already normalized.

        Returns:
            [N, output_size], normalized.
        """
        if self.use_conv1d:
            x = x.unsqueeze(1)
            if self.conv1d is None or self.conv_activation is None:
                raise RuntimeError("Convolution layers are not initialized")
            x = self.conv1d(x)
            x = self.conv_activation(x)
            x = x.flatten(1)

        return self.network(x)

    # ------------------------------------------------------------------
    # Physical-space inference
    # ------------------------------------------------------------------

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end inference: physical input → physical output.

        Args:
            x: [N, input_size], physical units.

        Returns:
            [N, output_size], physical units.
        """
        return self._denormalize_output(self.forward(self._normalize_input(x)))

    # ------------------------------------------------------------------
    # Jacobian (physical space) — private helper for wrappers
    # ------------------------------------------------------------------

    def _jac_physical(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full Jacobian ∂y_phys/∂x_phys via reverse-mode AD.

        Computed row-by-row (one backward pass per output dimension), then
        scaled from normalized to physical space with the chain rule:

            J_phys = diag(output_std) · J_norm · diag(1 / input_std)

        Args:
            inputs: [N, input_size], physical units.

        Returns:
            [N, output_size, input_size], physical units.
        """
        x_norm = self._normalize_input(inputs)

        rows: List[torch.Tensor] = []
        for i in range(self.output_size):
            x_copy = x_norm.detach().requires_grad_(True)
            y_norm = self.forward(x_copy)

            g = torch.zeros_like(y_norm)
            g[:, i] = 1.0

            # TorchScript requires grad_outputs as List[Optional[Tensor]]
            grad_outputs: List[Optional[torch.Tensor]] = [g]
            grads = torch.autograd.grad(
                [y_norm], [x_copy], grad_outputs=grad_outputs
            )
            # grads[0] is Optional[Tensor] in TorchScript; unwrap before append
            grad = grads[0]
            if grad is None:
                raise RuntimeError("Jacobian: gradient is None")
            rows.append(grad)

        jac_norm = torch.stack(rows, dim=1)  # [N, output_size, input_size]

        out_std = self.output_std.view(1, -1, 1)
        in_std = self.input_std.view(1, 1, -1)
        return jac_norm * out_std / in_std

    # ------------------------------------------------------------------
    # Normalization initialisation (called from build scripts / tests)
    # ------------------------------------------------------------------

    def init_norm(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        """Set per-feature normalization statistics."""
        self.input_mean.data = input_mean.clone()
        self.input_std.data = input_std.clone()
        self.output_mean.data = output_mean.clone()
        self.output_std.data = output_std.clone()

    def init_weights(self) -> None:
        """Xavier-normal weight initialisation for Conv1d and Linear layers."""
        if self.conv1d is not None:
            nn.init.xavier_normal_(self.conv1d.weight)
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)

        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def save_norm(self, path: str) -> None:
        """Save normalization buffers to a separate file.

        The file is read by build_ml_balance_emulator.py when constructing
        the TorchScript export.  Format mirrors the aibalance convention so
        that existing checkpoints remain loadable.
        """
        torch.save(
            {
                "input_mean": self.input_mean.cpu(),
                "input_std": self.input_std.cpu(),
                "output_mean": self.output_mean.cpu(),
                "output_std": self.output_std.cpu(),
            },
            path,
        )

    def load_norm(self, path: str) -> None:
        """Load normalization buffers saved by save_norm()."""
        moments = torch.load(path, weights_only=False)
        self.input_mean.data = moments["input_mean"]
        self.input_std.data = moments["input_std"]
        self.output_mean.data = moments["output_mean"]
        self.output_std.data = moments["output_std"]
