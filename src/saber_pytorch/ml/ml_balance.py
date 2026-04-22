"""ML balance surface emulator compatible with the SABER TorchBalance contract.

Background
----------
SABER TorchBalance loads TorchScript modules whose Jacobians encode the
linearised relationship between background-state variables.  This module
provides a surface emulator wrapper around the generic FFNN (ffnn.py) that
satisfies the C++ interface contract.

SABER TorchBalance surface emulator contract
--------------------------------------------
The emulator must be a TorchScript module with:

  Attributes:
    input_names  : List[str]  — one name per input feature
    input_levels : List[int]  — vertical-level index for each input feature
    output_names : List[str]  — exactly one output variable name
    output_levels: List[int]  — vertical-level index for each output feature

  Method called by C++ (TorchBalanceSurfaceEmulator.cc):
    jac_physical(inputs: Tensor, mask: Tensor) -> Tensor

    inputs : [nnodes, inputSize]   float32 — packed background state
    mask   : [nnodes, 1]           float32 — pre-computed binary mask from SABER
    return : [nnodes, outputSize, inputSize]  float32

The mask is already reduced to a per-node scalar by the C++ layer before
being passed in.  It is NOT ancillary data that needs thresholding inside the
model.

Design
------
FFNNSurfaceEmulator composes with FFNN (defined in ffnn.py).  The FFNN core
handles architecture, normalization buffers, and Jacobian computation.  This
wrapper only adds the required TorchScript attributes and the correct
jac_physical signature.

The separation keeps the generic FFNN reusable for other emulator types (e.g.
a future vertical ML balance) without re-encoding the TorchBalance interface
details in the core network class.
"""

from typing import List

import torch
import torch.nn as nn

from .ffnn import FFNN


class FFNNSurfaceEmulator(nn.Module):
    """FFNN-based surface balance emulator for SABER TorchBalance.

    Attributes (TorchScript-serialized)
    ------------------------------------
    input_names  : one CF-standard name per input feature.
    input_levels : vertical-level index for each input feature (single-level
                   surface variables all use level 0).
    output_names : exactly one output variable name.
    output_levels: vertical-level index for the output (typically [0]).

    Example
    -------
    >>> emulator = FFNNSurfaceEmulator(
    ...     input_names=["sea_water_potential_temperature", "sea_water_salinity"],
    ...     output_names=["sea_ice_area_fraction"],
    ...     input_levels=[0, 0],
    ...     output_levels=[0],
    ...     hidden_size=64,
    ...     hidden_layers=3,
    ... )
    >>> emulator.init_norm(im, is_, om, os_)
    >>> scripted = torch.jit.script(emulator.eval())
    >>> scripted.save("ml_balance.ts")
    """

    input_names: List[str]
    output_names: List[str]
    input_levels: List[int]
    output_levels: List[int]

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        input_levels: List[int],
        output_levels: List[int],
        hidden_size: int = 64,
        hidden_layers: int = 3,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            input_names:   CF-standard name for each input feature.
                           len(input_names) == FFNN input_size.
            output_names:  Exactly one output variable name.
            input_levels:  Vertical-level index for each input feature.
                           Must have the same length as input_names.
            output_levels: Vertical-level index for each output feature.
                           Must have the same length as output_names.
            hidden_size:   Neurons per hidden layer.
            hidden_layers: Number of hidden layers.
            activation:    Activation function name (gelu, relu, tanh, …).
        """
        super().__init__()

        if len(output_names) != 1:
            raise ValueError(
                "FFNNSurfaceEmulator requires exactly 1 output name "
                f"(got {len(output_names)})"
            )
        if len(input_levels) != len(input_names):
            raise ValueError(
                f"input_levels length ({len(input_levels)}) must match "
                f"input_names length ({len(input_names)})"
            )
        if len(output_levels) != len(output_names):
            raise ValueError(
                f"output_levels length ({len(output_levels)}) must match "
                f"output_names length ({len(output_names)})"
            )

        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.input_levels = list(input_levels)
        self.output_levels = list(output_levels)

        self.ffnn = FFNN(
            input_size=len(input_names),
            output_size=len(output_names),
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    # ------------------------------------------------------------------
    # Normalization initialisation — delegates to the FFNN core
    # ------------------------------------------------------------------

    def init_norm(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        """Set per-feature normalization statistics (forwarded to FFNN core)."""
        self.ffnn.init_norm(input_mean, input_std, output_mean, output_std)

    # ------------------------------------------------------------------
    # Forward (nonlinear prediction) — physical space
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Nonlinear prediction in physical space.

        Args:
            inputs: [nnodes, inputSize], physical units.

        Returns:
            [nnodes, outputSize], physical units.
        """
        return self.ffnn.predict(inputs)

    # ------------------------------------------------------------------
    # SABER C++ entry point
    # ------------------------------------------------------------------

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian from the background tensor assembled by SABER C++.

        Args:
            inputs: [nnodes, inputSize]  — packed background state in physical units.
            mask:   [nnodes, 1]          — pre-computed binary mask (1 = valid node,
                                           0 = masked out).  Applied by the C++
                                           TorchBalanceSurfaceEmulator before calling
                                           this method; passed straight through here.

        Returns:
            jac: [nnodes, outputSize, inputSize]
                 Jacobian ∂y_phys/∂x_phys, zeroed at masked nodes.
        """
        jac = self.ffnn._jac_physical(inputs)
        # mask: [nnodes, 1] → [nnodes, 1, 1] broadcasts over [nnodes, out, in]
        return jac * mask.unsqueeze(2)
