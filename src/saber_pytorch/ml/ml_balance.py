"""ML balance emulators compatible with the SABER TorchBalance contract.

Background
----------
SABER TorchBalance loads TorchScript modules whose Jacobians encode the
linearised relationship between background-state variables.  This module
provides surface and vertical emulator wrappers around the generic FFNN
(ffnn.py) that satisfy the C++ interface contract.

SABER TorchBalance emulator contract
-------------------------------------
The emulator must be a TorchScript module with:

  Attributes:
    input_names  : List[str]  — one name per input feature
    input_levels : List[int]  — vertical-level index for each input feature
    output_names : List[str]  — one name per output feature
    output_levels: List[int]  — vertical-level index for each output feature

  Method called by C++:
    jac_physical(inputs: Tensor, mask: Tensor) -> Tensor

    inputs : [nnodes, inputSize]   float32 — packed background state
    mask   : [nnodes, 1]           float32 — pre-computed binary mask from SABER
    return : [nnodes, outputSize, inputSize]  float32

The mask is already reduced to a per-node scalar by the C++ layer before
being passed in.  It is NOT ancillary data that needs thresholding inside the
model.

Design
------
FFNNSurfaceEmulator and FFNNVerticalEmulator both compose with FFNN (ffnn.py).
The FFNN core handles architecture, normalization buffers, and Jacobian
computation.  These wrappers add the required TorchScript attributes and the
correct jac_physical signature.

Surface emulator  (TorchBalanceSurfaceEmulator.cc)
  jac_physical(inputs, mask) → [nnodes, outputSize, inputSize]
  input_names: one entry per feature (each at a specific single level).
  input_levels: level index for each feature.
  output_names: exactly one entry.

Vertical emulator  (TorchBalanceVerticalEmulator.cc)
  jac_physical(inputs, mask, row_indices, col_indices) → [nnodes, nRequestedPairs]
  input_names: one entry per variable (C++ reads all levels from Atlas at runtime).
  output_names: exactly one entry.
  input_levels / output_levels: not read by the vertical C++ code.
  row_indices: LongTensor of output level indices whose Jacobians are requested.
  col_indices: LongTensor of input column indices whose Jacobians are requested.
"""

from typing import List, Optional

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
    >>> scripted.save("surface_ml_balance.ts")
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


class FFNNVerticalEmulator(nn.Module):
    """FFNN-based vertical balance emulator for SABER TorchBalanceVerticalEmulator.

    Maps full vertical profiles of N input variables to the full vertical
    profile of one output variable.

    Differences from FFNNSurfaceEmulator
    -------------------------------------
    - jac_physical takes row_indices and col_indices LongTensors and returns
      only the requested compact Jacobian entries: [nnodes, nRequestedPairs].
      This matches the TorchBalanceVerticalEmulator.cc contract.
    - input_names holds one entry per input variable (not per feature); the C++
      reads the actual level count from the Atlas FieldSet at runtime.
    - input_levels / output_levels are not used by the vertical C++ code.

    Example
    -------
    >>> emulator = FFNNVerticalEmulator(
    ...     input_variable_names=["sea_water_potential_temperature", "depth"],
    ...     output_variable_name="sea_water_salinity",
    ...     num_levels=50,
    ...     hidden_size=256,
    ...     hidden_layers=4,
    ...     use_conv1d=True,
    ...     conv_channels=32,
    ...     conv_kernel_size=5,
    ... )
    >>> emulator.init_norm(im, is_, om, os_)
    >>> scripted = torch.jit.script(emulator.eval())
    >>> scripted.save("vertical_ml_balance_salt_profile.ts")
    """

    input_names: List[str]
    output_names: List[str]

    def __init__(
        self,
        input_variable_names: List[str],
        output_variable_name: str,
        num_levels: int,
        hidden_size: int = 256,
        hidden_layers: int = 4,
        activation: str = "gelu",
        use_conv1d: bool = False,
        conv_channels: int = 32,
        conv_kernel_size: int = 5,
    ) -> None:
        """
        Args:
            input_variable_names: CF-standard name for each input variable.
            output_variable_name: CF-standard name for the single output variable.
            num_levels:           Number of vertical levels per variable.
            hidden_size:          Neurons per hidden layer.
            hidden_layers:        Number of hidden layers.
            activation:           Activation function name (gelu, relu, …).
            use_conv1d:           Prepend a Conv1d layer over the level dimension.
            conv_channels:        Output channels for the Conv1d layer.
            conv_kernel_size:     Kernel size for the Conv1d layer.
        """
        super().__init__()

        self.input_names = list(input_variable_names)
        self.output_names = [output_variable_name]

        input_size = len(input_variable_names) * num_levels
        output_size = num_levels

        self.ffnn = FFNN(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation=activation,
            use_conv1d=use_conv1d,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
        )

    def init_norm(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        """Set per-feature normalization statistics (forwarded to FFNN core)."""
        self.ffnn.init_norm(input_mean, input_std, output_mean, output_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Nonlinear prediction in physical space.

        Args:
            inputs: [nnodes, nTotalInputLevels], physical units.

        Returns:
            [nnodes, nOutputLevels], physical units.
        """
        return self.ffnn.predict(inputs)

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian for requested output/input pairs, called by TorchBalanceVerticalEmulator.cc.

        Args:
            inputs:      [nnodes, nTotalInputLevels] — all input levels packed, physical units.
            mask:        [nnodes, 1] — pre-computed binary mask (1 = valid node).
            row_indices: [nRequestedPairs] LongTensor — output level indices whose
                         Jacobians are needed.
            col_indices: [nRequestedPairs] LongTensor — input column indices whose
                         Jacobians are needed.

        Returns:
            jac: [nnodes, nRequestedPairs]
                 Jacobian ∂y_phys/∂x_phys for the requested pairs, zeroed at masked nodes.
        """
        jac = self.ffnn._jac_physical(inputs)             # [nnodes, nOutputLevels, nTotalInputLevels]
        jac_rows = jac.index_select(1, row_indices)       # [nnodes, nRequestedPairs, nTotalInputLevels]
        gather_cols = col_indices.view(1, -1, 1).expand(jac.shape[0], -1, 1)
        jac_pairs = jac_rows.gather(2, gather_cols).squeeze(2)
        return jac_pairs * mask


class FFNNSalinityProfileEmulator(nn.Module):
    """Salinity-profile emulator with salinity-specific vertical preprocessing.

    This wrapper is intentionally separate from FFNNVerticalEmulator.  It keeps
    the generic vertical emulator simple while giving the salinity profile path
    a dedicated place for layer-thickness coordinates, reduced-grid
    interpolation, and minimum-thickness masking.
    """

    input_names: List[str]
    output_names: List[str]
    source_num_levels: int
    target_num_levels: int
    fill_value_threshold: float
    min_layer_thickness: float
    use_temperature_gradient_grid: bool
    temperature_gradient_weight: float

    def __init__(
        self,
        temperature_variable_name: str,
        thickness_variable_name: str,
        output_variable_name: str,
        source_num_levels: int,
        target_num_levels: int,
        hidden_size: int = 256,
        hidden_layers: int = 4,
        activation: str = "gelu",
        use_conv1d: bool = False,
        conv_channels: int = 32,
        conv_kernel_size: int = 5,
        reduced_grid_method: str = "uniform_depth",
        temperature_gradient_weight: float = 0.0,
    ) -> None:
        super().__init__()

        if source_num_levels <= 0:
            raise ValueError("source_num_levels must be positive")
        if target_num_levels <= 0:
            raise ValueError("target_num_levels must be positive")

        self.input_names = [temperature_variable_name, thickness_variable_name]
        self.output_names = [output_variable_name]
        self.source_num_levels = int(source_num_levels)
        self.target_num_levels = int(target_num_levels)
        self.fill_value_threshold = 9000.0
        self.min_layer_thickness = 0.1
        method = reduced_grid_method.lower()
        self.use_temperature_gradient_grid = method in (
            "temperature_gradient",
            "temp_gradient",
        )
        self.temperature_gradient_weight = float(temperature_gradient_weight)

        self.ffnn = FFNN(
            input_size=2 * self.target_num_levels,
            output_size=self.target_num_levels,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation=activation,
            use_conv1d=use_conv1d,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
        )

    def init_norm(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        """Set reduced-grid normalization statistics (forwarded to FFNN core)."""
        self.ffnn.init_norm(input_mean, input_std, output_mean, output_std)

    def _temperature_block(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, : self.source_num_levels]

    def _thickness_block(self, inputs: torch.Tensor) -> torch.Tensor:
        start = self.source_num_levels
        end = 2 * self.source_num_levels
        return inputs[:, start:end]

    def _valid_thickness_mask(self, thickness: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(thickness) & (thickness > self.min_layer_thickness)

    def _valid_temperature_mask(self, temperature: torch.Tensor) -> torch.Tensor:
        return (
            torch.isfinite(temperature)
            & (torch.abs(temperature) < self.fill_value_threshold)
        )

    def _source_mid_depths(self, thickness: torch.Tensor) -> torch.Tensor:
        valid_thickness = self._valid_thickness_mask(thickness)
        safe_thickness = torch.where(
            valid_thickness, thickness, torch.zeros_like(thickness)
        )
        return torch.cumsum(safe_thickness, dim=1) - 0.5 * safe_thickness

    def _uniform_target_depth(
        self,
        target_level: int,
        first_depth: torch.Tensor,
        last_depth: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_num_levels == 1:
            return first_depth
        fraction = float(target_level) / float(self.target_num_levels - 1)
        return first_depth + fraction * (last_depth - first_depth)

    def _temperature_gradient_target_depth(
        self,
        target_level: int,
        first: int,
        last: int,
        temperature: torch.Tensor,
        depths: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        first_depth = depths[first]
        last_depth = depths[last]
        if self.target_num_levels == 1:
            return first_depth

        fraction = float(target_level) / float(self.target_num_levels - 1)
        max_slope = first_depth * 0.0
        prev = -1
        for level in range(first, self.source_num_levels):
            if bool(valid[level]):
                if prev >= 0:
                    dz = depths[level] - depths[prev]
                    if bool(dz > 1.0e-12):
                        slope = torch.abs((temperature[level] - temperature[prev]) / dz)
                        if bool(slope > max_slope):
                            max_slope = slope
                prev = level

        if bool(max_slope <= 1.0e-12):
            return self._uniform_target_depth(target_level, first_depth, last_depth)

        metric_total = first_depth * 0.0
        prev = -1
        for level in range(first, self.source_num_levels):
            if bool(valid[level]):
                if prev >= 0:
                    dz = depths[level] - depths[prev]
                    if bool(dz > 1.0e-12):
                        slope = torch.abs((temperature[level] - temperature[prev]) / dz)
                        metric_total = metric_total + dz * (
                            1.0 + self.temperature_gradient_weight * slope / max_slope
                        )
                prev = level

        if bool(metric_total <= 1.0e-12):
            return self._uniform_target_depth(target_level, first_depth, last_depth)

        target_metric = metric_total * fraction
        metric = first_depth * 0.0
        prev = -1
        for level in range(first, self.source_num_levels):
            if bool(valid[level]):
                if prev >= 0:
                    dz = depths[level] - depths[prev]
                    if bool(dz > 1.0e-12):
                        slope = torch.abs((temperature[level] - temperature[prev]) / dz)
                        step = dz * (
                            1.0 + self.temperature_gradient_weight * slope / max_slope
                        )
                        next_metric = metric + step
                        if bool(next_metric >= target_metric):
                            weight = (target_metric - metric) / step
                            return depths[prev] * (1.0 - weight) + depths[level] * weight
                        metric = next_metric
                prev = level

        return last_depth

    def _reduced_thickness_from_depths(
        self,
        target_depths: torch.Tensor,
    ) -> torch.Tensor:
        reduced_thickness = target_depths.new_zeros((self.target_num_levels,))
        if self.target_num_levels == 1:
            dz = 2.0 * target_depths[0]
            if bool(dz <= self.min_layer_thickness):
                dz = target_depths[0] * 0.0 + self.min_layer_thickness
            reduced_thickness[0] = dz
            return reduced_thickness

        interfaces = target_depths.new_zeros((self.target_num_levels + 1,))
        top_spacing = target_depths[1] - target_depths[0]
        top = target_depths[0] - 0.5 * top_spacing
        if bool(top < 0.0):
            top = target_depths[0] * 0.0
        interfaces[0] = top

        for level in range(1, self.target_num_levels):
            interfaces[level] = 0.5 * (
                target_depths[level - 1] + target_depths[level]
            )

        bottom_spacing = target_depths[self.target_num_levels - 1] - target_depths[
            self.target_num_levels - 2
        ]
        interfaces[self.target_num_levels] = (
            target_depths[self.target_num_levels - 1] + 0.5 * bottom_spacing
        )

        for level in range(self.target_num_levels):
            dz = interfaces[level + 1] - interfaces[level]
            if bool(dz <= self.min_layer_thickness):
                dz = target_depths[0] * 0.0 + self.min_layer_thickness
            reduced_thickness[level] = dz

        return reduced_thickness

    @torch.jit.export
    def reduced_temperature_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Interpolate source-grid temperature to the configured target grid.

        Source levels are valid only when layer thickness exceeds 0.1 m.
        """
        temperature = self._temperature_block(inputs)
        thickness = self._thickness_block(inputs)
        depths = self._source_mid_depths(thickness)
        valid = (
            self._valid_thickness_mask(thickness)
            & self._valid_temperature_mask(temperature)
        )

        nnodes = temperature.shape[0]
        reduced = temperature.new_zeros((nnodes, self.target_num_levels))

        for node in range(nnodes):
            first = -1
            last = -1
            for level in range(self.source_num_levels):
                if bool(valid[node, level]):
                    if first < 0:
                        first = level
                    last = level

            if first < 0:
                continue

            if first == last:
                for target_level in range(self.target_num_levels):
                    reduced[node, target_level] = temperature[node, first]
                continue

            first_depth = depths[node, first]
            last_depth = depths[node, last]

            for target_level in range(self.target_num_levels):
                if (
                    self.use_temperature_gradient_grid
                    and self.temperature_gradient_weight > 0.0
                ):
                    target_depth = self._temperature_gradient_target_depth(
                        target_level,
                        first,
                        last,
                        temperature[node],
                        depths[node],
                        valid[node],
                    )
                else:
                    target_depth = self._uniform_target_depth(
                        target_level, first_depth, last_depth
                    )

                lower = first
                upper = last
                for level in range(first, self.source_num_levels):
                    if bool(valid[node, level]):
                        if bool(depths[node, level] >= target_depth):
                            upper = level
                            break
                        lower = level

                if lower == upper:
                    reduced[node, target_level] = temperature[node, lower]
                else:
                    lower_depth = depths[node, lower]
                    upper_depth = depths[node, upper]
                    denom = upper_depth - lower_depth
                    if bool(torch.abs(denom) <= 1.0e-12):
                        reduced[node, target_level] = temperature[node, lower]
                    else:
                        weight = (target_depth - lower_depth) / denom
                        reduced[node, target_level] = (
                            temperature[node, lower] * (1.0 - weight)
                            + temperature[node, upper] * weight
                        )

        return reduced

    @torch.jit.export
    def reduced_thickness_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reduced-grid layer thickness used as learned FFNN geometry input."""
        temperature = self._temperature_block(inputs)
        thickness = self._thickness_block(inputs)
        depths = self._source_mid_depths(thickness)
        valid = (
            self._valid_thickness_mask(thickness)
            & self._valid_temperature_mask(temperature)
        )

        nnodes = temperature.shape[0]
        reduced = temperature.new_zeros((nnodes, self.target_num_levels))

        for node in range(nnodes):
            first = -1
            last = -1
            for level in range(self.source_num_levels):
                if bool(valid[node, level]):
                    if first < 0:
                        first = level
                    last = level

            if first < 0:
                continue

            if first == last:
                for target_level in range(self.target_num_levels):
                    reduced[node, target_level] = thickness[node, first]
                continue

            first_depth = depths[node, first]
            last_depth = depths[node, last]
            target_depths = temperature.new_zeros((self.target_num_levels,))

            for target_level in range(self.target_num_levels):
                if (
                    self.use_temperature_gradient_grid
                    and self.temperature_gradient_weight > 0.0
                ):
                    target_depths[target_level] = self._temperature_gradient_target_depth(
                        target_level,
                        first,
                        last,
                        temperature[node],
                        depths[node],
                        valid[node],
                    )
                else:
                    target_depths[target_level] = self._uniform_target_depth(
                        target_level, first_depth, last_depth
                    )

            reduced[node, :] = self._reduced_thickness_from_depths(target_depths)

        return reduced

    @torch.jit.export
    def reduced_profile_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reduced FFNN input: [temperature(target levels), thickness(target levels)]."""
        return torch.cat(
            (self.reduced_temperature_inputs(inputs), self.reduced_thickness_inputs(inputs)),
            dim=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict reduced-grid salinity from source-grid temperature and thickness."""
        return self.ffnn.predict(self.reduced_profile_inputs(inputs))

    def _jac_physical(self, inputs: torch.Tensor) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for i in range(self.target_num_levels):
            x_copy = inputs.detach().requires_grad_(True)
            y_phys = self.forward(x_copy)

            g = torch.zeros_like(y_phys)
            g[:, i] = 1.0

            grad_outputs: List[Optional[torch.Tensor]] = [g]
            grads = torch.autograd.grad([y_phys], [x_copy], grad_outputs=grad_outputs)
            grad = grads[0]
            if grad is None:
                raise RuntimeError("Jacobian: gradient is None")
            rows.append(grad)

        return torch.stack(rows, dim=1)

    @torch.jit.export
    def jac_physical(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobian entries requested by TorchBalanceVerticalEmulator.cc."""
        jac = self._jac_physical(inputs)
        jac_rows = jac.index_select(1, row_indices)
        gather_cols = col_indices.view(1, -1, 1).expand(jac.shape[0], -1, 1)
        jac_pairs = jac_rows.gather(2, gather_cols).squeeze(2)
        return jac_pairs * mask
