"""Residual and convergence helpers for transient flow/thermal runs.

The field operator in this module is deliberately independent of XLB and JAX.
It receives the array namespace to use (normally :mod:`numpy` or
``jax.numpy``), and therefore can be placed inside a caller's larger JIT
function without copying a device array to the host.  Only six scalar values
are returned from an evaluation:

``[velocity_residual, pressure_residual, temperature_residual,
  velocity_change_max, pressure_change_max, temperature_change_max]``.

The first three values are normalized L-infinity changes.  The final three
values retain the corresponding changes in physical units.  ``ConvergenceMonitor``
is a separate host-side scalar state machine that consumes those six values at
the requested sampling interval.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Iterable

import numpy as np


_RESIDUAL_KIND = "sample_change_linf_v1"
_VELOCITY_FLOOR_SI = 1.0e-6
_PRESSURE_FLOOR_PA = 1.0
_TEMPERATURE_FLOOR_K = 1.0
_VALUE_COUNT = 6


def _finite_scalar(value: Any, label: str) -> float:
    """Return a finite real scalar, with a useful public error."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_scalar(value: Any, label: str) -> float:
    result = _finite_scalar(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _bool_scalar(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a boolean")
    return bool(value)


def _masked_max(xp: Any, values: Any, mask: Any, zero: Any) -> Any:
    """Take a masked maximum without boolean indexing or empty reductions.

    ``where`` is important here: it keeps inactive cells out of the maximum,
    works with JAX tracers under ``jit``, and gives zero for an all-inactive
    mask.  The mesh validator guarantees a non-empty spatial shape.
    """

    return xp.max(xp.where(mask, values, zero))


def make_residual_operator(
    xp: Any,
    *,
    initial_temperature: float,
    velocity_scale: float,
    pressure_scale: float,
    flow_enabled: bool,
    thermal_enabled: bool,
) -> Callable[..., Any]:
    """Build a NumPy/JAX-compatible six-scalar residual evaluator.

    Parameters
    ----------
    xp:
        Array namespace, such as :mod:`numpy` or ``jax.numpy``.  The factory
        intentionally does not import JAX.
    initial_temperature, velocity_scale, pressure_scale:
        Finite physical constants.  Velocity and pressure scales must be
        positive; temperatures are expressed in kelvin.
    flow_enabled, thermal_enabled:
        Select which physics contributes metrics.  Disabled physics returns
        exact zero in its normalized and absolute slots.

    The returned ``evaluate`` accepts current and previous fields followed by
    ``fluid_mask`` and ``thermal_mask``.  Velocity fields have shape
    ``(3, *spatial_shape)``; density fields have shape ``(1, *spatial_shape)``;
    temperatures and masks have shape ``spatial_shape``.  All arithmetic is
    performed in FP64.  The pressure calculation converts density to FP64
    before subtracting its lattice reference value of one.
    """

    if not callable(getattr(xp, "asarray", None)):
        raise TypeError("xp must provide an asarray function")
    # Keep these as Python scalars in the closure.  They are small constants,
    # unlike the state arrays, and multiplying an FP64 input by them preserves
    # FP64 in NumPy and in a x64-enabled JAX context.
    initial_temperature_value = _finite_scalar(initial_temperature, "initial_temperature")
    velocity_scale_value = _positive_scalar(velocity_scale, "velocity_scale")
    pressure_scale_value = _positive_scalar(pressure_scale, "pressure_scale")
    flow_active = _bool_scalar(flow_enabled, "flow_enabled")
    thermal_active = _bool_scalar(thermal_enabled, "thermal_enabled")
    float64 = getattr(xp, "float64", np.float64)

    def _array(value: Any, label: str) -> Any:
        try:
            # Shape validation should not materialize a needless FP64 copy of
            # a disabled field.  Active-physics helpers cast their operands
            # immediately before arithmetic below.
            return xp.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a numeric array") from exc

    def _mask(value: Any, label: str) -> Any:
        try:
            return xp.asarray(value, dtype=bool)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be an array-like mask") from exc

    # Each helper owns its full-domain temporaries.  This keeps the reference
    # NumPy path from retaining all component fields at once while still
    # leaving one fused expression for a surrounding JAX ``jit``.
    def _velocity_metrics(current_velocity: Any, previous_velocity: Any, mask: Any, zero: Any) -> tuple[Any, Any]:
        """Return ``(absolute_change, normalized_change)`` for velocity."""

        # Accumulate one component at a time.  This matters for a large host
        # RAM run: a vector stack plus three component deltas would otherwise
        # remain live until the evaluator returns.
        delta_v_sq = xp.zeros(mask.shape, dtype=float64)
        for component in range(3):
            current_component = xp.asarray(current_velocity[component], dtype=float64)
            previous_component = xp.asarray(previous_velocity[component], dtype=float64)
            delta_component = (current_component - previous_component) * velocity_scale_value
            delta_v_sq += delta_component * delta_component
            del current_component, previous_component, delta_component
        velocity_change = xp.sqrt(xp.maximum(zero, _masked_max(xp, delta_v_sq, mask, zero)))
        del delta_v_sq

        current_speed_sq = xp.zeros(mask.shape, dtype=float64)
        for component in range(3):
            current_component = xp.asarray(current_velocity[component], dtype=float64)
            current_component = current_component * velocity_scale_value
            current_speed_sq += current_component * current_component
            del current_component
        current_speed = xp.sqrt(xp.maximum(zero, _masked_max(xp, current_speed_sq, mask, zero)))
        del current_speed_sq

        previous_speed_sq = xp.zeros(mask.shape, dtype=float64)
        for component in range(3):
            previous_component = xp.asarray(previous_velocity[component], dtype=float64)
            previous_component = previous_component * velocity_scale_value
            previous_speed_sq += previous_component * previous_component
            del previous_component
        previous_speed = xp.sqrt(xp.maximum(zero, _masked_max(xp, previous_speed_sq, mask, zero)))
        del previous_speed_sq
        denominator = xp.maximum(xp.maximum(current_speed, previous_speed), _VELOCITY_FLOOR_SI)
        return velocity_change, velocity_change / denominator

    def _pressure_metrics(current_density: Any, previous_density: Any, mask: Any, zero: Any) -> tuple[Any, Any]:
        """Return ``(absolute_change, normalized_change)`` for pressure."""

        # Density has already been converted to FP64 by the caller before the
        # lattice reference value is subtracted.
        current_pressure = (xp.asarray(current_density[0], dtype=float64) - 1.0) * pressure_scale_value
        previous_pressure = (xp.asarray(previous_density[0], dtype=float64) - 1.0) * pressure_scale_value
        pressure_change = _masked_max(
            xp,
            xp.abs(current_pressure - previous_pressure),
            mask,
            zero,
        )
        current_scale = _masked_max(xp, xp.abs(current_pressure), mask, zero)
        previous_scale = _masked_max(xp, xp.abs(previous_pressure), mask, zero)
        denominator = xp.maximum(xp.maximum(current_scale, previous_scale), _PRESSURE_FLOOR_PA)
        return pressure_change, pressure_change / denominator

    def _temperature_metrics(current_temp: Any, previous_temp: Any, mask: Any, zero: Any) -> tuple[Any, Any]:
        """Return ``(absolute_change, normalized_change)`` for temperature."""
        current_temp = xp.asarray(current_temp, dtype=float64)
        previous_temp = xp.asarray(previous_temp, dtype=float64)
        temperature_change = _masked_max(
            xp,
            xp.abs(current_temp - previous_temp),
            mask,
            zero,
        )
        current_scale = _masked_max(
            xp,
            xp.abs(current_temp - initial_temperature_value),
            mask,
            zero,
        )
        previous_scale = _masked_max(
            xp,
            xp.abs(previous_temp - initial_temperature_value),
            mask,
            zero,
        )
        denominator = xp.maximum(xp.maximum(current_scale, previous_scale), _TEMPERATURE_FLOOR_K)
        return temperature_change, temperature_change / denominator

    def evaluate(
        current_velocity_lbm: Any,
        current_rho: Any,
        current_temperature: Any,
        previous_velocity_lbm: Any,
        previous_rho: Any,
        previous_temperature: Any,
        fluid_mask: Any,
        thermal_mask: Any,
    ) -> Any:
        """Evaluate normalized and absolute changes for one sample pair."""

        current_velocity = _array(current_velocity_lbm, "current_velocity_lbm")
        previous_velocity = _array(previous_velocity_lbm, "previous_velocity_lbm")
        current_density = _array(current_rho, "current_rho")
        previous_density = _array(previous_rho, "previous_rho")
        current_temp = _array(current_temperature, "current_temperature")
        previous_temp = _array(previous_temperature, "previous_temperature")
        fluid = _mask(fluid_mask, "fluid_mask")
        thermal = _mask(thermal_mask, "thermal_mask")

        if current_velocity.ndim < 2 or current_velocity.shape[0] != 3:
            raise ValueError(
                "current_velocity_lbm must have shape (3, *spatial_shape), "
                f"got {current_velocity.shape}"
            )
        spatial_shape = tuple(current_velocity.shape[1:])
        expected_velocity = (3, *spatial_shape)
        expected_density = (1, *spatial_shape)
        if previous_velocity.shape != expected_velocity:
            raise ValueError(
                "previous_velocity_lbm must have shape "
                f"{expected_velocity}, got {previous_velocity.shape}"
            )
        if current_density.shape != expected_density:
            raise ValueError(f"current_rho must have shape {expected_density}, got {current_density.shape}")
        if previous_density.shape != expected_density:
            raise ValueError(f"previous_rho must have shape {expected_density}, got {previous_density.shape}")
        if current_temp.shape != spatial_shape:
            raise ValueError(f"current_temperature must have shape {spatial_shape}, got {current_temp.shape}")
        if previous_temp.shape != spatial_shape:
            raise ValueError(f"previous_temperature must have shape {spatial_shape}, got {previous_temp.shape}")
        if fluid.shape != spatial_shape:
            raise ValueError(f"fluid_mask must have shape {spatial_shape}, got {fluid.shape}")
        if thermal.shape != spatial_shape:
            raise ValueError(f"thermal_mask must have shape {spatial_shape}, got {thermal.shape}")

        zero = xp.asarray(0.0, dtype=float64)
        outputs: list[Any] = []

        if flow_active:
            # Scaling before taking the norm keeps all velocity comparisons in
            # SI units.  Component-wise expressions avoid constructing a
            # (3, nx, ny, nz) delta or speed stack for the full domain.
            velocity_change, velocity_residual = _velocity_metrics(
                current_velocity,
                previous_velocity,
                fluid,
                zero,
            )
            outputs.extend((velocity_residual,))
        else:
            outputs.extend((zero,))

        if flow_active:
            # Explicit FP64 density conversion happens before rho - 1.  This
            # matters when callers provide FP32 lattice fields with a small
            # reduced-pressure signal.
            pressure_change, pressure_residual = _pressure_metrics(
                current_density,
                previous_density,
                fluid,
                zero,
            )
            outputs.extend((pressure_residual,))
        else:
            outputs.extend((zero,))

        if thermal_active:
            temperature_change, temperature_residual = _temperature_metrics(
                current_temp,
                previous_temp,
                thermal,
                zero,
            )
            outputs.extend((temperature_residual,))
        else:
            outputs.extend((zero,))

        # Absolute slots are emitted after the three normalized slots.  Keep
        # their calculation adjacent to each normalized metric to make the
        # expression readable, while retaining the public vector contract.
        # ``outputs`` currently contains normalized [v, p, T]; recompute only
        # scalar references for absolute slots without any stacked field.
        if flow_active:
            outputs.extend((velocity_change, pressure_change))
        else:
            outputs.extend((zero, zero))
        if thermal_active:
            outputs.extend((temperature_change,))
        else:
            outputs.extend((zero,))

        return xp.stack(outputs).astype(float64)

    return evaluate


class ConvergenceMonitor:
    """Scalar convergence state machine for sampled residual vectors.

    ``update(step, None)`` establishes a fresh step baseline.  A vector passed
    after construction without a step baseline establishes the initial step;
    after an explicit ``None`` baseline, the first vector is compared at once
    when its step gap equals ``interval``.  A sample contributes to the
    consecutive pass count only when its step gap equals ``interval`` exactly
    and every enabled normalized residual is at or below ``tolerance``.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-5,
        consecutive_samples: int = 5,
        interval: int = 20,
        dt: float = 0.001,
        flow_enabled: bool = True,
        thermal_enabled: bool = True,
    ) -> None:
        self.tolerance = _positive_scalar(tolerance, "tolerance")
        if isinstance(consecutive_samples, (bool, np.bool_)) or not isinstance(consecutive_samples, numbers.Integral):
            raise ValueError("consecutive_samples must be an integer")
        self.required_samples = int(consecutive_samples)
        if self.required_samples < 1:
            raise ValueError("consecutive_samples must be at least 1")
        if self.required_samples > 100:
            raise ValueError("consecutive_samples must be at most 100")
        if isinstance(interval, (bool, np.bool_)) or not isinstance(interval, numbers.Integral):
            raise ValueError("interval must be an integer")
        self.interval = int(interval)
        if self.interval < 1:
            raise ValueError("interval must be at least 1")
        self.dt = _positive_scalar(dt, "dt")
        self.flow_enabled = _bool_scalar(flow_enabled, "flow_enabled")
        self.thermal_enabled = _bool_scalar(thermal_enabled, "thermal_enabled")

        self._baseline = False
        self._previous_step: int | None = None
        self._previous_values: tuple[float, ...] | None = None
        self._consecutive = 0
        self.satisfied = False

    @property
    def consecutive_samples(self) -> int:
        """Number of consecutive eligible samples currently passing."""

        return self._consecutive

    def _step_value(self, step: Any) -> int:
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, numbers.Integral):
            raise ValueError("step must be a non-negative integer")
        step_value = int(step)
        if step_value < 0:
            raise ValueError("step must be a non-negative integer")
        return step_value

    def _values(self, values: Iterable[Any]) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError("values must contain six residual metrics")
        try:
            raw = list(values)
        except (TypeError, ValueError) as exc:
            raise ValueError("values must contain six residual metrics") from exc
        if len(raw) != _VALUE_COUNT:
            raise ValueError("values must contain six residual metrics")
        try:
            converted = tuple(float(item) for item in raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("values must contain six numeric residual metrics") from exc

        active_indices: tuple[int, ...] = tuple(
            index
            for index in ((0, 1, 3, 4) if self.flow_enabled else ())
            + ((2, 5) if self.thermal_enabled else ())
        )
        if any(not math.isfinite(converted[index]) for index in active_indices):
            raise ValueError("active residual metrics must be finite")
        return converted

    def _row(
        self,
        step: int,
        values: tuple[float, ...] | None,
        *,
        sample_steps: int | None,
        eligible: bool,
    ) -> dict[str, Any]:
        if values is None:
            velocity_residual = pressure_residual = temperature_residual = None
            velocity_change = pressure_change = temperature_change = None
            residual = None
        else:
            velocity_residual = values[0] if self.flow_enabled else None
            pressure_residual = values[1] if self.flow_enabled else None
            temperature_residual = values[2] if self.thermal_enabled else None
            velocity_change = values[3] if self.flow_enabled else None
            pressure_change = values[4] if self.flow_enabled else None
            temperature_change = values[5] if self.thermal_enabled else None
            active_normalized = [
                value
                for value in (velocity_residual, pressure_residual, temperature_residual)
                if value is not None
            ]
            residual = max(active_normalized) if active_normalized else None
        return {
            "residual_kind": _RESIDUAL_KIND,
            "time": float(step) * self.dt,
            "sample_steps": sample_steps,
            "sample_time": None if sample_steps is None else float(sample_steps) * self.dt,
            "residual": residual,
            "velocity_residual": velocity_residual,
            "pressure_residual": pressure_residual,
            "temperature_residual": temperature_residual,
            "velocity_change_max": velocity_change,
            "pressure_change_max": pressure_change,
            "temperature_change_max": temperature_change,
            "monitor_tolerance": self.tolerance,
            "monitor_required_samples": self.required_samples,
            "monitor_consecutive_samples": self._consecutive,
            "monitor_satisfied": bool(self.satisfied),
            "monitor_eligible": bool(eligible),
        }

    def update(self, step: Any, values: Iterable[Any] | None = None) -> dict[str, Any]:
        """Consume one sample and return a serializable progress row.

        ``values=None`` explicitly starts a new baseline.  Non-finite values
        in enabled physics slots are rejected before monitor state changes;
        disabled slots are ignored and reported as ``None``.
        """

        step_value = self._step_value(step)
        if values is None:
            self._baseline = True
            self._previous_step = step_value
            self._previous_values = None
            self._consecutive = 0
            self.satisfied = False
            return self._row(step_value, None, sample_steps=None, eligible=False)

        converted = self._values(values)
        # The residual vector already represents the change from the previous
        # physical state.  A ``None`` update therefore only establishes the
        # step baseline; the first vector after that baseline is eligible when
        # its step gap matches ``interval``.
        if self._previous_step is None:
            self._baseline = True
            self._previous_step = step_value
            self._previous_values = converted
            self._consecutive = 0
            self.satisfied = False
            return self._row(step_value, None, sample_steps=None, eligible=False)

        previous_step = self._previous_step
        sample_steps = step_value - previous_step
        physics_active = self.flow_enabled or self.thermal_enabled
        eligible = physics_active and sample_steps == self.interval
        if eligible:
            active_normalized = []
            if self.flow_enabled:
                active_normalized.extend((converted[0], converted[1]))
            if self.thermal_enabled:
                active_normalized.append(converted[2])
            passes = all(value <= self.tolerance for value in active_normalized)
            if passes:
                self._consecutive += 1
            else:
                self._consecutive = 0
        else:
            # An irregularly spaced sample cannot establish a consecutive
            # interval guarantee, so it conservatively restarts the streak.
            self._consecutive = 0
        self.satisfied = bool(physics_active and self._consecutive >= self.required_samples)
        self._previous_step = step_value
        self._previous_values = converted
        self._baseline = True
        return self._row(step_value, converted, sample_steps=sample_steps, eligible=eligible)

    def diagnostics(self) -> dict[str, Any]:
        """Return stable monitor and residual-definition metadata."""

        return {
            "definition": _RESIDUAL_KIND,
            "normalized_scales": {
                "velocity": "max(current_speed, previous_speed, 1e-6 m/s)",
                "pressure": "max(abs(current_pressure), abs(previous_pressure), 1 Pa)",
                "temperature": "max(abs(current_temperature-initial), abs(previous_temperature-initial), 1 K)",
            },
            "floors": {
                "velocity_SI": _VELOCITY_FLOOR_SI,
                "pressure_Pa": _PRESSURE_FLOOR_PA,
                "temperature_K": _TEMPERATURE_FLOOR_K,
            },
            "tolerance": self.tolerance,
            "required_samples": self.required_samples,
            "consecutive_samples": self._consecutive,
            "satisfied": bool(self.satisfied),
            "interval": self.interval,
            "dt": self.dt,
            "flow_enabled": self.flow_enabled,
            "thermal_enabled": self.thermal_enabled,
        }


__all__ = ["ConvergenceMonitor", "make_residual_operator"]
