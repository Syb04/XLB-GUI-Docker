"""Small JAX field factories used by the GPU solver path.

The workbench keeps project parsing and validation on the host.  These
factories turn the already validated, immutable mesh/material descriptors into
pure array functions that can be composed with a caller's ``jax.jit`` and
device context.  Nothing in an evaluator converts a tracer back to NumPy or
uses boolean array indexing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


_PROPERTY_DEFAULTS = {
    "density": 1000.0,
    "viscosity": 1.0e-3,
    "heat_capacity": 4182.0,
    "conductivity": 0.6,
}
_PROPERTY_NAMES = tuple(_PROPERTY_DEFAULTS)


@dataclass(frozen=True)
class _PropertySpec:
    """Host-validated description of one material property."""

    kind: str
    constant: float | None = None
    temperatures: np.ndarray | None = None
    values: np.ndarray | None = None


def _property_spec(value: Any, default: float, label: str) -> _PropertySpec:
    """Normalize one property using the solver's defaults and rules.

    This intentionally mirrors ``solver._kind_value``: scalar values and
    ``constant`` mappings are accepted, table temperatures must be finite and
    strictly increasing, and values themselves only need to be finite here.
    Positivity is enforced by the project schema before a solver run.
    """

    if value is None:
        return _PropertySpec(kind="constant", constant=float(default))
    if isinstance(value, Mapping):
        kind = str(value.get("kind", "constant")).lower()
        raw = value.get("value", default) if kind == "constant" else value.get("points")
    else:
        kind = "constant"
        raw = value

    if kind == "constant":
        try:
            constant = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} constant must be numeric") from exc
        if not math.isfinite(constant):
            raise ValueError(f"{label} constant must be finite")
        return _PropertySpec(kind="constant", constant=constant)

    if kind != "table":
        raise ValueError(f"unsupported {label} property kind {kind!r}")
    try:
        points = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} table must contain at least two [temperature,value] points") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(f"{label} table must contain at least two [temperature,value] points")
    if not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValueError(f"{label} table temperatures must be finite and strictly increasing")
    # Copy once, on the host, so a caller cannot mutate a captured table while
    # a jitted evaluator is running.
    return _PropertySpec(
        kind="table",
        temperatures=np.array(points[:, 0], dtype=float, copy=True),
        values=np.array(points[:, 1], dtype=float, copy=True),
    )


def _material_specs(materials: Sequence[Mapping[str, Any]]) -> tuple[tuple[_PropertySpec, ...], ...]:
    if not isinstance(materials, Sequence) or isinstance(materials, (str, bytes)) or not materials:
        raise ValueError("materials must contain at least one material object")
    normalized: list[tuple[_PropertySpec, ...]] = []
    for index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise ValueError(f"materials[{index}] must be an object")
        normalized.append(
            tuple(
                _property_spec(material.get(name), default, f"material[{index}].{name}")
                for name, default in _PROPERTY_DEFAULTS.items()
            )
        )
    return tuple(normalized)


def make_material_operator(
    materials: Sequence[Mapping[str, Any]],
    material_index_np: Any,
    active_np: Any,
    *,
    jax: Any,
    jnp: Any,
):
    """Build a pure JAX material-property evaluator.

    ``material_index_np`` and ``active_np`` are host arrays describing the
    fixed mesh.  ``evaluate(temperature)`` returns
    ``(density, viscosity, heat_capacity, conductivity, clamped_flag)`` with
    the first four arrays shaped like ``temperature`` and the final value a
    device boolean scalar.  Inactive cells retain the neutral ``1`` defaults
    used by :func:`workbench.solver._material_fields`.
    """

    # Validate every Python/NumPy input before creating a captured JAX
    # function.  ``jax`` is part of the public injection point so callers can
    # select their device context; array creation intentionally goes through
    # the matching ``jnp`` module.
    del jax  # the factory only needs the injected namespace for composition
    specs = _material_specs(materials)
    try:
        material_index_host = np.asarray(material_index_np)
        active_host = np.asarray(active_np)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("material_index_np and active_np must be array-like") from exc
    if material_index_host.shape != active_host.shape:
        raise ValueError("material_index_np and active_np must have the same shape")
    if not np.issubdtype(material_index_host.dtype, np.integer):
        raise ValueError("material_index_np must contain integer material indices")
    if material_index_host.ndim == 0:
        # Scalar fields are useful in small operator tests and remain fully
        # compatible with JAX's zero-dimensional arrays.
        material_index_host = material_index_host.reshape(())
    material_index_host = np.asarray(material_index_host, dtype=np.int32)
    active_host = np.asarray(active_host, dtype=bool)
    shape = material_index_host.shape
    material_index = jnp.asarray(material_index_host)
    active = jnp.asarray(active_host, dtype=bool)

    # Convert immutable property tables once.  The evaluator only sees JAX
    # arrays and Python constants; no host conversion occurs under jit.
    jax_specs: list[tuple[tuple[int, float, Any, Any], ...]] = []
    for material_specs in specs:
        properties: list[tuple[int, float, Any, Any]] = []
        for spec in material_specs:
            if spec.kind == "constant":
                properties.append((0, float(spec.constant), None, None))
            else:
                properties.append(
                    (
                        1,
                        0.0,
                        jnp.asarray(spec.temperatures),
                        jnp.asarray(spec.values),
                    )
                )
        jax_specs.append(tuple(properties))
    frozen_specs = tuple(jax_specs)

    def evaluate(temperature: Any):
        """Evaluate all material properties for one temperature field."""

        temperature = jnp.asarray(temperature, dtype=float)
        if temperature.shape != shape:
            raise ValueError(f"temperature must have shape {shape}, got {temperature.shape}")
        density = jnp.ones_like(temperature)
        viscosity = jnp.ones_like(temperature)
        heat_capacity = jnp.ones_like(temperature)
        conductivity = jnp.ones_like(temperature)
        clamped_flag = jnp.asarray(False, dtype=bool)
        outputs = [density, viscosity, heat_capacity, conductivity]

        for material_number, material_properties in enumerate(frozen_specs):
            selected = jnp.logical_and(active, material_index == material_number)
            for property_number, (kind, constant, table_temperature, table_value) in enumerate(material_properties):
                if kind == 0:
                    candidate = jnp.full_like(temperature, constant)
                else:
                    # jnp.interp clamps to the first/last endpoint exactly as
                    # np.interp used by solver._evaluate_property.
                    candidate = jnp.interp(temperature, table_temperature, table_value)
                    low = table_temperature[0]
                    high = table_temperature[-1]
                    outside = jnp.logical_and(
                        selected,
                        jnp.logical_or(temperature < low, temperature > high),
                    )
                    clamped_flag = jnp.logical_or(clamped_flag, jnp.any(outside))
                outputs[property_number] = jnp.where(selected, candidate, outputs[property_number])

        return (*outputs, clamped_flag)

    return evaluate


def make_les_operator(
    fluid_np: Any,
    dx: float,
    coefficient: float,
    *,
    jax: Any,
    jnp: Any,
):
    """Build a pure JAX Smagorinsky eddy-viscosity evaluator.

    The returned function accepts ``velocity_phys`` with shape ``(3, nx, ny,
    nz)`` and returns a non-negative kinematic eddy-viscosity field.  Its
    valid-neighbour and one-sided boundary rules match
    :func:`workbench.solver._velocity_gradient` exactly.  Symmetric strain
    components are accumulated directly instead of materializing a 3×3×mesh
    gradient array, allowing a caller's larger ``jax.jit`` to fuse the rolls
    and invariant calculation.
    """

    del jax
    try:
        fluid_host = np.asarray(fluid_np)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fluid_np must be array-like") from exc
    if fluid_host.ndim != 3:
        raise ValueError("fluid_np must be a three-dimensional mask")
    fluid_host = np.asarray(fluid_host, dtype=bool)
    try:
        dx_value = float(dx)
        coefficient_value = float(coefficient)
    except (TypeError, ValueError) as exc:
        raise ValueError("dx and coefficient must be numeric") from exc
    if not math.isfinite(dx_value) or dx_value <= 0.0:
        raise ValueError("dx must be positive and finite")
    if not math.isfinite(coefficient_value) or coefficient_value < 0.0:
        raise ValueError("coefficient must be non-negative and finite")

    shape = fluid_host.shape
    fluid = jnp.asarray(fluid_host, dtype=bool)
    # Static boundary masks avoid tracer-dependent indexing while reproducing
    # the explicit edge invalidation in _velocity_gradient.
    edge_masks: list[tuple[Any, Any]] = []
    for axis, extent in enumerate(shape):
        coordinates = jnp.arange(extent)
        reshape = [1, 1, 1]
        reshape[axis] = extent
        low = (coordinates == 0).reshape(tuple(reshape))
        high = (coordinates == extent - 1).reshape(tuple(reshape))
        edge_masks.append((low, high))

    def derivative(field: Any, axis: int):
        previous = jnp.roll(field, 1, axis=axis)
        following = jnp.roll(field, -1, axis=axis)
        previous_valid = jnp.logical_and(jnp.roll(fluid, 1, axis=axis), ~edge_masks[axis][0])
        following_valid = jnp.logical_and(jnp.roll(fluid, -1, axis=axis), ~edge_masks[axis][1])
        central = (following - previous) / (2.0 * dx_value)
        forward = (following - field) / dx_value
        backward = (field - previous) / dx_value
        derivative_value = jnp.where(previous_valid & following_valid, central, 0.0)
        derivative_value = jnp.where(~previous_valid & following_valid, forward, derivative_value)
        derivative_value = jnp.where(previous_valid & ~following_valid, backward, derivative_value)
        return jnp.where(fluid, derivative_value, 0.0)

    def evaluate(velocity_phys: Any):
        """Return the local kinematic Smagorinsky eddy viscosity."""

        velocity = jnp.asarray(velocity_phys, dtype=float)
        expected = (3, *shape)
        if velocity.shape != expected:
            raise ValueError(f"velocity_phys must have shape {expected}, got {velocity.shape}")
        u0, u1, u2 = velocity[0], velocity[1], velocity[2]

        # Nine scalar derivative fields are consumed directly by the fused
        # invariant expression; no (3, 3, nx, ny, nz) temporary is stacked.
        g00 = derivative(u0, 0)
        g01 = derivative(u0, 1)
        g02 = derivative(u0, 2)
        g10 = derivative(u1, 0)
        g11 = derivative(u1, 1)
        g12 = derivative(u1, 2)
        g20 = derivative(u2, 0)
        g21 = derivative(u2, 1)
        g22 = derivative(u2, 2)
        s01 = 0.5 * (g01 + g10)
        s02 = 0.5 * (g02 + g20)
        s12 = 0.5 * (g12 + g21)
        strain_squared_twice = 2.0 * (
            g00 * g00
            + g11 * g11
            + g22 * g22
            + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12)
        )
        magnitude = jnp.sqrt(jnp.maximum(0.0, strain_squared_twice))
        eddy = (coefficient_value * dx_value) ** 2 * magnitude
        return jnp.where(fluid, jnp.maximum(eddy, 0.0), 0.0)

    return evaluate


__all__ = ["make_material_operator", "make_les_operator"]
