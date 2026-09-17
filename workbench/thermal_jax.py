"""JAX finite-volume thermal operators for the workbench solver.

The public solver keeps a NumPy implementation of the thermal update for a
portable reference path.  This module contains the same update expressed only
in array operations supported by :mod:`jax.numpy`.  Geometry is converted to
device arrays once, when :func:`make_thermal_operators` is called.  The two
returned callables can consequently be included in a larger ``jax.jit``
function without copying fields back to the host or inspecting tracer values.

The implementation deliberately receives ``jax`` and ``jax.numpy`` from the
caller.  XLB selects a device before constructing its operators, and this
keeps thermal code usable with either a CPU-only or a CUDA-enabled JAX
installation without importing JAX as a mandatory dependency for the package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _thermal_object(value: Any) -> Mapping[str, Any]:
    """Return a thermal descriptor from either a descriptor or BC object."""

    if not isinstance(value, Mapping):
        return {}
    nested = value.get("thermal")
    if isinstance(nested, Mapping):
        return nested
    return value


def _thermal_kind(value: Any) -> str:
    spec = _thermal_object(value)
    return str(spec.get("type", "adiabatic")).lower().replace("_", "-")


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _first_number(spec: Mapping[str, Any], keys: tuple[str, ...], label: str) -> float:
    for key in keys:
        if key in spec:
            return _number(spec[key], label)
    raise ValueError(f"{label} requires numeric value")


def _normalise_thermal(value: Any, label: str) -> tuple[str, float, float, float]:
    """Normalize one boundary to ``(kind, value, h, ambient)``.

    Keeping the normalized values as Python scalars means boundary descriptors
    remain static closure data while temperature, material properties, and
    velocity stay dynamic JAX arguments.
    """

    spec = _thermal_object(value)
    kind = _thermal_kind(spec)
    if kind in {"adiabatic", "insulated", "none"}:
        return "adiabatic", 0.0, 0.0, 0.0
    if kind in {"temperature", "dirichlet"}:
        return "temperature", _first_number(spec, ("value", "temperature"), f"{label} temperature"), 0.0, 0.0
    if kind in {"heat-flux", "flux", "neumann"}:
        return "heat-flux", _first_number(spec, ("value", "heat_flux"), f"{label} heat flux"), 0.0, 0.0
    if kind in {"convection", "convective"}:
        coefficient = _first_number(spec, ("h", "coefficient"), f"{label} convection coefficient")
        ambient = _first_number(spec, ("ambient_temperature", "ambient"), f"{label} ambient_temperature")
        if coefficient < 0.0:
            raise ValueError(f"{label} convection coefficient must be non-negative")
        return "convection", 0.0, coefficient, ambient
    raise ValueError(f"unsupported thermal boundary type {kind!r} on {label}")


def _static_mask(value: Any, shape: tuple[int, int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return array.copy()


def _plane(shape: tuple[int, int, int], axis: int, index: int) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    selector = [slice(None)] * 3
    selector[axis] = index
    result[tuple(selector)] = True
    return result


def _static_link(value: Any, shape: tuple[int, int, int], label: str, fluid: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    expected = (6, *shape)
    if array.shape == (*shape, 6):
        array = np.moveaxis(array, -1, 0)
    if array.shape != expected:
        raise ValueError(f"{label} must have shape {expected}")
    # Boundary links are fluid-cell contacts by contract.  Applying this
    # mask here also protects callers constructing links by hand from an
    # accidental source in a conductive solid.
    return array.copy() & fluid[None, ...]


def _contact_count(links: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.sum(links & active[None, ...], axis=0, dtype=np.int32).astype(np.float64, copy=False)


def _cad_contacts(fluid: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fluid-wall and active-inactive CAD contact counts.

    The first count mirrors the reference step's legacy CAD fallback (only
    fluid cells receive a wall exchange).  The second mirrors its stability
    check, which counts every active cell adjacent to an inactive cell.  Box
    borders are removed from both counts because they have their own face
    descriptors.
    """

    fluid_wall = np.zeros(fluid.shape, dtype=np.int32)
    active_boundary = np.zeros(fluid.shape, dtype=np.int32)
    for axis in range(3):
        for shift in (1, -1):
            neighbour_active = np.roll(active, shift, axis=axis)
            neighbour_fluid = np.roll(fluid, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            edge_t = tuple(edge)
            # A rolled value at a box edge is not a real neighbour.
            neighbour_active[edge_t] = True
            neighbour_fluid[edge_t] = True
            internal = np.ones(fluid.shape, dtype=bool)
            internal[edge_t] = False
            fluid_wall += (fluid & ~neighbour_active & internal).astype(np.int32)
            active_boundary += (active & ~neighbour_active & internal).astype(np.int32)
    # ``neighbour_fluid`` is intentionally constructed above for parity with
    # the reference geometry logic and to make the distinction explicit.  A
    # CAD fallback sees inactive solid cells through ``neighbour_active``.
    del neighbour_fluid
    return fluid_wall, active_boundary


def _box_pairs(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return masks for real left/right active neighbours per axis."""

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for axis in range(3):
        low = np.roll(active, 1, axis=axis)
        high = np.roll(active, -1, axis=axis)
        low[_plane(active.shape, axis, 0)] = False
        high[_plane(active.shape, axis, -1)] = False
        left.append(active & low)
        right.append(active & high)
    return np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)


def make_thermal_operators(
    fluid_np: Any,
    thermal_np: Any,
    dx: float,
    dt: float,
    box_thermal: Mapping[str, Any] | None,
    cad_thermal: Mapping[str, Any] | None,
    face_masks_np: Mapping[str, Any] | None,
    cad_link_actions_np: Iterable[tuple[Any, Mapping[str, Any]]] | None,
    *,
    jax: Any,
    jnp: Any,
) -> tuple[Any, Any]:
    """Build JAX thermal step and stability callables.

    Parameters are intentionally split into NumPy geometry and dynamic field
    inputs.  Call this factory inside the caller's ``jax.default_device``
    context.  The returned ``step`` has signature
    ``(temperature, velocity_phys, density, heat_capacity, conductivity)``;
    ``stability`` accepts the last three fields and returns device scalars
    ``(fourier_max, robin_max, loss_max)``.
    """

    # ``jax`` is part of the explicit API so callers can select the device;
    # touching it here also gives a useful error for accidental omission.
    if jax is None or jnp is None:
        raise TypeError("make_thermal_operators requires jax and jnp")
    del jax  # JAX arrays are created through the supplied jnp namespace.

    fluid = np.asarray(fluid_np, dtype=bool)
    thermal = np.asarray(thermal_np, dtype=bool)
    if fluid.ndim != 3 or not fluid.any():
        raise ValueError("fluid_np must be a non-empty 3-D mask")
    if thermal.shape != fluid.shape:
        raise ValueError("thermal_np must have the same shape as fluid_np")
    if np.any(fluid & ~thermal):
        raise ValueError("thermal_np must contain every fluid cell")
    shape = tuple(int(v) for v in fluid.shape)
    dx_value = _number(dx, "dx")
    dt_value = _number(dt, "dt")
    if dx_value <= 0.0 or dt_value <= 0.0:
        raise ValueError("dx and dt must be positive")

    # All six masks are materialized.  A missing entry means that the caller
    # did not request a box-face source, matching the reference loop over the
    # supplied face-mask mapping.  Normal solver calls provide all six masks.
    raw_face_masks = face_masks_np or {}
    if not isinstance(raw_face_masks, Mapping):
        raise ValueError("face_masks_np must be a mapping")
    face_masks: dict[str, np.ndarray] = {}
    for face in _FACES:
        raw = raw_face_masks.get(face)
        if raw is None:
            raw = np.zeros(shape, dtype=bool)
        face_masks[face] = _static_mask(raw, shape, f"face_masks_np[{face!r}]")

    raw_box = box_thermal or {}
    if not isinstance(raw_box, Mapping):
        raise ValueError("box_thermal must be a mapping")
    box_specs = tuple(
        _normalise_thermal(raw_box.get(face, {}), face)
        for face in _FACES
    )
    cad_spec = None if cad_thermal is None else _normalise_thermal(cad_thermal, "cad")

    # Materialize and normalize local actions now.  Their masks and contact
    # counts are geometry-only constants, while only the fields passed to the
    # returned functions remain dynamic.
    local_actions: list[tuple[Any, np.ndarray, tuple[str, float, float, float]]] = []
    raw_actions = () if cad_link_actions_np is None else tuple(cad_link_actions_np)
    for index, item in enumerate(raw_actions):
        try:
            links_raw, thermal_raw = item
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cad_link_actions_np[{index}] must be a (links, thermal) pair") from exc
        links = _static_link(links_raw, shape, f"CAD thermal links {index}", fluid)
        local_actions.append(
            (jnp.asarray(links), _contact_count(links, thermal), _normalise_thermal(thermal_raw, f"CAD patch {index}"))
        )

    # Static geometry fields live on the selected target device because the
    # factory is called under ``jax.default_device`` by the solver.
    fluid_j = jnp.asarray(fluid)
    active_j = jnp.asarray(thermal)
    face_masks_j = {face: jnp.asarray(mask) for face, mask in face_masks.items()}
    pair_left_np, pair_right_np = _box_pairs(thermal)
    pair_left_j = jnp.asarray(pair_left_np)
    pair_right_j = jnp.asarray(pair_right_np)
    plane_min_j = tuple(jnp.asarray(_plane(shape, axis, 0)) for axis in range(3))
    plane_max_j = tuple(jnp.asarray(_plane(shape, axis, -1)) for axis in range(3))

    cad_wall_np, cad_active_boundary_np = _cad_contacts(fluid, thermal)
    cad_wall_j = jnp.asarray(cad_wall_np)
    cad_active_boundary_j = jnp.asarray(cad_active_boundary_np)

    # Precompute masks used by the upwind ghost construction.  These are
    # intentionally based on fluid neighbors, exactly as in solver.py.
    previous_valid_j: list[Any] = []
    following_valid_j: list[Any] = []
    for axis in range(3):
        previous = np.roll(fluid, 1, axis=axis)
        following = np.roll(fluid, -1, axis=axis)
        previous[_plane(shape, axis, 0)] = False
        following[_plane(shape, axis, -1)] = False
        previous_valid_j.append(jnp.asarray(previous))
        following_valid_j.append(jnp.asarray(following))

    # Keep static arrays alongside a normalized tuple for each local action.
    local_static = tuple(
        (links_j, jnp.asarray(contacts), spec)
        for links_j, contacts, spec in local_actions
    )

    def _safe_rho_cp(density: Any, heat_capacity: Any) -> Any:
        rho_cp = density * heat_capacity
        # Solver-side validation catches bad active material values before the
        # JAX path.  The neutral denominator keeps inactive cells from
        # creating NaNs in masked arithmetic when a caller supplies zero
        # placeholders there.
        return jnp.where(rho_cp != 0.0, rho_cp, jnp.asarray(1.0, dtype=rho_cp.dtype))

    def _h_eff(conductivity: Any, coefficient: float) -> Any:
        coefficient_safe = max(coefficient, 1.0e-30)
        return 1.0 / jnp.maximum(
            1.0 / coefficient_safe + dx_value / (2.0 * conductivity),
            1.0e-30,
        )

    def _apply_wall_source(
        out: Any,
        temperature: Any,
        rho_cp: Any,
        conductivity: Any,
        contacts: Any,
        spec: tuple[str, float, float, float],
    ) -> Any:
        kind, value, coefficient, ambient = spec
        cells = contacts > 0.0
        if kind == "temperature":
            increment = contacts * 2.0 * dt_value * conductivity * (value - temperature) / (
                dx_value * dx_value * rho_cp
            )
        elif kind == "heat-flux":
            increment = contacts * dt_value * value / (dx_value * rho_cp)
        elif kind == "convection":
            h_eff = _h_eff(conductivity, coefficient)
            increment = contacts * dt_value * h_eff * (ambient - temperature) / (dx_value * rho_cp)
        else:
            return out
        return out + jnp.where(cells, increment, 0.0)

    def step(temperature: Any, velocity_phys: Any, density: Any, heat_capacity: Any, conductivity: Any) -> Any:
        """Advance one explicit thermal finite-volume step on the device."""

        temperature = jnp.asarray(temperature, dtype=float)
        velocity = jnp.asarray(velocity_phys, dtype=float)
        density = jnp.asarray(density, dtype=float)
        heat_capacity = jnp.asarray(heat_capacity, dtype=float)
        conductivity = jnp.asarray(conductivity, dtype=float)
        expected_velocity_shape = (3, *shape)
        for field, name in (
            (temperature, "temperature"),
            (density, "density"),
            (heat_capacity, "heat_capacity"),
            (conductivity, "conductivity"),
        ):
            if field.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {field.shape}")
        if velocity.shape != expected_velocity_shape:
            raise ValueError(f"velocity_phys must have shape {expected_velocity_shape}, got {velocity.shape}")
        rho_cp = _safe_rho_cp(density, heat_capacity)
        out = temperature

        # First-order upwind advection in each physical axis.  Every missing
        # or solid neighbor receives a zero-gradient ghost; Dirichlet box or
        # CAD links replace that ghost only for incoming characteristics.
        for axis in range(3):
            previous = jnp.roll(temperature, 1, axis=axis)
            following = jnp.roll(temperature, -1, axis=axis)
            previous = jnp.where(previous_valid_j[axis], previous, temperature)
            following = jnp.where(following_valid_j[axis], following, temperature)

            min_kind, min_value, _, _ = box_specs[2 * axis]
            max_kind, max_value, _, _ = box_specs[2 * axis + 1]
            if min_kind == "temperature":
                incoming = plane_min_j[axis] & (velocity[axis] > 0.0)
                previous = jnp.where(incoming, min_value, previous)
            if max_kind == "temperature":
                incoming = plane_max_j[axis] & (velocity[axis] < 0.0)
                following = jnp.where(incoming, max_value, following)

            for links, _, spec in local_static:
                kind, value, _, _ = spec
                if kind != "temperature":
                    continue
                incoming_min = links[2 * axis] & (velocity[axis] > 0.0)
                incoming_max = links[2 * axis + 1] & (velocity[axis] < 0.0)
                previous = jnp.where(incoming_min, value, previous)
                following = jnp.where(incoming_max, value, following)

            gradient_previous = (temperature - previous) / dx_value
            gradient_following = (following - temperature) / dx_value
            gradient = jnp.where(velocity[axis] >= 0.0, gradient_previous, gradient_following)
            out = out + jnp.where(fluid_j, -dt_value * velocity[axis] * gradient, 0.0)

        # Internal conduction is written as a divergence from both neighbors.
        # Pair masks are static and ensure a wrapped roll never couples the two
        # opposite domain borders.
        for axis in range(3):
            previous = jnp.roll(temperature, 1, axis=axis)
            following = jnp.roll(temperature, -1, axis=axis)
            previous_k = jnp.roll(conductivity, 1, axis=axis)
            following_k = jnp.roll(conductivity, -1, axis=axis)
            k_left = 2.0 * conductivity * previous_k / jnp.maximum(conductivity + previous_k, 1.0e-30)
            k_right = 2.0 * conductivity * following_k / jnp.maximum(conductivity + following_k, 1.0e-30)
            left_delta = dt_value * k_left * (previous - temperature) / (dx_value * dx_value * rho_cp)
            right_delta = dt_value * k_right * (following - temperature) / (dx_value * dx_value * rho_cp)
            out = out + jnp.where(pair_left_j[axis], left_delta, 0.0)
            out = out + jnp.where(pair_right_j[axis], right_delta, 0.0)

        # Explicit box face conditions use half-cell distance for Dirichlet
        # and the series resistance for Robin exchange.
        for index, face in enumerate(_FACES):
            kind, value, coefficient, ambient = box_specs[index]
            if kind == "adiabatic":
                continue
            mask = face_masks_j[face]
            if kind == "temperature":
                increment = 2.0 * dt_value * conductivity * (value - temperature) / (
                    dx_value * dx_value * rho_cp
                )
            elif kind == "heat-flux":
                increment = dt_value * value / (dx_value * rho_cp)
            else:
                h_eff = _h_eff(conductivity, coefficient)
                increment = dt_value * h_eff * (ambient - temperature) / (dx_value * rho_cp)
            out = out + jnp.where(mask, increment, 0.0)

        # Legacy CAD fallback applies to fluid cells adjacent to an inactive
        # cell.  Box border links are excluded in the precomputed contacts.
        if cad_spec is not None:
            # The fallback source is defined on fluid cells; conductive
            # solids use harmonic internal conduction instead.
            out = _apply_wall_source(
                out,
                temperature,
                rho_cp,
                conductivity,
                jnp.where(fluid_j, cad_wall_j, 0),
                cad_spec,
            )

        # Selector-local CAD sources use directional contact counts.  Their
        # masks have already been made disjoint by solver._cad_boundary_actions.
        for links, contacts, spec in local_static:
            del links
            out = _apply_wall_source(out, temperature, rho_cp, conductivity, contacts, spec)

        return jnp.where(active_j, out, temperature)

    def stability(density: Any, heat_capacity: Any, conductivity: Any) -> tuple[Any, Any, Any]:
        """Return device-side Fourier, Robin, and explicit loss maxima."""

        density = jnp.asarray(density, dtype=float)
        heat_capacity = jnp.asarray(heat_capacity, dtype=float)
        conductivity = jnp.asarray(conductivity, dtype=float)
        for field, name in (
            (density, "density"),
            (heat_capacity, "heat_capacity"),
            (conductivity, "conductivity"),
        ):
            if field.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {field.shape}")
        rho_cp = _safe_rho_cp(density, heat_capacity)
        alpha = conductivity / rho_cp
        fo_field = jnp.where(active_j, alpha * dt_value / (dx_value * dx_value), 0.0)
        fourier = jnp.max(fo_field)
        loss = jnp.zeros_like(rho_cp)
        robin = jnp.asarray(0.0, dtype=rho_cp.dtype)

        for axis in range(3):
            previous_k = jnp.roll(conductivity, 1, axis=axis)
            following_k = jnp.roll(conductivity, -1, axis=axis)
            k_left = 2.0 * conductivity * previous_k / jnp.maximum(conductivity + previous_k, 1.0e-30)
            k_right = 2.0 * conductivity * following_k / jnp.maximum(conductivity + following_k, 1.0e-30)
            left_coeff = dt_value * k_left / (dx_value * dx_value * rho_cp)
            right_coeff = dt_value * k_right / (dx_value * dx_value * rho_cp)
            loss = loss + jnp.where(pair_left_j[axis], left_coeff, 0.0)
            loss = loss + jnp.where(pair_right_j[axis], right_coeff, 0.0)

            min_kind, _, min_h, _ = box_specs[2 * axis]
            max_kind, _, max_h, _ = box_specs[2 * axis + 1]
            min_cells = active_j & plane_min_j[axis]
            max_cells = active_j & plane_max_j[axis]
            if min_kind == "temperature":
                loss = loss + jnp.where(min_cells, 2.0 * dt_value * conductivity / (dx_value * dx_value * rho_cp), 0.0)
            elif min_kind == "convection":
                min_coeff = dt_value * _h_eff(conductivity, min_h) / (dx_value * rho_cp)
                loss = loss + jnp.where(min_cells, min_coeff, 0.0)
                robin = jnp.maximum(robin, jnp.max(jnp.where(min_cells, min_coeff, 0.0)))
            if max_kind == "temperature":
                loss = loss + jnp.where(max_cells, 2.0 * dt_value * conductivity / (dx_value * dx_value * rho_cp), 0.0)
            elif max_kind == "convection":
                max_coeff = dt_value * _h_eff(conductivity, max_h) / (dx_value * rho_cp)
                loss = loss + jnp.where(max_cells, max_coeff, 0.0)
                robin = jnp.maximum(robin, jnp.max(jnp.where(max_cells, max_coeff, 0.0)))

        # Active/inactive internal contacts are the legacy CAD fallback
        # contribution.  Domain border contacts are not in this mask.  The
        # aggregate count is added once here; adding it in the axis loop would
        # count every contact six times.
        if cad_spec is not None:
            cad_kind, _, cad_h, _ = cad_spec
            if cad_kind == "temperature":
                loss = loss + jnp.where(
                    cad_active_boundary_j > 0,
                    cad_active_boundary_j * 2.0 * dt_value * conductivity / (dx_value * dx_value * rho_cp),
                    0.0,
                )
            elif cad_kind == "convection":
                cad_coeff = dt_value * _h_eff(conductivity, cad_h) / (dx_value * rho_cp)
                loss = loss + jnp.where(cad_active_boundary_j > 0, cad_active_boundary_j * cad_coeff, 0.0)
                # Robin is a per-contact coefficient; multiple contacts in a
                # corner increase the cell loss but do not change this
                # diagnostic's maximum single-face coefficient.
                robin = jnp.maximum(robin, jnp.max(jnp.where(cad_active_boundary_j > 0, cad_coeff, 0.0)))

        for _, contacts, spec in local_static:
            kind, _, coefficient, _ = spec
            cells = contacts > 0.0
            if kind == "temperature":
                coeff = contacts * 2.0 * dt_value * conductivity / (dx_value * dx_value * rho_cp)
                loss = loss + jnp.where(cells, coeff, 0.0)
            elif kind == "convection":
                coeff = contacts * dt_value * _h_eff(conductivity, coefficient) / (dx_value * rho_cp)
                loss = loss + jnp.where(cells, coeff, 0.0)
                robin = jnp.maximum(robin, jnp.max(jnp.where(cells, coeff, 0.0)))

        loss_max = jnp.max(jnp.where(active_j, loss, 0.0))
        return fourier, robin, loss_max

    return step, stability


__all__ = ["make_thermal_operators"]
