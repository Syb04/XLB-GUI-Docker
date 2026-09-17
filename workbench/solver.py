"""Small, self contained D3Q27/XLB solver used by the workbench.

The workbench deliberately keeps the solver independent of the older GUI
code.  XLB supplies the lattice, equilibrium, moments, and BGK collision
operators.  The pull stream below is kept in this module because the generic
XLB stream operator is periodic, whereas a workbench run needs physical box
faces and solid cells.

The flow variables use the usual weakly compressible LBM scaling.  ``dt`` and
the (isotropic) mesh spacing provide the conversion between lattice and SI
units.  Thermal transport is an explicit finite-volume update coupled to the
LBM velocity field.  The density used by the LBM is a dimensionless reference
density; a temperature-dependent material density is used in the
thermal storage/diffusion coefficients and optionally in the buoyancy force.
This reference-density incompressible approximation is
intentional until a compressible thermal lattice is added.
"""

from __future__ import annotations

import csv
from collections import deque
from functools import partial
import importlib
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .restart import load_restart, write_restart
from .snapshots import SnapshotWriter
from .gravity import GravityModel
from .residuals import ConvergenceMonitor, make_residual_operator


# D3Q27 ordering is the ordering used by xlb.velocity_set.D3Q27: the product
# is over x, y, z in [0, -1, 1].  Keeping a local copy is useful for the
# boundary stream, while the actual equilibrium/moment/collision operations
# are always delegated to XLB.
_C = np.asarray(
    [
        (0, 0, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, -1, 0),
        (0, -1, -1),
        (0, -1, 1),
        (0, 1, 0),
        (0, 1, -1),
        (0, 1, 1),
        (-1, 0, 0),
        (-1, 0, -1),
        (-1, 0, 1),
        (-1, -1, 0),
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, 0),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, 0, 0),
        (1, 0, -1),
        (1, 0, 1),
        (1, -1, 0),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, 0),
        (1, 1, -1),
        (1, 1, 1),
    ],
    dtype=np.int8,
).T
_OPPOSITE = np.asarray(
    [0, 2, 1, 6, 8, 7, 3, 5, 4, 18, 20, 19, 24, 26, 25, 21, 23, 22, 9, 11, 10, 15, 17, 16, 12, 14, 13],
    dtype=np.int8,
)
_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _as_float_vector(value: Any, name: str, length: int = 3) -> np.ndarray:
    """Convert a JSON vector to a finite float vector."""

    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite {length}-vector")
    return arr


def _mesh_arrays(
    mesh: Mapping[str, Any],
) -> tuple[
    tuple[int, int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Validate and normalize the public geometry/thermal mesh contract.

    Older meshes only contain ``fluid_mask``.  Such meshes remain all-fluid
    and receive the selected project material automatically.  New CHT meshes
    can provide ``solid_mask``, ``thermal_mask``, per-cell ``material_index``,
    and selector-local CAD ``boundary_links``.
    """

    try:
        mask = np.asarray(mesh["fluid_mask"], dtype=bool)
    except Exception as exc:  # pragma: no cover - defensive error wording
        raise ValueError("mesh must contain fluid_mask") from exc
    if mask.ndim != 3 or not mask.any():
        raise ValueError("mesh.fluid_mask must be a non-empty 3-D array")
    shape = tuple(int(v) for v in mask.shape)
    if any(v < 1 for v in shape):
        raise ValueError("each mesh dimension must contain at least one cell")
    spacing = _as_float_vector(mesh.get("spacing", ()), "mesh.spacing")
    origin = _as_float_vector(mesh.get("origin", (0.0, 0.0, 0.0)), "mesh.origin")
    if np.any(spacing <= 0.0):
        raise ValueError("mesh.spacing must be positive")
    # This restriction is part of the workbench contract.  It also prevents
    # silently using the wrong lattice scaling in the thermal update.
    if not np.allclose(spacing, spacing[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("D3Q27 workbench solver requires isotropic mesh spacing")
    solid = np.asarray(mesh.get("solid_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    if solid.shape != shape:
        raise ValueError("mesh.solid_mask must have the same shape as fluid_mask")
    if np.any(solid & mask):
        raise ValueError("mesh.solid_mask and fluid_mask may not overlap")
    if "thermal_mask" in mesh:
        thermal = np.asarray(mesh["thermal_mask"], dtype=bool)
        if thermal.shape != shape:
            raise ValueError("mesh.thermal_mask must have the same shape as fluid_mask")
    else:
        # Explicit solid masks opt into conductive solid cells.  Legacy
        # obstacle meshes without this field retain adiabatic interiors.
        thermal = mask | solid if "solid_mask" in mesh else mask.copy()
    if np.any(mask & ~thermal) or np.any(solid & ~thermal):
        raise ValueError("mesh.thermal_mask must contain all fluid and solid cells")

    if "material_index" in mesh:
        material_index = np.asarray(mesh["material_index"], dtype=np.int64)
        if material_index.shape != shape:
            raise ValueError("mesh.material_index must have the same shape as fluid_mask")
    else:
        material_index = np.full(shape, -1, dtype=np.int64)
    if np.any(material_index < -1):
        raise ValueError("mesh.material_index entries must be -1 or a material index")

    links: dict[str, np.ndarray] = {}
    raw_links = mesh.get("boundary_links", {})
    if raw_links is None:
        raw_links = {}
    if not isinstance(raw_links, Mapping):
        raise ValueError("mesh.boundary_links must be an object mapping selectors to six link masks")
    expected = (6, *shape)
    for selector, value in raw_links.items():
        selector_name = str(selector).lower()
        array = np.asarray(value, dtype=bool)
        if array.shape != expected:
            # A few mesh writers naturally emit [nx,ny,nz,6].  Accepting it is
            # harmless while the canonical public shape remains [6,nx,ny,nz].
            if array.shape == (*shape, 6):
                array = np.moveaxis(array, -1, 0)
            else:
                raise ValueError(f"mesh.boundary_links[{selector!r}] must have shape (6,nx,ny,nz)")
        links[selector_name] = array & mask[None, ...]
    return shape, mask, solid, thermal, material_index, origin, spacing, links


def _kind_value(prop: Any, default: float, name: str) -> tuple[str, Any]:
    """Normalize a material property without evaluating arbitrary code."""

    if prop is None:
        return "constant", float(default)
    if isinstance(prop, Mapping):
        kind = str(prop.get("kind", "constant")).lower()
        if kind == "constant":
            value = prop.get("value", default)
        elif kind == "table":
            value = prop.get("points")
        else:
            raise ValueError(f"unsupported {name} property kind {kind!r}")
    else:
        kind, value = "constant", prop
    if kind == "constant":
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} constant must be numeric") from exc
        if not math.isfinite(val):
            raise ValueError(f"{name} constant must be finite")
        return kind, val
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(f"{name} table must contain at least two [temperature,value] points")
    if not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValueError(f"{name} table temperatures must be finite and strictly increasing")
    return kind, points


def _evaluate_property(prop: Any, temperature: np.ndarray | float, default: float, name: str, warnings: list[str]) -> np.ndarray:
    """Evaluate a constant or linearly interpolated property table.

    Clamping is explicit and recorded once in ``warnings``.  No expression or
    callback from project JSON is ever executed.
    """

    kind, value = _kind_value(prop, default, name)
    temp = np.asarray(temperature, dtype=float)
    if kind == "constant":
        out = np.full(temp.shape, value, dtype=float)
    else:
        points = np.asarray(value, dtype=float)
        low, high = float(points[0, 0]), float(points[-1, 0])
        if np.any(temp < low) or np.any(temp > high):
            message = f"{name} table was clamped to its endpoint outside [{low:g}, {high:g}] K"
            if message not in warnings:
                warnings.append(message)
        out = np.interp(temp, points[:, 0], points[:, 1])
    if not np.isfinite(out).all():
        raise ValueError(f"{name} evaluation produced non-finite values")
    return out


def _material(project: Mapping[str, Any]) -> Mapping[str, Any]:
    materials = project.get("materials") or []
    if not materials:
        raise ValueError("project must contain at least one material")
    selected = project.get("physics", {}).get("material_id")
    for item in materials:
        if selected is None or item.get("id") == selected:
            return item
    raise ValueError(f"material_id {selected!r} is not present in project.materials")


def _material_table(project: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, int], int]:
    """Return materials, id lookup, and the selected fluid material index."""

    materials_value = project.get("materials") or []
    materials = [value for value in materials_value if isinstance(value, Mapping)]
    if len(materials) != len(materials_value) or not materials:
        raise ValueError("project.materials must contain material objects")
    by_id: dict[str, int] = {}
    for index, material in enumerate(materials):
        identifier = str(material.get("id", ""))
        if not identifier or identifier in by_id:
            raise ValueError("materials must have unique non-empty ids")
        by_id[identifier] = index
    selected_id = (project.get("physics") or {}).get("material_id")
    if selected_id is None:
        selected_index = 0
    else:
        try:
            selected_index = by_id[str(selected_id)]
        except KeyError as exc:
            raise ValueError(f"material_id {selected_id!r} is not present in project.materials") from exc
    return materials, by_id, selected_index


def _resolve_material_index(
    project: Mapping[str, Any],
    geometry: Mapping[str, Any],
    material_index: np.ndarray,
    fluid: np.ndarray,
    solid: np.ndarray,
    thermal: np.ndarray,
    material_ids: Mapping[str, int],
    selected_index: int,
) -> np.ndarray:
    """Fill legacy material indices and validate CHT material assignments."""

    result = np.asarray(material_index, dtype=np.int64).copy()
    # A mesh may deliberately use -1 for inactive cells.  Fluid cells are
    # always active and use the project fluid material when unspecified.
    result[fluid & (result < 0)] = selected_index
    solid_id = geometry.get("solid_material_id")
    if solid_id is not None:
        if str(solid_id) not in material_ids:
            raise ValueError(f"geometry.solid_material_id {solid_id!r} is not present in project.materials")
        result[solid & (result < 0)] = int(material_ids[str(solid_id)])
    if np.any(result[thermal] >= len(material_ids)):
        raise ValueError("mesh.material_index references a material outside project.materials")
    active_missing = thermal & (result < 0)
    if active_missing.any():
        raise ValueError("thermal cells require material_index or geometry.solid_material_id")
    return result


def _material_fields(
    materials: list[Mapping[str, Any]],
    material_index: np.ndarray,
    temperature: np.ndarray,
    active: np.ndarray,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ρ, μ, cp, and k per thermal cell, safely and table-wise."""

    shape = temperature.shape
    # Inactive values never enter a flux, but finite neutral values make array
    # arithmetic and diagnostics well behaved.
    density = np.ones(shape, dtype=float)
    viscosity = np.ones(shape, dtype=float)
    heat_capacity = np.ones(shape, dtype=float)
    conductivity = np.ones(shape, dtype=float)
    for index, material in enumerate(materials):
        selected = active & (material_index == index)
        if not selected.any():
            continue
        density[selected] = _evaluate_property(
            material.get("density"), temperature[selected], 1000.0, f"material[{index}].density", warnings
        )
        viscosity[selected] = _evaluate_property(
            material.get("viscosity"), temperature[selected], 1.0e-3, f"material[{index}].viscosity", warnings
        )
        heat_capacity[selected] = _evaluate_property(
            material.get("heat_capacity"), temperature[selected], 4182.0, f"material[{index}].heat_capacity", warnings
        )
        conductivity[selected] = _evaluate_property(
            material.get("conductivity"), temperature[selected], 0.6, f"material[{index}].conductivity", warnings
        )
    return density, viscosity, heat_capacity, conductivity


def _study_values(project: Mapping[str, Any]) -> tuple[int, int, float, str]:
    study = project.get("study") or {}
    try:
        steps = int(study.get("steps", 0))
        interval = int(study.get("output_interval", max(1, steps)))
        dt = float(study.get("dt", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("study.steps, output_interval, and dt must be numeric") from exc
    device = str(study.get("device", "cpu")).lower()
    if steps < 1:
        raise ValueError("study.steps must be at least one")
    if interval < 1:
        raise ValueError("study.output_interval must be at least one")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("study.dt must be positive and finite")
    if device in {"gpu", "cuda", "cuda:0", "gpu:0"}:
        device = "cuda:0"
    elif device not in {"cpu", "cuda:0-ram"}:
        raise ValueError("study.device must be 'cpu', 'cuda:0', or 'cuda:0-ram'")
    return steps, interval, dt, device


def _face_index(face: str) -> tuple[int, int]:
    face = str(face).lower()
    try:
        axis = "xyz".index(face[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unknown boundary face {face!r}") from exc
    if face not in _FACES:
        raise ValueError(f"unknown boundary face {face!r}")
    return axis, 0 if face.endswith("min") else -1


def _face_masks(shape: tuple[int, int, int], fluid: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for face in _FACES:
        axis, index = _face_index(face)
        m = np.zeros(shape, dtype=bool)
        sl = [slice(None)] * 3
        sl[axis] = index
        m[tuple(sl)] = fluid[tuple(sl)]
        masks[face] = m
    return masks


def _boundary_maps(
    project: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any] | None, dict[str, Mapping[str, Any]]]:
    """Return box faces, the global CAD fallback, and local CAD selectors."""

    boxes: dict[str, Mapping[str, Any]] = {}
    cad: Mapping[str, Any] | None = None
    cad_patches: dict[str, Mapping[str, Any]] = {}
    for boundary in project.get("boundaries") or []:
        if not isinstance(boundary, Mapping):
            raise ValueError("each boundary must be an object")
        face = str(boundary.get("face", "")).lower()
        if face == "cad":
            if cad is not None:
                raise ValueError("only one CAD wall boundary is supported")
            cad = boundary
        elif face.startswith("cad:") and len(face) > 4:
            if face in cad_patches:
                raise ValueError(f"duplicate boundary selector {face}")
            cad_patches[face] = boundary
        elif face in _FACES:
            if face in boxes:
                raise ValueError(f"duplicate boundary for face {face}")
            boxes[face] = boundary
        else:
            raise ValueError(f"unknown boundary face {face!r}")
    return boxes, cad, cad_patches


def _flow_type(spec: Mapping[str, Any]) -> str:
    flow = spec.get("flow")
    if not isinstance(flow, Mapping):
        return "no-slip"
    return str(flow.get("type", "no-slip")).lower().replace("_", "-")


def _thermal_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    value = spec.get("thermal")
    return value if isinstance(value, Mapping) else {"type": "adiabatic"}


def _thermal_type(spec: Mapping[str, Any]) -> str:
    return str(spec.get("type", "adiabatic")).lower().replace("_", "-")


def _property_scalar(prop: Any, temperature: float, default: float, name: str, warnings: list[str]) -> float:
    result = _evaluate_property(prop, np.asarray([temperature]), default, name, warnings)
    return float(result[0])


def _thermal_step(
    temperature: np.ndarray,
    velocity: np.ndarray,
    fluid: np.ndarray,
    density: np.ndarray,
    heat_capacity: np.ndarray,
    conductivity: np.ndarray,
    dt: float,
    dx: float,
    box_thermal: Mapping[str, Mapping[str, Any]],
    cad_thermal: Mapping[str, Any] | None,
    face_masks: Mapping[str, np.ndarray],
    warnings: list[str],
    thermal_mask: np.ndarray | None = None,
    cad_link_actions: Iterable[tuple[np.ndarray, Mapping[str, Any]]] = (),
) -> np.ndarray:
    """Advance temperature by one explicit finite-volume/advection step.

    ``velocity`` is in m/s, and the arrays use x, y, z cell order.  Advection
    is restricted to ``fluid`` while diffusion spans every active cell in
    ``thermal_mask`` (fluid-solid coupling therefore uses the same harmonic
    face conductivity as solid-solid conduction).  Explicit wall terms use
    half-cell distances for Dirichlet and convection conditions.
    """

    active = fluid if thermal_mask is None else np.asarray(thermal_mask, dtype=bool)
    if active.shape != fluid.shape:
        raise ValueError("thermal_mask must have the same shape as fluid")
    if np.any(fluid & ~active):
        raise ValueError("thermal_mask must contain all fluid cells")
    # The same links are consulted once per advection axis and again for the
    # conductive wall source; materialize a generator so both passes agree.
    cad_link_actions = tuple(cad_link_actions)
    for link_mask, _ in cad_link_actions:
        if np.asarray(link_mask, dtype=bool).shape != (6, *fluid.shape):
            raise ValueError("CAD thermal boundary links must have shape (6,nx,ny,nz)")
    out = np.array(temperature, dtype=float, copy=True)
    rho_cp = density * heat_capacity
    if np.any(rho_cp[active] <= 0.0) or np.any(conductivity[active] <= 0.0):
        raise ValueError("density, heat_capacity, and conductivity must be positive in thermal cells")

    # Upwind advection.  A missing/solid neighbour receives a zero gradient;
    # the box face condition itself is imposed below for temperature.
    for axis in range(3):
        prev = np.roll(temperature, 1, axis=axis)
        nxt = np.roll(temperature, -1, axis=axis)
        prev_fluid = np.roll(fluid, 1, axis=axis)
        next_fluid = np.roll(fluid, -1, axis=axis)
        sl_min = [slice(None)] * 3
        sl_min[axis] = 0
        sl_max = [slice(None)] * 3
        sl_max[axis] = -1
        prev_fluid[tuple(sl_min)] = False
        next_fluid[tuple(sl_max)] = False
        prev = np.where(prev_fluid, prev, temperature)
        nxt = np.where(next_fluid, nxt, temperature)
        # For an incoming upwind characteristic, use a prescribed boundary
        # temperature as the exterior ghost value.  Merely fixing the wall
        # cell after the update gives a zero-gradient inlet and under-heats a
        # flowing channel by one cell.  Heat-flux/Robin walls use a zero
        # advective gradient; their conductive term below carries the wall
        # exchange.
        min_face = ("xmin", "ymin", "zmin")[axis]
        max_face = ("xmax", "ymax", "zmax")[axis]
        min_thermal = box_thermal.get(min_face, {"type": "adiabatic"})
        max_thermal = box_thermal.get(max_face, {"type": "adiabatic"})
        min_kind = _thermal_type(min_thermal)
        max_kind = _thermal_type(max_thermal)
        if min_kind in {"temperature", "dirichlet"}:
            try:
                min_value = float(min_thermal.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"temperature boundary {min_face} requires numeric value") from exc
            if not math.isfinite(min_value):
                raise ValueError(f"temperature boundary {min_face} value must be finite")
            # The only cell that sees this ghost is the min-face cell.  It is
            # still useful for thin domains and is exact for the imposed
            # boundary characteristic.
            min_slice = [slice(None)] * 3
            min_slice[axis] = 0
            min_slice_t = tuple(min_slice)
            prev[min_slice_t] = np.where(velocity[axis][min_slice_t] > 0.0, min_value, prev[min_slice_t])
        if max_kind in {"temperature", "dirichlet"}:
            try:
                max_value = float(max_thermal.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"temperature boundary {max_face} requires numeric value") from exc
            if not math.isfinite(max_value):
                raise ValueError(f"temperature boundary {max_face} value must be finite")
            max_slice = [slice(None)] * 3
            max_slice[axis] = -1
            max_slice_t = tuple(max_slice)
            nxt[max_slice_t] = np.where(velocity[axis][max_slice_t] < 0.0, max_value, nxt[max_slice_t])
        # A selector-local CAD temperature port has the same incoming
        # characteristic as a box Dirichlet face.  ``boundary_links`` encode
        # the outward contact direction (x-, x+, y-, ...), so only the link
        # direction parallel to this advection sweep is touched.  Heat-flux
        # and Robin links retain the zero-gradient advective ghost; their
        # conductive exchange is applied after the sweep below.
        for link_masks, cad_temperature in cad_link_actions:
            kind = _thermal_type(cad_temperature)
            if kind not in {"temperature", "dirichlet"}:
                continue
            try:
                value = float(cad_temperature.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError("CAD patch temperature boundary requires numeric value") from exc
            if not math.isfinite(value):
                raise ValueError("CAD patch temperature boundary must be finite")
            minimum_links = np.asarray(link_masks[2 * axis], dtype=bool)
            maximum_links = np.asarray(link_masks[2 * axis + 1], dtype=bool)
            min_incoming = minimum_links & (velocity[axis] > 0.0)
            max_incoming = maximum_links & (velocity[axis] < 0.0)
            prev = np.where(min_incoming, value, prev)
            nxt = np.where(max_incoming, value, nxt)
        grad_pos = (temperature - prev) / dx
        grad_neg = (nxt - temperature) / dx
        out[fluid] += dt * (-velocity[axis, fluid] * np.where(velocity[axis, fluid] >= 0.0, grad_pos[fluid], grad_neg[fluid]))

    # Conductive fluxes at each internal fluid-fluid face.  Harmonic means
    # avoid an artificial high-conductivity shortcut at a material jump.
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left_t, right_t = tuple(left), tuple(right)
        pair = active[left_t] & active[right_t]
        if not pair.any():
            continue
        k0, k1 = conductivity[left_t], conductivity[right_t]
        kface = 2.0 * k0 * k1 / np.maximum(k0 + k1, 1.0e-30)
        flux = kface * (temperature[right_t] - temperature[left_t]) / dx
        left_rhocp = rho_cp[left_t]
        right_rhocp = rho_cp[right_t]
        delta_left = dt * flux / (dx * left_rhocp)
        delta_right = -dt * flux / (dx * right_rhocp)
        target_l = out[left_t]
        target_r = out[right_t]
        target_l[pair] += delta_left[pair]
        target_r[pair] += delta_right[pair]
        out[left_t], out[right_t] = target_l, target_r

    # Box face thermal conditions.  A positive heat flux is defined as heat
    # entering the fluid, matching the workbench API.
    for face, face_mask in face_masks.items():
        # ``box_thermal`` already contains the nested thermal object for each
        # face.  (Boundary objects are unwrapped by ``simulate``.)
        thermal = box_thermal.get(face, {"type": "adiabatic"})
        kind = _thermal_type(thermal)
        if kind in {"adiabatic", "insulated", "none"}:
            continue
        if kind in {"temperature", "dirichlet"}:
            try:
                value = float(thermal.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"temperature boundary {face} requires numeric value") from exc
            if not math.isfinite(value):
                raise ValueError(f"temperature boundary {face} value must be finite")
            axis, _ = _face_index(face)
            # Apply the Dirichlet value at the exterior face through a
            # half-cell conductive flux.  The boundary cell remains a real
            # cell-centred unknown; overwriting it would erase the FV update
            # and shift the wall by half a cell.
            out[face_mask] += 2.0 * dt * conductivity[face_mask] * (value - temperature[face_mask]) / (dx * dx * rho_cp[face_mask])
        elif kind in {"heat-flux", "flux", "neumann"}:
            try:
                flux = float(thermal.get("value", thermal.get("heat_flux")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"heat_flux boundary {face} requires numeric value") from exc
            out[face_mask] += dt * flux / (dx * rho_cp[face_mask])
        elif kind in {"convection", "convective"}:
            try:
                coefficient = float(thermal.get("h", thermal.get("coefficient")))
                ambient = float(thermal.get("ambient_temperature", thermal.get("ambient")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"convection boundary {face} requires h and ambient_temperature") from exc
            if not math.isfinite(coefficient) or coefficient < 0.0 or not math.isfinite(ambient):
                raise ValueError(f"convection boundary {face} has invalid h or ambient_temperature")
            # Cell-centred Robin condition: the half-cell conduction
            # resistance and the surface h resistance act in series.
            h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity[face_mask]), 1.0e-30)
            out[face_mask] += dt * h_eff * (ambient - temperature[face_mask]) / (dx * rho_cp[face_mask])
        else:
            raise ValueError(f"unsupported thermal boundary type {kind!r} on {face}")

    # A CAD wall is represented by fluid cells adjacent to a solid mask.  It
    # has no flux by default; explicit temperature/flux/convection terms are
    # applied once per solid lattice contact.
    if cad_thermal is not None:
        kind = _thermal_type(cad_thermal)
        if kind not in {"adiabatic", "insulated", "none", "temperature", "dirichlet", "heat-flux", "flux", "neumann", "convection", "convective"}:
            raise ValueError(f"unsupported thermal boundary type {kind!r} on cad")
        if kind not in {"adiabatic", "insulated", "none"}:
            wall_contacts = np.zeros(fluid.shape, dtype=np.int8)
            for axis in range(3):
                for shift in (-1, 1):
                    neighbour = np.roll(active, shift, axis=axis)
                    sl = [slice(None)] * 3
                    sl[axis] = 0 if shift == 1 else -1
                    # Out-of-domain box faces are handled by their own box
                    # thermal condition below.  They are not CAD contacts.
                    neighbour[tuple(sl)] = True
                    wall_contacts += fluid & ~neighbour
            wall = fluid & (wall_contacts > 0)
            contacts = np.maximum(wall_contacts[wall], 1)
            if kind in {"temperature", "dirichlet"}:
                try:
                    value = float(cad_thermal.get("value"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("CAD temperature boundary requires numeric value") from exc
                out[wall] += contacts * 2.0 * dt * conductivity[wall] * (value - temperature[wall]) / (dx * dx * rho_cp[wall])
            elif kind in {"heat-flux", "flux", "neumann"}:
                try:
                    flux = float(cad_thermal.get("value", cad_thermal.get("heat_flux")))
                except (TypeError, ValueError) as exc:
                    raise ValueError("CAD heat_flux boundary requires numeric value") from exc
                out[wall] += contacts * dt * flux / (dx * rho_cp[wall])
            else:
                try:
                    coefficient = float(cad_thermal.get("h", cad_thermal.get("coefficient")))
                    ambient = float(cad_thermal.get("ambient_temperature", cad_thermal.get("ambient")))
                except (TypeError, ValueError) as exc:
                    raise ValueError("CAD convection boundary requires h and ambient_temperature") from exc
                if not math.isfinite(coefficient) or coefficient < 0.0 or not math.isfinite(ambient):
                    raise ValueError("CAD convection boundary has invalid h or ambient_temperature")
                h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity[wall]), 1.0e-30)
                out[wall] += contacts * dt * h_eff * (ambient - temperature[wall]) / (dx * rho_cp[wall])

    # Selector-local CAD conditions are represented by six directional link
    # masks.  A contact count preserves heat-flux/Robin area at diagonally
    # faceted patches instead of collapsing all links to one cell mask.
    for link_mask, thermal in cad_link_actions:
        links = np.asarray(link_mask, dtype=bool)
        if links.shape != (6, *fluid.shape):
            raise ValueError("CAD thermal boundary links must have shape (6,nx,ny,nz)")
        contacts = np.sum(links & active[None, ...], axis=0).astype(float)
        cells = contacts > 0.0
        kind = _thermal_type(thermal)
        if kind in {"adiabatic", "insulated", "none"}:
            continue
        if kind in {"temperature", "dirichlet"}:
            try:
                value = float(thermal.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError("CAD patch temperature boundary requires numeric value") from exc
            if not math.isfinite(value):
                raise ValueError("CAD patch temperature boundary must be finite")
            out[cells] += contacts[cells] * 2.0 * dt * conductivity[cells] * (value - temperature[cells]) / (dx * dx * rho_cp[cells])
        elif kind in {"heat-flux", "flux", "neumann"}:
            try:
                flux = float(thermal.get("value", thermal.get("heat_flux")))
            except (TypeError, ValueError) as exc:
                raise ValueError("CAD patch heat_flux boundary requires numeric value") from exc
            out[cells] += contacts[cells] * dt * flux / (dx * rho_cp[cells])
        elif kind in {"convection", "convective"}:
            try:
                coefficient = float(thermal.get("h", thermal.get("coefficient")))
                ambient = float(thermal.get("ambient_temperature", thermal.get("ambient")))
            except (TypeError, ValueError) as exc:
                raise ValueError("CAD patch convection boundary requires h and ambient_temperature") from exc
            if not math.isfinite(coefficient) or coefficient < 0.0 or not math.isfinite(ambient):
                raise ValueError("CAD patch convection boundary has invalid h or ambient_temperature")
            h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity[cells]), 1.0e-30)
            out[cells] += contacts[cells] * dt * h_eff * (ambient - temperature[cells]) / (dx * rho_cp[cells])
        else:
            raise ValueError(f"unsupported thermal boundary type {kind!r} on CAD patch")

    out[~active] = temperature[~active]
    if not np.isfinite(out[active]).all():
        raise RuntimeError("thermal update produced a non-finite temperature")
    return out


def _thermal_stability_numbers(
    density: np.ndarray,
    heat_capacity: np.ndarray,
    conductivity: np.ndarray,
    fluid: np.ndarray,
    dt: float,
    dx: float,
    box_thermal: Mapping[str, Mapping[str, Any]],
    cad_thermal: Mapping[str, Any] | None,
    face_masks: Mapping[str, np.ndarray],
    thermal_mask: np.ndarray | None = None,
    cad_link_actions: Iterable[tuple[np.ndarray, Mapping[str, Any]]] = (),
) -> tuple[float, float, float]:
    """Return Fourier, Robin, and exact maximum loss numbers for one step.

    The explicit update has at most six conductive neighbours.  The returned
    loss coefficient sums the actual axis-neighbour, half-cell Dirichlet, and
    Robin terms for each cell, catching unstable thin-cell wall updates before
    they create non-finite fields.
    """

    active = fluid if thermal_mask is None else np.asarray(thermal_mask, dtype=bool)
    if active.shape != fluid.shape or np.any(fluid & ~active):
        raise ValueError("thermal_mask must contain all fluid cells")
    rho_cp = density * heat_capacity
    alpha = conductivity / rho_cp
    fo = float(np.max((alpha * dt / (dx * dx))[active]))
    robin = 0.0
    loss = np.zeros_like(rho_cp, dtype=float)

    # Build the exact largest explicit diagonal coefficient over axis-neighbour
    # contacts.  At a Dirichlet corner a half-cell face contributes 2*Fo,
    # while an interior face contributes Fo; counting contacts avoids a loose
    # global 6*Fo estimate that can miss a three-face corner.
    for axis in range(3):
        for shift in (1, -1):
            neighbour = np.roll(active, shift, axis=axis)
            outside = np.zeros(fluid.shape, dtype=bool)
            sl = [slice(None)] * 3
            if shift == 1:
                sl[axis] = 0
            else:
                sl[axis] = -1
            outside[tuple(sl)] = True
            current = active
            pair = current & neighbour & ~outside
            k_neighbour = np.roll(conductivity, shift, axis=axis)
            k_face = 2.0 * conductivity * k_neighbour / np.maximum(conductivity + k_neighbour, 1.0e-30)
            loss += np.where(pair, dt * k_face / (dx * dx * rho_cp), 0.0)

            boundary_face = ("xmin", "ymin", "zmin")[axis] if shift == 1 else ("xmax", "ymax", "zmax")[axis]
            boundary_cells = current & outside
            thermal = box_thermal.get(boundary_face, {"type": "adiabatic"})
            kind = _thermal_type(thermal)
            if kind in {"temperature", "dirichlet"}:
                coeff = 2.0 * dt * conductivity / (dx * dx * rho_cp)
                loss += np.where(boundary_cells, coeff, 0.0)
            elif kind in {"convection", "convective"}:
                try:
                    coefficient = float(thermal.get("h", thermal.get("coefficient")))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"convection boundary {boundary_face} requires h and ambient_temperature") from exc
                if not math.isfinite(coefficient) or coefficient < 0.0:
                    raise ValueError(f"convection boundary {boundary_face} has invalid h")
                h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity), 1.0e-30)
                coeff = dt * h_eff / (dx * rho_cp)
                loss += np.where(boundary_cells, coeff, 0.0)
                if boundary_cells.any():
                    robin = max(robin, float(np.max(coeff[boundary_cells])))

            # Solid cells inside a CAD mask are adiabatic unless a CAD wall
            # thermal object is supplied.  Accumulate every touching contact.
            solid = current & ~neighbour & ~outside
            if cad_thermal is not None:
                cad_kind = _thermal_type(cad_thermal)
                if cad_kind in {"temperature", "dirichlet"}:
                    coeff = 2.0 * dt * conductivity / (dx * dx * rho_cp)
                    loss += np.where(solid, coeff, 0.0)
                elif cad_kind in {"convection", "convective"}:
                    try:
                        coefficient = float(cad_thermal.get("h", cad_thermal.get("coefficient")))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("CAD convection boundary requires h and ambient_temperature") from exc
                    if not math.isfinite(coefficient) or coefficient < 0.0:
                        raise ValueError("CAD convection boundary has invalid h")
                    h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity), 1.0e-30)
                    coeff = dt * h_eff / (dx * rho_cp)
                    loss += np.where(solid, coeff, 0.0)
                    if solid.any():
                        robin = max(robin, float(np.max(coeff[solid])))
    for link_mask, thermal in cad_link_actions:
        links = np.asarray(link_mask, dtype=bool)
        if links.shape != (6, *fluid.shape):
            raise ValueError("CAD thermal boundary links must have shape (6,nx,ny,nz)")
        contacts = np.sum(links & active[None, ...], axis=0).astype(float)
        cells = contacts > 0.0
        kind = _thermal_type(thermal)
        if kind in {"temperature", "dirichlet"}:
            loss += np.where(cells, contacts * 2.0 * dt * conductivity / (dx * dx * rho_cp), 0.0)
        elif kind in {"convection", "convective"}:
            try:
                coefficient = float(thermal.get("h", thermal.get("coefficient")))
            except (TypeError, ValueError) as exc:
                raise ValueError("CAD patch convection boundary requires h and ambient_temperature") from exc
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise ValueError("CAD patch convection boundary has invalid h")
            h_eff = 1.0 / np.maximum(1.0 / max(coefficient, 1.0e-30) + dx / (2.0 * conductivity), 1.0e-30)
            coeff = contacts * dt * h_eff / (dx * rho_cp)
            loss += np.where(cells, coeff, 0.0)
            if cells.any():
                robin = max(robin, float(np.max(coeff[cells])))
    loss_max = float(np.max(loss[active]))
    return fo, robin, loss_max


def _valid_source_masks(shape: tuple[int, int, int]) -> np.ndarray:
    masks = np.ones((27, *shape), dtype=bool)
    for q, c in enumerate(_C.T):
        m = masks[q]
        for axis, component in enumerate(c):
            sl = [slice(None)] * 3
            if component > 0:
                sl[axis] = slice(0, int(component))
                m[tuple(sl)] = False
            elif component < 0:
                sl[axis] = slice(int(component), None)
                m[tuple(sl)] = False
    return masks


def _stream_jax(f_post: Any, fluid: Any, source_valid: Any, jnp: Any) -> Any:
    """Custom pull stream with full-way bounceback for solids and box edges."""

    values = []
    for q, c in enumerate(_C.T):
        shift = tuple(int(v) for v in c)
        pulled = jnp.roll(f_post[q], shift, axis=(0, 1, 2))
        source_is_fluid = jnp.roll(fluid, shift, axis=(0, 1, 2))
        valid = jnp.logical_and(source_valid[q], source_is_fluid)
        bounce = f_post[int(_OPPOSITE[q])]
        values.append(jnp.where(fluid, jnp.where(valid, pulled, bounce), jnp.asarray(0.0, dtype=f_post.dtype)))
    return jnp.stack(values, axis=0)


def _turbulence_config(physics: Mapping[str, Any]) -> tuple[str, float, float]:
    """Normalize the optional laminar/Smagorinsky LES settings."""

    value = physics.get("turbulence") or {}
    if not isinstance(value, Mapping):
        raise ValueError("physics.turbulence must be an object")
    model = str(value.get("model", "laminar")).lower().replace("_", "-")
    if model not in {"laminar", "smagorinsky"}:
        raise ValueError("physics.turbulence.model must be 'laminar' or 'smagorinsky'")
    try:
        coefficient = float(value.get("smagorinsky_constant", 0.17))
        turbulent_prandtl = float(value.get("turbulent_prandtl", 0.9))
    except (TypeError, ValueError) as exc:
        raise ValueError("Smagorinsky constant and turbulent_prandtl must be numeric") from exc
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("physics.turbulence.smagorinsky_constant must be positive and finite")
    if not math.isfinite(turbulent_prandtl) or turbulent_prandtl <= 0.0:
        raise ValueError("physics.turbulence.turbulent_prandtl must be positive and finite")
    return model, coefficient, turbulent_prandtl


def _velocity_gradient(velocity: np.ndarray, fluid: np.ndarray, dx: float) -> np.ndarray:
    """Return ∂u_i/∂x_j using centred differences and valid neighbours."""

    gradients = np.zeros((3, 3, *fluid.shape), dtype=float)
    for component in range(3):
        field = velocity[component]
        for axis in range(3):
            previous = np.roll(field, 1, axis=axis)
            following = np.roll(field, -1, axis=axis)
            previous_valid = np.roll(fluid, 1, axis=axis)
            following_valid = np.roll(fluid, -1, axis=axis)
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = 0
            high[axis] = -1
            previous_valid[tuple(low)] = False
            following_valid[tuple(high)] = False
            # A one-sided difference is preferable to a zero gradient at an
            # open face.  Solid neighbours are treated as no-slip mirrors.
            central = (following - previous) / (2.0 * dx)
            forward = (following - field) / dx
            backward = (field - previous) / dx
            derivative = np.where(previous_valid & following_valid, central, 0.0)
            derivative = np.where(~previous_valid & following_valid, forward, derivative)
            derivative = np.where(previous_valid & ~following_valid, backward, derivative)
            gradients[component, axis] = np.where(fluid, derivative, 0.0)
    return gradients


def _smagorinsky_eddy_viscosity(
    velocity: np.ndarray,
    fluid: np.ndarray,
    dx: float,
    coefficient: float,
) -> np.ndarray:
    """Compute non-negative Smagorinsky kinematic eddy viscosity in SI units."""

    gradient = _velocity_gradient(velocity, fluid, dx)
    strain = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    # |S| = sqrt(2 S_ij S_ij), the conventional Smagorinsky invariant.
    magnitude = np.sqrt(np.maximum(0.0, 2.0 * np.sum(strain * strain, axis=(0, 1))))
    eddy = (coefficient * dx) ** 2 * magnitude
    return np.where(fluid, np.maximum(eddy, 0.0), 0.0)


_LINK_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _cad_boundary_actions(
    links: Mapping[str, np.ndarray],
    box_specs: Mapping[str, Mapping[str, Any]],
    cad_global: Mapping[str, Any] | None,
    cad_patches: Mapping[str, Mapping[str, Any]],
    fluid: np.ndarray,
    thermal: np.ndarray,
) -> tuple[
    list[tuple[np.ndarray, Mapping[str, Any]]],
    list[tuple[np.ndarray, Mapping[str, Any]]],
    Mapping[str, Any] | None,
]:
    """Resolve selector-local CAD flow/thermal actions from link masks.

    A patch selector wins over the global ``cad`` fallback on overlapping
    links.  Box-face masks win over both.  Two distinct local selectors that
    overlap are rejected because applying two incompatible equilibria or heat
    fluxes would be order dependent.
    """

    def prepared(selector: str) -> np.ndarray | None:
        array = links.get(selector.lower())
        if array is None:
            return None
        output = np.asarray(array, dtype=bool).copy()
        # A CAD link at a box corner belongs to the explicit box face when one
        # exists.  This also prevents double thermal source terms.
        for direction, face in enumerate(_LINK_FACES):
            for box_face in box_specs:
                if box_face == face:
                    output[direction] &= ~(_face_mask_for_active(box_face, fluid.shape, thermal))
        output &= fluid[None, ...]
        return output

    def touches_active_solid(array: np.ndarray) -> bool:
        """Return whether any CAD link points into a conductive solid cell."""

        # A selector link is a fluid-cell contact directed toward the CAD
        # surface.  A non-adiabatic CAD source at a thermal solid interface is
        # ambiguous (it could mean wall exchange or solid conduction), so it
        # is rejected instead of silently applying both models.
        for direction in range(6):
            axis = direction // 2
            shift = 1 if direction % 2 == 0 else -1
            neighbour = np.roll(thermal, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            neighbour[tuple(edge)] = False
            fluid_neighbour = np.roll(fluid, shift, axis=axis)
            fluid_neighbour[tuple(edge)] = False
            if np.any(array[direction] & neighbour & ~fluid_neighbour):
                return True
        return False

    local_actions: list[tuple[np.ndarray, Mapping[str, Any]]] = []
    local_thermal: list[tuple[np.ndarray, Mapping[str, Any]]] = []
    occupied = np.zeros((6, *fluid.shape), dtype=bool)
    for selector, spec in cad_patches.items():
        array = prepared(selector)
        flow = spec.get("flow")
        thermal_spec = _thermal_spec(spec)
        flow_kind = _flow_type(spec)
        thermal_kind = _thermal_type(thermal_spec)
        if array is None or not array.any():
            raise ValueError(f"CAD selector {selector} has no mapped boundary links")
        if np.any(occupied & array):
            raise ValueError(f"CAD selectors overlap on mapped links: {selector}")
        occupied |= array
        if flow_kind in {"velocity", "pressure", "density"}:
            if touches_active_solid(array):
                raise ValueError(f"CAD selector {selector} has a flow condition on a fluid-solid interface; use wall")
            local_actions.append((np.any(array, axis=0), flow))
        if thermal_kind not in {"adiabatic", "insulated", "none"}:
            if touches_active_solid(array):
                raise ValueError(f"CAD selector {selector} has a non-adiabatic condition on a fluid-solid interface")
            local_thermal.append((array, thermal_spec))

    global_array = prepared("cad")
    global_thermal_fallback: Mapping[str, Any] | None = _thermal_spec(cad_global) if cad_global is not None else None
    if global_array is not None:
        global_array &= ~occupied
        if not global_array.any() and cad_global is not None:
            if _flow_type(cad_global) in {"velocity", "pressure", "density"} or _thermal_type(_thermal_spec(cad_global)) not in {"adiabatic", "insulated", "none"}:
                raise ValueError("global CAD boundary has no mapped boundary links")
        if global_array.any():
            if cad_global is not None:
                flow_kind = _flow_type(cad_global)
                thermal_spec = _thermal_spec(cad_global)
                if flow_kind in {"velocity", "pressure", "density"}:
                    if touches_active_solid(global_array):
                        raise ValueError("global CAD boundary has a flow condition on a fluid-solid interface; use wall")
                    local_actions.append((np.any(global_array, axis=0), cad_global.get("flow", {})))
                if _thermal_type(thermal_spec) not in {"adiabatic", "insulated", "none"}:
                    if touches_active_solid(global_array):
                        raise ValueError("global CAD boundary has a non-adiabatic condition on a fluid-solid interface")
                    local_thermal.append((global_array, thermal_spec))
        # Link masks are a complete CAD boundary description.  Do not apply
        # the legacy solid-neighbour fallback a second time.
        global_thermal_fallback = None
    elif cad_global is not None:
        flow_kind = _flow_type(cad_global)
        if flow_kind in {"velocity", "pressure", "density"}:
            raise ValueError("global CAD flow boundary requires mesh.boundary_links['cad']")

    return local_actions, local_thermal, global_thermal_fallback


def _face_mask_for_active(face: str, shape: tuple[int, int, int], active: np.ndarray) -> np.ndarray:
    """Return one box-face mask using the supplied active-cell array."""

    axis, index = _face_index(face)
    mask = np.zeros(shape, dtype=bool)
    sl = [slice(None)] * 3
    sl[axis] = index
    mask[tuple(sl)] = active[tuple(sl)]
    return mask


def _flow_target(
    flow: Mapping[str, Any],
    current_rho: Any,
    current_u: Any,
    mask: Any,
    equilibrium: Callable[[Any, Any], Any],
    jnp: Any,
    dt: float,
    dx: float,
    pressure_scale: float,
    mach_limit: float,
    acceleration: Any = None,
    force_source: Any = None,
) -> Any:
    def target_equilibrium(rho_target, velocity_target):
        value = equilibrium(rho_target, velocity_target)
        if force_source is not None:
            value = value - 0.5 * force_source(rho_target, velocity_target, acceleration)
        return value

    kind = str(flow.get("type", "no-slip")).lower().replace("_", "-")
    if kind in {"no-slip", "noslip", "stationary", "wall", "none", "adiabatic"}:
        return None
    if kind in {"velocity", "moving-wall", "moving"}:
        velocity = _as_float_vector(flow.get("velocity", ()), "boundary velocity")
        lattice = velocity * dt / dx
        if float(np.linalg.norm(lattice)) / math.sqrt(1.0 / 3.0) >= mach_limit:
            raise ValueError("boundary velocity exceeds the D3Q27 low-Mach limit")
        target_u = jnp.zeros_like(current_u)
        target_u = target_u + jnp.asarray(lattice, dtype=current_u.dtype).reshape((3, 1, 1, 1))
        # Preserve a locally measured density for velocity boundaries.  This
        # is the non-equilibrium extrapolation reference used by the simple
        # workbench boundary and avoids inventing a pressure level.
        return target_equilibrium(current_rho, target_u)
    if kind in {"pressure", "density"}:
        try:
            pressure = float(flow.get("value", flow.get("pressure", 0.0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("pressure boundary requires a numeric value") from exc
        if not math.isfinite(pressure):
            raise ValueError("pressure boundary must be finite")
        rho_value = 1.0 + pressure / pressure_scale
        if rho_value <= 0.0 or abs(rho_value - 1.0) > 0.20:
            raise ValueError("pressure boundary creates an unstable density excursion")
        # Current velocity provides the zero-gradient tangential and normal
        # velocity for an outlet.  It is clipped before equilibrium evaluation
        # so a transient cannot violate the low-Mach guard.
        speed = jnp.sqrt(jnp.sum(current_u * current_u, axis=0, keepdims=True))
        scale = jnp.minimum(1.0, (mach_limit * math.sqrt(1.0 / 3.0)) / jnp.maximum(speed, 1.0e-12))
        target_u = current_u * scale
        target_rho = jnp.full_like(current_rho, rho_value)
        return target_equilibrium(target_rho, target_u)
    raise ValueError(f"unsupported flow boundary type {kind!r}")


def _write_vtk(
    path: Path,
    velocity: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
    fluid: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    solid: np.ndarray | None = None,
    thermal: np.ndarray | None = None,
    eddy_viscosity: np.ndarray | None = None,
) -> None:
    nx, ny, nz = fluid.shape
    if solid is None:
        solid = np.zeros_like(fluid, dtype=bool)
    if thermal is None:
        thermal = fluid
    center = origin + 0.5 * spacing

    def values(array: np.ndarray) -> Iterable[str]:
        # VTK's x index is the fastest changing index; fields in this package
        # are indexed [x, y, z], hence Fortran flattening here.
        return (f"{float(v):.9g}" for v in np.asarray(array).ravel(order="F"))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("XLB workbench D3Q27 result\nASCII\n")
        handle.write(f"DATASET STRUCTURED_POINTS\nDIMENSIONS {nx} {ny} {nz}\n")
        handle.write(f"ORIGIN {center[0]:.12g} {center[1]:.12g} {center[2]:.12g}\n")
        handle.write(f"SPACING {spacing[0]:.12g} {spacing[1]:.12g} {spacing[2]:.12g}\n")
        handle.write(f"POINT_DATA {nx * ny * nz}\n")
        handle.write("SCALARS fluid_mask int 1\nLOOKUP_TABLE default\n")
        handle.writelines(f"{int(v)}\n" for v in fluid.ravel(order="F"))
        handle.write("SCALARS solid_mask int 1\nLOOKUP_TABLE default\n")
        handle.writelines(f"{int(v)}\n" for v in solid.ravel(order="F"))
        handle.write("SCALARS thermal_mask int 1\nLOOKUP_TABLE default\n")
        handle.writelines(f"{int(v)}\n" for v in thermal.ravel(order="F"))
        handle.write("SCALARS pressure float 1\nLOOKUP_TABLE default\n")
        handle.writelines(v + "\n" for v in values(pressure))
        handle.write("SCALARS temperature float 1\nLOOKUP_TABLE default\n")
        handle.writelines(v + "\n" for v in values(temperature))
        handle.write("SCALARS speed float 1\nLOOKUP_TABLE default\n")
        handle.writelines(v + "\n" for v in values(np.linalg.norm(velocity, axis=0)))
        if eddy_viscosity is not None:
            handle.write("SCALARS eddy_viscosity float 1\nLOOKUP_TABLE default\n")
            handle.writelines(v + "\n" for v in values(eddy_viscosity))
        handle.write("VECTORS velocity float\n")
        flat = [velocity[d].ravel(order="F") for d in range(3)]
        handle.writelines(f"{float(flat[0][i]):.9g} {float(flat[1][i]):.9g} {float(flat[2][i]):.9g}\n" for i in range(nx * ny * nz))


_HISTORY_FIELDS = ["step", "time", "progress", "residual", "velocity_residual", "pressure_residual", "temperature_residual",
                   "velocity_change_max", "pressure_change_max", "temperature_change_max", "sample_steps", "sample_time",
                   "monitor_tolerance", "monitor_required_samples", "monitor_consecutive_samples", "monitor_satisfied", "monitor_eligible",
                   "residual_kind", "temperature_min", "temperature_max", "temperature_mean", "speed_mean", "speed_max", "mach_max", "eddy_viscosity_max"]


def _append_history(path: Path, row: Mapping[str, Any], *, initial: bool = False) -> None:
    # Close after each sample so the downloadable CSV survives a stopped or
    # failed calculation and contains the same values as the live monitor.
    with path.open("w" if initial else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS)
        if initial:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in _HISTORY_FIELDS})


def _load_xlb(device: str) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Initialize XLB/JAX and construct the real D3Q27 operators."""

    try:
        import jax
        import jax.numpy as jnp
        import xlb
        from xlb import ComputeBackend, PrecisionPolicy
        from xlb.operator.collision import BGK
        from xlb.operator.equilibrium import QuadraticEquilibrium
        from xlb.operator.macroscopic import Macroscopic
    except ImportError as exc:  # pragma: no cover - native dev machines may lack WSL XLB
        raise RuntimeError("XLB 0.3.2 and JAX are required; run the workbench with /home/tk/src/XLB/.venv/bin/python") from exc

    # Macroscopic/equilibrium contractions must use the same accumulation
    # precision on CPU and CUDA.  Leaving JAX's default matmul policy in place
    # permits reduced precision on some GPUs and produces visible CPU/CUDA
    # drift in long coupled runs.  This is process-wide by design; operators
    # are constructed immediately below and all later jobs share the setting.
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
        # Thermal/material/LES device operators use float64 to retain the
        # reference NumPy path's accumulation semantics.  XLB's precision
        # policy below still creates the lattice state and its operators as
        # explicit FP32 values.
        jax.config.update("jax_enable_x64", True)
    except (AttributeError, RuntimeError, ValueError):  # pragma: no cover - old JAX fallback
        pass

    if device == "cuda:0":
        try:
            devices = jax.devices("gpu")
        except RuntimeError as exc:
            raise RuntimeError("study.device='cuda:0' requested but no CUDA JAX device is available") from exc
        if not devices:
            raise RuntimeError("study.device='cuda:0' requested but no CUDA JAX device is available")
        target_device = devices[0]
    else:
        target_device = jax.devices("cpu")[0]
    policy = PrecisionPolicy.FP32FP32
    # XLB's JAX operators are pure callables, but the velocity set also owns
    # JAX constants.  Constructing it under the requested default device is
    # important when a single server process runs CPU and CUDA jobs in turn.
    with jax.default_device(target_device):
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=policy, compute_backend=ComputeBackend.JAX)
        xlb.init(velocity_set=velocity_set, default_backend=ComputeBackend.JAX, default_precision_policy=policy)
        macro = Macroscopic(velocity_set=velocity_set, precision_policy=policy, compute_backend=ComputeBackend.JAX)
        equilibrium = QuadraticEquilibrium(velocity_set=velocity_set, precision_policy=policy, compute_backend=ComputeBackend.JAX)
        collision = BGK(velocity_set=velocity_set, precision_policy=policy, compute_backend=ComputeBackend.JAX)
    return jax, jnp, target_device, velocity_set, macro, equilibrium, collision


def _compute_backend(device: str) -> str:
    """Select the host reference or JAX-resident coupled compute path.

    CPU remains on the reference path for ``auto`` so existing numerical
    results remain reproducible.  CUDA selects the device path automatically;
    ``XLB_COMPUTE_BACKEND=device`` can force the same path on CPU for an
    apples-to-apples comparison.  The older one-bit switch is retained for
    private benchmarks, but an explicit backend always wins.
    """

    # The explicit RAM choice must never be promoted to the all-device path
    # by a server-wide benchmark override: that would defeat its memory bound.
    if device == "cuda:0-ram":
        return "ram"
    requested = os.environ.get("XLB_COMPUTE_BACKEND")
    if requested is None:
        legacy = os.environ.get("XLB_DEVICE_PIPELINE", "").strip().lower()
        requested = "device" if legacy in {"1", "true", "yes", "on"} else "auto"
    requested = str(requested).strip().lower().replace("_", "-")
    if requested == "auto":
        return "device" if device == "cuda:0" else "reference"
    if requested in {"reference", "ref", "numpy", "host"}:
        return "reference"
    if requested in {"device", "jax", "gpu", "accelerated"}:
        return "device"
    raise ValueError("XLB_COMPUTE_BACKEND must be 'auto', 'reference', or 'device'")


def _device_block(jax: Any, value: Any) -> Any:
    """Synchronize a JAX value/tree without converting it to a host array."""

    try:
        return jax.tree_util.tree_map(
            lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item,
            value,
        )
    except Exception:  # pragma: no cover - compatibility with older JAX trees
        if hasattr(value, "block_until_ready"):
            return value.block_until_ready()
        return value


def _device_scalar(jax: Any, value: Any) -> float:
    """Read one scalar reduction from a device, rejecting accidental fields."""

    result = jax.device_get(value)
    array = np.asarray(result)
    if array.size != 1:
        raise RuntimeError("device scalar reduction unexpectedly returned a full field")
    return float(array.reshape(-1)[0])


def _device_bool(jax: Any, value: Any) -> bool:
    return bool(_device_scalar(jax, value))


def _device_factories(
    *,
    materials: list[Mapping[str, Any]],
    material_index: np.ndarray,
    fluid: np.ndarray,
    thermal: np.ndarray,
    dx: float,
    dt: float,
    box_thermal: Mapping[str, Mapping[str, Any]],
    cad_thermal: Mapping[str, Any] | None,
    thermal_face_masks: Mapping[str, np.ndarray],
    cad_thermal_actions: Iterable[tuple[np.ndarray, Mapping[str, Any]]],
    turbulence_model: str,
    smagorinsky_constant: float,
    jax: Any,
    jnp: Any,
) -> tuple[Callable[..., Any], Callable[..., Any] | None, Callable[..., Any] | None, Callable[..., Any] | None]:
    """Load and JIT the device-side property, LES, and thermal factories.

    Device mode intentionally fails clearly when the factories are missing;
    silently switching a requested CUDA run back to host computation would
    defeat the memory/synchronization contract.
    """

    try:
        thermal_module = importlib.import_module("workbench.thermal_jax")
        material_module = importlib.import_module("workbench.fields_jax")
    except ImportError as exc:
        raise RuntimeError(
            "device compute requires workbench.thermal_jax and workbench.fields_jax"
        ) from exc
    make_thermal = getattr(thermal_module, "make_thermal_operators", None)
    make_material = getattr(material_module, "make_material_operator", None)
    if not callable(make_thermal) or not callable(make_material):
        raise RuntimeError("device compute factories are incomplete: material and thermal operators are required")

    # Keep the index compact when it is captured by a factory closure.  The
    # canonical mesh representation and output remain unchanged.
    material_index_arg = np.asarray(material_index, dtype=np.int32)
    material_factory = make_material(
        materials,
        material_index_arg,
        np.asarray(thermal, dtype=bool),
        jax=jax,
        jnp=jnp,
    )
    material_eval = getattr(material_factory, "evaluate", material_factory)
    if not callable(material_eval):
        raise RuntimeError("make_material_operator must return a callable or an object with evaluate(T)")
    material_eval = jax.jit(material_eval)

    thermal_factory = make_thermal(
        np.asarray(fluid, dtype=bool),
        np.asarray(thermal, dtype=bool),
        float(dx),
        float(dt),
        box_thermal,
        cad_thermal,
        thermal_face_masks,
        tuple(cad_thermal_actions),
        jax=jax,
        jnp=jnp,
    )
    if not isinstance(thermal_factory, (tuple, list)) or len(thermal_factory) != 2:
        raise RuntimeError("make_thermal_operators must return (step, stability)")
    thermal_step, thermal_stability = thermal_factory
    if not callable(thermal_step) or not callable(thermal_stability):
        raise RuntimeError("thermal device factory returned non-callable operators")
    thermal_step = jax.jit(thermal_step)
    thermal_stability = jax.jit(thermal_stability)

    les_operator: Callable[..., Any] | None = None
    if turbulence_model == "smagorinsky":
        make_les = getattr(material_module, "make_les_operator", None)
        if not callable(make_les):
            # The LES factory is intentionally in fields_jax with the
            # material evaluator so both operators share array conventions.
            raise RuntimeError("device Smagorinsky LES requires make_les_operator")
        les_operator = make_les(
            np.asarray(fluid, dtype=bool),
            float(dx),
            float(smagorinsky_constant),
            jax=jax,
            jnp=jnp,
        )
        les_operator = getattr(les_operator, "evaluate", les_operator)
        if not callable(les_operator):
            raise RuntimeError("make_les_operator must return a callable")
        les_operator = jax.jit(les_operator)
    return material_eval, les_operator, thermal_step, thermal_stability


def _device_flow_operator(
    *,
    macro: Callable[..., Any],
    equilibrium: Callable[..., Any],
    collision: Callable[..., Any],
    box_flow_items: list[tuple[int, Mapping[str, Any]]],
    cad_flow_items: list[Mapping[str, Any]],
    dt: float,
    dx: float,
    pressure_scale: float,
    jnp: Any,
    jax: Any,
    force_source: Any = None,
) -> Callable[..., Any]:
    """Build one jitted XLB collide/stream/boundary update.

    Grid-sized masks are dynamic operands rather than closure constants.  This
    keeps XLA from embedding a second copy of a multi-million-cell mesh in
    the compiled executable.
    """

    def update(f: Any, rho: Any, velocity: Any, omega: Any, fluid: Any, source_valid: Any, face_masks: Any, cad_masks: Any, acceleration: Any = None) -> tuple[Any, Any, Any]:
        rho_current, velocity_current = macro(f, rho, velocity)
        if force_source is not None:
            velocity_current = velocity_current + 0.5 * acceleration
        feq = equilibrium(rho_current, velocity_current)
        f_post = collision(f, feq, omega)
        if force_source is not None:
            f_post = f_post + (1.0 - 0.5 * omega) * force_source(rho_current, velocity_current, acceleration)
        f_next = _stream_jax(f_post, fluid, source_valid, jnp)
        for face_index, flow in box_flow_items:
            target = _flow_target(
                flow,
                rho_current,
                velocity_current,
                face_masks[face_index],
                equilibrium,
                jnp,
                dt,
                dx,
                pressure_scale,
                0.30,
                acceleration,
                force_source,
            )
            if target is not None:
                f_next = jnp.where(face_masks[face_index][None, ...], target, f_next)
        for cad_index, flow in enumerate(cad_flow_items):
            target = _flow_target(
                flow,
                rho_current,
                velocity_current,
                cad_masks[cad_index],
                equilibrium,
                jnp,
                dt,
                dx,
                pressure_scale,
                0.30,
                acceleration,
                force_source,
            )
            if target is not None:
                f_next = jnp.where(cad_masks[cad_index][None, ...], target, f_next)
        rho_next, velocity_next = macro(f_next, rho_current, velocity_current)
        if force_source is not None:
            velocity_next = velocity_next + 0.5 * acceleration
        return f_next, rho_next, velocity_next

    return jax.jit(update)


def _force_boundary_refresh(*, box_items, cad_items, equilibrium, force_source, dt, dx, pressure_scale, jax, jnp):
    """Keep prescribed physical velocities fixed when thermal buoyancy changes."""
    def refresh(f, rho, velocity, acceleration, face_masks, cad_masks):
        for index, flow in box_items:
            if str(flow.get('type', '')).lower().replace('_', '-') not in {'velocity', 'moving-wall', 'moving'}:
                continue
            target = _flow_target(flow, rho, velocity, face_masks[index], equilibrium, jnp,
                                  dt, dx, pressure_scale, .30, acceleration, force_source)
            f = jnp.where(face_masks[index][None, ...], target, f)
        for index, flow in enumerate(cad_items):
            if str(flow.get('type', '')).lower().replace('_', '-') not in {'velocity', 'moving-wall', 'moving'}:
                continue
            target = _flow_target(flow, rho, velocity, cad_masks[index], equilibrium, jnp,
                                  dt, dx, pressure_scale, .30, acceleration, force_source)
            f = jnp.where(cad_masks[index][None, ...], target, f)
        return f
    return jax.jit(refresh)


def _device_rest_operator(equilibrium: Callable[..., Any], jax: Any, jnp: Any) -> Callable[..., Any]:
    """JIT the zero-flow state update used by a thermal-only device run."""

    def update(rho: Any, velocity: Any) -> tuple[Any, Any, Any]:
        zero = jnp.zeros_like(velocity)
        return equilibrium(rho, zero), rho, zero

    return jax.jit(update)


def _simulate_device(
    *,
    project: Mapping[str, Any],
    mesh: Mapping[str, Any],
    shape: tuple[int, int, int],
    fluid_np: np.ndarray,
    solid_np: np.ndarray,
    thermal_np: np.ndarray,
    material_index_np: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    materials: list[Mapping[str, Any]],
    initial_temperature: float,
    flow_enabled: bool,
    thermal_enabled: bool,
    turbulence_model: str,
    smagorinsky_constant: float,
    turbulent_prandtl: float,
    dt: float,
    dx: float,
    device: str,
    rho_reference: float,
    pressure_scale: float,
    max_boundary_mach: float,
    box_boundaries: Mapping[str, Mapping[str, Any]],
    thermal_by_face: Mapping[str, Mapping[str, Any]],
    cad_thermal: Mapping[str, Any] | None,
    cad_flow_actions: list[tuple[np.ndarray, Mapping[str, Any]]],
    cad_thermal_actions: list[tuple[np.ndarray, Mapping[str, Any]]],
    flow_face_masks: Mapping[str, np.ndarray],
    thermal_face_masks: Mapping[str, np.ndarray],
    warnings: list[str],
    run_path: Path,
    on_progress: Callable[[dict[str, Any]], Any] | None,
    should_stop: Callable[[], bool] | None,
    steps: int,
    output_interval: int,
    start_step: int,
    restart_temperature: np.ndarray | None,
    jax: Any,
    jnp: Any,
    target_device: Any,
    macro: Callable[..., Any],
    equilibrium: Callable[..., Any],
    collision: Callable[..., Any],
    fluid: Any,
    source_valid: Any,
    rho: Any,
    velocity: Any,
    f: Any,
    gravity_model: Any = None,
    force_source: Any = None,
) -> dict[str, Any]:
    """Run the coupled fields with all mutable fields resident on JAX.

    The NumPy implementation in :func:`simulate` remains the reference
    backend. This path transfers full fields only at requested snapshots and
    finalization; other iterations transfer only scalar reductions.
    """

    run_path.mkdir(parents=True, exist_ok=True)
    with jax.default_device(target_device):
        # Thermal/material operators create their static geometry arrays on
        # the selected device.  Missing factories are a hard error in this
        # explicitly requested path; a CUDA run must never silently become a
        # host run.
        material_eval, les_operator, thermal_step, thermal_stability = _device_factories(
            materials=materials,
            material_index=material_index_np,
            fluid=fluid_np,
            thermal=thermal_np,
            dx=dx,
            dt=dt,
            box_thermal=thermal_by_face,
            cad_thermal=cad_thermal,
            thermal_face_masks=thermal_face_masks,
            cad_thermal_actions=cad_thermal_actions,
            turbulence_model=turbulence_model,
            smagorinsky_constant=smagorinsky_constant,
            jax=jax,
            jnp=jnp,
        )

        if restart_temperature is None:
            temperature = jnp.full(shape, initial_temperature, dtype=jnp.float64)
        else:
            # ``load_restart`` has already checked shape, dtype, and finite
            # values before this host array is copied to the target device.
            temperature = jnp.asarray(restart_temperature, dtype=jnp.float64)
        fluid_device = jnp.asarray(fluid_np, dtype=bool)
        thermal_device = jnp.asarray(thermal_np, dtype=bool)
        material_index_device = jnp.asarray(material_index_np, dtype=jnp.int32)
        # ``material_index_device`` is retained to make the fixed mesh
        # ownership explicit; the factory owns its compact captured copy.
        del material_index_device
        gravity_operator = jax.jit(gravity_model.make_operator(jnp)) if force_source is not None else None

        # Flow masks are dynamic operands to the fused update, rather than
        # closure constants.  Empty CAD lists use a zero-length leading axis,
        # which remains a valid static JAX shape.
        face_stack = jnp.stack([jnp.asarray(flow_face_masks[face], dtype=bool) for face in _FACES], axis=0)
        if cad_flow_actions:
            cad_stack = jnp.stack([jnp.asarray(mask, dtype=bool) for mask, _ in cad_flow_actions], axis=0)
        else:
            cad_stack = jnp.zeros((0, *shape), dtype=bool)
        box_flow_items: list[tuple[int, Mapping[str, Any]]] = []
        valid_flow_types = {"velocity", "moving-wall", "moving", "pressure", "density"}
        for face, spec in box_boundaries.items():
            flow = spec.get("flow")
            if isinstance(flow, Mapping) and _flow_type(spec) in valid_flow_types:
                box_flow_items.append((_FACES.index(face), flow))
        cad_flow_items = [flow for _, flow in cad_flow_actions]
        refresh_force_boundaries = _force_boundary_refresh(
            box_items=box_flow_items, cad_items=cad_flow_items, equilibrium=equilibrium,
            force_source=force_source, dt=dt, dx=dx, pressure_scale=pressure_scale, jax=jax, jnp=jnp,
        ) if force_source is not None and gravity_model.mode == 'buoyancy' and thermal_enabled else None
        if flow_enabled:
            flow_step = _device_flow_operator(
                macro=macro,
                equilibrium=equilibrium,
                collision=collision,
                box_flow_items=box_flow_items,
                cad_flow_items=cad_flow_items,
                dt=dt,
                dx=dx,
                pressure_scale=pressure_scale,
                jnp=jnp,
                jax=jax,
                force_source=force_source,
            )
        else:
            flow_step = None
            rest_step = _device_rest_operator(equilibrium, jax, jnp)

    # The scalar helper is deliberately the only host read used by the hot
    # loop.  It rejects an accidental field conversion and makes the transfer
    # count in diagnostics auditable.
    scalar_transfers = 0

    def scalar(value: Any) -> float:
        nonlocal scalar_transfers
        scalar_transfers += 1
        return _device_scalar(jax, value)

    def scalar_bool(value: Any) -> bool:
        return bool(scalar(value))

    def acceleration_at(current_temperature):
        if gravity_operator is None:
            return None
        acceleration, fraction = gravity_operator(current_temperature, fluid_device)
        gravity_model.check(scalar(fraction))
        return acceleration

    def physical_macro(current_f, current_rho, current_velocity):
        r, u = macro(current_f, current_rho, current_velocity)
        if force_source is not None:
            u = u + 0.5 * acceleration_at(temperature)
        return r, u

    def actual_device(value: Any) -> str:
        try:
            actual = getattr(value, "device", None)
            if callable(actual):
                actual = actual()
            if actual is None and hasattr(value, "devices"):
                devices = value.devices()
                actual = next(iter(devices), None)
            if isinstance(actual, (set, frozenset, tuple, list)):
                actual = next(iter(actual), None)
            return str(actual if actual is not None else target_device)
        except Exception:  # pragma: no cover - JAX version compatibility
            return str(target_device)

    def evaluate_properties(current_temperature: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Evaluate material fields and validate scalar stability on device."""

        result = material_eval(current_temperature)
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise RuntimeError("material device evaluator must return (density, viscosity, cp, conductivity, clamped)")
        density_d, viscosity_d, cp_d, conductivity_d, clamped = result
        density_d = jnp.asarray(density_d, dtype=jnp.float64)
        viscosity_d = jnp.asarray(viscosity_d, dtype=jnp.float64)
        cp_d = jnp.asarray(cp_d, dtype=jnp.float64)
        conductivity_d = jnp.asarray(conductivity_d, dtype=jnp.float64)
        clamped_d = jnp.asarray(clamped, dtype=bool)
        finite = jnp.all(
            jnp.isfinite(density_d)
            & jnp.isfinite(viscosity_d)
            & jnp.isfinite(cp_d)
            & jnp.isfinite(conductivity_d)
        )
        positive = jnp.all(
            jnp.where(
                thermal_device,
                (density_d > 0.0) & (viscosity_d > 0.0) & (cp_d > 0.0) & (conductivity_d > 0.0),
                True,
            )
        )
        if not scalar_bool(finite):
            raise RuntimeError("device material evaluator produced non-finite values")
        if not scalar_bool(positive):
            raise ValueError("temperature-dependent material properties must stay positive")
        if scalar_bool(jnp.any(clamped_d)):
            message = "temperature-dependent material property table was clamped to an endpoint on the device"
            if message not in warnings:
                warnings.append(message)
        tau_d = 0.5 + (viscosity_d / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
        if flow_enabled:
            tau_finite = scalar_bool(jnp.all(jnp.where(fluid_device, jnp.isfinite(tau_d), True)))
            tau_min = scalar(jnp.min(jnp.where(fluid_device, tau_d, jnp.inf)))
            tau_max = scalar(jnp.max(jnp.where(fluid_device, tau_d, -jnp.inf)))
            if not tau_finite or tau_min <= 0.500001:
                raise ValueError("temperature-dependent viscosity produced an unstable relaxation time tau <= 0.5")
            if tau_max > 5.0 and "temperature-dependent relaxation time is high; BGK accuracy may be poor" not in warnings:
                warnings.append("temperature-dependent relaxation time is high; BGK accuracy may be poor")
        return density_d, viscosity_d, cp_d, conductivity_d, tau_d, clamped_d

    def stability_numbers(density_d: Any, cp_d: Any, conductivity_d: Any) -> tuple[float, float, float]:
        if thermal_stability is None:
            return 0.0, 0.0, 0.0
        result = thermal_stability(density_d, cp_d, conductivity_d)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError("thermal stability operator must return (fourier, robin, loss)")
        values = tuple(scalar(value) for value in result)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("device thermal stability reduction was non-finite")
        return values  # type: ignore[return-value]

    # Initialize material fields before the first step.  This call also
    # compiles the property operator and establishes the initial thermal
    # stability guard before any XLB state is advanced.
    with jax.default_device(target_device):
        density, viscosity, cp, conductivity, tau_molecular = evaluate_properties(temperature)[:5]
        if thermal_enabled:
            thermal_number, robin_number, thermal_loss_number = stability_numbers(density, cp, conductivity)
            if thermal_loss_number > 0.95:
                raise ValueError("explicit thermal update is unstable for this dt and mesh (CFL/diffusion/wall loss sum > 0.95)")
        else:
            thermal_number = robin_number = thermal_loss_number = 0.0

    history: deque[dict[str, Any]] = deque(maxlen=2)
    last_velocity: Any | None = None
    last_temperature: Any | None = None
    last_rho: Any | None = None
    monitor = ConvergenceMonitor(**project['study'].get('monitor', {}), interval=output_interval, dt=dt,
                                 flow_enabled=flow_enabled, thermal_enabled=thermal_enabled)
    with jax.default_device(target_device):
        residual_operator = jax.jit(make_residual_operator(jnp, initial_temperature=initial_temperature,
            velocity_scale=dx / dt, pressure_scale=pressure_scale, flow_enabled=flow_enabled, thermal_enabled=thermal_enabled))
    final_step = int(start_step)
    stopped = False
    max_mach_seen = float(max_boundary_mach)
    eddy_viscosity = jnp.zeros(shape, dtype=jnp.float64)
    turbulent_conductivity = jnp.zeros(shape, dtype=jnp.float64)
    full_field_transfers = 0
    compilation_seconds = 0.0
    warm_compute_seconds = 0.0
    first_device_step = True
    snapshots = SnapshotWriter(run_path, project['study'].get('snapshot_interval', 0), dt,
                               fluid_mask=fluid_np, solid_mask=solid_np, thermal_mask=thermal_np,
                               material_index=material_index_np, origin=origin, spacing=spacing)
    snapshot_seconds = 0.0

    def checkpoint_device(step: int) -> None:
        nonlocal full_field_transfers, snapshot_seconds
        started = time.perf_counter()
        with jax.default_device(target_device):
            velocity_si = jnp.where(fluid_device[None, ...], velocity, 0.0).astype(jnp.float64) * (dx / dt)
            if not flow_enabled:
                velocity_si = jnp.zeros_like(velocity_si)
            eddy = les_operator(velocity_si) if flow_enabled and turbulence_model == 'smagorinsky' else jnp.zeros(shape)
            pressure = jnp.where(fluid_device, (rho[0] - 1.0) * pressure_scale, 0.0)
            fields = jax.device_get(tuple(jnp.asarray(value, dtype=jnp.float32)
                                          for value in (velocity_si, pressure, temperature, eddy)))
        full_field_transfers += 1
        snapshots.write(step, velocity=fields[0], pressure=fields[1], temperature=fields[2], eddy_viscosity=fields[3])
        snapshot_seconds += time.perf_counter() - started

    def record_device(step: int) -> None:
        nonlocal last_velocity, last_temperature, last_rho, max_mach_seen, scalar_transfers
        rho_field = rho[0]
        velocity_field = jnp.where(fluid_device[None, ...], velocity, 0.0)
        finite = jnp.all(
            jnp.where(fluid_device, jnp.isfinite(rho_field), True)
        ) & jnp.all(jnp.isfinite(velocity_field))
        positive = jnp.all(jnp.where(fluid_device, rho_field > 0.0, True))
        if not scalar_bool(finite) or not scalar_bool(positive):
            raise RuntimeError("LBM produced a non-finite or non-positive density/velocity")
        speed_lbm = jnp.sqrt(jnp.sum(velocity_field * velocity_field, axis=0))
        max_mach = scalar(jnp.max(jnp.where(fluid_device, speed_lbm / math.sqrt(1.0 / 3.0), 0.0)))
        if max_mach >= 0.30:
            raise RuntimeError(f"LBM Mach number {max_mach:g} exceeded the low-Mach limit 0.30")
        max_mach_seen = max(max_mach_seen, max_mach)
        values = None
        if last_velocity is not None:
            with jax.default_device(target_device):
                reduced = residual_operator(velocity_field, rho, temperature, last_velocity, last_rho, last_temperature,
                                            fluid_device, thermal_device)
                values = np.asarray(jax.device_get(reduced))
            scalar_transfers += 6
        monitor_row = monitor.update(step, values)
        velocity_phys = jnp.asarray(velocity, dtype=jnp.float64) * (dx / dt)
        speed = jnp.sqrt(jnp.sum(velocity_phys * velocity_phys, axis=0))
        progress_span = steps - start_step
        progress_value = 1.0 if progress_span == 0 else float((step - start_step) / progress_span)
        row: dict[str, Any] = {
            "step": int(step),
            "progress": progress_value,
            **monitor_row,
            "temperature_min": scalar(jnp.min(jnp.where(thermal_device, temperature, jnp.inf))),
            "temperature_max": scalar(jnp.max(jnp.where(thermal_device, temperature, -jnp.inf))),
            "temperature_mean": scalar(jnp.sum(jnp.where(thermal_device, temperature, 0.0)) / jnp.maximum(jnp.sum(thermal_device), 1)),
            "speed_mean": scalar(jnp.sum(jnp.where(fluid_device, speed, 0.0)) / jnp.maximum(jnp.sum(fluid_device), 1)),
            "speed_max": scalar(jnp.max(jnp.where(fluid_device, speed, 0.0))),
            "mach_max": float(max_mach),
            "eddy_viscosity_max": scalar(jnp.max(jnp.where(fluid_device, eddy_viscosity, 0.0))) if flow_enabled else 0.0,
        }
        history.append(row)
        _append_history(run_path / 'history.csv', row, initial=len(history) == 1)
        if on_progress is not None:
            on_progress(dict(row))
        # These are device references, not host snapshots.  JAX arrays are
        # immutable, so the next update cannot mutate the previous sample.
        last_velocity = velocity_field
        last_temperature = temperature
        last_rho = rho

    # Emit an actual initial state and honor a stop request before stepping.
    # For a resumed run this is the checkpoint's absolute physical step.
    if force_source is not None:
        with jax.default_device(target_device):
            rho, velocity = physical_macro(f, rho, velocity)
    record_device(start_step)
    if should_stop is not None and should_stop():
        stopped = True
    else:
        if snapshots.interval:
            checkpoint_device(start_step)
        for step in range(start_step + 1, steps + 1):
            started = time.perf_counter()
            with jax.default_device(target_device):
                density, viscosity, cp, conductivity, tau_molecular = evaluate_properties(temperature)[:5]
                velocity_masked = jnp.where(fluid_device[None, ...], velocity, 0.0)
                velocity_phys_pre = jnp.asarray(velocity_masked, dtype=jnp.float64) * (dx / dt)
                if flow_enabled and turbulence_model == "smagorinsky":
                    if les_operator is None:  # pragma: no cover - factory guard
                        raise RuntimeError("device Smagorinsky LES operator is unavailable")
                    eddy_viscosity = les_operator(velocity_phys_pre)
                    eddy_viscosity = jnp.asarray(eddy_viscosity, dtype=jnp.float64)
                else:
                    eddy_viscosity = jnp.zeros(shape, dtype=jnp.float64)
                turbulent_conductivity = density * cp * eddy_viscosity / turbulent_prandtl
                if turbulence_model != "smagorinsky":
                    turbulent_conductivity = jnp.zeros_like(turbulent_conductivity)
                conductivity_transport = conductivity + turbulent_conductivity

                if flow_enabled:
                    viscosity_effective = viscosity + rho_reference * eddy_viscosity
                    tau = 0.5 + (viscosity_effective / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
                    tau_finite = scalar_bool(jnp.all(jnp.where(fluid_device, jnp.isfinite(tau), True)))
                    tau_min = scalar(jnp.min(jnp.where(fluid_device, tau, jnp.inf)))
                    tau_max = scalar(jnp.max(jnp.where(fluid_device, tau, -jnp.inf)))
                    if not tau_finite or tau_min <= 0.500001:
                        raise ValueError("effective Smagorinsky viscosity produced an unstable relaxation time tau <= 0.5")
                    if tau_max > 5.0 and "effective Smagorinsky relaxation time is high; BGK accuracy may be poor" not in warnings:
                        warnings.append("effective Smagorinsky relaxation time is high; BGK accuracy may be poor")
                else:
                    tau = tau_molecular

                if thermal_enabled:
                    thermal_number, robin_number, thermal_loss_number = stability_numbers(density, cp, conductivity_transport)
                    if thermal_loss_number > 0.95:
                        raise ValueError("temperature-dependent thermal diffusivity made the explicit update unstable")

                if flow_enabled:
                    f, rho, velocity = flow_step(
                        f,
                        rho,
                        velocity,
                        jnp.asarray(1.0 / tau, dtype=jnp.float32),
                        fluid,
                        source_valid,
                        face_stack,
                        cad_stack,
                        acceleration_at(temperature),
                    )
                else:
                    # A flow-disabled run has no lattice evolution.  Keep the
                    # actual distribution byte-for-byte stable across a
                    # restart; rebuilding equilibrium from a reduced rho
                    # field can introduce tiny rounding differences.
                    velocity = jnp.zeros_like(velocity)

                velocity_masked = jnp.where(fluid_device[None, ...], velocity, 0.0)
                velocity_phys = jnp.asarray(velocity_masked, dtype=jnp.float64) * (dx / dt)
                if thermal_enabled:
                    cfl = scalar(jnp.max(jnp.where(fluid_device, jnp.sum(jnp.abs(velocity_phys), axis=0) * dt / dx, 0.0)))
                    if not math.isfinite(cfl) or cfl >= 1.0 or cfl + thermal_loss_number > 0.95:
                        raise RuntimeError("thermal advection/diffusion CFL stability limit exceeded")
                    temperature = thermal_step(temperature, velocity_phys, density, cp, conductivity_transport)
                if force_source is not None:
                    # Report physical velocity at the new temperature/force,
                    # and use it for the next LES and thermal update.
                    rho, velocity = physical_macro(f, rho, velocity)
                    if refresh_force_boundaries is not None:
                        f = refresh_force_boundaries(f, rho, velocity, acceleration_at(temperature), face_stack, cad_stack)
                        rho, velocity = physical_macro(f, rho, velocity)
                # Synchronize only a scalar-shaped reduction to make timing
                # honest and prevent asynchronous device work from leaking
                # into the next iteration's timing bucket.
                _device_block(jax, (f, temperature))
            elapsed = time.perf_counter() - started
            if first_device_step:
                compilation_seconds += elapsed
                first_device_step = False
            else:
                warm_compute_seconds += elapsed
            if not scalar_bool(jnp.all(jnp.isfinite(temperature))):
                raise RuntimeError("thermal field became non-finite")

            final_step = step
            if snapshots.due(step) and step != steps:
                checkpoint_device(step)
            if (step - start_step) % output_interval == 0 or step == steps:
                record_device(step)
            if should_stop is not None and should_stop():
                stopped = True
                break

    # A stop between output intervals still gets a scalar-only final sample.
    if not history or history[-1]["step"] != final_step:
        record_device(final_step)

    # Recompute final material/LES fields on-device, then download all fields
    # in one batched call. Without snapshots this is the only full-field
    # transfer; scalar reductions above intentionally remain separate.
    save_started = time.perf_counter()
    with jax.default_device(target_device):
        rho, velocity = physical_macro(f, rho, velocity)
        if not flow_enabled:
            # The rest equilibrium has only floating-point roundoff momentum
            # after a macroscopic reduction.  Keep the public thermal-only
            # velocity field exactly zero, matching the physical model and
            # the reference branch's explicit flow-disabled state.
            velocity = jnp.zeros_like(velocity)
        density_final, viscosity_final, cp_final, conductivity_final, tau_molecular_final = evaluate_properties(temperature)[:5]
        velocity_masked_final = jnp.where(fluid_device[None, ...], velocity, 0.0)
        velocity_phys_final = jnp.asarray(velocity_masked_final, dtype=jnp.float64) * (dx / dt)
        if flow_enabled and turbulence_model == "smagorinsky":
            if les_operator is None:  # pragma: no cover - factory guard
                raise RuntimeError("device Smagorinsky LES operator is unavailable")
            eddy_final = jnp.asarray(les_operator(velocity_phys_final), dtype=jnp.float64)
            tau_final = 0.5 + ((viscosity_final + rho_reference * eddy_final) / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
        else:
            eddy_final = jnp.zeros(shape, dtype=jnp.float64)
            tau_final = tau_molecular_final
        turbulent_conductivity_final = density_final * cp_final * eddy_final / turbulent_prandtl
        if turbulence_model != "smagorinsky":
            turbulent_conductivity_final = jnp.zeros_like(turbulent_conductivity_final)
        tau_min_final = scalar(jnp.min(jnp.where(fluid_device, tau_final, jnp.inf)))
        tau_max_final = scalar(jnp.max(jnp.where(fluid_device, tau_final, -jnp.inf)))
        if flow_enabled and (not math.isfinite(tau_min_final) or tau_min_final <= 0.500001):
            raise RuntimeError("final effective viscosity produced an unstable relaxation time tau <= 0.5")
        _device_block(jax, (f, rho, velocity, temperature, eddy_final))
        full_field_transfers += 1
        f_host, rho_host, velocity_host, temperature_host, eddy_host = jax.device_get(
            (f, rho, velocity, temperature, eddy_final)
        )

    rho_np = np.asarray(rho_host)[0]
    velocity_lbm_np = np.asarray(velocity_host, dtype=float)
    velocity_lbm_np[:, ~fluid_np] = 0.0
    velocity_phys_np = velocity_lbm_np * dx / dt
    pressure_np = (rho_np - 1.0) * pressure_scale
    pressure_np[~fluid_np] = 0.0
    temperature_np = np.asarray(temperature_host, dtype=float)
    eddy_viscosity_np = np.asarray(eddy_host, dtype=float)
    if not np.isfinite(pressure_np[fluid_np]).all() or not np.isfinite(temperature_np[thermal_np]).all():
        raise RuntimeError("final device fields became non-finite")
    snapshots.write(final_step, velocity=velocity_phys_np, pressure=pressure_np,
                    temperature=temperature_np, eddy_viscosity=eddy_viscosity_np)
    np.savez_compressed(
        run_path / "fields.npz",
        velocity=velocity_phys_np.astype(np.float32),
        pressure=pressure_np.astype(np.float32),
        temperature=temperature_np.astype(np.float32),
        eddy_viscosity=eddy_viscosity_np.astype(np.float32),
        fluid_mask=fluid_np,
        solid_mask=solid_np,
        thermal_mask=thermal_np,
        material_index=material_index_np.astype(np.int32),
        origin=origin.astype(float),
        spacing=spacing.astype(float),
    )
    _write_vtk(
        run_path / "fields.vtk",
        velocity_phys_np,
        pressure_np,
        temperature_np,
        fluid_np,
        origin,
        spacing,
        solid=solid_np,
        thermal=thermal_np,
        eddy_viscosity=eddy_viscosity_np,
    )
    write_restart(
        run_path / "restart.npz",
        step=final_step,
        dt=dt,
        f=np.asarray(f_host),
        temperature=np.asarray(temperature_host, dtype=np.float64),
        project=project,
        mesh=mesh,
        status="stopped" if stopped else "completed",
    )
    save_seconds = time.perf_counter() - save_started

    converged = bool((not stopped) and monitor.satisfied)
    if turbulence_model == "smagorinsky":
        warnings.append("Smagorinsky LES uses cell-centred gradients, voxel walls, and first-order thermal advection; short runs are not a turbulence benchmark")
    if thermal_enabled:
        warnings.append("thermal advection/diffusion uses an explicit JAX finite-volume update on the selected device")
    if flow_enabled:
        les_backend = "JAX Smagorinsky device operator" if turbulence_model == "smagorinsky" else "disabled (laminar)"
    else:
        les_backend = "disabled (flow disabled)"
    diagnostics = {
        "backend": "XLB D3Q27 BGK via JAX",
        "gravity": gravity_model.diagnostics() if gravity_model is not None else {"enabled": False},
        "pressure_kind": "reduced" if force_source is not None and gravity_model.mode == "buoyancy" else "gauge",
        "compute_backend": "device",
        "convergence_monitor": monitor.diagnostics(),
        "device": device,
        "device_actual": actual_device(f),
        "jax_matmul_precision": "highest",
        "velocity_set": "D3Q27",
        "collision": "BGK",
        "tau_min": float(tau_min_final),
        "tau_max": float(tau_max_final),
        "mach_max": float(max_mach_seen),
        "thermal_fourier_max": float(thermal_number),
        "thermal_robin_loss_max": float(robin_number),
        "thermal_loss_coefficient_max": float(thermal_loss_number),
        "temperature_residual": history[-1].get("temperature_residual"),
        "eddy_viscosity_max_m2_s": float(scalar(jnp.max(jnp.where(fluid_device, eddy_final, 0.0)))) if flow_enabled else 0.0,
        "turbulent_conductivity_max_w_m_k": float(scalar(jnp.max(jnp.where(thermal_device, turbulent_conductivity_final, 0.0)))) if thermal_enabled else 0.0,
        "turbulence_model": turbulence_model,
        "smagorinsky_constant": float(smagorinsky_constant),
        "turbulent_prandtl": float(turbulent_prandtl),
        "flow_cells": int(np.count_nonzero(fluid_np)),
        "solid_cells": int(np.count_nonzero(solid_np)),
        "thermal_cells": int(np.count_nonzero(thermal_np)),
        "rho_reference_kg_m3": float(rho_reference),
        "density_model": "reference-density incompressible LBM; local density(T) is used in thermal storage/diffusion coefficients and, when enabled, buoyancy",
        "thermal_backend": "JAX finite-volume on device" if thermal_enabled else "disabled",
        "les_backend": les_backend,
        "property_backend": "JAX material evaluator on device",
        "full_field_transfers": int(full_field_transfers),
        "snapshot_count": len(snapshots.frames),
        "snapshot_bytes": snapshots.bytes_written,
        "device_scalar_transfers": int(scalar_transfers),
        "timings": {
            # JAX compilation is lazy and the first timed iteration includes
            # both compilation and its execution.  Keep that distinction
            # explicit instead of labelling the whole interval as compile
            # time.
            "first_step_s": float(compilation_seconds),
            "warm_compute_s": float(warm_compute_seconds),
            "save_s": float(save_seconds),
            "snapshots_s": float(snapshot_seconds),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
    return {
        "status": "stopped" if stopped else "completed",
        "steps": int(final_step),
        "converged": bool(converged),
        "diagnostics": diagnostics,
    }


def simulate(
    project: Mapping[str, Any],
    mesh: Mapping[str, Any],
    run_dir: str | Path,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    restart_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run a real D3Q27 BGK/thermal simulation and write workbench artifacts.

    Parameters follow ``CONTRACT.md``.  Invalid physical or lattice settings
    raise ``ValueError`` before stepping.  Runtime numerical failures raise a
    ``RuntimeError`` and never return fabricated fields.
    """

    shape, fluid_np, solid_np, thermal_np, material_index_np, origin, spacing, boundary_links = _mesh_arrays(mesh)
    steps, output_interval, dt, device = _study_values(project)
    restart_state: dict[str, Any] | None = None
    start_step = 0
    if restart_from is not None:
        # This performs all archive integrity and compatibility checks before
        # XLB/JAX is imported or any GPU array is allocated.
        restart_state = load_restart(
            restart_from,
            project=project,
            mesh=mesh,
            dt=dt,
            shape=shape,
            steps=steps,
        )
        start_step = int(restart_state["metadata"]["step"])
    physics = project.get("physics") or {}
    if not isinstance(physics, Mapping):
        raise ValueError("project.physics must be an object")
    geometry = project.get("geometry") or {}
    if not isinstance(geometry, Mapping):
        raise ValueError("project.geometry must be an object")
    flow_enabled = bool(physics.get("flow", True))
    thermal_enabled = bool(physics.get("thermal", True))
    try:
        initial_temperature = float(physics.get("initial_temperature", 293.15))
    except (TypeError, ValueError) as exc:
        raise ValueError("physics.initial_temperature must be numeric") from exc
    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        raise ValueError("physics.initial_temperature must be positive and finite")

    materials, material_ids, selected_index = _material_table(project)
    material_index_np = _resolve_material_index(
        project,
        geometry,
        material_index_np,
        fluid_np,
        solid_np,
        thermal_np,
        material_ids,
        selected_index,
    )
    turbulence_model, smagorinsky_constant, turbulent_prandtl = _turbulence_config(physics)
    warnings: list[str] = []
    fluid_j_np = fluid_np.astype(bool, copy=False)
    solid_j_np = solid_np.astype(bool, copy=False)
    thermal_j_np = thermal_np.astype(bool, copy=False)
    dx = float(spacing[0])

    box_boundaries, cad_boundary, cad_patches = _boundary_maps(project)
    thermal_by_face = {face: _thermal_spec(spec) for face, spec in box_boundaries.items()}
    flow_face_masks = _face_masks(shape, fluid_j_np)
    thermal_face_masks = _face_masks(shape, thermal_j_np)
    cad_flow_actions, cad_thermal_actions, cad_thermal_fallback = _cad_boundary_actions(
        boundary_links,
        box_boundaries,
        cad_boundary,
        cad_patches,
        fluid_j_np,
        thermal_j_np,
    )
    # A global CAD selector with no link field retains the legacy solid-mask
    # wall fallback.  When links exist, _cad_boundary_actions has converted
    # its thermal object into directional actions and suppresses this path.
    cad_thermal = cad_thermal_fallback

    material = materials[selected_index]
    gravity_model = GravityModel(physics, material, dt, dx, shape=shape)
    # Reference properties define the incompressible LBM scaling.  Every
    # thermal cell, including conductive solids, still receives its own local
    # rho/cp/k values through properties() below.
    density0 = _property_scalar(material.get("density"), initial_temperature, 1000.0, "density", warnings)
    if gravity_model.enabled and gravity_model.mode == "buoyancy":
        density0 = _property_scalar(material.get("density"), gravity_model.config['reference_temperature'], 1000.0, "density", warnings)
        if gravity_model.spec.kind == 'constant':
            warnings.append('Buoyancy is zero for constant density; select a temperature-dependent density material to model natural convection')
    viscosity0 = _property_scalar(material.get("viscosity"), initial_temperature, 1.0e-3, "viscosity", warnings)
    cp0 = _property_scalar(material.get("heat_capacity"), initial_temperature, 4182.0, "heat_capacity", warnings)
    conductivity0 = _property_scalar(material.get("conductivity"), initial_temperature, 0.6, "conductivity", warnings)
    if density0 <= 0.0 or viscosity0 <= 0.0 or cp0 <= 0.0 or conductivity0 <= 0.0:
        raise ValueError("material density, viscosity, heat_capacity, and conductivity must be positive")
    rho_reference = density0
    nu_lbm0 = viscosity0 / rho_reference * dt / (dx * dx)
    tau0 = 0.5 + nu_lbm0 / (1.0 / 3.0)
    if flow_enabled and (not math.isfinite(tau0) or tau0 <= 0.500001):
        raise ValueError(f"unstable initial relaxation time tau={tau0:g}; increase dt or resolve the mesh")
    if flow_enabled and tau0 > 5.0:
        warnings.append(f"initial relaxation time tau={tau0:g} is high; BGK accuracy may be poor")

    pressure_scale = rho_reference * (dx / dt) ** 2 * (1.0 / 3.0)
    if not math.isfinite(pressure_scale) or pressure_scale <= 0.0:
        raise ValueError("invalid pressure scaling from mesh spacing and dt")

    def _validate_flow_descriptor(flow: Any, label: str) -> float:
        """Validate a boundary flow object and return its requested Mach."""

        if not isinstance(flow, Mapping):
            return 0.0
        kind = str(flow.get("type", "no-slip")).lower().replace("_", "-")
        if kind not in {"no-slip", "noslip", "stationary", "wall", "none", "adiabatic", "velocity", "moving-wall", "moving", "pressure", "density"}:
            raise ValueError(f"unsupported flow boundary type {kind!r} on {label}")
        if kind in {"velocity", "moving-wall", "moving"}:
            value = _as_float_vector(flow.get("velocity", ()), f"boundary velocity {label}")
            lattice = value * dt / dx
            mach = float(np.linalg.norm(lattice) / math.sqrt(1.0 / 3.0))
            if not math.isfinite(mach) or mach >= 0.30:
                max_dt = 0.30 * dx / (math.sqrt(3.0) * float(np.linalg.norm(value)))
                raise ValueError(f"boundary Mach number {mach:g} exceeds the low-Mach limit 0.30; "
                                 f"reduce study.dt below {max_dt:.9g} s (current {dt:g} s)")
            return mach
        if kind in {"pressure", "density"}:
            try:
                pressure = float(flow.get("value", flow.get("pressure", 0.0)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"pressure boundary {label} requires a finite value") from exc
            if not math.isfinite(pressure):
                raise ValueError(f"pressure boundary {label} requires a finite value")
            rho_value = 1.0 + pressure / pressure_scale
            if rho_value <= 0.0 or abs(rho_value - 1.0) > 0.20:
                raise ValueError(f"pressure boundary {label} creates an unstable density excursion")
        return 0.0

    max_boundary_mach = 0.0
    if flow_enabled:
        for face, spec in box_boundaries.items():
            kind = _flow_type(spec)
            if kind in {"velocity", "moving-wall", "moving", "pressure", "density"} and not flow_face_masks[face].any():
                raise ValueError(f"flow boundary {face} does not touch any fluid cells")
            max_boundary_mach = max(max_boundary_mach, _validate_flow_descriptor(spec.get("flow"), face))
        for index, (mask, flow) in enumerate(cad_flow_actions):
            if np.asarray(mask, dtype=bool).any():
                max_boundary_mach = max(max_boundary_mach, _validate_flow_descriptor(flow, f"CAD link {index}"))

    compute_backend = _compute_backend(device)
    acceleration_seed = None
    if gravity_model.enabled:
        seed_temperature = restart_state['temperature'] if restart_state is not None else np.full(shape, initial_temperature)
        acceleration_seed, fraction = gravity_model.make_operator(np)(seed_temperature, fluid_j_np)
        gravity_model.check(fraction)
        del seed_temperature
        from .forcing import guo_source

    # Importing XLB is intentionally delayed until contract and stability
    # checks above have passed.  All JAX arrays are placed on target_device.
    ram_collision = None
    collision_device = None
    if compute_backend == "ram":
        from .ram_backend import RamCollision

        batch_cells = project["study"].get("gpu_batch_cells", 65536)
        if isinstance(batch_cells, bool) or not isinstance(batch_cells, int) or not 1024 <= batch_cells <= 1048576:
            raise ValueError("study.gpu_batch_cells must be an integer between 1024 and 1048576")
        jax, jnp, collision_device, gpu_velocity_set, _, gpu_equilibrium, gpu_collision = _load_xlb("cuda:0")
        gpu_force = partial(guo_source, jnp=jnp, c=gpu_velocity_set.c, w=gpu_velocity_set.w) if gravity_model.enabled else None
        ram_collision = RamCollision(jax, jnp, collision_device, gpu_equilibrium, gpu_collision, batch_cells=batch_cells, forcing_source=gpu_force)
        warnings.append("GPU+RAM: full state is stored in CPU RAM; only batched BGK collision runs on CUDA. Streaming, boundaries, thermal, LES and material evaluation run on CPU.")
    jax, jnp, target_device, velocity_set, macro, equilibrium, collision = _load_xlb("cpu" if compute_backend == "ram" else device)
    force_source = jax.jit(partial(guo_source, jnp=jnp, c=velocity_set.c, w=velocity_set.w)) if gravity_model.enabled else None
    source_valid_np = _valid_source_masks(shape)
    with jax.default_device(target_device):
        fluid = jnp.asarray(fluid_j_np)
        source_valid = jnp.asarray(source_valid_np)
        rho = jnp.ones((1, *shape), dtype=jnp.float32)
        velocity = jnp.zeros((3, *shape), dtype=jnp.float32)
        if restart_state is None:
            f = equilibrium(rho, velocity)
            if force_source is not None:
                # Zero *physical* initial velocity requires raw momentum -F/2.
                f = f - 0.5 * force_source(rho, velocity, jnp.asarray(acceleration_seed))
        else:
            # The distribution is the authoritative state.  Derive the
            # macroscopic auxiliaries from it instead of reconstructing an
            # equilibrium from the plotted velocity field.
            f = jnp.asarray(restart_state["f"], dtype=jnp.float32)
            if flow_enabled:
                rho, velocity = macro(f, rho, velocity)
            else:
                # Flow-disabled runs intentionally keep these auxiliaries at
                # the same neutral values as a fresh run.  The final macro
                # reduction still reports the loaded distribution exactly,
                # while thermal-only stepping never perturbs f.
                rho = jnp.ones((1, *shape), dtype=jnp.float32)
                velocity = jnp.zeros((3, *shape), dtype=jnp.float32)
        face_masks_jax = {face: jnp.asarray(mask) for face, mask in flow_face_masks.items()}
        cad_flow_masks_jax = [(jnp.asarray(mask), flow) for mask, flow in cad_flow_actions]

    if compute_backend == "ram" or force_source is not None:
        # Commit the initial/reloaded state to the requested device, like subsequent
        # streamed states. Uncommitted arrays may otherwise let JIT select
        # the process-default GPU for a macroscopic reduction after restart.
        f, rho, velocity = jax.device_put((f, rho, velocity), target_device)

    if compute_backend == "device":
        return _simulate_device(
            project=project,
            mesh=mesh,
            shape=shape,
            fluid_np=fluid_j_np,
            solid_np=solid_j_np,
            thermal_np=thermal_j_np,
            material_index_np=material_index_np,
            origin=origin,
            spacing=spacing,
            materials=materials,
            initial_temperature=initial_temperature,
            flow_enabled=flow_enabled,
            thermal_enabled=thermal_enabled,
            turbulence_model=turbulence_model,
            smagorinsky_constant=smagorinsky_constant,
            turbulent_prandtl=turbulent_prandtl,
            dt=dt,
            dx=dx,
            device=device,
            rho_reference=rho_reference,
            pressure_scale=pressure_scale,
            max_boundary_mach=max_boundary_mach,
            box_boundaries=box_boundaries,
            thermal_by_face=thermal_by_face,
            cad_thermal=cad_thermal,
            cad_flow_actions=cad_flow_actions,
            cad_thermal_actions=cad_thermal_actions,
            flow_face_masks=flow_face_masks,
            thermal_face_masks=thermal_face_masks,
            warnings=warnings,
            run_path=Path(run_dir),
            on_progress=on_progress,
            should_stop=should_stop,
            steps=steps,
            output_interval=output_interval,
            start_step=start_step,
            restart_temperature=(None if restart_state is None else restart_state["temperature"]),
            jax=jax,
            jnp=jnp,
            target_device=target_device,
            macro=macro,
            equilibrium=equilibrium,
            collision=collision,
            fluid=fluid,
            source_valid=source_valid,
            rho=rho,
            velocity=velocity,
            f=f,
            gravity_model=gravity_model,
            force_source=force_source,
        )

    if restart_state is None:
        temperature = np.full(shape, initial_temperature, dtype=float)
    else:
        # Keep the checkpoint's FP64 thermal state.  It is validated before
        # this point and copied so later updates cannot mutate the archive's
        # loaded buffer unexpectedly.
        temperature = np.array(restart_state["temperature"], dtype=np.float64, copy=True)
    if compute_backend == "ram":
        # The evolved CPU arrays now own everything needed for continuation.
        # Do not retain an extra full checkpoint distribution for the run.
        restart_state = None
    acceleration_np = acceleration_seed
    acceleration_seed = None
    gravity_operator_host = gravity_model.make_operator(np) if force_source is not None else None

    def update_acceleration():
        nonlocal acceleration_np
        if gravity_operator_host is not None:
            acceleration_np, fraction = gravity_operator_host(temperature, fluid_j_np)
            gravity_model.check(fraction)

    def physical_macro(current_f, current_rho, current_velocity):
        r, u = macro(current_f, current_rho, current_velocity)
        if force_source is not None:
            with jax.default_device(target_device):
                u = u + 0.5 * jnp.asarray(acceleration_np)
        return r, u

    refresh_force_boundaries = _force_boundary_refresh(
        box_items=[(_FACES.index(face), spec['flow']) for face, spec in box_boundaries.items() if isinstance(spec.get('flow'), Mapping)],
        cad_items=[flow for _, flow in cad_flow_masks_jax], equilibrium=equilibrium,
        force_source=force_source, dt=dt, dx=dx, pressure_scale=pressure_scale, jax=jax, jnp=jnp,
    ) if force_source is not None and gravity_model.mode == 'buoyancy' and thermal_enabled else None
    last_velocity: np.ndarray | None = None
    last_temperature: np.ndarray | None = None
    last_rho: np.ndarray | None = None
    monitor = ConvergenceMonitor(**project['study'].get('monitor', {}), interval=output_interval, dt=dt,
                                 flow_enabled=flow_enabled, thermal_enabled=thermal_enabled)
    residual_operator = make_residual_operator(np, initial_temperature=initial_temperature,
        velocity_scale=dx / dt, pressure_scale=pressure_scale, flow_enabled=flow_enabled, thermal_enabled=thermal_enabled)
    history: deque[dict[str, Any]] = deque(maxlen=2)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    stopped = False
    final_step = int(start_step)
    max_mach_seen = max_boundary_mach
    eddy_viscosity_np = np.zeros(shape, dtype=float)
    turbulent_conductivity_np = np.zeros(shape, dtype=float)
    snapshots = SnapshotWriter(run_path, project['study'].get('snapshot_interval', 0), dt,
                               fluid_mask=fluid_j_np, solid_mask=solid_j_np, thermal_mask=thermal_j_np,
                               material_index=material_index_np, origin=origin, spacing=spacing)
    snapshot_seconds = 0.0

    def checkpoint_reference(step: int) -> None:
        nonlocal snapshot_seconds
        started = time.perf_counter()
        rho_host, velocity_host = jax.device_get((rho, velocity))
        velocity_si = np.array(velocity_host, dtype=float) * dx / dt
        velocity_si[:, ~fluid_j_np] = 0.0
        if not flow_enabled:
            velocity_si.fill(0.0)
        pressure = (np.asarray(rho_host)[0] - 1.0) * pressure_scale
        pressure[~fluid_j_np] = 0.0
        eddy = (_smagorinsky_eddy_viscosity(velocity_si, fluid_j_np, dx, smagorinsky_constant)
                if flow_enabled and turbulence_model == 'smagorinsky' else np.zeros(shape))
        snapshots.write(step, velocity=velocity_si, pressure=pressure, temperature=temperature, eddy_viscosity=eddy)
        snapshot_seconds += time.perf_counter() - started

    def properties(current_temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        density, viscosity, cp, conductivity = _material_fields(
            materials, material_index_np, current_temperature, thermal_j_np, warnings
        )
        if (
            np.any(density[thermal_j_np] <= 0.0)
            or np.any(viscosity[thermal_j_np] <= 0.0)
            or np.any(cp[thermal_j_np] <= 0.0)
            or np.any(conductivity[thermal_j_np] <= 0.0)
        ):
            raise ValueError("temperature-dependent material properties must stay positive")
        tau = 0.5 + (viscosity / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
        tau_fluid = tau[fluid_j_np]
        if flow_enabled and (not np.isfinite(tau_fluid).all() or np.min(tau_fluid) <= 0.500001):
            raise ValueError("temperature-dependent viscosity produced an unstable relaxation time tau <= 0.5")
        if flow_enabled and np.max(tau_fluid) > 5.0 and "temperature-dependent relaxation time is high; BGK accuracy may be poor" not in warnings:
            warnings.append("temperature-dependent relaxation time is high; BGK accuracy may be poor")
        return density, viscosity, cp, conductivity, tau

    density, viscosity, cp, conductivity, tau = properties(temperature)
    if thermal_enabled:
        thermal_number, robin_number, thermal_loss_number = _thermal_stability_numbers(
            density,
            cp,
            conductivity,
            fluid_j_np,
            dt,
            dx,
            thermal_by_face,
            cad_thermal,
            thermal_face_masks,
            thermal_mask=thermal_j_np,
            cad_link_actions=cad_thermal_actions,
        )
        if thermal_loss_number > 0.95:
            raise ValueError("explicit thermal update is unstable for this dt and mesh (CFL/diffusion/wall loss sum > 0.95)")
    else:
        thermal_number = robin_number = thermal_loss_number = 0.0

    def _actual_device(value: Any) -> str:
        try:
            actual = getattr(value, "device", None)
            if callable(actual):
                actual = actual()
            if actual is None and hasattr(value, "devices"):
                devices = value.devices()
                actual = next(iter(devices), None)
            if isinstance(actual, (set, frozenset, tuple, list)):
                actual = next(iter(actual), None)
            return str(actual if actual is not None else target_device)
        except Exception:  # pragma: no cover - JAX version compatibility
            return str(target_device)

    def record(step: int) -> None:
        nonlocal rho, velocity, last_velocity, last_temperature, last_rho, max_mach_seen
        # Refresh from the post-stream distribution so samples and diagnostics
        # describe the advertised physical time.
        rho, velocity = physical_macro(f, rho, velocity)
        rho_np = np.asarray(jax.device_get(rho))[0]
        velocity_lbm_np = np.array(jax.device_get(velocity), dtype=float, copy=True)
        if not np.isfinite(rho_np[fluid_j_np]).all() or not np.isfinite(velocity_lbm_np[:, fluid_j_np]).all() or np.any(rho_np[fluid_j_np] <= 0.0):
            raise RuntimeError("LBM produced a non-finite or non-positive density/velocity")
        velocity_lbm_np[:, ~fluid_j_np] = 0.0
        mach = np.linalg.norm(velocity_lbm_np, axis=0) / math.sqrt(1.0 / 3.0)
        max_mach = float(np.max(mach[fluid_j_np]))
        max_mach_seen = max(max_mach_seen, max_mach)
        if max_mach >= 0.30:
            raise RuntimeError(f"LBM Mach number {max_mach:g} exceeded the low-Mach limit 0.30")
        active = thermal_j_np
        values = None if last_velocity is None else residual_operator(
            velocity_lbm_np, rho_np[None, ...], temperature, last_velocity, last_rho, last_temperature,
            fluid_j_np, thermal_j_np)
        monitor_row = monitor.update(step, values)
        velocity_phys = velocity_lbm_np * dx / dt
        speed = np.linalg.norm(velocity_phys, axis=0)
        progress_span = steps - start_step
        progress_value = 1.0 if progress_span == 0 else float((step - start_step) / progress_span)
        row: dict[str, Any] = {
            "step": int(step),
            "progress": progress_value,
            **monitor_row,
            "temperature_min": float(np.min(temperature[active])),
            "temperature_max": float(np.max(temperature[active])),
            "temperature_mean": float(np.mean(temperature[active])),
            "speed_mean": float(np.mean(speed[fluid_j_np])),
            "speed_max": float(np.max(speed[fluid_j_np])),
            "mach_max": max_mach,
            "eddy_viscosity_max": float(np.max(eddy_viscosity_np[fluid_j_np])) if flow_enabled else 0.0,
        }
        history.append(row)
        _append_history(run_path / 'history.csv', row, initial=len(history) == 1)
        if on_progress is not None:
            on_progress(dict(row))
        last_velocity = velocity_lbm_np.copy()
        last_temperature = temperature.copy()
        last_rho = rho_np[None, ...].copy()

    # Always emit an initial, actual state.  This also makes a run stopped
    # before its first step inspectable.
    record(start_step)
    if should_stop is not None and should_stop():
        stopped = True
    else:
        if snapshots.interval:
            checkpoint_reference(start_step)
        for step in range(start_step + 1, steps + 1):
            density, viscosity, cp, conductivity, tau_molecular = properties(temperature)
            rho, velocity = physical_macro(f, rho, velocity)
            velocity_pre_np = np.asarray(jax.device_get(velocity)) * dx / dt
            velocity_pre_np[:, ~fluid_j_np] = 0.0
            if flow_enabled and turbulence_model == "smagorinsky":
                eddy_viscosity_np = _smagorinsky_eddy_viscosity(velocity_pre_np, fluid_j_np, dx, smagorinsky_constant)
            else:
                eddy_viscosity_np = np.zeros(shape, dtype=float)
            turbulent_conductivity_np = density * cp * eddy_viscosity_np / turbulent_prandtl
            if turbulence_model != "smagorinsky":
                turbulent_conductivity_np.fill(0.0)
            conductivity_transport = conductivity + turbulent_conductivity_np
            if flow_enabled:
                # Smagorinsky adds local kinematic eddy viscosity to the
                # molecular dynamic viscosity in the BGK relaxation time.
                viscosity_effective = viscosity + rho_reference * eddy_viscosity_np
                tau = 0.5 + (viscosity_effective / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
                tau_fluid = tau[fluid_j_np]
                if not np.isfinite(tau_fluid).all() or np.min(tau_fluid) <= 0.500001:
                    raise ValueError("effective Smagorinsky viscosity produced an unstable relaxation time tau <= 0.5")
                if np.max(tau_fluid) > 5.0 and "effective Smagorinsky relaxation time is high; BGK accuracy may be poor" not in warnings:
                    warnings.append("effective Smagorinsky relaxation time is high; BGK accuracy may be poor")
            else:
                tau = tau_molecular

            if thermal_enabled:
                thermal_number, robin_number, thermal_loss_number = _thermal_stability_numbers(
                    density,
                    cp,
                    conductivity_transport,
                    fluid_j_np,
                    dt,
                    dx,
                    thermal_by_face,
                    cad_thermal,
                    thermal_face_masks,
                    thermal_mask=thermal_j_np,
                    cad_link_actions=cad_thermal_actions,
                )
                if thermal_loss_number > 0.95:
                    raise ValueError("temperature-dependent thermal diffusivity made the explicit update unstable")

            if flow_enabled:
                omega_np = 1.0 / tau
                with jax.default_device(target_device):
                    acceleration = jnp.asarray(acceleration_np) if force_source is not None else None
                    if ram_collision is None:
                        omega = jnp.asarray(omega_np, dtype=jnp.float32)
                        feq = equilibrium(rho, velocity)
                        f_post = collision(f, feq, omega)
                        if force_source is not None:
                            f_post = f_post + (1.0 - 0.5 * omega) * force_source(rho, velocity, acceleration)
                    else:
                        # Collision is cell-local. Global CPU streaming below
                        # preserves every cross-batch neighbour exactly, with
                        # the same CAD and box boundary treatment as before.
                        f_post = jax.device_put(ram_collision(f, rho, velocity, omega_np, acceleration=acceleration_np), target_device)
                    f_next = _stream_jax(f_post, fluid, source_valid, jnp)
                    # Explicit box faces are applied first.  Local CAD links
                    # then override a global CAD fallback and are disjoint
                    # from box links by construction.
                    for face, spec in box_boundaries.items():
                        flow = spec.get("flow")
                        if not isinstance(flow, Mapping):
                            continue
                        target = _flow_target(
                            flow,
                            rho,
                            velocity,
                            face_masks_jax[face],
                            equilibrium,
                            jnp,
                            dt,
                            dx,
                            pressure_scale,
                            0.30,
                            acceleration,
                            force_source,
                        )
                        if target is not None:
                            f_next = jnp.where(face_masks_jax[face][None, ...], target, f_next)
                    for mask_jax, flow in cad_flow_masks_jax:
                        target = _flow_target(
                            flow,
                            rho,
                            velocity,
                            mask_jax,
                            equilibrium,
                            jnp,
                            dt,
                            dx,
                            pressure_scale,
                            0.30,
                            acceleration,
                            force_source,
                        )
                        if target is not None:
                            f_next = jnp.where(mask_jax[None, ...], target, f_next)
                    f = f_next
            else:
                # Keep the real XLB state at rest while thermal transport may
                # continue.  Collision is intentionally disabled in this
                # mode, so tau is diagnostic only.  Preserve the actual
                # distribution instead of rebuilding equilibrium from rho;
                # that keeps thermal-only checkpoints exactly continuous.
                with jax.default_device(target_device):
                    velocity = jnp.zeros_like(velocity)

            # The distribution after boundary treatment is the velocity used
            # by this thermal step and by the progress sample at this step.
            rho, velocity = physical_macro(f, rho, velocity)
            velocity_phys = np.asarray(jax.device_get(velocity)) * dx / dt
            velocity_phys[:, ~fluid_j_np] = 0.0
            if thermal_enabled:
                # Upwind advection consumes the sum of directional Courant
                # numbers.  Combining that with the exact conductive/wall
                # diagonal gives a conservative explicit stability check.
                cfl = float(np.max(np.sum(np.abs(velocity_phys[:, fluid_j_np]), axis=0) * dt / dx))
                if not math.isfinite(cfl) or cfl >= 1.0 or cfl + thermal_loss_number > 0.95:
                    raise RuntimeError("thermal advection/diffusion CFL stability limit exceeded")
                temperature = _thermal_step(
                    temperature,
                    velocity_phys,
                    fluid_j_np,
                    density,
                    cp,
                    conductivity_transport,
                    dt,
                    dx,
                    thermal_by_face,
                    cad_thermal,
                    thermal_face_masks,
                    warnings,
                    thermal_mask=thermal_j_np,
                    cad_link_actions=cad_thermal_actions,
                )
            if not np.isfinite(temperature[thermal_j_np]).all():
                raise RuntimeError("thermal field became non-finite")
            if force_source is not None:
                update_acceleration()
                rho, velocity = physical_macro(f, rho, velocity)
                if refresh_force_boundaries is not None:
                    with jax.default_device(target_device):
                        f = refresh_force_boundaries(f, rho, velocity, jnp.asarray(acceleration_np),
                                                     [face_masks_jax[face] for face in _FACES],
                                                     [mask for mask, _ in cad_flow_masks_jax])
                    rho, velocity = physical_macro(f, rho, velocity)

            final_step = step
            if snapshots.due(step) and step != steps:
                checkpoint_reference(step)
            if (step - start_step) % output_interval == 0 or step == steps:
                record(step)
            if should_stop is not None and should_stop():
                stopped = True
                break

    # Ensure the final state is represented in all artifacts, even when a stop
    # request landed between output intervals.
    if not history or history[-1]["step"] != final_step:
        record(final_step)

    # Final fields are read from the actual XLB state.  Pressure is a gauge
    # pressure using the incompressible reference-density approximation.
    rho, velocity = physical_macro(f, rho, velocity)
    rho_np = np.asarray(jax.device_get(rho))[0]
    velocity_lbm_np = np.array(jax.device_get(velocity), dtype=float, copy=True)
    velocity_lbm_np[:, ~fluid_j_np] = 0.0
    velocity_phys_np = velocity_lbm_np * dx / dt
    pressure_np = (rho_np - 1.0) * pressure_scale
    pressure_np[~fluid_j_np] = 0.0
    if not np.isfinite(pressure_np[fluid_j_np]).all():
        raise RuntimeError("pressure field became non-finite")
    # Re-evaluate material properties at the final thermal state so the
    # reported relaxation and turbulent transport coefficients correspond to
    # the fields written below, rather than the pre-step values from the last
    # collision.
    density_final, viscosity_final, cp_final, conductivity_final, tau_molecular_final = properties(temperature)
    if flow_enabled and turbulence_model == "smagorinsky":
        eddy_viscosity_np = _smagorinsky_eddy_viscosity(velocity_phys_np, fluid_j_np, dx, smagorinsky_constant)
        tau = 0.5 + ((viscosity_final + rho_reference * eddy_viscosity_np) / rho_reference) * dt / (dx * dx) / (1.0 / 3.0)
    else:
        eddy_viscosity_np = np.zeros(shape, dtype=float)
        tau = tau_molecular_final
    if flow_enabled and (not np.isfinite(tau[fluid_j_np]).all() or np.min(tau[fluid_j_np]) <= 0.500001):
        raise RuntimeError("final effective viscosity produced an unstable relaxation time tau <= 0.5")
    turbulent_conductivity_np = density_final * cp_final * eddy_viscosity_np / turbulent_prandtl
    if turbulence_model != "smagorinsky":
        turbulent_conductivity_np.fill(0.0)
    snapshots.write(final_step, velocity=velocity_phys_np, pressure=pressure_np,
                    temperature=temperature, eddy_viscosity=eddy_viscosity_np)
    np.savez_compressed(
        run_path / "fields.npz",
        velocity=velocity_phys_np.astype(np.float32),
        pressure=pressure_np.astype(np.float32),
        temperature=temperature.astype(np.float32),
        eddy_viscosity=eddy_viscosity_np.astype(np.float32),
        fluid_mask=fluid_j_np,
        solid_mask=solid_j_np,
        thermal_mask=thermal_j_np,
        material_index=material_index_np.astype(np.int32),
        origin=origin.astype(float),
        spacing=spacing.astype(float),
    )
    _write_vtk(
        run_path / "fields.vtk",
        velocity_phys_np,
        pressure_np,
        temperature,
        fluid_j_np,
        origin,
        spacing,
        solid=solid_j_np,
        thermal=thermal_j_np,
        eddy_viscosity=eddy_viscosity_np,
    )
    write_restart(
        run_path / "restart.npz",
        step=final_step,
        dt=dt,
        f=np.asarray(jax.device_get(f)),
        temperature=np.asarray(temperature, dtype=np.float64),
        project=project,
        mesh=mesh,
        status="stopped" if stopped else "completed",
    )

    converged = bool((not stopped) and monitor.satisfied)
    if turbulence_model == "smagorinsky":
        warnings.append("Smagorinsky LES uses cell-centred gradients, voxel walls, and first-order thermal advection; short runs are not a turbulence benchmark")
    if thermal_enabled:
        warnings.append("thermal advection/diffusion uses an explicit NumPy finite-volume update on the host")
    actual_device = _actual_device(f)
    diagnostics = {
        "backend": "XLB D3Q27 BGK via JAX",
        "compute_backend": compute_backend,
        "convergence_monitor": monitor.diagnostics(),
        "device": device,
        "device_actual": actual_device,
        "gravity": gravity_model.diagnostics(),
        "pressure_kind": "reduced" if force_source is not None and gravity_model.mode == "buoyancy" else "gauge",
        "state_device": actual_device,
        "collision_device": str(collision_device) if ram_collision is not None and flow_enabled else actual_device,
        "ram_offload": dict(ram_collision.stats) if ram_collision is not None else None,
        "jax_matmul_precision": "highest",
        "velocity_set": "D3Q27",
        "collision": "BGK",
        "tau_min": float(np.min(tau[fluid_j_np])),
        "tau_max": float(np.max(tau[fluid_j_np])),
        "mach_max": float(max_mach_seen),
        "thermal_fourier_max": float(thermal_number),
        "thermal_robin_loss_max": float(robin_number),
        "thermal_loss_coefficient_max": float(thermal_loss_number),
        "temperature_residual": history[-1].get("temperature_residual"),
        "eddy_viscosity_max_m2_s": float(np.max(eddy_viscosity_np[fluid_j_np])) if flow_enabled else 0.0,
        "turbulent_conductivity_max_w_m_k": float(np.max(turbulent_conductivity_np[thermal_j_np])) if thermal_enabled else 0.0,
        "turbulence_model": turbulence_model,
        "smagorinsky_constant": float(smagorinsky_constant),
        "turbulent_prandtl": float(turbulent_prandtl),
        "flow_cells": int(np.count_nonzero(fluid_j_np)),
        "solid_cells": int(np.count_nonzero(solid_j_np)),
        "thermal_cells": int(np.count_nonzero(thermal_j_np)),
        "rho_reference_kg_m3": float(rho_reference),
        "density_model": "reference-density incompressible LBM; local density(T) is used in thermal storage/diffusion coefficients and, when enabled, buoyancy",
        "thermal_backend": "NumPy finite-volume on host",
        "les_backend": "NumPy Smagorinsky host operator" if turbulence_model == "smagorinsky" and flow_enabled else "disabled",
        "property_backend": "NumPy material evaluator",
        "full_field_transfers": None,
        "snapshot_count": len(snapshots.frames),
        "snapshot_bytes": snapshots.bytes_written,
        "device_scalar_transfers": None,
        "timings": {
            "first_step_s": 0.0,
            "warm_compute_s": 0.0,
            "save_s": 0.0,
            "snapshots_s": float(snapshot_seconds),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
    return {
        "status": "stopped" if stopped else "completed",
        "steps": int(final_step),
        "converged": bool(converged),
        "diagnostics": diagnostics,
    }


__all__ = ["simulate"]
