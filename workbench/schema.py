"""Project schema and temperature dependent material properties.

The workbench keeps project files deliberately small and JSON compatible.  This
module is the single place where project input is checked and normalised before
it reaches the mesh builder or solver.  Property tables are intentionally
limited to linear interpolation; accepting Python expressions here would make
saved projects non portable and unsafe to evaluate.
"""

from __future__ import annotations

import copy
import math
import numbers
import re
import warnings
from typing import Any, Mapping

import numpy as np
from .limits import MAX_AXIS_CELLS, MAX_TOTAL_CELLS
from .mesh_planning import plan_box


SCHEMA_VERSION = 1

_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_ALL_FACES = frozenset((*_FACES, "cad"))
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class PropertyClampWarning(UserWarning):
    """Warning emitted when a temperature table is evaluated outside its range."""


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return true for real numeric scalars, excluding booleans."""

    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _finite(value: Any, label: str) -> float:
    if not _is_number(value):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be a finite number")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        _fail(f"{label} must be positive")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        _fail(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        _fail(f"{label} must be at least {minimum}")
    return result


def _vector(value: Any, label: str, *, positive: bool = False, length: int = 3) -> list[float]:
    if isinstance(value, (str, bytes)):
        _fail(f"{label} must be a {length}-component vector")
    try:
        values = list(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be a {length}-component vector")
    if len(values) != length:
        _fail(f"{label} must be a {length}-component vector")
    return [
        _positive(item, f"{label}[{index}]") if positive else _finite(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not _ID_RE.fullmatch(value):
        _fail(f"{label} must be a non-empty identifier using letters, digits, '_', '.', ':', or '-'")
    return value


def _text(value: Any, label: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        _fail(f"{label} must be a non-empty string of at most {max_length} characters")
    if any(ord(character) < 32 for character in value):
        _fail(f"{label} contains a control character")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        _fail(f"{label} contains unknown field(s): {names}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _property(prop: Any, label: str, *, require_positive: bool, temperature_positive: bool = False) -> dict[str, Any]:
    prop = _mapping(prop, label)
    _keys(prop, {"kind", "value", "points"}, label)
    kind = prop.get("kind")
    if kind == "constant":
        if "value" not in prop or "points" in prop:
            _fail(f"{label} constant property requires value only")
        value = _positive(prop["value"], f"{label}.value") if require_positive else _finite(prop["value"], f"{label}.value")
        return {"kind": "constant", "value": value}
    if kind == "table":
        if "points" not in prop or "value" in prop:
            _fail(f"{label} table property requires points only")
        raw_points = prop["points"]
        if isinstance(raw_points, (str, bytes)):
            _fail(f"{label}.points must be an array of [temperature, value] pairs")
        try:
            points = list(raw_points)
        except (TypeError, ValueError):
            _fail(f"{label}.points must be an array of [temperature, value] pairs")
        if len(points) < 2:
            _fail(f"{label}.points must contain at least two points")
        normalized: list[list[float]] = []
        previous_temperature: float | None = None
        for index, point in enumerate(points):
            if isinstance(point, (str, bytes)):
                _fail(f"{label}.points[{index}] must contain temperature and value")
            try:
                pair = list(point)
            except (TypeError, ValueError):
                _fail(f"{label}.points[{index}] must contain temperature and value")
            if len(pair) != 2:
                _fail(f"{label}.points[{index}] must contain temperature and value")
            temperature = _positive(pair[0], f"{label}.points[{index}][0]") if temperature_positive else _finite(pair[0], f"{label}.points[{index}][0]")
            value = _positive(pair[1], f"{label}.points[{index}][1]") if require_positive else _finite(pair[1], f"{label}.points[{index}][1]")
            if previous_temperature is not None and temperature <= previous_temperature:
                _fail(f"{label}.points temperatures must be strictly increasing")
            previous_temperature = temperature
            normalized.append([temperature, value])
        return {"kind": "table", "points": normalized}
    _fail(f"{label}.kind must be 'constant' or 'table'")
    raise AssertionError("unreachable")


def _property_value_without_warning(prop: Mapping[str, Any], temperature: float) -> float:
    if prop["kind"] == "constant":
        return float(prop["value"])
    points = np.asarray(prop["points"], dtype=float)
    return float(np.interp(float(temperature), points[:, 0], points[:, 1]))


def _computational_box(geometry: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    """Read the optional obstacle domain descriptor without mutating geometry."""

    descriptor: Any = None
    for key in ("computational_box", "domain"):
        if key in geometry:
            if descriptor is not None:
                _fail("geometry may specify only one of computational_box or domain")
            descriptor = geometry[key]
    if descriptor is not None:
        descriptor = _mapping(descriptor, "geometry.computational_box")
        _keys(descriptor, {"size", "origin"}, "geometry.computational_box")
        if "size" not in descriptor:
            _fail("geometry.computational_box.size is required")
        size = _vector(descriptor["size"], "geometry.computational_box.size", positive=True)
        origin = _vector(descriptor.get("origin", [0.0, 0.0, 0.0]), "geometry.computational_box.origin")
        return size, origin
    size_key = next((key for key in ("domain_size", "box_size") if key in geometry), None)
    origin_key = next((key for key in ("domain_origin", "box_origin") if key in geometry), None)
    if size_key is None:
        if origin_key is not None:
            _fail(f"geometry.{origin_key} requires geometry.{('domain_size' if origin_key == 'domain_origin' else 'box_size')}")
        return None
    if origin_key is None:
        origin_key = "domain_origin" if size_key == "domain_size" else "box_origin"
    size = _vector(geometry[size_key], f"geometry.{size_key}", positive=True)
    origin = _vector(geometry.get(origin_key, [0.0, 0.0, 0.0]), f"geometry.{origin_key}")
    return size, origin


def _normalise_solid_region(value: Any, index: int) -> dict[str, Any]:
    label = f"geometry.solids[{index}]"
    value = _mapping(value, label)
    _keys(value, {"id", "name", "origin", "size", "material_id"}, label)
    for key in ("id", "name", "origin", "size", "material_id"):
        if key not in value:
            _fail(f"{label}.{key} is required")
    return {
        "id": _identifier(value["id"], f"{label}.id"),
        "name": _text(value["name"], f"{label}.name"),
        "origin": _vector(value["origin"], f"{label}.origin"),
        "size": _vector(value["size"], f"{label}.size", positive=True),
        "material_id": _identifier(value["material_id"], f"{label}.material_id"),
    }


def _validate_solid_region_bounds(geometry: Mapping[str, Any], solids: list[dict[str, Any]]) -> None:
    """Check boxes whose containing computational domain is known in JSON."""

    if not solids:
        return
    domain_size: list[float] | None = None
    domain_origin: list[float] | None = None
    if geometry.get("kind") == "box" and geometry.get("role") == "fluid" and "size" in geometry:
        domain_size = list(geometry["size"])
        domain_origin = list(geometry.get("origin", [0.0, 0.0, 0.0]))
    elif geometry.get("role") == "obstacle":
        descriptor = _computational_box(geometry)
        if descriptor is not None:
            domain_size, domain_origin = descriptor
        elif geometry.get("kind") == "cad" and "size" in geometry and "origin" in geometry:
            domain_size = list(geometry["size"])
            domain_origin = list(geometry["origin"])
    if domain_size is None or domain_origin is None:
        return
    lower = np.asarray(domain_origin, dtype=float)
    upper = lower + np.asarray(domain_size, dtype=float)
    for solid in solids:
        solid_lower = np.asarray(solid["origin"], dtype=float)
        solid_upper = solid_lower + np.asarray(solid["size"], dtype=float)
        if np.any(solid_lower < lower) or np.any(solid_upper > upper):
            _fail(f"geometry.solids[{solid['id']}] must lie inside the computational box")
    for index, first in enumerate(solids):
        first_lower = np.asarray(first["origin"], dtype=float)
        first_upper = first_lower + np.asarray(first["size"], dtype=float)
        for second in solids[index + 1 :]:
            second_lower = np.asarray(second["origin"], dtype=float)
            second_upper = second_lower + np.asarray(second["size"], dtype=float)
            overlap = np.minimum(first_upper, second_upper) - np.maximum(first_lower, second_lower)
            if np.all(overlap > 0.0):
                _fail(f"geometry.solids regions '{first['id']}' and '{second['id']}' overlap")


def _normalise_turbulence(value: Any) -> dict[str, Any]:
    """Validate the optional turbulence model descriptor.

    The constants are retained in the normalized representation even for a
    laminar project.  This makes a project round-trip stable and gives the
    solver an explicit, deterministic default if the model is later changed
    in the UI.
    """

    label = "physics.turbulence"
    if value is None:
        _fail(f"{label} must be an object when provided")
    value = _mapping(value, label)
    _keys(value, {"model", "smagorinsky_constant", "turbulent_prandtl"}, label)
    model = value.get("model", "laminar")
    if model not in {"laminar", "smagorinsky"}:
        _fail(f"{label}.model must be 'laminar' or 'smagorinsky'")
    smagorinsky_constant = _positive(value.get("smagorinsky_constant", 0.17), f"{label}.smagorinsky_constant")
    turbulent_prandtl = _positive(value.get("turbulent_prandtl", 0.9), f"{label}.turbulent_prandtl")
    return {
        "model": model,
        "smagorinsky_constant": smagorinsky_constant,
        "turbulent_prandtl": turbulent_prandtl,
    }


def normalise_gravity(value: Any) -> dict[str, Any]:
    """Validate and normalize the optional gravity/buoyancy descriptor.

    ``None`` is treated like an omitted descriptor and returns a fresh set of
    disabled defaults.  Keeping this as a public helper lets the solver apply
    the exact same strict input contract when it receives a direct simulation
    request rather than a complete project document.
    """

    label = "physics.gravity"
    if value is None:
        value = {}
    value = _mapping(value, label)
    _keys(value, {"enabled", "mode", "vector", "reference_temperature"}, label)
    enabled = value.get("enabled", False)
    if not _is_bool(enabled):
        _fail(f"{label}.enabled must be a boolean")
    mode = value.get("mode", "uniform")
    if not isinstance(mode, str) or mode not in {"uniform", "buoyancy"}:
        _fail(f"{label}.mode must be 'uniform' or 'buoyancy'")
    vector = _vector(value.get("vector", [0.0, 0.0, -9.80665]), f"{label}.vector")
    reference_temperature = _positive(
        value.get("reference_temperature", 293.15),
        f"{label}.reference_temperature",
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "vector": vector,
        "reference_temperature": reference_temperature,
    }


def _mesh_cells(mesh: Any, label: str = "mesh", *, enforce_limits: bool = True) -> list[int]:
    mesh = _mapping(mesh, label)
    _keys(mesh, {"cells"}, label)
    if "cells" not in mesh:
        _fail("mesh.cells is required")
    raw_cells = mesh["cells"]
    if isinstance(raw_cells, (str, bytes)):
        _fail("mesh.cells must be a three-component integer vector")
    try:
        raw_cells = list(raw_cells)
    except (TypeError, ValueError):
        _fail("mesh.cells must be a three-component integer vector")
    cells = [_integer(value, f"{label}.cells[{index}]", minimum=1) for index, value in enumerate(raw_cells)]
    if len(cells) != 3:
        _fail("mesh.cells must be a three-component integer vector")
    if enforce_limits and any(value > MAX_AXIS_CELLS for value in cells):
        _fail(f"mesh cells cannot exceed {MAX_AXIS_CELLS} per axis (XLB_MAX_AXIS_CELLS)")
    total = math.prod(cells)
    if enforce_limits and total > MAX_TOTAL_CELLS:
        _fail(f"mesh contains {total} cells; the limit is {MAX_TOTAL_CELLS} (XLB_MAX_TOTAL_CELLS)")
    return cells


def _mesh_target(mesh: Any) -> int | None:
    """Validate the optional scalar target without requiring legacy cells."""

    mesh = _mapping(mesh, "mesh")
    _keys(mesh, {"cells", "target_cells"}, "mesh")
    if "target_cells" not in mesh:
        return None
    result = _integer(mesh["target_cells"], "mesh.target_cells", minimum=1)
    if result > MAX_TOTAL_CELLS:
        _fail(f"mesh.target_cells cannot exceed {MAX_TOTAL_CELLS} (XLB_MAX_TOTAL_CELLS)")
    return result


def _outer_domain_descriptor(geometry: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    """Return the actual computational box for known box/obstacle forms."""

    kind = geometry.get("kind")
    role = geometry.get("role")
    if kind == "box" and role == "fluid" and "size" in geometry:
        return list(geometry["size"]), list(geometry.get("origin", [0.0, 0.0, 0.0]))
    if role == "obstacle":
        descriptor = _computational_box(geometry)
        if descriptor is not None:
            return descriptor
        # CAD obstacle shorthand uses ``size`` for the outer box.  A box
        # obstacle without a descriptor is rejected by build_mesh, so it is
        # intentionally not treated as a computational domain here.
        if kind == "cad" and "size" in geometry:
            return list(geometry["size"]), list(geometry.get("origin", [0.0, 0.0, 0.0]))
    return None


def _validate_box_spacing(geometry: Mapping[str, Any], cells: list[int]) -> None:
    size: Any = None
    if geometry.get("kind") == "box":
        if geometry.get("role") == "obstacle" and any(key in geometry for key in ("computational_box", "domain", "domain_size", "box_size")):
            descriptor = _computational_box(geometry)
            size = descriptor[0] if descriptor is not None else None
        elif "size" in geometry:
            size = geometry["size"]
    elif geometry.get("kind") == "cad" and geometry.get("role") == "obstacle" and "size" in geometry and not any(key in geometry for key in ("computational_box", "domain", "domain_size", "box_size")):
        # CAD obstacle shorthand: size is the outer computational box.
        size = geometry["size"]
    if size is None:
        return
    spacing = np.asarray(size, dtype=float) / np.asarray(cells, dtype=float)
    if not np.allclose(spacing, spacing[0], rtol=1e-9, atol=1e-15):
        _fail("geometry.size / mesh.cells must produce isotropic Cartesian spacing")


def _normalise_boundary(boundary: Any, index: int, geometry_kind: str) -> dict[str, Any]:
    label = f"boundaries[{index}]"
    boundary = _mapping(boundary, label)
    _keys(boundary, {"id", "name", "face", "flow", "thermal"}, label)
    for key in ("id", "name", "face", "flow", "thermal"):
        if key not in boundary:
            _fail(f"{label}.{key} is required")
    identifier = _identifier(boundary["id"], f"{label}.id")
    name = _text(boundary["name"], f"{label}.name")
    face = boundary["face"]
    if not isinstance(face, str):
        _fail(f"{label}.face must be one of {', '.join(sorted(_ALL_FACES))} or cad:<patch_id>")
    is_cad_selector = face == "cad" or face.startswith("cad:")
    if is_cad_selector:
        if geometry_kind != "cad":
            _fail("a CAD boundary selector requires geometry.kind='cad'")
        if face.startswith("cad:"):
            patch_id = face[4:]
            if not patch_id:
                _fail(f"{label}.face must use cad:<patch_id> with a non-empty patch id")
            _identifier(patch_id, f"{label}.face patch id")
    elif face not in _FACES:
        _fail(f"{label}.face must be one of {', '.join(_FACES)}, 'cad', or 'cad:<patch_id>'")

    flow = _mapping(boundary["flow"], f"{label}.flow")
    flow_type = flow.get("type")
    if flow_type in ("wall", "no-slip", "noslip", "stationary"):
        _keys(flow, {"type"}, f"{label}.flow")
        flow_normalized = {"type": "wall"}
    elif flow_type == "velocity":
        _keys(flow, {"type", "velocity"}, f"{label}.flow")
        if "velocity" not in flow:
            _fail(f"{label}.flow.velocity is required")
        flow_normalized = {"type": "velocity", "velocity": _vector(flow["velocity"], f"{label}.flow.velocity")}
    elif flow_type == "pressure":
        _keys(flow, {"type", "value"}, f"{label}.flow")
        if "value" not in flow:
            _fail(f"{label}.flow.value is required")
        flow_normalized = {"type": "pressure", "value": _finite(flow["value"], f"{label}.flow.value")}
    else:
        _fail(f"{label}.flow.type must be 'wall', 'velocity', or 'pressure'")
    # A global CAD selector represents the remaining surface and therefore
    # defaults to a stationary wall.  Individual native/derived patches may
    # carry inlet, outlet, or wall flow conditions.
    if face == "cad" and flow_normalized["type"] != "wall":
        _fail("a CAD surface boundary must use flow.type='wall'")

    thermal = _mapping(boundary["thermal"], f"{label}.thermal")
    thermal_type = thermal.get("type")
    if thermal_type == "temperature":
        _keys(thermal, {"type", "value"}, f"{label}.thermal")
        if "value" not in thermal:
            _fail(f"{label}.thermal.value is required")
        thermal_normalized = {"type": "temperature", "value": _positive(thermal["value"], f"{label}.thermal.value")}
    elif thermal_type == "adiabatic":
        _keys(thermal, {"type"}, f"{label}.thermal")
        thermal_normalized = {"type": "adiabatic"}
    elif thermal_type in ("heat_flux", "heat-flux"):
        _keys(thermal, {"type", "value"}, f"{label}.thermal")
        if "value" not in thermal:
            _fail(f"{label}.thermal.value is required for heat_flux")
        thermal_normalized = {"type": "heat_flux", "value": _finite(thermal["value"], f"{label}.thermal.value")}
    elif thermal_type in ("convection", "convective"):
        _keys(thermal, {"type", "h", "ambient_temperature"}, f"{label}.thermal")
        for key in ("h", "ambient_temperature"):
            if key not in thermal:
                _fail(f"{label}.thermal.{key} is required for convection")
        coefficient = _finite(thermal["h"], f"{label}.thermal.h")
        if coefficient < 0.0:
            _fail(f"{label}.thermal.h must be non-negative")
        ambient = _positive(thermal["ambient_temperature"], f"{label}.thermal.ambient_temperature")
        thermal_normalized = {"type": "convection", "h": coefficient, "ambient_temperature": ambient}
    else:
        _fail(f"{label}.thermal.type must be 'temperature', 'heat_flux', 'convection', or 'adiabatic'")
    return {"id": identifier, "name": name, "face": face, "flow": flow_normalized, "thermal": thermal_normalized}


def default_project() -> dict[str, Any]:
    """Return a fresh, physically scaled starter project.

    With water-like properties, ``dx=0.002 m`` and ``dt=0.01 s`` give a
    lattice kinematic viscosity of about 0.0025 and BGK relaxation time
    ``tau ≈ 0.5075``.  This keeps the initial project above the collision
    singularity while retaining SI values for the solver.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "name": "加熱矩形流路",
        "geometry": {
            "kind": "box",
            "size": [0.06, 0.02, 0.02],
            "asset_id": None,
            "solid_asset_id": None,
            "role": "fluid",
            "solids": [],
        },
        "materials": [
            {
                "id": "water",
                "name": "Water",
                "density": {"kind": "constant", "value": 998.0},
                "viscosity": {"kind": "constant", "value": 0.001},
                "heat_capacity": {"kind": "constant", "value": 4182.0},
                "conductivity": {"kind": "constant", "value": 0.6},
            }
        ],
        "physics": {
            "flow": True,
            "thermal": True,
            "material_id": "water",
            "initial_temperature": 293.15,
            "gravity": {
                "enabled": False,
                "mode": "uniform",
                "vector": [0.0, 0.0, -9.80665],
                "reference_temperature": 293.15,
            },
            "turbulence": {
                "model": "laminar",
                "smagorinsky_constant": 0.17,
                "turbulent_prandtl": 0.9,
            },
        },
        "boundaries": [
            {
                "id": "inlet",
                "name": "Inlet",
                "face": "xmin",
                "flow": {"type": "velocity", "velocity": [0.01, 0.0, 0.0]},
                "thermal": {"type": "temperature", "value": 293.15},
            },
            {
                "id": "outlet",
                "name": "Outlet",
                "face": "xmax",
                "flow": {"type": "pressure", "value": 0.0},
                "thermal": {"type": "adiabatic"},
            },
            {
                "id": "heated-wall",
                "name": "Heated wall",
                "face": "zmax",
                "flow": {"type": "wall"},
                "thermal": {"type": "heat_flux", "value": 1000.0},
            },
        ],
        "mesh": {"cells": [30, 10, 10]},
        "study": {"steps": 200, "output_interval": 20, "snapshot_interval": 20, "dt": 0.01, "device": "cpu", "gpu_batch_cells": 65536, "monitor": {"tolerance": 1.0e-5, "consecutive_samples": 5}},
    }


def validate_project(project: Any) -> dict[str, Any]:
    """Validate and return a normalised copy of a project.

    ``ValueError`` is used consistently so the HTTP layer can return a useful
    client error.  The input object is never mutated.
    """

    if not isinstance(project, Mapping):
        _fail("project must be an object")
    source = copy.deepcopy(dict(project))
    _keys(source, {"schema_version", "name", "geometry", "materials", "physics", "boundaries", "mesh", "study"}, "project")
    for key in ("schema_version", "name", "geometry", "materials", "physics", "boundaries", "mesh", "study"):
        if key not in source:
            _fail(f"project.{key} is required")
    if isinstance(source["schema_version"], (bool, np.bool_)) or source["schema_version"] != SCHEMA_VERSION:
        _fail(f"project.schema_version must be {SCHEMA_VERSION}")

    name = _text(source["name"], "project.name", max_length=200)

    geometry = _mapping(source["geometry"], "geometry")
    _keys(
        geometry,
        {
            "kind", "size", "asset_id", "solid_asset_id", "role", "origin", "computational_box", "domain",
            "domain_size", "domain_origin", "box_size", "box_origin",
            "solid_material_id", "solids",
        },
        "geometry",
    )
    kind = geometry.get("kind")
    if kind not in ("box", "cad"):
        _fail("geometry.kind must be 'box' or 'cad'")
    role = geometry.get("role")
    if role not in ("fluid", "obstacle"):
        _fail("geometry.role must be 'fluid' or 'obstacle'")
    if kind == "box" and "size" not in geometry:
        _fail("geometry.size is required for a box")
    if "size" in geometry:
        geometry_size = _vector(geometry["size"], "geometry.size", positive=True)
    else:
        geometry_size = None
    asset_id = geometry.get("asset_id")
    if asset_id is not None:
        asset_id = _identifier(asset_id, "geometry.asset_id")
    if kind == "cad" and not asset_id:
        _fail("geometry.asset_id is required for CAD geometry")
    if kind == "box" and asset_id is not None:
        _fail("geometry.asset_id is only valid for CAD geometry")
    solid_asset_id = geometry.get("solid_asset_id")
    if solid_asset_id is not None:
        solid_asset_id = _identifier(solid_asset_id, "geometry.solid_asset_id")
        if kind != "cad" or role != "fluid":
            _fail("geometry.solid_asset_id is only valid for CAD fluid geometry")
        if solid_asset_id == asset_id:
            _fail("geometry.solid_asset_id must identify a separate CAD asset")
    origin = _vector(geometry["origin"], "geometry.origin") if "origin" in geometry else None
    solid_material_id = geometry.get("solid_material_id")
    if solid_material_id is not None:
        solid_material_id = _identifier(solid_material_id, "geometry.solid_material_id")
        if role == "obstacle":
            pass
        elif kind == "cad" and role == "fluid":
            if solid_asset_id is None:
                _fail("geometry.solid_material_id requires geometry.solid_asset_id for CAD fluid geometry")
        else:
            _fail("geometry.solid_material_id is only valid for obstacle geometry or paired CAD fluid geometry")
    if solid_asset_id is not None and solid_material_id is None:
        _fail("geometry.solid_asset_id requires geometry.solid_material_id")
    raw_solids = geometry.get("solids", [])
    if isinstance(raw_solids, (str, bytes)) or not isinstance(raw_solids, list):
        _fail("geometry.solids must be an array")
    solids: list[dict[str, Any]] = []
    solid_ids: set[str] = set()
    for index, solid_value in enumerate(raw_solids):
        normalized_solid = _normalise_solid_region(solid_value, index)
        if normalized_solid["id"] in solid_ids:
            _fail(f"duplicate solid region id: {normalized_solid['id']}")
        solid_ids.add(normalized_solid["id"])
        solids.append(normalized_solid)
    # Validate aliases and the canonical obstacle descriptor even if build_mesh
    # will later resolve the imported CAD bounds.
    descriptor = _computational_box(geometry)
    normalized_geometry: dict[str, Any] = {"kind": kind}
    if geometry_size is not None:
        normalized_geometry["size"] = geometry_size
    normalized_geometry["asset_id"] = asset_id
    normalized_geometry["solid_asset_id"] = solid_asset_id
    normalized_geometry["role"] = role
    normalized_geometry["solids"] = solids
    if solid_material_id is not None:
        normalized_geometry["solid_material_id"] = solid_material_id
    if origin is not None:
        normalized_geometry["origin"] = origin
    if "computational_box" in geometry:
        normalized_geometry["computational_box"] = {"size": descriptor[0], "origin": descriptor[1]}  # type: ignore[index]
    elif "domain" in geometry:
        normalized_geometry["domain"] = {"size": descriptor[0], "origin": descriptor[1]}  # type: ignore[index]
    else:
        for key in ("domain_size", "domain_origin", "box_size", "box_origin"):
            if key in geometry:
                if key.endswith("size"):
                    normalized_geometry[key] = _vector(geometry[key], f"geometry.{key}", positive=True)
                else:
                    normalized_geometry[key] = _vector(geometry[key], f"geometry.{key}")

    _validate_solid_region_bounds(normalized_geometry, solids)

    materials_value = source["materials"]
    if isinstance(materials_value, (str, bytes)) or not isinstance(materials_value, list) or not materials_value:
        _fail("materials must be a non-empty array")
    material_ids: set[str] = set()
    materials: list[dict[str, Any]] = []
    material_keys = {"id", "name", "density", "viscosity", "heat_capacity", "conductivity"}
    for index, material_value in enumerate(materials_value):
        label = f"materials[{index}]"
        material = _mapping(material_value, label)
        _keys(material, material_keys, label)
        for key in material_keys:
            if key not in material:
                _fail(f"{label}.{key} is required")
        material_id = _identifier(material["id"], f"{label}.id")
        if material_id in material_ids:
            _fail(f"duplicate material id: {material_id}")
        material_ids.add(material_id)
        materials.append(
            {
                "id": material_id,
                "name": _text(material["name"], f"{label}.name"),
                "density": _property(material["density"], f"{label}.density", require_positive=True, temperature_positive=True),
                "viscosity": _property(material["viscosity"], f"{label}.viscosity", require_positive=True, temperature_positive=True),
                "heat_capacity": _property(material["heat_capacity"], f"{label}.heat_capacity", require_positive=True, temperature_positive=True),
                "conductivity": _property(material["conductivity"], f"{label}.conductivity", require_positive=True, temperature_positive=True),
            }
        )

    physics = _mapping(source["physics"], "physics")
    _keys(physics, {"flow", "thermal", "material_id", "initial_temperature", "gravity", "turbulence"}, "physics")
    for key in ("flow", "thermal", "material_id", "initial_temperature"):
        if key not in physics:
            _fail(f"physics.{key} is required")
    if not _is_bool(physics["flow"]) or not _is_bool(physics["thermal"]):
        _fail("physics.flow and physics.thermal must be booleans")
    material_id = _identifier(physics["material_id"], "physics.material_id")
    if material_id not in material_ids:
        _fail(f"physics.material_id references unknown material: {material_id}")
    initial_temperature = _positive(physics["initial_temperature"], "physics.initial_temperature")
    normalized_physics = {
        "flow": physics["flow"],
        "thermal": physics["thermal"],
        "material_id": material_id,
        "initial_temperature": initial_temperature,
        "turbulence": _normalise_turbulence(physics["turbulence"] if "turbulence" in physics else {}),
    }
    # Gravity was added after the original project/checkpoint format.  Keep
    # the key absent when an older project omitted it so its normalized
    # representation remains byte-for-byte compatible with old consumers.
    if "gravity" in physics:
        normalized_physics["gravity"] = normalise_gravity(physics["gravity"])

    if solid_material_id is not None and solid_material_id not in material_ids:
        _fail(f"geometry.solid_material_id references unknown material: {solid_material_id}")
    for solid in solids:
        if solid["material_id"] not in material_ids:
            _fail(f"geometry.solids[{solid['id']}] references unknown material: {solid['material_id']}")

    boundaries_value = source["boundaries"]
    if isinstance(boundaries_value, (str, bytes)) or not isinstance(boundaries_value, list):
        _fail("boundaries must be an array")
    boundaries: list[dict[str, Any]] = []
    boundary_ids: set[str] = set()
    boundary_faces: set[str] = set()
    for index, boundary_value in enumerate(boundaries_value):
        normalized_boundary = _normalise_boundary(boundary_value, index, kind)
        if normalized_boundary["id"] in boundary_ids:
            _fail(f"duplicate boundary id: {normalized_boundary['id']}")
        if normalized_boundary["face"] in boundary_faces:
            _fail(f"duplicate boundary face: {normalized_boundary['face']}")
        boundary_ids.add(normalized_boundary["id"])
        boundary_faces.add(normalized_boundary["face"])
        boundaries.append(normalized_boundary)

    mesh_source = _mapping(source["mesh"], "mesh")
    target_cells = _mesh_target(mesh_source)
    if target_cells is None:
        cells = _mesh_cells(mesh_source)
        _validate_box_spacing(normalized_geometry, cells)
        normalized_mesh = {"cells": cells}
    else:
        # A target is authoritative.  Keep validating a stale legacy vector's
        # shape and integer values when present, but do not let its anisotropy
        # or old memory estimate reject the target request.
        if "cells" in mesh_source:
            _mesh_cells({"cells": mesh_source["cells"]}, enforce_limits=False)
        outer_descriptor = _outer_domain_descriptor(normalized_geometry)
        if outer_descriptor is not None:
            outer_size, outer_origin = outer_descriptor
            planned = plan_box(outer_size, target_cells, origin=outer_origin)
            normalized_mesh = {
                "target_cells": target_cells,
                # Known box domains can carry the resolved shape for older
                # consumers while the target remains the source of truth.
                "cells": list(planned["shape"]),
            }
        else:
            # CAD fluid bounds are asset metadata and are resolved by
            # geometry.build_mesh/estimate_mesh.  Do not use a stale
            # geometry.size or legacy cells as a proxy here.
            normalized_mesh = {"target_cells": target_cells}

    study = _mapping(source["study"], "study")
    _keys(study, {"steps", "output_interval", "snapshot_interval", "dt", "device", "gpu_batch_cells", "monitor"}, "study")
    for key in ("steps", "output_interval", "dt", "device"):
        if key not in study:
            _fail(f"study.{key} is required")
    steps = _integer(study["steps"], "study.steps", minimum=1)
    output_interval = _integer(study["output_interval"], "study.output_interval", minimum=1)
    if output_interval > steps:
        _fail("study.output_interval cannot exceed study.steps")
    snapshot_interval = _integer(study.get("snapshot_interval", 0), "study.snapshot_interval", minimum=0)
    dt = _positive(study["dt"], "study.dt")
    device = study["device"]
    if not isinstance(device, str) or device not in {"cpu", "cuda:0", "cuda:0-ram"}:
        _fail("study.device must be 'cpu', 'cuda:0', or 'cuda:0-ram'")
    gpu_batch_cells = _integer(study.get("gpu_batch_cells", 65536), "study.gpu_batch_cells", minimum=1024)
    if gpu_batch_cells > 1048576:
        _fail("study.gpu_batch_cells cannot exceed 1048576")
    monitor = None
    if "monitor" in study:
        descriptor = _mapping(study["monitor"], "study.monitor")
        _keys(descriptor, {"tolerance", "consecutive_samples"}, "study.monitor")
        tolerance = _positive(descriptor.get("tolerance", 1.0e-5), "study.monitor.tolerance")
        consecutive = _integer(descriptor.get("consecutive_samples", 5), "study.monitor.consecutive_samples", minimum=1)
        if consecutive > 100:
            _fail("study.monitor.consecutive_samples cannot exceed 100")
        monitor = {"tolerance": tolerance, "consecutive_samples": consecutive}

    # Check the BGK relaxation range when a physical box spacing is known.
    # CAD bounds are loaded by geometry.build_mesh, so their spacing is checked
    # there before the same calculation is used by the solver.
    spacing: float | None = None
    if target_cells is None:
        known_domain = _outer_domain_descriptor(normalized_geometry)
        if known_domain is not None:
            domain_size = np.asarray(known_domain[0], dtype=float)
            spacing = float(domain_size[0] / cells[0])
    elif normalized_mesh.get("cells") is not None:
        known_domain = _outer_domain_descriptor(normalized_geometry)
        if known_domain is not None:
            spacing = float(np.asarray(known_domain[0], dtype=float)[0] / normalized_mesh["cells"][0])
    if spacing is not None and physics["flow"]:
        material = next(item for item in materials if item["id"] == material_id)
        density_temperature = initial_temperature
        gravity = normalized_physics.get("gravity")
        if gravity and gravity["enabled"] and gravity["mode"] == "buoyancy":
            density_temperature = gravity["reference_temperature"]
        rho = _property_value_without_warning(material["density"], density_temperature)
        mu = _property_value_without_warning(material["viscosity"], initial_temperature)
        tau = 0.5 + 3.0 * (mu / rho) * dt / (spacing * spacing)
        if not math.isfinite(tau) or not 0.500001 < tau <= 5.0:
            _fail(f"study.dt gives an unstable BGK relaxation time tau={tau:g}; require 0.500001 < tau <= 5")

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "geometry": normalized_geometry,
        "materials": materials,
        "physics": normalized_physics,
        "boundaries": boundaries,
        "mesh": normalized_mesh,
        "study": {"steps": steps, "output_interval": output_interval, "snapshot_interval": snapshot_interval, "dt": dt, "device": device, "gpu_batch_cells": gpu_batch_cells, **({"monitor": monitor} if monitor is not None else {})},
    }


def evaluate_property(prop: Any, temperature: Any) -> float | np.ndarray:
    """Evaluate a constant or linearly interpolated temperature table.

    Values outside a table's temperature range are clamped to the nearest
    endpoint and produce :class:`PropertyClampWarning`.  Scalar temperatures
    return ``float``; array-like temperatures return a float NumPy array.
    """

    # Public property evaluation follows the project convention that the
    # independent variable is an absolute temperature in Kelvin.
    normalized = _property(prop, "property", require_positive=False, temperature_positive=True)
    if isinstance(temperature, (str, bytes)):
        _fail("temperature must be a finite scalar or numeric array")
    try:
        values = np.asarray(temperature)
    except (TypeError, ValueError):
        _fail("temperature must be a finite scalar or numeric array")
    if values.dtype.kind not in "biuf":
        _fail("temperature must be a finite scalar or numeric array")
    if values.dtype.kind == "b":
        _fail("temperature must be a finite scalar or numeric array")
    try:
        temperatures = values.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError):
        _fail("temperature must be a finite scalar or numeric array")
    if not np.isfinite(temperatures).all():
        _fail("temperature must contain only finite values")
    if np.any(temperatures <= 0.0):
        _fail("temperature must be positive Kelvin values")

    if normalized["kind"] == "constant":
        evaluated = np.full(temperatures.shape, float(normalized["value"]), dtype=float)
    else:
        points = np.asarray(normalized["points"], dtype=float)
        below = temperatures < points[0, 0]
        above = temperatures > points[-1, 0]
        if np.any(below) or np.any(above):
            sides: list[str] = []
            if np.any(below):
                sides.append(f"below {points[0, 0]:g} K")
            if np.any(above):
                sides.append(f"above {points[-1, 0]:g} K")
            warnings.warn(
                "temperature property table clamped for values " + " and ".join(sides),
                PropertyClampWarning,
                stacklevel=2,
            )
        evaluated = np.interp(temperatures, points[:, 0], points[:, 1]).astype(float, copy=False)
    if evaluated.ndim == 0:
        return float(evaluated)
    return evaluated


__all__ = [
    "MAX_AXIS_CELLS",
    "MAX_TOTAL_CELLS",
    "PropertyClampWarning",
    "SCHEMA_VERSION",
    "default_project",
    "evaluate_property",
    "normalise_gravity",
    "validate_project",
]
