"""Read transient and final workbench results.

The solver writes only the fields which are expensive to produce (velocity,
pressure, temperature, and eddy viscosity) in a frame.  This module keeps the
HTTP layer small by resolving the published frame first and deriving material
and diagnostic fields on demand.  All paths accepted from a frame manifest
are deliberately stricter than normal filesystem paths: a frame is exactly a
single ``step_<nine digits>.npz`` file below ``frames``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
import zipfile

import numpy as np


_FRAME_NAME = re.compile(r"^step_(\d{9})\.npz$")
_STATIC_KEYS = ("fluid_mask", "solid_mask", "thermal_mask", "material_index", "origin", "spacing", "dx")

# Keep this order and spelling in sync with the browser result selector.  The
# ``mask`` value names the class of cells in which the field is meaningful.
FIELD_SPECS = (
    {"id": "temperature", "label": "温度", "unit": "K", "mask": "thermal"},
    {"id": "speed", "label": "速度", "unit": "m/s", "mask": "fluid"},
    {"id": "velocity_x", "label": "速度 X", "unit": "m/s", "mask": "fluid"},
    {"id": "velocity_y", "label": "速度 Y", "unit": "m/s", "mask": "fluid"},
    {"id": "velocity_z", "label": "速度 Z", "unit": "m/s", "mask": "fluid"},
    {"id": "pressure", "label": "圧力", "unit": "Pa", "mask": "fluid"},
    {"id": "density", "label": "材料密度", "unit": "kg/m³", "mask": "thermal"},
    {"id": "eddy_viscosity", "label": "渦粘性", "unit": "m²/s", "mask": "fluid"},
    {"id": "vorticity", "label": "渦度", "unit": "s⁻¹", "mask": "fluid"},
    {"id": "vorticity_x", "label": "渦度 X", "unit": "s⁻¹", "mask": "fluid"},
    {"id": "vorticity_y", "label": "渦度 Y", "unit": "s⁻¹", "mask": "fluid"},
    {"id": "vorticity_z", "label": "渦度 Z", "unit": "s⁻¹", "mask": "fluid"},
    {"id": "heat_flux", "label": "熱流束", "unit": "W/m²", "mask": "thermal"},
    {"id": "heat_flux_x", "label": "熱流束 X", "unit": "W/m²", "mask": "thermal"},
    {"id": "heat_flux_y", "label": "熱流束 Y", "unit": "W/m²", "mask": "thermal"},
    {"id": "heat_flux_z", "label": "熱流束 Z", "unit": "W/m²", "mask": "thermal"},
    {"id": "conductivity", "label": "熱伝導率", "unit": "W/(m·K)", "mask": "thermal"},
    {"id": "dynamic_viscosity", "label": "粘性係数", "unit": "Pa·s", "mask": "fluid"},
    {"id": "heat_capacity", "label": "比熱容量", "unit": "J/(kg·K)", "mask": "thermal"},
)
_FIELD_BY_ID = {item["id"]: item for item in FIELD_SPECS}
_ALIASES = {
    "velocity_xyz": "velocity_xyz",
    "velocity": "velocity_xyz",
    "vorticity_magnitude": "vorticity",
    "heat_flux_magnitude": "heat_flux",
    "viscosity": "dynamic_viscosity",
    "mu": "dynamic_viscosity",
    "cp": "heat_capacity",
    "k": "conductivity",
}
_STATIC_CACHE: dict[tuple[tuple[str, int, int], ...], "StaticFields"] = {}


@dataclass
class StaticFields:
    """Static geometry arrays used by both base and derived result fields."""

    shape: tuple[int, int, int]
    fluid_mask: np.ndarray
    solid_mask: np.ndarray
    thermal_mask: np.ndarray
    material_index: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    source: Path | None = None


def field_specs() -> list[dict[str, str]]:
    """Return a copy of the public field metadata."""

    return [dict(item) for item in FIELD_SPECS]


def _read_input(directory: Path) -> dict[str, Any]:
    path = directory / "input.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _npz_values(path: Path, keys: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    """Read only requested NPZ members and detach them from the archive."""

    values: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            if key in archive:
                values[key] = np.array(archive[key], copy=True)
    return values


def _vector(value: Any, name: str, *, positive: bool = False, default: Any = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is missing")
        value = default
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), 3)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _input_spacing(project: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    geometry = project.get("geometry") if isinstance(project.get("geometry"), Mapping) else {}
    mesh = project.get("mesh") if isinstance(project.get("mesh"), Mapping) else {}
    cells = mesh.get("cells")
    size = geometry.get("size")
    if cells is None or size is None:
        return None
    try:
        cells_array = np.asarray(cells, dtype=float)
        size_array = np.asarray(size, dtype=float)
        if cells_array.shape != (3,) or size_array.shape != (3,):
            return None
        if np.any(cells_array <= 0.0) or np.any(size_array <= 0.0):
            return None
        origin = _vector(geometry.get("origin"), "geometry.origin", default=(0.0, 0.0, 0.0))
        return origin, size_array / cells_array
    except (TypeError, ValueError, OverflowError):
        return None


def _static_candidates(directory: Path, final_path: Path | None = None) -> list[Path]:
    paths = [directory / "frames" / "static.npz", directory / "mesh.npz"]
    if final_path is not None:
        paths.append(final_path)
    else:
        paths.append(directory / "fields.npz")
    return paths


def load_static(directory: str | Path, final_path: str | Path | None = None) -> StaticFields:
    """Load static masks from frames, mesh, or the legacy final fields file.

    Keys are collected one by one, allowing a mesh written by an older
    solver to supply masks while a newer final fields file supplies a missing
    ``material_index``.  Dynamic arrays are never read here.
    """

    directory = Path(directory).resolve()
    selected_final = Path(final_path).resolve() if final_path is not None else None
    candidates = _static_candidates(directory, selected_final)
    # Once a run has a static frame/mesh file, its masks are independent of
    # the selected dynamic frame.  Excluding that frame from the cache key is
    # significant for large meshes during timeline playback.
    signature_candidates = candidates[:2] if any(path.is_file() for path in candidates[:2]) else candidates[2:]
    signature: list[tuple[str, int, int]] = []
    for candidate in signature_candidates:
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if candidate.is_file():
            signature.append((str(candidate), int(stat.st_mtime_ns), int(stat.st_size)))
    cache_key = tuple(signature)
    cached = _STATIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values: dict[str, np.ndarray] = {}
    source: Path | None = None
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            if not candidate.resolve().is_relative_to(directory):
                continue
        except OSError:
            continue
        try:
            found = _npz_values(candidate, _STATIC_KEYS)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            # A final artifact may be in the middle of a non-atomic legacy
            # write while a run is still executing.  Continue to a safe older
            # source when one is available.
            continue
        if found and source is None:
            source = candidate
        for key, value in found.items():
            values.setdefault(key, value)
        if (
            {"fluid_mask", "solid_mask", "thermal_mask", "material_index", "origin"}.issubset(values)
            and ("spacing" in values or "dx" in values)
        ):
            break

    if "fluid_mask" not in values:
        raise FileNotFoundError("No result mesh or fields snapshot exists")
    fluid = np.asarray(values["fluid_mask"], dtype=bool)
    if fluid.ndim != 3 or not fluid.any():
        raise ValueError("fluid_mask must be a non-empty 3-D array")
    shape = tuple(int(value) for value in fluid.shape)

    solid = np.asarray(values.get("solid_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    if solid.shape != shape:
        raise ValueError("solid_mask has a different shape from fluid_mask")
    if "thermal_mask" in values:
        thermal = np.asarray(values["thermal_mask"], dtype=bool)
        if thermal.shape != shape:
            raise ValueError("thermal_mask has a different shape from fluid_mask")
    else:
        # A legacy all-fluid file has no conductive solid cells.  If a solid
        # mask is present, treating it as thermal preserves old conjugate runs
        # whose writer predated the explicit thermal mask.
        thermal = fluid | solid if "solid_mask" in values else fluid.copy()
    if np.any(fluid & ~thermal) or np.any(solid & ~thermal):
        raise ValueError("thermal_mask must contain fluid and solid cells")

    material_index = np.asarray(values.get("material_index", np.full(shape, -1, dtype=np.int64)), dtype=np.int64)
    if material_index.shape != shape:
        raise ValueError("material_index has a different shape from fluid_mask")
    origin = values.get("origin")
    spacing = values.get("spacing", values.get("dx"))
    project = _read_input(directory)
    inferred = _input_spacing(project)
    if origin is None:
        origin = inferred[0] if inferred is not None else (0.0, 0.0, 0.0)
    if spacing is None:
        spacing = inferred[1] if inferred is not None else (1.0, 1.0, 1.0)
    origin = _vector(origin, "origin")
    spacing = _vector(spacing, "spacing", positive=True)
    result = StaticFields(shape, fluid, solid, thermal, material_index, origin, spacing, source)
    if cache_key:
        _STATIC_CACHE[cache_key] = result
    return result


def _manifest(directory: Path) -> list[dict[str, Any]]:
    """Return manifest entries whose exact frame file is present."""

    path = directory / "frames.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("frames", []), list):
        raise ValueError("Invalid frames manifest")
    frames_root = (directory / "frames").resolve()
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in value["frames"]:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid frame entry")
        step = item.get("step")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or int(step) < 0:
            raise ValueError("Frame step must be a non-negative integer")
        step = int(step)
        if step in seen:
            raise ValueError("Duplicate frame step")
        filename = item.get("file")
        if not isinstance(filename, str) or _FRAME_NAME.fullmatch(filename) is None:
            raise ValueError("Invalid frame file")
        if int(_FRAME_NAME.fullmatch(filename).group(1)) != step:
            raise ValueError("Frame file does not match step")
        candidate = (frames_root / filename).resolve()
        if not candidate.is_relative_to(frames_root) or not candidate.is_file():
            # An entry is not publicly usable until its NPZ exists.  This is
            # also tolerant of archives made while a writer was interrupted.
            continue
        raw_time = item.get("time")
        if raw_time is None:
            raw_time = step * float(_read_input(directory).get("study", {}).get("dt", 0.0) or 0.0)
        if isinstance(raw_time, (bool, np.bool_)):
            raise ValueError("Frame time must be finite")
        try:
            time_value = float(raw_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Frame time must be finite") from exc
        if not math.isfinite(time_value):
            raise ValueError("Frame time must be finite")
        seen.add(step)
        entries.append({"step": step, "time": time_value, "file": filename})
    return sorted(entries, key=lambda item: item["step"])


def frames_result(directory: str | Path) -> dict[str, Any]:
    """Return published frame metadata and the available field contract."""

    directory = Path(directory).resolve()
    entries = _manifest(directory)
    static: StaticFields | None = None
    try:
        static = load_static(directory)
    except (FileNotFoundError, ValueError):
        # A queued run can be listed before the mesher has created any result
        # artifact.  The field contract is still useful to the UI.
        static = None
    result: dict[str, Any] = {
        "frames": [{"step": item["step"], "time": item["time"]} for item in entries],
        "fields": field_specs(),
    }
    if static is not None:
        result.update(
            shape=list(static.shape),
            origin=static.origin.tolist(),
            spacing=static.spacing.tolist(),
            dx=float(static.spacing[0]),
            fluid_cells=int(np.count_nonzero(static.fluid_mask)),
            solid_cells=int(np.count_nonzero(static.solid_mask)),
            thermal_cells=int(np.count_nonzero(static.thermal_mask)),
        )
    result["final_available"] = (directory / "fields.npz").is_file()
    return result


def _resolve_field(field: Any) -> tuple[str, str]:
    if not isinstance(field, str):
        raise ValueError("Unknown field")
    value = _ALIASES.get(field, field)
    if value == "velocity_xyz":
        return value, "velocity_xyz"
    if value not in _FIELD_BY_ID:
        raise ValueError("Unknown field")
    return value, value


def _frame_source(directory: Path, query: Mapping[str, list[str]]) -> tuple[Path, int | None, float | None]:
    """Resolve a final artifact or an exact published frame."""

    step_values = query.get("step")
    if step_values is not None:
        if len(step_values) != 1:
            raise ValueError("step must be one integer")
        raw = step_values[0]
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
            raise ValueError("step must be a non-negative integer")
        step = int(raw)
        if step > 999_999_999:
            raise ValueError("step is too large")
        entries = {item["step"]: item for item in _manifest(directory)}
        item = entries.get(step)
        if item is None:
            raise FileNotFoundError("Requested timestep is not published")
        # _manifest has already checked the filename and containment.
        path = (directory / "frames" / item["file"]).resolve()
        return path, step, float(item["time"])
    path = directory / "fields.npz"
    if not path.is_file():
        raise FileNotFoundError("Final result does not exist")
    return path, _final_step(directory), _final_time(directory)


def _final_step(directory: Path) -> int | None:
    for name in ("result.json", "status.json"):
        path = directory / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            raw = value.get("steps")
            if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool) and int(raw) >= 0:
                return int(raw)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _final_time(directory: Path) -> float | None:
    step = _final_step(directory)
    if step is None:
        return None
    project = _read_input(directory)
    try:
        dt = float(project.get("study", {}).get("dt", 0.0))
        return step * dt if math.isfinite(dt) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _materials(project: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int, dict[str, int]]:
    raw = project.get("materials")
    materials = [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []
    by_id = {str(item.get("id")): index for index, item in enumerate(materials) if item.get("id") is not None}
    selected_id = project.get("physics", {}).get("material_id") if isinstance(project.get("physics"), Mapping) else None
    selected = by_id.get(str(selected_id), 0) if materials else 0
    return materials, selected, by_id


def _property_value(prop: Any, temperature: np.ndarray, name: str) -> np.ndarray:
    if prop is None:
        raise ValueError(f"input.json does not contain material {name} property")
    if isinstance(prop, Mapping):
        kind = str(prop.get("kind", "constant")).lower()
        value = prop.get("value") if kind == "constant" else prop.get("points")
    else:
        kind, value = "constant", prop
    if kind == "constant":
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} property is not numeric") from exc
        if not math.isfinite(scalar):
            raise ValueError(f"{name} property is not finite")
        return np.full(temperature.shape, scalar, dtype=float)
    if kind != "table":
        raise ValueError(f"Unsupported {name} property kind")
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} property table is invalid") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2 or not np.isfinite(points).all():
        raise ValueError(f"{name} property table is invalid")
    if np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValueError(f"{name} property table temperatures are not increasing")
    return np.interp(temperature, points[:, 0], points[:, 1]).astype(float, copy=False)


def _material_field(
    project: Mapping[str, Any], static: StaticFields, temperature: np.ndarray, name: str
) -> np.ndarray:
    if temperature.shape != static.shape:
        raise ValueError("temperature has a different shape from the result mesh")
    materials, selected, by_id = _materials(project)
    if not materials:
        raise ValueError("input.json material properties are unavailable")
    active = static.thermal_mask
    indices = np.asarray(static.material_index, dtype=np.int64).copy()
    geometry = project.get("geometry") if isinstance(project.get("geometry"), Mapping) else {}
    solids = geometry.get("solids") if isinstance(geometry, Mapping) else None
    missing_solid = static.solid_mask & (indices < 0)
    if missing_solid.any():
        # A list of CHT boxes may contain different materials.  Without the
        # mesh's per-cell index there is no safe way to recover which box a
        # solid cell belongs to, and a CAD-wide fallback would silently apply
        # the wrong material.  One material is the only unambiguous inference.
        if len(materials) == 1:
            indices[missing_solid] = 0
        elif isinstance(solids, list) and solids:
            raise ValueError("solid material_index is unavailable for geometry.solids")
        else:
            solid_id = geometry.get("solid_material_id") if isinstance(geometry, Mapping) else None
            if solid_id is None:
                raise ValueError("solid cells require material_index or geometry.solid_material_id")
            try:
                indices[missing_solid] = by_id[str(solid_id)]
            except KeyError as exc:
                raise ValueError("geometry.solid_material_id references an unknown material") from exc
    indices[active & (indices < 0)] = selected
    if np.any(indices[active] < 0) or np.any(indices[active] >= len(materials)):
        raise ValueError("material_index references an unknown material")
    result = np.zeros(static.shape, dtype=float)
    for index, material in enumerate(materials):
        selected_cells = active & (indices == index)
        if not selected_cells.any():
            continue
        result[selected_cells] = _property_value(
            material.get(name),
            np.asarray(temperature[selected_cells], dtype=float),
            name,
        )
    return result


def _field_array(archive: Any, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"This result does not contain the requested field: {name}")
    value = np.asarray(archive[name], dtype=float)
    if value.shape != shape:
        raise ValueError(f"{name} has a different shape from the result mesh")
    return value


def _velocity(archive: Any, shape: tuple[int, int, int]) -> np.ndarray:
    if "velocity" not in archive:
        raise ValueError("This result does not contain velocity")
    value = np.asarray(archive["velocity"], dtype=float)
    if value.shape != (3, *shape):
        raise ValueError("velocity must have shape (3,nx,ny,nz)")
    return value


def _masked_derivative(value: np.ndarray, active: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """One cell-centred derivative which does not cross inactive cells."""

    value = np.asarray(value, dtype=float)
    valid = np.asarray(active, dtype=bool) & np.isfinite(value)
    plus = np.roll(value, -1, axis=axis)
    minus = np.roll(value, 1, axis=axis)
    plus_valid = np.roll(valid, -1, axis=axis)
    minus_valid = np.roll(valid, 1, axis=axis)
    # np.roll wraps at the boundary; those neighbours do not exist.
    plus_slice = [slice(None)] * 3
    plus_slice[axis] = -1
    plus_valid[tuple(plus_slice)] = False
    minus_slice = [slice(None)] * 3
    minus_slice[axis] = 0
    minus_valid[tuple(minus_slice)] = False

    gradient = np.zeros(value.shape, dtype=float)
    central = valid & plus_valid & minus_valid
    forward = valid & plus_valid & ~minus_valid
    backward = valid & minus_valid & ~plus_valid
    gradient[central] = (plus[central] - minus[central]) / (2.0 * dx)
    gradient[forward] = (plus[forward] - value[forward]) / dx
    gradient[backward] = (value[backward] - minus[backward]) / dx
    return gradient


def _masked_gradient(value: np.ndarray, active: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all three cell-centred derivatives without crossing inactive cells."""

    return tuple(_masked_derivative(value, active, axis, float(spacing[axis])) for axis in range(3))  # type: ignore[return-value]


def _vorticity_component(velocity: np.ndarray, static: StaticFields, component: int) -> np.ndarray:
    flow_active = static.fluid_mask | static.solid_mask
    wall_velocity = np.where(flow_active[None, ...], velocity, 0.0)
    if component == 0:
        # ωx = ∂w/∂y − ∂v/∂z
        result = _masked_derivative(wall_velocity[2], flow_active, 1, float(static.spacing[1]))
        result -= _masked_derivative(wall_velocity[1], flow_active, 2, float(static.spacing[2]))
    elif component == 1:
        # ωy = ∂u/∂z − ∂w/∂x
        result = _masked_derivative(wall_velocity[0], flow_active, 2, float(static.spacing[2]))
        result -= _masked_derivative(wall_velocity[2], flow_active, 0, float(static.spacing[0]))
    else:
        # ωz = ∂v/∂x − ∂u/∂y
        result = _masked_derivative(wall_velocity[1], flow_active, 0, float(static.spacing[0]))
        result -= _masked_derivative(wall_velocity[0], flow_active, 1, float(static.spacing[1]))
    return np.where(static.fluid_mask, result, 0.0)


def _vorticity(velocity: np.ndarray, static: StaticFields) -> np.ndarray:
    """Compute vorticity in SI units, retaining no values in solids."""

    return np.stack([_vorticity_component(velocity, static, component) for component in range(3)], axis=0)


def _heat_flux(
    temperature: np.ndarray,
    conductivity: np.ndarray,
    static: StaticFields,
    component: int | None = None,
) -> np.ndarray:
    if component is None:
        gradients = _masked_gradient(temperature, static.thermal_mask, static.spacing)
        result = -conductivity[None, ...] * np.stack(gradients, axis=0)
        return np.where(static.thermal_mask[None, ...], result, 0.0)
    gradient = _masked_derivative(temperature, static.thermal_mask, component, float(static.spacing[component]))
    return np.where(static.thermal_mask, -conductivity * gradient, 0.0)


def _derive_field(field: str, archive: Any, static: StaticFields, project: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, bool]:
    shape = static.shape
    if field == "velocity_xyz":
        return _velocity(archive, shape), static.fluid_mask, True
    if field == "temperature":
        return _field_array(archive, "temperature", shape), static.thermal_mask, False
    if field == "speed":
        velocity = _velocity(archive, shape)
        return np.linalg.norm(velocity, axis=0), static.fluid_mask, False
    if field.startswith("velocity_"):
        velocity = _velocity(archive, shape)
        component = "xyz".index(field[-1])
        return velocity[component], static.fluid_mask, False
    if field == "pressure":
        return _field_array(archive, "pressure", shape), static.fluid_mask, False
    if field == "eddy_viscosity":
        return _field_array(archive, "eddy_viscosity", shape), static.fluid_mask, False
    if field in {"density", "conductivity", "dynamic_viscosity", "heat_capacity"}:
        temperature = _field_array(archive, "temperature", shape)
        name = {"dynamic_viscosity": "viscosity"}.get(field, field)
        active = static.thermal_mask if _FIELD_BY_ID[field]["mask"] == "thermal" else static.fluid_mask
        return _material_field(project, static, temperature, name), active, False
    if field.startswith("vorticity"):
        velocity = _velocity(archive, shape)
        if field == "vorticity":
            return np.linalg.norm(_vorticity(velocity, static), axis=0), static.fluid_mask, False
        component = "xyz".index(field[-1])
        return _vorticity_component(velocity, static, component), static.fluid_mask, False
    if field.startswith("heat_flux"):
        temperature = _field_array(archive, "temperature", shape)
        conductivity = _material_field(project, static, temperature, "conductivity")
        if field == "heat_flux":
            heat_flux = _heat_flux(temperature, conductivity, static)
            return np.linalg.norm(heat_flux, axis=0), static.thermal_mask, False
        component = "xyz".index(field[-1])
        return _heat_flux(temperature, conductivity, static, component), static.thermal_mask, False
    raise ValueError("Unknown field")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, bool)) or value is None:
        return value
    return float(value) if isinstance(value, (np.floating,)) and math.isfinite(float(value)) else None


def slice_result(directory: str | Path, query: Mapping[str, list[str]]) -> dict[str, Any]:
    """Read one scalar/vector plane from a published timestep or final file."""

    directory = Path(directory).resolve()
    field_values = query.get("field", ["temperature"])
    if not field_values:
        raise ValueError("Unknown field")
    field_raw = field_values[0]
    field, _ = _resolve_field(field_raw)
    axis_values = query.get("axis", ["z"])
    if not axis_values:
        raise ValueError("Unknown slice axis")
    axis = axis_values[0]
    if axis not in ("x", "y", "z"):
        raise ValueError("Unknown slice axis")
    final_path, step, time_value = _frame_source(directory, query)
    static = load_static(directory, final_path)
    project = _read_input(directory)
    with np.load(final_path, allow_pickle=False) as archive:
        values, active, vector = _derive_field(field, archive, static, project)
        idx = "xyz".index(axis)
        shape_for_index = values.shape[idx + 1] if vector else values.shape[idx]
        index_values = query.get("index")
        if index_values is None:
            index = shape_for_index // 2
        elif len(index_values) != 1:
            raise ValueError("index must be one integer")
        else:
            raw_index = index_values[0]
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_index):
                raise ValueError("Slice index must be an integer")
            index = int(raw_index)
        if not 0 <= index < shape_for_index:
            raise ValueError("Slice index outside mesh")

        if vector:
            plane = np.take(values, index, axis=idx + 1)
            plane = np.moveaxis(plane, 0, -1)
            plane_mask = np.take(active, index, axis=idx)
            masked = np.where(plane_mask[..., None], plane, np.nan)
            finite = masked[np.isfinite(masked)]
            magnitude = np.linalg.norm(masked, axis=-1)
            unit = "m/s"
            response_shape = list(values.shape)
        else:
            plane = np.take(values, index, axis=idx)
            plane_mask = np.take(active, index, axis=idx)
            masked = np.where(plane_mask, plane, np.nan)
            finite = masked[np.isfinite(masked)]
            magnitude = masked
            unit = _FIELD_BY_ID.get(field, {"unit": "m/s"})["unit"]
            response_shape = list(values.shape)
        solid_plane = np.take(static.solid_mask, index, axis=idx)
        thermal_plane = np.take(static.thermal_mask, index, axis=idx)
        fluid_plane = np.take(static.fluid_mask, index, axis=idx)
        finite_magnitude = magnitude[np.isfinite(magnitude)]
        return {
            "values": _json_value(masked),
            "min": float(finite_magnitude.min()) if finite_magnitude.size else 0.0,
            "max": float(finite_magnitude.max()) if finite_magnitude.size else 0.0,
            "solid_mask": solid_plane.tolist(),
            "thermal_mask": thermal_plane.tolist(),
            "fluid_mask": fluid_plane.tolist(),
            "mask": plane_mask.tolist(),
            "unit": unit,
            "field": field_raw,
            "field_id": field,
            "axis": axis,
            "index": index,
            "shape": response_shape,
            "extent": [
                float(static.origin[other])
                for other in range(3)
                if other != idx
            ]
            + [
                float(static.origin[other] + static.shape[other] * static.spacing[other])
                for other in range(3)
                if other != idx
            ],
            "step": step,
            "time": time_value,
        }


__all__ = ["FIELD_SPECS", "StaticFields", "field_specs", "frames_result", "load_static", "slice_result"]
