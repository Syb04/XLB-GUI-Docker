"""Versioned, validated restart archives for the workbench solver.

The restart format deliberately stores the evolved lattice distribution rather
than a field reconstructed from the final plot.  ``f`` is kept at the solver's
native FP32 precision and ``temperature`` is kept in FP64 so that a resumed
run starts from the same state as the run that produced the checkpoint.

The archive is an ordinary ``.npz`` file.  A small ``metadata`` member is
read directly from the ZIP container by :func:`read_restart_metadata`; this
keeps status polling independent of the potentially large field arrays.
"""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np


RESTART_VERSION = 1
_FORMAT = "xlb-workbench-restart"
_METADATA_MEMBER = "metadata.npy"
_REQUIRED_METADATA = ("version", "step", "dt", "shape")
_SCHEDULE_KEYS = {"steps", "output_interval", "snapshot_interval", "device", "dt", "gpu_batch_cells", "monitor"}
_F_DTYPE = np.dtype(np.float32)
_TEMPERATURE_DTYPE = np.dtype(np.float64)


def _json_value(value: Any, *, remove_asset_id: bool = False) -> Any:
    """Return a deterministic JSON-compatible representation.

    Project inputs normally originate in JSON, but callers and tests often
    retain NumPy scalars/arrays.  Normalising those here makes the fingerprint
    independent of the in-memory container types.  ``asset_id`` is an opaque
    server-local CAD handle; the solver-effective mesh hash protects the
    actual imported geometry, so it is intentionally excluded.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            key_text = str(key)
            if remove_asset_id and key_text == "asset_id":
                continue
            result[key_text] = _json_value(value[key], remove_asset_id=remove_asset_id)
        return result
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), remove_asset_id=remove_asset_id)
    if isinstance(value, np.generic):
        return _json_value(value.item(), remove_asset_id=remove_asset_id)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, remove_asset_id=remove_asset_id) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("project fingerprint cannot contain non-finite values")
        # JSON's representation is stable across the supported Python
        # versions and preserves the input value exactly enough for matching.
        return value
    raise ValueError(f"project contains unsupported fingerprint value {type(value).__name__}")


def _project_fingerprint_payload(project: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(project, Mapping):
        raise ValueError("project must be an object")
    # Keep non-scheduling study keys for forward compatibility.  The current
    # schema has only the keys in _SCHEDULE_KEYS, while dt is checked
    # separately with exact compatibility below.
    source = dict(project)
    physics = source.get("physics")
    if isinstance(physics, Mapping) and isinstance(physics.get("gravity"), Mapping) and not physics["gravity"].get("enabled", False):
        # An explicitly disabled new setting is physically identical to old
        # archives which predate the gravity option.
        source["physics"] = {key: value for key, value in physics.items() if key != "gravity"}
    study = source.get("study")
    if isinstance(study, Mapping):
        retained_study = {key: value for key, value in study.items() if str(key) not in _SCHEDULE_KEYS}
        if retained_study:
            source["study"] = retained_study
        else:
            source.pop("study", None)
    else:
        source.pop("study", None)
    return _json_value(source, remove_asset_id=True)


def project_fingerprint(project: Mapping[str, Any]) -> str:
    """Hash solver inputs while excluding study scheduling and CAD handles."""

    payload = _project_fingerprint_payload(project)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mesh_effective_arrays(mesh: Mapping[str, Any]) -> tuple[tuple[int, int, int], dict[str, np.ndarray]]:
    """Normalise the arrays that can affect a solver step.

    Surface vertices/faces and importer metadata are deliberately ignored.
    Boundary links are canonicalised to ``(6, nx, ny, nz)`` and masked to
    fluid cells, matching :func:`workbench.solver._mesh_arrays`.
    """

    if not isinstance(mesh, Mapping):
        raise ValueError("mesh must be an object")
    if "fluid_mask" not in mesh:
        raise ValueError("mesh must contain fluid_mask")
    fluid = np.asarray(mesh["fluid_mask"], dtype=bool)
    if fluid.ndim != 3 or not fluid.any():
        raise ValueError("mesh.fluid_mask must be a non-empty 3-D array")
    shape = tuple(int(item) for item in fluid.shape)
    solid = np.asarray(mesh.get("solid_mask", np.zeros(shape, dtype=bool)), dtype=bool)
    if solid.shape != shape:
        raise ValueError("mesh.solid_mask must have the same shape as fluid_mask")
    if "thermal_mask" in mesh:
        thermal = np.asarray(mesh["thermal_mask"], dtype=bool)
    else:
        thermal = solid | fluid if "solid_mask" in mesh else fluid.copy()
    if thermal.shape != shape:
        raise ValueError("mesh.thermal_mask must have the same shape as fluid_mask")
    material = np.asarray(mesh.get("material_index", np.full(shape, -1, dtype=np.int64)), dtype=np.int64)
    if material.shape != shape:
        raise ValueError("mesh.material_index must have the same shape as fluid_mask")
    try:
        origin = np.asarray(mesh.get("origin", (0.0, 0.0, 0.0)), dtype=np.float64)
        spacing = np.asarray(mesh.get("spacing", ()), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh origin and spacing must be numeric vectors") from exc
    if origin.shape != (3,) or spacing.shape != (3,):
        raise ValueError("mesh origin and spacing must be three-vectors")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("mesh origin and spacing must be finite, with positive spacing")

    links: dict[str, np.ndarray] = {}
    raw_links = mesh.get("boundary_links", {})
    if raw_links is None:
        raw_links = {}
    if not isinstance(raw_links, Mapping):
        raise ValueError("mesh.boundary_links must be an object")
    expected = (6, *shape)
    for selector, value in raw_links.items():
        selector_name = str(selector).lower()
        array = np.asarray(value, dtype=bool)
        if array.shape == (*shape, 6):
            array = np.moveaxis(array, -1, 0)
        if array.shape != expected:
            raise ValueError(f"mesh.boundary_links[{selector!r}] has the wrong shape")
        links[selector_name] = np.ascontiguousarray(array & fluid[None, ...])

    arrays = {
        "fluid_mask": np.ascontiguousarray(fluid),
        "solid_mask": np.ascontiguousarray(solid),
        "thermal_mask": np.ascontiguousarray(thermal),
        "material_index": np.ascontiguousarray(material),
        "origin": np.ascontiguousarray(origin),
        "spacing": np.ascontiguousarray(spacing),
    }
    for selector in sorted(links):
        arrays[f"boundary_links:{selector}"] = links[selector]
    return shape, arrays


def mesh_fingerprint(mesh: Mapping[str, Any]) -> str:
    """Hash shape, coordinates, masks, material indices, and boundary links."""

    shape, arrays = _mesh_effective_arrays(mesh)
    digest = hashlib.sha256()
    digest.update(b"xlb-workbench-effective-mesh-v1\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _metadata_json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("restart metadata must be a scalar JSON string")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("restart metadata must be a JSON string")
    return value


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    try:
        result = json.loads(_metadata_json(metadata))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("restart metadata is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("restart metadata must be a JSON object")
    missing = [key for key in _REQUIRED_METADATA if key not in result]
    if missing:
        raise ValueError(f"restart metadata is missing {', '.join(missing)}")
    version = result.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != RESTART_VERSION:
        raise ValueError(f"unsupported restart version {version!r}")
    step = result.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("restart metadata step must be a non-negative integer")
    try:
        dt = float(result.get("dt"))
    except (TypeError, ValueError) as exc:
        raise ValueError("restart metadata dt must be finite and positive") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("restart metadata dt must be finite and positive")
    shape = result.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError("restart metadata shape must contain three dimensions")
    normalized_shape: list[int] = []
    for item in shape:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 1:
            raise ValueError("restart metadata shape must contain positive integers")
        normalized_shape.append(int(item))
    result["version"] = int(version)
    result["step"] = int(step)
    result["dt"] = dt
    result["shape"] = normalized_shape
    # These fields are required for our solver-produced archives.  A strict
    # check here ensures a hand-edited metadata member cannot make a corrupt
    # archive look resumable.
    if result.get("format") != _FORMAT:
        raise ValueError("restart metadata format is invalid")
    if result.get("f_dtype") != _F_DTYPE.str:
        raise ValueError("restart metadata f_dtype is invalid")
    if result.get("temperature_dtype") != _TEMPERATURE_DTYPE.str:
        raise ValueError("restart metadata temperature_dtype is invalid")
    for key in ("project_fingerprint", "mesh_fingerprint"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"restart metadata {key} is invalid")
    status = result.get("status")
    if status is not None and status not in {"completed", "stopped"}:
        raise ValueError("restart metadata status must be completed or stopped")
    return result


def _read_metadata_member(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            try:
                info = archive.getinfo(_METADATA_MEMBER)
            except KeyError as exc:
                raise ValueError("restart archive has no metadata member") from exc
            # Metadata is intentionally tiny.  Refuse an unexpectedly large
            # member so a polling request cannot be used to inflate memory.
            if info.file_size > 1 << 20:
                raise ValueError("restart metadata is too large")
            raw = archive.read(info)
        array = np.load(BytesIO(raw), allow_pickle=False)
        return _validate_metadata(array)
    except zipfile.BadZipFile as exc:
        raise ValueError("restart archive is not a valid npz file") from exc
    except OSError:
        raise
    except (EOFError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("restart metadata cannot be read") from exc


def read_restart_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate only the small metadata member of a restart archive."""

    return _read_metadata_member(Path(path))


def _validate_array_fields(
    fields: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(item) for item in metadata["shape"])
    try:
        f = np.asarray(fields["f"])
        temperature = np.asarray(fields["temperature"])
    except KeyError as exc:
        raise ValueError(f"restart archive is missing {exc.args[0]} array") from exc
    if f.dtype != _F_DTYPE:
        raise ValueError(f"restart f must have dtype {_F_DTYPE}, got {f.dtype}")
    if temperature.dtype != _TEMPERATURE_DTYPE:
        raise ValueError(f"restart temperature must have dtype {_TEMPERATURE_DTYPE}, got {temperature.dtype}")
    if f.shape != (27, *shape):
        raise ValueError(f"restart f must have shape {(27, *shape)}, got {f.shape}")
    if temperature.shape != shape:
        raise ValueError(f"restart temperature must have shape {shape}, got {temperature.shape}")
    if not np.isfinite(f).all():
        raise ValueError("restart f contains non-finite values")
    if not np.isfinite(temperature).all():
        raise ValueError("restart temperature contains non-finite values")
    return np.ascontiguousarray(f), np.ascontiguousarray(temperature)


def load_restart(
    path: str | Path,
    *,
    project: Mapping[str, Any] | None = None,
    mesh: Mapping[str, Any] | None = None,
    dt: float | None = None,
    shape: tuple[int, int, int] | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """Load a restart archive and validate it against current run inputs.

    Project/mesh/dt/target-step checks happen while the process is still on
    the host.  The returned arrays can therefore be copied to a JAX device
    only after all compatibility and integrity checks have passed.
    """

    archive_path = Path(path)
    metadata = _read_metadata_member(archive_path)
    if dt is not None:
        try:
            requested_dt = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("current study.dt must be finite and positive") from exc
        if not math.isfinite(requested_dt) or requested_dt <= 0.0 or requested_dt != metadata["dt"]:
            raise ValueError("restart checkpoint dt does not match current study.dt")
    checkpoint_shape = tuple(metadata["shape"])
    if shape is not None and tuple(int(item) for item in shape) != checkpoint_shape:
        raise ValueError("restart checkpoint shape does not match current mesh")
    if steps is not None:
        if isinstance(steps, bool) or int(steps) < metadata["step"]:
            raise ValueError("study.steps must be at least the restart checkpoint step")
    if project is not None and project_fingerprint(project) != metadata["project_fingerprint"]:
        raise ValueError("restart checkpoint project physics/geometry/material/boundary inputs do not match")
    if mesh is not None and mesh_fingerprint(mesh) != metadata["mesh_fingerprint"]:
        raise ValueError("restart checkpoint mesh does not match current mesh")

    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in ("f", "temperature") if key in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"restart fields cannot be read: {exc}") from exc
    f, temperature = _validate_array_fields(fields, metadata)
    return {"metadata": metadata, "f": f, "temperature": temperature}


def write_restart(
    path: str | Path,
    *,
    step: int,
    dt: float,
    f: Any,
    temperature: Any,
    project: Mapping[str, Any] | None = None,
    mesh: Mapping[str, Any] | None = None,
    project_fingerprint_value: str | None = None,
    mesh_fingerprint_value: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Atomically write a solver restart archive and return its metadata."""

    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 0:
        raise ValueError("restart step must be a non-negative integer")
    try:
        dt_value = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError("restart dt must be finite and positive") from exc
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("restart dt must be finite and positive")
    f_array = np.asarray(f)
    temperature_array = np.asarray(temperature)
    if f_array.dtype != _F_DTYPE:
        raise ValueError(f"restart f must have dtype {_F_DTYPE}, got {f_array.dtype}")
    if temperature_array.dtype != _TEMPERATURE_DTYPE:
        raise ValueError(f"restart temperature must have dtype {_TEMPERATURE_DTYPE}, got {temperature_array.dtype}")
    if f_array.ndim != 4 or f_array.shape[0] != 27 or temperature_array.ndim != 3:
        raise ValueError("restart fields have invalid dimensions")
    if f_array.shape[1:] != temperature_array.shape:
        raise ValueError("restart f and temperature shapes do not match")
    if not np.isfinite(f_array).all() or not np.isfinite(temperature_array).all():
        raise ValueError("restart fields must be finite")
    if project_fingerprint_value is None:
        if project is None:
            raise ValueError("project or project_fingerprint_value is required")
        project_fingerprint_value = project_fingerprint(project)
    if mesh_fingerprint_value is None:
        if mesh is None:
            raise ValueError("mesh or mesh_fingerprint_value is required")
        mesh_fingerprint_value = mesh_fingerprint(mesh)
    # Validate the supplied hashes before writing; this also catches accidental
    # use of a human-readable label in place of a digest.
    for name, value in (("project_fingerprint", project_fingerprint_value), ("mesh_fingerprint", mesh_fingerprint_value)):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{name} is invalid")
    metadata: dict[str, Any] = {
        "format": _FORMAT,
        "version": RESTART_VERSION,
        "step": int(step),
        "dt": dt_value,
        "shape": [int(item) for item in temperature_array.shape],
        "f_dtype": _F_DTYPE.str,
        "temperature_dtype": _TEMPERATURE_DTYPE.str,
        "project_fingerprint": project_fingerprint_value,
        "mesh_fingerprint": mesh_fingerprint_value,
    }
    if status is not None:
        if status not in {"completed", "stopped"}:
            raise ValueError("restart status must be completed or stopped")
        metadata["status"] = status
    _validate_metadata(json.dumps(metadata, separators=(",", ":")))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                f=np.ascontiguousarray(f_array),
                temperature=np.ascontiguousarray(temperature_array),
                metadata=np.asarray(metadata_json),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


__all__ = [
    "RESTART_VERSION",
    "load_restart",
    "mesh_fingerprint",
    "project_fingerprint",
    "read_restart_metadata",
    "write_restart",
]
