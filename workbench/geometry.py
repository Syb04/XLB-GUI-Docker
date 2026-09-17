"""CAD import and CPU voxel preparation for the XLB workbench.

The geometry API stores the original source beside an SI-normalised triangle
surface.  STL and OBJ files are read by :mod:`trimesh`; STEP, IGES and BREP
files are tessellated by the optional :mod:`gmsh` Python API.  Voxelisation is
cell-centre classification of that actual surface, with no synthetic proxy or
bounding-box substitution.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .schema import MAX_AXIS_CELLS, MAX_TOTAL_CELLS, validate_project
from .limits import MAX_BOUNDARY_LINK_BYTES
from .mesh_planning import plan_box, plan_cad


MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_FACES = 200_000
MAX_VERTICES = 300_000
MAX_COORDINATE_M = 1.0e12
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SCALE = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}
_MESH_EXTENSIONS = {".stl", ".obj"}
_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".brep"}


def _unit_scale(unit: str) -> tuple[str, float]:
    if not isinstance(unit, str):
        raise ValueError("unit must be one of m, cm, or mm")
    normalized = unit.strip().lower()
    if normalized not in _SCALE:
        raise ValueError("unit must be one of m, cm, or mm")
    return normalized, _SCALE[normalized]


def _safe_asset_id(value: Any) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError("asset_id is not a valid asset identifier")
    return value


def _source_path(path: str | Path) -> Path:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"CAD source does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError("CAD source file is empty")
    if size > MAX_SOURCE_BYTES:
        raise ValueError(f"CAD source exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MiB limit")
    return source


def _trimesh_module() -> Any:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # ImportError and missing native optional deps
        raise ImportError("STL/OBJ import and CAD containment require the optional 'trimesh' package") from exc
    return trimesh


def _load_mesh_source(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load STL/OBJ with trimesh while avoiding automatic repair operations."""

    trimesh = _trimesh_module()
    extension = path.suffix.lower()
    try:
        loaded = trimesh.load(path, file_type=extension[1:], force="scene", process=False)
    except TypeError:
        # Older trimesh versions do not accept all keyword combinations for a
        # few loaders; process=False remains the important invariant.
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as exc:
        raise ValueError(f"could not read {extension[1:].upper()} geometry: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("CAD scene contains no mesh geometry")
        # ``dump(concatenate=True)`` applies each scene transform before
        # concatenating.  No welding or repair is requested here; exact
        # duplicate welding is performed below for STL facet topology only.
        try:
            if hasattr(loaded, "to_geometry"):
                loaded = loaded.to_geometry()
            else:
                loaded = loaded.dump(concatenate=True)
        except Exception as exc:
            raise ValueError(f"could not combine CAD scene geometry: {exc}") from exc
        if isinstance(loaded, (list, tuple)):
            meshes = [item for item in loaded if isinstance(item, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("CAD scene contains no triangle mesh geometry")
            loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("CAD source did not produce a triangle mesh")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError("CAD source must contain triangular surface faces")
    original_vertex_count = int(len(loaded.vertices))
    if original_vertex_count > MAX_VERTICES or len(loaded.faces) > MAX_FACES:
        raise ValueError(f"CAD surface limits are {MAX_VERTICES} vertices and {MAX_FACES} triangles")
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64), original_vertex_count


def _load_gmsh_surface(path: Path) -> tuple[np.ndarray, np.ndarray, int, list[str], dict[str, str]]:
    """Tessellate a CAD solid and retain each native dimension-2 surface tag."""

    try:
        import gmsh  # type: ignore
    except Exception as exc:
        raise ImportError(
            "STEP/IGES/BREP import requires the optional 'gmsh' package and its native libGLU dependency"
        ) from exc

    was_initialized = False
    try:
        checker = getattr(gmsh, "isInitialized", None)
        was_initialized = bool(checker()) if checker is not None else False
        if not was_initialized:
            # gmsh registers signal handlers by default, which is invalid when
            # the HTTP importer runs in a worker thread.
            try:
                gmsh.initialize(interruptible=False)
            except TypeError:
                gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.open(str(path))
        gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ValueError("gmsh produced no CAD surface nodes")
        nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        tags = np.asarray(node_tags, dtype=np.int64)
        if len(nodes) > MAX_VERTICES:
            raise ValueError(f"CAD surface limits are {MAX_VERTICES} vertices and {MAX_FACES} triangles")
        index = {int(tag): position for position, tag in enumerate(tags.tolist())}
        triangles: list[np.ndarray] = []
        triangle_groups: list[str] = []
        group_names: dict[str, str] = {}
        entities = list(gmsh.model.getEntities(2))
        if entities:
            element_batches = []
            for dimension_entity, entity_tag in entities:
                if int(dimension_entity) != 2:
                    continue
                group_id = f"surface-{int(entity_tag)}"
                try:
                    native_name = str(gmsh.model.getEntityName(2, int(entity_tag)) or "").strip()
                except Exception:
                    native_name = ""
                group_names[group_id] = native_name or f"Surface {int(entity_tag)}"
                element_batches.append((group_id, gmsh.model.mesh.getElements(2, int(entity_tag))))
        else:
            # A few older gmsh builds do not expose CAD entities after import;
            # retain one deterministic group rather than dropping triangles.
            group_names["surface-1"] = "Surface 1"
            element_batches = [("surface-1", gmsh.model.mesh.getElements(2))]
        for group_id, elements in element_batches:
            element_types, _, element_nodes = elements
            for element_type, node_buffer in zip(element_types, element_nodes):
                try:
                    properties = gmsh.model.mesh.getElementProperties(int(element_type))
                    node_count = int(properties[3])
                    dimension = int(properties[1])
                except Exception:
                    continue
                if dimension != 2 or node_count < 3:
                    continue
                raw = np.asarray(node_buffer, dtype=np.int64)
                if raw.size % node_count:
                    raise ValueError("gmsh returned malformed surface elements")
                cells = raw.reshape(-1, node_count)
                # ElementOrder=1 normally gives triangles.  Keep the corner
                # nodes for quadratic triangles and fan triangulate a retained
                # surface quad/polygon so no CAD area is silently discarded.
                if node_count in (3, 6):
                    corners = cells[:, :3]
                elif node_count in (4, 9):
                    corners = np.concatenate((cells[:, :3], cells[:, 3:4]), axis=1)
                else:
                    fans = [np.column_stack((cells[:, 0], cells[:, index], cells[:, index + 1])) for index in range(1, node_count - 1)]
                    corners = np.concatenate(fans, axis=0)
                if sum(len(item) for item in triangles) + len(corners) > MAX_FACES:
                    raise ValueError(f"CAD surface limits are {MAX_VERTICES} vertices and {MAX_FACES} triangles")
                try:
                    triangles.append(np.asarray([[index[int(tag)] for tag in row] for row in corners], dtype=np.int64))
                except KeyError as exc:
                    raise ValueError("gmsh surface references an unknown node") from exc
                triangle_groups.extend([group_id] * len(corners))
        if not triangles:
            raise ValueError("gmsh produced no triangular CAD surface")
        faces = np.concatenate(triangles, axis=0)
        return nodes, faces, int(len(nodes)), triangle_groups, group_names
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"gmsh could not tessellate CAD source: {exc}") from exc
    finally:
        try:
            if not was_initialized:
                gmsh.finalize()
            else:
                gmsh.clear()
        except Exception:
            # Preserve the actual import error; cleanup should never hide it.
            pass


def _normalise_surface(vertices: Any, faces: Any, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Validate a surface and weld only exactly equal vertex coordinates."""

    try:
        raw_vertices = np.asarray(vertices)
        raw_faces = np.asarray(faces)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CAD surface arrays must be numeric") from exc
    if raw_vertices.dtype.kind not in "iuf" or raw_faces.dtype.kind not in "iuf":
        raise ValueError("CAD surface arrays must contain real numeric coordinates and integer face indices")
    try:
        vertices_array = raw_vertices.astype(np.float64, copy=False)
        if raw_faces.dtype.kind == "f" and (not np.isfinite(raw_faces).all() or not np.all(raw_faces == np.trunc(raw_faces))):
            raise ValueError("CAD face indices must be integers")
        if raw_faces.size and (
            np.any(raw_faces > np.iinfo(np.int64).max)
            or np.any(raw_faces < np.iinfo(np.int64).min)
        ):
            raise ValueError("CAD face indices exceed the supported integer range")
        faces_array = raw_faces.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CAD surface arrays must be numeric with integer face indices") from exc
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError("CAD vertices must have shape (N, 3)")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("CAD faces must have shape (M, 3)")
    if len(vertices_array) > MAX_VERTICES or len(faces_array) > MAX_FACES:
        raise ValueError(f"CAD surface limits are {MAX_VERTICES} vertices and {MAX_FACES} triangles")
    if len(faces_array) < 4:
        raise ValueError("CAD surface must contain at least four triangles")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("CAD unit scale must be positive and finite")
    vertices_array = vertices_array * float(scale)
    if not np.isfinite(vertices_array).all() or np.max(np.abs(vertices_array)) > MAX_COORDINATE_M:
        raise ValueError("CAD coordinates must be finite and within the supported range")
    if np.any(faces_array < 0) or np.any(faces_array >= len(vertices_array)):
        raise ValueError("CAD face index is outside the vertex array")

    # Binary STL stores each facet's three corners independently.  Exact
    # welding restores shared topology without changing any geometric point or
    # silently repairing near gaps.
    used = vertices_array[faces_array.reshape(-1)]
    unique_vertices, inverse = np.unique(used, axis=0, return_inverse=True)
    normalized_faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
    if np.any(normalized_faces[:, 0] == normalized_faces[:, 1]) or np.any(normalized_faces[:, 1] == normalized_faces[:, 2]) or np.any(normalized_faces[:, 0] == normalized_faces[:, 2]):
        raise ValueError("CAD surface contains a zero-area triangle")
    if len(np.unique(np.sort(normalized_faces, axis=1), axis=0)) != len(normalized_faces):
        raise ValueError("CAD surface contains duplicate triangles")
    triangles = unique_vertices[normalized_faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    extent = float(np.linalg.norm(np.ptp(unique_vertices, axis=0)))
    if extent <= 0.0 or not math.isfinite(extent):
        raise ValueError("CAD surface has no finite three-dimensional extent")
    areas_twice = np.linalg.norm(cross, axis=1)
    if np.any(areas_twice <= max(extent * extent * 1.0e-14, 1.0e-30)):
        raise ValueError("CAD surface contains a zero-area or extremely small triangle")
    if np.any(np.ptp(unique_vertices, axis=0) <= max(extent * 1.0e-12, 1.0e-15)):
        raise ValueError("CAD surface must span all three spatial axes")
    return unique_vertices.astype(np.float64, copy=False), normalized_faces


def _surface_group_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    triangle_groups: list[str],
    group_names: Mapping[str, str] | None = None,
    native_tags: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe summaries for the supplied per-triangle patch labels."""

    if len(triangle_groups) != len(faces):
        raise ValueError("CAD triangle group metadata does not match the surface")
    names = group_names or {}
    native = native_tags or {}
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    centroids = triangles.mean(axis=1)
    summaries: list[dict[str, Any]] = []
    # Preserve the first appearance order.  For derived groups this is the
    # deterministic patch-number order; for gmsh this is the native entity
    # order returned by the kernel.
    indices_by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(triangle_groups):
        indices_by_group.setdefault(str(group_id), []).append(index)
    ordered_ids = list(indices_by_group)
    for group_id in ordered_ids:
        indices = np.asarray(indices_by_group[group_id], dtype=np.int64)
        weights = areas[indices]
        area = float(weights.sum())
        if area > 0.0:
            centroid = np.average(centroids[indices], axis=0, weights=weights)
            normal_vector = cross[indices].sum(axis=0)
        else:  # _normalise_surface already rejects this, retained defensively.
            centroid = centroids[indices].mean(axis=0)
            normal_vector = cross[indices].sum(axis=0)
        normal_length = float(np.linalg.norm(normal_vector))
        if normal_length <= 1.0e-30:
            normal_vector = cross[indices[0]]
            normal_length = float(np.linalg.norm(normal_vector))
        normal_vector = normal_vector / normal_length
        bounds = np.asarray([vertices[faces[indices]].min(axis=(0, 1)), vertices[faces[indices]].max(axis=(0, 1))])
        summary: dict[str, Any] = {
            "id": str(group_id),
            "name": str(names.get(group_id, group_id)),
            "triangle_count": int(len(indices)),
            "area": area,
            "normal": normal_vector.astype(float).tolist(),
            "centroid": np.asarray(centroid, dtype=float).tolist(),
            "center": np.asarray(centroid, dtype=float).tolist(),
            "bounds": bounds.astype(float).tolist(),
            "axis_aligned": bool(np.max(np.abs(normal_vector)) >= 1.0 - 1.0e-6),
        }
        if group_id in native:
            summary["native_tag"] = native[group_id]
        summaries.append(summary)
    return summaries


def _derive_surface_groups(vertices: np.ndarray, faces: np.ndarray) -> tuple[list[dict[str, Any]], list[str]]:
    """Group connected coplanar STL/OBJ triangles into selectable patches."""

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(cross, axis=1)
    normals = cross / normal_lengths[:, None]
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    plane_tolerance = max(diagonal * 1.0e-7, 1.0e-12)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces.tolist()):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)

    parent = np.arange(len(faces), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for adjacent in edge_to_faces.values():
        if len(adjacent) < 2:
            continue
        first = adjacent[0]
        for second in adjacent[1:]:
            if abs(float(np.dot(normals[first], normals[second]))) < 1.0 - 1.0e-6:
                continue
            plane = normals[first]
            offset = float(np.dot(plane, triangles[first, 0]))
            if float(np.max(np.abs(triangles[second] @ plane - offset))) <= plane_tolerance:
                union(first, second)

    components: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        components.setdefault(find(face_index), []).append(face_index)
    ordered_components = sorted(components.values(), key=lambda values: min(values))
    triangle_groups = [""] * len(faces)
    group_names: dict[str, str] = {}
    for number, indices in enumerate(ordered_components, start=1):
        group_id = f"patch-{number}"
        group_names[group_id] = f"Patch {number}"
        for index in indices:
            triangle_groups[index] = group_id
    return _surface_group_summary(vertices, faces, triangle_groups, group_names), triangle_groups


def _normalise_native_groups(
    vertices: np.ndarray,
    faces: np.ndarray,
    triangle_groups: list[str],
    group_names: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize gmsh native group labels and add missing defensive labels."""

    if len(triangle_groups) != len(faces):
        raise ValueError("gmsh triangle group metadata does not match the surface")
    safe_groups: list[str] = []
    safe_names: dict[str, str] = {}
    safe_native: dict[str, Any] = {}
    for index, group_id in enumerate(triangle_groups):
        text = str(group_id)
        if not _ID_RE.fullmatch(text):
            text = f"surface-{index + 1}"
        safe_groups.append(text)
        safe_names[text] = str(group_names.get(group_id, group_names.get(text, text)))
        suffix = text.removeprefix("surface-")
        if suffix.isdigit():
            safe_native[text] = int(suffix)
    return _surface_group_summary(vertices, faces, safe_groups, safe_names, safe_native), safe_groups


def _surface_status(vertices: np.ndarray, faces: np.ndarray) -> tuple[bool, bool]:
    trimesh = _trimesh_module()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return bool(mesh.is_watertight), bool(mesh.is_winding_consistent)


def _require_closed_surface(vertices: np.ndarray, faces: np.ndarray) -> None:
    watertight, winding_consistent = _surface_status(vertices, faces)
    if not watertight:
        raise ValueError("CAD volume classification requires a watertight closed surface")
    if not winding_consistent:
        raise ValueError("CAD volume classification requires consistently oriented surface faces")


def _write_metadata(asset_root: Path, metadata: Mapping[str, Any]) -> None:
    target = asset_root / "metadata.json"
    temporary = asset_root / f".metadata-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def import_cad(path: str | Path, asset_dir: str | Path, unit: str = "m") -> dict[str, Any]:
    """Import and persist a CAD surface, returning JSON-compatible metadata.

    ``unit`` is the unit of the source coordinates.  The returned vertices and
    bounds are always metres; the untouched source is copied under the asset
    directory for later export and audit.
    """

    source = _source_path(path)
    normalized_unit, scale = _unit_scale(unit)
    extension = source.suffix.lower()
    native_triangle_groups: list[str] | None = None
    native_group_names: dict[str, str] | None = None
    if extension in _MESH_EXTENSIONS:
        vertices, faces, original_vertex_count = _load_mesh_source(source)
    elif extension in _CAD_EXTENSIONS:
        vertices, faces, original_vertex_count, native_triangle_groups, native_group_names = _load_gmsh_surface(source)
    else:
        supported = ", ".join(sorted(_MESH_EXTENSIONS | _CAD_EXTENSIONS))
        raise ValueError(f"unsupported CAD extension {extension!r}; use {supported}")
    raw_vertices = np.asarray(vertices)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or len(raw_vertices) == 0:
        raise ValueError("CAD source did not produce a non-empty (N, 3) vertex array")
    original_bounds = np.asarray([raw_vertices.min(axis=0), raw_vertices.max(axis=0)], dtype=np.float64).tolist()
    vertices, faces = _normalise_surface(vertices, faces, scale)
    bounds_array = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float64)
    watertight, winding_consistent = _surface_status(vertices, faces)
    if native_triangle_groups is None:
        surface_groups, triangle_groups = _derive_surface_groups(vertices, faces)
    else:
        surface_groups, triangle_groups = _normalise_native_groups(
            vertices,
            faces,
            native_triangle_groups,
            native_group_names or {},
        )

    root = Path(asset_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    asset_id = uuid.uuid4().hex
    directory = root / asset_id
    directory.mkdir()
    source_name = f"source{extension}"
    try:
        shutil.copyfile(source, directory / source_name)
        np.savez_compressed(directory / "surface.npz", vertices=vertices, faces=faces)
        metadata: dict[str, Any] = {
            "id": asset_id,
            "original_name": source.name,
            "source_extension": extension,
            "source_file": source_name,
            "unit": normalized_unit,
            "scale_to_m": scale,
            "original_bounds": original_bounds,
            "original_vertex_count": int(original_vertex_count),
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(faces)),
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "bounds": bounds_array.tolist(),
            "bounds_m": bounds_array.tolist(),
            "watertight": watertight,
            "winding_consistent": winding_consistent,
            "surface_file": "surface.npz",
            "surface_groups": surface_groups,
            "triangle_groups": triangle_groups,
        }
        _write_metadata(directory, metadata)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return metadata


def _asset_surface(asset_root: Path, asset_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    asset_id = _safe_asset_id(asset_id)
    root = asset_root.resolve()
    directory = (root / asset_id).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("invalid CAD asset path")
    metadata_path = directory / "metadata.json"
    surface_path = directory / "surface.npz"
    if not metadata_path.exists() or not surface_path.exists():
        raise FileNotFoundError(f"CAD asset does not exist: {asset_id}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata.json must contain an object")
        with np.load(surface_path, allow_pickle=False) as data:
            if "vertices" in data and "faces" in data:
                vertices = np.asarray(data["vertices"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            elif "vertices_m" in data and "faces" in data:
                vertices = np.asarray(data["vertices_m"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
            else:
                raise ValueError("CAD surface.npz must contain vertices and faces")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ValueError(f"CAD asset {asset_id} is invalid: {exc}") from exc
    vertices, faces = _normalise_surface(vertices, faces, 1.0)
    raw_triangle_groups = metadata.get("triangle_groups")
    raw_surface_groups = metadata.get("surface_groups")
    if isinstance(raw_triangle_groups, list) and len(raw_triangle_groups) == len(faces):
        triangle_groups = [str(value) for value in raw_triangle_groups]
        group_names = {
            str(group.get("id")): str(group.get("name", group.get("id")))
            for group in raw_surface_groups or []
            if isinstance(group, Mapping) and group.get("id") is not None
        }
        surface_groups = _surface_group_summary(vertices, faces, triangle_groups, group_names)
    else:
        surface_groups, triangle_groups = _derive_surface_groups(vertices, faces)
    metadata = dict(metadata)
    metadata["surface_groups"] = surface_groups
    metadata["triangle_groups"] = triangle_groups
    return vertices, faces, metadata


def load_asset_metadata(asset_dir: str | Path, asset_id: str) -> dict[str, Any]:
    """Read an asset manifest and lazily enrich older assets with patch data.

    Older workbench assets contain only the normalized surface and basic bounds.
    This read-only helper derives the same ``surface_groups`` and
    ``triangle_groups`` used by :func:`build_mesh`, allowing asset listings and
    preview endpoints to expose selectable CAD faces without rewriting files.
    """

    vertices, faces, metadata = _asset_surface(Path(asset_dir), asset_id)
    enriched = dict(metadata)
    enriched["vertices"] = vertices.tolist()
    enriched["faces"] = faces.tolist()
    enriched.setdefault("bounds", np.asarray([vertices.min(axis=0), vertices.max(axis=0)]).tolist())
    enriched.setdefault("bounds_m", enriched["bounds"])
    return enriched


def _spacing(size: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if size.shape != (3,) or cells.shape != (3,):
        raise ValueError("mesh size and cells must be three-component vectors")
    spacing = size / cells
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("mesh spacing must be positive and finite")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-7, atol=1.0e-15):
        raise ValueError("Cartesian mesh requires isotropic physical spacing")
    return spacing.astype(np.float64)


def _cad_grid(bounds: np.ndarray, requested_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], list[str]]:
    """Choose one physical cell size and round CAD dimensions conservatively.

    ``mesh.cells`` is a resolution request for arbitrary CAD.  The largest
    requested cell width normally becomes the common Cartesian spacing, capped
    at one third of the thinnest extent so slender solids retain interior cell
    centres.  Each axis is rounded up to an integer number of cells.  Any
    excess extent is distributed evenly on both sides so the first and last
    cell centres stay inside the CAD bounds instead of putting a whole padded
    face outside the volume.
    """

    extent = np.asarray(bounds[1] - bounds[0], dtype=np.float64)
    requested = np.asarray(requested_cells, dtype=np.float64)
    raw_spacing = extent / requested
    # A spacing larger than one third of the thinnest extent can leave every
    # cell centre outside a valid volume (a common failure for slender CAD).
    # Keep the requested coarse spacing where possible, but cap it so every
    # axis receives at least three interior samples.
    spacing_value = float(min(np.max(raw_spacing), np.min(extent) / 3.0))
    if not math.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("CAD bounds do not define a positive finite extent")
    shape_array = np.maximum(np.ceil(extent / spacing_value - 1.0e-12).astype(np.int64), 3)
    if np.any(shape_array > MAX_AXIS_CELLS) or int(np.prod(shape_array)) > MAX_TOTAL_CELLS:
        raise ValueError("CAD mesh exceeds the bounded workbench cell limit")
    spacing = np.full(3, spacing_value, dtype=np.float64)
    padding = shape_array.astype(float) * spacing - extent
    origin = np.asarray(bounds[0], dtype=np.float64) - 0.5 * padding
    shape = tuple(int(value) for value in shape_array)
    warnings: list[str] = []
    if not np.array_equal(shape_array, requested_cells.astype(np.int64)):
        warnings.append(
            "CAD mesh rounded to isotropic spacing; requested cells "
            f"{[int(value) for value in requested_cells]} became {list(shape)}"
        )
    if np.any(padding > max(spacing_value * 1.0e-8, 1.0e-15)):
        warnings.append("CAD mesh adds evenly centred exterior padding after isotropic rounding")
    return origin, spacing, shape, warnings


def _box_descriptor(geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    descriptor: Any = None
    for key in ("computational_box", "domain"):
        if key in geometry:
            descriptor = geometry[key]
            break
    if descriptor is not None:
        if not isinstance(descriptor, Mapping) or "size" not in descriptor:
            raise ValueError("geometry.computational_box.size is required")
        size = np.asarray(descriptor["size"], dtype=np.float64)
        origin = np.asarray(descriptor.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
    else:
        size_key = next((key for key in ("domain_size", "box_size") if key in geometry), None)
        if size_key is None:
            raise ValueError("obstacle geometry requires geometry.computational_box")
        origin_key = "domain_origin" if size_key == "domain_size" else "box_origin"
        size = np.asarray(geometry[size_key], dtype=np.float64)
        origin = np.asarray(geometry.get(origin_key, [0.0, 0.0, 0.0]), dtype=np.float64)
    if size.shape != (3,) or origin.shape != (3,) or not np.isfinite(size).all() or not np.isfinite(origin).all() or np.any(size <= 0.0):
        raise ValueError("computational box size and origin must be finite three-vectors with positive size")
    return size, origin


def _grid_points(origin: np.ndarray, spacing: np.ndarray, shape: tuple[int, int, int], start: int, stop: int) -> np.ndarray:
    ny, nz = shape[1], shape[2]
    flat = np.arange(start, stop, dtype=np.int64)
    x = flat // (ny * nz)
    y = (flat // nz) % ny
    z = flat % nz
    indices = np.column_stack((x, y, z)).astype(np.float64)
    return origin + (indices + 0.5) * spacing


def _contains_fallback(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Chunked Möller-Trumbore parity test used when trimesh lacks an index."""

    triangles = vertices[faces]
    direction = np.asarray([1.0, 0.3713906763541037, 0.1732050807568877], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge_2.shape), edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, h)
    valid = np.abs(determinant) > 1.0e-14
    inverse_det = np.zeros_like(determinant)
    inverse_det[valid] = 1.0 / determinant[valid]
    result = np.zeros(len(points), dtype=bool)
    chunk = 4096
    for start in range(0, len(points), chunk):
        sample = points[start : start + chunk]
        vector = sample[:, None, :] - triangles[None, :, 0, :]
        u = inverse_det[None, :] * np.einsum("cmj,mj->cm", np.cross(vector, np.broadcast_to(direction, edge_2.shape)), edge_2)
        q = np.cross(vector, edge_1[None, :, :])
        v = inverse_det[None, :] * np.einsum("j,cmj->cm", direction, q)
        distance = inverse_det[None, :] * np.einsum("mj,cmj->cm", edge_1, q)
        hit = valid[None, :] & (u >= -1.0e-10) & (v >= -1.0e-10) & (u + v <= 1.0 + 1.0e-10) & (distance > 1.0e-12)
        result[start : start + len(sample)] = np.count_nonzero(hit, axis=1) % 2 == 1
    return result


def _contains(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    trimesh = _trimesh_module()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        values = np.asarray(mesh.contains(points), dtype=bool)
        if values.shape != (len(points),):
            raise ValueError("trimesh containment returned an invalid result")
        return values
    except Exception as exc:
        # A normal trimesh installation without rtree cannot build the ray
        # index.  Keep a bounded CPU implementation for small/medium assets;
        # the face/cell limits above prevent unbounded allocations.
        combinations = len(points) * len(faces)
        if combinations > 25_000_000:
            raise ImportError(
                "CAD containment needs trimesh with rtree (or Warp CPU) for this surface and cell count"
            ) from exc
        return _contains_fallback(points, vertices, faces)


def _classify_surface(origin: np.ndarray, spacing: np.ndarray, shape: tuple[int, int, int], vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    _require_closed_surface(vertices, faces)
    total = math.prod(shape)
    labels = np.zeros(total, dtype=bool)
    chunk = 65_536
    for start in range(0, total, chunk):
        points = _grid_points(origin, spacing, shape, start, min(start + chunk, total))
        labels[start : start + len(points)] = _contains(points, vertices, faces)
    return labels.reshape(shape)


def _normalise_cells(project: Mapping[str, Any]) -> np.ndarray:
    raw = project["mesh"]["cells"]
    cells = np.asarray(raw, dtype=np.int64)
    if cells.shape != (3,) or np.any(cells <= 0) or not np.all(np.asarray(raw) == cells):
        raise ValueError("mesh.cells must be three positive integer values")
    if np.any(cells > MAX_AXIS_CELLS) or int(np.prod(cells)) > MAX_TOTAL_CELLS:
        raise ValueError("mesh cell count exceeds the bounded workbench limit")
    return cells


def _plan_mesh(
    project: Mapping[str, Any],
    assets_dir: str | Path,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, dict[str, Any]] | None]:
    """Resolve one mesh plan shared by :func:`estimate_mesh` and build_mesh.

    The returned CAD surface tuple is loaded only when CAD metadata is needed
    for planning; build_mesh reuses it for classification and contact links.
    No voxel-sized array is allocated here.
    """

    geometry = project["geometry"]
    mesh = project["mesh"]
    kind = geometry["kind"]
    role = geometry["role"]
    target = mesh.get("target_cells")
    surface: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    if kind == "cad":
        surface = _asset_surface(Path(assets_dir), geometry["asset_id"])

    if target is not None:
        if kind == "box" and role == "fluid":
            size = np.asarray(geometry["size"], dtype=np.float64)
            origin = np.asarray(geometry.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
            return plan_box(size, target, origin=origin), surface
        if role == "obstacle":
            if any(key in geometry for key in ("computational_box", "domain", "domain_size", "box_size")):
                domain_size, domain_origin = _box_descriptor(geometry)
            elif kind == "cad" and "size" in geometry:
                domain_size = np.asarray(geometry["size"], dtype=np.float64)
                if "origin" in geometry:
                    domain_origin = np.asarray(geometry["origin"], dtype=np.float64)
                else:
                    assert surface is not None
                    vertices = surface[0]
                    domain_origin = vertices.mean(axis=0) - domain_size / 2.0
            else:
                # Preserve the established error for a box obstacle without
                # an explicit computational box descriptor.
                domain_size, domain_origin = _box_descriptor(geometry)
            return plan_box(domain_size, target, origin=domain_origin), surface
        # CAD fluid target mode is metadata-driven.  In particular, an old
        # geometry.size must never replace the imported CAD bounds.
        assert surface is not None
        vertices = surface[0]
        bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float64)
        return plan_cad(bounds, target), surface

    cells = _normalise_cells(project)
    if kind == "box" and role == "fluid":
        size = np.asarray(geometry["size"], dtype=np.float64)
        origin = np.asarray(geometry.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        spacing = _spacing(size, cells.astype(np.float64))
        return {
            "shape": [int(value) for value in cells],
            "total_cells": int(np.prod(cells, dtype=np.int64)),
            "spacing": spacing.tolist(),
            "origin": origin.tolist(),
            "target_cells": None,
            "warnings": [],
        }, surface
    if kind == "box" and role == "obstacle":
        domain_size, domain_origin = _box_descriptor(geometry)
        spacing = _spacing(domain_size, cells.astype(np.float64))
        return {
            "shape": [int(value) for value in cells],
            "total_cells": int(np.prod(cells, dtype=np.int64)),
            "spacing": spacing.tolist(),
            "origin": domain_origin.tolist(),
            "target_cells": None,
            "warnings": [],
        }, surface

    assert surface is not None
    vertices, _faces, _metadata = surface
    bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float64)
    if role == "fluid":
        origin, spacing, shape, warnings = _cad_grid(bounds, cells)
        return {
            "shape": [int(value) for value in shape],
            "total_cells": int(np.prod(shape, dtype=np.int64)),
            "spacing": spacing.tolist(),
            "origin": origin.tolist(),
            "target_cells": None,
            "warnings": warnings,
        }, surface

    if any(key in geometry for key in ("computational_box", "domain", "domain_size", "box_size")):
        domain_size, domain_origin = _box_descriptor(geometry)
    else:
        if "size" not in geometry:
            raise ValueError("CAD obstacle requires geometry.size or geometry.computational_box")
        domain_size = np.asarray(geometry["size"], dtype=np.float64)
        if "origin" in geometry:
            domain_origin = np.asarray(geometry["origin"], dtype=np.float64)
        else:
            domain_origin = vertices.mean(axis=0) - domain_size / 2.0
    spacing = _spacing(domain_size, cells.astype(np.float64))
    return {
        "shape": [int(value) for value in cells],
        "total_cells": int(np.prod(cells, dtype=np.int64)),
        "spacing": spacing.tolist(),
        "origin": domain_origin.tolist(),
        "target_cells": None,
        "warnings": [f"CAD obstacle centroid [m]: {float(value):.9g}" for value in vertices.mean(axis=0)],
    }, surface


_BOUNDARY_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_BOUNDARY_NORMALS = {
    "xmin": np.asarray([-1.0, 0.0, 0.0]),
    "xmax": np.asarray([1.0, 0.0, 0.0]),
    "ymin": np.asarray([0.0, -1.0, 0.0]),
    "ymax": np.asarray([0.0, 1.0, 0.0]),
    "zmin": np.asarray([0.0, 0.0, -1.0]),
    "zmax": np.asarray([0.0, 0.0, 1.0]),
}


def _box_boundary_links(fluid_mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Represent Cartesian domain contacts in the public six-link layout."""

    shape = tuple(int(value) for value in fluid_mask.shape)
    links: dict[str, np.ndarray] = {}
    normals: dict[str, list[float]] = {}
    for direction, face in enumerate(_BOUNDARY_FACES):
        array = np.zeros((6, *shape), dtype=bool)
        axis = direction // 2
        index = 0 if direction % 2 == 0 else -1
        selector = [slice(None)] * 3
        selector[axis] = index
        array[(direction, *selector)] = fluid_mask[tuple(selector)]
        links[face] = array
        normals[face] = _BOUNDARY_NORMALS[face].tolist()
    return links, normals


def _region_cell_mask(
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    region_origin: np.ndarray,
    region_size: np.ndarray,
) -> np.ndarray:
    """Rasterize one axis-aligned solid region using cell-centre inclusion."""

    coordinates = [origin[axis] + (np.arange(shape[axis], dtype=float) + 0.5) * spacing[axis] for axis in range(3)]
    per_axis = [
        (coordinates[axis] >= region_origin[axis] - spacing[axis] * 1.0e-8)
        & (coordinates[axis] <= region_origin[axis] + region_size[axis] + spacing[axis] * 1.0e-8)
        for axis in range(3)
    ]
    return np.einsum("i,j,k->ijk", per_axis[0], per_axis[1], per_axis[2], optimize=True)


def _apply_solid_regions(
    fluid_mask: np.ndarray,
    solid_mask: np.ndarray,
    material_index: np.ndarray,
    geometry: Mapping[str, Any],
    origin: np.ndarray,
    spacing: np.ndarray,
    material_ids: Mapping[str, int],
    *,
    require_inside_fluid: bool,
) -> None:
    """Apply user-declared CHT boxes and assign their material indices."""

    for region in geometry.get("solids", []) or []:
        region_origin = np.asarray(region["origin"], dtype=float)
        region_size = np.asarray(region["size"], dtype=float)
        selected = _region_cell_mask(origin, spacing, fluid_mask.shape, region_origin, region_size)
        if not selected.any():
            raise ValueError(f"solid region {region['id']} contains no mesh cell centres; refine mesh.cells")
        if require_inside_fluid and np.any(selected & ~fluid_mask):
            raise ValueError(f"solid region {region['id']} is outside the CAD fluid volume")
        try:
            index = int(material_ids[region["material_id"]])
        except KeyError as exc:  # validate_project normally catches this first
            raise ValueError(f"solid region {region['id']} references an unknown material") from exc
        # Regions are non-overlapping by schema.  For a CAD obstacle, a user
        # box may coincide with imported solid cells to give that solid a CHT
        # material, so an already-solid cell is allowed and simply receives
        # the explicit region material.
        fluid_mask[selected] = False
        solid_mask[selected] = True
        material_index[selected] = index


def _closest_surface_points(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_mesh: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest surface points, distances, and triangle indices."""

    trimesh = _trimesh_module()
    mesh = surface_mesh if surface_mesh is not None else trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        closest, distance, triangle_index = trimesh.proximity.closest_point(mesh, points)
        return (
            np.asarray(closest, dtype=float),
            np.asarray(distance, dtype=float),
            np.asarray(triangle_index, dtype=np.int64),
        )
    except Exception as exc:
        # Keep the no-rtree path exact for small assets.  Large fallback jobs
        # would require an unbounded point-by-triangle temporary, so fail with
        # an actionable optional-dependency message instead.
        if len(points) * len(faces) > 25_000_000:
            raise ImportError(
                "CAD boundary mapping needs trimesh with rtree (or another spatial index) for this surface"
            ) from exc
        triangles = vertices[faces]
        closest = np.empty_like(points, dtype=float)
        distances = np.full(len(points), np.inf, dtype=float)
        triangle_index = np.full(len(points), -1, dtype=np.int64)
        # trimesh's triangle routine computes the exact closest point for
        # corresponding point/triangle pairs.  Iterate over point chunks and
        # broadcast each chunk over bounded triangle batches.
        for point_start in range(0, len(points), 256):
            point_chunk = points[point_start : point_start + 256]
            best_distance = np.full(len(point_chunk), np.inf, dtype=float)
            best_point = np.zeros_like(point_chunk)
            best_triangle = np.full(len(point_chunk), -1, dtype=np.int64)
            for face_start in range(0, len(triangles), 2048):
                face_chunk = triangles[face_start : face_start + 2048]
                pair_triangles = np.repeat(face_chunk, len(point_chunk), axis=0)
                pair_points = np.tile(point_chunk, (len(face_chunk), 1))
                pair_closest = np.asarray(trimesh.triangles.closest_point(pair_triangles, pair_points), dtype=float)
                pair_distance = np.linalg.norm(pair_closest - pair_points, axis=1).reshape(len(face_chunk), len(point_chunk))
                candidate_face = np.argmin(pair_distance, axis=0)
                candidate_distance = pair_distance[candidate_face, np.arange(len(point_chunk))]
                update = candidate_distance < best_distance
                if np.any(update):
                    best_distance[update] = candidate_distance[update]
                    closest_pairs = pair_closest.reshape(len(face_chunk), len(point_chunk), 3)
                    best_point[update] = closest_pairs[candidate_face, np.arange(len(point_chunk))][update]
                    best_triangle[update] = face_start + candidate_face[update]
            output_slice = slice(point_start, point_start + len(point_chunk))
            closest[output_slice] = best_point
            distances[output_slice] = best_distance
            triangle_index[output_slice] = best_triangle
        return closest, distances, triangle_index


def _exposed_link_indices(fluid_mask: np.ndarray, direction: int) -> np.ndarray:
    """Return fluid cells whose neighbour in one lattice direction is inactive."""

    axis = direction // 2
    positive = direction % 2 == 1
    neighbour = np.zeros_like(fluid_mask, dtype=bool)
    source = [slice(None)] * 3
    target = [slice(None)] * 3
    if positive:
        source[axis] = slice(1, None)
        target[axis] = slice(0, -1)
    else:
        source[axis] = slice(0, -1)
        target[axis] = slice(1, None)
    neighbour[tuple(target)] = fluid_mask[tuple(source)]
    return np.argwhere(fluid_mask & ~neighbour)


def _triangle_contact_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    triangle_group: list[str],
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    fluid_mask: np.ndarray,
    requested_groups: Iterable[str] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Map each exposed fluid link to its nearest imported surface triangle.

    The link set is generated from the actual fluid/inactive interface, so no
    link can point at a fluid neighbour.  A distance gate avoids assigning a
    padded computational-domain face to an unrelated CAD patch; a CAD face
    that lies on that domain border still maps because its link midpoint is on
    the surface.
    """

    dx = float(spacing[0])
    requested_groups = set(requested_groups)
    global_links = np.zeros((6, *shape), dtype=bool)
    present_groups = set(triangle_group)
    links: dict[str, np.ndarray] = {
        group_id: np.zeros((6, *shape), dtype=bool)
        for group_id in requested_groups
        if group_id in present_groups
    }
    face_groups = np.asarray(triangle_group, dtype=object)
    trimesh = _trimesh_module()
    surface_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    directions = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    # The midpoint of a link from a cell centre to its neighbour is at most
    # half a cell diagonal from a planar boundary.  The modest margin handles
    # tilted facets while keeping centred CAD padding excluded.
    contact_distance = max(1.25 * dx, 0.75 * math.sqrt(3.0) * dx)
    for direction in range(6):
        indices = _exposed_link_indices(fluid_mask, direction)
        if len(indices) == 0:
            continue
        for start in range(0, len(indices), 4_096):
            chunk = indices[start : start + 4_096]
            points = origin + (chunk.astype(float) + 0.5) * spacing + 0.5 * dx * directions[direction]
            _, distance, triangle_index = _closest_surface_points(points, vertices, faces, surface_mesh)
            accepted = np.isfinite(distance) & (triangle_index >= 0) & (distance <= contact_distance)
            if not np.any(accepted):
                continue
            accepted_indices = chunk[accepted]
            accepted_triangles = triangle_index[accepted]
            global_links[direction, accepted_indices[:, 0], accepted_indices[:, 1], accepted_indices[:, 2]] = True
            if links:
                accepted_groups = face_groups[accepted_triangles]
                for group_id, group_links in links.items():
                    selected = accepted_groups == group_id
                    if np.any(selected):
                        group_indices = accepted_indices[selected]
                        group_links[direction, group_indices[:, 0], group_indices[:, 1], group_indices[:, 2]] = True
    return global_links, links


def _cad_boundary_links(
    project: Mapping[str, Any],
    fluid_mask: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Build global and selector-local CAD contact masks.

    A global ``cad`` mask covers every imported surface patch.  A
    ``cad:<id>`` entry is emitted for every selector in the project, including
    unresolved selectors as a correctly shaped all-false array so callers can
    report the unresolved condition deterministically.
    """

    raw_groups = metadata.get("triangle_groups")
    if isinstance(raw_groups, list) and len(raw_groups) == len(faces):
        triangle_groups = [str(value) for value in raw_groups]
    else:
        _, triangle_groups = _derive_surface_groups(vertices, faces)
    requested_selectors = {
        str(boundary.get("face")).lower()[4:]
        for boundary in project.get("boundaries", [])
        if isinstance(boundary, Mapping) and str(boundary.get("face", "")).lower().startswith("cad:")
    }
    # One global CAD mask, six Cartesian border masks, and one mask for each
    # explicitly selected patch are dense bool arrays.  Check the upper bound
    # before allocating any of them; unresolved selectors share one zero mask
    # but are counted conservatively so a malformed project cannot request an
    # unbounded number of large arrays.
    estimated_bytes = 6 * int(np.prod(fluid_mask.shape, dtype=np.int64)) * (7 + len(requested_selectors))
    if estimated_bytes > MAX_BOUNDARY_LINK_BYTES:
        raise ValueError(
            f"CAD boundary links need {estimated_bytes / (1024**2):.1f} MiB; "
            f"limit is {MAX_BOUNDARY_LINK_BYTES / (1024**2):g} MiB "
            "(XLB_MAX_BOUNDARY_LINK_MIB); coarsen the mesh or select fewer CAD patches"
        )
    all_cad, groups = _triangle_contact_mask(
        vertices,
        faces,
        triangle_groups,
        origin,
        spacing,
        tuple(int(value) for value in fluid_mask.shape),
        fluid_mask,
        requested_selectors,
    )
    shape = tuple(int(value) for value in fluid_mask.shape)
    links, normals = _box_boundary_links(fluid_mask)
    links["cad"] = all_cad
    normals["cad"] = [0.0, 0.0, 0.0]
    group_summary_by_id = {
        str(item.get("id")): item
        for item in metadata.get("surface_groups", [])
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    for group_id, value in groups.items():
        summary = group_summary_by_id.get(group_id)
        if summary is not None and isinstance(summary.get("normal"), list) and len(summary["normal"]) == 3:
            normals[f"cad:{group_id}"] = [float(component) for component in summary["normal"]]
        else:
            normals[f"cad:{group_id}"] = [0.0, 0.0, 0.0]
    zero = np.zeros((6, *shape), dtype=bool)
    for patch_id in requested_selectors:
        selector = f"cad:{patch_id}"
        links[selector] = groups.get(patch_id, zero)
        normals.setdefault(selector, [0.0, 0.0, 0.0])
    return links, normals


def _result(
    mask: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    warnings: Iterable[str] = (),
    *,
    solid_mask: np.ndarray | None = None,
    thermal_mask: np.ndarray | None = None,
    material_index: np.ndarray | None = None,
    boundary_links: Mapping[str, np.ndarray] | None = None,
    boundary_normals: Mapping[str, list[float]] | None = None,
    surface_groups: list[dict[str, Any]] | None = None,
    triangle_groups: list[str] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "fluid_mask": np.asarray(mask, dtype=bool),
        "origin": np.asarray(origin, dtype=np.float64).tolist(),
        "spacing": np.asarray(spacing, dtype=np.float64).tolist(),
        "shape": [int(value) for value in mask.shape],
        "warnings": list(warnings),
        "solid_mask": np.asarray(solid_mask if solid_mask is not None else np.zeros_like(mask, dtype=bool), dtype=bool),
        "thermal_mask": np.asarray(thermal_mask if thermal_mask is not None else mask, dtype=bool),
        "material_index": np.asarray(material_index if material_index is not None else np.full(mask.shape, -1, dtype=np.int32), dtype=np.int32),
        "boundary_links": {str(key): np.asarray(value, dtype=bool) for key, value in (boundary_links or {}).items()},
        "boundary_normals": {str(key): [float(component) for component in value] for key, value in (boundary_normals or {}).items()},
    }
    if vertices is not None and faces is not None:
        output["surface_vertices"] = np.asarray(vertices, dtype=np.float64)
        output["surface_faces"] = np.asarray(faces, dtype=np.int32)
        # The HTTP preview and older callers use these short names.  Both are
        # views of the same normalised surface, never a fabricated box mesh.
        output["vertices"] = output["surface_vertices"]
        output["faces"] = output["surface_faces"]
        output["surface_groups"] = list(surface_groups or [])
        output["triangle_groups"] = list(triangle_groups or [])
    return output


def estimate_mesh(project: Mapping[str, Any], assets_dir: str | Path) -> dict[str, Any]:
    """Return a mesh plan without allocating voxel or boundary-link arrays."""

    normalized = validate_project(project)
    plan, _surface = _plan_mesh(normalized, assets_dir)
    velocities = [np.asarray(boundary['flow']['velocity'], dtype=float)
                  for boundary in normalized['boundaries']
                  if boundary['flow']['type'] == 'velocity' and normalized['physics']['flow']]
    speeds = [float(np.linalg.norm(velocity)) for velocity in velocities]
    speed = max(speeds, default=0.0) if normalized['physics']['flow'] else 0.0
    max_dt = 0.30 * float(plan['spacing'][0]) / (math.sqrt(3.0) * speed) if speed > 0 else None
    # Match the solver's multidimensional upwind Courant number: sum of
    # absolute directional speeds, not the Euclidean speed used for Mach.
    cfl_rate = max((float(np.sum(np.abs(velocity) / np.asarray(plan['spacing'])))
                    for velocity in velocities), default=0.0)
    cfl_target = 0.10
    dt_cfl = cfl_target / cfl_rate if cfl_rate > 0 else None
    # This is an advection estimate from prescribed velocities, not a proof
    # of thermal/material stability or a bound on the evolving interior flow.
    recommended_dt = min(dt_cfl, 0.8 * max_dt) if dt_cfl is not None else None
    gravity = normalized['physics'].get('gravity', {})
    force_dt_max = force_dt_basis = None
    if normalized['physics']['flow'] and gravity.get('enabled'):
        # Buoyancy is restricted to <=10% density contrast by the solver.
        # Use this conservative envelope even before the hot region forms.
        fraction = 0.1 if gravity['mode'] == 'buoyancy' else 1.0
        acceleration = math.hypot(*gravity['vector']) * fraction
        if acceleration > 0:
            force_dt_max = math.sqrt(0.05 * float(plan['spacing'][0]) / acceleration)
            head = float(np.dot(np.abs(gravity['vector']), np.asarray(plan['shape']) * np.asarray(plan['spacing']))) * fraction
            if head > 0:
                force_dt_max = min(force_dt_max, float(plan['spacing'][0]) * math.sqrt(0.1 / (3.0 * head)))
            force_dt_basis = '10% buoyancy density contrast' if gravity['mode'] == 'buoyancy' else 'uniform gravity'
            recommended_dt = min(recommended_dt, 0.8 * force_dt_max) if recommended_dt is not None else 0.8 * force_dt_max
    return {
        "shape": list(plan["shape"]),
        "total_cells": int(plan["total_cells"]),
        "spacing": list(plan["spacing"]),
        "origin": list(plan["origin"]),
        "target_cells": plan.get("target_cells"),
        "warnings": list(plan.get("warnings", [])),
        "flow_stability": {
            "boundary_speed": speed,
            "boundary_mach": math.sqrt(3.0) * speed * normalized['study']['dt'] / float(plan['spacing'][0]),
            "mach_limit": 0.30,
            "dt_max_exclusive": max_dt,
            "force_dt_max": force_dt_max,
            "force_dt_basis": force_dt_basis,
            "boundary_cfl": cfl_rate * normalized['study']['dt'],
            "cfl_target": cfl_target,
            "dt_cfl_target": dt_cfl,
            "recommended_dt": recommended_dt,
            "recommended_cfl": cfl_rate * recommended_dt if recommended_dt is not None else None,
        },
    }


def build_mesh(project: Mapping[str, Any], assets_dir: str | Path) -> dict[str, Any]:
    """Build a bounded Cartesian mask and CHT/contact metadata.

    ``fluid_mask`` identifies cells available to the fluid solver.  Explicit
    solid cells are removed from that mask and reported in ``solid_mask``;
    ``material_index`` uses the order of ``project.materials`` and is ``-1``
    for inactive cells or solids without a CHT material.  All returned link
    arrays use ``(6, nx, ny, nz)`` with directions ordered x-, x+, y-, y+,
    z-, z+.
    """

    normalized = validate_project(project)
    geometry = normalized["geometry"]
    plan, cad_surface = _plan_mesh(normalized, assets_dir)
    cells_array = np.asarray(plan["shape"], dtype=np.int64)
    cells_tuple = tuple(int(value) for value in cells_array)
    kind = geometry["kind"]
    role = geometry["role"]
    material_ids = {str(material["id"]): index for index, material in enumerate(normalized["materials"])}
    fluid_material = int(material_ids[normalized["physics"]["material_id"]])

    def finish(
        fluid_mask: np.ndarray,
        solid_mask: np.ndarray,
        material_index: np.ndarray,
        origin: np.ndarray,
        spacing: np.ndarray,
        warnings: Iterable[str] = (),
        *,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        fluid_mask = np.asarray(fluid_mask, dtype=bool)
        solid_mask = np.asarray(solid_mask, dtype=bool)
        if np.any(fluid_mask & solid_mask):
            raise ValueError("fluid and solid voxel masks may not overlap")
        material_index = np.asarray(material_index, dtype=np.int32)
        if material_index.shape != fluid_mask.shape:
            raise ValueError("material_index must have the same shape as fluid_mask")
        # A solid with no assigned material is still a geometric solid, but it
        # has no conductive CHT state.  The solver receives the complete
        # thermal connectivity mask and can treat -1 as its adiabatic fallback.
        thermal_mask = fluid_mask | solid_mask
        links, normals = {}, {}
        if vertices is not None and faces is not None and metadata is not None:
            cad_links, cad_normals = _cad_boundary_links(
                normalized,
                fluid_mask,
                np.asarray(origin, dtype=float),
                np.asarray(spacing, dtype=float),
                vertices,
                faces,
                metadata,
            )
            links.update(cad_links)
            normals.update(cad_normals)
        else:
            links, normals = _box_boundary_links(fluid_mask)
        groups = metadata.get("surface_groups") if metadata is not None else None
        triangle_groups = metadata.get("triangle_groups") if metadata is not None else None
        return _result(
            fluid_mask,
            origin,
            spacing,
            vertices,
            faces,
            warnings,
            solid_mask=solid_mask,
            thermal_mask=thermal_mask,
            material_index=material_index,
            boundary_links=links,
            boundary_normals=normals,
            surface_groups=list(groups) if isinstance(groups, list) else None,
            triangle_groups=[str(value) for value in triangle_groups] if isinstance(triangle_groups, list) else None,
        )

    if kind == "box" and role == "fluid":
        size = np.asarray(geometry["size"], dtype=np.float64)
        origin = np.asarray(plan["origin"], dtype=np.float64)
        spacing = np.asarray(plan["spacing"], dtype=np.float64)
        fluid_mask = np.ones(cells_tuple, dtype=bool)
        solid_mask = np.zeros(cells_tuple, dtype=bool)
        material_index = np.full(cells_tuple, -1, dtype=np.int32)
        material_index[fluid_mask] = fluid_material
        _apply_solid_regions(
            fluid_mask,
            solid_mask,
            material_index,
            geometry,
            origin,
            spacing,
            material_ids,
            require_inside_fluid=True,
        )
        if not fluid_mask.any():
            raise ValueError("solid regions consume every computational cell")
        return finish(
            fluid_mask,
            solid_mask,
            material_index,
            origin,
            spacing,
            list(plan.get("warnings", [])),
        )

    if kind == "box" and role == "obstacle":
        obstacle_size = np.asarray(geometry["size"], dtype=np.float64)
        obstacle_origin = np.asarray(geometry.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        domain_size, domain_origin = _box_descriptor(geometry)
        domain_origin = np.asarray(plan["origin"], dtype=np.float64)
        spacing = np.asarray(plan["spacing"], dtype=np.float64)
        domain_max = domain_origin + domain_size
        obstacle_max = obstacle_origin + obstacle_size
        tolerance = max(float(np.max(spacing)) * 1.0e-7, 1.0e-15)
        if np.any(obstacle_origin < domain_origin - tolerance) or np.any(obstacle_max > domain_max + tolerance):
            raise ValueError("solid obstacle must lie inside the computational box")
        fluid_mask = np.ones(cells_tuple, dtype=bool)
        total = math.prod(cells_tuple)
        flat = fluid_mask.reshape(-1)
        for start in range(0, total, 65_536):
            points = _grid_points(domain_origin, spacing, cells_tuple, start, min(start + 65_536, total))
            inside = np.all((points >= obstacle_origin - tolerance) & (points <= obstacle_max + tolerance), axis=1)
            flat[start : start + len(points)] = ~inside
        if not fluid_mask.any():
            raise ValueError("solid obstacle consumes every computational cell")
        solid_mask = ~fluid_mask
        material_index = np.full(cells_tuple, -1, dtype=np.int32)
        material_index[fluid_mask] = fluid_material
        if geometry.get("solid_material_id") is not None:
            material_index[solid_mask] = int(material_ids[geometry["solid_material_id"]])
        _apply_solid_regions(
            fluid_mask,
            solid_mask,
            material_index,
            geometry,
            domain_origin,
            spacing,
            material_ids,
            require_inside_fluid=False,
        )
        return finish(
            fluid_mask,
            solid_mask,
            material_index,
            domain_origin,
            spacing,
            list(plan.get("warnings", [])),
        )

    assert cad_surface is not None
    vertices, faces, metadata = cad_surface
    bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.float64)
    if role == "fluid":
        origin = np.asarray(plan["origin"], dtype=np.float64)
        spacing = np.asarray(plan["spacing"], dtype=np.float64)
        actual_shape = tuple(int(value) for value in plan["shape"])
        grid_warnings = list(plan.get("warnings", []))
        fluid_mask = _classify_surface(origin, spacing, actual_shape, vertices, faces)
        if not fluid_mask.any():
            raise ValueError("CAD fluid volume contains no cell centres; refine mesh resolution")
        solid_mask = np.zeros(actual_shape, dtype=bool)
        material_index = np.full(actual_shape, -1, dtype=np.int32)
        material_index[fluid_mask] = fluid_material
        _apply_solid_regions(
            fluid_mask,
            solid_mask,
            material_index,
            geometry,
            origin,
            spacing,
            material_ids,
            require_inside_fluid=True,
        )
        if not fluid_mask.any():
            raise ValueError("solid regions consume every CAD fluid cell")
        return finish(
            fluid_mask,
            solid_mask,
            material_index,
            origin,
            spacing,
            grid_warnings,
            vertices=vertices,
            faces=faces,
            metadata=metadata,
        )

    # A CAD obstacle is classified inside a computational box; ``geometry.size``
    # is the UI shorthand for that box and is centred on the imported surface
    # centroid when no explicit origin is supplied.
    if any(key in geometry for key in ("computational_box", "domain", "domain_size", "box_size")):
        domain_size, domain_origin = _box_descriptor(geometry)
    else:
        if "size" not in geometry:
            raise ValueError("CAD obstacle requires geometry.size or geometry.computational_box")
        domain_size = np.asarray(geometry["size"], dtype=np.float64)
        if "origin" in geometry:
            domain_origin = np.asarray(geometry["origin"], dtype=np.float64)
        else:
            centroid = vertices.mean(axis=0)
            domain_origin = centroid - domain_size / 2.0
        if domain_size.shape != (3,) or not np.isfinite(domain_size).all() or np.any(domain_size <= 0.0):
            raise ValueError("CAD obstacle computational box size must be a positive finite three-vector")
    domain_origin = np.asarray(plan["origin"], dtype=np.float64)
    spacing = np.asarray(plan["spacing"], dtype=np.float64)
    domain_max = domain_origin + domain_size
    tolerance = max(float(np.max(spacing)) * 1.0e-7, 1.0e-15)
    if np.any(bounds[0] < domain_origin - tolerance) or np.any(bounds[1] > domain_max + tolerance):
        raise ValueError("solid CAD obstacle must lie inside the computational box")
    _require_closed_surface(vertices, faces)
    fluid_mask = np.ones(cells_tuple, dtype=bool)
    total = math.prod(cells_tuple)
    flat = fluid_mask.reshape(-1)
    for start in range(0, total, 65_536):
        points = _grid_points(domain_origin, spacing, cells_tuple, start, min(start + 65_536, total))
        flat[start : start + len(points)] = ~_contains(points, vertices, faces)
    if not fluid_mask.any():
        raise ValueError("solid CAD obstacle consumes every computational cell")
    solid_mask = ~fluid_mask
    material_index = np.full(cells_tuple, -1, dtype=np.int32)
    material_index[fluid_mask] = fluid_material
    if geometry.get("solid_material_id") is not None:
        material_index[solid_mask] = int(material_ids[geometry["solid_material_id"]])
    _apply_solid_regions(
        fluid_mask,
        solid_mask,
        material_index,
        geometry,
        domain_origin,
        spacing,
        material_ids,
        require_inside_fluid=False,
    )
    warnings = list(plan.get("warnings", []))
    return finish(
        fluid_mask,
        solid_mask,
        material_index,
        domain_origin,
        spacing,
        warnings,
        vertices=vertices,
        faces=faces,
        metadata=metadata,
    )


__all__ = [
    "MAX_FACES",
    "MAX_BOUNDARY_LINK_BYTES",
    "MAX_SOURCE_BYTES",
    "MAX_VERTICES",
    "build_mesh",
    "estimate_mesh",
    "import_cad",
    "load_asset_metadata",
]
