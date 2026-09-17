"""Pure, bounded Cartesian mesh planning helpers.

The target-cell mode is intentionally kept independent from CAD loading and
voxel allocation.  Box planning searches only integer cell counts whose
physical cell widths agree to the configured isotropic tolerance.  CAD
planning uses the imported bounds and permits the centred padding convention
used by the existing CAD voxeliser.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .limits import MAX_AXIS_CELLS, MAX_TOTAL_CELLS


ISOTROPIC_RTOL = 1.0e-9
ISOTROPIC_ATOL = 1.0e-15


def _limits(max_axis: int | None, max_total: int | None) -> tuple[int, int]:
    axis = MAX_AXIS_CELLS if max_axis is None else int(max_axis)
    total = MAX_TOTAL_CELLS if max_total is None else int(max_total)
    if axis < 1 or total < 1:
        raise ValueError("mesh limits must be positive")
    return axis, total


def _positive_vector(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive finite three-vector") from exc
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{label} must be a positive finite three-vector")
    return result


def _positive_target(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("mesh.target_cells must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("mesh.target_cells must be a positive integer")
    return result


def _cell_counts(value: Any, label: str = "mesh.cells") -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a three-component integer vector") from exc
    if result.shape != (3,) or result.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a three-component integer vector")
    result = result.astype(np.int64, copy=False)
    if np.any(result <= 0):
        raise ValueError(f"{label} must contain positive integers")
    return result


def _isotropic(spacing: np.ndarray) -> bool:
    return bool(np.allclose(spacing, spacing[0], rtol=ISOTROPIC_RTOL, atol=ISOTROPIC_ATOL))


def _plan_result(
    shape: np.ndarray,
    spacing: np.ndarray,
    origin: Any = (0.0, 0.0, 0.0),
    *,
    target_cells: int | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    shape = np.asarray(shape, dtype=np.int64)
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("mesh shape must be a positive three-vector")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("mesh spacing must be a positive finite three-vector")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("mesh origin must be a finite three-vector")
    return {
        "shape": [int(value) for value in shape],
        "total_cells": int(np.prod(shape, dtype=np.int64)),
        # Keep size/count division in the output for boxes.  This preserves
        # the requested physical dimensions exactly while still satisfying
        # the isotropic equality tolerance used by the lattice.
        "spacing": spacing.astype(float).tolist(),
        "origin": origin.astype(float).tolist(),
        "target_cells": target_cells,
        "warnings": list(warnings or []),
    }


def plan_box(
    size: Any,
    target_cells: Any | None = None,
    *,
    cells: Any | None = None,
    origin: Any = (0.0, 0.0, 0.0),
    max_axis: int | None = None,
    max_total: int | None = None,
) -> dict[str, Any]:
    """Plan a box mesh, preserving all three physical extents.

    With ``target_cells`` set, integer counts are enumerated by count along
    the longest physical axis.  The other two counts are rounded around their
    proportional values and retained only when all three widths agree within
    ``1e-9`` relative tolerance.  The admissible product nearest the target
    is selected.  A legacy ``cells`` request is returned unchanged after the
    same isotropic validation used by the old mesh path.
    """

    extent = _positive_vector(size, "geometry size")
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
        raise ValueError("mesh origin must be a finite three-vector")
    axis_limit, total_limit = _limits(max_axis, max_total)

    if target_cells is None:
        if cells is None:
            raise ValueError("mesh.cells or mesh.target_cells is required")
        shape = _cell_counts(cells)
        if np.any(shape > axis_limit) or int(np.prod(shape, dtype=np.int64)) > total_limit:
            raise ValueError("mesh cell count exceeds the bounded workbench limit")
        spacing = extent / shape.astype(np.float64)
        if not _isotropic(spacing):
            raise ValueError("geometry size / mesh.cells must produce isotropic Cartesian spacing")
        return _plan_result(shape, spacing, origin_array, target_cells=None)

    target = _positive_target(target_cells)
    if target > total_limit:
        raise ValueError(
            f"mesh.target_cells cannot exceed {total_limit} (XLB_MAX_TOTAL_CELLS)"
        )

    longest = int(np.argmax(extent))
    longest_extent = float(extent[longest])
    ratios = extent / longest_extent
    candidates: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}

    # At the configured maximum of 1024 this is a tiny bounded search.  The
    # floor/ceil neighbourhood is sufficient for a 1e-9 width tolerance; the
    # extra adjacent values make the search robust around exact half-way
    # floating-point representations without introducing a Cartesian brute
    # force over all three axes.
    for longest_count in range(1, axis_limit + 1):
        desired = ratios * float(longest_count)
        per_axis: list[list[int]] = []
        for axis in range(3):
            if axis == longest:
                per_axis.append([longest_count])
                continue
            center = float(desired[axis])
            lower = math.floor(center)
            upper = math.ceil(center)
            values = {lower - 1, lower, upper, upper + 1}
            values = {int(value) for value in values if 1 <= int(value) <= axis_limit}
            per_axis.append(sorted(values))
        for first in per_axis[0]:
            for second in per_axis[1]:
                for third in per_axis[2]:
                    shape = np.asarray([first, second, third], dtype=np.int64)
                    product = int(np.prod(shape, dtype=np.int64))
                    if product > total_limit:
                        continue
                    spacing = extent / shape.astype(np.float64)
                    if not _isotropic(spacing):
                        continue
                    key = tuple(int(value) for value in shape)
                    candidates[key] = (product, spacing)

    if not candidates:
        raise ValueError(
            "exact isotropic mesh is impossible within configured limits "
            f"(max axis {axis_limit}, max total {total_limit})"
        )

    selected_key = min(
        candidates,
        key=lambda key: (abs(candidates[key][0] - target), candidates[key][0], key),
    )
    selected_total, spacing = candidates[selected_key]
    warnings: list[str] = []
    if selected_total != target:
        warnings.append(
            f"target total cell count {target} adjusted to nearest admissible "
            f"isotropic grid {selected_total} ({list(selected_key)})"
        )
    return _plan_result(
        np.asarray(selected_key, dtype=np.int64),
        spacing,
        origin_array,
        target_cells=target,
        warnings=warnings,
    )


def _ceil_shape(extent: np.ndarray, spacing: float, axis_limit: int) -> np.ndarray:
    raw = np.ceil(extent / float(spacing) - 1.0e-12).astype(np.int64)
    return np.maximum(raw, 3)


def plan_cad(
    bounds: Any,
    target_cells: Any,
    *,
    max_axis: int | None = None,
    max_total: int | None = None,
) -> dict[str, Any]:
    """Plan a target-cell CAD fluid mesh using actual imported bounds.

    CAD bounds may require centred padding because each extent is rounded to
    an integer count at one common spacing.  The planner considers the target
    volume spacing and the nearby spacing transition points, choosing the
    admissible resulting product nearest the requested total.
    """

    raw_bounds = np.asarray(bounds, dtype=np.float64)
    if raw_bounds.shape != (2, 3) or not np.isfinite(raw_bounds).all():
        raise ValueError("CAD bounds must be a finite (2, 3) array")
    extent = raw_bounds[1] - raw_bounds[0]
    if np.any(extent <= 0.0):
        raise ValueError("CAD bounds do not define a positive finite extent")
    target = _positive_target(target_cells)
    axis_limit, total_limit = _limits(max_axis, max_total)
    if target > total_limit:
        raise ValueError(
            f"mesh.target_cells cannot exceed {total_limit} (XLB_MAX_TOTAL_CELLS)"
        )

    volume = float(np.prod(extent, dtype=np.float64))
    spacing_target = (volume / float(target)) ** (1.0 / 3.0)
    # Keep at least three samples in every direction, as required by the
    # existing CAD classification path.
    max_spacing = float(np.min(extent) / 3.0)
    spacing_target = min(spacing_target, max_spacing)
    if not math.isfinite(spacing_target) or spacing_target <= 0.0:
        raise ValueError("CAD bounds do not define a positive finite target spacing")

    spacing_candidates: set[float] = {float(spacing_target)}
    # Every ceil transition can change the admissible product.  Sampling all
    # transitions is bounded by 3 * max_axis and avoids any voxel allocation.
    for axis_extent in extent:
        for count in range(3, axis_limit + 1):
            spacing_candidates.add(float(axis_extent / count))
    # Include a small neighbourhood around the volume-derived spacing to make
    # the nearest product deterministic even when it lies on a floating edge.
    for factor in (1.0 - 1.0e-12, 1.0 + 1.0e-12):
        spacing_candidates.add(float(spacing_target * factor))

    candidates: dict[tuple[int, int, int], tuple[int, float]] = {}
    for spacing in spacing_candidates:
        if spacing <= 0.0 or not math.isfinite(spacing) or spacing > max_spacing * (1.0 + ISOTROPIC_RTOL):
            continue
        shape = _ceil_shape(extent, spacing, axis_limit)
        if np.any(shape > axis_limit):
            continue
        total = int(np.prod(shape, dtype=np.int64))
        if total > total_limit:
            continue
        key = tuple(int(value) for value in shape)
        # For the same rounded shape, use the spacing closest to the target
        # volume request.  The actual grid spacing is the common value and the
        # resulting exterior padding is centred around the imported bounds.
        old = candidates.get(key)
        if old is None or abs(math.log(spacing / spacing_target)) < abs(math.log(old[1] / spacing_target)):
            candidates[key] = (total, spacing)

    if not candidates:
        raise ValueError(
            "CAD target mesh cannot satisfy configured axis/total limits "
            f"(max axis {axis_limit}, max total {total_limit})"
        )
    selected_key = min(
        candidates,
        key=lambda key: (abs(candidates[key][0] - target), candidates[key][0], key),
    )
    selected_total, spacing_value = candidates[selected_key]
    shape = np.asarray(selected_key, dtype=np.int64)
    padding = shape.astype(np.float64) * spacing_value - extent
    origin = raw_bounds[0] - 0.5 * padding
    warnings: list[str] = []
    if selected_total != target:
        warnings.append(
            f"target total cell count {target} adjusted to nearest permitted CAD "
            f"grid {selected_total} ({list(selected_key)})"
        )
    if np.any(padding > max(spacing_value * 1.0e-8, 1.0e-15)):
        warnings.append("CAD mesh adds evenly centred exterior padding after isotropic rounding")
    return _plan_result(shape, np.full(3, spacing_value, dtype=np.float64), origin, target_cells=target, warnings=warnings)


__all__ = ["ISOTROPIC_ATOL", "ISOTROPIC_RTOL", "plan_box", "plan_cad"]
