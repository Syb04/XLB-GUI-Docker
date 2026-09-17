"""Verify CUDA transient snapshots against the NumPy reference solver.

This is a self-contained, deterministic check for the short LES + CHT case
used by the transient snapshot review.  It intentionally calls ``simulate``
directly inside the CUDA image, so no workbench HTTP server is stopped or
restarted and no existing ``data/runs`` entry is touched.

Typical Docker invocation from WSL::

    docker run --rm --gpus all \
      -v /mnt/h/20260902_XLB/xlb_workbench/workbench:/app/workbench:ro \
      -v /mnt/h/20260902_XLB/xlb_workbench/scripts/verify_transient_solver.py:/app/verify_transient_solver.py:ro \
      -v /mnt/h/20260902_XLB/xlb_workbench/development_archive/transient-gpu:/output \
      -v /mnt/h/20260902_XLB/xlb_workbench/docs:/evidence \
      --entrypoint python xlb-workbench:0.1.0-cuda /app/verify_transient_solver.py \
      --output-dir /output --evidence /evidence/transient-gpu-verification.json

Only ``--output-dir`` and ``--evidence`` are written by the script.  The
output directory is deliberately refused when it already contains files, so
rerunning the check cannot remove a prior evidence run by accident.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
from typing import Any, Iterator, Mapping

import numpy as np

from workbench.geometry import build_mesh
from workbench.schema import default_project, validate_project
from workbench.solver import simulate


FIELDS = ("velocity", "pressure", "temperature", "eddy_viscosity")
SNAPSHOT_STEPS = [0, 2, 4, 6, 7]
COMPARISON_TOLERANCES: dict[str, dict[str, float]] = {
    # These are the same tolerances used by scripts/verify_gpu.py, with the
    # all-field comparison applied at every transient frame.
    "temperature": {"rtol": 0.0, "atol": 1.0e-3},
    "velocity": {"rtol": 3.0e-3, "atol": 2.0e-5},
    "pressure": {"rtol": 3.0e-3, "atol": 5.0e-2},
    "eddy_viscosity": {"rtol": 3.0e-3, "atol": 1.0e-7},
}


def make_project(*, device: str, flow: bool = True, thermal: bool = True) -> dict[str, Any]:
    """Return the small LES + CHT project used by every run in this check."""

    project = default_project()
    project["name"] = "Transient LES + CHT snapshot verification"
    project["geometry"].update(
        {
            "size": [0.04, 0.02, 0.02],
            "role": "fluid",
            "origin": [0.0, 0.0, 0.0],
            "solids": [
                {
                    "id": "aluminum-insert",
                    "name": "Internal aluminum CHT insert",
                    "origin": [0.012, 0.005, 0.005],
                    "size": [0.01, 0.01, 0.01],
                    "material_id": "aluminum",
                }
            ],
        }
    )
    project["materials"].append(
        {
            "id": "aluminum",
            "name": "Aluminum",
            "density": {"kind": "constant", "value": 2700.0},
            "viscosity": {"kind": "constant", "value": 0.001},
            "heat_capacity": {"kind": "constant", "value": 900.0},
            "conductivity": {"kind": "constant", "value": 16.0},
        }
    )
    project["physics"].update(
        {
            "flow": flow,
            "thermal": thermal,
            "material_id": "water",
            "initial_temperature": 300.0,
            "turbulence": {
                "model": "smagorinsky",
                "smagorinsky_constant": 0.17,
                "turbulent_prandtl": 0.9,
            },
        }
    )
    if flow and thermal:
        project["boundaries"] = [
            {
                "id": "inlet",
                "name": "Inlet",
                "face": "xmin",
                "flow": {"type": "velocity", "velocity": [0.01, 0.0, 0.0]},
                "thermal": {"type": "temperature", "value": 305.0},
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
                "thermal": {"type": "temperature", "value": 330.0},
            },
        ]
    elif not flow and thermal:
        project["boundaries"] = [
            {
                "id": "hot-wall",
                "name": "Hot wall",
                "face": "xmin",
                "flow": {"type": "wall"},
                "thermal": {"type": "temperature", "value": 310.0},
            },
            {
                "id": "cold-wall",
                "name": "Cold wall",
                "face": "xmax",
                "flow": {"type": "wall"},
                "thermal": {"type": "temperature", "value": 290.0},
            },
        ]
    project["mesh"] = {"cells": [8, 4, 4]}
    project["study"].update(
        {"steps": 7, "output_interval": 1, "snapshot_interval": 2, "dt": 0.001, "device": device}
    )
    return validate_project(project)


def make_mesh(project: Mapping[str, Any], assets_dir: Path) -> dict[str, Any]:
    """Build the canonical box mesh, including CHT material/contact masks."""

    mesh = build_mesh(project, assets_dir)
    if not np.asarray(mesh["solid_mask"], dtype=bool).any():
        raise AssertionError("the verification mesh must contain solid CHT cells")
    if not np.asarray(mesh["thermal_mask"], dtype=bool).all():
        raise AssertionError("all box cells must be in the thermal CHT mask")
    return mesh


@contextmanager
def backend_environment(value: str) -> Iterator[None]:
    """Set the solver backend selector for one in-process run."""

    previous = os.environ.get("XLB_COMPUTE_BACKEND")
    os.environ["XLB_COMPUTE_BACKEND"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("XLB_COMPUTE_BACKEND", None)
        else:
            os.environ["XLB_COMPUTE_BACKEND"] = previous


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar values used in diagnostics to JSON values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_frames(run_dir: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    manifest = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))
    entries = list(manifest.get("frames", []))
    frame_arrays: dict[int, dict[str, np.ndarray]] = {}
    for entry in entries:
        step = int(entry["step"])
        with np.load(run_dir / "frames" / entry["file"], allow_pickle=False) as data:
            frame_arrays[step] = {field: np.asarray(data[field]) for field in data.files}
    return entries, frame_arrays


def _field_stats(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for field in FIELDS:
        array = np.asarray(arrays[field])
        stats[field] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "max_abs": float(np.max(np.abs(array))),
        }
    return stats


def _verify_frames(run_dir: Path, expected_steps: list[int]) -> dict[str, Any]:
    entries, frames = _read_frames(run_dir)
    actual_steps = [int(entry["step"]) for entry in entries]
    if actual_steps != expected_steps:
        raise AssertionError(f"{run_dir}: expected frame steps {expected_steps}, got {actual_steps}")
    if set(frames) != set(expected_steps):
        raise AssertionError(f"{run_dir}: frame files do not match manifest")

    frame_reports: list[dict[str, Any]] = []
    for entry in entries:
        step = int(entry["step"])
        arrays = frames[step]
        if tuple(sorted(arrays)) != tuple(sorted(FIELDS)):
            raise AssertionError(f"{run_dir}: step {step} fields are {sorted(arrays)}")
        if not all(np.isfinite(np.asarray(arrays[field])).all() for field in FIELDS):
            raise AssertionError(f"{run_dir}: step {step} contains a non-finite field")
        frame_reports.append(
            {
                "step": step,
                "time": float(entry["time"]),
                "file": str(entry["file"]),
                "finite": True,
                "stats": _field_stats(arrays),
            }
        )

    changes: list[dict[str, Any]] = []
    for previous, current in zip(expected_steps, expected_steps[1:]):
        field_deltas = {
            field: float(np.max(np.abs(frames[current][field] - frames[previous][field])))
            for field in FIELDS
        }
        overall = max(field_deltas.values())
        if not math_is_positive(overall):
            raise AssertionError(f"{run_dir}: states at {previous} and {current} did not change")
        changes.append({"from_step": previous, "to_step": current, "max_abs_delta": field_deltas, "overall_max_abs_delta": overall})

    with np.load(run_dir / "fields.npz", allow_pickle=False) as final_fields:
        final_arrays = {field: np.asarray(final_fields[field]) for field in FIELDS}
    final_step = expected_steps[-1]
    final_match = {
        field: bool(np.array_equal(frames[final_step][field], final_arrays[field]))
        for field in FIELDS
    }
    if not all(final_match.values()):
        raise AssertionError(f"{run_dir}: final fields do not exactly match the last snapshot: {final_match}")

    return {
        "directory": str(run_dir),
        "frame_steps": actual_steps,
        "frame_count": len(entries),
        "frames": frame_reports,
        "state_changes": changes,
        "final_field_exact_match": final_match,
    }


def math_is_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def _run(
    *,
    name: str,
    project: Mapping[str, Any],
    mesh: Mapping[str, Any],
    root: Path,
    backend_selector: str,
    stop_step: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = root / name
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing verification directory: {run_dir}")
    run_dir.mkdir(parents=True)
    progress: list[dict[str, Any]] = []

    def stop() -> bool:
        return stop_step is not None and bool(progress) and int(progress[-1]["step"]) >= stop_step

    with backend_environment(backend_selector):
        result = simulate(project, mesh, run_dir, on_progress=progress.append, should_stop=stop if stop_step is not None else None)
    return result, {"progress": progress, "run_dir": str(run_dir)}


def _compare_frame_sets(
    cpu_dir: Path,
    cuda_dir: Path,
    steps: list[int],
) -> dict[str, dict[str, float]]:
    _, cpu_frames = _read_frames(cpu_dir)
    _, cuda_frames = _read_frames(cuda_dir)
    comparison: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        max_abs = 0.0
        max_rel = 0.0
        for step in steps:
            cpu = np.asarray(cpu_frames[step][field])
            cuda = np.asarray(cuda_frames[step][field])
            if cpu.shape != cuda.shape:
                raise AssertionError(f"{field} step {step}: CPU/CUDA shape mismatch")
            tolerance = COMPARISON_TOLERANCES[field]
            np.testing.assert_allclose(
                cuda,
                cpu,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=f"CPU/reference vs CUDA mismatch for {field} at step {step}",
            )
            difference = np.abs(cuda - cpu)
            max_abs = max(max_abs, float(np.max(difference)))
            denominator = np.maximum(np.abs(cpu), tolerance["atol"])
            max_rel = max(max_rel, float(np.max(difference / denominator)))
        comparison[field] = {
            "max_absolute_difference": max_abs,
            "max_relative_difference_using_atol_floor": max_rel,
            **tolerance,
        }
    return comparison


def _mesh_report(mesh: Mapping[str, Any]) -> dict[str, Any]:
    fluid = np.asarray(mesh["fluid_mask"], dtype=bool)
    solid = np.asarray(mesh["solid_mask"], dtype=bool)
    thermal = np.asarray(mesh["thermal_mask"], dtype=bool)
    return {
        "shape": list(fluid.shape),
        "total_cells": int(fluid.size),
        "fluid_cells": int(np.count_nonzero(fluid)),
        "solid_cells": int(np.count_nonzero(solid)),
        "thermal_cells": int(np.count_nonzero(thermal)),
        "origin": np.asarray(mesh["origin"], dtype=float).tolist(),
        "spacing": np.asarray(mesh["spacing"], dtype=float).tolist(),
        "material_indices": sorted(int(value) for value in np.unique(np.asarray(mesh["material_index"]))),
        "boundary_link_counts": {
            key: int(np.count_nonzero(np.asarray(value, dtype=bool)))
            for key, value in (mesh.get("boundary_links") or {}).items()
        },
    }


def _assert_diagnostics(result: Mapping[str, Any], *, expect_transfers: int, thermal_only: bool = False) -> None:
    if result.get("status") != "completed":
        raise AssertionError(f"run did not complete: {result}")
    diagnostics = result["diagnostics"]
    if diagnostics.get("compute_backend") != "device":
        raise AssertionError(f"CUDA run did not use the device backend: {diagnostics}")
    if "cuda" not in str(diagnostics.get("device_actual", "")).lower():
        raise AssertionError(f"CUDA run selected an unexpected device: {diagnostics}")
    if diagnostics.get("full_field_transfers") != expect_transfers:
        raise AssertionError(
            f"expected {expect_transfers} CUDA full-field transfers, got {diagnostics.get('full_field_transfers')}"
        )
    if diagnostics.get("thermal_backend") != "JAX finite-volume on device":
        raise AssertionError(f"unexpected thermal backend: {diagnostics.get('thermal_backend')}")
    if diagnostics.get("turbulence_model") != "smagorinsky":
        raise AssertionError(f"unexpected turbulence model: {diagnostics.get('turbulence_model')}")
    if thermal_only and diagnostics.get("flow_cells", 0) <= 0:
        raise AssertionError("thermal-only run unexpectedly has no fluid cells")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="development_archive/transient-gpu")
    parser.add_argument("--evidence", default="docs/transient-gpu-verification.json")
    parser.add_argument("--image", default="xlb-workbench:0.1.0-cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    evidence_path = Path(args.evidence).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty verification directory {output_dir}; choose a new --output-dir"
        )

    # Build once so CPU/reference and CUDA receive identical masks and links.
    base_project = make_project(device="cuda:0")
    mesh = make_mesh(base_project, output_dir)
    mesh_report = _mesh_report(mesh)

    cpu_project = copy.deepcopy(base_project)
    cpu_project["study"]["device"] = "cpu"
    cuda_project = copy.deepcopy(base_project)
    cuda_project["study"]["device"] = "cuda:0"

    cpu_result, cpu_meta = _run(
        name="cpu_reference_les_cht",
        project=cpu_project,
        mesh=mesh,
        root=output_dir,
        backend_selector="reference",
    )
    cuda_result, cuda_meta = _run(
        name="cuda_les_cht_snapshots",
        project=cuda_project,
        mesh=mesh,
        root=output_dir,
        backend_selector="auto",
    )
    _assert_diagnostics(cuda_result, expect_transfers=5)
    cpu_frames = _verify_frames(Path(cpu_meta["run_dir"]), SNAPSHOT_STEPS)
    cuda_frames = _verify_frames(Path(cuda_meta["run_dir"]), SNAPSHOT_STEPS)
    frame_comparison = _compare_frame_sets(
        Path(cpu_meta["run_dir"]), Path(cuda_meta["run_dir"]), SNAPSHOT_STEPS
    )

    # Optional edge cases are kept in their own directories.  They use the
    # same CUDA project and make the requested stop/snapshot behavior visible
    # without changing the main 0,2,4,6,7 assertion.
    stop0_result, stop0_meta = _run(
        name="cuda_stop_before_first_step",
        project=copy.deepcopy(cuda_project),
        mesh=mesh,
        root=output_dir,
        backend_selector="auto",
        stop_step=0,
    )
    stop0_report = _verify_frames(Path(stop0_meta["run_dir"]), [0])
    if stop0_result.get("status") != "stopped" or stop0_result.get("steps") != 0:
        raise AssertionError(f"stop-before-first-step run has unexpected result: {stop0_result}")

    stop3_result, stop3_meta = _run(
        name="cuda_stop_at_step_3",
        project=copy.deepcopy(cuda_project),
        mesh=mesh,
        root=output_dir,
        backend_selector="auto",
        stop_step=3,
    )
    stop3_report = _verify_frames(Path(stop3_meta["run_dir"]), [0, 2, 3])
    if stop3_result.get("status") != "stopped" or stop3_result.get("steps") != 3:
        raise AssertionError(f"stop-at-step-3 run has unexpected result: {stop3_result}")

    thermal_project = make_project(device="cuda:0", flow=False, thermal=True)
    thermal_result, thermal_meta = _run(
        name="cuda_thermal_only",
        project=thermal_project,
        mesh=mesh,
        root=output_dir,
        backend_selector="auto",
    )
    _assert_diagnostics(thermal_result, expect_transfers=5, thermal_only=True)
    thermal_report = _verify_frames(Path(thermal_meta["run_dir"]), SNAPSHOT_STEPS)
    _, thermal_frames = _read_frames(Path(thermal_meta["run_dir"]))
    # ``array_equal(array, 0.0)`` does not broadcast a scalar, so use an
    # elementwise exact comparison for this physical invariant.
    thermal_zero_velocity = {
        str(step): bool(np.all(thermal_frames[step]["velocity"] == 0.0))
        for step in SNAPSHOT_STEPS
    }
    if not all(thermal_zero_velocity.values()):
        raise AssertionError(f"thermal-only CUDA velocity is not exactly zero: {thermal_zero_velocity}")
    with np.load(Path(thermal_meta["run_dir"]) / "fields.npz", allow_pickle=False) as fields:
        thermal_zero_velocity["final_fields"] = bool(np.all(fields["velocity"] == 0.0))
    if not thermal_zero_velocity["final_fields"]:
        raise AssertionError("thermal-only final velocity is not exactly zero")

    try:
        import jax

        jax_runtime = {
            "version": str(jax.__version__),
            "devices": [str(device) for device in jax.devices()],
            "gpu_devices": [str(device) for device in jax.devices("gpu")],
        }
    except Exception as exc:  # pragma: no cover - only reached on a broken image
        jax_runtime = {"error": f"{type(exc).__name__}: {exc}"}

    evidence = {
        "purpose": "Real CUDA transient LES + CHT snapshot verification against the CPU reference path",
        "image": args.image,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax_runtime,
            "workbench_source": "/app/workbench mounted read-only",
            "server": "direct simulate call; port 8766 untouched",
        },
        "case": {
            "study": base_project["study"],
            "snapshot_steps": SNAPSHOT_STEPS,
            "project": base_project,
            "mesh": mesh_report,
        },
        "cuda_main": {
            "result": _json_value(cuda_result),
            "progress": _json_value(cuda_meta["progress"]),
            "verification": cuda_frames,
        },
        "cpu_reference": {
            "result": _json_value(cpu_result),
            "progress": _json_value(cpu_meta["progress"]),
            "verification": cpu_frames,
        },
        "cpu_cuda_frame_comparison": frame_comparison,
        "optional_checks": {
            "stop_before_first_step": {
                "result": _json_value(stop0_result),
                "progress": _json_value(stop0_meta["progress"]),
                "verification": stop0_report,
            },
            "stop_at_step_3": {
                "result": _json_value(stop3_result),
                "progress": _json_value(stop3_meta["progress"]),
                "verification": stop3_report,
            },
            "thermal_only_zero_velocity": {
                "result": _json_value(thermal_result),
                "progress": _json_value(thermal_meta["progress"]),
                "verification": thermal_report,
                "zero_velocity": thermal_zero_velocity,
            },
        },
        "reproduction": {
            "command_shape": "wsl.exe -d Ubuntu-24.04 -u root --cd /mnt/h/20260902_XLB/xlb_workbench -- docker run --rm --gpus all ...",
            "mounts": [
                "workbench:/app/workbench:ro",
                "development_archive/transient-gpu:/output",
                "docs:/evidence",
            ],
            "script": "scripts/verify_transient_solver.py",
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "evidence": str(evidence_path),
        "output_dir": str(output_dir),
        "cuda_status": cuda_result["status"],
        "cuda_device": cuda_result["diagnostics"].get("device_actual"),
        "frame_steps": SNAPSHOT_STEPS,
        "full_field_transfers": cuda_result["diagnostics"].get("full_field_transfers"),
        "cpu_cuda_comparison": frame_comparison,
        "thermal_only_zero_velocity": thermal_zero_velocity,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
