"""Restart archive and split-run checks for the workbench solver."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from workbench.restart import (
    load_restart,
    mesh_fingerprint,
    project_fingerprint,
    read_restart_metadata,
    write_restart,
)


_HAS_XLB = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("xlb") is not None
if _HAS_XLB:
    from workbench.solver import simulate


def _project(*, steps: int = 4, output_interval: int = 1) -> dict:
    return {
        "schema_version": 1,
        "name": "restart test",
        "geometry": {"kind": "box", "size": [0.02, 0.01, 0.01], "role": "fluid"},
        "materials": [
            {
                "id": "water",
                "name": "Water",
                "density": {"kind": "constant", "value": 1000.0},
                "viscosity": {"kind": "table", "points": [[280.0, 0.0008], [320.0, 0.0016]]},
                "heat_capacity": {"kind": "constant", "value": 4182.0},
                "conductivity": {"kind": "constant", "value": 0.6},
            }
        ],
        "physics": {
            "flow": True,
            "thermal": True,
            "material_id": "water",
            "initial_temperature": 300.0,
        },
        "boundaries": [
            {
                "id": "inlet",
                "name": "Inlet",
                "face": "xmin",
                "flow": {"type": "velocity", "velocity": [0.005, 0.0, 0.0]},
                "thermal": {"type": "temperature", "value": 305.0},
            },
            {
                "id": "outlet",
                "name": "Outlet",
                "face": "xmax",
                "flow": {"type": "pressure", "value": 0.0},
                "thermal": {"type": "adiabatic"},
            },
        ],
        "mesh": {"cells": [4, 2, 2]},
        "study": {"steps": steps, "output_interval": output_interval, "snapshot_interval": 2, "dt": 0.001, "device": "cpu"},
    }


def _mesh() -> dict:
    shape = (4, 2, 2)
    return {
        "fluid_mask": np.ones(shape, dtype=bool),
        "origin": np.zeros(3, dtype=np.float64),
        "spacing": np.full(3, 0.005, dtype=np.float64),
        "shape": list(shape),
    }


class RestartArchiveTests(unittest.TestCase):
    def test_memory_schedule_does_not_change_restart_physics(self):
        project = _project()
        original = project_fingerprint(project)
        project["study"].update(device="cuda:0-ram", gpu_batch_cells=65536)
        self.assertEqual(project_fingerprint(project), original)
        project["study"]["gpu_batch_cells"] = 1024
        self.assertEqual(project_fingerprint(project), original)

    def test_metadata_is_read_without_loading_fields(self) -> None:
        project = _project(steps=2)
        mesh = _mesh()
        with tempfile.TemporaryDirectory(prefix="xlb-restart-metadata-") as directory:
            path = Path(directory) / "restart.npz"
            write_restart(
                path,
                step=2,
                dt=0.001,
                f=np.zeros((27, 4, 2, 2), dtype=np.float32),
                temperature=np.full((4, 2, 2), 300.0, dtype=np.float64),
                project=project,
                mesh=mesh,
                status="completed",
            )
            metadata = read_restart_metadata(path)
            self.assertEqual({key: metadata[key] for key in ("version", "step", "dt", "shape")},
                             {"version": 1, "step": 2, "dt": 0.001, "shape": [4, 2, 2]})
            self.assertEqual(load_restart(path, project=project, mesh=mesh, dt=0.001, shape=(4, 2, 2), steps=4)["f"].dtype,
                             np.dtype(np.float32))

    def test_project_and_mesh_compatibility_rejects_changes(self) -> None:
        project = _project(steps=2)
        mesh = _mesh()
        with tempfile.TemporaryDirectory(prefix="xlb-restart-validation-") as directory:
            path = Path(directory) / "restart.npz"
            write_restart(
                path,
                step=2,
                dt=0.001,
                f=np.zeros((27, 4, 2, 2), dtype=np.float32),
                temperature=np.full((4, 2, 2), 300.0, dtype=np.float64),
                project=project,
                mesh=mesh,
                status="stopped",
            )
            scheduling_change = _project(steps=9, output_interval=3)
            scheduling_change["study"]["snapshot_interval"] = 0
            load_restart(path, project=scheduling_change, mesh=mesh, dt=0.001, shape=(4, 2, 2), steps=9)
            remapped_asset = _project(steps=9)
            remapped_asset["geometry"]["asset_id"] = "server-local-remapped-id"
            load_restart(path, project=remapped_asset, mesh=mesh, dt=0.001, shape=(4, 2, 2), steps=9)

            changed_project = _project(steps=9)
            changed_project["physics"]["initial_temperature"] = 301.0
            with self.assertRaisesRegex(ValueError, "project"):
                load_restart(path, project=changed_project, mesh=mesh, dt=0.001, shape=(4, 2, 2), steps=9)

            changed_mesh = _mesh()
            changed_mesh["fluid_mask"][0, 0, 0] = False
            with self.assertRaisesRegex(ValueError, "mesh"):
                load_restart(path, project=project, mesh=changed_mesh, dt=0.001, shape=(4, 2, 2), steps=9)

            mesh_with_links = _mesh()
            mesh_with_links["boundary_links"] = {"cad": np.zeros((6, 4, 2, 2), dtype=bool)}
            link_path = Path(directory) / "links.npz"
            write_restart(
                link_path,
                step=2,
                dt=0.001,
                f=np.zeros((27, 4, 2, 2), dtype=np.float32),
                temperature=np.full((4, 2, 2), 300.0, dtype=np.float64),
                project=project,
                mesh=mesh_with_links,
                status="stopped",
            )
            changed_links = _mesh()
            changed_links["boundary_links"] = {"cad": np.zeros((6, 4, 2, 2), dtype=bool)}
            changed_links["boundary_links"]["cad"][0, 0, 0, 0] = True
            with self.assertRaisesRegex(ValueError, "mesh"):
                load_restart(link_path, project=project, mesh=changed_links, dt=0.001, shape=(4, 2, 2), steps=9)

            with self.assertRaisesRegex(ValueError, "dt"):
                load_restart(path, project=project, mesh=mesh, dt=0.002, shape=(4, 2, 2), steps=9)

            bad_dtype = Path(directory) / "bad-dtype.npz"
            with np.load(path, allow_pickle=False) as source:
                np.savez_compressed(
                    bad_dtype,
                    f=source["f"].astype(np.float64),
                    temperature=source["temperature"],
                    metadata=source["metadata"],
                )
            with self.assertRaisesRegex(ValueError, "dtype"):
                load_restart(bad_dtype, project=project, mesh=mesh, dt=0.001, shape=(4, 2, 2), steps=9)

            corrupt = Path(directory) / "corrupt.npz"
            corrupt.write_bytes(b"not a restart archive")
            with self.assertRaises(ValueError):
                read_restart_metadata(corrupt)


@unittest.skipUnless(_HAS_XLB, "the pinned XLB/JAX environment is required")
class SolverRestartTests(unittest.TestCase):
    def _run_pair(self, *, backend: str | None = None, flow: bool = True) -> tuple[Path, Path, Path]:
        old_backend = os.environ.get("XLB_COMPUTE_BACKEND")
        if backend is None:
            os.environ.pop("XLB_COMPUTE_BACKEND", None)
        else:
            os.environ["XLB_COMPUTE_BACKEND"] = backend
        try:
            root = Path(tempfile.mkdtemp(prefix="xlb-restart-split-"))
            self.addCleanup(shutil.rmtree, root, ignore_errors=True)
            continuous = root / "continuous"
            first = root / "first"
            resumed = root / "resumed"
            mesh = _mesh()
            first_project = _project(steps=2)
            first_project["physics"]["flow"] = flow
            continuous_project = _project(steps=4)
            continuous_project["physics"]["flow"] = flow
            simulate(continuous_project, mesh, continuous)
            simulate(first_project, mesh, first)
            resumed_project = _project(steps=4)
            resumed_project["physics"]["flow"] = flow
            simulate(resumed_project, mesh, resumed, restart_from=first / "restart.npz")
            return continuous, first, resumed
        finally:
            if old_backend is None:
                os.environ.pop("XLB_COMPUTE_BACKEND", None)
            else:
                os.environ["XLB_COMPUTE_BACKEND"] = old_backend

    def _assert_pair(self, continuous: Path, resumed: Path) -> None:
        with np.load(continuous / "fields.npz") as expected, np.load(resumed / "fields.npz") as actual:
            for name in ("velocity", "pressure", "temperature", "eddy_viscosity"):
                np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)
        with np.load(continuous / "restart.npz") as expected, np.load(resumed / "restart.npz") as actual:
            np.testing.assert_array_equal(expected["f"], actual["f"])
            np.testing.assert_array_equal(expected["temperature"], actual["temperature"])

    def test_reference_continuous_equals_two_part_run(self) -> None:
        continuous, _, resumed = self._run_pair()
        self._assert_pair(continuous, resumed)

    def test_reference_thermal_only_continuous_equals_two_part_run(self) -> None:
        continuous, _, resumed = self._run_pair(flow=False)
        self._assert_pair(continuous, resumed)

    def test_device_cpu_continuous_equals_two_part_run(self) -> None:
        continuous, _, resumed = self._run_pair(backend="device")
        self._assert_pair(continuous, resumed)

    def test_device_cpu_thermal_only_continuous_equals_two_part_run(self) -> None:
        continuous, _, resumed = self._run_pair(backend="device", flow=False)
        self._assert_pair(continuous, resumed)

    def test_stopped_checkpoint_resumes_with_absolute_steps(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="xlb-restart-stop-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        progress: list[dict] = []
        stop = lambda: bool(progress and progress[-1]["step"] >= 2)
        first = simulate(_project(steps=4), _mesh(), root / "first", on_progress=progress.append, should_stop=stop)
        self.assertEqual(first["status"], "stopped")
        self.assertEqual(first["steps"], 2)
        resumed_progress: list[dict] = []
        resumed = simulate(_project(steps=4), _mesh(), root / "resumed", on_progress=resumed_progress.append,
                           restart_from=root / "first" / "restart.npz")
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual([row["step"] for row in resumed_progress], [2, 3, 4])
        self.assertEqual(resumed_progress[0]["progress"], 0.0)
        self.assertEqual(resumed_progress[-1]["progress"], 1.0)
