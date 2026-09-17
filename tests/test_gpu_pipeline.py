"""Integration checks for the JAX-resident solver path.

The tests force the device pipeline on ``cpu`` so they run in the pinned XLB
environment even when a CUDA wheel is unavailable.  A CUDA run uses the same
code path automatically when ``study.device`` is ``cuda:0``.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workbench.solver import _compute_backend


_HAS_XLB = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("xlb") is not None
if _HAS_XLB:
    from workbench.solver import simulate
    try:
        # ``unittest discover`` under the source tree can import tests as a
        # namespace package, while the Docker image discovers this directory
        # as top-level modules.  Support both layouts.
        from tests.test_solver import _mesh, _project
    except ModuleNotFoundError:  # pragma: no cover - exercised by Docker discovery
        from test_solver import _mesh, _project


class ComputeBackendSelectionTests(unittest.TestCase):
    def test_backend_selection_is_explicit_and_cuda_auto_is_device(self) -> None:
        self.assertEqual(_compute_backend("cpu"), "reference")
        self.assertEqual(_compute_backend("cuda:0"), "device")
        old = os.environ.get("XLB_COMPUTE_BACKEND")
        try:
            os.environ["XLB_COMPUTE_BACKEND"] = "device"
            self.assertEqual(_compute_backend("cpu"), "device")
            self.assertEqual(_compute_backend("cuda:0-ram"), "ram")
            os.environ["XLB_COMPUTE_BACKEND"] = "invalid"
            with self.assertRaises(ValueError):
                _compute_backend("cpu")
        finally:
            if old is None:
                os.environ.pop("XLB_COMPUTE_BACKEND", None)
            else:
                os.environ["XLB_COMPUTE_BACKEND"] = old


@unittest.skipUnless(_HAS_XLB, "the pinned WSL XLB/JAX environment is required")
class DevicePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_backend = os.environ.get("XLB_COMPUTE_BACKEND")
        os.environ["XLB_COMPUTE_BACKEND"] = "device"

    def tearDown(self) -> None:
        if self._old_backend is None:
            os.environ.pop("XLB_COMPUTE_BACKEND", None)
        else:
            os.environ["XLB_COMPUTE_BACKEND"] = self._old_backend

    def _run(self, project: dict | None = None, mesh: dict | None = None) -> tuple[dict, Path]:
        project = _project(steps=4) if project is None else project
        mesh = _mesh() if mesh is None else mesh
        directory = Path(tempfile.mkdtemp(prefix="xlb-device-pipeline-"))
        return simulate(project, mesh, directory), directory

    def test_forced_cpu_device_pipeline_uses_device_factories_and_one_final_download(self) -> None:
        result, directory = self._run()
        diagnostics = result["diagnostics"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(diagnostics["compute_backend"], "device")
        self.assertIn("JAX", diagnostics["thermal_backend"])
        self.assertIn("JAX", diagnostics["property_backend"])
        self.assertEqual(diagnostics["full_field_transfers"], 1)
        self.assertGreaterEqual(diagnostics["device_scalar_transfers"], 1)
        self.assertGreaterEqual(diagnostics["timings"]["first_step_s"], 0.0)
        self.assertGreaterEqual(diagnostics["timings"]["warm_compute_s"], 0.0)
        self.assertGreaterEqual(diagnostics["timings"]["save_s"], 0.0)
        with np.load(directory / "fields.npz") as fields:
            self.assertTrue(np.isfinite(fields["velocity"]).all())
            self.assertTrue(np.isfinite(fields["temperature"]).all())
            self.assertGreater(float(np.max(np.abs(fields["velocity"]))), 0.0)

    def test_thermal_only_device_path_keeps_flow_zero(self) -> None:
        project = _project(flow=False, thermal=True, steps=3)
        project["boundaries"] = [
            {"id": "hot", "face": "xmin", "flow": {"type": "wall"}, "thermal": {"type": "temperature", "value": 310.0}},
            {"id": "cold", "face": "xmax", "flow": {"type": "wall"}, "thermal": {"type": "temperature", "value": 290.0}},
        ]
        result, directory = self._run(project)
        self.assertEqual(result["diagnostics"]["thermal_backend"], "JAX finite-volume on device")
        with np.load(directory / "fields.npz") as fields:
            np.testing.assert_array_equal(fields["velocity"], 0.0)
            self.assertGreater(float(fields["temperature"][0].mean()), 300.0)
            self.assertLess(float(fields["temperature"][-1].mean()), 300.0)

    def test_flow_only_device_path_skips_thermal_update(self) -> None:
        project = _project(flow=True, thermal=False, steps=3)
        initial = float(project["physics"]["initial_temperature"])
        result, directory = self._run(project)
        self.assertEqual(result["diagnostics"]["thermal_backend"], "disabled")
        with np.load(directory / "fields.npz") as fields:
            np.testing.assert_allclose(fields["temperature"], initial)

    def test_stop_before_first_step_and_between_outputs_write_final_state(self) -> None:
        project = _project(steps=5)
        # The callback is checked after the initial sample, before step 1.
        calls = {"count": 0}

        def stop_now() -> bool:
            calls["count"] += 1
            return True

        stopped_directory = Path(tempfile.mkdtemp(prefix="xlb-device-stop-"))
        stopped = simulate(project, _mesh(), stopped_directory, should_stop=stop_now)
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(stopped["steps"], 0)
        self.assertEqual(stopped["diagnostics"]["full_field_transfers"], 1)
        self.assertTrue((stopped_directory / "fields.npz").exists())

        # With output_interval=5, stopping after two callbacks lands between
        # samples and must still produce a scalar final history row.
        calls = {"count": 0}

        def stop_between() -> bool:
            calls["count"] += 1
            return calls["count"] >= 2

        between_directory = Path(tempfile.mkdtemp(prefix="xlb-device-stop-between-"))
        between = simulate(project, _mesh(), between_directory, should_stop=stop_between)
        self.assertEqual(between["status"], "stopped")
        self.assertEqual(between["steps"], 1)
        self.assertTrue((between_directory / "history.csv").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
