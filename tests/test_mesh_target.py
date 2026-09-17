import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - optional CAD dependency
    trimesh = None

from workbench.geometry import build_mesh, estimate_mesh, import_cad
from workbench.mesh_planning import plan_box, plan_cad
from workbench.schema import default_project, validate_project


class TargetMeshTests(unittest.TestCase):
    def test_one_meter_inlet_reports_time_step_limit_before_solver_start(self):
        from workbench.solver import simulate
        project = default_project()
        project['geometry']['size'][0] = 1.0
        project['mesh'] = {'target_cells': 20000}
        project['boundaries'][0]['flow']['velocity'] = [1.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as temporary:
            estimate = estimate_mesh(project, temporary)
            self.assertEqual(estimate['shape'], [350, 7, 7])
            stability = estimate['flow_stability']
            self.assertAlmostEqual(stability['boundary_mach'], 6.062177826)
            self.assertAlmostEqual(stability['dt_max_exclusive'], 0.0004948716593)
            self.assertAlmostEqual(stability['boundary_cfl'], 3.5)
            self.assertAlmostEqual(stability['recommended_dt'], 0.000285714285714)
            self.assertAlmostEqual(stability['recommended_cfl'], 0.1)
            mesh = build_mesh(project, temporary)
            with self.assertRaisesRegex(ValueError, r'reduce study.dt below 0.000494871659 s'):
                simulate(project, mesh, Path(temporary) / 'run')
            project['study']['dt'] = 0.0001
            self.assertLess(estimate_mesh(project, temporary)['flow_stability']['boundary_mach'], 0.30)

    def test_cfl_recommendation_sums_directions_and_handles_zero_velocity(self):
        project = default_project()
        for velocity in ([1.0, 0.0, 0.0], [1.0, -1.0, 1.0]):
            project['boundaries'][0]['flow']['velocity'] = velocity
            info = estimate_mesh(project, tempfile.gettempdir())['flow_stability']
            rate = sum(abs(component) for component in velocity) / 0.002
            self.assertAlmostEqual(info['boundary_cfl'], rate * project['study']['dt'])
            self.assertAlmostEqual(info['recommended_dt'], 0.1 / rate)
            self.assertLess(info['recommended_dt'], info['dt_max_exclusive'])
        project['boundaries'][0]['flow']['velocity'] = [0, 0, 0]
        info = estimate_mesh(project, tempfile.gettempdir())['flow_stability']
        self.assertEqual(info['boundary_cfl'], 0.0)
        self.assertIsNone(info['recommended_dt'])
        self.assertIsNone(info['recommended_cfl'])
        project['boundaries'][0]['flow']['velocity'] = [1, 0, 0]
        project['physics']['flow'] = False
        self.assertIsNone(estimate_mesh(project, tempfile.gettempdir())['flow_stability']['recommended_dt'])

    def test_cfl_recommendation_uses_each_boundary_vector(self):
        project = default_project()
        project['boundaries'][0]['flow']['velocity'] = [1, 0, 0]
        project['boundaries'][1]['flow'] = {'type': 'velocity', 'velocity': [0.6, 0.6, 0]}
        info = estimate_mesh(project, tempfile.gettempdir())['flow_stability']
        # The second boundary controls directional CFL; the first controls Mach.
        self.assertAlmostEqual(info['boundary_cfl'], 6.0)
        self.assertAlmostEqual(info['boundary_speed'], 1.0)
        self.assertAlmostEqual(info['recommended_dt'], 0.1 * 0.002 / 1.2)

    def test_default_box_target_three_million_and_exact_extents(self):
        project = default_project()
        project["mesh"] = {"target_cells": 3_000_000}
        normalized = validate_project(project)
        self.assertEqual(normalized["mesh"]["cells"], [300, 100, 100])
        with tempfile.TemporaryDirectory() as temporary:
            estimate = estimate_mesh(project, temporary)
            mesh = build_mesh(project, temporary)
        self.assertEqual(estimate["shape"], [300, 100, 100])
        self.assertEqual(estimate["total_cells"], 3_000_000)
        self.assertEqual(mesh["shape"], estimate["shape"])
        np.testing.assert_allclose(mesh["spacing"], estimate["spacing"])
        np.testing.assert_allclose(
            np.asarray(estimate["spacing"]) * np.asarray(estimate["shape"]),
            project["geometry"]["size"],
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_nonexact_target_selects_nearest_admissible_total(self):
        project = default_project()
        project["mesh"] = {"target_cells": 3500, "cells": [2, 2, 2]}
        result = estimate_mesh(project, tempfile.gettempdir())
        self.assertEqual(result["shape"], [33, 11, 11])
        self.assertEqual(result["total_cells"], 3993)
        self.assertTrue(result["warnings"])

    def test_target_ignores_anisotropic_legacy_vector(self):
        project = default_project()
        project["mesh"] = {"target_cells": 3500, "cells": [1, 2, 3]}
        expected = estimate_mesh({**project, "mesh": {"target_cells": 3500}}, tempfile.gettempdir())
        actual = estimate_mesh(project, tempfile.gettempdir())
        self.assertEqual(actual["shape"], expected["shape"])
        self.assertEqual(actual["total_cells"], expected["total_cells"])

    def test_exact_isotropic_box_failure_does_not_mutate_input(self):
        size = [1.0, float(np.sqrt(2.0)), 1.0]
        with self.assertRaisesRegex(ValueError, "exact isotropic.*configured limits"):
            plan_box(size, 100)
        project = default_project()
        project["geometry"]["size"] = size
        project["mesh"] = {"target_cells": 100}
        original = copy.deepcopy(project)
        with self.assertRaisesRegex(ValueError, "exact isotropic"):
            validate_project(project)
        self.assertEqual(project, original)

    def test_limits_and_cad_thin_axis_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "exact isotropic.*configured limits"):
            plan_box([3.0, 1.0, 1.0], 3_000_000, max_axis=1, max_total=10_000_000)
        with self.assertRaisesRegex(ValueError, "configured axis/total limits"):
            plan_cad([[0.0, 0.0, 0.0], [1.0, 1.0, 0.001]], 1000)

    def test_legacy_box_shape_and_spacing_remain_unchanged(self):
        project = default_project()
        with tempfile.TemporaryDirectory() as temporary:
            mesh = build_mesh(project, temporary)
        self.assertEqual(mesh["shape"], [30, 10, 10])
        np.testing.assert_allclose(mesh["spacing"], [0.002, 0.002, 0.002])

    def test_cad_target_uses_imported_bounds_and_matches_voxel_grid(self):
        import trimesh
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'unequal.stl'
            trimesh.creation.box(extents=[17.3, 8.7, 5.3]).export(source)
            asset = import_cad(source, root / 'assets', unit='mm')
            project = default_project()
            project['geometry'] = {'kind': 'cad', 'role': 'fluid', 'asset_id': asset['id'], 'size': [1, 1, 1]}
            project['mesh'] = {'target_cells': 1000}
            estimate = estimate_mesh(project, root / 'assets')
            mesh = build_mesh(project, root / 'assets')
        self.assertEqual(mesh['shape'], estimate['shape'])
        self.assertGreater(mesh['shape'][0], mesh['shape'][1])
        self.assertGreater(mesh['shape'][1], mesh['shape'][2])
        np.testing.assert_allclose(mesh['spacing'], [estimate['spacing'][0]] * 3)
        np.testing.assert_allclose(mesh['origin'], estimate['origin'])
        self.assertTrue(mesh['fluid_mask'].any())
        bounds = np.asarray(asset['bounds'])
        grid_size = np.asarray(mesh['spacing']) * mesh['shape']
        np.testing.assert_allclose(mesh['origin'] + grid_size / 2, bounds.mean(axis=0), atol=1e-12)
        self.assertTrue(np.all(grid_size >= bounds[1] - bounds[0] - 1e-12))

    def test_obstacle_target_uses_outer_box_for_resolution_and_relaxation(self):
        descriptors = [
            {'computational_box': {'size': [0.06, 0.02, 0.02], 'origin': [-0.03, -0.01, -0.01]}},
            {'domain': {'size': [0.06, 0.02, 0.02], 'origin': [-0.03, -0.01, -0.01]}},
            {'domain_size': [0.06, 0.02, 0.02], 'domain_origin': [-0.03, -0.01, -0.01]},
            {'box_size': [0.06, 0.02, 0.02], 'box_origin': [-0.03, -0.01, -0.01]},
        ]
        for descriptor in descriptors:
            with self.subTest(descriptor=descriptor):
                project = default_project()
                project['geometry'] = {'kind': 'box', 'role': 'obstacle', 'asset_id': None,
                                       'size': [0.004] * 3, 'origin': [-0.002] * 3, **descriptor}
                project['mesh'] = {'target_cells': 3000}
                project['study']['dt'] = 0.1
                with tempfile.TemporaryDirectory() as temporary:
                    estimate = estimate_mesh(project, temporary)
                    mesh = build_mesh(project, temporary)
                self.assertEqual(estimate['shape'], [30, 10, 10])
                self.assertEqual(mesh['shape'], estimate['shape'])
                np.testing.assert_allclose(mesh['spacing'], [0.002] * 3)
                np.testing.assert_allclose(mesh['origin'], [-0.03, -0.01, -0.01])
                self.assertTrue(mesh['solid_mask'].any())

    def test_obstacle_bgk_validation_uses_outer_computational_box(self):
        project = default_project()
        project["geometry"] = {
            "kind": "box",
            "size": [1.0e-6, 1.0e-6, 1.0e-6],
            "origin": [0.009, 0.009, 0.009],
            "asset_id": None,
            "role": "obstacle",
            "computational_box": {"size": [0.02, 0.02, 0.02], "origin": [0.0, 0.0, 0.0]},
        }
        project["mesh"] = {"target_cells": 8000}
        normalized = validate_project(project)
        self.assertEqual(normalized["mesh"]["cells"], [20, 20, 20])
        with tempfile.TemporaryDirectory() as temporary:
            estimate = estimate_mesh(project, temporary)
            mesh = build_mesh(project, temporary)
        self.assertEqual(estimate["shape"], [20, 20, 20])
        self.assertEqual(mesh["shape"], estimate["shape"])

    @unittest.skipIf(trimesh is None, "trimesh is required for CAD planning tests")
    def test_cad_target_uses_actual_noninteger_bounds_and_matches_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = __import__("pathlib").Path(temporary)
            source = root / "noninteger.obj"
            trimesh.creation.box(extents=[13.0, 8.0, 8.0]).export(source)
            from workbench.geometry import import_cad

            asset = import_cad(source, root / "assets", unit="mm")
            project = default_project()
            project["geometry"] = {
                "kind": "cad",
                "asset_id": asset["id"],
                "role": "fluid",
                # Deliberately stale UI data; target mode must use metadata.
                "size": [0.06, 0.02, 0.02],
            }
            project["mesh"] = {"target_cells": 1000}
            estimate = estimate_mesh(project, root / "assets")
            mesh = build_mesh(project, root / "assets")
        self.assertEqual(mesh["shape"], estimate["shape"])
        self.assertEqual(mesh["warnings"], estimate["warnings"])
        np.testing.assert_allclose(mesh["spacing"], estimate["spacing"])
        np.testing.assert_allclose(mesh["origin"], estimate["origin"])
        self.assertGreater(estimate["total_cells"], 0)
        self.assertTrue(any("padding" in warning for warning in estimate["warnings"]))
        lower = np.asarray(estimate["origin"])
        upper = lower + np.asarray(estimate["shape"]) * np.asarray(estimate["spacing"])
        np.testing.assert_allclose(upper - lower, [0.013, 0.008, 0.008], atol=0.002)


if __name__ == "__main__":
    unittest.main()
