"""Physical forcing checks beyond the algebraic Guo moment tests."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from workbench.gravity import GravityModel
from workbench.restart import project_fingerprint
from workbench.schema import default_project
from tests.test_solver import _mesh, _project


def gravity(mode='uniform', vector=(0, 0, -0.2), reference=300.0):
    return {'enabled': True, 'mode': mode, 'vector': list(vector), 'reference_temperature': reference}


class GravityModelTests(unittest.TestCase):
    def test_gravity_only_time_step_estimate(self):
        from workbench.geometry import estimate_mesh
        p = default_project()
        p['boundaries'] = []
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(estimate_mesh(p, tmp)['flow_stability']['recommended_dt'])
            p['physics']['gravity'] = gravity(vector=(0, 0, -9.80665))
            estimate = estimate_mesh(p, tmp)
            stability = estimate['flow_stability']
            self.assertIsNone(stability['dt_max_exclusive'])
            self.assertGreater(stability['recommended_dt'], 0)
            self.assertLess(stability['recommended_dt'], stability['force_dt_max'])
            model = GravityModel(p['physics'], p['materials'][0], stability['recommended_dt'], estimate['spacing'][0], estimate['shape'])
            model.check(1.0)
            model.dt = stability['force_dt_max'] * 1.01
            with self.assertRaisesRegex(ValueError, 'reduce dt'):
                model.check(1.0)

    def test_buoyancy_sign_reference_temperature_and_solid_mask(self):
        material = {'density': {'kind': 'table', 'points': [[290., 1010.], [310., 990.]]}}
        model = GravityModel({'flow': True, 'gravity': gravity('buoyancy')}, material, .001, .005)
        temperature = np.array([290., 300., 310., 310.]).reshape(4, 1, 1)
        active = np.array([True, True, True, False]).reshape(4, 1, 1)
        a, contrast = model.make_operator(np)(temperature, active)
        self.assertLess(a[2, 0, 0, 0], 0)
        self.assertEqual(a[2, 1, 0, 0], 0)
        self.assertGreater(a[2, 2, 0, 0], 0)
        np.testing.assert_array_equal(a[:, 3], 0)
        self.assertAlmostEqual(float(contrast), .01)
        model.check(contrast)
        with self.assertRaisesRegex(ValueError, '10%'):
            model.check(.11)

    def test_force_dt_guard_and_disabled_checkpoint_compatibility(self):
        p = default_project()
        original = project_fingerprint(p)
        p['physics']['gravity'] = dict(gravity(), enabled=False)
        self.assertEqual(project_fingerprint(p), original)
        p['physics']['gravity']['enabled'] = True
        self.assertNotEqual(project_fingerprint(p), original)
        model = GravityModel({'flow': True, 'gravity': gravity(vector=(0, 0, -9.80665))}, p['materials'][0], .01, .002)
        with self.assertRaisesRegex(ValueError, 'reduce dt'):
            model.check(1)


_HAS_XLB = importlib.util.find_spec('xlb') is not None and importlib.util.find_spec('jax') is not None


@unittest.skipUnless(_HAS_XLB, 'XLB/JAX required')
class GravitySolverTests(unittest.TestCase):
    def test_heating_does_not_shift_prescribed_inlet_velocity(self):
        from workbench.solver import simulate
        for backend in ['reference', 'device']:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, XLB_COMPUTE_BACKEND=backend):
                p = _project(steps=2)
                p['study'].update(output_interval=1)
                p['physics']['gravity'] = gravity('buoyancy', vector=(0, 0, -9.80665))
                p['materials'][0]['density'] = {'kind': 'table', 'points': [[290., 1010.], [320., 980.]]}
                p['materials'][0]['conductivity'] = {'kind': 'constant', 'value': 2000.}
                p['materials'][0]['heat_capacity'] = {'kind': 'constant', 'value': 1000.}
                simulate(p, _mesh(), Path(tmp))
                with np.load(Path(tmp) / 'fields.npz') as fields:
                    self.assertGreater(float(fields['temperature'][0, 1, 1]), 300.5)
                    np.testing.assert_allclose(fields['velocity'][:, 0, 1, 1], [.01, 0, 0], atol=3e-7)

    def test_uniform_acceleration_and_zero_physical_initial_velocity(self):
        from workbench.solver import simulate
        for backend in ['reference', 'device']:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, XLB_COMPUTE_BACKEND=backend):
                p = _project(steps=1, thermal=False)
                p['study'].update(output_interval=1, snapshot_interval=1)
                p['boundaries'] = []
                p['physics']['gravity'] = gravity()
                simulate(p, _mesh(), Path(tmp))
                with np.load(Path(tmp) / 'frames/step_000000000.npz') as initial:
                    np.testing.assert_allclose(initial['velocity'], 0, atol=2e-7)
                with np.load(Path(tmp) / 'fields.npz') as final:
                    np.testing.assert_allclose(final['velocity'][:, 3, 1, 1], [0, 0, -.2 * .001], atol=3e-7)

    def test_forced_restart_preserves_lattice_and_temperature(self):
        from workbench.solver import simulate
        for backend in ['reference', 'device']:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, XLB_COMPUTE_BACKEND=backend):
                root = Path(tmp)
                p = _project(steps=7)
                p['physics']['gravity'] = gravity('buoyancy', vector=(0, 0, -9.80665))
                p['materials'][0]['density'] = {'kind': 'table', 'points': [[290., 1010.], [320., 980.]]}
                p['study'].update(output_interval=1, snapshot_interval=2)
                simulate(p, _mesh(), root / 'full')
                first = copy.deepcopy(p)
                first['study']['steps'] = 3
                simulate(first, _mesh(), root / 'first')
                simulate(p, _mesh(), root / 'split', restart_from=root / 'first/restart.npz')
                with np.load(root / 'full/restart.npz') as a, np.load(root / 'split/restart.npz') as b:
                    np.testing.assert_array_equal(a['f'], b['f'])
                    np.testing.assert_array_equal(a['temperature'], b['temperature'])


if __name__ == '__main__':
    unittest.main()
