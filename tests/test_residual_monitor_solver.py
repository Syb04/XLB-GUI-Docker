"""Monitor sampled physical fields, persistence and restart semantics."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_solver import _mesh, _project


_HAS_XLB = importlib.util.find_spec('xlb') is not None and importlib.util.find_spec('jax') is not None


@unittest.skipUnless(_HAS_XLB, 'XLB/JAX required')
class ResidualMonitorSolverTests(unittest.TestCase):
    def test_live_components_match_saved_physical_fields_and_csv(self):
        from workbench.solver import simulate
        for backend in ('reference', 'device'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, XLB_COMPUTE_BACKEND=backend):
                root = Path(tmp)
                p = _project(steps=6)
                p['study'].update(output_interval=2, snapshot_interval=2)
                p['materials'][0]['conductivity'] = {'kind': 'constant', 'value': 600.0}
                rows = []
                def progress(row):
                    # CSV is available before completion, including every emitted sample.
                    with (root / 'history.csv').open(encoding='utf-8', newline='') as handle:
                        saved = list(csv.DictReader(handle))
                    self.assertEqual(int(saved[-1]['step']), row['step'])
                    self.assertEqual(len(saved), len(rows) + 1)
                    json.dumps(row, allow_nan=False)
                    rows.append(row)
                result = simulate(p, _mesh(), root, on_progress=progress)
                self.assertFalse(result['converged'])
                self.assertIsNone(rows[0]['residual'])
                self.assertEqual([r['sample_steps'] for r in rows[1:]], [2, 2, 2])
                self.assertTrue(all(r['monitor_eligible'] for r in rows[1:]))
                for row in rows[1:]:
                    self.assertGreater(row['velocity_residual'], 0)
                    self.assertGreater(row['pressure_residual'], 0)
                    self.assertGreater(row['temperature_residual'], 0)
                    self.assertEqual(row['residual'], max(row[k] for k in ['velocity_residual', 'pressure_residual', 'temperature_residual']))
                    with np.load(root / f"frames/step_{row['step']:09d}.npz") as cur, np.load(root / f"frames/step_{row['step']-2:09d}.npz") as prev:
                        u, v = cur['velocity'].astype(float), prev['velocity'].astype(float)
                        delta = np.linalg.norm(u-v, axis=0).max()
                        scale = max(np.linalg.norm(u, axis=0).max(), np.linalg.norm(v, axis=0).max(), 1e-6)
                        self.assertAlmostEqual(row['velocity_residual'], delta/scale, delta=2e-5)
                        pressure_delta = np.max(np.abs(cur['pressure'].astype(float)-prev['pressure'].astype(float)))
                        pressure_scale = max(np.max(np.abs(cur['pressure'])), np.max(np.abs(prev['pressure'])), 1)
                        self.assertAlmostEqual(row['pressure_residual'], pressure_delta/pressure_scale, delta=2e-5)
                        tdelta = np.max(np.abs(cur['temperature'].astype(float)-prev['temperature'].astype(float)))
                        self.assertAlmostEqual(row['temperature_change_max'], tdelta, delta=4e-5)

    def test_equilibrium_requires_consecutive_full_samples_and_restart_resets(self):
        from workbench.solver import simulate
        for backend in ('reference', 'device'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, XLB_COMPUTE_BACKEND=backend):
                root = Path(tmp)
                p = _project(flow=False, steps=6)
                p['boundaries'] = []
                p['study'].update(output_interval=2, monitor={'tolerance': 1e-5, 'consecutive_samples': 3})
                rows = []
                result = simulate(p, _mesh(), root / 'steady', on_progress=rows.append)
                self.assertEqual([r['monitor_satisfied'] for r in rows], [False, False, False, True])
                self.assertIsNone(rows[-1]['velocity_residual'])
                self.assertIsNone(rows[-1]['pressure_residual'])
                self.assertEqual(rows[-1]['temperature_residual'], 0)
                self.assertTrue(result['converged'])
                p['study']['steps'] = 9
                resumed = []
                result = simulate(p, _mesh(), root / 'resumed', restart_from=root / 'steady/restart.npz', on_progress=resumed.append)
                self.assertEqual([r['step'] for r in resumed], [6, 8, 9])
                self.assertIsNone(resumed[0]['residual'])
                self.assertFalse(resumed[-1]['monitor_eligible'])
                self.assertFalse(result['converged'])


if __name__ == '__main__':
    unittest.main()
