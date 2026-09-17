import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from workbench.schema import default_project, validate_project
from workbench.snapshots import SnapshotWriter

HAS_XLB = importlib.util.find_spec('xlb') is not None and importlib.util.find_spec('jax') is not None


class SnapshotTests(unittest.TestCase):
    def test_schema_legacy_default_and_interval_validation(self):
        project = default_project()
        self.assertEqual(validate_project(project)['study']['snapshot_interval'], 20)
        project['study'].pop('snapshot_interval')
        self.assertEqual(validate_project(project)['study']['snapshot_interval'], 0)
        for invalid in (-1, 1.2, True, '2'):
            project['study']['snapshot_interval'] = invalid
            with self.assertRaises(ValueError):
                validate_project(project)

    def test_failed_write_never_publishes_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = SnapshotWriter(temporary, 1, .01, fluid_mask=np.ones((2, 2, 2), bool))
            with patch('workbench.snapshots.np.savez', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    writer.write(1, velocity=np.zeros((3, 2, 2, 2)), pressure=np.zeros((2, 2, 2)),
                                 temperature=np.full((2, 2, 2), 300), eddy_viscosity=np.zeros((2, 2, 2)))
            self.assertEqual(json.loads((Path(temporary) / 'frames.json').read_text())['frames'], [])
            self.assertFalse(list(Path(temporary).rglob('*.tmp')))


@unittest.skipUnless(HAS_XLB, 'XLB/JAX runtime required')
class SnapshotSolverTests(unittest.TestCase):
    def test_interval_fields_are_real_states_and_downloads_are_requested_only(self):
        import jax
        from workbench.solver import simulate
        from tests.test_solver import _mesh, _project
        for backend in ('reference', 'device'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(os.environ, {'XLB_COMPUTE_BACKEND': backend}):
                project = _project(steps=5)
                project['study'].update(snapshot_interval=2, output_interval=1)
                downloads = []
                original = jax.device_get

                def observe(value, *args, **kwargs):
                    if any(getattr(item, 'size', 0) > 32 for item in jax.tree_util.tree_leaves(value)):
                        downloads.append(True)
                    return original(value, *args, **kwargs)

                live_frames = []

                def progress(row):
                    path = Path(temporary) / 'frames.json'
                    if row['step'] == 2:
                        live_frames.extend(json.loads(path.read_text())['frames'])

                with patch.object(jax, 'device_get', side_effect=observe):
                    result = simulate(project, _mesh(), temporary, on_progress=progress)
                manifest = json.loads((Path(temporary) / 'frames.json').read_text())['frames']
                self.assertEqual([entry['step'] for entry in manifest], [0, 2, 4, 5])
                self.assertEqual([entry['step'] for entry in live_frames], [0, 2])
                if backend == 'device':
                    self.assertEqual(len(downloads), 4)
                    self.assertEqual(result['diagnostics']['full_field_transfers'], 4)
                with np.load(Path(temporary) / 'frames' / manifest[-1]['file']) as frame, \
                        np.load(Path(temporary) / 'fields.npz') as final:
                    for key in frame.files:
                        np.testing.assert_array_equal(frame[key], final[key])
                project['study'].update(steps=2, snapshot_interval=0)
                independent = Path(temporary) / 'independent'
                simulate(project, _mesh(), independent)
                with np.load(Path(temporary) / 'frames' / manifest[1]['file']) as frame, \
                        np.load(independent / 'fields.npz') as final:
                    for key in frame.files:
                        np.testing.assert_allclose(frame[key], final[key], rtol=2e-5, atol=1e-6)

    def test_stop_between_snapshots_retains_actual_last_step(self):
        from workbench.solver import simulate
        from tests.test_solver import _mesh, _project
        for backend in ('reference', 'device'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(os.environ, {'XLB_COMPUTE_BACKEND': backend}):
                project = _project(steps=8)
                project['study'].update(snapshot_interval=4, output_interval=1)
                progress = []
                result = simulate(project, _mesh(), temporary, on_progress=progress.append,
                                  should_stop=lambda: bool(progress and progress[-1]['step'] >= 3))
                self.assertEqual(result['status'], 'stopped')
                frames = json.loads((Path(temporary) / 'frames.json').read_text())['frames']
                self.assertEqual([entry['step'] for entry in frames], [0, 3])
                self.assertAlmostEqual(frames[-1]['time'], .003)
