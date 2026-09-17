"""Observe actual JAX download calls, independently of solver diagnostics."""
import os
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

_HAS_XLB = importlib.util.find_spec('jax') is not None and importlib.util.find_spec('xlb') is not None
if _HAS_XLB:
    import jax
    from workbench.solver import simulate
    from tests.test_solver import _mesh, _project


@unittest.skipUnless(_HAS_XLB, 'XLB/JAX runtime required')
class DeviceTransferGuardTests(unittest.TestCase):
    def test_progress_downloads_only_reductions_until_final_artifacts(self):
        project = _project(steps=4)
        project['study']['output_interval'] = 1
        project['physics']['turbulence'] = {
            'model': 'smagorinsky', 'smagorinsky_constant': .17, 'turbulent_prandtl': .9}
        completed_step = -1
        bulk_download_steps = []
        original_get = jax.device_get

        def observe(value, *args, **kwargs):
            leaves = jax.tree_util.tree_leaves(value)
            if any(getattr(leaf, 'size', 0) > 32 for leaf in leaves):
                bulk_download_steps.append(completed_step)
            return original_get(value, *args, **kwargs)

        def progress(sample):
            nonlocal completed_step
            completed_step = sample['step']

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'XLB_COMPUTE_BACKEND': 'device'}), \
                patch.object(jax, 'device_get', side_effect=observe):
            result = simulate(project, _mesh(), directory, on_progress=progress)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(bulk_download_steps, [4], 'Full fields were downloaded before finalization')
