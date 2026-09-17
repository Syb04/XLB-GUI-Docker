"""Cross-module regressions for physical CAD patch assignment and thermal area."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from workbench.geometry import build_mesh, import_cad
from workbench.schema import default_project
from workbench.solver import simulate


class CadThermalIntegrationTests(unittest.TestCase):
    def test_global_cad_flux_and_patch_override_have_exact_voxel_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'cube.stl'
            cad = trimesh.creation.box(extents=[4., 4., 4.])
            cad.apply_translation([2., 2., 2.])
            cad.export(source)
            asset = import_cad(source, root / 'assets')
            patch = min(asset['surface_groups'], key=lambda group: group['normal'][0])
            project = default_project()
            project['geometry'].update(kind='cad', asset_id=asset['id'])
            project['mesh']['cells'] = [4, 4, 4]
            project['physics'].update(flow=False, initial_temperature=10.)
            for prop in ('density', 'viscosity', 'heat_capacity', 'conductivity'):
                project['materials'][0][prop] = {'kind': 'constant', 'value': 1.}
            project['study'].update(steps=1, dt=.01, output_interval=1)
            project['boundaries'] = [
                {'id': 'all', 'name': 'All CAD', 'face': 'cad',
                 'flow': {'type': 'wall'}, 'thermal': {'type': 'heat_flux', 'value': 2.}},
                {'id': 'patch', 'name': 'One face', 'face': 'cad:' + patch['id'],
                 'flow': {'type': 'wall'}, 'thermal': {'type': 'heat_flux', 'value': 5.}},
            ]
            mesh = build_mesh(project, root / 'assets')
            self.assertEqual(np.count_nonzero(mesh['boundary_links']['cad']), 96)
            self.assertEqual(np.count_nonzero(mesh['boundary_links']['cad:' + patch['id']]), 16)
            result = simulate(project, mesh, root / 'run')
            self.assertEqual(result['status'], 'completed')
            with np.load(root / 'run' / 'fields.npz') as fields:
                energy_gain = float(np.sum(fields['temperature'].astype(float) - 10.))
            # Five unit-area 4x4 faces at q=2; one face overrides q=5.
            self.assertAlmostEqual(energy_gain, .01 * (5 * 16 * 2 + 16 * 5), delta=1e-4)


if __name__ == '__main__':
    unittest.main()
