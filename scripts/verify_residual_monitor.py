"""Actual CPU/CUDA/RAM monitor verification with LES, CHT and buoyancy."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from scripts.verify_transient_solver import make_project, make_mesh, COMPARISON_TOLERANCES
from workbench.solver import simulate


def main():
    os.environ['XLB_COMPUTE_BACKEND'] = 'auto'
    evidence = {'cases': []}
    with tempfile.TemporaryDirectory(prefix='residual-monitor-') as tmp:
        root = Path(tmp)
        p = make_project(device='cpu')
        p['mesh']['cells'] = [24, 12, 12]
        p['study'].update(steps=12, output_interval=3, snapshot_interval=0, dt=.0003, gpu_batch_cells=1024,
                          monitor={'tolerance': 1e-5, 'consecutive_samples': 3})
        p['physics']['gravity'] = {'enabled': True, 'mode': 'buoyancy', 'vector': [0, 0, -9.80665], 'reference_temperature': 300.}
        p['materials'][0]['density'] = {'kind': 'table', 'points': [[290., 1010.], [330., 970.]]}
        mesh = make_mesh(p, root / 'assets')
        for device in ('cpu', 'cuda:0', 'cuda:0-ram'):
            p['study']['device'] = device
            directory = root / device.replace(':', '_')
            rows = []
            result = simulate(p, mesh, directory, on_progress=rows.append)
            assert result['status'] == 'completed'
            assert rows[0]['residual'] is None
            assert all(r['monitor_eligible'] for r in rows[1:])
            assert all(r['velocity_residual'] > 0 and r['pressure_residual'] > 0 and r['temperature_residual'] > 0 for r in rows[1:])
            if device == 'cuda:0':
                assert 'cuda' in result['diagnostics']['device_actual']
                assert result['diagnostics']['full_field_transfers'] == 1
            if device == 'cuda:0-ram':
                assert 'cpu' in result['diagnostics']['state_device']
                assert 'cuda' in result['diagnostics']['collision_device']
            different_interval = copy.deepcopy(p)
            different_interval['study']['output_interval'] = 1
            simulate(different_interval, mesh, directory / 'every_step')
            with np.load(directory / 'restart.npz') as a, np.load(directory / 'every_step/restart.npz') as b:
                for name in ('f', 'temperature'):
                    np.testing.assert_array_equal(a[name], b[name])
            with np.load(root / 'cpu/fields.npz') as expected, np.load(directory / 'fields.npz') as actual:
                for name, tolerance in COMPARISON_TOLERANCES.items():
                    np.testing.assert_allclose(actual[name], expected[name], **tolerance)
            evidence['cases'].append({'device': device, 'status': result['status'], 'sampling_does_not_change_solution': True,
                                      'samples': rows, 'diagnostics': result['diagnostics']})
            print(device, 'live fields and sample interval invariance OK', flush=True)
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
