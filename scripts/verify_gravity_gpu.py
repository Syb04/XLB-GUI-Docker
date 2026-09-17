"""Compare Guo gravity/buoyancy on CPU, CUDA and RAM; verify continuation."""
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
    with tempfile.TemporaryDirectory(prefix='gravity-check-') as tmp:
        root = Path(tmp)
        for mode in ('uniform', 'buoyancy'):
            p = make_project(device='cpu')
            p['mesh']['cells'] = [24, 12, 12]
            p['study'].update(steps=20, output_interval=5, snapshot_interval=5, dt=.0003, gpu_batch_cells=1024)
            p['physics']['gravity'] = {'enabled': True, 'mode': mode, 'vector': [0, 0, -.2 if mode == 'uniform' else -9.80665], 'reference_temperature': 300.}
            p['materials'][0]['density'] = {'kind': 'table', 'points': [[290., 1010.], [330., 970.]]}
            if mode == 'buoyancy':
                p['materials'][0]['heat_capacity'] = {'kind': 'constant', 'value': 1000.}
                p['materials'][0]['conductivity'] = {'kind': 'constant', 'value': 200.}
                p['boundaries'] = [
                    {'id': 'hot', 'name': 'Hot wall', 'face': 'xmin', 'flow': {'type': 'wall'}, 'thermal': {'type': 'temperature', 'value': 310.}},
                    {'id': 'cold', 'name': 'Cold wall', 'face': 'xmax', 'flow': {'type': 'wall'}, 'thermal': {'type': 'temperature', 'value': 290.}},
                ]
            mesh = make_mesh(p, root / 'assets')
            runs = {}
            for device in ('cpu', 'cuda:0', 'cuda:0-ram'):
                p['study']['device'] = device
                run = root / mode / device.replace(':', '_')
                result = simulate(p, mesh, run)
                assert result['status'] == 'completed'
                assert result['diagnostics']['gravity']['active']
                if device == 'cuda:0':
                    assert 'cuda' in result['diagnostics']['device_actual'].lower()
                if device == 'cuda:0-ram':
                    assert 'cpu' in result['diagnostics']['state_device'].lower()
                    assert 'cuda' in result['diagnostics']['collision_device'].lower()
                    assert result['diagnostics']['ram_offload']['acceleration_channels'] == 3
                runs[device] = run
                differences = {}
                with np.load(runs['cpu'] / 'fields.npz') as expected, np.load(run / 'fields.npz') as actual:
                    for field, tolerance in COMPARISON_TOLERANCES.items():
                        np.testing.assert_allclose(actual[field], expected[field], **tolerance)
                        differences[field] = float(np.max(np.abs(actual[field] - expected[field])))
                    if mode == 'buoyancy':
                        # A hot/light column rises; a cold/heavy one descends.
                        hot_vertical = float(actual['velocity'][2, 1, 6, 6])
                        cold_vertical = float(actual['velocity'][2, -2, 6, 6])
                        assert hot_vertical > 0, hot_vertical
                        assert cold_vertical < 0, cold_vertical
                short = copy.deepcopy(p)
                short['study']['steps'] = 8
                simulate(short, mesh, run / 'first')
                simulate(p, mesh, run / 'resumed', restart_from=run / 'first/restart.npz')
                with np.load(run / 'restart.npz') as a, np.load(run / 'resumed/restart.npz') as b:
                    for field in ('f', 'temperature'):
                        np.testing.assert_array_equal(a[field], b[field])
                frames = json.loads((run / 'frames.json').read_text())['frames']
                for frame in frames:
                    with np.load(runs['cpu'] / 'frames' / frame['file']) as expected, np.load(run / 'frames' / frame['file']) as actual:
                        for field, tolerance in COMPARISON_TOLERANCES.items():
                            np.testing.assert_allclose(actual[field], expected[field], **tolerance)
                evidence['cases'].append({'mode': mode, 'device': device, 'steps': 20, 'cells': int(mesh['fluid_mask'].size),
                                          'cpu_max_absolute_differences': differences, 'restart_exact': True,
                                          'hot_vertical_m_s': hot_vertical if mode == 'buoyancy' else None,
                                          'cold_vertical_m_s': cold_vertical if mode == 'buoyancy' else None,
                                          'diagnostics': result['diagnostics']})
                print(mode, device, 'fields, snapshots and exact restart OK', flush=True)
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
